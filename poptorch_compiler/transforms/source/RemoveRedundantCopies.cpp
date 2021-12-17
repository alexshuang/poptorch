// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "passes/CommonPasses.hpp"
#include "passes/PassUtils.hpp"

#include "dialect/PoptorchDialect.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

namespace {
/*
  Converts the MLIR graph into a poplar graph which can then be compiled.
 */
class RemoveRedundantCopies final
    : public mlir::PassWrapper<RemoveRedundantCopies,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  RemoveRedundantCopies() = default;

  void runOnOperation() override;
};

void RemoveRedundantCopies::runOnOperation() {
  mlir::ModuleOp module = this->getOperation();
  std::vector<mlir::Operation *> to_remove;

  poptorch::logging::trace("Graph right before RemoveRedundantCopies:\n{}",
                           mlirOpToStr(module));

  mlir::FuncOp main_graph = *module.getOps<mlir::FuncOp>().begin();
  ERROR_ON(main_graph.getName().str() != "MainGraph");
  for (mlir::Operation &op : main_graph.getOps()) {
    // Case 1: (empty_tensor - op - copy_) pattern
    // This pattern is generated by torch ops with output parameters
    // (e.g. avg_pool2d.out). These ops needlessly copy the result
    // of an op into an empty tensor. We can remove both the copy and the
    // empty tensor, and replace all uses of the empty tensor with the
    // result of the op itself.
    if (mlir::isa<empty_tensor>(op)) {
      // If the empty_tensor op has no uses we can immediately mark it for
      // deletion and move on
      if (op.use_empty()) {
        to_remove.push_back(&op);
        continue;
      }
      // The first user is the last element in getUsers
      mlir::Operation *first_user = nullptr;
      // Unfortunately we need to traverse the entire list to get the last
      // element because the iterator isn't bidirectional
      for (mlir::Operation *user : op.getUsers()) {
        first_user = user;
      }

      // Remove only if this is a copy.
      if (poptorch_ir::copy_ copy =
              llvm::dyn_cast<poptorch_ir::copy_>(*first_user)) { // NOLINT
        // Check we are copying *to* the empty tensor.
        if (copy.self() != op.getResult(0)) {
          continue;
        }
        // The second operand of copy_ is the result of the operation we
        // want to replace the empty_tensor with
        copy.self().replaceAllUsesWith(copy.src());
        to_remove.push_back(&op);
        to_remove.push_back(first_user);
      }
    }
    // TODO(T44785): Add more copy removal cases
  }
  for (auto *op : to_remove) {
    op->erase();
  }
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRemoveRedundantCopiesPass() {
  return std::make_unique<RemoveRedundantCopies>();
}

} // namespace poptorch_ir

static mlir::PassRegistration<poptorch_ir::RemoveRedundantCopies>
    remove_redundant_copies("remove-redundant-copies", "");
