// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_LOWER_TO_POPLAR_PASS_HPP_
#define POPTORCH_LOWER_TO_POPLAR_PASS_HPP_

#include <memory>

namespace mlir {
template <typename T> class OperationPass;
class ModuleOp;
} // namespace mlir

namespace poptorch_ir {

class CompilerContext;

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLowerToPoplarPass(CompilerContext &context);

} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPLAR_PASS_HPP_