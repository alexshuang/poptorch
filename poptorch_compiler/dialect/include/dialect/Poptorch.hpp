// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_CODEGEN_POPTORCH_IR_H_
#define POPTORCH_CODEGEN_POPTORCH_IR_H_

#include <algorithm>
#include <numeric>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "poptorch_logging/Error.hpp"

/*
 * Custom traits.
 */

// TODO(T62821) in next LLVM version (Current 13.0) we can move this into our
// own namespace. Note: this comment was also present for LLVM 13.0 so remove it
// if we can't do this in LLVM 14
namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class ViewOp : public TraitBase<ConcreteType, ViewOp> {
public:
  // Return the value this operation is a view of.
  mlir::Value isViewOf() const;
};

template <typename ConcreteType>
class NotImplementedOp : public TraitBase<ConcreteType, NotImplementedOp> {};

const size_t max_implicit_casting_operands = 3;
template <unsigned idx> class ImplicitCastOperand {
public:
  static_assert(idx < max_implicit_casting_operands);

  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, ImplicitCastOperand<idx>::Impl> {
  };
};

} // namespace OpTrait
} // namespace mlir

namespace poptorch_ir {

struct CompilerContext;

inline std::int64_t convertToPositiveDim(std::int64_t dim,
                                         std::size_t numDims) {
  // Pytorch allows you to index into the dimensions of scalar values using 0
  // and -1
  if (numDims == 0) {
    ERROR_ON_MSG(!(dim == -1 || dim == 0),
                 "When indexing into a scalar value the dimension must be in "
                 "the range [-1, 0]");
    return 0;
  }

  if (dim < 0) {
    dim += numDims;
  }

  ERROR_ON_MSG(dim >= static_cast<std::int64_t>(numDims) ||
                   dim < std::int64_t{0},
               "Dimension must be within the range of the tensor");

  return dim;
}
inline std::vector<std::int64_t>
convertToPositiveDim(std::vector<std::int64_t> dim, std::size_t numDims) {
  for (auto &d : dim) {
    d = convertToPositiveDim(d, numDims);
  }

  return dim;
}

// Add our interfaces. These allow us to call poptorch specific functions on
// generic ops.
#include "dialect/PoptorchInterfaces.h.inc"

} // namespace poptorch_ir

// Include the operation definition.
#define GET_OP_CLASSES
#include "dialect/Poptorch.h.inc"

#endif // POPTORCH_CODEGEN_POPTORCH_IR_H_
