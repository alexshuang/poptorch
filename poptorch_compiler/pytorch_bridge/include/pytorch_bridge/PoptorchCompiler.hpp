// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_HPP_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/PoplarExecutorWrapper.hpp"

namespace mlir {
class RankedTensorType;
class Operation;
} // namespace mlir

namespace poptorch {
struct CompilerOptions;
}

namespace poptorch_ir {

namespace detail {
class IMLIRCompiler;
} // namespace detail

enum class ExecutionType { StaticGraph, EagerMode };

enum class CompilerBackend { Poplar, PopIR };

class PoptorchCompiler {
public:
  PoptorchCompiler();
  ~PoptorchCompiler();

  void startTraceTiming();
  void endTraceTiming();
  void getTimingInfo();

  void dump();

  void init(ExecutionType execution_type, CompilerBackend compiler_backend,
            const poptorch::CompilerOptions &options);

  const poptorch::CompilerOptions &getOptions() const;
  poptorch::CompilerOptions &getMutableOptions();

  void setCurrentPythonCodeLocation(const char *filename, std::uint64_t line,
                                    std::uint64_t col);

  TensorId addInput(const Buffer &ptr, const std::vector<std::int64_t> &shape,
                    Type, const char *);
  TensorId addParameter(const Buffer &ptr,
                        const std::vector<std::int64_t> &shape, Type,
                        const char *);
  void addOutput(TensorId id, void *ptr, const char *);

  // Only if ExecutionType::EagerMode is used.
  void compileRunAndReset();

  // Only if ExecutionType::StaticGraph is used
  std::unique_ptr<PoplarExecutorWrapper> compileAndLoad();

  std::vector<std::int64_t> getSize(TensorId id) const;
  Type getType(TensorId id) const;

  // Non popart.
  void addReturn();

  bool isView(TensorId id) const;

  // Return true if all the ops in the graph can be lowered to
  // Poplar.
  bool allOpsCanBeLoweredToPoplar() const;

  // Called by the dispatcher after it added an op to the graph.
  // (Can be used by the backend as a trigger for eager execution)
  void onOpAdded();

// Each tablegen entry will automatically generate a C++ method and impl which
// can be used by PyTorch. This means Compiler will have a function to add any
// op using non-pytorch, non-mlir types. Tensors are poptorch_ir::TensorId.
// Functions return void, poptorch_ir::TensorId, or
// std::vector<poptorch_ir::TensorId> depending on their type.
#include "dialect/AutogenCompiler.hpp.inc"

private:
  mlir::RankedTensorType getRankedTensorType(TensorId id) const;

  std::unique_ptr<detail::IMLIRCompiler> _impl;
};

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_HPP_
