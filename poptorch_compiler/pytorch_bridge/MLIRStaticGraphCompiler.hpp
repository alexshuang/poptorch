// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_MLIR_STATIC_GRAPH_COMPILER_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_MLIR_STATIC_GRAPH_COMPILER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "IMLIRCompiler.hpp"
#include "lower_to_poplar/PoplarExecutor.hpp"

namespace poptorch_ir {

class Buffer;

namespace detail {

class MLIRStaticGraphCompiler : public IMLIRCompiler {
public:
  explicit MLIRStaticGraphCompiler(const poptorch::CompilerOptions &options);
  virtual ~MLIRStaticGraphCompiler() = default;
  // Compile graph by running both PopTorch compiler passes and poplar
  // compilation.
  poptorch_ir::PoplarExecutor compile(const PoplarTarget &target);
  TensorId addInput(const mlir::RankedTensorType &input,
                    const char *name) override;

  TensorId addParameter(Buffer &ptr, const mlir::RankedTensorType &parameter,
                        const char *name) override;
  void addOutput(TensorId id, const char *name) override;
  void addReturn() override;

private:
  // Program to write weights onto the chip.
  Graph _write_weights_graph;

  // Program to read weights off the chip.
  Graph _read_weights_graph;

public:
  // Input and output callbacks to give to poplar.
  std::vector<StreamInfo> input_callbacks;
  std::vector<StreamInfo> output_callbacks;
  std::vector<StreamInfo> weight_callbacks;
};

} // namespace detail

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_MLIR_STATIC_GRAPH_COMPILER_HPP_
