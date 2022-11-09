// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_LOWER_TO_POPLAR_POPIT_EXECUTOR_HPP_
#define POPTORCH_LOWER_TO_POPLAR_POPIT_EXECUTOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/IpuSession.hpp"

namespace popit {
struct MemRef;
using Mem_t = MemRef;
using FunctionId_t = unsigned;
} // namespace popit

namespace mlir {
class ModuleOp;
class TimingScope;
class Value;
class RankedTensorType;
} // namespace mlir

namespace poptorch_ir {

class EagerIpuSession;
class NonRestartingMLIRTimer;

class PopitDeviceFunction {
public:
  PopitDeviceFunction(EagerIpuSession &context, mlir::ModuleOp module,
                      NonRestartingMLIRTimer &timer);

  void run(const std::vector<popit::Mem_t *> &inputs,
           const std::vector<popit::Mem_t *> &outputs);

  friend class LowerToPopit;

private:
  // Only the headless ipu session should be making device functions that don't
  // do anything
  friend class HeadlessIpuSession;
  PopitDeviceFunction() = default;

  // These attributes get populated by LowerToPopit
  popit::FunctionId_t _popit_fn;

  // Note we need to be careful that PopitFunctions aren't called after their
  // context is destroyed
  EagerIpuSession *_context = nullptr;
};

} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPLAR_POPIT_EXECUTOR_HPP_
