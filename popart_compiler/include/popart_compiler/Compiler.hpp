// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popart_compiler/PopartEnums.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace popart {
enum class DataType;
class ConstVoidData;
} // namespace popart

namespace poptorch {

using TensorId = std::size_t;

static constexpr TensorId NoneTensor = 0; // NOLINT

namespace detail {
struct CompilerImpl;
struct SessionOptionsImpl;
} // namespace detail

struct OutputType {
  enum class Type { Tensor, Tuple, List };
  Type type;
  int64_t num_elements{0};
};

// Extract the value from the map or return zero.
static std::pair<float, bool> findInMapOrDefault(
    const std::unordered_map<std::string, std::pair<float, bool>> &opts,
    const std::string &name, float defaultValue = 0.0f) {

  // Lookup map.
  auto itr = opts.find(name);
  if (itr != opts.end()) {
    return itr->second;
  }

  logging::info("Optimizer map didn't have field for {}, defaulting to {}",
                name, defaultValue);
  return {defaultValue, false};
}

/* Returns true if the system contains a device with numIpus
 * Note: This function doesn't check if the devices are currently in use.
 */
bool ipuHardwareIsAvailable(std::uint64_t num_ipus = 1);

struct Optimizer {
  using ParamType = std::pair<float, bool>;
  using ParamList = std::unordered_map<std::string, ParamType>;

  explicit Optimizer(OptimizerType t, const ParamList &opts) : type(t) {
    // It is valid to not pass in a optimizer.
    if (opts.empty() || type == OptimizerType::NONE) {
      return;
    }

    // We will assume all optimizers will have a learning rate.
    auto itr = opts.find("lr");
    ERROR_ON_MSG(itr == opts.end(),
                 "Learning rate was not provided in optimizer dictionary!");
    learning_rate = itr->second;
    weight_decay = findInMapOrDefault(opts, "weight_decay");
    loss_scaling = findInMapOrDefault(opts, "loss_scaling", 1.0f);
    velocity_scaling = findInMapOrDefault(opts, "velocity_scaling", 1.0f);

    switch (type) {
    case OptimizerType::SGD: {
      momentum = findInMapOrDefault(opts, "momentum");
      dampening = findInMapOrDefault(opts, "dampening");
      break;
    }
    case OptimizerType::LAMB:
    case OptimizerType::LAMB_NO_BIAS:
    case OptimizerType::ADAMW: {
      beta1 = findInMapOrDefault(opts, "beta1", 0.9);
      beta2 = findInMapOrDefault(opts, "beta2", 0.999);
      eps = findInMapOrDefault(opts, "eps", 1e-08);
      break;
    }
    case OptimizerType::RMSPROP_CENTERED:
    case OptimizerType::RMSPROP: {
      alpha = findInMapOrDefault(opts, "alpha", 0.99);
      eps = findInMapOrDefault(opts, "eps", 1e-08);
      break;
    }
    case OptimizerType::NONE:
    default:
      ERROR("UNREACHABLE: Unsupported optimizer type");
    }
  }

  OptimizerType type;

  ParamType learning_rate;
  ParamType weight_decay;
  ParamType loss_scaling;
  ParamType velocity_scaling;

  // Shared by SGD and RMSprop
  ParamType momentum;

  // Shared by AdamW, RMSprop and LAMB
  ParamType eps;

  // Shared by AdamW and LAMB
  ParamType beta1;
  ParamType beta2;

  // Unique to SGD
  ParamType dampening;

  // Unique to RMSprop
  ParamType alpha;
};

class Compiler;
class SessionOptions {
public:
  SessionOptions();
  SessionOptions(SessionOptions &&);
  ~SessionOptions();
  // Disable copy: Move only
  SessionOptions(const SessionOptions &) = delete;
  SessionOptions &operator=(const SessionOptions &) = delete;

  void setMemoryProportion(std::uint32_t ipu, float memory);

  void setPatternsLevel(std::uint64_t level);
  void addPattern(const char *pattern, bool enabled);
  void setTensorLocation(const char *tensor, const char *option,
                         std::uint64_t value);

  void addStringOption(const char *option, const char *value);
  void addUint64Option(const char *option, std::uint64_t value);
  void addBoolOption(const char *option, bool value);
  void addDoubleOption(const char *option, double value);
  // Insert a string option in an option container (set / list / vector)
  void insertStringOption(const char *option, const char *value);
  // Insert a key / value pair in an option map
  void insertStringPairOption(const char *option, const char *key,
                              const char *value);

private:
  std::unique_ptr<detail::SessionOptionsImpl> _impl;
  friend Compiler;
};

// A class to store all the data and info required to create a constant in the
// popart builder for convenience. Internally, it is a simple wrapper to
// popart::ConstVoidData.
class PopartConstant {
public:
  PopartConstant(const PopartType &popart_type, const void *data,
                 const std::vector<std::int64_t> &shape);

  ~PopartConstant(); // Required for opaque pointer

  const popart::ConstVoidData *getPopartData() const { return _data.get(); }

private:
  // Use an opaque pointer to avoid the need for popart headers
  std::unique_ptr<popart::ConstVoidData> _data;
};

// A class to store a constant which is simply returned, (possibly in a tuple
// or list) and is not inserted into Popart
class HostSideConstant {
public:
  HostSideConstant(const PopartType &popart_type, void *data, size_t data_size,
                   std::vector<std::int64_t> shape);

  PopartType popartType() const { return _popart_type; }

  const std::vector<std::int64_t> &shape() const { return _shape; }

  void copyDataTo(void *ptr) const;

private:
  const PopartType _popart_type;
  std::vector<uint8_t> _data;
  std::vector<std::int64_t> _shape;
};

class Compiler {
public:
  Compiler(bool is_training, const SessionOptions &options);
  ~Compiler();
  Compiler(Compiler &&compiler);

  poptorch::TensorId addInputTensor(const char *type,
                                    const std::vector<std::int64_t> &dims);

#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<double>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define STRING const char *
#define NONE
#define ARG(Type, Name) , Type Name
#define POPART_CONST_ARG(Name) , const PopartConstant &Name
#define HOST_SIDE_CONST_ARG(Name) , const HostSideConstant &Name
#define BODY_ARG(Name) NONE

// Create a function decl with the given call and arguments.
#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs)       \
  poptorch::TensorId function(                                                 \
      const std::vector<poptorch::TensorId> &inputs Args);

#include "SupportedOperations.inc.hpp"

#undef OP_DECL
#undef BODY_ARG
#undef HOST_SIDE_CONST_ARG
#undef POPART_CONST_ARG
#undef ARG
#undef NONE
#undef STRING
#undef BOOL
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC

  poptorch::TensorId
  addInitializedInputTensor(const char *name, const char *type,
                            const std::vector<std::int64_t> &dims, void *data);

  bool tensorIdIsValid(poptorch::TensorId id) const;
  const char *tensorName(poptorch::TensorId id) const;

  std::vector<std::int64_t> getSize(poptorch::TensorId id) const;

  std::unique_ptr<char[]> getTensorDTypeString(poptorch::TensorId id) const;

  bool isHostSideConstant(poptorch::TensorId id) const;

  poptorch::TensorId
  customOperation(const char *op,
                  const std::vector<poptorch::TensorId> &inputs);

  void addOutputType(OutputType type);

  // This function marks |output| as being read back from the device by the
  // host. |anchor_mode| determines how frequently that should happen.
  // clang-format off
  // "ALL":  Will return all popart batches.
  // "SUM": Will return the sum of all popart batches (I.E device iterations)
  // "EVERYN": Will return every N batch
  // "FINAL": Will return the last batch only
  // clang-format on
  void addOutputTensor(poptorch::TensorId output);

  void setUpInputOp(poptorch::TensorId id, float *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(poptorch::TensorId id, std::int32_t *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(poptorch::TensorId id, bool *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(poptorch::TensorId id, std::int16_t *ptr,
                    const std::vector<std::int64_t> &dims,
                    bool float16 = false);

  void setUpOutputOp(poptorch::TensorId id, float *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, std::int32_t *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, bool *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, std::int16_t *ptr,
                     const std::vector<std::int64_t> &dims);

  void
  setAvailableMemoryProportion(const std::vector<poptorch::TensorId> &inputs,
                               float availableMemoryProportion);

  void setMatMulSerialization(poptorch::TensorId matmul, const char *mode,
                              std::uint64_t factor,
                              std::uint64_t keep_precision);
  void clearActiveIpu();
  void setActiveIpu(std::uint64_t stage_id, std::int64_t phase_id,
                    std::int64_t ipu_id);

  void initSession(const std::vector<Optimizer> &opt);

  // Write the weights into IPU memory from the pytorch tensor buffers in the
  // model.
  void copyWeightsToDevice(const std::vector<void *> &host_buffers);

  // Read the weights from IPU memory into the pytorch tensor buffers.
  void copyWeightsToHost(const std::vector<void *> &host_buffers);

  // Return the type of the given tensor.
  PopartType getPopartType(poptorch::TensorId id) const;

  /*
   * Execute the compiled popart graph using poplar. An optimizer can be
   * provided to update the optimizer currently being run by the graph. If there
   * is nothing to update the optimizer will be set to OptimizerType::None
   * otherwise the new optimizer will be written to device.
   */
  void run(const std::vector<Optimizer> &optimizer);

  std::uint64_t batchPerStep() const;

  // Return the PopART batch dimensions [DeviceIterations * ReplicationFactor *
  // GradientAccumulation]
  std::uint64_t popartBatchDim() const;

  // Take the above and work out how much of it is being returned. ID must anbe
  // an anchor d the batch dim will be mutated depending on what the anchor is
  // returning.
  std::uint64_t popartBatchDimForAnchor(poptorch::TensorId id) const;

  // Return a flat representation of the output types
  // For example: ( T0, T2, (T3, T4)) is represented as:
  // [ Tuple3, Tensor, Tensor, Tuple2, Tensor, Tensor ]
  const std::vector<OutputType> &outputTypes() const;

  // We return this as a unique char pointer to avoid leaking memory while
  // protecting the ABI boundry.
  std::unique_ptr<char[]> getPopartIR() const;

  void optimizerGroup(const std::vector<poptorch::TensorId> &inputs,
                      int64_t group);

  std::unique_ptr<char[]> getExecutionInfo() const;

private:
  void assertTensorIs(PopartType dataType, poptorch::TensorId id,
                      const char *caller) const;
  std::unique_ptr<detail::CompilerImpl> _impl;
};

} // namespace poptorch
