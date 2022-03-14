// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popart_compiler/PopartEnums.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace popart {
class any;
enum class DataType;
class ConstVoidData;
} // namespace popart

namespace poptorch {

// PopTorch abstraction of popart::MutableVoidData to be used across the ABI
// boundary
struct TensorMetadata {
  const char *id;
  std::vector<int64_t> shape;
  const char *dtype;
  void *data = nullptr;
  int64_t num_bytes = -1;
};

/*
  We use this callback structure to capture data from the poptorch python
  frontend. We get the function to call as well as pointers to the output/input
  storage waiting on CPU. From this we derive more data, see
  CallbackInternalMetadata in CompilerImpl.hpp.
*/
struct CallbackMetadata {
  // The thing we are calling back.
  std::function<void()> the_callback;

  // Due to tracing complexities we have to register the buffers as a seperate
  // step after the model has been traced.
  std::function<void()> buffer_registration_callback;

  // Pointers to the buffers we created on host.
  std::vector<void *> input_pointers;
  std::vector<void *> output_pointers;
};

using TensorId = std::size_t;

static constexpr TensorId NoneTensor = 0; // NOLINT

namespace detail {
struct CompilerImpl;
struct SessionOptionsImpl;
struct ExceptionInfoImpl;
} // namespace detail

enum class OutputElemType { Tensor, Tuple, List };

enum class ErrorCategory { RuntimeRecoverable, RuntimeUnrecoverable, Other };

// For testing only: throw an exception of the selected type.
enum class TestErrorType {
  Poptorch,
  Popart,
  PopartInternal,
  Poplibs,
  PoplarUnrecoverable,
  PoplarUnknown,
  PoplarRecoverableFullReset,
  PoplarLinkError
};
void throwTestError(TestErrorType type);

class ExceptionInfo : public std::exception {
public:
  explicit ExceptionInfo(const std::exception &e, const char *type,
                         const char *filename, uint64_t line);
  ~ExceptionInfo() override;
  const char *what() const noexcept override;
  const char *type() const;
  int64_t stackDepth() const;
  const char *stack(int64_t level) const;
  const char *filename() const;
  uint64_t line() const;
  const char *recoveryAction() const;
  ErrorCategory category() const;

private:
  std::unique_ptr<detail::ExceptionInfoImpl> _impl;
};

struct OutputType {
  OutputElemType type;
  int64_t num_elements{0};
};

/** Returns the IPU version of the device if the system contains a device with
 * num_ipus -1 if there is a device but the architecture is unknown. 0 if there
 * is no device with num_ipus.
 *
 * Note: This function doesn't check if the devices are currently in use.
 */
std::int64_t ipuHardwareVersion(std::uint64_t num_ipus = 1);

void setPopartLogLevel(logging::Level level);

struct Optimizer {
  struct Parameter {
    char name[32];
    float value;
    bool is_const;
  };
  using ParamType = std::pair<float, bool>;

  explicit Optimizer(OptimizerType t, bool useTfVariant)
      : type(t), accum_types_provided(false), use_tf_variant(useTfVariant) {}
  explicit Optimizer(OptimizerType t, bool useTfVariant, float maxGradNorm)
      : type(t), accum_types_provided(false), use_tf_variant(useTfVariant),
        max_grad_norm(maxGradNorm) {}
  Optimizer(OptimizerType t, bool accumType, bool firstOrderType,
            bool secondOrderType, bool useTfVariant, float maxGradNorm)
      : type(t), accum_types_provided(true), accum_type_is_half(accumType),
        first_order_momentum_accum_type_is_half(firstOrderType),
        second_order_momentum_accum_type_is_half(secondOrderType),
        use_tf_variant(useTfVariant), max_grad_norm(maxGradNorm) {}

  // Copies the value and constness of one parameter to another
  void copyParam(const Optimizer &source_optim, const char *source,
                 const char *dest);

  void copyParam(const char *source, const char *dest) {
    copyParam(*this, source, dest);
  }

  OptimizerType type;
  // True if the main, first and second order accum types have been set.
  bool accum_types_provided;
  // Special parameters for adam/lamb. If true accumulations will be half
  // otherwise will be float.
  bool accum_type_is_half;
  bool first_order_momentum_accum_type_is_half;
  bool second_order_momentum_accum_type_is_half;
  bool use_tf_variant;
  float max_grad_norm;

  std::vector<Parameter> parameters;
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
  void
  setCompilationProgressLogger(const std::function<void(int, int)> &logger);

  void addStringOption(const char *option, const char *value);
  void addUint64Option(const char *option, std::uint64_t value);
  void addBoolOption(const char *option, bool value);
  void addDoubleOption(const char *option, double value);
  // Insert a string option in an option container (set / list / vector)
  void insertStringOption(const char *option, const char *value);
  // Insert a key / value pair in an option map
  void insertStringPairOption(const char *option, const char *key,
                              const char *value);

  bool broadcastBuffers() const;
  std::uint64_t replicationFactor() const;

private:
  std::unique_ptr<detail::SessionOptionsImpl> _impl;
  friend Compiler;
};

// Represents an attribute used in a custom operation: popart uses popart::any
// to store the different values
class PopartAttribute {
public:
  // Templating works with g++ but not clang++
  PopartAttribute(const char *name, const int64_t &value);
  PopartAttribute(const char *name, const std::vector<int64_t> &values);
  PopartAttribute(const char *name, const float &value);
  PopartAttribute(const char *name, const std::vector<float> &values);
  PopartAttribute(const char *name, const std::unique_ptr<char[]> &str);
  PopartAttribute(const char *name,
                  const std::vector<std::unique_ptr<char[]>> &strs);

  // Required for opaque pointer
  PopartAttribute(PopartAttribute &&);
  PopartAttribute &operator=(PopartAttribute &&);
  ~PopartAttribute();

  popart::any *getValue();

  const char *name() const { return _name.get(); }

private:
  // Convert a "const char *" to a std::unique_ptr char*
  static std::unique_ptr<const char[]> cStrToUP(const char *name);

  // Use a pointer to circumvent the C++ ABI problems with std::string
  std::unique_ptr<const char[]> _name;

  // Use an opaque pointer to avoid the need for popart headers
  std::unique_ptr<popart::any> _any;
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
                                    const std::vector<std::int64_t> &dims,
                                    const char *overlap = "no_overlap");

  poptorch::TensorId createTensorId(const char *name);

#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<float>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define STRING const char *
#define STRING_VEC std::vector<const char *>
#define NONE
#define ARG(Type, Name) , Type Name
#define POPART_CONST_ARG(Name) , const PopartConstant &Name
#define HOST_SIDE_CONST_ARG(Name) , const HostSideConstant &Name
#define POPART_ATTRIB_VEC_ARG(Name)                                            \
  , std::shared_ptr<std::vector<PopartAttribute>> Name
#define BODY_ARG(Name) NONE

// Create a function decl with the given call and arguments.
#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs)       \
  poptorch::TensorId function(                                                 \
      const std::vector<poptorch::TensorId> &inputs Args);

// Create a function decl with the given call and arguments which returns void.
#define OP_DECL_NO_RETURN(Namespace, FuncName, function, OnnxImpl, Args,       \
                          BodyArgs)                                            \
  void function(const std::vector<poptorch::TensorId> &inputs Args);

#include "SupportedOperations.inc.hpp"

#undef OP_DECL
#undef OP_DECL_NO_RETURN
#undef BODY_ARG
#undef POPART_ATTRIB_VEC_ARG
#undef HOST_SIDE_CONST_ARG
#undef POPART_CONST_ARG
#undef ARG
#undef NONE
#undef STRING_VEC
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

  void addOutputType(OutputType type);

  // This function marks |output| as being read back from the device by the
  // host. |output_mode| determines how frequently that should happen.
  // clang-format off
  // "ALL":  Will return all popart batches.
  // "SUM": Will return the sum of all popart batches (I.E device iterations)
  // "EVERYN": Will return every N batch
  // "FINAL": Will return the last batch only
  // clang-format on
  void addOutputTensor(poptorch::TensorId output,
                       PopartOutputMode output_mode = PopartOutputMode::N,
                       size_t output_return_period = 1,
                       const char *overlap = "no_overlap");

  void setUpInputOp(poptorch::TensorId id, float *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(poptorch::TensorId id, std::int32_t *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(poptorch::TensorId id, bool *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(poptorch::TensorId id, std::int16_t *ptr,
                    const std::vector<std::int64_t> &dims,
                    bool float16 = false);

  // at::ScalarType::Byte
  void setUpInputOp(poptorch::TensorId id, std::uint8_t *ptr,
                    const std::vector<std::int64_t> &dims);

  // at::ScalarType::Char
  void setUpInputOp(poptorch::TensorId id, std::int8_t *ptr,
                    const std::vector<std::int64_t> &dims);

  // at::ScalarType::Byte
  void setUpOutputOp(poptorch::TensorId id, std::uint8_t *ptr,
                     const std::vector<std::int64_t> &dims);

  // at::ScalarType::Char
  void setUpOutputOp(poptorch::TensorId id, std::int8_t *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, float *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, std::int32_t *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, bool *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, std::int16_t *ptr,
                     const std::vector<std::int64_t> &dims);

  // Each std::set of tensors represents all the outputs of a node to set
  // the available memory proportion on. This function loops over the outer
  // vector, so the total number of nodes it will set the proportion on
  // will be inputs.size().
  void setAvailableMemoryProportion(
      const std::vector<std::set<poptorch::TensorId>> &inputs,
      float availableMemoryProportion);

  void setMatMulSerialization(poptorch::TensorId matmul, const char *mode,
                              std::uint64_t factor,
                              std::uint64_t keep_precision);
  void clearActiveIpu();
  void setActiveIpu(std::uint64_t stage_id, std::int64_t phase_id,
                    std::int64_t ipu_id);

  void initSession(const std::vector<Optimizer> &opt,
                   const char *export_proto_filename);
  void setRngState(std::uint64_t seed,
                   const std::vector<std::uint32_t> &rng_state);

  std::vector<std::uint32_t> getRngState() const;
  std::uint64_t getRandomSeed() const;

  void saveExecutableToFile(const char *export_filename) const;
  void compileAndPrepareDevice();
  void loadEngineAndConnectStreams();
  void loadExecutableAndPrepareDevice(const char *import_filename);

  static void
  appendPoptorchMetadataToFile(const char *serialized_poptorch_metadata,
                               size_t metadata_length,
                               const char *export_filename);
  static std::vector<char>
  importPoptorchMetadataFromFile(const char *import_filename);

  poptorch::TensorId
  addCPUCallback(const std::vector<poptorch::TensorId> &inputs,
                 const CallbackMetadata &callback,
                 std::vector<poptorch::PopartType> input_types,
                 std::vector<std::vector<std::size_t>> input_shapes,
                 std::vector<poptorch::PopartType> output_types,
                 std::vector<std::vector<std::size_t>> output_shapes);

  void startSubgraph();

  poptorch::TensorId endForLoop(std::int32_t trip_count,
                                std::int64_t num_outputs,
                                const std::vector<poptorch::TensorId> &inputs);

  void pushNameScope(const char *name);

  void popNameScope();

  poptorch::TensorId addUntypedInputTensor();
  // Write the weights into IPU memory from the pytorch tensor buffers in the
  // model.
  void copyWeightsToDevice(const std::vector<void *> &host_buffers);

  // Read the weights from IPU memory into the pytorch tensor buffers.
  void copyWeightsToHost(const std::vector<void *> &host_buffers);

  // Return the type of the given tensor.
  PopartType getPopartType(poptorch::TensorId id) const;

  // Execute the compiled popart graph using poplar.
  void run();

  // Update the optimizers currently being run by the graph.
  void updateOptimizers(const std::vector<Optimizer> &optimizers);

  std::uint64_t batchPerStep() const;

  // Return the PopART batch dimensions [DeviceIterations * ReplicationFactor *
  // GradientAccumulation]
  std::uint64_t popartBatchDim() const;

  // Take the above and work out how much of it is being returned. ID must be
  // an anchor. The batch dim will be mutated depending on what the anchor is
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

  std::vector<TensorMetadata> optimizerTensorMetadataList() const;

  void
  fillHostOptimizerStateTensorData(const std::vector<void *> &host_buffers);

  void
  writeDeviceOptimizerStateTensorData(const std::vector<void *> &host_buffers);

  std::unique_ptr<char[]> getExecutionInfo() const;

  void addMultiConvPart(const std::vector<poptorch::TensorId> &inputs,
                        const std::vector<int64_t> &dilations,
                        const std::vector<int64_t> &kernel_shape,
                        const std::vector<int64_t> &pads,
                        const std::vector<int64_t> &strides);

  void setMultiConvAvailableMemoryProportions(const std::vector<double> &v);

  void setMultiConvPartialsTypes(const std::vector<int64_t> &partials_types);
  void
  setMultiConvEnableConvDithering(const std::vector<int64_t> &conv_dithering);

  void setMultiConvPlanType(int64_t plan_type);

  void setMultiConvPerConvReservedTiles(int64_t v);

  void setMultiConvCycleBackOff(double c);

  std::vector<poptorch::TensorId> endMultiConv();

  void setAttribute(const char *attribute, const char *key, const char *value);
  void clearAttribute(const char *attribute, const char *key);

  void detachFromDevice();
  void attachToDevice();
  bool isAttachedToDevice() const;

  const std::vector<double> &getInputTimestamps(size_t index) const;
  const std::vector<double> &getInputCompleteTimestamps(size_t index) const;
  const std::vector<double> &getOutputTimestamps(size_t index) const;
  const std::vector<double> &getOutputCompleteTimestamps(size_t index) const;

  // Returns the number of cycles (on replica 0) run by the IPU for the last
  // model run.
  uint64_t getCycleCount() const;

  size_t getNumInputs() const;
  size_t getNumOutputs() const;

private:
  void assertTensorIs(PopartType dataType, poptorch::TensorId id) const;

  // Make sure no overlap is specified for pipelined mode and that the output
  // mode is supported by PopART.
  void verifySettingsForOverlappedIO(PopartOutputMode output_mode);

  std::unique_ptr<detail::CompilerImpl> _impl;

  // Store the cycle account of last run, if the relevant option is enabled,
  // otherwise no_cycles
  int64_t _cycle_count;
  static constexpr int64_t no_cycles = -1;
  static constexpr const char *poptorch_opaque_name = "poptorch";
};

} // namespace poptorch
