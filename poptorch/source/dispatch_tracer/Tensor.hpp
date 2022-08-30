// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_TENSOR_HPP_
#define POPTORCH_DISPATCH_TENSOR_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch {

using Buffer = poptorch_ir::Buffer;

struct IpuTensorImpl;
class ValueMapper;

// We cannot reliably store references to TensorImpls, as they may be deleted
// without notice if the tensor goes out of scope. We can't hold an
// intrusive_ptr or weak_intrusive_ptr to them either, as they may be
// ExclusivelyOwned, which will trigger an assert if an intrusive_ptr
// exists at the time of destruction. So instead, we store the details we
// care about in this structure, which is referenced by a shared_ptr from
// IpuTensorImpl and also from the ValueMapper.
struct IpuTensorDetails {
  // Raw pointer to the parent IpuTensorImpl. If the parent is destroyed,
  // the details of the tensor will be copied into this structure so they can
  // still be accessed through the ValueMapper.
  IpuTensorImpl *parent;
  ValueMapper *mapper = nullptr;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::string name;
  bool is_parameter = false;
  Buffer host_buffer;
  std::optional<std::uint64_t> alias_of;

  int64_t dim();
  c10::IntArrayRef sizesArrayref();
  c10::IntArrayRef stridesArrayref();
  int64_t numel();
};

uint64_t tensorImplDataSize(const at::TensorImpl &impl);

// Mark an IPU tensor as being a parameter or not.
void setIsParameter(at::Tensor &tensor, bool is_parameter);

// Return true if the given IPU tensor is a parameter.
bool isParameter(const at::Tensor &tensor);

// Return true if the given IPU tensor implementation is a parameter.
bool isParameter(const at::TensorImpl &tensor);

// Return the data size in bytes of the given at::Tensor.
uint64_t tensorDataSize(const at::Tensor &tensor);

// Return the tensor ID of the given IPU tensor.
uint64_t ipuTensorId(const at::Tensor &tensor);

// Return the tensor ID of the given IPU tensor implementation.
uint64_t ipuTensorId(const at::TensorImpl &tensor);

// Return true if the given at::Tensor is an IPU tensor.
bool isIpuTensor(const at::Tensor &tensor);

// Return a string containing the given tensor's metadata (device, shape, etc).
std::string str(const at::Tensor &tensor);

// Make a copy of the given CPU tensor's content and store it inside the given
// IPU tensor.
void copyDataFromCpuSource(at::Tensor &ipu_tensor, const at::Tensor &cpu_src);

// Returns a reference to the CPU buffer of the given IPU tensor.
Buffer &getHostBuffer(const at::Tensor &ipu_tensor);

// Returns a reference to the CPU buffer of the given IPU tensor implementation.
Buffer &getHostBuffer(const at::TensorImpl &ipu_tensor);

std::shared_ptr<IpuTensorDetails>
getTensorDetails(const at::TensorImpl &ipu_tensor_impl);

inline std::shared_ptr<IpuTensorDetails>
getTensorDetails(const at::Tensor &ipu_tensor) {
  return getTensorDetails(*ipu_tensor.unsafeGetTensorImpl());
}

void errorOnZeroSizedTensor(const at::Tensor &tensor);

void initHostBuffer(at::Tensor &ipu_tensor);

/** Host-side storage for `ipu` tensors.
 *
 *  This allows the user to convert tensors and modules to `ipu` using
 *  `t.to("ipu")` even when the dispatcher is off, and even outside eager mode.
 *
 *  We simply copy the tensor in to our ownership, then when we go to load and
 *  execute an executable, we can upload these tensors to the device. We'll
 *  also retrieve them from the device when the user copies a tensor back to the
 *  CPU (`t.to("cpu")`).
 */
class TensorStore {
public:
  TensorStore() = default;
  TensorStore(const TensorStore &) = delete;
  TensorStore(TensorStore &&) = delete;

  // Create a new IPU tensor.
  at::Tensor
  allocateTensor(c10::IntArrayRef sizes,
                 c10::optional<at::ScalarType> dtype = c10::nullopt,
                 c10::optional<at::Device> device = c10::nullopt,
                 c10::optional<at::Layout> layout = c10::nullopt,
                 c10::optional<bool> pin_memory = c10::nullopt,
                 c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

  // Create a new IPU tensor by copying the data from the given CPU tensor in to
  // the new IPU tensor's host_buffer. Input tensor is a CPU tensor, returns an
  // IPU tensor.
  at::Tensor copyCpuTensorAsIpuTensor(const at::Tensor &cpu_tensor);

private:
  poptorch_ir::TensorId _next_tensor_id{1};
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_TENSOR_HPP_
