// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {
torch::jit::Node *poolingHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  torch::jit::Symbol kind = node->kind();
  // aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[]
  // padding, int[] dilation, bool ceil_mode) -> Tensor
  //
  // aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[]
  //                   padding, bool ceil_mode, bool count_include_pad,
  //                   int? divisor_override) -> Tensor
  auto kernel_size = constantToLongVec(node->input(1)->node());
  auto stride = constantToLongVec(node->input(2)->node());
  auto padding = constantToLongVec(node->input(3)->node());

  // Pytorch gives the padding as being the amount to pad in both
  // directions. Popart two arguments for each axis, the amount to pad in
  // each direction along that axis. In the form (Axis0Left, AxisNLeft...,
  // Axis0Right, AxisNRight) where left and right refer to the direction
  // along the axis to add zeros to.
  const std::size_t num_pads = padding.size();
  for (std::size_t pad_index = 0; pad_index < num_pads; ++pad_index) {
    padding.push_back(padding[pad_index]);
  }

  if (kind == c10::aten::max_pool1d || kind == c10::aten::max_pool2d ||
      kind == c10::aten::max_pool3d) {
    auto dilations = constantToLongVec(node->input(4)->node());
    auto ceil_mode = constantToLong(node->input(5)->node());

    return createMaxpool(graph, {node->input(0)}, 1, kernel_size, ceil_mode,
                         dilations, padding, 0, stride);
  }

  // divisor_override is ignored for now due to not being supported directly in
  // popart.
  auto ceil_mode = constantToLong(node->input(4)->node());

  torch::jit::Value *new_value = node->input(0);

  bool count_include_pad = constantToBool(node->input(5)->node());
  // count_include_pad isn't supported in PopART so we check and pad manually if
  // the average pool is supposed to include the padding in its average.
  if (count_include_pad) {
    new_value = createConstantPad(graph, new_value, padding, 0.f)->output();
    // Ensure that padding isn't added twice.
    padding = {};
  }

  // popart only supports float types for avgpool
  auto input_type = getNodeScalarType(new_value);

  if (input_type == c10::kFloat) {
    return createAveragepool(graph, {new_value}, kernel_size, ceil_mode, 0,
                             padding, stride);
  }

  // all ather types require casting via float
  auto new_node = createCast(graph, new_value, c10::kFloat);
  new_node = createAveragepool(graph, {new_node->output()}, kernel_size,
                               ceil_mode, 0, padding, stride);
  return createCast(graph, new_node->output(), input_type);
}

torch::jit::Node *adaptivePoolingHandler(torch::jit::Graph *graph,
                                         torch::jit::Node *node) {
  // aten::adaptive_avg_pool1d(Tensor self, int[] output_size) -> Tensor
  // aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor
  // aten::adaptive_avg_pool3d(Tensor self, int[] output_size) -> Tensor

  torch::jit::Value *x = node->input(0);
  std::vector<std::int64_t> output_shape =
      constantToLongVec(node->input(1)->node());
  std::size_t n_output_dims = output_shape.size();

  std::vector<std::int64_t> input_shape = shapeFromTensor(x);
  std::size_t input_offset = input_shape.size() - n_output_dims;

  std::vector<std::int64_t> stride(n_output_dims);
  std::vector<std::int64_t> kernel_shape(n_output_dims);
  for (std::size_t i = 0; i < n_output_dims; i++) {
    std::int64_t in_dim = input_shape[input_offset + i];
    std::int64_t out_dim = output_shape[i];
    // This matches PyTorch's implementation as long as each input dim is
    // divisible by the corresponding output dim. If this is not the case, the
    // shape will be correct but the output will differ.
    if (in_dim % out_dim != 0) {
      std::stringstream ss;
      ss << "Input dim " << i << " (" << in_dim
         << ") is not divisible by the "
            "corresponding output dim ("
         << out_dim
         << "). The results will differ "
            "numerically from PyTorch's implementation.";
      ERROR(ss.str());
    }
    stride[i] = in_dim / out_dim;
    kernel_shape[i] = in_dim - (out_dim - 1) * stride[i];
  }

  std::vector<std::int64_t> padding(n_output_dims * 2, 0);
  return createAveragepool(graph, {x}, kernel_shape, 0, 0, padding, stride);
}

} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::max_pool1d, poolingHandler,
    c10::aten::avg_pool1d, poolingHandler,
    c10::aten::max_pool2d, poolingHandler,
    c10::aten::avg_pool2d, poolingHandler,
    c10::aten::max_pool3d, poolingHandler,
    c10::aten::avg_pool3d, poolingHandler,
    c10::aten::adaptive_avg_pool1d, adaptivePoolingHandler,
    c10::aten::adaptive_avg_pool2d, adaptivePoolingHandler,
    c10::aten::adaptive_avg_pool3d, adaptivePoolingHandler);
// clang-format on

} // namespace poptorch
