// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/CompilerHelpers.hpp"
#include <poplin/ConvParams.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/Broadcast.hpp>

namespace pe = popops::expr;

namespace poptorch_ir {

void conv::lowerToPoplar(CompilerContext &context) {
  // Convert the inputs to poplar tensors.
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor weight = context.fromSsa(this->weight());

  // Extract the attributes into something usable. Oddly/annoyingly the API
  // requests unsigned but stores as size_t.
  std::vector<unsigned> strides = convertIntArray<unsigned>(this->strides());
  std::vector<unsigned> padding = convertIntArray<unsigned>(this->padding());
  std::vector<unsigned> dilation = convertIntArray<unsigned>(this->dilation());
  std::vector<unsigned> output_padding =
      convertIntArray<unsigned>(this->output_padding());
  const std::size_t groups = this->groups();

  std::vector<std::size_t> input_field_shape;
  std::vector<std::size_t> weight_shape;

  for (std::size_t i = 0; i < weight.rank() - 2; ++i) {
    input_field_shape.push_back(input.dim(i + 2));
    weight_shape.push_back(weight.dim(i + 2));
  }

  // We will create this as part of the planning stage and cache it somewhere in
  // the future.
  poplin::ConvParams params(
      poplar::FLOAT /*input type*/, input.dim(0) /*batch_size*/,
      input_field_shape /*input field shape*/, weight_shape /*kernel shape*/,
      weight.dim(1) /*input channels per group*/,
      weight.dim(0) / groups /*output channels per group*/, groups /*groups*/);

  params.inputTransform.paddingLower = padding;
  params.inputTransform.paddingUpper = padding;

  params.kernelTransform.dilation = dilation;

  params.outputTransform.stride = strides;
  params.outputTransform.paddingLower = output_padding;
  params.outputTransform.paddingUpper = output_padding;

  weight = weight.reshapePartial(
      0, 2, {groups, weight.dim(0) / groups, weight.dim(1)});

  poplar::Tensor output = poplin::convolution(context.graph, input, weight,
                                              params, false, context.seq);

  // MLIR value converts to bool so optional biases will be ignored.
  if (this->bias()) {
    poplar::Tensor bias = context.fromSsa(this->bias());
    poplin::addBias(context.graph, output, bias, context.seq);
  }

  context.tensors.insert({this->result(), output});
}

void matmul::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor input2 = context.fromSsa(this->in2());

  // We are always going to want to add a batch dim, even if one exists it's
  // only 1 so should be free.
  input1 = input1.expand({0});
  input2 = input2.expand({0});

  // Broadcast Vector-Vector.
  if (input1.rank() == 2 && input2.rank() == 2) {
    input1 = input1.expand({1});
    input2 = input2.expand({2});
  }

  // Broadcast Matrix-Vector to Batch-matrix.
  if (input1.rank() == 2) {
    input1 = input1.expand({2});
  }

  if (input2.rank() == 2) {
    input2 = input2.expand({2});
  }

  if (input1.rank() > 3) {
    input1 = input1.flatten(0, input1.rank() - 2);
  }

  if (input2.rank() > 3) {
    input2 = input2.flatten(0, input2.rank() - 2);
  }

  poplar::Tensor out = poplin::matMulGrouped(context.graph, input1, input2,
                                             context.seq, input1.elementType());

  // Reshape into target shape. Jump through some array ref stuff.
  auto llvm_ref =
      this->result().getType().cast<mlir::RankedTensorType>().getShape();
  poplar::ArrayRef<std::uint64_t> poplar_ref{
      (const std::uint64_t *)llvm_ref.data(), llvm_ref.size()};

  out = out.reshape(poplar_ref);

  // Record the result.
  context.tensors.insert({this->result(), out});
}

void addmm::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor mat1 = context.fromSsa(this->mat1());
  poplar::Tensor mat2 = context.fromSsa(this->mat2());
  float beta = this->beta().convertToFloat();
  float alpha = this->alpha().convertToFloat();

  poplar::Tensor out = poplin::matMul(context.graph, mat1, mat2, context.seq,
                                      mat1.elementType());

  // alpha * mm
  if (alpha != 1.0f) {
    auto expr = pe::Mul(pe::_1, pe::Const(alpha));
    popops::mapInPlace(context.graph, expr, {out}, context.seq);
  }

  if (beta != 1.0f) {
    poputil::broadcastToMatch(out, input);
    // scaledAdd(alpha * mm, input, beta)
    popops::scaledAddTo(context.graph, out, input, beta, context.seq);
  } else {
    popops::addInPlace(context.graph, out, input, context.seq);
  }

  context.tensors[this->result()] = out;
}

} // namespace poptorch_ir