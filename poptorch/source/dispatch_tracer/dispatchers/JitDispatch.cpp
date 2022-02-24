// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "JitDispatch.hpp"

#include <string>
#include <unordered_set>
#include <utility>

#include "../../PoptorchSymbols.hpp"
#include "../../popart_canonicalization/PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../CommonHelperFunctions.hpp"

namespace poptorch {

namespace {

void fixFakeTargetOutput(torch::jit::Node *fake_target,
                         const c10::Stack &stack) {
  std::uint32_t index = 0;
  for (c10::IValue value : stack) {
    // Add any missing outputs. They frequently return scalars which we just
    // ignore here as our canonicalisation only returns tensors.
    while (index >= fake_target->outputs().size()) {
      fake_target->addOutput();
    }

    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();
      // Sometimes "Tensors" are actually "Not tensors" but still stored as a
      // tensor and will assert in the infer type.
      if (tensor.sizes().size() == 1 && tensor.sizes()[0] == 0) {
        continue;
      }

      torch::jit::Value *val = fake_target->output(index);
      val->inferTypeFrom(tensor);
    } else if (value.isTensorList()) {
      auto list_type = value.type()->expect<c10::ListType>();
      torch::jit::Value *val = fake_target->output(index);
      val->setType(list_type);
    }
    index++;
  }
}

} // namespace

void JITDispatch::createGraph(const std::vector<at::Tensor> &inputs,
                              const std::vector<at::Tensor> &parameters) {
  // We build up the torch IR graph as well.
  auto add = [&](at::Tensor &tensor) {
    torch::jit::Value *value = graph.addInput(tensor.name());
    value->inferTypeFrom(tensor);

    _mapper.addTensor(tensor, value);
  };

  // Add any inputs.
  for (at::Tensor tensor : inputs) {
    add(tensor);
  }

  // Add the parameters.
  for (at::Tensor tensor : parameters) {
    add(tensor);
  }
}

void JITDispatch::markOutputs(
    const std::vector<at::Tensor> &outputs,
    const std::vector<at::Tensor> &persistent_data_storage, bool output_tuple) {
  // 'persistent_data_storage' is only needed by the MLIR dispatcher.
  UNUSED(persistent_data_storage);

  int64_t output_num = 0;
  torch::jit::Node *construct_node = nullptr;
  std::vector<c10::TypePtr> construct_element_types;
  if (output_tuple) {
    construct_node = graph.create(c10::prim::TupleConstruct);
    graph.insertNode(construct_node);
    graph.registerOutput(construct_node->output());
  }
  for (const at::Tensor &tensor : outputs) {
    torch::jit::Value *val = _mapper.getValueForTensor(tensor);
    ERROR_ON_MSG(
        val == nullptr,
        "Internal: graph output tensor not present in the value mapper");

    logging::trace(
        "[TRACING-2][JIT] Graph output: Tensor ptr {}, jit ir %{} {}",
        reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
        val->debugNameBase(), toString(tensor));

    if (construct_node != nullptr) {
      construct_element_types.push_back(val->type());
      construct_node->addInput(val);
    } else {
      graph.registerOutput(val);
    }

    // For now, disable overlapping host IO on every output
    auto overlap_symbol = getOverlapSymbol("_for_output", output_num);
    const std::string value_str = "no_overlap";
    graph.return_node()->s_(overlap_symbol, value_str);

    output_num++;
  }
  if (construct_node != nullptr) {
    construct_node->output()->setType(
        c10::TupleType::create(construct_element_types));
  }
  logging::trace("[TRACING-2][JIT] Graph after marking outputs\n{}\n", graph);
}

const at::Tensor &JITDispatch::copyInplace(const at::Tensor &self,
                                           const at::Tensor &other) {
  if (other.unsafeGetTensorImpl()->is_wrapped_number()) {
    torch::jit::Value *val = graph.insertConstant(other);
    _mapper.addTensor(other, val, true);
  }

  ValueMapper::TrackedTensor *dest = _mapper.rawTensorRecord(self);
  ValueMapper::TrackedTensor *src = _mapper.rawTensorRecord(other);
  ERROR_ON(dest == nullptr);
  ERROR_ON(src == nullptr);

  dest->jit = src->jit;
  dest->is_const = src->is_const;

  return self;
}

// _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device?
// device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat?
// memory_format=None) -> Tensor
// Apears in 1.10.
at::Tensor JITDispatch::toCopyInplace(
    const at::Tensor &self, c10::optional<at::ScalarType> /*dtype*/,
    c10::optional<at::Layout> /*layout*/, c10::optional<at::Device> /*device*/,
    c10::optional<bool> /*pin*/, c10::optional<c10::MemoryFormat> /*fmt*/) {
  // TODO(T45469): Renable.
  return self;
}

void JITDispatch::registerEmptyTensor(const at::Tensor &tensor) {
  torch::jit::Node *n =
      graph.createUninitialized(c10::TensorType::create(tensor));
  _mapper.addTensor(tensor, n->output(0), true);
}

at::Tensor JITDispatch::convolution(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, const at::IntArrayRef stride,
    const at::IntArrayRef padding, const at::IntArrayRef dilation,
    const bool transposed, const at::IntArrayRef output_padding,
    const int64_t groups) {

  c10::OperatorName name{"aten::convolution", ""};

  c10::Dispatcher &dispatcher = c10::Dispatcher::singleton();
  c10::OperatorHandle op = *dispatcher.findOp(name);
  const c10::FunctionSchema &schema = op.schema();

  // Turn our convolution inputs into a generic stack input.
  c10::Stack stack;
  stack.push_back(input);
  stack.push_back(weight);
  stack.push_back(bias);
  stack.push_back(stride);
  stack.push_back(padding);
  stack.push_back(dilation);
  stack.push_back(transposed);
  stack.push_back(output_padding);
  stack.push_back(groups);

  // Add it to the graph as a normal output.
  torch::jit::Node *fake_target =
      lowerFromSchema(schema, &stack, graph, _mapper);

  // Get the handler for the convolution.
  auto op_typed = op.typed<decltype(at::convolution)>();

  // Rerun on CPU to see the sizes.
  at::Tensor output = op_typed.redispatch(
      c10::DispatchKeySet({c10::DispatchKey::AutogradOther}), input, weight,
      bias, stride, padding, dilation, transposed, output_padding, groups);

  // Add the output into the stack.
  stack.clear();
  stack.push_back(output);

  // Fix the fake tensor so it can still work with our canonicalisation
  // functions which check the output.
  fixFakeTargetOutput(fake_target, stack);

  logging::trace("[TRACING-2][JIT] Node tensor output size: ={}",
                 output.sizes());

  // Run our normal canonicalisation passes on it.
  canonicaliseAndFixOutput(schema, stack, &fake_target);

  return output;
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
at::Tensor JITDispatch::detach(const at::Tensor &self) { return self; }

// Convert the operation into our normal IR style operation.
void JITDispatch::canonicaliseAndFixOutput(const c10::FunctionSchema &schema,
                                           c10::Stack &stack,
                                           torch::jit::Node **fake_target) {
  torch::jit::Node *new_node = canonicalise(schema, *fake_target, graph, false);

  // Point fake_target at the new node
  *fake_target = new_node;

  logging::trace("[TRACING-2][JIT] Post canonicalisation {}", *new_node);

  // Fix up the outputs.
  std::uint32_t output_index = 0;
  for (c10::IValue value : stack) {
    // PopART doesn't always match these 1:1.
    if (output_index >= new_node->outputs().size()) {
      break;
    }

    // Start tracking the tensors, i.e. add them to the value mapper.
    torch::jit::Value *val = new_node->output(output_index);
    // Check whether the handler replaced this value.
    auto *replacement = wasReplaced(val);
    if (replacement != nullptr) {
      val = replacement;
    }

    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();

      auto st = tensor.scalar_type();
      auto st_coerced = coerceToSupportedType(st);
      if (st != st_coerced) {
        logging::warn("[TRACING-2][JIT] Type coerced from {} to {}", st,
                      st_coerced);
        val->inferTypeFrom(tensor.to(st_coerced));
      } else {
        val->inferTypeFrom(tensor);
      }
      _mapper.addTensor(tensor, val);

      logging::trace("[TRACING-2][JIT] Output: Tensor ptr {}, jit ir %{} {}",
                     reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                     val->debugNameBase(), toString(tensor));
    } else if (value.isTensorList()) {
      val->setType(value.type()->expect<c10::ListType>());
      auto tensor_list = value.toTensorVector();
      // Always insert list unpack if output value is a list.
      auto *unpack = graph.createListUnpack(val, tensor_list.size());
      graph.insertNode(unpack);

      for (size_t i = 0; i < tensor_list.size(); ++i) {
        at::Tensor tensor = tensor_list.at(i);
        val = unpack->output(i);
        _mapper.addTensor(tensor, val);
        logging::trace("[TRACING-2][JIT] Output: Tensor ptr {}, jit ir %{} {}",
                       reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                       val->debugNameBase(), toString(tensor));
      }
    }

    output_index++;
  }
}

namespace {
// The in-place versions of these functions break the rule that in place ops
// don't change the shape of the tensor they operate on.
const std::unordered_set<std::string> in_place_reshapes{
    "aten::squeeze", "aten::unsqueeze", "aten::transpose"};
} // namespace

void JITDispatch::fallback(const c10::OperatorHandle &initial_op,
                           c10::Stack *stack) {
  std::string name = initial_op.schema().name();
  std::string overload = initial_op.schema().overload_name();
  c10::Dispatcher &dispatcher = c10::Dispatcher::singleton();
  c10::OperatorHandle op = initial_op;

  // Run through the schema to find out if one of the operators is supposed to
  // be inplace, this could be the 'out' argument of a non-inplace op.
  c10::intrusive_ptr<at::TensorImpl> inplace_tensor =
      getInplaceArgument(*stack, initial_op.schema());
  bool is_truly_inplace = isTrulyInplace(*stack, initial_op.schema());
  // If ends with '_', it's inplace. Remove the "_" and use the outplace version
  // instead.
  bool name_indicates_is_inplace = name[name.size() - 1] == '_';

  if (name_indicates_is_inplace) {
    // These are special cases because there is no zero / fill.
    if (name == "aten::zero_") {
      name = "aten::zeros_like";
    } else if (name == "aten::fill_") {
      name = "aten::full_like";
    } else {
      name.erase(name.end() - 1, name.end());
    }
    auto opt_op = dispatcher.findOp({name, overload});
    if (opt_op) {
      op = *opt_op;
    } else {
      op = *dispatcher.findOp({name, ""});
    }
  }

  const c10::FunctionSchema &schema = op.schema();

  // Create a fake IR node for us to target using the schema.
  torch::jit::Node *fake_target =
      lowerFromSchema(schema, stack, graph, _mapper);

  if ((name_indicates_is_inplace || is_truly_inplace) &&
      in_place_reshapes.count(name) == 0) {
    // The Op is in place: we don't need to run the CPU version.
    // Just clear the stack and keep the first input.
    at::Tensor t = stack->at(0).toTensor();
    stack->clear();
    stack->push_back(t);
  } else {
    // Convert any halves to floats
    for (size_t i = 0; i < stack->size(); i++) {
      auto &value = stack->at(i);
      if (!value.isTensor()) {
        continue;
      }
      auto tensor = value.toTensor();
      if (!tensor.defined()) {
        continue;
      }
      auto tt = value.type()->cast<at::TensorType>();
      if (tt->scalarType() == at::ScalarType::Half) {
        at::Tensor t = value.toTensor();
        value = t.toType(at::ScalarType::Float);
      }
    }
    // Call the CPU version to get the output shape
    dispatcher.callBoxed(op, stack);
  }

  // Fix the fake tensor so it can still work with our canonicalisation
  // functions which check the output.
  fixFakeTargetOutput(fake_target, *stack);
  logging::trace("[TRACING-2][JIT] Pre canonicalisation {}", *fake_target);

  // Run our normal canonicalisation passes on it.
  // The original fake_target node will be deleted but replaced with a new node.
  canonicaliseAndFixOutput(schema, *stack, &fake_target);

  logging::trace("[TRACING-2][JIT] Post canonicalisation and fix output {}",
                 *fake_target);

  std::size_t i = 0;
  for (c10::IValue value : *stack) {
    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();
      logging::trace(
          "[TRACING-2][JIT] Node tensor output at index {} size: ={}", i++,
          tensor.sizes());
    } else {
      logging::trace("[TRACING-2][JIT] Node scalar output at index {}", i++);
    }
  }

  // Switcheroo the output so the inplace tensor reference is now pointing to
  // the output.
  if (inplace_tensor) {
    at::Tensor inplace{inplace_tensor};
    at::Tensor output = stack->at(0).toTensor();

    // Get the jit value we are tracking for the output.
    torch::jit::Value *value = _mapper.getValueForTensor(output);

    // Overwrite the inplace tensor with that jit. Now a reference to the
    // inplace tensor correctly points to this outplace value.
    ValueMapper::TrackedTensor *record = _mapper.rawTensorRecord(inplace);
    ERROR_ON_MSG(
        !record,
        "[TRACING-2][JIT] Inplace op is not tracking inplace argument");
    record->jit = value;
  }

  logging::trace("[TRACING-2][JIT] Graph after interception of {}=\n{}\n",
                 schema.name(), graph);
}

} // namespace poptorch
