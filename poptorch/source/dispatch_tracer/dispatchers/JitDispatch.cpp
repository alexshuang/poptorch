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

void JITDispatch::createGraph(const std::vector<at::Tensor> &inputs,
                              const std::vector<at::Tensor> &parameters) {
  auto add_input = [&](at::Tensor &tensor) {
    torch::jit::Value *value = graph.addInput(tensor.name());
    value->inferTypeFrom(tensor);
    _mapper.addTensor(tensor, value);
  };

  // Add any inputs.
  for (at::Tensor tensor : inputs) {
    add_input(tensor);
  }

  // Add the parameters.
  for (at::Tensor tensor : parameters) {
    at::ScalarType type = tensor.scalar_type();
    // PopART doesn't allow non-floating point variables so add them as
    // constants instead. These will be deleted from parameters and buffers
    // in python before passed to lowering.
    if (!at::isFloatingType(type)) {
      if (type == at::ScalarType::Long) {
        tensor = tensor.to(at::ScalarType::Int);
      }
      auto *new_node = tensorToConstant(&graph, tensor);
      _mapper.addTensor(tensor, new_node->output());
    } else {
      add_input(tensor);
    }
  }
}

namespace {
struct ConstructElement {
  torch::jit::Node *node;
  std::vector<c10::TypePtr> element_types;

  explicit ConstructElement(torch::jit::Node *_node) : node(_node) {}
};
} // namespace

void JITDispatch::markOutputs(
    const std::vector<at::Tensor> &outputs,
    const std::vector<at::Tensor> &persistent_data_storage,
    const std::string &output_structure) {
  // 'persistent_data_storage' is only needed by the MLIR dispatcher.
  UNUSED(persistent_data_storage);

  int64_t output_num = 0;
  auto output_it = std::begin(outputs);
  std::vector<ConstructElement> construct_elem_stack;

  for (size_t i = 0; i < output_structure.size(); i++) {
    char s = output_structure[i];
    if (s == '(' || s == '[') {
      auto *construct_node = graph.create(s == '(' ? c10::prim::TupleConstruct
                                                   : c10::prim::ListConstruct);
      construct_elem_stack.emplace_back(construct_node);
      continue;
    }
    torch::jit::Value *val = nullptr;
    if (s == ')' || s == ']') {
      auto &construct_elem = construct_elem_stack.back();
      auto *node = construct_elem.node;
      if (s == ')') {
        node->output()->setType(
            c10::TupleType::create(construct_elem.element_types));
      } else {
        if (construct_elem.element_types.empty()) {
          node->output()->setType(c10::ListType::create(at::NoneType::get()));
        } else {
          ERROR_ON_MSG(
              std::adjacent_find(
                  std::begin(construct_elem.element_types),
                  std::end(construct_elem.element_types),
                  [](const c10::TypePtr &a, const c10::TypePtr &b) {
                    return a->kind() != b->kind();
                  }) != std::end(construct_elem.element_types),
              "All elements in an output list must have the same type");
          node->output()->setType(
              c10::ListType::create(construct_elem.element_types[0]));
        }
      }
      construct_elem_stack.pop_back();
      graph.insertNode(node);
      val = node->output();
    } else {
      const auto &tensor = *output_it;
      ++output_it;
      val = _mapper.getValueForTensor(tensor);
      ERROR_ON_MSG(
          val == nullptr,
          "Internal: graph output tensor not present in the value mapper");

      logging::trace(
          "[TRACING-2][JIT] Graph output: Tensor ptr {}, jit ir %{} {}",
          reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
          val->debugNameBase(), toString(tensor));
    }
    if (construct_elem_stack.empty()) {
      graph.registerOutput(val);
    } else {
      auto &next_up_stack = construct_elem_stack.back();
      next_up_stack.node->addInput(val);
      next_up_stack.element_types.push_back(val->type());
    }
    if (s == 'x') {
      // For now, disable overlapping host IO on every output
      auto overlap_symbol = getOverlapSymbol("_for_output", output_num);
      const std::string value_str = "no_overlap";
      graph.return_node()->s_(overlap_symbol, value_str);

      output_num++;
    }
  }
  ERROR_ON_MSG(output_it != std::end(outputs), "Didn't consume all outputs");
  logging::trace("[TRACING-2][JIT] Graph after marking outputs\n{}\n", graph);
}

const at::Tensor &JITDispatch::copyInplace(const at::Tensor &self,
                                           const at::Tensor &other) {
  if (other.unsafeGetTensorImpl()->is_wrapped_number()) {
    torch::jit::Value *val = graph.insertConstant(other);
    _mapper.addTensor(other, val);
  }

  ValueMapper::TrackedTensor *dest = _mapper.rawTensorRecord(self);
  ValueMapper::TrackedTensor *src = _mapper.rawTensorRecord(other);
  ERROR_ON(dest == nullptr);
  ERROR_ON(src == nullptr);

  logging::trace("[TRACING-2][JIT] copyInplace: src tensor {}, dest tensor {}",
                 static_cast<void *>(other.unsafeGetTensorImpl()),
                 static_cast<void *>(self.unsafeGetTensorImpl()));

  torch::jit::Value *copy;
  if (self.scalar_type() == other.scalar_type()) {
    copy = createIdentity(&graph, {src->jit})->output();
    copy->inferTypeFrom(self);
  } else {
    copy = createCast(&graph, src->jit, self.scalar_type())->output();
  }

  dest->jit = copy;
  dest->is_empty = src->is_empty;

  if (_mapper.isHalfTensor(other)) {
    _mapper.markHalfTensor(self);
  }

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

// aten::detach(Tensor(a) self) -> (Tensor(a))
at::Tensor JITDispatch::detach(const at::Tensor &self) { return self; }

void JITDispatch::setCurrentCodeLocation(
    const torch::jit::SourceRange &source_location) {
  setCurrentPythonCodeLocation(source_location);
}

// Convert the operation into our normal IR style operation.
void JITDispatch::canonicaliseAndFixOutput(const c10::FunctionSchema &schema,
                                           c10::Stack &stack,
                                           torch::jit::Node **node,
                                           ValueMapper &mapper) {
  torch::jit::Node *new_node = canonicalise(schema, *node, graph, false);
  *node = new_node;

  logging::trace("[TRACING-2][JIT] Post canonicalisation {}", *new_node);

  // Fix up the outputs.
  std::uint32_t output_index = 0;
  for (c10::IValue value : stack) {
    // PopART doesn't always match these 1:1.
    if (output_index >= new_node->outputs().size()) {
      break;
    }

    // Start tracking the output tensors, i.e. add them to the value mapper.
    torch::jit::Value *val = new_node->output(output_index);
    // Check whether the handler replaced this value.
    auto *replacement = wasReplaced(val);
    if (replacement != nullptr) {
      val = replacement;
    }

    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();

      if (mapper.isHalfTensor(tensor)) {
        val->inferTypeFrom(tensor.to(at::ScalarType::Half));
      } else {
        auto st = tensor.scalar_type();
        auto st_coerced = coerceToSupportedType(st);
        if (st != st_coerced) {
          logging::warn("[TRACING-2][JIT] Type coerced from {} to {}", st,
                        st_coerced);
          val->inferTypeFrom(tensor.to(st_coerced));
        } else {
          val->inferTypeFrom(tensor);
        }
      }
      _mapper.addTensor(tensor, val);

      logging::trace("[TRACING-2][JIT] Output: Tensor ptr {}, jit ir %{} {}",
                     reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                     val->debugNameBase(), toString(tensor));
    } else if (value.isTensorList()) {
      logging::trace("[TRACING-2][JIT] Output tensor list: jit ir %{}",
                     val->debugName());
      val->setType(value.type()->expect<c10::ListType>());
      auto tensor_list = value.toTensorVector();
      // Always insert list unpack if output value is a list.
      auto *unpack = graph.createListUnpack(val, tensor_list.size());
      graph.insertNode(unpack);

      for (size_t i = 0; i < tensor_list.size(); ++i) {
        at::Tensor tensor = tensor_list.at(i);
        val = unpack->output(i);
        _mapper.addTensor(tensor, val);
        logging::trace("[TRACING-2][JIT] Output tensor list element: Tensor "
                       "ptr {}, jit ir %{} {}",
                       reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                       val->debugNameBase(), toString(tensor));
      }
    }

    output_index++;
  }
}

void JITDispatch::fallback(const c10::OperatorHandle &initial_op,
                           c10::Stack *stack) {
  c10::Dispatcher &dispatcher = c10::Dispatcher::singleton();

  const c10::FunctionSchema &initial_schema = initial_op.schema();
  // Run through the schema to find out if one of the operators is supposed to
  // be inplace, this could be the 'out' argument of a non-inplace op.
  c10::intrusive_ptr<at::TensorImpl> inplace_tensor =
      getInplaceArgument(*stack, initial_schema);
  bool is_truly_inplace = isTrulyInplace(*stack, initial_schema);

  c10::OperatorHandle op = initial_op;
  bool op_changed = getOutplaceOpHandle(op, dispatcher);
  const c10::FunctionSchema &schema = op.schema();

  // Create a fake IR node for us to target using the schema.
  torch::jit::Node *node = lowerFromSchema(schema, stack, graph, _mapper);

  if (shouldRunOnCpu(is_truly_inplace || op_changed, schema.name())) {
    HalfFloatConverter hf_conv(stack, schema, _mapper);
    hf_conv.pre();
    // Call the CPU version to get the output shape
    dispatcher.callBoxed(op, stack);
    hf_conv.post();
  } else {
    // The Op is in place: we don't need to run the CPU version.
    // Just clear the stack and keep the first input.
    at::Tensor t = stack->at(0).toTensor();
    stack->clear();
    stack->push_back(t);
  }
  // Fix the fake tensor so it can still work with our canonicalisation
  // functions which check the output.
  fixNodeOutput(node, *stack, _mapper);
  logging::trace("[TRACING-2][JIT] Pre canonicalisation {}", *node);

  // Run our normal canonicalisation passes on it.
  // The original node will be deleted but replaced with a new node.
  canonicaliseAndFixOutput(schema, *stack, &node, _mapper);

  // Annotate for loops as subgraphs.
  annotateSubgraphsDispatch(&graph, node);

  logging::trace("[TRACING-2][JIT] Post canonicalisation and fix output {}",
                 *node);

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
    record->is_empty = false;
    if (_mapper.isHalfTensor(output)) {
      _mapper.markHalfTensor(inplace);
    }
  }

  logging::trace("[TRACING-2][JIT] Graph after interception of {}=\n{}\n",
                 schema.name(), graph);
}

} // namespace poptorch
