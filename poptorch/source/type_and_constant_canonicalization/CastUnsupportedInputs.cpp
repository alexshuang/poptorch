// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "torch/csrc/jit/ir/ir.h"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {
namespace {
void processInputTensor(torch::jit::Graph *graph, torch::jit::Value *input) {
  auto tensor_type = input->type()->expect<c10::TensorType>();
  auto current_type = tensor_type->scalarType().value();

  at::ScalarType new_type;

  if (current_type == at::ScalarType::Double) {
    new_type = at::ScalarType::Float;
  } else if (current_type == at::ScalarType::Long) {
    new_type = at::ScalarType::Int;
  } else if (current_type == at::ScalarType::BFloat16) {
    new_type = at::ScalarType::Half;
  } else {
    // No need for a host side cast
    return;
  }

  auto earliest_user = findEarliestUser(input);
  if (earliest_user == nullptr) {
    logging::warn("Unused input");
    return;
  }

  // This is an identity op but used just to make sure the implicit cast
  // does not end up promoting to a Double/Long
  auto new_node = graph->create(symbols::poptorch::host_side_cast);

  new_node->insertBefore(earliest_user);
  input->replaceAllUsesWith(new_node->output());
  new_node->addInput(input);

  new_node->output()->setType(tensor_type->withScalarType(new_type));
}

void processInput(torch::jit::Graph *graph, torch::jit::Value *input) {
  switch (input->type()->kind()) {
  case c10::TypeKind::TensorType:
    processInputTensor(graph, input);
    break;
  case c10::TypeKind::TupleType: {
    // Find the [List/Tuple]Unpack node
    if (input->hasUses()) {
      auto unpack = input->uses()[0].user;
      ERROR_ON((unpack->kind() != c10::prim::TupleUnpack) &&
               (unpack->kind() != c10::prim::ListUnpack));
      unsigned int idx = 0;
      for (auto element : unpack->outputs()) {
        logging::LogContext ctx(std::string("element: ") + std::to_string(idx));
        idx++;
        // Recurse for nested tuple support
        processInput(graph, element);
      }
    }
  } break;

  default:
    ERROR("Unsupported input type '"
          << c10::typeKindToString(input->type()->kind()) << "'");
  }
}

} // namespace

void castUnsupportedInputs(torch::jit::Graph *graph) {
  for (torch::jit::Value *input : graph->inputs()) {
    logging::LogContext ctx("castUnsupportedInputs processing" +
                            input->debugName());
    processInput(graph, input);
  }
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
