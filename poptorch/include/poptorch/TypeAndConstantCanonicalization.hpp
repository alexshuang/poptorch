// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_TYPE_AND_CONSTANT_CANONICALIZATION_H
#define INCLUDE_POPTORCH_TYPE_AND_CONSTANT_CANONICALIZATION_H

namespace torch {
namespace jit {
struct Graph;
} // namespace jit
} // namespace torch

namespace poptorch {
namespace type_and_constant_canonicalization {

// Change the graph to add a poptorch::host_side_cast node after every graph
// input whose type is unsupported (Long, Double, BFloat16) to reflect the
// the casting which would happen on the host and the correct types as they
// would be on the graph.
void castUnsupportedInputs(torch::jit::Graph *graph);

} // namespace type_and_constant_canonicalization
} // namespace poptorch

#endif // INCLUDE_POPTORCH_TYPE_AND_CONSTANT_CANONICALIZATION_H