# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# pragma pylint: disable=function-redefined
import poptorch
import torch


# Cases in which casting resolves to the correct type
# correct_cast_start
class Model(torch.nn.Module):
    def forward(self, x, y):
        # y.dtype is ignored, however the type is resolved to be the type of y
        return x.to(y.dtype) + y


native_model = Model()

float16_tensor = torch.tensor([1.0], dtype=torch.float16)
float32_tensor = torch.tensor([1.0], dtype=torch.float32)

assert native_model(float16_tensor, float16_tensor).dtype == torch.float16
assert native_model(float16_tensor, float32_tensor).dtype == torch.float32
assert native_model(float32_tensor, float16_tensor).dtype == torch.float16
assert native_model(float32_tensor, float32_tensor).dtype == torch.float32

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float16_tensor, float16_tensor).dtype == torch.float16

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float16_tensor, float32_tensor).dtype == torch.float32

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float32_tensor, float16_tensor).dtype == torch.float16

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float32_tensor, float32_tensor).dtype == torch.float32

# correct_cast_end


# Cases in which casting resolves to an incorrect type
# incorrect_cast_start
class Model(torch.nn.Module):
    def forward(self, x, y):
        # torch.float32 is ignored and the type is resolved to be the type of y
        return x.to(torch.float32) + y


native_model = Model()

float16_tensor = torch.tensor([1.0], dtype=torch.float16)
float32_tensor = torch.tensor([1.0], dtype=torch.float32)

assert native_model(float16_tensor, float16_tensor).dtype == torch.float32
assert native_model(float32_tensor, float16_tensor).dtype == torch.float32

# This incorrectly results in a float 16 tensor
poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float16_tensor, float16_tensor).dtype == torch.float16

# This incorrectly results in a float 16 tensor
poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float32_tensor, float16_tensor).dtype == torch.float16
# incorrect_cast_end
