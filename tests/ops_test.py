#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import poptorch


def test_print_tensor():
    class Model(torch.nn.Module):
        def forward(self, x):
            return poptorch.ipu_print_tensor(x)

    m = poptorch.inferenceModel(Model())
    m(torch.randn(5))


def test_print_tensor_with_title():
    class Model(torch.nn.Module):
        def forward(self, x):
            return poptorch.ipu_print_tensor(x, "my_tensor")

    m = poptorch.inferenceModel(Model())
    m(torch.randn(5))


def test_nop():
    class Model(torch.nn.Module):
        def forward(self, x):
            return poptorch.nop(x) * 2

    m = poptorch.inferenceModel(Model())
    m(torch.randn(5))
