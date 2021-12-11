#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
from torch import nn
import pytest
import helpers
import poptorch
from poptorch.enums import Compiler

spatial_dim_map = {
    nn.MaxPool1d: 1,
    nn.MaxPool2d: 2,
    nn.MaxPool3d: 3,
    nn.AvgPool1d: 1,
    nn.AvgPool2d: 2,
    nn.AvgPool3d: 3,
    nn.AdaptiveAvgPool1d: 1,
    nn.AdaptiveAvgPool2d: 2,
    nn.AdaptiveAvgPool3d: 3,
}


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize(
    "params",
    [
        # op, kernel_size, stride, padding, ceil_mode, count_include_pad
        (3, 2, 0, False, True),
        (3, 2, 0, True, True),
        ((3, 2, 2), (2, 1, 2), 0, False, True),
        (3, 2, 1, False, True),
        (3, 2, 0, False, False),
    ])
@pytest.mark.parametrize("op", [
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
])
def test_pool(op, params):
    torch.manual_seed(42)

    spatial_dims = spatial_dim_map[op]

    kernel_size, stride, padding, ceil_mode, count_include_pad = params
    extra_args = {}
    if op in (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d):
        extra_args["count_include_pad"] = count_include_pad
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[:spatial_dims]
        stride = stride[:spatial_dims]

    shape = [1, 2]
    shape.extend([10 for _ in range(spatial_dims)])
    t = torch.randn(shape)

    pool = op(kernel_size, stride, padding, ceil_mode=ceil_mode, **extra_args)

    # Run pytorch native on CPU.
    torch_out = pool(t)

    # Run on IPU.
    with poptorch.IPUScope([t], compile_using=Compiler.MLIR) as ipu:
        ipu.outputs([pool(t)])

    # pylint: disable=no-member
    helpers.assert_allclose(actual=ipu(t), expected=torch_out)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize(
    "op", [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d])
def test_adaptive_avg_pool(op):
    torch.manual_seed(42)
    # AdaptiveAvgPool1d: [1, 2, 4]       -> [1, 2, 2]
    # AdaptiveAvgPool2d: [1, 2, 4, 6]    -> [1, 2, 2, 3]
    # AdaptiveAvgPool3d: [1, 2, 4, 6, 8] -> [1, 2, 2, 3, 4]
    # TODO(T31335): Match PyTorch's implementation so that we can test cases where
    #               input dims are not divisible by corresponding output dims
    spatial_dims = spatial_dim_map[op]

    shape = [1, 2]
    shape.extend([2 * i + 4 for i in range(spatial_dims)])

    t = torch.randn(shape)
    output_size = [i + 2 for i in range(spatial_dims)]

    pool = op(output_size)
    # Run pytorch native on CPU.
    torch_out = pool(t)

    # Run on IPU.
    with poptorch.IPUScope([t], compile_using=Compiler.MLIR) as ipu:
        ipu.outputs([pool(t)])

    # pylint: disable=no-member
    helpers.assert_allclose(actual=ipu(t), expected=torch_out)