# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
from poptorch.experimental import IPUContext
import helpers


def harness(op, **kwargs):
    torch.manual_seed(42)
    t = torch.randn(10)

    ipu_result = IPUContext(op)(t, **kwargs)
    cpu_result = op(t, **kwargs)

    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("min_param,max_param", [(None, 0.5), (-0.5, None),
                                                 (-0.5, 0.5)])
def test_clamp(min_param, max_param):
    if min_param is None or max_param is None:
        pytest.skip("TODO(T62028): There is currently a bug with the name "
                    "mangling in poplar which causes this to break. Poplar "
                    "task T62028")
    kwargs = {}
    if min_param is not None:
        kwargs["min"] = min_param
    if max_param is not None:
        kwargs["max"] = max_param
    harness(torch.clamp, **kwargs)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("min_param,max_param", [(None, 0.5), (-0.5, None),
                                                 (-0.5, 0.5)])
def test_hardtanh(min_param, max_param):
    kwargs = {}
    if min_param is not None:
        kwargs["min_val"] = min_param
    if max_param is not None:
        kwargs["max_val"] = max_param

    harness(torch.nn.Hardtanh(**kwargs))


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("clamp_min,clamp_max", [(True, True), (True, False),
                                                 (False, True)])
def test_clampTensor(clamp_min, clamp_max):
    kwargs = {}
    if clamp_min:
        kwargs["min"] = torch.linspace(-0.5, 0.0, 10)
    if clamp_max:
        kwargs["max"] = torch.linspace(0.0, 0.5, 10)

    harness(torch.clamp, **kwargs)
