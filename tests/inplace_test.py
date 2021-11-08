#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch

import poptorch
import helpers


def test_inplace_add():
    class Model(torch.nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] += 1
            elif isinstance(x, (dict)):
                x['input'] += 1
            else:
                x += 1

    poptorch_model = poptorch.inferenceModel(Model())
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 2.0
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 3.0
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == 3.0

    list_in = (torch.Tensor([1.0]), )
    assert poptorch_model(list_in) is None
    assert list_in[0] == 2.0
    assert poptorch_model(list_in) is None
    assert list_in[0] == 3.0


def test_inplace_add_multi_elements():
    class Model(torch.nn.Module):
        def forward(self, _x, y):
            y += 1

    poptorch_model = poptorch.inferenceModel(Model())
    nested_tuple_in = ((torch.Tensor([1.0]), torch.Tensor([1.0])),
                       (torch.Tensor([1.0])))
    tensor_in = torch.Tensor([1.0])

    assert poptorch_model(nested_tuple_in, tensor_in) is None
    assert tensor_in == 2.0


def test_inplace_sub():
    class Model(torch.nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] -= 1
            elif isinstance(x, (dict)):
                x['input'] -= 1
            else:
                x -= 1

    poptorch_model = poptorch.inferenceModel(Model())
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 0.0
    assert poptorch_model(tensor_in) is None
    assert tensor_in == -1.0
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == -1.0

    list_in = (torch.Tensor([1.0]), )
    assert poptorch_model(list_in) is None
    assert list_in[0] == 0.0
    assert poptorch_model(list_in) is None
    assert list_in[0] == -1.0


def test_inplace_div():
    class Model(torch.nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] /= 2
            elif isinstance(x, (dict)):
                x['input'] /= 2
            else:
                x /= 2

    poptorch_model = poptorch.inferenceModel(Model())
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 0.5
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 0.25
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == 0.25

    list_in = (torch.Tensor([1.0]), )
    assert poptorch_model(list_in) is None
    assert list_in[0] == 0.5
    assert poptorch_model(list_in) is None
    assert list_in[0] == 0.25


def test_inplace_mul():
    class Model(torch.nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] *= 2
            elif isinstance(x, (dict)):
                x['input'] *= 2
            else:
                x *= 2

    poptorch_model = poptorch.inferenceModel(Model())
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 2.0
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 4.0
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == 4.0

    list_in = (torch.Tensor([1.0]), )
    assert poptorch_model(list_in) is None
    assert list_in[0] == 2.0
    assert poptorch_model(list_in) is None
    assert list_in[0] == 4.0


def test_inplace_masked_fill():
    class Model(torch.nn.Module):
        def forward(self, x):
            x.masked_fill_(x > 0.5, 1.0)

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.tensor([[0, 0.7], [0.2, 3.5]])
    poptorch_model(x)

    assert x[0][0] == 0
    assert x[0][1] == 1.0
    assert x[1][0] == 0.2
    assert x[1][1] == 1.0


def test_inplace_zero():
    class Model(torch.nn.Module):
        def forward(self, x):
            # (Simply setting it to zero gets pruned by PopART)
            a = torch.sum(x)
            x.zero_()
            x += a

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.tensor([[0, 0.5], [0.25, 2.0]])
    poptorch_model(x)

    assert x[0][0] == 2.75
    assert x[0][1] == 2.75
    assert x[1][0] == 2.75
    assert x[1][1] == 2.75


def test_inplace_fill():
    class Model(torch.nn.Module):
        def forward(self, x):
            a = torch.sum(x)
            x.fill_(1.0)
            x += a

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.tensor([[0, 0.5], [0.25, 2.0]])
    poptorch_model(x)

    assert x[0][0] == 3.75
    assert x[0][1] == 3.75
    assert x[1][0] == 3.75
    assert x[1][1] == 3.75


def test_inplace_non_input():
    class Model(torch.nn.Module):
        def forward(self, x):
            a = x + 1
            a += 1
            return a

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.tensor([[0, 0.5], [0.25, 2.0]])

    y = poptorch_model(x)

    assert x[0][0] == 0
    assert x[0][1] == 0.5
    assert x[1][0] == 0.25
    assert x[1][1] == 2.0

    assert y[0][0] == 2
    assert y[0][1] == 2.5
    assert y[1][0] == 2.25
    assert y[1][1] == 4.0


def test_double_underscore():
    # This tests aten::__and__ is not treated as inplace

    class Model(torch.nn.Module):
        def forward(self, x, l):
            return x[x[0][0].int() & l.item()]

    model = Model()
    poptorch_model = poptorch.inferenceModel(model, options=poptorch.Options())
    inp, l = torch.rand(10, 10), torch.LongTensor([2])

    out = model(inp, l)
    popout = poptorch_model(inp, l)

    helpers.assert_allclose(actual=popout, expected=out)


def test_half_buffer_inplace():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('buff', torch.ones(5, dtype=torch.float16))

        def forward(self, x):
            # pylint: disable=no-member
            out = x + self.buff
            self.buff += 1
            return out

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float16)
    out = poptorch_model(x)

    helpers.assert_allclose(actual=out,
                            expected=torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5],
                                                  dtype=torch.float16))
    poptorch_model.copyWeightsToHost()
    helpers.assert_allclose(actual=poptorch_model.buff,
                            expected=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0],
                                                  dtype=torch.float16))


def test_float_to_half_buffer_inplace_with_training():
    torch.manual_seed(42)

    # pylint: disable=attribute-defined-outside-init
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            # need at least one parameter for a training model
            self.param = torch.nn.Parameter(torch.ones(5, 5))

            self.register_buffer("buff", torch.ones(5))
            self.loss = torch.nn.MSELoss()

        def forward(self, x):
            # pylint: disable=no-member
            out = self.buff + self.param
            self.buff += 1
            return out, self.loss(out, x)

    model = Model().train().half()
    poptorch_model = poptorch.trainingModel(model)

    x = torch.rand(5, 5).half()
    native_out, native_loss = model(x)

    # Reset buff
    model.buff = torch.ones(5, 5)

    poptorch_out, poptorch_loss = poptorch_model(x)

    helpers.assert_allclose(actual=native_out, expected=poptorch_out)
    helpers.assert_allclose(actual=native_loss, expected=poptorch_loss)
