#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import pytest
import helpers
import poptorch
from poptorch.experimental import IPUContext


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("num_dims", [1, 2, 4])
def test_mse_loss_forward(reduction, num_dims):
    torch.manual_seed(42)

    if num_dims == 1:
        input_dims = [5]
    if num_dims == 2:
        input_dims = [3, 4]
    if num_dims == 4:
        input_dims = [2, 4, 5, 3]

    input1 = torch.rand(input_dims)
    input2 = torch.rand(input_dims)

    def mse_loss(t1, t2):
        return F.mse_loss(
            t1,
            t2,
            reduction=reduction,
        )

    cpu_result = mse_loss(input1, input2)
    ipu_result = IPUContext(mse_loss)(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("num_dims", [1, 2, 4])
def test_mse_loss_backward(reduction, num_dims):
    torch.manual_seed(42)

    if num_dims == 1:
        input_dims = [5]
    if num_dims == 2:
        input_dims = [3, 4]
    if num_dims == 4:
        input_dims = [2, 4, 5, 3]

    input1 = torch.rand(input_dims)
    input2 = torch.rand(input_dims)

    def mse_loss_backward(t1, t2):
        t1.requires_grad = True
        t1.retain_grad()
        loss = F.mse_loss(
            t1,
            t2,
            reduction=reduction,
        )
        if reduction == "none":
            loss = loss.sum()
        assert t1.requires_grad
        loss.backward()
        return t1.grad

    ipu_result = IPUContext(mse_loss_backward)(input1, input2)
    cpu_result = mse_loss_backward(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=False)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("num_dims", [1, 2, 4])
def test_nll_loss_forward(reduction, num_dims):
    torch.manual_seed(42)

    if num_dims == 1:
        # (C) and ()
        input1 = torch.randn([4])
        input2 = torch.tensor(2)
    if num_dims == 2:
        # (N, C) and (N)
        input1 = torch.randn([4, 10])
        input2 = torch.tensor([1, 2, 3, 4])
    if num_dims == 4:
        return  # not yet implemented TODO T62380
        # (N, C, d1, d2) and (N, d1, d2)
        # input1 = torch.randn([4, 10, 2, 2])
        # input2 = torch.tensor([1, 2, 3, 4]).unsqueeze(1).unsqueeze(1)
        # input2 = input2.expand(4, 2, 2)

    def nll_loss(t1, t2):
        return F.nll_loss(
            t1,
            t2,
            reduction=reduction,
        )

    cpu_result = nll_loss(input1, input2)
    ipu_result = IPUContext(nll_loss)(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.parametrize("reduction", ["mean"])
@pytest.mark.parametrize("num_dims", [1, 2, 4])
def test_nll_loss_backward(reduction, num_dims):
    torch.manual_seed(42)

    if num_dims == 1:
        # (C) and ()
        input1 = torch.randn([4])
        input2 = torch.tensor(2)
    if num_dims == 2:
        # (N, C) and (N)
        input1 = torch.randn([4, 10])
        input2 = torch.tensor([1, 2, 3, 4])
    if num_dims == 4:
        # (N, C, d1, d2) and (N, d1, d2)
        return  # not yet implemented TODO T62380
        # input1 = torch.nn.parameter.Parameter(torch.randn([4, 10, 2, 2]))
        # input2 = torch.tensor([1, 2, 3, 4]).unsqueeze(1).unsqueeze(1)

    def nll_loss_backward(t1, t2):
        t1.requires_grad = True
        t1.retain_grad()

        loss = F.nll_loss(t1, t2, reduction=reduction)
        if reduction == "none":
            loss = loss.sum()
        loss.backward()
        return t1.grad

    ipu_result = IPUContext(nll_loss_backward)(input1, input2)

    cpu_result = nll_loss_backward(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("ignore_index", [2, -100])
def test_nll_loss_forward_ignore(reduction, ignore_index):
    torch.manual_seed(42)
    input1 = torch.randn([4, 10])
    input2 = torch.Tensor([1, 2, 3, 4]).long()

    def nll_loss(t1, t2):
        return F.nll_loss(t1,
                          t2,
                          reduction=reduction,
                          ignore_index=ignore_index)

    cpu_result = nll_loss(input1, input2)
    ipu_result = IPUContext(nll_loss)(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_nll_loss_forward_ignore_all(reduction):
    torch.manual_seed(42)
    input1 = torch.randn([4, 10])
    input2 = torch.Tensor([1, 1, 1, 1]).long()

    def nll_loss(t1, t2):
        return F.nll_loss(t1, t2, reduction=reduction, ignore_index=1)

    cpu_result = nll_loss(input1, input2)
    ipu_result = IPUContext(nll_loss)(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("ignore_index", [2, -100])
def test_nll_loss_backward_ignore(reduction, ignore_index):
    torch.manual_seed(42)
    input1 = torch.randn([4, 10])
    input2 = torch.tensor([1, 2, 3, 4])

    def nll_loss_backward(t1, t2):
        t1.requires_grad = True
        t1.retain_grad()
        loss = F.nll_loss(t1,
                          t2,
                          reduction=reduction,
                          ignore_index=ignore_index)
        if reduction == "none":
            loss = loss.sum()
        loss.backward()
        return t1.grad

    ipu_result = IPUContext(nll_loss_backward)(input1, input2)

    cpu_result = nll_loss_backward(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_nll_loss_backward_ignore_all(reduction):
    torch.manual_seed(42)
    input1 = torch.randn([4, 10])
    input2 = torch.tensor([1, 1, 1, 1])

    def nll_loss_backward(t1, t2):
        t1.requires_grad = True
        t1.retain_grad()
        loss = F.nll_loss(t1, t2, reduction=reduction, ignore_index=1)
        if reduction == "none":
            loss = loss.sum()
        loss.backward()
        return t1.grad

    ipu_result = IPUContext(nll_loss_backward)(input1, input2)

    cpu_result = nll_loss_backward(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("num_dims", [1, 2])
@pytest.mark.parametrize("weight", [None, "batch", "bigger"])
def test_binary_cross_entropy_forward(reduction, num_dims, weight):
    torch.manual_seed(42)

    if weight is None:
        weight_tensor = None

    if num_dims == 1:
        input1 = torch.clip(torch.randn([4]), min=0.001, max=0.999)
        input2 = torch.Tensor([1, 0, 0, 1])
        if weight == "batch":
            weight_tensor = torch.rand([1])
        if weight == "bigger":
            weight_tensor = torch.rand([4])

    if num_dims == 2:
        input1 = torch.clip(torch.randn([3, 2]), min=0.001, max=0.999)
        input2 = torch.Tensor([[1, 0], [0, 1], [1, 1]])
        if weight == "batch":
            weight_tensor = torch.rand([3, 1])
        if weight == "bigger":
            weight_tensor = torch.rand([3, 2])

    def bce(t1, t2, weight_tensor):
        return F.binary_cross_entropy(t1,
                                      t2,
                                      reduction=reduction,
                                      weight=weight_tensor)

    ipu_result = IPUContext(bce)(input1, input2, weight_tensor)
    cpu_result = bce(input1, input2, weight_tensor)

    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.parametrize("num_dims", [1, 2])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("weight", [None, "batch", "bigger"])
def test_binary_cross_entropy_backward(reduction, num_dims, weight):
    torch.manual_seed(42)

    if weight is None:
        weight_tensor = None

    if num_dims == 1:
        input1 = torch.clip(torch.randn([4]), min=0.001, max=0.999)
        input2 = torch.Tensor([1, 0, 0, 1])
        if weight == "batch":
            weight_tensor = torch.rand([1])
        if weight == "bigger":
            weight_tensor = torch.rand([4])

    if num_dims == 2:
        input1 = torch.clip(torch.randn([3, 2]), min=0.001, max=0.999)
        input2 = torch.Tensor([[1, 0], [0, 1], [1, 1]])
        if weight == "batch":
            weight_tensor = torch.rand([3, 1])
        if weight == "bigger":
            weight_tensor = torch.rand([3, 2])

    def bce_backward(t1, t2, weight_tensor):
        t1.requires_grad = True
        t1.retain_grad()

        loss = F.binary_cross_entropy(t1,
                                      t2,
                                      reduction=reduction,
                                      weight=weight_tensor)
        if reduction == "none":
            loss = loss.sum()
        loss.backward()
        return t1.grad

    ipu_result = IPUContext(bce_backward)(input1, input2, weight_tensor)
    cpu_result = bce_backward(input1, input2, weight_tensor)

    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.parametrize("num_dims", [1, 2])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_binary_cross_entropy_with_logits_forward(num_dims, reduction):
    torch.manual_seed(42)

    if num_dims == 1:
        input1 = torch.randn([4])
        input2 = torch.Tensor([1, 0, 0, 1])

    if num_dims == 2:
        input1 = torch.randn([3, 2])
        input2 = torch.Tensor([[1, 0], [0, 1], [1, 1]])

    def bce_logit(t1, t2):
        return F.binary_cross_entropy_with_logits(t1, t2, reduction=reduction)

    cpu_result = IPUContext(bce_logit)(input1, input2)
    ipu_result = (bce_logit)(input1, input2)

    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.parametrize("num_dims", [1, 2])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_binary_cross_entropy_with_logits_backward(reduction, num_dims):
    torch.manual_seed(42)
    if num_dims == 1:
        input1 = torch.randn([4])
        input2 = torch.Tensor([1, 0, 0, 1])

    if num_dims == 2:
        input1 = (torch.randn([3, 2]))
        input2 = torch.Tensor([[1, 0], [0, 1], [1, 1]])

    def bce_logit_backward(t1, t2):
        t1.requires_grad = True
        t1.retain_grad()
        loss = F.binary_cross_entropy_with_logits(t1, t2, reduction=reduction)
        if reduction == "none":
            loss = loss.sum()
        loss.backward()
        return t1.grad

    ipu_result = IPUContext(bce_logit_backward)(input1, input2)
    cpu_result = bce_logit_backward(input1, input2)

    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("num_dims", [1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
def test_smooth_l1_loss_forward(reduction, num_dims, beta):
    torch.manual_seed(42)

    input_shape = tuple(range(3, 3 + num_dims))
    input1 = torch.rand(input_shape)
    input2 = torch.rand(input_shape)

    def smooth_l1_loss(t1, t2):
        return F.smooth_l1_loss(t1, t2, reduction=reduction, beta=beta)

    smooth_l1_loss(input1, input2)

    err_msg = "smooth_l1_loss cannot currently be lowered to Poplar"
    with pytest.raises(poptorch.poptorch_core.Error, match=err_msg):
        IPUContext(smooth_l1_loss)(input1, input2)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("num_dims", [1, 2])
def test_l1_loss_forward(reduction, num_dims):
    torch.manual_seed(42)

    input_shape = tuple(range(3, 3 + num_dims))
    input1 = torch.rand(input_shape)
    input2 = torch.rand(input_shape)

    def l1_loss(t1, t2):
        return F.l1_loss(t1, t2, reduction=reduction)

    l1_loss(input1, input2)

    err_msg = r"l1_loss(\.out)? cannot currently be lowered to Poplar"
    with pytest.raises(poptorch.poptorch_core.Error, match=err_msg):
        IPUContext(l1_loss)(input1, input2)
