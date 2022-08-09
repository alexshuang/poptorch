#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import inspect
import torch
import torchvision.models as models
import pytest
import helpers


def simple_add(capfd):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    capfd.readouterr()  # Clear the log

    def fn(input, check_log=False):
        x = input + 5
        if check_log:
            # Check the add was lowered and executed.
            log = helpers.LogChecker(capfd)
            log.assert_not_contains("CPU -> IPU")
            log.assert_not_contains("IPU -> CPU")
            log.assert_contains("Graph lowered to popit")
        return x * 3

    input = torch.ones([10])
    cpu = fn(input)
    log = helpers.LogChecker(capfd)
    log.assert_isEmpty()
    input = input.to("xla")
    log = helpers.LogChecker(capfd)
    log.assert_contains("CPU -> IPU")
    ipu = fn(input, check_log=True)
    # Check the multiplication was also lowered and executed.
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_not_contains("IPU -> CPU")
    log.assert_contains("Graph lowered to popit")
    print(f"Result cpu: {cpu} ipu: {ipu}")
    # Check the print triggered a copy to host
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_contains("IPU -> CPU")
    helpers.assert_allclose(expected=cpu, actual=ipu.cpu())
    # Check .cpu() triggered a copy to host
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_contains("IPU -> CPU")


@pytest.mark.mlirSupportRequired
@pytest.mark.ipuHardwareRequired
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_simple_add_hw(capfd):
    simple_add(capfd)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("mode", ["default", "show_all", "hide_all"])
@pytest.mark.mlirSupportRequired
@pytest.mark.extendedTestingOnly
def test_source_location(capfd, mode):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    layer = torch.nn.Linear(1, 2).to('xla')
    expected_filename = inspect.stack()[0].filename
    # +3 -> We expect to see f()'s return line in the log
    expected_line = inspect.stack()[0].lineno + 3

    def f(x):
        return layer(x)

    if mode == "show_all":
        # Clear the list: show everything
        poptorch.eager.eager_options.source_location_excludes = []
    elif mode == "hide_all":
        # All paths have a '/' in them so we essentially exclude everything.
        poptorch.eager.eager_options.source_location_excludes += ['/']

    input = torch.Tensor([[1.], [-1.]]).to('xla')
    f(input)

    log = helpers.LogChecker(capfd)
    if mode == "show_all":
        # If we clear the list of exclusions we will point at Torch's internals
        log.assert_matches(
            "poptorch.transpose.*site-packages/torch/nn/functional.py")
        log.assert_no_matches(
            f"poptorch.transpose.*{expected_filename}:{expected_line}")
    elif mode == "hide_all":
        log.assert_matches(r"poptorch.transpose.*\[unknown\]")  # no filename
        log.assert_no_matches(
            "poptorch.transpose.*site-packages/torch/nn/functional.py")
        log.assert_no_matches(
            f"poptorch.transpose.*{expected_filename}:{expected_line}")
    else:
        # By default: we point at the user code
        log.assert_no_matches(
            "poptorch.transpose.*site-packages/torch/nn/functional.py")
        log.assert_matches(
            f"poptorch.transpose.*{expected_filename}:{expected_line}")


@pytest.mark.mlirSupportRequired
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_simple_add(capfd):
    pytest.skip("PopIT doesn't currently support IPUModel")
    simple_add(capfd)


@pytest.mark.ipuHardwareRequired
@pytest.mark.mlirSupportRequired
@pytest.mark.extendedTestingOnly
def test_squeezenet():
    pytest.skip("TODO(T67125): Tensor-likes are not close")

    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    input = torch.randn([1, 3, 224, 224])

    model = models.squeezenet1_1(pretrained=False)
    model.eval()

    cpu = model(input)

    model.to("xla")
    input = input.to("xla")

    ipu = model(input)

    print(f"Result cpu: {cpu} ipu: {ipu}")
    helpers.assert_allclose(expected=cpu, actual=ipu.cpu())


@pytest.mark.ipuHardwareRequired
@pytest.mark.mlirSupportRequired
@pytest.mark.extendedTestingOnly
def test_resnet18():
    pytest.skip("TODO(T64252): 'std::exception': Trying to allocate a tensor "
                "to an allocated region")

    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    input = torch.randn([1, 3, 224, 224])

    model = models.resnet18(pretrained=False)
    model.eval()

    cpu = model(input)

    model.to("xla")
    input = input.to("xla")

    ipu = model(input)

    print(f"Result cpu: {cpu} ipu: {ipu}")
    helpers.assert_allclose(expected=cpu, actual=ipu.cpu())
