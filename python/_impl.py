# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import inspect
import torch
import torch.optim as optim
import poptorch.poptorch_core as poptorch_core
import poptorch.optim

from . import enums
from .logging import logger
from .options import Options


def to_poptorch_optimizer(optimizer):
    if isinstance(optimizer, optim.SGD):
        return enums.OptimizerType.SGD

    if isinstance(optimizer, optim.AdamW):
        if isinstance(optimizer, poptorch.optim.AdamW):
            bias_correction = optimizer.param_groups[0]["biasCorrection"]
            if not bias_correction:
                return enums.OptimizerType.ADAMW_NO_BIAS

        return enums.OptimizerType.ADAMW

    if isinstance(optimizer, optim.RMSprop):
        centered = optimizer.param_groups[0]["centered"]

        return enums.OptimizerType.RMSPROP_CENTERED if centered \
            else enums.OptimizerType.RMSPROP

    if isinstance(optimizer, poptorch.optim.LAMB):  # pylint: disable=no-member
        bias_correction = optimizer.param_groups[0]["biasCorrection"]

        return enums.OptimizerType.LAMB if bias_correction \
            else enums.OptimizerType.LAMB_NO_BIAS
    return None


def convertOptimizerToDict(optimizer):
    optimizer_type = to_poptorch_optimizer(optimizer)

    assert optimizer_type is not None, """Unsupported optimizer type.
         Types supported %s""" % str(list(enums.OptimizerType))

    num_groups = len(optimizer.param_groups)

    accumType = getattr(optimizer, "accumType", False)
    firstOrderTy = getattr(optimizer, "firstOrderMomentumAccumType", False)
    secondOrderTy = getattr(optimizer, "secondOrderMomentumAccumType", False)

    the_dict = {
        "optimizer_type": optimizer_type,
        "num_groups": num_groups,
        "accumType": accumType,
        "firstOrderMomentumAccumType": firstOrderTy,
        "secondOrderMomentumAccumType": secondOrderTy
    }

    for index in range(0, num_groups):
        learning_rate = optimizer.param_groups[index]["lr"]
        weight_decay = optimizer.param_groups[index]["weight_decay"]
        loss_scaling = getattr(optimizer, "loss_scaling", 1.0)

        if isinstance(optimizer, optim.SGD):
            velocity_scaling = getattr(optimizer, "velocity_scaling", 1.0)
            momentum = optimizer.param_groups[0]["momentum"]
            dampening = optimizer.param_groups[0]["dampening"]
            # We will default momentum, weight decay, and dampening, to be
            # constant if they are set to zero.
            the_dict[index] = {
                "lr": (learning_rate, False),
                "momentum": (momentum, momentum == 0.0),
                "weight_decay": (weight_decay, weight_decay == 0.0),
                "dampening": (dampening, dampening == 0.0),
                "loss_scaling": (loss_scaling, loss_scaling == 1.0),
                "velocity_scaling":
                (velocity_scaling, velocity_scaling == 1.0),
            }
        if isinstance(optimizer, (optim.AdamW, poptorch.optim.LAMB)):  # pylint: disable=no-member
            beta1 = optimizer.param_groups[index]["betas"][0]
            beta2 = optimizer.param_groups[index]["betas"][1]
            eps = optimizer.param_groups[index]["eps"]

            if isinstance(optimizer, optim.AdamW):
                assert not optimizer.param_groups[index]["amsgrad"], (
                    "Only non-amsgrad "
                    "AdamW optimizers are supported.")
            the_dict[index] = {
                "lr": (learning_rate, False),
                "beta1": (beta1, False),
                "beta2": (beta2, False),
                "weight_decay": (weight_decay, weight_decay == 0.01),
                "eps": (eps, eps == 1e-08),
                "loss_scaling": (loss_scaling, loss_scaling == 1.0)
            }

        if isinstance(optimizer, optim.RMSprop):
            momentum = optimizer.param_groups[index]["momentum"]
            alpha = optimizer.param_groups[index]["alpha"]
            eps = optimizer.param_groups[index]["eps"]
            the_dict[index] = {
                "lr": (learning_rate, False),
                "momentum": (momentum, momentum == 0.0),
                "alpha": (alpha, False),
                "eps": (eps, eps == 1e-08),
                "weight_decay": (weight_decay, weight_decay == 0.01),
                "loss_scaling": (loss_scaling, loss_scaling == 1.0)
            }

    return the_dict


class ArgsParser:
    class Args:
        def __init__(self):
            self._args = []
            self.first_none = None

        def _forEach(self, data, fn):
            if isinstance(data, (tuple, list)):
                return type(data)(self._forEach(d, fn) for d in data)
            if isinstance(data, dict):
                return {
                    key: self._forEach(value, fn)
                    for key, value in data.items()
                }
            return fn(data)

        def _forEachMatched(self, data, condition, doOnTrue, conditionMatches):
            if isinstance(data, (tuple, list)):
                return type(data)(self._forEachMatched(
                    d, condition, doOnTrue, conditionMatches) for d in data)
            if isinstance(data, dict):
                return {
                    key: self._forEachMatched(value, condition, doOnTrue,
                                              conditionMatches)
                    for key, value in data.items()
                }
            if condition(data):
                conditionMatches.setTrue()
                return doOnTrue(data)
            return data

        def forEachMatchedAtLeastOnce(self, condition, doOnTrue=None):
            class ConditionMatches:
                def __init__(self):
                    self._matches = False

                def __bool__(self):
                    return self._matches

                def setTrue(self):
                    self._matches = True

            matches = ConditionMatches()
            self._args = self._forEachMatched(self._args, condition, doOnTrue,
                                              matches)
            return bool(matches)

        def forEach(self, fn):
            self._args = self._forEach(self._args, fn)

        def asTuple(self):
            return tuple(self._args)

    def __init__(self, model):
        # Combine args and kwargs:
        if isinstance(model, poptorch._OptimizerWrapper):
            sig = inspect.signature(model.model.forward)
        else:
            sig = inspect.signature(model.forward)

        self._has_variadic_arguments = any([
            p.kind in [p.VAR_POSITIONAL, p.VAR_KEYWORD]
            for p in sig.parameters.values()
        ])
        self._varnames = list(sig.parameters.keys())
        self._defaults = [p.default for p in sig.parameters.values()]

    def __call__(self, args, kwargs):
        """Calls the wrapped model with the given tensors. Inputs must be
        tensors or tuples/lists of tensors.
        Will compile for IPU on the first invocation.
        """
        a = ArgsParser.Args()
        assert self._has_variadic_arguments or len(args) + len(kwargs) <= len(
            self._varnames), ("Too many arguments provided: expected %s (%d) "
                              "but got %d") % (self._varnames,
                                               len(self._varnames),
                                               len(args) + len(kwargs))
        first_optional = len(self._varnames) - len(self._defaults)
        none_passed = []

        # Make sure all the arguments provided are allowed.
        for k in kwargs.keys():
            assert k in self._varnames, (
                f"{k} is not a valid parameter."
                f"Allowed values are {self._varnames}")

        for i, name in enumerate(self._varnames):
            if i < len(args):
                self._errorOnListOrDict(args[i], name, [])
                a._args.append(args[i])
                assert name not in kwargs, ("Parameter %s was passed more "
                                            "than once") % name
            elif name in kwargs:
                assert not none_passed, (
                    "Torch doesn't support passing tensors (%s)"
                    " after the following parameters have defaulted to None."
                    " %s") % (name, ", ".join(none_passed))
                self._errorOnListOrDict(kwargs[name], name, [])
                a._args.append(kwargs[name])
            else:
                assert i >= first_optional, ("Mandatory parameter %s "
                                             "missing") % name
                value = self._defaults[i - first_optional]
                if value is None:
                    if a.first_none is None:
                        a.first_none = i
                    none_passed.append("%s (%d)" % (name, i))
                if not none_passed:
                    a._args.append(value)
        if a.first_none is None:
            a.first_none = len(self._varnames)

        return a

    def _errorOnListOrDict(self, data, arg_name, stack_list):
        if isinstance(data, (tuple)):
            for idx, d in enumerate(data):
                stack_list.append(idx)
                self._errorOnListOrDict(d, arg_name, stack_list)
                stack_list.pop()

        if isinstance(data, (dict, list)):
            stack_list = [str(s) for s in stack_list]
            end_msg = arg_name
            if stack_list:
                end_msg += "[" + "][".join(stack_list) + "]"
            end_msg += " = " + str(data)

        if isinstance(data, dict):
            raise TypeError(
                "Dictionaries are not supported as input arguments,"
                " including when nested in tuples.\nReceived dict " + end_msg)

        if isinstance(data, list):
            raise TypeError(
                "Lists are not supported as input arguments,"
                " including when nested in tuples.\nReceived list " + end_msg)


class PoplarExecutor:
    """ This class should not be created directly but is a wrapper around
    the model that was passed into `inferenceModel` or `trainingModel`.
    It only has a few methods which can be used to interface with the IPU.
    """

    def __init__(self,
                 model,
                 options,
                 training,
                 optimizer=None,
                 user_model=None):
        options = options or Options()
        self._model = model
        self._user_model = user_model or self._model
        self._host_weights_version = 0
        if training:
            if options.defaultAnchorMode():
                # In training it makes sense to see only the last result, by default.
                options.anchorMode(enums.AnchorMode.Final)
            if not optimizer:
                optimizer = optim.SGD(self._user_model.parameters(), lr=0.01)

            optimizer = convertOptimizerToDict(optimizer)
        else:
            if options.defaultAnchorMode():
                # In inference it makes sense to see all the results, by default.
                options.anchorMode(enums.AnchorMode.All)
            assert options.Training.gradient_accumulation == 1, (
                "Gradient accumulation"
                " should be left to its default value (1) for inference")
            assert not optimizer, "Optimizer should be None for inference"

        self._executable = None
        self._options = options
        # The args parser needs to be initilialised before the model gets wrapped
        # otherwise we will not be able to retrieve the real arguments list
        self._args_parser = ArgsParser(model)
        self._first_none_arg = None

        self._training = training
        self._optimizer = optimizer or {}
        self._new_optimizer = optimizer or {}
        self._warned_not_contiguous_input = False
        self._dirty_host_weights = False
        self._trace = None

        if self._training:
            parent = self

            class PoptorchModel(type(self._user_model)):
                def copyWeightsToHostIfNeeded(self):
                    """ Return True if the weights on the host were dirty and
                    have been updated.
                    Return False if the weights were already up to date.
                    """
                    if parent._dirty_host_weights:  # pylint: disable=protected-access
                        logger.debug("Implicit copyWeightsToHost()")
                        parent._dirty_host_weights = False  # pylint: disable=protected-access
                        parent.copyWeightsToHost()
                        return True
                    return False

                def __getattribute__(self, name):
                    if name == "_host_weights_version":
                        return parent._host_weights_version
                    if name in ("_parameters", "forward"):
                        self.copyWeightsToHostIfNeeded()
                    return object.__getattribute__(self, name)

                def __getattr__(self, name):
                    attribute = super().__getattr__(name)
                    if isinstance(attribute, torch.nn.parameter.Parameter):
                        self.copyWeightsToHostIfNeeded()
                    return attribute

            class PoptorchParameter(torch.nn.Parameter):
                def __getattribute__(self, name):
                    parent._user_model.copyWeightsToHostIfNeeded()  # pylint: disable=protected-access

                    return object.__getattribute__(self, name)

            for p in self._user_model.parameters():
                p.__class__ = PoptorchParameter

            # __getattr__ and __getattribute__ are attributes, not methods, unfortunately we cannot just
            # replace them in the model object: we have to create a wrapper class
            # and change the object's class.
            PoptorchModel.__name__ = "Poptorch%s" % type(
                self._user_model).__name__
            self._user_model.__class__ = PoptorchModel

    def __getattr__(self, attr):
        return getattr(self._user_model, attr)

    @property
    def model(self):
        return self._user_model

    def _debugGetPopartIR(self):
        return poptorch_core._getPopartIR(self._executable)  # pylint: disable=protected-access

    # Copy weights from the device into the memory of the model given on wrapper creation.
    def copyWeightsToHost(self):
        """ Updates the parameters used in `model` with the weights stored on device.
        (The weights in ``model.parameters()``)
        """
        weights = {
            **dict(self._model.named_parameters()),
            **dict(self._model.named_buffers())
        }
        poptorch_core.copyWeightsToHost_impl(self._executable,
                                             tuple(weights.keys()),
                                             tuple(weights.values()))
        self._host_weights_version += 1

    # Write from host memory to IPU memory. This is done automatically on
    # compilation so should be rarely used.
    def copyWeightsToDevice(self):
        """Copies the weights from ``model.parameters()`` to the IPU device.
        Implicitly called on first call.
        """
        # Don't trigger a copyToHost by accessing `named_parameters`
        saved_dirty_flag = self._dirty_host_weights
        self._dirty_host_weights = False

        weights = {
            **dict(self._model.named_parameters()),
            **dict(self._model.named_buffers())
        }
        poptorch_core.copyWeightsToDevice_impl(self._executable,
                                               tuple(weights.keys()),
                                               tuple(weights.values()))

        # Restore dirtiness flag
        self._dirty_host_weights = saved_dirty_flag

    def setOptimizer(self, optimizer):
        """Sets the optimiser for a training model. Will overwrite the
        previous one. Supported optimisers: ``optim.SGD``, ``optim.AdamW``,
        ``optim.RMSProp``.
        """
        self._new_optimizer = convertOptimizerToDict(optimizer)

    def _parseArgsAndCompile(self, args, kwargs):
        # Convert single tensor to tuple.
        in_tensors = self._args_parser(args, kwargs)

        if in_tensors.forEachMatchedAtLeastOnce(
                condition=lambda t: isinstance(t, torch.Tensor
                                               ) and not t.is_contiguous(),
                doOnTrue=lambda t: t.contiguous()):
            if not self._warned_not_contiguous_input:
                logger.warning("At least one input tensor is not contiguous: "
                               "non-contiguous tensors will be converted.")
                self._warned_not_contiguous_input = True

        if self._executable is None:
            self._first_none_arg = in_tensors.first_none

            # Input will be in form of [BatchSize* BatchPerStep, ...] so we
            # should slice it up so we compile by the batch size alone.
            extra_poplar_batch_dims = self._options.device_iterations * \
                self._options.replication_factor * \
                self._options.Training.gradient_accumulation

            # There are two concepts of batch size. First is the "model" batch size then there is the
            # concept of batching at the popart level. Here we divide by the popart batch size so the
            # trace "sees" the model batch size but when we call execute we pass the full batch and popart
            # will partition it up.
            in_tensors_trace_view = self._args_parser(args, kwargs)

            def narrowTensor(tensor):
                if not isinstance(tensor, torch.Tensor):
                    return tensor
                b_size = 1 if not tensor.size() else tensor.size()[0]
                assert b_size % extra_poplar_batch_dims == 0, (
                    "Invalid batch dimension: In the input %s, the batch "
                    "dimension (%d) must be a multiple of "
                    "Options.deviceIterations(%d) * "
                    "Options.replicationFactor(%d) * "
                    "Options.Training.gradientAccumulation(%d) = %d "
                    "because it is used to calculate the batch size which will "
                    "be executed on the device in any given iteration. For a "
                    "full explanation see the batching semantics page of the "
                    "documentation.") % (
                        tensor.shape, b_size, self._options.device_iterations,
                        self._options.replication_factor,
                        self._options.Training.gradient_accumulation,
                        extra_poplar_batch_dims)
                return tensor.narrow(0, 0, b_size // extra_poplar_batch_dims)

            in_tensors_trace_view.forEach(narrowTensor)

            # Normal bools don't get captured in python.
            hasConvertedAnyHalf = [False]

            def possiblyConvertFromHalf(tensor):
                if isinstance(tensor,
                              torch.Tensor) and tensor.dtype == torch.half:
                    hasConvertedAnyHalf[0] = True
                    return tensor.float()
                return tensor

            in_tensors_trace_view.forEach(possiblyConvertFromHalf)

            # Compile the poplar executable based on the batchsize.
            if self._options.Jit.trace_model:
                logger.info('Compiling the model using tracing')

                convertedLayers = []

                for name, layer in self._model.named_modules():
                    anyIsHalf = False
                    for param in layer.parameters():
                        if param.dtype == torch.half:
                            anyIsHalf = True
                            break

                    if anyIsHalf:
                        layer.float()

                        convertedLayers.append(name)

                # We will trace using the normal trace view.
                self._options._execution_strategy.onStartTracing()
                self._trace = torch.jit.trace(self._model,
                                              in_tensors_trace_view.asTuple())
                self._options._execution_strategy.onEndTracing()

                # Save the inputs of the traced graph printout as it will be
                # different after getting originals back.
                # NB empty if log level is not TRACE.
                if hasConvertedAnyHalf[0]:
                    trace_input_string = poptorch_core.getTraceInputStr(
                        self._trace._c).strip()
                else:
                    trace_input_string = ""

                # Convert any converted params back to half.
                for name, layer in self._trace.named_modules():
                    if name in convertedLayers:
                        layer.half()

                parameters = {
                    **dict(self._trace.named_parameters()),
                    **dict(self._trace.named_buffers())
                }
                if hasConvertedAnyHalf[0]:
                    # Get the originals back.
                    in_tensors_as_half = self._args_parser(args, kwargs)
                    in_tensors_as_half.forEach(narrowTensor)

                    # Compile using the actual halves.
                    self._executable = poptorch_core.compileWithTrace(
                        self._trace._c, tuple(parameters.keys()),
                        tuple(parameters.values()),
                        in_tensors_as_half.asTuple(), trace_input_string,
                        self._options.toDict(), self._training,
                        self._optimizer)
                else:
                    self._executable = poptorch_core.compileWithTrace(
                        self._trace._c, tuple(parameters.keys()),
                        tuple(parameters.values()),
                        in_tensors_trace_view.asTuple(), trace_input_string,
                        self._options.toDict(), self._training,
                        self._optimizer)
            else:
                logger.info('Compiling the model using scripting')
                self._trace = torch.jit.script(self._model)
                graphInputs = list(self._trace.graph.inputs())
                for graphInput, argIn in zip(graphInputs[1:],
                                             in_tensors_trace_view.asTuple()):
                    if isinstance(argIn, torch.Tensor):
                        graphInput.inferTypeFrom(argIn)

                parameters = {
                    **dict(self._trace.named_parameters()),
                    **dict(self._trace.named_buffers())
                }
                self._executable = poptorch_core.compileWithScript(
                    self._trace._c, self._trace.graph,
                    tuple(parameters.keys()), tuple(parameters.values()),
                    in_tensors_trace_view.asTuple(), self._options.toDict(),
                    self._training)

            # Upload the weights to the IPU
            self.copyWeightsToDevice()
        return in_tensors

    def compile(self, *args, **kwargs):
        """Takes the same arguments as the wrapped PyTorch `model.__call__`.

        Trace and compile the wrapped model if no executable has been
        created yet.
        """
        self._parseArgsAndCompile(args, kwargs)

    def __call__(self, *args, **kwargs):
        """
        Takes the same arguments as the wrapped PyTorch `model.__call__`.

        .. note:: The first time the PoplarExecutor wrapper is called, the
            wrapped model will be traced and compiled.

        """
        assert self._options.connectionType != enums.ConnectionType.Never, (
            "Trying to run a model on an offline device "
            " (ConnectionType.Never): use model.compile(inputs) instead of"
            " model(inputs)")
        in_tensors = self._parseArgsAndCompile(args, kwargs)

        # If this is an inference model: check if the same model is not being
        # trained on a different IPU.
        # If it is: make sure the weights are updated.
        if not self._training:
            copyWeightsToHostIfNeeded = getattr(self._user_model,
                                                "copyWeightsToHostIfNeeded",
                                                None)
            if callable(copyWeightsToHostIfNeeded):
                copyWeightsToHostIfNeeded()
                if self._host_weights_version != \
                        self._user_model._host_weights_version:
                    # Weights have now been updated on the Host: copy them to
                    # the second IPU.
                    logger.debug("Implicit copyWeightsToDevice()")
                    self.copyWeightsToDevice()
                    self._host_weights_version = \
                            self._user_model._host_weights_version

        assert in_tensors.first_none == self._first_none_arg, (
            f"Number of arguments mismatch: {self._first_none_arg} "
            f"arguments used to compile the model and "
            f"{in_tensors.first_none} provided this time")
        # Execute the poplar executable with the full size (batch * device interations)
        if self._new_optimizer and self._new_optimizer != self._optimizer:
            self._optimizer = self._new_optimizer
            output = poptorch_core.execute(self._executable,
                                           in_tensors.asTuple(),
                                           self._optimizer)
        else:
            output = poptorch_core.execute(self._executable,
                                           in_tensors.asTuple(), {})

        if self._training:
            self._dirty_host_weights = True

        if len(output) > 1:
            return output
        return output[0]

    def destroy(self):
        """Destroy the model: release the IPUs and the executable.
        """
        if not self._executable:
            return
        if self._training:
            self.copyWeightsToHostIfNeeded()
        del self._executable
        self._executable = None
