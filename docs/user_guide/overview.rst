========
Features
========

.. contents::
  :local:

Options
=======

The compilation and execution on the IPU can be controlled using :class:`poptorch.Options`:

See :ref:`efficient_data_batching`  for a full
explanation of how ``device_iterations`` greater than 1, ``gradient_accumulation``, and
``replication_factor`` interact with the output and input sizes.

.. autoclass:: poptorch.Options
   :members:

You can choose to use the IPU model or the real IPU hardware
via :py:class:`poptorch.Options.useIpuModel`.


.. autoclass:: poptorch.options._DistributedOptions
   :members:

.. autoclass:: poptorch.options._JitOptions
   :members:

.. autoclass:: poptorch.options._TrainingOptions
   :members:

.. autoclass:: poptorch.options._PopartOptions
   :members:

.. autoclass:: poptorch.options._TensorLocationOptions
   :members:

.. autoclass:: poptorch.TensorLocationSettings
   :members:

Model wrapping functions
========================


The basis of PopTorch integration comes from these two model wrapping functions.

poptorch.trainingModel
----------------------

.. autofunction:: poptorch.trainingModel

.. literalinclude:: trainingModel.py
    :language: python
    :caption: An example of the use of :py:func:`poptorch.trainingModel`
    :linenos:
    :emphasize-lines: 22
    :start-after: training_model_start

poptorch.inferenceModel
-----------------------

.. autofunction:: poptorch.inferenceModel

.. literalinclude:: inferenceModel.py
    :language: python
    :caption: An example of the use of :py:func:`poptorch.inferenceModel`
    :linenos:
    :start-after: inference_model_start
    :emphasize-lines: 14


poptorch.PoplarExecutor
-----------------------

.. autoclass:: poptorch.PoplarExecutor
   :special-members: __call__
   :members:

.. note:: The ``PoplarExecutor`` will implicitly keep in sync the parameters
  of the source PyTorch model and the PopTorch model(s).
  However, weights need to be explicitly copied if the
  model is trained on the CPU and inference is run on the IPU.

  .. code-block:: python

    model = Model()
    poptorch_train = poptorch.trainingModel(model)
    poptorch_inf = poptorch.inferenceModel(model)

    train(poptorch_train)
    torch.save(model.state_dict(), "model.save") # OK
    validate(poptorch_inf) # OK
    validate(model) # OK

    train(model)
    # Explicit copy needed
    poptorch_inf.copyWeightsToDevice()
    validate(poptorch_inf)

.. _parallel_execution:

Parallel execution
==================

This section demonstrates multi-IPU strategies for parallel execution in
PopTorch.
We recommended that you start such parallel programming from
PopTorch code that is working properly on a single IPU.

There are four kinds of execution strategies in total to run a model on a
multi-IPU device:
:py:class:`poptorch.ShardedExecution`,
:py:class:`poptorch.PipelinedExecution`,
:py:class:`poptorch.SerialPhasedExecution`.
and :py:class:`poptorch.ParallelPhasedExecution`.
These execution strategies are set through
:py:func:`poptorch.Options.setExecutionStrategy`.
The default execution strategy is :py:class:`poptorch.PipelinedExecution`.
In the following,
we first introduce the general APIs that will be applied to all four
parallel execution strategies.
Finally, we explain the four strategies with examples.

Annotation tools
----------------

poptorch.Block and poptorch.BeginBlock
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`poptorch.BeginBlock` and :py:class:`poptorch.Block` are
indispensable wrapper classes to define model
parallelism in a multi-IPU device.
You can use :py:class:`poptorch.Block` to define a scope in the context of the
model.

.. autoclass:: poptorch.Block
   :special-members: __init__, useAutoId

:py:class:`poptorch.BeginBlock` is an annotation defined outside the
model, and applied to current and onward layers.

.. autoclass:: poptorch.BeginBlock
   :special-members: __init__

:py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block`
alone is enough to enable parallel execution in the simplest case.
By default, the layers before the first :py:class:`poptorch.BeginBlock`
will be placed on IPU 0.
The complete code examples for :py:class:`poptorch.BeginBlock` and
:py:class:`poptorch.Block` are shown below.
All layers before ``model.bert.encoder.layer[0]``
will be on IPU 0 and all layers from ``model.bert.encoder.layer[0]``
onwards (inclusive) will be on IPU 1.

.. literalinclude:: pipeline_simple.py
    :language: python
    :linenos:
    :start-after: annotations_start
    :end-before: annotations_end
    :emphasize-lines: 15-16, 19-20, 23-24
    :caption: Annotations can be attached to layers in existing models.


.. literalinclude:: pipeline_simple.py
    :language: python
    :linenos:
    :start-after: annotations_inline_start
    :end-before: annotations_inline_end
    :emphasize-lines: 16, 19, 22, 26
    :caption: PopTorch also supports annotating the model directly.
		          Both forms can be used interchangeably.

Both :py:class:`poptorch.BeginBlock` and :py:class:`poptorch.Block`
need to follow a set of rules:

* All the layers must be declared inside a :py:class:`poptorch.Block` scope.
  It is to avoid missing annotation. :py:class:`poptorch.BeginBlock`
  doesn't have the same constraint because all the layers called after will
  automatically be added to the last :py:class:`poptorch.BeginBlock`.
* Please note that PopTorch needs to reserve IPUs in powers of 2 or
  multiples of 64. You are advised to configure your model accordingly
  to take full advantage of the IPUs available. However, if you need to run
  with a different number of IPUs, you can use
  ``poptorch.Options().autoRoundNumIPUs(True)`` to allow
  PopTorch to reserve more IPUs than the model specifies.
* Unused or dead layers should NOT be included in any
  :py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block`.
* If layer A happens before layer B inside the model and each layer has
  a :py:class:`poptorch.BeginBlock` associated with it,
  you need to write :py:class:`poptorch.BeginBlock` for layer A before
  :py:class:`poptorch.BeginBlock` for layer B.

Failing to obey above rules will result in compilation errors.


poptorch.Stage and poptorch.AutoStage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conceptually :py:class:`poptorch.BeginBlock` or
:py:class:`poptorch.Block` collects the
layers of a model into a :py:class:`poptorch.Stage`,
multiple stages can be combined into a :py:class:`poptorch.Phase`,
and multiple phases form a parallel execution strategy.

poptorch.Stage
""""""""""""""

:py:class:`poptorch.Stage` defines some layers of model to run on one IPU.
It can be made of one or more blocks created using
:py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block`
and identified by their `user_id`.
Consecutive layers in a model can be defined either in the same
:py:class:`poptorch.Stage` or consecutive stages.
Whether stages run in parallel or sequentially depends on specific
parallel execution strategies.

.. autoclass:: poptorch.Stage
   :special-members: __init__

Internally, each operation in a model is assigned a `stage_id`
through :py:class:`poptorch.Stage`.

poptorch.AutoStage
""""""""""""""""""

You can use :py:class:`poptorch.AutoStage` if you don't want to
specify :py:class:`poptorch.Stage` by hand.
It will assign one :py:class:`poptorch.Stage`
per :py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block`.

.. autoclass:: poptorch.AutoStage

By default ``poptorch.AutoStage.SameAsIpu`` is in use, which means the
`stage_id` of :py:class:`poptorch.Stage` will be set to the `ipu_id`
specified for the :py:class:`poptorch.BeginBlock` or
:py:class:`poptorch.Block`.
Please note that `stage_id` must be ascending in
:py:class:`poptorch.PipelinedExecution`.
Let's use the code example above.
If your blocks "0", "1", and "2" are assigned to IPU 0, 1, and 0.
Then the :py:class:`poptorch.Block`
"2" will be assigned `stage_id` 0. This will make
the compiler fail to
schedule the last two stages "1" and "2" due to a conflict:

* The model implies "1" should run earlier than "2".
* their `stage_id` values suggest "2" should run earier than "1".

When ``poptorch.AutoStage.AutoIncrement`` is in use, each new
:py:class:`poptorch.BeginBlock` or
:py:class:`poptorch.Block` will be assigned an automatically incremented
`stage_id`.
In the previous example the last stage would be assigned `stage_id` 2 and
the compilation would succeed.

poptorch.Phase
^^^^^^^^^^^^^^

:py:class:`poptorch.Phase` defines a processing unit of phased execution.
It may contain one or more :py:class:`poptorch.Stage`.
:py:class:`poptorch.Phase` is only used in
:py:class:`poptorch.SerialPhasedExecution` and
:py:class:`poptorch.ParallelPhasedExecution`.
It is not used in
:py:class:`poptorch.ShardedExecution` and
:py:class:`poptorch.PipelinedExecution`.

.. autoclass:: poptorch.Phase
   :special-members: __init__

In the last two lines above,
"A" and "B" will run in parallel on IPU 0 and 1 simultaneously since
they are placed in two stages. They will run sequentially in one IPU
if they are placed in one stage only.


Advanced annotation with strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use Python strings to represent the `user_id` and `ipu_id` for a
:py:class:`poptorch.Block` or
:py:class:`poptorch.BeginBlock`.
Since strings are evaluated at runtime,
they allow for a dynamic number of stages and phases.
Here is an example below to use formatted strings(f-strings) in
:py:class:`poptorch.ParallelPhasedExecution`.

Inside the code example below, there are two lines that f-strings are
used in the ``forward()`` class.
One is ``f"phase{phase}_ipu{ipu}"`` at Line 25,
where ``phase`` is
0, 1, 1, 2, 3, 3, 4, 5, and 5 respectively,
and ``ipu`` ranges from 0 to 1.
The total number of instances for this f-string is 12 due to
6 phases and 2 IPUs.
The other is ``f"phase{N*2-1}_ipu1"`` at Line 32,
where ``phase`` is 5 and ``ipu`` is 1.
When defining :py:class:`poptorch.Stage`,
four f-strings are used where ``n`` ranges from 0 to 2
at Line 46-47 and 50-51:

* ``f"phase_{2*n}_ipu0"``
* ``f"phase{2*n}_ipu1"``
* ``f"phase_{2*n+1}_ipu0"``
* ``f"phase{2*n+1}_ipu1"``

They refer to ``phase`` 0, 2, 4 and 1, 3, 5, with ``ipu0`` and ``ipu1``
respectively.
So all these 12 f-strings are defined in :py:class:`poptorch.BeginBlock`,
and used in :py:class:`poptorch.Stage` dynamically. They match exactly.

.. literalinclude:: phased_execution.py
  :caption: An example of parallel phased execution
  :language: python
  :linenos:
  :start-after: annotations_start
  :end-before: annotations_end
  :emphasize-lines: 24, 31, 45-46, 49-50


Parallel execution strategies
-----------------------------

With the above APIs as building blocks, we can set execution strategies
using the four kinds of execution modes, as shown below.
Note that the same annotation can be used for each of them.
They only differ in the method of parallelisation and tensor locations.

.. _sharded_execution:

poptorch.ShardedExecution
^^^^^^^^^^^^^^^^^^^^^^^^^

In this strategy, each IPU
will sequentially execute a distinct part of the model.
A single unit of processing :py:class:`poptorch.ShardedExecution` is a
shard.
A shard is specified using :py:class:`poptorch.Stage`,
or if no :py:class:`poptorch.Stage` is specified,
the `user_id` passed by
:py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block` is used.
Each shard is executed sequentially on a single IPU.
Multiple shards can be placed on multiple IPUs.
However, only one IPU is used at a time, while
the other IPUs are idle.
If an IPU is allocated to run consecutive stages,
PopART will merge consecutive stages into one on the same IPU.
Weights and activations will use the on-chip memory of the IPUs.
Layers sharing weights need be placed on the same IPU.

:py:class:`poptorch.ShardedExecution` can be useful
for processing a single sample or debugging.
Overall it has low efficiency since only one IPU is used at a time.

.. autoclass:: poptorch.ShardedExecution
   :inherited-members:

poptorch.PipelinedExecution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the default execution strategy.
It extends :ref:`sharded_execution` with parallel execution on multiple
IPUs.

Parallelisation in :py:class:`poptorch.PipelinedExecution`
requires :py:meth:`~poptorch.Options.deviceIterations`
and :py:meth:`~poptorch.options._TrainingOptions.gradientAccumulation`.
as explained in :ref:`efficient_data_batching`.
After one :py:class:`poptorch.Stage` is finished with processing a batch
on one IPU, it starts immediately processing the next batch.
This creates a pipeline where multiple batches are processed in parallel.
An IPU can only start its own :py:class:`poptorch.Stage` of a batch if
its previous :py:class:`poptorch.Stage` of the current batch is processed.
Hence, all IPUs will be occupied after a warm-up period.
A cool-down period is required to aggregate the results and apply weight
changes.

.. autoclass:: poptorch.PipelinedExecution
   :special-members: __init__
   :inherited-members:

Phased execution
^^^^^^^^^^^^^^^^

:py:class:`poptorch.ParallelPhasedExecution` and
:py:class:`poptorch.SerialPhasedExecution` have the following
features in common:

* A portion of the weights and activations are transferred to and from
  streaming memory, before and after each phase.
* If the desired weights and activations are already stored in an IPU
  of the same group of IPUs,
  intra-phase cross-IPU copies can replace the copies
  to and from streaming memory.
* This specific portion is needed by the layers of the model wrapped in
  :py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block` in current
  :py:class:`poptorch.Phase`.
* They both trade off some performance
  for larger models with higher memory needs.
* Any number of phases is allowed.
* The number of stages in each :py:class:`poptorch.Phase`
  should match the number of IPUs in each group of IPUs.
* Stages inside each :py:class:`poptorch.Phase` can run in parallel.

Although you only define the :py:class:`poptorch.Phase` for forward passes,
the corresponding phases for backward passes are created correspondingly.
The order of phased execution for backward passes won't change
but you can decide whether a phase is shared by both
forward and backward passes. In other words, you decide whether to avoid
a memory transfer of a portion of the weights and activations.

poptorch.SerialPhasedExecution
""""""""""""""""""""""""""""""

In :py:class:`poptorch.SerialPhasedExecution`,
phases execute on a single group of IPUs sequentially.

.. autoclass:: poptorch.SerialPhasedExecution
   :special-members: __init__
   :inherited-members:

poptorch.ParallelPhasedExecution
""""""""""""""""""""""""""""""""

In :py:class:`poptorch.ParallelPhasedExecution`,
phases are executed in parallel alternating between two groups of IPUs.
Even phases must run on even IPUs and odd phases on odd IPUs.
Inter-phase cross-IPU copies can replace the memory transfers to and from
the streaming memory, if the desired weights and activations are already
available in another group of IPUs.

.. autoclass:: poptorch.ParallelPhasedExecution
   :special-members: __init__
   :inherited-members:

In the code example above, there are three phases.
Each phase has two stages and
each IPU group has two IPUs, so these two numbers match.
Even phases 0 and 2 run on IPU 0 and 2, while odd phase 1 runs on IPU 1 and
3 as required. This allows for faster
cross-IPU copies, both inter-phase and intra-phase.

poptorch.Liveness
"""""""""""""""""

:py:class:`poptorch.Liveness` controls the availability of tensors on IPU,
and is only needed for
:py:class:`poptorch.ParallelPhasedExecution`
and :py:class:`poptorch.SerialPhasedExecution`.

.. autoclass:: poptorch.Liveness

The default :py:class:`poptorch.Liveness` is ``AlwaysLive``.
``OffChipAfterFwd`` and
``OffChipAfterEachPhase`` may be helpful if you run a large model
with a tight memory budget.


Optimizers
==========

You can use a number of optimizers with PopTorch.
In addition, PopTorch has additional features to support float16 models such as loss scaling.

.. autoclass:: poptorch.optim.SGD
   :special-members: __init__
   :members:

.. autoclass:: poptorch.optim.AdamW
   :special-members: __init__
   :members:

.. autoclass:: poptorch.optim.RMSprop
   :special-members: __init__
   :members:

.. autoclass:: poptorch.optim.LAMB
   :special-members: __init__
   :members:


Loss scaling
------------

When training models which use half/float16 values, you can use loss scaling  to prevent the gradients from becoming too small and underflowing.
Before calculating the gradients, PopTorch will scale the loss by the value of the ``loss_scaling`` parameter.
PopTorch will multiply the gradients by the inverse scale prior to updating the optimizer state.
Therefore, beyond improving numerical stability, neither the training nor the hyper-parameters are affected.

 Higher ``loss_scaling`` values can improve numerical stability by minimising underflow.
 However, too high a value can result in overflow.
 The optimal loss scaling factor depends on the model.


Velocity scaling (SGD only)
---------------------------

The SGD optimizer, when used with momentum, updates weights based on the velocity values.
At each update step, the new velocity is a combination of the gradients derived from the loss function and the previous velocity value.
Similar to loss scaling, the ``velocity_scaling`` parameter allows the velocity values to be scaled to improve numerical precision when using half/float16 values.
(Note that the gradients are, in effect, scaled by ``velocity_scaling/loss_scaling`` so the ``loss_scaling`` has no impact on the effective scaling of velocity parameters.)

As with loss scaling, higher values can minimise underflow of the velocity values but may result in overflow.


Custom ops
==========

These are helper operations to be used within a model.

poptorch.ipu_print_tensor
-------------------------

.. py:class:: ipu_print_tensor(tensor_to_print, optional_title)

    Adds a tensor to be printed on the IPU. When this is executed the tensor
    will be copied back to host and printed.

    When this operation is called in the backward pass it
    will print the gradient of the tensor.

    The operation is an identity operation and it will return the exact same
    tensor. The returned tensor should be used in place of the original tensor,
    in the rest of the program to make sure that the print operation isn't optimised away.

    For example if the original code looks like this:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        return a + b

    And you want to print the value of `a`.
    If you do:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        poptorch.ipu_print_tensor(a)
        return a + b

    Optionally, you may add a second string parameter to be used as a title.

    .. code-block:: python

    def forward(self, c, d, b)
        a = c + d
        poptorch.ipu_print_tensor(a, "summation"))
        return a + b

    The result of ``ipu_print_tensor`` is not used,therefore it will be optimised out by the
    graph optimiser and ``a`` will not be printed.

    Instead you should do:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        x = poptorch.ipu_print_tensor(a)
        return x + b

    .. warning::
       In order for the print operation to not be optimised out by the graph
       optimiser, you must use the output of the print.

    :param ipu_print_tensor: The tensor to print.
    :returns: The input unchanged.

.. literalinclude:: api.py
    :language: python
    :linenos:
    :start-after: print_tensor_start
    :end-before: print_tensor_end
    :emphasize-lines: 10


poptorch.identity_loss
----------------------

This function is used to implement custom losses. This takes in a single PyTorch tensor
and will backpropagate a gradient of ones through it.

.. warning::
   Passing a PyTorch loss function or another ``identity_loss`` to this function is not
   supported. Multiple losses must be implemented via composite PyTorch ops.

.. autofunction:: poptorch.identity_loss

.. literalinclude:: api.py
  :language: python
  :linenos:
  :start-after: identity_start
  :end-before: identity_end
  :emphasize-lines: 5

poptorch.MultiConv
----------------------

Use :py:class:`poptorch.MultiConv` wrapper class to define multi-convolutions.

.. autoclass:: poptorch.MultiConv
   :members:

.. autoclass:: poptorch.MultiConvPartialsType

.. autoclass:: poptorch.MultiConvPlanType

Half / float 16 support
=======================

PopTorch supports the half-precision floating point (float 16) format.
You can convert models to float 16 by using a ``Module``'s .``half()`` method and by using float 16 rather than float 32 tensors for the input.
(You can convert a tensor to float 16 using ``tensor = tensor.half()``)

.. literalinclude:: inferenceModel.py
    :language: python
    :caption: How to run a model using half precision
    :linenos:
    :start-after: inference_half_start
    :end-before: inference_half_end
    :emphasize-lines: 1, 2

Because PopTorch relies on the ``torch.jit.trace`` API, it is limited to tracing operations which run on the CPU.
Many of these operations do not support float 16 inputs.
To allow the full range of operations, PopTorch converts all float 16 inputs to float 32 before tracing and then restores the inputs to float 16 as part of the canonicalization process.
Some operations may result in the model running in float 32 where float 16 would
be expected, or vice versa (see :ref:`float_16_op_support` for full details).

Profiling
=========

You can profile a graph produced by PopTorch for analysis using the PopVision Graph Analyser, which can be downloaded from the Graphcore support portal.
To do this, use the :ref:`POPLAR_ENGINE_OPTIONS<profiling_env>` environment variable.

Environment variables
=====================

Logging level
-------------
PopTorch uses the following levels of logging:
  * ``OFF``: No logging.
  * ``ERR``: Errors only.
  * ``WARN``: Warnings and errors only.
  * ``INFO``: Info, warnings and errors. (Default)
  * ``DEBUG``: Adds some extra debugging information.
  * ``TRACE`` and ``TRACE_ALL``: Trace everything inside PopTorch.

The ``POPTORCH_LOG_LEVEL`` environment variable can be used to set the logging level:

.. code-block:: bash

  export POPTORCH_LOG_LEVEL=DEBUG

.. _profiling_env :

Profiling
---------

When running programs using PopTorch, you can enable profiling by using the ``POPLAR_ENGINE_OPTIONS`` environment variable used by Poplar.

In order to capture the reports needed for the PopVision Graph Analyser you only need to set ``POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true"}'``:

.. code-block:: bash

  export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true"}'

By default, report files are output to the current working directory. You can specify a different output directory by setting ``autoReport.directory``, for example:

.. code-block:: bash

  export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./tommyFlowers"}'

For more options, please refer to the `PopVision Graph Analyser User Guide <https://docs.graphcore.ai/projects/graphcore-popvision-user-guide>`_.

In order to capture the ``pvti`` reports needed for the PopVision System Analyser you only need to set ``PVTI_OPTIONS='{"enable":"true"}'``

You can also add extra tracepoints in your own code by using

.. autoclass:: poptorch.profiling.Channel
   :members:

IPU Model
---------

By default PopTorch will try to attach to a physical IPU.
If instead you want to use the model, you can do so by setting ``POPTORCH_IPU_MODEL`` to ``1``:

.. code-block:: bash

  export POPTORCH_IPU_MODEL=1

Please see the `Poplar and PopLibs User Guide <https://docs.graphcore.ai/projects/poplar-user-guide>`_ for the limitations of the IPU Model.

Wait for an IPU to become available
-----------------------------------

By default if you try to attach to an IPU but all the IPUs in the system are
already in use, an exception will be raised.
If you would rather wait for an IPU to become available, you can do so by setting ``POPTORCH_WAIT_FOR_IPU`` to ``1``.

.. code-block:: bash

  export POPTORCH_WAIT_FOR_IPU=1
