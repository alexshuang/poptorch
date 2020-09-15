# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import glob
import json
import os
import torch
from . import enums
from .logging import logger


class _OptionsDict:
    """Safe dictionary to store options: only keys which have been passed to
    the constructor can later be updated.
    """

    def __init__(self, **default_values):
        self._values = default_values

    def set(self, **kwargs):
        for option, value in kwargs.items():
            assert self.exists(option), ("Invalid option %s, valid options"
                                         " are %s") % (option,
                                                       self._values.keys())
            assert isinstance(
                value, type(self._values[option])
            ), "Unexpected type %s for option %s. Expected %s" % (
                type(value), option, type(self._values[option]))
            self._values[option] = value

    def createOrSet(self, **kwargs):
        for option, value in kwargs.items():
            if option in self._values:
                self.set(**{option: value})
            else:
                self._values[option] = value

    def exists(self, option):
        return option in self._values

    def __getstate__(self):
        return self._values

    def __setstate__(self, state):
        self._values = state

    def __getattr__(self, option):
        assert self.exists(
            option), ("Invalid option %s, "
                      "valid options are %s") % (option, self._values.keys())
        return self._values[option]

    def update(self, other):
        assert not set(self._values.keys()).intersection(
            other), "Can't merge dictionaries, they have some keys in common"
        other.update(self._values)
        return other

    def __call__(self, option):
        assert self.exists(
            option), ("Invalid option %s, "
                      "valid options are %s") % (option, self._values.keys())
        return self._values[option]


class _JitOptions(_OptionsDict):
    """Options related to Pytorch's JIT compiler.

    Can be accessed via `poptorch.Options`:

    >>> opts = poptorch.Options()
    >>> opts.Jit.traceModel(True)
    """

    def __init__(self):
        super().__init__(trace_model=True)

    def traceModel(self, trace_model):
        """
        If True: use torch.jit.trace
        If False: use torch.jit.script (Experimental)

        Trace model is enabled by default.
        """
        self.set(trace_model=trace_model)
        return self


class _TrainingOptions(_OptionsDict):
    """Options specific to model training.

    Note: You must not set these options for inference models.

    Can be accessed via `poptorch.Options`:

    >>> opts = poptorch.Options()
    >>> opts.Training.gradientAccumulation(4)
    """

    def __init__(self):
        super().__init__(gradient_accumulation=1)

    def gradientAccumulation(self, gradient_accumulation):
        """Number of samples to accumulate for the gradient calculation.
        Might be called "pipeline depth" in some other frameworks."""
        self.set(gradient_accumulation=gradient_accumulation)
        return self


class _PopartOptions:
    """Options specific to the PopART backend.

    Only for advanced users.

    Any option from `popart.SessionOptions` can be set using this class.
    Note: there is no mapping for the various PopART enums so integers need
    to be used instead.

    Can be accessed via `poptorch.Options`:

    >>> opts = poptorch.Options()
    >>> opts.Popart.set("autoRecomputation", 3) # RecomputationType::Pipeline
    """

    def __init__(self):
        self.options = {}

    def set(self, key, value):
        self.options[key] = value
        return self


class _DistributedOptions(_OptionsDict):
    """Options related to distributed execution.

    Can be accessed via `poptorch.Options`:

    >>> opts = poptorch.Options()
    >>> opts.Distributed.configureProcessId(0, 2)
    """

    def __init__(self):
        super().__init__(num_distributed_processes=1,
                         distributed_process_id=0,
                         ipuof_configs={})
        self._gcd_mappings = {}
        self.setEnvVarNames("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK")

    def disable(self):
        """Ignore the current options / environment variables and disable
        distributed execution.
        """
        self.set(num_distributed_processes=1, distributed_process_id=0)
        return self

    def IPUoFConfigFiles(self, files):
        """ List of IPUoF configuration files to use for the different
        GCDs.

        Important: One and exactly one configuration file must be provided
        for each GCD.

        :param files: one or more glob compatible expressions

        By default: `~/.ipuof.conf.d/*.conf`

        The default value will work if you only own one partition.

        If you own several then you will need to narrow down the number of
        configuration files so that only the configuration files corresponding
        to the partition to use are selected.

        For example: `~/.ipuof.conf.d/partitionA_*.conf`
        """
        if isinstance(files, str):
            files = [files]
        # Find all the config files
        all_files = []
        for f in files:
            all_files += glob.glob(os.path.expanduser(f))
        # remove duplicates
        all_files = set(all_files)
        self._gcd_mappings = {}
        for f in all_files:
            id = json.load(open(f))["attributes"].get("GCD Id")
            gcd = int(id) if id else 0
            assert gcd not in self._gcd_mappings, (
                f"Multiple config files "
                f"are registered to handle GCD {gcd}: {self._gcd_mappings[gcd]}"
                f" and {f}")
            self._gcd_mappings[gcd] = f
        return self

    def setEnvVarNames(self, var_num_processes, var_process_id):
        """Utility to read and set `processId` and `numProcesses` from
        environment variables.

        Useful if you use a third party library to manage the processes used for
        the distributed execution such as mpirun.

        For example: mpirun -np 4 myscript.py

        By default the OpenMPI "OMPI_COMM_WORLD_SIZE" and "OMPI_COMM_WORLD_RANK"
        variables are used.
        """
        return self.configureProcessId(
            int(os.environ.get(var_process_id, "0")),
            int(os.environ.get(var_num_processes, "1")))

    def configureProcessId(self, process_id, num_processes):
        """Manually set the current process ID and the total number of processess.

        :param int process_id: The ID of this process.
        :param int num_processes: The total number of processes the execution is
            distributed over.
        """
        self.set(distributed_process_id=process_id)
        self.set(num_distributed_processes=num_processes)
        return self

    def getGcdConfigFile(self):
        """Return all the GCD ids <-> file mappings.

        :meta private:
        """
        if not self._gcd_mappings:
            self.IPUoFConfigFiles("~/.ipuof.conf.d/*.conf")
        return self._gcd_mappings.get(self.processId)

    @property
    def processId(self):
        """Id of the current process."""
        return self.distributed_process_id

    @property
    def numProcesses(self):
        """Total number of processes the execution is distributed over."""
        return self.num_distributed_processes


class Options(_OptionsDict):
    """Options controlling how a model is run on the IPU.
    """

    def __init__(self):
        self._jit = _JitOptions()
        self._training = _TrainingOptions()
        self._popart = _PopartOptions()
        self._distributed = _DistributedOptions()

        super().__init__(replication_factor=1,
                         device_iterations=1,
                         log_dir=".",
                         anchor_mode=enums.AnchorMode.Default.value,
                         anchor_return_period=1,
                         use_model=False,
                         connection_type=enums.ConnectionType.Always.value,
                         sync_pattern=enums.SyncPattern.Full.value,
                         available_memory_proportion={})

    @property
    def Distributed(self):
        """Options specific to distributed execution.

        .. seealso:: :py:class:`poptorch.options._DistributedOptions`"""
        return self._distributed

    @property
    def Jit(self):
        """Options specific to upstream PyTorch's JIT compiler.

        .. seealso:: :py:class:`poptorch.options._JitOptions`"""
        return self._jit

    @property
    def Training(self):
        """Options specific to training.

        .. seealso:: :py:class:`poptorch.options._TrainingOptions`"""
        return self._training

    @property
    def Popart(self):
        """Options specific to the PopART backend.
        (Advanced users only).

        .. seealso:: :py:class:`poptorch.options._PopartOptions`"""
        return self._popart

    def deviceIterations(self, device_iterations):
        """Number of iterations the device should run over the data before
        returning to the user. (Default: 1)"""
        self.set(device_iterations=device_iterations)
        return self

    def enablePipelining(self, enable_pipelining):
        """Enable pipelining of virtual graphs (Default: False if 1 IPU used,
        True otherwise)"""
        self.createOrSet(enable_pipelining=enable_pipelining)
        return self

    def setAvailableMemoryProportion(self, available_memory_proportion):
        """Memory is set on a per IPU basis, this should be a dictionary
        of IPU ids and float values between 0 and 1.

        For example: {"IPU0": 0.5}
        """
        actual_memory = {}

        for key, mem in available_memory_proportion.items():
            assert key.startswith("IPU"), (
                "Available memory proportions are expected"
                " to be in a dictionary of {\"IPU0\": 0.5}"
                " where the 0 in IPU is the index of the"
                " IPU. Invalid key: %s" % key)

            ipu_id = int(key[3:])
            actual_memory[ipu_id] = mem

        self.createOrSet(available_memory_proportion=actual_memory)
        return self

    def replicationFactor(self, replication_factor):
        """Number of model replications (Default: 1).

        For example if your model uses 1 IPU, a
        replication factor of 2 will use 2 IPUs. If your model is
        pipelined across 4 IPUs, a replication factor of 4 will use 16 IPUs
        total.
        """
        self.set(replication_factor=replication_factor)
        return self

    def logDir(self, log_dir):
        """Where to save log files (Default: Current directory)"""
        self.set(log_dir=log_dir)
        return self

    def useIpuModel(self, use_model):
        """Use the IPU model or physical hardware.

        Default: False (Real Hardware).

        This setting takes precedence over the `POPTORCH_IPU_MODEL` environment
        variable.
        """
        self.set(use_model=use_model)
        return self

    def connectionType(self, connection_type):
        """set the IPU connection type to one of:

        :param poptorch.ConnectionType connection_type:
            * Always: Attach to the IPU from the start (Default).
            * OnDemand: Wait until the compilation is complete and the
              executable is ready to be run to attach to the IPU.
            * Never: Never try to attach to an IPU. (Useful for offline
              compilation, but trying to run an executable will raise
              an exception).
        """
        assert isinstance(connection_type, enums.ConnectionType)
        self.set(connection_type=connection_type.value)
        return self

    def syncPattern(self, sync_pattern):
        """Set the IPU SyncPattern.

        :param poptorch.SyncPattern sync_pattern:
            * Full
            * SinglePipeline
            * ReplicaAndLadder
        """
        assert isinstance(sync_pattern, enums.SyncPattern)
        self.set(sync_pattern=sync_pattern.value)
        return self

    def useIpuId(self, ipu_id):
        """ Use the specified IPU id as provided by `gc-info`.

        The number of IPUs associated with the id must be equal to the number
        of IPUs used by your grpah multiplied by the replication factor.

        For example if your model uses 1 IPU and the replication factor is 2
        you will need to provide an id with 2 IPUs.

        If your model is pipelined across 4 IPUs, the replication factor is 4,
        you will need to provide an id containing 16 IPUs total.

        :param int ipu_id: IPU id as provided by `gc-info`.
        """
        assert isinstance(ipu_id, int)
        self.createOrSet(ipu_id=ipu_id)
        return self

    def useOfflineIpuTarget(self, ipu_version=1):
        """Create an offline IPU target that can only be used for offline compilation.

        Note: the offline IPU target cannot be used if the IPU model is enabled.

        :param int ipu_version: IPU version to target (1 for mk1, 2 for mk2).
            Default: 1.
        """
        self.connectionType(enums.ConnectionType.Never)
        self.createOrSet(ipu_version=ipu_version)
        return self

    def anchorMode(self, anchor_mode, anchor_return_period=None):
        """ How much data to return from a model

        :param poptorch.AnchorMode anchor_mode:
                * All: Return a result for each batch.
                * Sum: Return the sum of all the batches
                * Final: Return the last batch.
                * EveryN: Return every N batches. N is passed in as
                    `anchor_return_period`.
                * Default: `All` for inference, `Final` for training.
        """
        assert isinstance(anchor_mode, enums.AnchorMode)

        # Check the anchor return period makes sense.
        if anchor_mode == enums.AnchorMode.EveryN:
            assert anchor_return_period and anchor_return_period > 0, (
                "EveryN"
                " anchor must have anchor_return_period set to valid"
                " positive integer")
        elif anchor_return_period:
            logger.info(
                "Anchor return period argument ignored with anchor_mode"
                " set to %s", anchor_mode)

        self.set(anchor_mode=anchor_mode.value,
                 anchor_return_period=anchor_return_period or 1)
        return self

    def defaultAnchorMode(self):
        """Return True if the anchor_mode is currently set to Default,
        False otherwise."""
        return self.anchor_mode == enums.AnchorMode.Default

    def randomSeed(self, random_seed):
        """Set the seed for the random number generator on the IPU.
        """
        assert isinstance(random_seed, int)
        torch.manual_seed(random_seed)
        self.createOrSet(random_seed=random_seed)
        return self

    def toDict(self):
        """ Merge all the options, except for the Jit ones, into a single
        dictionary to be serialised and passed to the C++ backend.

        :meta private:
        """
        assert not self.defaultAnchorMode(
        ), "An anchor mode must be picked before serialisation"
        out = {}
        out.update(self._popart.options)
        out = self.update(out)
        out = self._training.update(out)
        out = self._distributed.update(out)
        config_file = self._distributed.getGcdConfigFile()
        if self._distributed.numProcesses > 1 or config_file:
            assert config_file, ("No IPUoF configuration file found for "
                                 "processId %d" % self._distributed.processId)
            os.environ["IPUOF_CONFIG_PATH"] = config_file
            logger.debug("'IPUOF_CONFIG_PATH' set to %s for processId %d",
                         config_file, self._distributed.processId)

        return out
