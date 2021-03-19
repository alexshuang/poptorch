# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import abc
import logging
import torch
import tqdm

from ._logging import logger

_begin_ipu_block = torch.ops.poptorch.begin_ipu_block


class ProgressBar:
    def __init__(self):
        self._bar = None
        self._last = 0

    def __call__(self, progress: int, total: int):
        if self._bar is None:
            # Remove {rate_fmt}{postfix} from the default format
            # as it doesn't really make sense for a compilation process
            #
            # Note: this is *not* a f-string
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            bar_format += "[{elapsed}<{remaining}]"
            self._bar = tqdm.tqdm(desc="Graph compilation",
                                  total=total,
                                  bar_format=bar_format)
        self._bar.update(progress - self._last)
        self._last = progress
        if progress == total:
            self._bar.close()
            self._bar = None


class OptionsDict:
    """Safe dictionary to store options: only keys which have been passed to
    the constructor can later be updated.
    """

    def __init__(self, **default_values):
        self._values = default_values

        # Keep a dictionary of warnings messages based on the parameter options:
        # these are printed when the dictionarys are consolidated. The use of a
        # dictionary allows a warning to be removed by the key, e.g. if there is
        # a warning that the default parameter has changed but the parameter is
        # specified explicitly.
        self._warnings = {}

        # Allow warnings to be disabled by adding them to the list
        self._warnings_disabled = set()

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

    def deleteIfExists(self, option):
        if self.exists(option):
            del self._values[option]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, option):
        assert self.exists(
            option), ("Invalid option %s, "
                      "valid options are %s") % (option, self._values.keys())
        return self._values[option]

    def update(self, other):
        for warning in self._warnings.values():
            if warning not in self._warnings_disabled:
                logger.warning(warning)

        assert not set(self._values.keys()).intersection(
            other), "Can't merge dictionaries, they have some keys in common"
        other.update(self._values)
        return other

    def toDict(self):
        return self.update({})

    def __call__(self, option):
        assert self.exists(
            option), ("Invalid option %s, "
                      "valid options are %s") % (option, self._values.keys())
        return self._values[option]


class IStageManager(abc.ABC):
    def __init__(self):
        self._next_auto_id = 0
        self._current_ipu = None
        # We expect Torch to trace the graph 3 times, so to avoid printing
        # the same messages several times we store all the messages and
        # print the first third of them at the end.
        self._debug_messages = []

    def clearDebug(self):
        self._debug_messages = []

    def _debug(self, *args):
        if logger.isEnabledFor(logging.DEBUG):
            self._debug_messages.append(args)

    def printDebug(self):
        n = len(self._debug_messages)
        # We assume the graph was traced 3 times if:
        # - Number of messages can be divided by 3
        # - The first message is identical to the n/3th and 2n/3th ones.
        is_triple_trace = n > 0 and n % 3 == 0\
                and self._debug_messages[0] == self._debug_messages[n//3] \
                == self._debug_messages[2*n//3]
        if is_triple_trace:
            for i in range(n // 3):
                logger.debug(*self._debug_messages[i])
        else:
            # Not sure what happened: in doubt print everything
            for m in self._debug_messages:
                logger.debug(m)

    def nextAutoId(self):
        id = self._next_auto_id
        self._next_auto_id += 1
        return str(id)

    @abc.abstractmethod
    def getStage(self, block_id):
        """Return the stage corresponding to the given block_id.
        """

    def beginStage(self, user_id, ipu_id_from_block):
        user_id = user_id or self.nextAutoId()
        self._current_ipu = ipu_id_from_block
        stage = self.getStage(user_id)
        # If the user specified an ipu_id in the option use that one
        ipu = stage._ipu if stage._ipu is not None else ipu_id_from_block  # pylint: disable=protected-access
        if ipu is None:
            self._debug(
                "No IPU specified for block %s: default to stage_id %d",
                user_id, stage._stage_id)  # pylint: disable=protected-access
            ipu = stage._stage_id  # pylint: disable=protected-access
        self._debug("Starting block id=%s stage=%d phase=%d ipu=%d", user_id,
                    stage._stage_id, stage._phase_id, ipu)  # pylint: disable=protected-access
        _begin_ipu_block(stage._stage_id, stage._phase_id, ipu)  # pylint: disable=protected-access

    def resetAutoId(self):
        self._next_auto_id = 0
