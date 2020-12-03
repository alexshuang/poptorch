#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import poptorch

#annotations_start
poptorch.setLogLevel(1)  # Force debug logging
N = 3
size = 10


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = []
        for n in range(N * 6):
            weight = torch.nn.Parameter(torch.rand(size, size),
                                        requires_grad=True)
            self.register_parameter(f"w{n}", weight)
            self.weights.append(weight)

    def forward(self, in0, target=None):
        phase = 0
        weight = iter(self.weights)
        with poptorch.Block("phase0_ipu0"):
            ins = torch.split(in0, size)
        for n in range(N * 3):
            out = []
            for ipu in range(2):
                x = ins[ipu]
                with poptorch.Block(f"phase{phase}_ipu{ipu}"):
                    x = torch.matmul(next(weight), x)
                    out.append(F.relu(x))
            ins = out[1], out[0]
            # We want 2 matmuls in the same phase
            if n % 3 != 1:
                phase += 1
        with poptorch.Block(f"phase{N*2-1}_ipu1"):
            res = ins[0] + ins[1]
            if target is None:
                return res
            return res, torch.nn.L1Loss(reduction="mean")(res, target)


input = torch.rand(size * 2, 1)
target = torch.rand(size, 1)
model = Model()
opts = poptorch.Options()
phases = []
# Alternate between 0-2 and 1-3
for n in range(N):
    phases.append([
        poptorch.Stage(f"phase{2*n}_ipu0").ipu(0),
        poptorch.Stage(f"phase{2*n}_ipu1").ipu(2)
    ])
    phases.append([
        poptorch.Stage(f"phase{2*n+1}_ipu0").ipu(1),
        poptorch.Stage(f"phase{2*n+1}_ipu1").ipu(3)
    ])
opts.setExecutionStrategy(poptorch.ParallelPhasedExecution(*phases))
poptorch_model = poptorch.trainingModel(model, opts)
poptorch_model.compile(input, target)
#annotations_end
