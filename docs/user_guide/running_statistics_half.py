# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch

# half_stats_begin
model = torch.nn.Sequential()
model.add_module('lin', torch.nn.Linear(16, 16))
model.add_module('bn', torch.nn.BatchNorm1d(16))
model.float()

opts = poptorch.Options()
opts.Precision.runningStatisticsAlwaysFloat(False)
poptorch_model = poptorch.inferenceModel(model, opts)
# half_stats_end