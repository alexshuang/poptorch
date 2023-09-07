import torch
import torch.nn as nn
import torchvision
import poptorch

# Normal pytorch batch size
training_batch_size = 20
validation_batch_size = 100

# Define the network using the above blocks.
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(320, 256)
        self.layer2_act = nn.ReLU()
        self.layer3 = nn.Linear(256, 10)

        self.softmax = nn.LogSoftmax(1)
        self.loss = nn.NLLLoss(reduction="mean")

    def forward(self, x, target=None):
        x = self.layer2_act(self.layer1(x))
        x = self.layer3(x)
        x = self.softmax(x)

        if target is not None:
            loss = self.loss(x, target)
            return x, loss
        return x

# Create our model.
model = Network()


opts = poptorch.Options()
opts.randomSeed(42)

inputs = torch.randn((5, 320))
# Same model as above, they will share weights (in 'model') which once training is finished can be copied back.
inference_model = poptorch.inferenceModel(model, opts)
output = inference_model(inputs)
print(output)
