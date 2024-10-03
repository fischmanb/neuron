import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.name = "MLP"
        self.layers = nn.Sequential(
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.layers(x)
        return y
