import torch.nn as nn
import torch.nn.functional as F


class ModernM5(nn.Module):
    def __init__(self, n_input=1, num_classes=35, stride=2, n_channel=32, dropout=0.3):
        super().__init__()
        self.name = "ModernM5"
        # Input (Batch, 1, 13, 1000)
        # Conv1 output: (2, 32, 11, 461)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=(3, 80), stride=(1, stride), padding=(1, 39))
        self.bn1 = nn.BatchNorm2d(n_channel)

        # Conv2 output: (2, 32, 9, 459)
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(n_channel)

        # Conv3 output: (2, 64, 7, 457)
        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(2 * n_channel)

        # Conv4 output: (2, 64, 5, 500)
        self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=(3, 3), padding=(1, 23))  # Padding adjusted
        self.bn4 = nn.BatchNorm2d(2 * n_channel)

        # Pooling and Fully Connected layers as before
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * n_channel, num_classes)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return self.sigmoid(x)

