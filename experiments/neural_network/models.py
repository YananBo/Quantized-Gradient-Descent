import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarCNN(torch.nn.Module):
     def __init__(self, classes=10):
        super().__init__()

        in_channels = 3
        kernel_size = 5
        in1, in2, in3 = 512, 84, 84

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=kernel_size, padding=1)#, padding=2)
        self.conv3 = nn.Conv2d(64, out_channels=128, kernel_size=kernel_size, padding=1)

        self.dropout = nn.Dropout2d(0.25)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.dense1 = nn.Linear(in1, in2)
        self.dense2 = nn.Linear(in3, classes)

     def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 512)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.dense = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x