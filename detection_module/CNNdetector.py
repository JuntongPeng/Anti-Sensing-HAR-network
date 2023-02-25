import torch
import torch.nn as nn


class CNNdetector(nn.Module):
    def __init__(self):
        super(CNNdetector, self).__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 8, kernel_size=1))

        self.conv3 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 8, kernel_size=1))
        self.conv1 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=1))

        self.fcn = nn.Sequential(nn.Linear(64 * 128 * 8, 128 * 8),
                                 nn.ReLU(),
                                 nn.Linear(128*8, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 8))

    def forward(self, x):

        y1 = self.conv5(x)

        y2 = self.conv3(x)

        y = torch.cat((y1, y2), 1)
        y = self.conv1(y)

        y = y.view(-1, y.shape[1] * y.shape[2] * y.shape[3])

        out = self.fcn(y).softmax(dim=1)


        return out
