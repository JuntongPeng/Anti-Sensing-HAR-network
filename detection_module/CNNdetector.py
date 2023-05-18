import torch
import torch.nn as nn


class CNNblock(nn.Module):
    def __init__(self):
        super(CNNblock, self).__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(8),
                                   nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 8, kernel_size=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(8))

        self.conv3 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(8),
                                   nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 8, kernel_size=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(8))

        self.conv1 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1))

    def forward(self, x):
        y1 = self.conv5(x)
        y2 = self.conv3(x)
        y = torch.cat((y1, y2), 1)
        y = self.conv1(y)
        return y


class CNNdetector(nn.Module):
    def __init__(self):
        super(CNNdetector, self).__init__()
        self.cnn1 = CNNblock()
        self.cnn2 = CNNblock()
        self.cnn3 = CNNblock()
        self.cnn4 = CNNblock()

        self.fcn = nn.Sequential(nn.Linear(64 * 256 * 1, 128 * 8),
                                 nn.ReLU(),
                                 nn.Linear(128 * 8, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 8))

    def forward(self, x):

        y = self.cnn1(x)+x
        y = self.cnn2(y)+y
        y = self.cnn3(y)+y
        y = self.cnn4(y)+y

        y = y.view(-1, y.shape[1] * y.shape[2] * y.shape[3])

        out = self.fcn(y).softmax(dim=1)

        return out
