import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, bit_len):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=16 * bit_len, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64 * 256)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = x.view(-1, 64, 256)

        return x


class Decoder(nn.Module):
    def __init__(self, bit_len):
        super(Decoder, self).__init__()
        self.bit_len = bit_len
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2))
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1))

        self.fc1 = nn.Linear(in_features=4096, out_features=16 * bit_len)
        self.fc2 = nn.Linear(in_features=16 * bit_len, out_features=1 * bit_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), 1)
        x = self.pooling(x)
        x = self.conv3(x)

        x = x.view(x.shape[0], 4096)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(-1, self.bit_len)

        return x
