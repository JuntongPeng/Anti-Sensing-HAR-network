import torch


class BaseEncDec(torch.nn.Module):

    def __int__(self):
        super(BaseEncDec, self).__init__()

    def forward(self, x):
        raise NotImplementedError
