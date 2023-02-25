import numpy as np
import scipy.io as scio
import torch
from torch import nn


class CSIdataset(nn.Module):
    def __init__(self, phase='train'):
        super(CSIdataset, self).__init__()
        assert phase in ['train', 'test']
        self.phase = phase

        if phase == 'train':
            self.dataset = scio.loadmat('dataset/csi_train_128_v3.mat')
        elif phase == 'test':
            self.dataset = scio.loadmat('dataset/csi_eval_128_v3.mat')
        self.dataset_csi = self.dataset['csi']
        self.dataset_label = torch.tensor(self.dataset['label'][0].astype(np.int64))
        self.dataset_label = self.dataset_label.unsqueeze(1)
        self.dataset_label = self.make_one_hot(self.dataset_label, 8)



    def make_one_hot(self,input, num_classes):
        """Convert class index tensor to one hot encoding tensor.
        Args:
             input: A tensor of shape [bs, 1, *]
             num_classes: An int of number of class
        Returns:
            A tensor of shape [bs, num_classes, *]
        """
        shape = np.array(input.shape)
        shape[1] = num_classes
        shape = tuple(shape)
        result = torch.zeros(shape)
        result = result.scatter_(1, input.cpu(), 1)



        return result
    def __getitem__(self,idx):

        csi = torch.tensor(self.dataset_csi[idx].astype(np.float32)).unsqueeze(0)
        label = self.dataset_label[idx]

        return csi,label

    def __len__(self):
        return self.dataset['csi'].shape[0]


