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

        csi = self.dataset['csi']
        real=np.real(csi)
        img=np.imag(csi)
        print(real.shape)
        print(img.shape)
        csi=np.concatenate((real,img),axis=2)
        print(csi.shape)
        self.dataset_csi = torch.tensor(csi.astype(np.float32))

        self.dataset_label = torch.tensor(self.dataset['label'][0].astype(np.int64))

        # ones = torch.sparse.torch.eye(8)
        # self.dataset_label = ones.index_select(0, dataset_label)

    def __getitem__(self, idx):

        csi = self.dataset_csi[idx].unsqueeze(0)
        label = self.dataset_label[idx]

        return csi, label

    def __len__(self):
        return self.dataset['csi'].shape[0]
