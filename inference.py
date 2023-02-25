import time

import torch
import argparse
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from detection_module.CNNdetector import CNNdetector
import numpy as np
from dataset.csi_dataset import CSIdataset


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")

    parser.add_argument('--model_dir', default='',
                        help='Continued training path')

    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    print("-----loading dataset-----")

    val_dataset = CSIdataset(phase='test')
    print("-----dataset loaded-----")

    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    if opt.model_dir != '':
        print('loading model')
        model = torch.load(opt.model_dir)
    else:
        raise ValueError('model path not specified')

    device = torch.device("cpu")
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda:0")

    model.eval()
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        acc_num = 0
        all_num = 0
        for i, (csi, label) in enumerate(val_dataloader):
            csi = csi.to(device)
            label = label.to(device)

            detect_result = model(csi).argmax(dim=1)
            print(detect_result)
            print(label)
            acc_num += (label == detect_result).sum().item()

            all_num += label.shape[0]

        print('acc: %f' % (acc_num / all_num))
    t2 = time.time()
    print('time: %f' % (t2 - t1))


if __name__ == '__main__':
    main()
