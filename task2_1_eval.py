import time

import torch
import argparse
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from enc_dec.base_encdec import Encoder, Decoder
import numpy as np
from dataset.csi_dataset import CSIdataset


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")

    parser.add_argument('--model_dir', default='archived/best_model_encdec.pth',
                        help='Stored Model Path')

    opt = parser.parse_args()
    return opt


def main():
    batch_size = 16
    bit_len = 16

    opt = test_parser()
    print("-----loading dataset-----")

    val_dataset = CSIdataset(phase='test')
    print("-----dataset loaded-----")

    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    if opt.model_dir != '':
        print('loading model')
        state = torch.load(opt.model_dir)
        encoder = Encoder()
        decoder = Decoder()
        encoder.load_state_dict(state['encoder'])
        decoder.load_state_dict(state['decoder'])
    else:
        raise ValueError('model path not specified')

    device = torch.device("cpu")
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        device = torch.device("cuda:0")

    encoder.eval()
    decoder.eval()
    t1 = time.time()

    with torch.no_grad():
        acc_num = 0
        for i, (csi, label) in tqdm(enumerate(val_dataloader)):
            csi = csi.to(device)
            label = label.to(device)
            csi = torch.squeeze(csi, dim=1)
            input = torch.randint(0, 2, size=(batch_size, bit_len)).float().to(device)
            target = input.clone().detach()

            enc_output = encoder(input)
            dec_input = torch.multiply(enc_output, csi)
            dec_output = decoder(dec_input)

            predicted = (dec_output > 0.5).int()
            batch_acc_num = (predicted == target.int()).float().mean()
            acc_num = batch_acc_num + acc_num
        all_num = len(val_dataloader)
        print('acc: %f' % (acc_num / all_num))

    t2 = time.time()
    print('time: %f' % (t2 - t1))


if __name__ == '__main__':
    main()