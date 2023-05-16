import torch
import argparse
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import numpy as np
from dataset.csi_dataset import CSIdataset
from enc_dec.deep_encdec import Encoder, Decoder


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")

    parser.add_argument('--model_dir', default='',
                        help='Continued training path')

    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()

    batch_size = 32
    bit_len = 128

    print("-----loading dataset-----")
    train_dataset = CSIdataset(phase='train')
    val_dataset = CSIdataset(phase='test')
    print("-----dataset loaded-----")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)

    datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists('logs/enc_dec/%s' % datetime):
        os.makedirs('logs/enc_dec/%s' % datetime)

    print("-----training start-----")

    if opt.model_dir != '':
        print('loading model')
        state = torch.load(opt.model_dir)
        encoder = Encoder(bit_len)
        decoder = Decoder(bit_len)
        encoder.load_state_dict(state['encoder'])
        decoder.load_state_dict(state['decoder'])
    else:
        print('building model')
        encoder = Encoder(bit_len)
        decoder = Decoder(bit_len)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        device = torch.device("cuda:0")
    encoder.train()
    decoder.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=3e-5)
    scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

    best_acc = 0
    best_epoch = 0
    for epoch in range(1000):
        if epoch - best_epoch > 50:
            break

        if epoch % 5 == 0:
            state = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}
            torch.save(state, 'logs/enc_dec/%s/model_%d.pth' % (datetime, epoch))
            print('model saved')
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                acc_num = 0
                for i, (csi, label) in enumerate(val_dataloader):
                    csi = csi.to(device)
                    label = label.to(device)
                    csi = torch.squeeze(csi, dim=1)
                    input = torch.randint(0, 2, size=(batch_size, bit_len)).float().to(device)
                    #input = torch.zeros((csi.shape[0], bit_len)).float().to(device)
                    target = input.clone().detach()

                    enc_output = encoder(input)
                    dec_input = torch.multiply(enc_output, csi)
                    dec_output = decoder(dec_input)

                    predicted = (dec_output > 0.5).int()
                    batch_acc_num = (predicted == target.int()).float().mean()
                    acc_num = batch_acc_num + acc_num

                all_num = len(val_dataloader)
                print('epoch: %d, acc: %f' % (epoch, acc_num / all_num))
                if acc_num / all_num > best_acc:
                    best_acc = acc_num / all_num
                    torch.save(state, 'archived/best_model_encdec_%s.pth' % bit_len)
                    best_epoch = epoch
                    print('best model saved')

        for i, (csi, label) in enumerate(train_dataloader):
            csi = csi.to(device)
            label = label.to(device)
            csi = torch.squeeze(csi, dim=1)
            input = torch.randint(0, 2, size=(batch_size, bit_len)).float().to(device)
            target = input.clone().detach()

            enc_output = encoder(input)
            dec_input = torch.multiply(enc_output, csi)
            dec_output = decoder(dec_input)

            loss = criterion(dec_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        print('epoch: %d, loss: %f' % (epoch, loss.item()))
        scheduler.step()

if __name__ == '__main__':
    main()

