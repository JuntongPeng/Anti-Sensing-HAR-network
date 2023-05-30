import torch
import argparse
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import numpy as np
from dataset.csi_dataset import CSIdataset
from enc_dec.base_encdec import Encoder, Decoder
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")

    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--bit_len', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--random_mask', action='store_true', default=False)

    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    print(opt.random_mask)
    batch_size = 32
    bit_len = opt.bit_len
    print(bit_len)
    writer = SummaryWriter('./tf-logs')
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

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    best_acc = 0
    best_epoch = 0
    for epoch in range(1000):
        if epoch - best_epoch > 50:
            break

        if epoch % 10 == 0:
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
                    input = torch.randint(0, 2, size=(csi.shape[0], bit_len)).float().to(device)
                    # input = torch.zeros((csi.shape[0], bit_len)).float().to(device)
                    target = input.clone().detach()

                    enc_output = encoder(input)

                    writer.add_image('enc_output', enc_output[0].cpu().numpy(), epoch)

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
            input = torch.randint(0, 2, size=(csi.shape[0], bit_len)).float().to(device)
            target = input.clone().detach()
            enc_output = encoder(input)

            # random mask
            if opt.random_mask:
                rand_mat = torch.rand_like(enc_output)
                mask_index = torch.where(rand_mat > 0.95)
                enc_output[mask_index] = 1

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
