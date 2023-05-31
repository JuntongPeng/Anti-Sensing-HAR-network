import torch
import argparse
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from detection_module.CNNdetector import CNNdetector
import numpy as np
from dataset.csi_dataset import CSIdataset
from enc_dec.base_encdec import Encoder, Decoder


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")

    parser.add_argument('--sensing_module_dir', default='archived/best_model_sensing_module.pth', required=False,
                        help='Pretrained Sensing Module')
    parser.add_argument('--encdec_dir', default='archived/best_model_encdec_16.pth', required=False,
                        help='Pretrained Encoder Decoder')
    parser.add_argument('--stage', default='enc_dec', required=False)
    parser.add_argument('--bit_len', default=16, type=int, required=False)
    parser.add_argument('--random_mask', action='store_true', default=False)
    parser.add_argument('--mask_ratio', default=0.05, type=float, required=False)
    parser.add_argument('--preamble', action='store_true', default=False)

    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    print("-----loading dataset-----")
    train_dataset = CSIdataset(phase='train')
    val_dataset = CSIdataset(phase='test')
    print("-----dataset loaded-----")
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists('logs/%s' % datetime):
        os.makedirs('logs/%s' % datetime)
    if not os.path.exists('archived'):
        os.makedirs('archived')
    batch_size = 32
    bit_len = opt.bit_len
    print(bit_len)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print('-----loading model-----')
    sensing_module = torch.load(opt.sensing_module_dir)
    state = torch.load(opt.encdec_dir)
    best_enc_dir = "tuned_" + opt.encdec_dir
    encoder = Encoder(bit_len)
    decoder = Decoder(bit_len)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])

    print("-----training start-----")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        sensing_module = sensing_module.to(device)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

    if opt.stage == 'sensing_module':
        sensing_module.train()
        encoder.eval()
        decoder.eval()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(sensing_module.parameters(), lr=1e-6)
    elif opt.stage == 'enc_dec':
        sensing_module.eval()
        encoder.train()
        decoder.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-5)
    else:
        raise ValueError('stage not specified')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    best_acc = 0
    if opt.stage == 'enc_dec':
        best_acc = 1.0
    best_epoch = 0
    coef_trans = 10
    for epoch in range(1000):
        if epoch - best_epoch > 50:
            break

        if epoch % 2 == 0:
            if opt.stage == 'enc_dec':
                state = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}
                torch.save(state, 'logs/%s/tuned_encdec_%d.pth' % (datetime, epoch))

            elif opt.stage == 'sensing_module':
                torch.save(sensing_module, 'logs/%s/tuned_sensing_module_%d.pth' % (datetime, epoch))
            print('model saved')
            encoder.eval()
            decoder.eval()
            sensing_module.eval()
            with torch.no_grad():
                acc_num = 0
                detect_acc_num = 0
                for i, (csi, label) in enumerate(val_dataloader):
                    csi = csi.to(device)
                    label = label.to(device)
                    csi = torch.squeeze(csi, dim=1)
                    if opt.preamble:
                        input_bits = torch.zeros(csi.shape[0], bit_len).float().to(device)
                    else:
                        input_bits = torch.randint(0, 2, size=(csi.shape[0], bit_len)).float().to(device)
                    target = input_bits.clone().detach()

                    enc_output = encoder(input_bits)

                    if opt.random_mask:
                        rand_mat = torch.rand_like(enc_output)
                        mask_index = torch.where(rand_mat > 1-opt.mask_ratio)
                        enc_output[mask_index] = 0

                    dec_input = torch.multiply(enc_output, csi)

                    detect_result = sensing_module(dec_input.unsqueeze(1)).argmax(dim=1)
                    detect_acc_num += (label == detect_result).sum().item() / label.shape[0]

                    dec_output = decoder(dec_input)

                    predicted = (dec_output > 0.5).int()
                    batch_acc_num = (predicted == target.int()).float().mean()
                    acc_num = batch_acc_num + acc_num

                all_num = len(val_dataloader)
                print('epoch: %d, detection_acc: %f, transmission_ber:%f' \
                      % (epoch, detect_acc_num / all_num, (1 - acc_num / all_num) * 100))

                if opt.stage == 'sensing_module':
                    if detect_acc_num / all_num > best_acc:
                        best_acc = detect_acc_num / all_num
                        torch.save(sensing_module, 'archived/tuned_model_sensing_module.pth')
                        best_epoch = epoch
                        print('best model saved')
                elif opt.stage == 'enc_dec':
                    if (1 - acc_num / all_num) * 100 > 1:
                        coef_trans = 10
                    else:
                        coef_trans = 0
                    if detect_acc_num / all_num < best_acc and (1 - acc_num / all_num) * 100 < 1:
                        best_acc = detect_acc_num / all_num
                        torch.save(state, 'archived/tuned_model_encdec.pth')
                        best_epoch = epoch
                        print('best model saved')

        if opt.stage == 'sensing_module':
            sensing_module.train()
            encoder.eval()
            decoder.eval()
        elif opt.stage == 'enc_dec':
            sensing_module.eval()
            encoder.train()
            decoder.train()

        for i, (csi, label) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            csi = csi.to(device)
            label = label.to(device)
            csi = torch.squeeze(csi, dim=1)
            if opt.stage == 'enc_dec':
                input_bits = torch.randint(0, 2, size=(csi.shape[0], bit_len)).float().to(device)
            elif opt.stage == 'sensing_module':
                if opt.preamble:
                    input_bits = torch.zeros((csi.shape[0], bit_len)).float().to(device)
                else:
                    input_bits = torch.randint(0, 2, size=(csi.shape[0], bit_len)).float().to(device)
            target = input_bits.clone().detach()
            enc_output = encoder(input_bits)
            dec_input = torch.multiply(enc_output, csi)

            dec_output = decoder(dec_input)
            detect_result = sensing_module(dec_input.unsqueeze(1))

            if opt.stage == 'enc_dec':

                detection_loss = criterion(detect_result, label)
                transmission_loss = criterion(dec_output, target)
                loss = - detection_loss + coef_trans * transmission_loss
            elif opt.stage == 'sensing_module':
                detection_loss = criterion(detect_result, label)
                loss = detection_loss
            else:
                raise ValueError('stage not specified')

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        scheduler.step()
        print('epoch: %d, loss: %f, lr: %f, ratio:%f' % (epoch, loss.item(), optimizer.param_groups[0]['lr'],coef_trans))


if __name__ == '__main__':
    main()
