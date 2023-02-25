import torch
import argparse
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from detection_module.CNNdetector import CNNdetector
import numpy as np
from dataset.csi_dataset import CSIdataset


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")

    parser.add_argument('--model_dir', default='',
                        help='Continued training path')

    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    print("-----loading dataset-----")
    train_dataset = CSIdataset(phase='train')
    val_dataset = CSIdataset(phase='test')
    print("-----dataset loaded-----")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    print("-----training start-----")

    if opt.model_dir != '':
        print('loading model')
        model = torch.load(opt.model_dir)
    else:
        print('building model')
        model = CNNdetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda:0")
    model.train()
    criterion = nn.MSELoss()
    best_acc = 0
    for epoch in range(200):

        for i, (csi, label) in enumerate(train_dataloader):
            csi = csi.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            detect_result = model(csi)

            loss = criterion(detect_result, label)
            loss.backward()
            optimizer.step()
            # if i % 500 == 0:
            #     print('epoch: %d, step: %d, loss: %f' % (epoch, i, loss.item()))

            torch.cuda.empty_cache()
        print('epoch: %d, loss: %f' % (epoch, loss.item()))

        if epoch % 3 == 0:
            torch.save(model, 'logs/model_%d.pth' % epoch)
            print('model saved')
            model.eval()
            with torch.no_grad():
                acc_num = 0
                all_num = 0
                for i, (csi, label) in enumerate(val_dataloader):
                    csi = csi.to(device)
                    label = np.where(label.cpu().numpy() == 1)[1]

                    detect_result = model(csi).argmax(dim=1).cpu().numpy()

                    acc_num += np.sum(label == detect_result)
                    all_num += len(label)

                    torch.cuda.empty_cache()
                print('epoch: %d, acc: %f' % (epoch, acc_num / all_num))
                if acc_num / all_num > best_acc:
                    best_acc = acc_num / all_num
                    torch.save(model, 'archived/best_model_sensing_module.pth')
                    print('best model saved')


if __name__ == '__main__':
    main()
