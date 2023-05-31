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
from torch.utils.tensorboard import SummaryWriter


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")

    parser.add_argument('--model_dir', default='',
                        help='Continued training path')

    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()

    datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists('logs/%s' % datetime):
        os.makedirs('logs/%s' % datetime)
    if not os.path.exists('archived'):
        os.makedirs('archived')
    print("-----loading dataset-----")
    train_dataset = CSIdataset(phase='train')
    val_dataset = CSIdataset(phase='test')
    print("-----dataset loaded-----")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("-----training start-----")
    writer = SummaryWriter('./tf-logs')
    model = CNNdetector()
    if opt.model_dir != '':
        print('loading model')
        model = torch.load(opt.model_dir)
    else:
        print('building model')
        model = CNNdetector()
    # writer.add_graph(model, torch.rand(1, 1, 64, 256))
    # writer.close()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda:0")
        model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    best_epoch = 0
    for epoch in range(1000):
        if epoch - best_epoch > 50:
            break
        if epoch % 1 == 0:
            torch.save(model, 'logs/%s/model_%d.pth' % (datetime, epoch))
            print('model saved')
            model.eval()
            with torch.no_grad():
                acc_num = 0
                all_num = 0
                for i, (csi, label) in enumerate(val_dataloader):
                    csi = csi.to(device)
                    label = label.to(device)

                    detect_result = model(csi).argmax(dim=1)

                    batch_acc_num = (label == detect_result).sum().item()
                    acc_num = batch_acc_num + acc_num

                    all_num += label.shape[0]

                print('epoch: %d, acc: %f' % (epoch, acc_num / all_num))
                writer.add_scalar('acc', acc_num / all_num, epoch)
                if acc_num / all_num > best_acc:
                    best_acc = acc_num / all_num
                    torch.save(model, 'archived/best_model_sensing_module.pth')
                    best_epoch = epoch
                    print('best model saved')
        for i, (csi, label) in enumerate(train_dataloader):
            csi = csi.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            detect_result = model(csi)
            loss = criterion(detect_result, label)
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
        scheduler.step()
        print('epoch: %d, loss: %f, lr: %f' % (epoch, loss.item(), optimizer.param_groups[0]['lr']))
        writer.add_scalar('loss', loss.item(), epoch)

if __name__ == '__main__':
    main()
