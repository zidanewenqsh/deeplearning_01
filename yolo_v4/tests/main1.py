'''
@Descripttion: 
@version: 
@Author: QsWen
@Date: 2020-04-27 22:45:03
@LastEditors: QsWen
@LastEditTime: 2020-04-27 22:45:22
'''

import torch
import torch.nn as nn
from PIL import Image
import os
import sys
import time
import numpy as np
from torchvision import transforms
from torch.utils import data
from torch import optim


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 0),
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(64*9*9,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )


    def forward(self, x):
        #
        # self.inputs = nn.Parameter(x,requires_grad=True)

        y1 = self.layer_1(x)
        y1 = y1.reshape(-1, 64*9*9)
        y2 = self.layer_2(y1)
        return y2

class Net2(nn.Module):
    def __init__(self, x):
        super(Net2, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 0),
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(64 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # self.layer_3 = nn.Parameter(torch.randn(3,3))
        # self.inputs = nn.Parameter(torch.randn(1,1))

        self.inputs = nn.Parameter(x, requires_grad=True)

    def forward(self, x):
        # print("x",x)
        # self.inputs = nn.Parameter(x, requires_grad=True)
        # print("self.inputs",self.inputs)
        y1 = self.layer_1(x)
        y1 = y1.reshape(-1, 64 * 9 * 9)
        y2 = self.layer_2(y1)
        return y2


class Dataset(data.Dataset):
    def __init__(self, datadir=r"D:\PycharmProjects\deeplearning_01\yolo_v4\datas\catdog"):
        self.transformer = transforms.Compose([
            transforms.Resize([50, 50]),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.datadir = datadir
        self.dataset = []

        for filename in os.listdir(datadir):
            self.dataset.append(filename)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        label = self.dataset[index].split('_')[0]
        # print("label",label)
        label = torch.Tensor([int(label)])
        # print(label2)
        img_file = os.path.join(self.datadir, self.dataset[index])
        with Image.open(img_file) as img:
            img_data = self.transformer(img)
        # print(img_data)
        return img_data, label


class Trainer:
    def __init__(self):
        self.module1 = Net1()
        # self.module2 = Net2()
        # self.optimizer = optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}])
        self.optimizer = optim.Adam(self.module1.parameters())
        # self.optimizer = optim.SGD(net.parameters(),lr=0.001)
        # self.optimizer = optim.SGD([{'params':net.layer_1.parameters(),'lr':0.001},{'params':net.layer_2.parameters(),'lr':0.001}])
        # self.optimizer_2 = optim.SGD(net.inputs,lr=0.001)
        self.loss_fn = nn.MSELoss()

    def train(self):
        torch.save(self.module1.state_dict(), 'net01.pt')
        torch.save(self.module1, 'net01.pth')
        # a0 = self.module.state_dict()['layer_2.4.weight']
        # print("a0", a0)
        epoch = 10
        dataset = Dataset()
        dataloader = data.DataLoader(dataset, 40, shuffle=False)

        for i in range(epoch):

            for j, (img_data, label) in enumerate(dataloader):

                net2 = Net2(img_data)
                net2.load_state_dict(net1.state_dict(), strict=False)
                # torch.save(net2, self)
                optimizer_2 = optim.Adam([{"params":net2.inputs,"lr":0.01}])
                # y = self.module(img_data)
                # print("image_data",img_data)
                y = self.module1(img_data)
                # print("inputs",self.module.inputs)
                # print("inputs2", net.state_dict()["inputs"][0])
                img_ = transforms.ToPILImage(mode='L')(net2.state_dict()["inputs"][0].detach())
                # print(torch.from_numpy(np.array(img_)))
                # print("1",torch.from_numpy(np.array(img_))[0, 0:10])
                # print("11",self.module.state_dict()['layer_2.4.weight'][0])
                # img_.save("./001.jpg")
                if i == 0 and j == 0:
                    a1 = net2.state_dict()['inputs'].detach()
                    print("a1", a1[-1])
                img_2 = transforms.ToPILImage(mode='L')(img_data[-1])
                img_2.save("./002.jpg")
                if i == 9 and j == 9:
                    print("imgdata",img_data[-1])
                # img_.show()

                # print("y",y,label)
                # print("1", self.module.state_dict()["inputs"][0, 0, 0, 0:10])
                # a1 = self.module.state_dict()['layer_2.4.weight'].detach().numpy()
                # print("a1",a1)
                loss = self.loss_fn(y, label)
                self.optimizer.zero_grad()
                optimizer_2.zero_grad()
                loss.backward()
                self.optimizer.step()
                optimizer_2.step()
                img_3 = transforms.ToPILImage(mode='L')(net2.state_dict()["inputs"][0].detach())
                # a2 = self.module.state_dict()['layer_2.4.weight'].detach().numpy()
                # print("a2",a2)
                # print("31", self.module.state_dict()['layer_2.4.weight'][0])
                # print("3",self.module.state_dict()["inputs"][0,0,0,0:10])
                # print("3",torch.from_numpy(np.array(img_3))[0, 0:10])
                # print((a1==a2))
                img_3.save("./003.jpg")
                # self.optimizer_2.step()
                if j % 10 ==0:
                    print("{}:{} loss:{}".format(i, j, loss))
                # print(i,j)
        torch.save(self.module1.state_dict(), 'net02.pt')
        img_ = transforms.ToPILImage(mode='L')(net2.state_dict()["inputs"][-1])
        # print(torch.from_numpy(np.array(img_)))
        torch.save(self.module1, 'net02.pth')
        img_.save("./000.jpg")
        a2 = net2.state_dict()['inputs'].detach()
        print("a2", a2[-1])

        # img_.show()

        return 0


if __name__ == '__main__':
    np.set_printoptions(precision=8, threshold=np.inf, suppress=True)
    torch.set_printoptions(sci_mode=False, precision=8,threshold=100000)
    print(0)
    x = torch.randn(1, 1, 50, 50)
    net1 = Net1()
    net2 = Net2(x)
    # y = net(x)
    # print(y.shape)
    print("net1")
    print(net1.state_dict()['layer_1.0.weight'][0])
    print("net2")
    net2.load_state_dict(net1.state_dict(),strict=False)
    print(net2.state_dict()['layer_1.0.weight'][0])
    print("*************")
    # print(net2.state_dict()['inputs'][0])
    # print(x[0])

    # for n, p in net1.named_parameters():
    #     print(n)
    #     print(p.shape)
    #     print(p)
    #     break
    # for n, p in net2.named_parameters():
    #     print(n)
    #     print(p.shape)
    #     print(p)
    #     break
    # print(net.inputs)
    # print(net.)


    # exit()
    # dataset = Dataset()
    # print(len(dataset))
    # dataloader = data.DataLoader(dataset, 1)

    # for img_data, label in dataloader:
    #     print(img_data)
    #     print(label.size())
    #     print(img_data[0,0,0,0:10])
    #     y = net(img_data)
    # print(net.parameters())
    # print(net.state_dict()["inputs"][0,0,0,0:10])
    # exit()
    #     # print(img_data)
    trainer = Trainer()
    trainer.train()
    # exit()
