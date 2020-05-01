import torch
import torch.nn as nn
# from main1 import Net2

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

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False, precision=6, threshold=100000)
    print(0)
    # net1 = Net()
        # net2 = Net()
        # net1.load_state_dict(torch.load(r"D:\PycharmProjects\deeplearning_01\yolo_v4\tests\net01.pt"))
        # net2.load_state_dict(torch.load(r"D:\PycharmProjects\deeplearning_01\yolo_v4\tests\net02.pt"))
    net1 = torch.load("./net01.pth")

    net2 = torch.load("./net02.pth")
    a = "abd"
    for n,p in net1.named_parameters():
        if n.find("bias")>-2 :
            print(n)
            print(p)
            print("***")
    print("****+++")
    for n,p in net2.named_parameters():
        # print(n.find("bias"),n)
        a = n.find("bias")
        b = False
        if n.find("bias")>-2 :
            print(n,n.find("bias"))
            print(p)
            print("***")
