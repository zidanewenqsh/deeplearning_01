import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Parameter(torch.zeros(2,2))
        self.layer2 = nn.Linear(2,2)

        for p in self.layer2.parameters():
            init.constant_(p, 2)

    def forward(self, x):
        self.layer = nn.Parameter(torch.ones(2, 2))
        y1 = torch.matmul(x, self.layer)
        y = self.layer2(y1)
        return y
if __name__ == '__main__':
    net= Net()
    x = torch.Tensor([1,2,3,4]).reshape(2,2)
    y = net(x)
    print(y)
    print(y.mean())
    label = 6
    optimizer = optim.SGD(net.parameters(),lr=0.001)
    for n,p in net.named_parameters():
        print(n)
        print(p)
    loss = (y.mean()-label)**2
    print("loss:",loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for n,p in net.named_parameters():
        print(n)
        print(p)