
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.ExampleNet import ExampleNet

def main():

	epochs = 4
	batch_size = 512

	net = ExampleNet()
	criterion = nn.MSELoss(reduce = None, size_average = None)
	optimizer = optim.Adam(net.parameters(), weight_decay = 0, amsgrad = False, lr = 0.001, betas = (0.9, 0.999), eps = 1e-08)

	transform = transforms.Compose([
		transforms.Resize(28),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,)),
	])
	dataset = datasets.MNIST("datasets/", train=True, download=False,transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	testdataset = datasets.MNIST("datasets/", train=False, download=True,transform=transform)
	testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False)

	losses = []
	for i in range(epochs):
		print("epochs: {}".format(i))
		for j, (input, target) in enumerate(dataloader):
			output = net(input)
			target = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 1)
			loss = criterion(output, target)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if j % 10 == 0:
				losses.append(loss.float())
				print("[epochs - {0} - {1}/{2}]loss: {3}".format(i, j, len(dataloader), loss.float()))
				plt.clf()
				plt.plot(losses)
				plt.pause(0.01)
		with torch.no_grad():
			correct = 0
			total = 0
			for input, target in testdataloader:
				output = net(input)
				_, predicted = torch.max(output.data, 1)
				total += target.size(0)
				correct += (predicted == target).sum()
				accuracy = correct.float() / total
			print("[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))
	save_model = torch.jit.trace(net,  torch.rand(1, 1, 28, 28))
	save_model.save("models/net.pth")


