
import torch
import torch.nn as nn

class ExampleNet(nn.Module):

	def __init__(self):
		super(ExampleNet, self).__init__()
		self.conv2d_4 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True)
		self.reLU_5 = nn.ReLU(inplace = False)
		self.maxPool2D_6 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, return_indices = False, ceil_mode = False)
		self.conv2d_9 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True)
		self.reLU_10 = nn.ReLU(inplace = False)
		self.maxPool2D_11 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, return_indices = False, ceil_mode = False)
		self.conv2d_12 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True)
		self.reLU_13 = nn.ReLU(inplace = False)
		self.linear_7 = nn.Linear(in_features = 3*3*32, out_features = 120, bias = True)
		self.reLU_16 = nn.ReLU(inplace = False)
		self.linear_17 = nn.Linear(in_features = 120, out_features = 84, bias = True)
		self.reLU_19 = nn.ReLU(inplace = False)
		self.linear_18 = nn.Linear(in_features = 84, out_features = 10, bias = True)

	def forward(self, x_para_1):
		x_conv2d_4 = self.conv2d_4(x_para_1)
		x_reLU_5 = self.reLU_5(x_conv2d_4)
		x_maxPool2D_6 = self.maxPool2D_6(x_reLU_5)
		x_conv2d_9 = self.conv2d_9(x_maxPool2D_6)
		x_reLU_10 = self.reLU_10(x_conv2d_9)
		x_maxPool2D_11 = self.maxPool2D_11(x_reLU_10)
		x_conv2d_12 = self.conv2d_12(x_maxPool2D_11)
		x_reLU_13 = self.reLU_13(x_conv2d_12)
		x_reshape_15 = torch.reshape(x_reLU_13,shape = (-1,3*3*32))
		x_linear_7 = self.linear_7(x_reshape_15)
		x_reLU_16 = self.reLU_16(x_linear_7)
		x_linear_17 = self.linear_17(x_reLU_16)
		x_reLU_19 = self.reLU_19(x_linear_17)
		x_linear_18 = self.linear_18(x_reLU_19)
		return x_linear_18
