import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import scipy.io as io
class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

MyBinarize = MySign.apply

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class GCIM(nn.Module):
    def __init__(self, dim):
        super(GCIM, self).__init__()
        self.cat = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, stride=1)
        self.grad = GradBlock(dim)
        # base
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, stride=1, bias=True)
        self.res1 = ResBlock(dim * 2)
        self.res2 = ResBlock(dim * 2)
        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, stride=1, bias=True)
        # lstm
        pad_x = 1
        self.conv_xf = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_x)
        self.conv_xi = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_x)
        self.conv_xo = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_x)
        self.conv_xj = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_x)

        pad_h = 1
        self.conv_hf = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_h)
        self.conv_hi = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_h)
        self.conv_ho = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_h)
        self.conv_hj = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_h)

    def forward(self, x, h, c, y, Phi, PhiT):
        if h is None and c is None:

            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            x = self.grad(x, y, Phi, PhiT)
            c = x + self.conv2(self.res2(self.res1(self.conv1(self.cat(torch.cat([x, i * j], dim=1))))))
            h = o * F.tanh(c)

        else:

            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c
            c = self.grad(c, y, Phi, PhiT)
            c = c + self.conv2(self.res2(self.res1(self.conv1(self.cat(torch.cat([c, i * j], dim=1))))))
            h = o * F.tanh(c)

        return c, h, c


class GradBlock(nn.Module):
    def __init__(self, dim):
        super(GradBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)
        self.res1 = ResBlock(dim)

    def forward(self, x, y, Phi, PhiT):

        x_pixel = self.conv1(x)
        Phix = F.conv2d(x_pixel, Phi, padding=0, stride=32, bias=None)
        delta = y - Phix
        x_pixel = nn.PixelShuffle(32)(F.conv2d(delta, PhiT, padding=0, bias=None))
        x_delta = self.conv2(x_pixel)
        x = self.res1(x_delta) + x
        return x


class RGCDUN(nn.Module):
    def __init__(self, sensing_rate, LayerNo):
        super(RGCDUN, self).__init__()

        self.measurement = int(sensing_rate * 1024)
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))
        self.base = 16
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)
        layer1 = []
        self.LayerNo = LayerNo
        for i in range(LayerNo):
            layer1.append(GCIM(self.base))
        self.fcs1 = nn.ModuleList(layer1)

    def forward(self, x):
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_
        Phi = Phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)
        y = F.conv2d(x, Phi, padding=0, stride=32, bias=None)
        x = F.conv2d(y, PhiT, padding=0, bias=None)
        x = nn.PixelShuffle(32)(x)
        x = self.conv1(x)
        h = None
        c = None
        for i in range(self.LayerNo):
            if i == 0:
                c, h, x = self.fcs1[i](x, h, c, y, Phi, PhiT)
            elif i == self.LayerNo - 1:
                c, h, x = self.fcs1[i](x, h, c, y, Phi, PhiT)
            else:
                for j in range(5):
                    c, h, x = self.fcs1[i](x, h, c, y, Phi, PhiT)
        x = self.conv2(c)
        phi_cons = torch.mm(self.Phi, self.Phi.t())
        return x, phi_cons
