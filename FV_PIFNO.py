import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, size, padding=0, padding_add=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.size = size
        self.padding = padding
        self.padding_add = padding_add
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        b = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(self.size + self.padding + self.padding_add*2,
                                        self.size + self.padding + self.padding_add*2))
        return x

class Net2d(nn.Module):
    def __init__(self, modes: int, width: int, size: int, input_dim: int):
        super().__init__()
        self.modes = modes
        self.size = size
        self.padding = 0
        self.padding_add = 3

        self.fc1 = nn.Linear(input_dim, width)
        self.conv1 = SpectralConv2d_fast(width, width, modes, modes, size, self.padding, self.padding_add)
        self.conv2 = SpectralConv2d_fast(width, width, modes, modes, size, self.padding, self.padding_add)
        self.conv3 = SpectralConv2d_fast(width, width, modes, modes, size, self.padding, self.padding_add)
        self.fc2 = nn.Linear(width, 1)

        self.bn1 = nn.BatchNorm2d(width)
        self.bn2 = nn.BatchNorm2d(width)
        self.bn3 = nn.BatchNorm2d(width)

        self.a1 = nn.Parameter(torch.full((1, width, size + self.padding_add*2, size + self.padding_add*2), 0.5))
        self.a2 = nn.Parameter(torch.full((1, width, size + self.padding_add*2, size + self.padding_add*2), 0.5))
        self.a3 = nn.Parameter(torch.full((1, width, size + self.padding_add*2, size + self.padding_add*2), 0.5))

        self.W1 = nn.Conv2d(width, width, 1)
        self.W2 = nn.Conv2d(width, width, 1)
        self.W3 = nn.Conv2d(width, width, 1)

    def forward(self, x):
        # x: (B, H, W, C)
        b, h, w, c = x.size()
        x = self.fc1(x)                    # (B,H,W,width)
        x = x.permute(0, 3, 1, 2)          # (B,width,H,W)

        x = F.pad(x, [self.padding_add, self.padding + self.padding_add,
                      self.padding_add, self.padding + self.padding_add])

        x = self.bn1(x); x = F.relu(x * self.a1)
        x = self.conv1(x) + self.W1(x)

        x = self.bn2(x); x = F.relu(x * self.a2)
        x = self.conv2(x) + self.W2(x)

        x = self.bn3(x); x = F.relu(x * self.a3)
        x = self.conv3(x) + self.W3(x)

        x = x[..., self.padding_add:-(self.padding + self.padding_add),
                  self.padding_add:-(self.padding + self.padding_add)]
        x = x.permute(0, 2, 3, 1)          # (B,H,W,width)
        x = self.fc2(x)                    # (B,H,W,1)
        x = x.permute(0, 3, 1, 2)          # (B,1,H,W)
        x = x.view(b, h, w).squeeze()      # (B,H,W) -> pressure
        return x
