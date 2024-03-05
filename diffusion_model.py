# import packages
import math

import torch
import torch.nn as nn

from utils import device

# define time embedding block to turn batch of timestep into time embeddings
class time_embedding_block(nn.Module):
    def __init__(self, time_embedding_dim):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.frequencies = torch.exp(- 2 * (torch.arange(self.time_embedding_dim // 2) + 1) / self.time_embedding_dim * math.log(10000)).to(device)
        self.Linear = nn.Linear(time_embedding_dim, time_embedding_dim)
        self.ReLU = nn.ReLU()

    def forward(self, timestep):
        tmp_device = timestep.device
        time_embeddings = torch.zeros(timestep.shape[0], self.time_embedding_dim).to(tmp_device)
        self.frequencies = self.frequencies.to(tmp_device)
        timestep = timestep.unsqueeze(dim=1)
        values = self.frequencies * timestep
        time_embeddings[:, 0::2] = torch.sin(values)
        time_embeddings[:, 1::2] = torch.cos(values)
        time_embeddings = self.Linear(time_embeddings)
        time_embeddings = self.ReLU(time_embeddings)
        return time_embeddings

# define convolution building block used in UNet
class convolution_block(nn.Module):
    def __init__(self, in_dim, out_dim, time_embedding_dim):
        super().__init__()
        self.convolution_layer_1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

        self.time_embedding_layer = nn.Sequential(
            nn.Linear(time_embedding_dim, out_dim),
            nn.ReLU(),
            nn.Unflatten(1, (out_dim, 1, 1))
        )

        self.convolution_layer_2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x, time_embeddings):
        output = self.convolution_layer_1(x)
        time_embeddings = self.time_embedding_layer(time_embeddings)
        output = output + time_embeddings
        output = self.convolution_layer_2(output)
        return output

# define downsampling block used in UNet
class downsample_block(nn.Module):
    def __init__(self, in_dim, out_dim, time_embedding_dim):
        super().__init__()
        self.convolution_block = convolution_block(in_dim, out_dim, time_embedding_dim)
        self.MaxPool = nn.MaxPool2d((2, 2))

    def forward(self, x, time_embeddings):
        residual_x = self.convolution_block(x, time_embeddings)
        output = self.MaxPool(residual_x)
        return residual_x, output

# define the attention block
class attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.ReLU = nn.ReLU()
    
    def forward(self, g, x):
        output_g = self.W_g(g)
        output_x = self.W_x(x)
        output_psi = self.ReLU(output_g + output_x)
        output_psi = self.psi(output_psi)
        output = output_psi * x
        return output

# define upsampling block used in UNet
class upsample_block(nn.Module):
    def __init__(self, in_dim, out_dim, time_embedding_dim, use_attention):
        super().__init__()
        self.use_attention = use_attention
        self.upsample = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0)
        self.attention_block = attention_block(out_dim, out_dim, out_dim // 2)
        self.convolution_block = convolution_block(2 * out_dim, out_dim, time_embedding_dim)

    def forward(self, x, residual_x, time_embeddings):
        x = self.upsample(x)
        if self.use_attention:
            residual_x = self.attention_block(x, residual_x)
        x = torch.cat([x, residual_x], axis=1)
        output = self.convolution_block(x, time_embeddings)
        return output

# define the UNet model
class UNet_Model(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        self.time_embedding_block = time_embedding_block(time_embedding_dim=32)

        self.downsample_block_1 = downsample_block(in_dim=3, out_dim=64, time_embedding_dim=32)
        self.downsample_block_2 = downsample_block(in_dim=64, out_dim=128, time_embedding_dim=32)
        self.downsample_block_3 = downsample_block(in_dim=128, out_dim=256, time_embedding_dim=32)
        self.downsample_block_4 = downsample_block(in_dim=256, out_dim=512, time_embedding_dim=32)

        self.convolution_block = convolution_block(in_dim=512, out_dim=1024, time_embedding_dim=32)

        self.upsample_block_4 = upsample_block(in_dim=1024, out_dim=512, time_embedding_dim=32, use_attention=use_attention)
        self.upsample_block_3 = upsample_block(in_dim=512, out_dim=256, time_embedding_dim=32, use_attention=use_attention)
        self.upsample_block_2 = upsample_block(in_dim=256, out_dim=128, time_embedding_dim=32, use_attention=use_attention)
        self.upsample_block_1 = upsample_block(in_dim=128, out_dim=64, time_embedding_dim=32, use_attention=use_attention)

        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, timestep):
        time_embeddings = self.time_embedding_block(timestep)
        residual_x_1, x = self.downsample_block_1(x, time_embeddings)
        residual_x_2, x = self.downsample_block_2(x, time_embeddings)
        residual_x_3, x = self.downsample_block_3(x, time_embeddings)
        residual_x_4, x = self.downsample_block_4(x, time_embeddings)
        x = self.convolution_block(x, time_embeddings)
        x = self.upsample_block_4(x, residual_x_4, time_embeddings)
        x = self.upsample_block_3(x, residual_x_3, time_embeddings)
        x = self.upsample_block_2(x, residual_x_2, time_embeddings)
        x = self.upsample_block_1(x, residual_x_1, time_embeddings)
        output = self.output_layer(x)
        return output