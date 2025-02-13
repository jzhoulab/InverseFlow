import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from utils import pad_dims_like


class ConsistencyModel(nn.Module):
    """Consistency models with skip connections."""

    def __init__(
        self,
        model: nn.Module,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
    ):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min

    def forward(self, x, sigma, **kwargs):
        c_skip = self.skip_scaling(sigma)
        c_out = self.output_scaling(sigma)
        c_skip = pad_dims_like(c_skip, x)
        c_out = pad_dims_like(c_out, x)

        output = self.model(x, sigma, **kwargs)

        return c_skip * x + c_out * output

    def skip_scaling(
        self,
        sigma: Tensor,
    ) -> Tensor:
        return self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)

    def output_scaling(self, sigma: Tensor) -> Tensor:
        return (self.sigma_data * (sigma - self.sigma_min)) / (
            self.sigma_data**2 + sigma**2
        ) ** 0.5


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class N2NUNet(nn.Module):
    def __init__(
        self,
        input_channel=1,
        channels=[32, 64, 128, 256, 512],
        embed_dim=256,
        embed_scale=1.0,
    ) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, channels[0], 3, stride=1, padding=1)

        self.conv = nn.ModuleList(
            [nn.Conv2d(channels[0], channels[0], 3, stride=1, padding=1)]
            + [
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=1, padding=1)
                for i in range(len(channels) - 1)
            ]
        )
        self.dense = nn.ModuleList(
            [Dense(embed_dim, channels[i]) for i in range(len(channels))]
        )
        self.gnorm = nn.ModuleList(
            [nn.GroupNorm(32, num_channels=channels[i]) for i in range(len(channels))]
        )

        self.convM = nn.Conv2d(channels[-1], channels[-1], 3, stride=1, padding=1)

        self.tconv = nn.ModuleList(
            [
                nn.Conv2d(
                    channels[i] + channels[i],
                    channels[i - 1] + channels[i - 1],
                    3,
                    stride=1,
                    padding=1,
                )
                for i in range(len(channels) - 1, 0, -1)
            ]
        )
        self.tconvb = nn.ModuleList(
            [
                nn.Conv2d(
                    channels[i - 1] + channels[i - 1],
                    channels[i - 1],
                    3,
                    stride=1,
                    padding=1,
                )
                for i in range(len(channels) - 1, 0, -1)
            ]
        )
        self.denseT = nn.ModuleList(
            [Dense(embed_dim, channels[i - 1]) for i in range(len(channels) - 1, 0, -1)]
        )
        self.gnormT = nn.ModuleList(
            [
                nn.GroupNorm(32, num_channels=channels[i - 1])
                for i in range(len(channels) - 1, 0, -1)
            ]
        )

        self.tconv_final = nn.Conv2d(
            channels[0] + channels[0], channels[0], 3, stride=1, padding=1
        )
        self.tconv_finalb = nn.Conv2d(
            channels[0], input_channel, 3, stride=1, padding=1
        )

        self.act = lambda x: x * torch.sigmoid(x)
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim, scale=embed_scale),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, t, output_mid=False):
        h = []
        embed = self.act(self.embed(t))
        x = self.conv0(x)
        x = self.act(x)
        for l, d, g in zip(self.conv, self.dense, self.gnorm):
            x = l(x)
            x += d(embed)
            x = g(x)
            x = self.act(x)
            h.append(x)
            x = F.max_pool2d(x, 2)

        x = self.act(self.convM(x))
        mid = x

        for l, tb, d, g in zip(self.tconv, self.tconvb, self.denseT, self.gnormT):
            x = F.interpolate(x, scale_factor=2)
            x = torch.cat([x, h.pop()], dim=1)
            x = self.act(l(x))
            x = tb(x)
            x += d(embed)
            x = g(x)
            x = self.act(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, h.pop()], dim=1)
        x = self.act(self.tconv_final(x))
        x = self.tconv_finalb(x)

        if output_mid == True:
            return x, mid
        else:
            return x


class sMLP(nn.Module):
    def __init__(
        self,
        input_channel=1,
        channels=[8, 32, 32, 32],
        embed_dim=32,
        embed_scale=1.0,
        act=nn.SiLU(),
    ):
        super().__init__()
        self.denses = nn.ModuleList(
            [nn.Linear(embed_dim, channels[i]) for i in range(len(channels))]
        )
        self.layers = nn.ModuleList(
            [nn.Linear(input_channel, channels[0])]
            + [
                nn.Linear(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )
        self.final = nn.Linear(channels[-1], input_channel)

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim, scale=embed_scale),
            nn.Linear(embed_dim, embed_dim),
        )
        self.act = act

    def forward(self, h, t):
        embed = self.act(self.embed(t))
        for l, d in zip(self.layers, self.denses):
            cur_h = l(h)
            cur_h += d(embed)
            h = self.act(cur_h)

        h = self.final(h)

        return h
