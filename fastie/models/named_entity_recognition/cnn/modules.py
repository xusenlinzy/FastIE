import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim = dim
        self.eps = 1e-6

    def forward(self, x):
        mu = x.mean(dim=self.dim, keepdim=True)
        s = (x - mu).pow(2).mean(dim=self.dim, keepdim=True)
        x = (x - mu) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1):
        super(MaskedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            groups=groups
        )

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        return self.conv2d(x)


class MaskedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, depth=3):
        super(MaskedCNN, self).__init__()

        layers = []
        for _ in range(depth):
            layers.extend([
                MaskedConv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2),
                LayerNorm((1, in_channels, 1, 1), dim=1),
                nn.GELU(),
            ])
        layers.append(MaskedConv2d(in_channels, out_channels, kernel_size=3, padding=3//2))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x, mask)
            else:
                x = layer(x)
        return _x


class MultiHeadBiaffine(nn.Module):
    def __init__(self, hidden_size, out_size=None, num_heads=4):
        super(MultiHeadBiaffine, self).__init__()
        assert hidden_size % num_heads == 0
        in_head_dim = hidden_size // num_heads

        out_size = hidden_size if out_size is None else out_size
        assert out_size % num_heads == 0
        out_head_dim = out_size // num_heads

        self.num_heads = num_heads
        self.W = nn.Parameter(torch.randn(self.num_heads, out_head_dim, in_head_dim, in_head_dim))
        nn.init.xavier_normal_(self.W.data)
        self.out_size = out_size

    def forward(self, h, v):
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.num_heads, -1)
        v = v.reshape(bsz, max_len, self.num_heads, -1)
        w = torch.einsum('blhx, hdxy, bkhy -> bhdlk', h, self.W, v)
        return w.reshape(bsz, self.out_size, max_len, max_len)
