from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CrossAttention(nn.Module):
    def __init__(self, dim=640, num_heads=8, largest_num=500, qkv_bias=False, scale=1, bias=0):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE: scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale
        self.bias = bias
        self.largest_num = torch.Tensor([largest_num]).cuda()
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.memory_bank = None

    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.memory_bank == None or self.memory_bank.size(0) > self.largest_num:
            # self.memory_bank = x.detach()
            return x
        N, C = x.size()
        kv_input = torch.cat([self.memory_bank, x], dim=0)
        q = self.wq(x) # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(kv_input)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(kv_input)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale + self.bias  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v) + x  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # self.memory_bank = torch.cat([self.memory_bank, x.detach()], dim=0)
        return x

class CrossAttentionV2(nn.Module):
    def __init__(self, dim=640, num_heads=8, qkv_bias=False, scale=1, bias=0):
        super(CrossAttentionV2, self).__init__(dim, num_heads, qkv_bias, scale, bias)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE: scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale
        self.bias = bias
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.memory_bank = None

    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.memory_bank == None:
            return x
        N, C = x.size()
        kv_input = torch.cat([self.memory_bank, x], dim=0)
        q = self.wq(x) # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(kv_input)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(kv_input)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale + self.bias # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v) # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        # self.memory_bank = torch.cat([self.memory_bank, x.detach()], dim=0)
        return x