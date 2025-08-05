import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) regularization"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        # print(net.shape)
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding

    params:
        kernel_size: kernel size
        stride: the stride of the window. Default value is kernel_size
    
    input: (n_sample, n_channel, n_length)
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        p = max(0, self.kernel_size - 1)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class BasicBlock(nn.Module):
    """
    Inverted Bottleneck Block: 
        1x1 Conv (expand) -> kxk Conv (depthwise/grouped) -> 1x1 Conv (reduce)

    params:
        in_channels: number of input channels
        out_channels: number of output channels
        expansion_ratio: ratio to expand channels in the middle conv
        kernel_size: kernel window length
        stride: kernel step size
        groups: number of groups in convk (for depthwise, groups = middle_channels)
        downsample: whether downsample length
        drop_path_rate: dropout path rate

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, expansion_ratio, kernel_size, stride, groups, downsample, drop_path_rate=0.0):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = expansion_ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        self.drop_path_rate = drop_path_rate

        self.expanded_channels = int(self.in_channels * self.expansion_ratio)

        # 1x1 Conv (upsample/expand)
        self.conv1 = MyConv1dPadSame(
            in_channels=self.in_channels, 
            out_channels=self.expanded_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)
        self.norm1 = nn.BatchNorm1d(self.expanded_channels, eps=1e-3)
        self.activation1 = GELU()

        # kxk Conv (deepthwise/grouped)
        self.conv2 = MyConv1dPadSame(
            in_channels=self.expanded_channels, 
            out_channels=self.expanded_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride,
            groups=self.expanded_channels) # Depthwise convolution
        self.norm2 = nn.BatchNorm1d(self.expanded_channels, eps=1e-3)
        self.activation2 = GELU()

        # 1x1 Conv (downsample/reduce)
        self.conv3 = MyConv1dPadSame(
            in_channels=self.expanded_channels, 
            out_channels=self.out_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)
        self.norm3 = nn.BatchNorm1d(self.out_channels, eps=1e-3)

        # Squeeze-and-Excitation
        r = 2
        self.se_fc1 = nn.Linear(self.out_channels, self.out_channels//r)
        self.se_fc2 = nn.Linear(self.out_channels//r, self.out_channels)
        self.se_activation = GELU()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        out = x
        # 1x1 Conv (upsample/expand)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation1(out)
        
        # kxk Conv (depthwise/grouped)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)
        
        # 1x1 Conv (downsample/reduce)
        out = self.conv3(out)
        out = self.norm3(out)

        # Squeeze-and-Excitation
        se = out.mean(-1) # (n_sample, n_channel)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = torch.clamp(torch.sigmoid(se), min=1e-7, max=1-1e-7)
        out = torch.einsum('abc,ab->abc', out, se)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)       
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut with DropPath
        out = self.drop_path(out) + identity
        return out

class BasicStage(nn.Module):
    """
    Basic Stage:
        block_1 -> block_2 -> ... -> block_M
    """
    def __init__(self, in_channels, out_channels, expansion_ratio, kernel_size, stride, groups, i_stage, m_blocks, drop_path_rates, verbose=False):
        super(BasicStage, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = expansion_ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.i_stage = i_stage
        self.m_blocks = m_blocks
        self.verbose = verbose

        self.block_list = nn.ModuleList()
        for i_block in range(self.m_blocks):
            
            # downsample, stride, input
            if i_block == 0:
                self.downsample = True
                self.stride = stride
                self.tmp_in_channels = self.in_channels
            else:
                self.downsample = False
                self.stride = 1
                self.tmp_in_channels = self.out_channels
            
            # build block
            tmp_block = BasicBlock(
                in_channels=self.tmp_in_channels, 
                out_channels=self.out_channels, 
                expansion_ratio=self.expansion_ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=self.groups, 
                downsample=self.downsample, 
                drop_path_rate=drop_path_rates[i_block])
            self.block_list.append(tmp_block)

    def forward(self, x):

        out = x

        for i_block in range(self.m_blocks):
            net = self.block_list[i_block]
            out = net(out)
            if self.verbose:
                print('stage: {}, block: {}, in_channels: {}, out_channels: {}, outshape: {}'.format(self.i_stage, i_block, net.in_channels, net.out_channels, list(out.shape)))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv1.in_channels, net.conv1.out_channels, net.conv1.kernel_size, net.conv1.stride, net.conv1.groups))
                print('stage: {}, block: {}, convk: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv2.in_channels, net.conv2.out_channels, net.conv2.kernel_size, net.conv2.stride, net.conv2.groups))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv3.in_channels, net.conv3.out_channels, net.conv3.kernel_size, net.conv3.stride, net.conv3.groups))

        return out

class ECGFeatureExtractor(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    params:
        in_channels
        base_filters
        filter_list: list, filters for each stage
        m_blocks_list: list, number of blocks of each stage
        kernel_size
        stride
        groups_width
        n_stages
        n_classes

    """

    def __init__(self, in_channels, base_filters, expansion_ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes, return_features=False, verbose=False, drop_path_rate=0.0):
        super(ECGFeatureExtractor, self).__init__()
        
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.expansion_ratio = expansion_ratio
        self.filter_list = filter_list
        self.m_blocks_list = m_blocks_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_width = groups_width
        self.n_stages = len(filter_list)
        self.n_classes = n_classes
        self.return_features = return_features
        self.verbose = verbose

        # calculate drop path rates
        total_blocks = sum(m_blocks_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        dpr_idx = 0

        # first conv
        self.first_conv = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=self.base_filters, 
            kernel_size=self.kernel_size, 
            stride=2)
        self.first_norm = nn.BatchNorm1d(base_filters, eps=1e-3)
        self.first_activation = GELU()

        # stages
        self.stage_list = nn.ModuleList()
        in_channels = self.base_filters
        for i_stage in range(self.n_stages):

            out_channels = self.filter_list[i_stage]
            m_blocks = self.m_blocks_list[i_stage]
            
            stage_drop_path_rates = dpr[dpr_idx : dpr_idx + m_blocks]
            dpr_idx += m_blocks

            tmp_stage = BasicStage(
                in_channels=in_channels, 
                out_channels=out_channels, 
                expansion_ratio=self.expansion_ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=out_channels//self.groups_width, 
                i_stage=i_stage,
                m_blocks=m_blocks, 
                drop_path_rates=stage_drop_path_rates, 
                verbose=self.verbose)
            self.stage_list.append(tmp_stage)
            in_channels = out_channels

        # final prediction
        self.dense = nn.Linear(in_channels, n_classes)
        
    def forward(self, x):
        
        out = x # input shape: (n_samples, n_channels, n_length)
        
        # first conv
        out = self.first_conv(out)
        out = self.first_norm(out)
        out = self.first_activation(out)
        
        # stages
        for i_stage in range(self.n_stages):
            net = self.stage_list[i_stage]
            out = net(out)
            
        # final prediction
        deep_features = out.mean(-1)
        out = self.dense(deep_features)

        if self.return_features:
            return out, deep_features
        else:
            return out

if __name__ == "__main__":
  model = ECGFeatureExtractor(
      in_channels=12, 
      base_filters=96, # ConvNeXt base filters
      expansion_ratio=4, # Inverted bottleneck expansion ratio
      filter_list=[96,192,384,768],    # ConvNeXt channels
      m_blocks_list=[3,3,9,3],   # ConvNeXt depths
      kernel_size=16, 
      stride=2, 
      groups_width=1, # This parameter is not directly used for groups in BasicBlock due to depthwise conv
      verbose=True, 
      drop_path_rate=0.1, # Example drop path rate
      n_classes=150)
  print(model)
