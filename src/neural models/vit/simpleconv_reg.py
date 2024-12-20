# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import random
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

from common_reg import (
    ConvSequence, ScaledEmbedding, SubjectLayers,
    DualPathRNN, ChannelMerger, ChannelDropout, pad_multiple, ConvBlock
)

class TemporalAggregation(nn.Module):
    def __init__(self, input_time_length = 281):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_time_length))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: (B, C, T)
        return torch.sum(x * self.weight.unsqueeze(0).unsqueeze(0), dim=2) + self.bias

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, extra_dropout = 0.0):
        super().__init__()
        
        layers = []
        layers.extend([
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            nn.Dropout(p=extra_dropout)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SimpleConv(nn.Module):
    def __init__(self,
                 # Channels
                 in_channels: int,
                 out_channels: int = 2048,
                 hidden_channels: int = 512,
                 initial_linear= True,
                 # Overall structure
                 depth: int = 2,
                 dilation_growth: int = 2,
                 dilation_type: str = "expo",
                 gelu: bool = True,
                 # Subject specific settings
                 n_subjects: int = 4,
                 subject_layers: bool = True,
                 subject_layers_dim: str = "input",  # or "hidden"
                 subject_layers_id: bool = False,
                 # Attention multi-dataset support
                 merger: bool = True,
                 merger_pos_dim: int = 256,
                 merger_dropout: float = 0.2,
                 dropout_rescale: bool = True,
                 merger_penalty: float = 0.0,
                 merger_per_subject: bool = False,
                 # Projection heads
                 target_dim: int = 768,
                 device: str = "cpu",
                 temporal_dim: int = 181,
                # Extra dropout for projection head and conv blocks, not in original model
                 extra_dropout: float = 0.0,
                 stop_at_attention: bool = False,
                 ):
        super().__init__()

        # Set activation function
        self.stop_at_attention = stop_at_attention
        activation = nn.GELU
        self.device = device
        # Initialize the ChannelMerger (attention layer) if merger is True
        self.merger = None
        if merger_dropout > 0.:
            self.dropout = ChannelDropout(merger_dropout, dropout_rescale)
        if merger:
            self.merger = ChannelMerger(
                chout=in_channels-2, 
                pos_dim=merger_pos_dim,
                dropout=merger_dropout,
                usage_penalty=merger_penalty,
                n_subjects=n_subjects,
                per_subject=merger_per_subject
            )

        if not stop_at_attention:
            print("Not stopping model at Spatial Attention")
            in_channels = in_channels - 2 # 270
            self.initial_linear = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        

            # Initialize the SubjectLayers if subject_layers is True
            self.subject_layers = None
            if subject_layers:
                dim = hidden_channels if subject_layers_dim == "hidden" else in_channels
                self.subject_layers = SubjectLayers(in_channels, dim, n_subjects, subject_layers_id)
                in_channels = dim  # Update in_channels after subject_layers

            # Build convolutional blocks
            self.conv_blocks = nn.ModuleList()
            # dilation = 1
            for i in range(depth):
                print(f"ConvBlock {i}: passing k = {i+1}")
                k = i+1
                block = ConvBlock(in_channels, hidden_channels, k, dilation_type, extra_dropout)
                self.conv_blocks.append(block)
                self.add_module(f'conv_block_{i}', block)
                in_channels = hidden_channels  # Update in_channels for next block
                # dilation *= dilation_growth
                # print(f"Dilation {i+1}", dilation)

            self.final = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=1),
                nn.Conv1d(hidden_channels*2, out_channels, kernel_size=1),
            )

            self.temporal_aggregation = TemporalAggregation(input_time_length = temporal_dim)

            self.clip_head = ProjectionHead(out_channels, target_dim, extra_dropout)
            # self.mse_head = ProjectionHead(out_channels, target_dim)
        print("Stopping model at Spatial Attention")

    def forward(self, batch):
        meg = batch['meg'].to(self.device)
        pos = batch['ch_pos'].to(self.device)
        # Apply merger if exists
        if self.dropout is not None:
            x = self.dropout(meg, batch, pos)
        else:
            x = meg
        if self.merger is not None:
            x = self.merger(x, batch, pos)
        else:
            x = meg
        if self.stop_at_attention:
            return x
        x = self.initial_linear(x)
        # Apply subject_layers if exists
        if self.subject_layers is not None:
            subjects = batch['subject_index'].to(self.device)
            x = self.subject_layers(x, subjects)

        # Pass through convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Apply final projection
        x = self.final(x)

        # Temporal aggregation
        x = self.temporal_aggregation(x)  # Now x is (B, out_channels)

        # Projection heads
        clip_output = self.clip_head(x)
        # mse_output = self.mse_head(x)
        # print("Clip Output Shape", clip_output.shape)
        return clip_output