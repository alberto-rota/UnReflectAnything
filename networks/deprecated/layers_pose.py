
import torch
import torch.nn as nn
import transformers

import matplotlib.pyplot as plt

from utilities import *
import projections as proj
from rich import print
import inspect

import inspect
import copy
import sys

import importlib


class Reassemble(nn.Module):
    def __init__(self):
        super(Reassemble, self).__init__()

    def forward(self, tokens):
        spatialdim = int(math.sqrt(tokens.shape[-2]))
        return tokens.reshape(-1, tokens.shape[-1], spatialdim, spatialdim)


class Disassemble(nn.Module):
    def __init__(self):
        super(Disassemble, self).__init__()

    def forward(self, tokens):
        spatialdim = int(math.sqrt(tokens.shape[-2]))
        return tokens.view(-1, tokens.shape[-1] ** 2, tokens.shape[-2])


class ProjectReader(nn.Module):
    def __init__(self, sequence_length: int):
        super(ProjectReader, self).__init__()
        self.projector = nn.Linear(sequence_length + 1, sequence_length)

    def forward(self, source, return_cls=False):
        projectedsource = self.projector(source)
        if return_cls:
            return source[..., 0], projectedsource
        return projectedsource


class IgnoreReader(nn.Module):
    def __init__(self):
        super(IgnoreReader, self).__init__()

    def forward(self, source, return_cls=False):
        if return_cls:
            return source[..., 0], source[..., 1:]
        return source[..., 1:]


class Hidden_State_Selector(nn.Module):
    def __init__(
        self, selected_hidden_states: list, sequence_length: int = 256, reader="project"
    ):
        super(Hidden_State_Selector, self).__init__()
        self.selected_hidden_states = selected_hidden_states
        readers = {
            "project": ProjectReader(sequence_length),
            "ignore": IgnoreReader(),
            "noread": nn.Identity(),
        }
        self.reader = readers[reader]

    def forward(self, source: list):
        return [
            self.reader(source[i].permute(0, 2, 1)).permute(0, 2, 1)
            for i in self.selected_hidden_states
        ]


class CrossMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        attention_dim: int = 256,
        sequence_length: int = 256,
        layer_norm: bool = True,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super(CrossMultiHeadAttention, self).__init__()

        self.mha_ln = nn.ModuleDict(
            {"True": nn.LayerNorm(attention_dim), "False": nn.Identity()}
        )
        self.mha_ln = self.mha_ln[str(layer_norm)]
        self.linear_ln = nn.ModuleDict(
            {"True": nn.LayerNorm(attention_dim), "False": nn.Identity()}
        )
        self.linear_ln = self.linear_ln[str(layer_norm)]
        self.atn_projetor_source = nn.Linear(embed_dim, attention_dim)
        self.atn_projetor_target = nn.Linear(embed_dim, attention_dim)
        self.gelu = nn.GELU()
        self.dropout_source = nn.Dropout(p=dropout)
        self.dropout_target = nn.Dropout(p=dropout)
        self.linear1_source = nn.Linear(attention_dim, attention_dim)
        self.linear1_target = nn.Linear(attention_dim, attention_dim)
        self.linear2_source = nn.Linear(attention_dim, attention_dim)
        self.linear2_target = nn.Linear(attention_dim, attention_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_projector = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(attention_dim, attention_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, source, target):

        source = self.atn_projetor_source(source)
        source = self.linear2_source(self.gelu(self.linear1_source(source)))
        source = self.dropout_source(source)
        target = self.atn_projetor_target(target)
        target = self.linear2_target(self.gelu(self.linear1_target(target)))
        target = self.dropout_target(target)

        mha_out = self.mha(source, target, target, need_weights=False)[0]

        mha_out = self.mha_ln(mha_out + source)
        mha_proj = self.out_projector(mha_out)
        mha_out = self.linear_ln(mha_proj + mha_out)
        return mha_out


class ResConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ResConv2d, self).__init__()
        self.groups = 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.groups,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.groups,
        )
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(num_features=in_channels)
        self.batchnorm2 = nn.BatchNorm2d(num_features=in_channels)
        self.channel_reducer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            groups=self.groups,
        )

    def forward(self, x):
        residual = x
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = x + residual
        x = self.channel_reducer(x)
        return x


class GlobalAveragePooling2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling2d, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.pool(x)
