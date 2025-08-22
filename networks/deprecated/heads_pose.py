
# %%
import torch
import torch.nn as nn
import transformers

import matplotlib.pyplot as plt

from utilities import *
import projections as proj
from rich import print
import inspect
import copy
import sys

import importlib
import networks.combiners_readers_pose

importlib.reload(networks.combiners_readers_pose)
from networks.combiners_readers_pose import *


class RegressionHead_FC(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        sequence_length: int = 256,
        output_dim: int = 3,
        dropout: float = 0.0,
    ):
        super(RegressionHead_FC, self).__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.flat = nn.Flatten()
        self.regressor = nn.Linear(
            self.embed_dim * self.sequence_length, self.output_dim * 10
        )
        self.preout1 = nn.Linear(self.output_dim * 10, self.output_dim * 10)
        self.dropout = nn.Dropout(p=dropout)
        self.gelu = nn.GELU()
        self.output = nn.Linear(self.output_dim * 10, self.output_dim)
        self.tanh = nn.Tanh()

    def forward(self, combined):
        y = self.regressor(self.flat(combined))
        y = self.tanh(y)
        y = self.dropout(y)
        y = self.preout1(y)
        y = self.output(y)
        return y


class RegressionHead_MLP(nn.Module):

    def __init__(
        self,
        embed_dim: int = 384,
        sequence_length: int = 256,
        output_dim: int = 3,
        dropout: float = 0.0,
    ):
        super(RegressionHead_MLP, self).__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.flat = nn.Sequential(
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(self.embed_dim * sequence_length, 256)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(256, 64)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(64, 64)
        self.act3 = nn.GELU()
        self.dropout3 = nn.Dropout(p=dropout)

        self.output = nn.Linear(64, self.output_dim)
        self.tanh = nn.Tanh()

    def forward(self, combined):
        y = self.flat(combined)
        y = self.dropout1(y)
        y = self.fc1(y)
        y = self.tanh(y)
        y = self.dropout2(y)
        y = self.fc2(y)
        y = self.tanh(y)
        y = self.dropout3(y)
        y = self.fc3(y)
        y = self.output(y)
        return y


class HomogeneousHead(nn.Module):
    def __init__(
        self, embed_dim: int = 384, sequence_length: int = 196, dropout: float = 0.0
    ):
        super(HomogeneousHead, self).__init__()

    def forward(self, combined):
        pass


class DuplicateHead(nn.Module):
    def __init__(
        self, embed_dim: int = 384, sequence_length: int = 196, dropout: float = 0.0
    ):
        super(DuplicateHead, self).__init__()

    def forward(self, combined):
        return combined


class TwoHeads(nn.Module):
    def __init__(self, head):
        super(TwoHeads, self).__init__()

        assert len(head) == 2
        self.pos = head[0]
        self.rot = head[1]

    def forward(self, x):
        return torch.cat([self.pos(x), self.rot(x)], dim=1)
