import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Just two layer MLP
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.d_model
        self.encoder = nn.Linear(self.seq_len, self.hidden_dim)
        self.norm = torch.nn.LayerNorm(self.hidden_dim)
        self.act_fn=nn.ReLU()
        self.decoder = nn.Linear(self.hidden_dim, self.pred_len)

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1)
        x = self.encoder(x)
        x=self.norm(x)
        x = self.act_fn(x)
        x = self.decoder(x)
        x = x.permute(0,2,1)
        return x # [Batch, Output length, Channel]