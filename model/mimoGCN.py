import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from typing import List, Optional, Union
from torch import Tensor
from torch_scatter import scatter

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

class mimoGCN(torch.nn.Module):
    def __init__(self, in_dim, num_classes, num_layers, hidden, dropout=0.5, pool_type='mean', use_jk=False, jk_mode='cat'):
        super(mimoGCN, self).__init__()
        num_features = in_dim
        self.conv1s = nn.ModuleList([GCNConv(num_features, hidden), GCNConv(num_features, hidden)])
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        
        self.use_jk = use_jk
        if use_jk:
            self.jk = JumpingKnowledge(jk_mode)

        lin_in_dim = num_layers * hidden if use_jk and jk_mode == 'cat' else hidden
        self.lin1s = nn.ModuleList([Linear(lin_in_dim, hidden), Linear(lin_in_dim, hidden)])
        self.lin2s = nn.ModuleList([Linear(hidden, num_classes), Linear(hidden, num_classes)])
        self.dropout = dropout

        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'sum':
            self.pool = global_add_pool
        elif pool_type == 'max':
            self.pool = global_max_pool

    def reset_parameters(self):
        self.conv1s[0].reset_parameters()
        self.conv1s[1].reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1s[0].reset_parameters()
        self.lin2s[0].reset_parameters()
        self.lin1s[1].reset_parameters()
        self.lin2s[1].reset_parameters()
        if self.use_jk:
            self.jk.reset_parameters()
    
    def forward(self, data, data2):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        if 'edge_weights' in data:
            edge_weight = data.edge_weights.float()
            edge_weight2 = data2.edge_weights.float()
        else:
            edge_weight = None
            edge_weight2 = None

        x_list = [x, x2]
        edge_index_list = [edge_index, edge_index2]
        batch_list = [batch, batch2]
        edge_weight_list = [edge_weight, edge_weight2]
        hs = []
        ys = []
        xss = [[],[]]
        edge_indexs = [edge_index, edge_index2]

        for i in range(2):
            x = F.relu(self.conv1s[i](x_list[i], edge_index_list[i], edge_weight = edge_weight_list[i]))
            xss[i].append(x)
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index_list[i], edge_weight = edge_weight_list[i]))
                xs += [x]
                xss[i].append(x)

            if self.use_jk:
                x = self.jk(xs)
            h = self.pool(x, batch_list[i])

            x = F.relu(self.lin1s[i](h))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2s[i](x)

            hs.append(h), ys.append(x)

        return [edge_indexs, xss, hs], ys
        # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
  