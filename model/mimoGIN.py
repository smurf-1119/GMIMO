import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import torch_geometric.transforms as T
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from typing import List, Optional, Union, Callable
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.models import MLP
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn.inits import reset


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        os.environ['PYTHONHASHSEED'] = str(42)  # 为了禁止hash随机化，使得实验可复现。
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)      # 为当前GPU设置随机种子（只用一块GPU）
        torch.cuda.manual_seed_all(42)   # 为所有GPU设置随机种子（多块GPU）

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        # if (edge_weight):
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)
        # else:
        #     out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)


    # def message(self, x_j: Tensor) -> Tensor:
    #     return x_j

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class mimoGIN(torch.nn.Module):
    def __init__(self, in_dim, num_classes, 
                 num_layers, hidden, dropout=0.5,
                 pool_type='mean',
                 use_jk=False, jk_mode='cat', num_ensemble=2):
        super(mimoGIN, self).__init__()
        num_features = in_dim
        self.conv1s = nn.ModuleList([GINConv(Sequential(
            Linear(num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ), train_eps=True) for _ in range(num_ensemble)])
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                    train_eps=True))
        self.lin1s = nn.ModuleList([Linear(hidden, hidden) for _ in range(num_ensemble)])
        self.lin2s = nn.ModuleList([Linear(hidden, num_classes) for _ in range(num_ensemble)])
        # self.lin2 = Linear(hidden, 1)
        self.dropout = dropout

        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'sum':
            self.pool = global_add_pool
        elif pool_type == 'max':
            self.pool = global_max_pool
        self.use_jk = False

    def reset_parameters(self):
        for conv in self.conv1s:
            conv.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

        for lin in self.lin1s:
            lin.reset_parameters()

        for lin in self.lin2s:
            lin.reset_parameters()

    def forward(self, datas):
        x_list = []
        edge_index_list = []
        batch_list = []
        edge_weight_list = []

        for data in datas:
            x_list.append(data.x)
            edge_index_list.append(data.edge_index)
            batch_list.append(data.batch)

            if 'edge_weights' in data:
                edge_weight_list.append(data.edge_weights.float())
            else:
                edge_weight_list.append(None)

        hs = []
        ys = []
        xss = [[] for _ in range(len(self.conv1s))]

        for i in range(len(self.conv1s)):
            x = F.relu(self.conv1s[i](x_list[i],
                        edge_index_list[i],
                        edge_weight=edge_weight_list[i]))
            xss[i].append(x)
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x,
                                edge_index_list[i],
                                edge_weight=edge_weight_list[i]))
                xs += [x]
                xss[i].append(x)

            if self.use_jk:
                x = self.jk(xs)
            h = self.pool(x, batch_list[i])

            x = F.relu(self.lin1s[i](h))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2s[i](x)

            hs.append(h), ys.append(x)

        return [edge_index_list, xss, hs], ys

    def M_mixup_forward(self, data1, data2, lambd):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        x1 = self.conv1(x1, edge_index1)
        for conv in self.convs:
            x1 = conv(x1, edge_index1)
        embed1 = self.pool(x1, batch1)

        x2 = self.conv1(x2, edge_index2)
        for conv in self.convs:
            x2 = conv(x2, edge_index2)
        embed2 = self.pool(x2, batch2)

        mixup_embed = lambd * embed1 + (1.0 - lambd) * embed2
        embed = F.relu(self.lin1(mixup_embed))
        embed = F.dropout(embed, p=self.dropout, training=self.training)
        embed = self.lin2(embed)
        # return F.log_softmax(embed, dim=-1)
        return mixup_embed, embed

    def __repr__(self):
        return self.__class__.__name__

