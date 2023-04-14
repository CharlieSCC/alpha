import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class PairNorm(nn.Module):
    def __init__(self,
                 mode,
                 scale=1):
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        elif self.mode == 'PN':
            x = x - x.mean(dim=0)
            x = self.scale * x / (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        elif self.mode == 'PN-SI':
            x = x - x.mean(dim=0)
            x = self.scale * x / (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
        elif self.mode == 'PN-SCS':
            x = self.scale * x / (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt() - x.mean(dim=0)
        else:
            raise "No such a mode!"
        return x


class HeterogeneousGATLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,):
        super(HeterogeneousGATLayer, self).__init__()
        self.w = nn.Sequential(nn.Linear(input_size, hidden_size),
                               nn.Tanh(),
                               nn.Linear(hidden_size, 1, bias=False))

    def forward(self,
                inputs,
                require_weight=False):
        attention = self.w(inputs)
        attention = torch.softmax(attention, dim=1)
        if require_weight:
            return torch.mul(attention, inputs).sum(dim=1), attention.squeeze()
        else:
            return torch.mul(attention, inputs).sum(dim=1)


class hats(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 out_features,):
        super(hats, self).__init__()
        self.encoding = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.upstream_gcn = GraphConvolution(
            in_features=hidden_size,
            out_features=out_features,
        )
        self.downstream_gcn = GraphConvolution(
            in_features=hidden_size,
            out_features=out_features,
        )
        self.nn_self = nn.Linear(hidden_size, hidden_size)
        self.nn_upstream = nn.Linear(out_features, hidden_size)
        self.nn_downstream = nn.Linear(out_features, hidden_size)

        self.pair_norm = PairNorm(mode='PN')
        self.Heterogeneous_GAT = HeterogeneousGATLayer(hidden_size,
                                                       hidden_size)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

    def forward(self,
                inputs,
                upstream_adj,
                downstream_adj,
                require_weight):
        _, x = self.encoding(inputs)
        x_upstream = self.upstream_gcn(x.squeeze(), upstream_adj,)
        x_downstream = self.downstream_gcn(x.squeeze(), downstream_adj,)
        x = self.nn_self(x.squeeze())
        x_upstream = self.nn_upstream(x_upstream)
        x_downstream = self.nn_downstream(x_downstream)
        x = torch.stack((x, x_upstream, x_downstream), dim=1)
        x, heterogeneous_attention = self.Heterogeneous_GAT(x, require_weight)
        x = self.pair_norm(x)
        if require_weight:
            return self.predictor(x).squeeze(), heterogeneous_attention
        else:
            return self.predictor(x).squeeze()
