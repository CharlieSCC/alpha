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


class lstm_gcn(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 out_features):
        super(lstm_gcn, self).__init__()
        self.encoding = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.gcn = GraphConvolution(
            in_features=hidden_size,
            out_features=out_features,
        )
        self.pair_norm = PairNorm(mode='PN-SCS')
        self.predictor = nn.Sequential(
            nn.Linear(out_features, 1),
        )

    def forward(self,
                inputs,
                adj_matrix,):
        x, _ = self.encoding(inputs)
        x = self.gcn(x[:, -1, :].squeeze(), adj_matrix)
        #x = self.pair_norm(x)
        return self.predictor(x.squeeze()).squeeze()
