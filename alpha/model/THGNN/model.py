import torch
from torch import nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 alpha=-0.2,
                 is_bias=True,
                 is_residual=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_heads = num_heads
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * num_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, num_heads)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.is_bias = is_bias
        self.is_residual = is_residual

        if self.is_residual:
            self.residual = nn.Linear(in_features, out_features * num_heads)

        if self.is_bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features * num_heads))

    def forward(self, inputs, adj, require_weight=False):
        h = torch.mm(inputs, self.W)
        # [N,  out_features * num_heads]
        N = h.size()[0]
        input_concat = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                               2 * self.out_features,
                                                                                               self.num_heads)
        # [N, N, 2*out_features, num_heads]
        e = self.leaky_relu(torch.mul(input_concat, self.a).sum(-2).squeeze().permute(2, 0, 1, ))
        # [num_heads, N, N]
        attention = torch.mul(e, adj)
        # [num_heads, N, N]
        attention = F.softmax(attention, dim=-1)
        # [num_heads, N, N]
        output_h = torch.matmul(attention, h.view(N, self.out_features, self.num_heads).permute(2, 0, 1, )).permute(1, 0, 2).reshape(N, -1)

        if self.is_bias:
            output_h = output_h + self.bias
        if self.is_residual:
            output_h = output_h + self.residual(inputs)
        if require_weight:
            return output_h, attention
        else:
            return output_h


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


class THGNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 out_features,
                 num_heads):
        super(THGNN, self).__init__()
        self.encoding = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.upstream_GAT = GATLayer(
            in_features=hidden_size,
            out_features=out_features,
            num_heads=num_heads
        )
        self.downstream_GAT = GATLayer(
            in_features=hidden_size,
            out_features=out_features,
            num_heads=num_heads
        )
        self.nn_self = nn.Linear(hidden_size, hidden_size)
        self.nn_upstream = nn.Linear(out_features * num_heads, hidden_size)
        self.nn_downstream = nn.Linear(out_features * num_heads, hidden_size)
        self.pair_norm = PairNorm(mode='PN-SI')
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
        x_upstream, attention_upstream = self.upstream_GAT(x.squeeze(),
                                                           upstream_adj,
                                                           require_weight)
        x_downstream, attention_downstream = self.downstream_GAT(x.squeeze(),
                                                                 downstream_adj,
                                                                 require_weight)
        x = self.nn_self(x.squeeze())
        x_upstream = self.nn_upstream(x_upstream)
        x_downstream = self.nn_downstream(x_downstream)
        x = torch.stack((x, x_upstream, x_downstream), dim=1)
        x, heterogeneous_attention = self.Heterogeneous_GAT(x, require_weight)
        print(heterogeneous_attention)
        x = self.pair_norm(x)
        if require_weight:
            return self.predictor(x).squeeze(), (attention_upstream, attention_downstream, heterogeneous_attention)
        else:
            return self.predictor(x).squeeze()
