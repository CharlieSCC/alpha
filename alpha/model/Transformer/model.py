import torch
from torch import nn


class transformer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_heads,):
        super(transformer, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

    def forward(self,
                inputs,):
        x = self.linear(inputs)
        x = self.encoder_layer(x)
        return self.predictor(x[:, -1, :].squeeze()).squeeze()
