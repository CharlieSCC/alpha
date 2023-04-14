import torch
from torch import nn


class lstm(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,):
        super(lstm, self).__init__()
        self.gru = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

    def forward(self,
                inputs,):
        x, _ = self.gru(inputs)
        return self.predictor(x[:, -1, :].squeeze()).squeeze()