import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, ipt_dim, opt_dim,look_back_window, pre_rnn_dim = 128, hid_dim=128, encoder_rnn_layers=1,dropout=0.0):
        super(Encoder, self).__init__()
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.look_back_window = look_back_window
        self.pre_rnn_dim = pre_rnn_dim
        self.opt_dim = opt_dim
        self.encoder_rnn_layers = encoder_rnn_layers
        self.dropout = dropout


        self.pre_rnn = nn.Sequential(
            nn.Linear(self.ipt_dim, self.pre_rnn_dim)
        )
        self.bn = nn.BatchNorm2d(self.look_back_window)
        self.relu = nn.LeakyReLU()
        
        self.rnn = nn.GRU(self.pre_rnn_dim, self.hid_dim, self.encoder_rnn_layers,dropout=self.dropout, batch_first = True)
        
        self.post_rnn = nn.Sequential(
            nn.Linear(self.hid_dim, self.opt_dim))

    def forward(self, x):
        x = self.pre_rnn(x).unsqueeze(3)
        x = self.relu(self.bn(x).squeeze(3))
        output, hn= self.rnn(x)
        out = self.post_rnn(output)
        return {
            'past_ret': out, 
            'gru_hidden': hn, 
            "gru_output": output
        }

class Decoder(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128, decoder_rnn_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.decoder_rnn_layers = decoder_rnn_layers
        self.dropout = dropout
        
        self.rnn = nn.GRU(self.hid_dim, self.hid_dim, self.decoder_rnn_layers, dropout=self.dropout, batch_first = True)
        
        self.post_rnn = nn.Sequential(
            nn.Linear(self.hid_dim, int(self.hid_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.hid_dim/2), self.opt_dim)
        )

    def forward(self, x, hidden):
        x, hn= self.rnn(x, hidden)
        out = self.post_rnn(x)
        return {
            'future_price': out,
            'decoder_gru_hidden': hn,
            'decoder_gru_output': x
        }

class Classifier(nn.Module):
    def __init__(self, ipt_dim, opt_dim=3, hid_dim=32):
        super(Classifier, self).__init__()
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.fc1 = nn.Linear(self.ipt_dim, self.hid_dim)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(self.hid_dim)

        self.fc2 = nn.Linear(self.hid_dim, int(self.hid_dim/2))
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(int(self.hid_dim/2))

        self.fc3 = nn.Linear(int(self.hid_dim/2), self.opt_dim)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1((x))))
        x = self.relu2(self.bn2(self.fc2((x))))
        output = self.fc3(x)
        return {
            'output': output
        }

class StockNet(nn.Module):
    def __init__(self, encoder, decoder, classifier, future_window, is_gpu=True):
        super(StockNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.future_window = future_window
        self.is_gpu = is_gpu
        # if(pre_encoder_path != ''):
        #     self.encoder.load_state_dict(torch.load(pre_encoder_path, map_location=torch.device(device)))
        #     print("encoder model loaded")

    def forward(self, data):
        future_seq_len = self.future_window
        batch_size = data.shape[0]

        #outputs: future price
        outputs = torch.zeros(batch_size, future_seq_len)
        if self.is_gpu:
            outputs = outputs.cuda()


        encoder_pred = self.encoder(data) #process encoder
        pred_past_price = encoder_pred['past_ret'].squeeze(2) #[batch_size, past_future_len] 
        past_hidden = encoder_pred['gru_hidden'] # [num_layers, batch_size, hidden_dim] 
        past_output_sequence = encoder_pred['gru_output'] #[batch_size, past_future_len, hidden_dim] 
        
        output = past_output_sequence[:, -1, :]
        output = output.unsqueeze(1)

        classifier_input = past_output_sequence[:, -1, :]

        for t in range(0, future_seq_len):
            if t == 0:
                prediction = self.decoder(output, past_hidden)
            else:
                prediction = self.decoder(output, hidden)
            outputs[:, t] = prediction['future_price'].squeeze(1).squeeze(1)
            hidden = prediction['decoder_gru_hidden']
            output = prediction['decoder_gru_output']

        classifier_output = self.classifier(classifier_input)['output']
        return {
            'past_ret': pred_past_price,
            'future_ret': outputs.float(),
            'pred_cl': classifier_output
        }