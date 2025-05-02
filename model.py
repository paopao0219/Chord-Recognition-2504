import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, use_gpu, bidirectional, dropout=(0.4, 0.0, 0.0)):
        super().__init__()
        self.use_gpu = use_gpu
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dropout1 = nn.Dropout(p=dropout[0])
        self.lstm = nn.LSTM(
            input_size, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout[1]
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.num_directions)
        self.dropout2 = nn.Dropout(p=dropout[2])
        self.hidden2out = nn.Linear(hidden_dim * self.num_directions, output_size)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, dtype=torch.float32)
        c = torch.zeros_like(h)
        if self.use_gpu:
            h, c = h.cuda(), c.cuda()
        return (h, c)

    def forward(self, batch, lengths=None):
        hidden = self.init_hidden(batch.size(0))
        batch = self.dropout1(batch)
        if lengths is not None:
            lengths = lengths.cpu()
            batch = pack_padded_sequence(batch, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(batch, hidden)
        if lengths is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.bn1(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.dropout2(output)
        return self.hidden2out(output)
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, use_gpu, bidirectional,
                 dropout=(0.4, 0.0, 0.0)):
        super(GRUClassifier, self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.dropout1 = nn.Dropout(p=dropout[0])
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=self.num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout[1])
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.num_directions)
        self.hidden2out = nn.Linear(hidden_dim * self.num_directions, output_size)
        self.dropout2 = nn.Dropout(p=dropout[2])

    def disable_dropout(self):
        self.gru.dropout = .0
        self.dropout1.p = .0
        self.dropout2.p = .0

    def init_hidden(self, batch_size):
        if self.use_gpu:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                            dtype=torch.float32).cuda())
        else:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, dtype=torch.float32))

    def forward(self, batch, lengths=None):
        self.hidden = self.init_hidden(batch.size(0))
        batch = self.dropout1(batch)
        # pack sequence if lengths available(during training)
        if lengths is not None:
            lengths = lengths.cpu() 
            batch = pack_padded_sequence(batch, lengths, batch_first=True, enforce_sorted=False)
        output, self.hidden = self.gru(batch, self.hidden)
        if lengths is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.bn1(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.dropout2(output)
        output = self.hidden2out(output)
        return output
