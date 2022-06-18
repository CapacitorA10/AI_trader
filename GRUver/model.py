import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, num_classes, input_size, gru_out_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers  = num_layers
        self.input_size  = input_size
        self.hidden_size = gru_out_size
        self.seq_length  = seq_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=gru_out_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(gru_out_size, 64)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h_0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to('cuda')
        output, (hn) = self.gru(x, (h_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.tanh(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
