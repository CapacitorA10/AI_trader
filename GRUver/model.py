import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers  = num_layers
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.seq_length  = seq_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 64)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        h_0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to('cpu')
        #output, (hn) = self.gru(x, (h_0))
        #hn = hn.view(-1, self.hidden_size)
        #out = self.tanh(hn)

        #new
        out, (h_n) = self.gru(x, (h_0))
        #print(out.shape)
        out = out[:,-1,:] # LAST OUTPUT 추출
        #out = self.tanh(out.view(-1, self.hidden_size))
        out = out.view(-1, self.hidden_size)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
