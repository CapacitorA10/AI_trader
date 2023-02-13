import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, input_features, hidden_features, kernel_size, dilation_rate, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(input_features, hidden_features, kernel_size,
                               dilation=dilation_rate,
                               padding=(kernel_size-1)*dilation_rate,
                               bias=False))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(hidden_features, hidden_features, kernel_size,
                               dilation=dilation_rate,
                               padding=(kernel_size-1)*dilation_rate,
                               bias=False))
        self.dropout = nn.Dropout(dropout)
        if input_features != hidden_features:
            self.residual_conv = nn.Conv1d(input_features, hidden_features, 1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = x[:, :, :-(self.conv1.dilation[0] * (self.conv1.kernel_size[0] - 1))]
        x = self.dropout(torch.relu(x))
        x = self.conv2(x)
        x = x[:, :, :-(self.conv2.dilation[0] * (self.conv2.kernel_size[0] - 1))]
        x = self.dropout(torch.relu(x))
        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(residual)
        x = x + residual
        return x

class TCN(nn.Module):
    def __init__(self, input_features, hidden_features, kernel_size, dilation_rates, num_layers, dropout=0.2):
        super(TCN, self).__init__()
        self.dilation_rates = dilation_rates
        self.num_layers = num_layers
        self.residual_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.residual_blocks.append(ResidualBlock(input_features if i == 0 else hidden_features,
                                                      hidden_features,
                                                      kernel_size,
                                                      dilation_rates[i],
                                                      dropout))
        self.fc = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for i, residual_block in enumerate(self.residual_blocks):
            x = residual_block(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x[:, -1, :])
        return x

class TCN_vanilla(nn.Module):
    def __init__(self, input_features, hidden_features, kernel_size, dilation_rates, num_layers, dropout=0.2):
        super(TCN_vanilla, self).__init__()
        self.dilation_rates = dilation_rates
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(nn.Conv1d(input_features, hidden_features, kernel_size,
                                   dilation=dilation_rates[0],
                                   padding=(kernel_size-1)*dilation_rates[0],
                                   bias=False))
        for i in range(1, self.num_layers):
            self.convs.append(nn.Conv1d(hidden_features, hidden_features, kernel_size,
                                        dilation=dilation_rates[i],
                                        padding=(kernel_size-1)*dilation_rates[i],
                                        bias=False))
        self.fc = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = self.dropout(torch.relu(x[:, :, :-(conv.dilation[0] * (conv.kernel_size[0] - 1))]))
        x = x.permute(0, 2, 1)
        x = self.fc(x[:, -1, :])
        return x