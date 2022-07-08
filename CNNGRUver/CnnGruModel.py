import torch

## MODEL DEFINE
class CNNGRU(torch.nn.Module):

    def __init__(self, num_classes, input_size, gru_out_size, num_layers, seq_length):
        super(CNNGRU, self).__init__()
        self.num_classes = num_classes  # output Class 수
        self.num_layers = num_layers  # GRU layer 수
        self.input_size = input_size  # input feature class 수
        self.hidden_size = gru_out_size
        self.seq_length = seq_length
        cnn_feature = 64

        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv1d(2, cnn_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(cnn_feature, cnn_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            # torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        self.cnn2 = torch.nn.Sequential(
            torch.nn.Conv1d(2, cnn_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(cnn_feature, cnn_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            # torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        self.cnn3 = torch.nn.Sequential(
            torch.nn.Conv1d(2, cnn_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(cnn_feature, cnn_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            # torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        comb_feature = cnn_feature * (input_size // 2)
        self.cnn_comb = torch.nn.Sequential(
            torch.nn.Conv1d(comb_feature, comb_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(comb_feature),
            torch.nn.ReLU(),
            torch.nn.Conv1d(comb_feature, comb_feature, kernel_size=3, padding=1, bias=True),
            torch.nn.BatchNorm1d(comb_feature),
            torch.nn.ReLU()
        )
        self.gru = torch.nn.Sequential(
            torch.nn.GRU(input_size=comb_feature, hidden_size=gru_out_size, num_layers=num_layers, batch_first=True),
            torch.nn.Tanh()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(gru_out_size, 32, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes, bias=True)
        )

    def forward(self, input):
        feat_size = input.shape[-1]
        c1 = self.cnn1(input[:,:, 0:feat_size:feat_size>>1])
        c2 = self.cnn2(input[:,:, 1:feat_size:feat_size>>1])
        c3 = self.cnn3(input[:,:, 2:feat_size:feat_size>>1])
        out = torch.cat((c1, c2, c3), 1)
        out = self.cnn_comb(out)
        out = self.gru(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.fc(out)
        return out
