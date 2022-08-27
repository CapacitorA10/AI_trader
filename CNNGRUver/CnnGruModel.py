import torch

## MODEL DEFINE
class CNNGRU(torch.nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(CNNGRU, self).__init__()
        self.num_classes = num_classes  # output Class 수
        self.num_layers = num_layers  # GRU layer 수
        self.input_size = input_size  # input feature class 수
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        cnn_feature = 64

        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv1d(2, cnn_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(cnn_feature, cnn_feature, kernel_size=3, padding=1, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            # torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        self.cnn2 = torch.nn.Sequential(
            torch.nn.Conv1d(2, cnn_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(cnn_feature, cnn_feature, kernel_size=3, padding=1, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            # torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        self.cnn3 = torch.nn.Sequential(
            torch.nn.Conv1d(2, cnn_feature, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(cnn_feature),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(cnn_feature, cnn_feature, kernel_size=3, padding=1, bias=True),
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

        class extract_tensor(torch.nn.Module):
            def forward(self, x):
                # Output shape (batch, features, hidden)
                tensor, _ = x
                # Reshape shape (batch, hidden)
                return tensor[:, -1, :]
        self.gru = torch.nn.Sequential(
            torch.nn.GRU(input_size=comb_feature, hidden_size=hidden_size,
                         num_layers=num_layers, batch_first=True),
            extract_tensor()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 32, bias=True),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes, bias=True)
        )

    def forward(self, input):
        feat_size = input.shape[-2]
        print(f"input: {input.shape}")
        print(f"cnn input: {input[:, 0::feat_size>>1, :].shape}")
        c1 = self.cnn1(input[:, 0::feat_size>>1, :])
        c2 = self.cnn2(input[:, 1::feat_size>>1, :])
        c3 = self.cnn3(input[:, 2::feat_size>>1, :])
        print(f"c1: {c1.shape}, c2: {c2.shape}, c3: {c3.shape} ")
        out = torch.cat((c1, c2, c3), 1)
        print(f"cat: {out.shape}")
        out = self.cnn_comb(out).transpose(1,2)
        print(f"comb: {out.shape}")
        h_0 = (torch.zeros(self.num_layers, out.size(0), self.hidden_size)).to('cpu')
        out = self.gru(out, (h_0))
        print(f"gru out: {out.shape}")
        out = out.view(out.size(0), -1)  # Flatten them for FC
        print(f"gru view: {out.shape}")
        out = self.fc(out)
        print(f"output: {out.shape}")
        return out
