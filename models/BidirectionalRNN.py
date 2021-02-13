import torch
from torch import nn
from torch.autograd import Variable

class LayerNormLSTM(nn.Module):
    def __init__(self, num_features, num_nodes, num_layers, dropout, layer_norm):
        super(LayerNormLSTM, self).__init__()

        if layer_norm:
            self.lstms = nn.ModuleList([nn.LSTM(num_features if i==0 else num_nodes*2, num_nodes, num_layers=1, bidirectional=True) for i in range(num_layers)])
            self.layer_norms = nn.ModuleList([nn.LayerNorm(num_nodes*2) for _ in range(num_layers)])
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers-1)] + [None])
        else:
            self.lstms = nn.ModuleList([nn.LSTM(num_features, num_nodes, num_layers=num_layers, dropout=dropout, bidirectional=True)])
            self.layer_norms = [None]
            self.dropouts = [None]

    def forward(self, h):
        for lstm, layer_norm, dropout in zip(self.lstms, self.layer_norms, self.dropouts):
            h, hs = lstm(h)
            if layer_norm is not None: h = layer_norm(h)
            if dropout is not None: h = dropout(h)

        return h, hs

class BidirectionalRNN(nn.Module):
    def __init__(self, num_features, num_nodes, num_layers, num_classes, dropout, layer_norm):
        super(BidirectionalRNN, self).__init__()

        self.lstm = LayerNormLSTM(num_features, num_nodes, num_layers, dropout, layer_norm)
        self.classifier = nn.Sequential(
            nn.Linear(num_nodes*2, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        X = X.permute(1,0,2)

        h, _ = self.lstm(X)
        h = h.view(-1, h.shape[2])

        y_hat = self.classifier(h)
        y_hat = y_hat.view(X.shape[0], X.shape[1], -1)

        #y_hat_lens = torch.IntTensor([y_hat.shape[0] for _ in range(y_hat.shape[1])])
        return y_hat#, y_hat_lens
