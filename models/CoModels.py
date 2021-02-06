from torch import nn
import copy

class CoModels(nn.Module):
    def __init__(self, models):
        super(CoModels, self).__init__()

        self.model_1, self.model_2 = models

    def forward(self, batch):
        y_hat_1, y_hat_lens = self.model_1(batch)
        y_hat_2, _ = self.model_2(batch)

        return (y_hat_1, y_hat_2), y_hat_lens
