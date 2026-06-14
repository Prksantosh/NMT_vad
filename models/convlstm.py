import torch
import torch.nn as nn

###############################################
# ConvLSTM Cell
###############################################
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            3,
            padding=1
        )

    def forward(self, x, h, c):

        combined = torch.cat([x, h], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)

        g = torch.tanh(g)

        c = f * c + i * g

        h = o * torch.tanh(c)

        return h, c


###############################################
# ConvLSTM Layer
###############################################
class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x):

        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        c = torch.zeros_like(h)

        outputs = []

        for t in range(T):

            h, c = self.cell(x[:, t], h, c)

            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs, h
