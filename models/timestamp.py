import torch.nn as nn

###############################################
# Timestamp Transformation
###############################################
class TimestampTransform(nn.Module):

    def forward(self, x, B, T):

        C, H, W = x.shape[1:]

        x = x.view(B, T, C, H, W)

        return x

