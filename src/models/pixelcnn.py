
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())

        _, _, h, w = self.weight.shape
        self.mask.fill_(1)

        self.mask[:, :, h//2, w//2 + (mask_type == "B"):] = 0
        self.mask[:, :, h//2 + 1:] = 0

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        self.stride, self.padding)



class GatedResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = MaskedConv2d("B", channels, 2*channels, 3, padding=1)
        self.conv1x1 = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout2d(0.1)   

    def forward(self, x):
        out = self.conv(F.relu(x))

        a, b = torch.chunk(out, 2, dim=1)
        out = torch.tanh(a) * torch.sigmoid(b)

        out = self.dropout(out)   
        out = self.conv1x1(out)

        return x + out



class PixelCNN(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, 512)

       
        self.input_conv = MaskedConv2d("A", 512, 512, 7, padding=3)

       
        self.res_blocks = nn.Sequential(
            *[GatedResidualBlock(512) for _ in range(30)]
        )

       
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, num_embeddings, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2)

        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.output(x)

        return x