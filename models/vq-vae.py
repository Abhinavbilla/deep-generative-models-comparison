import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple
from torch import Tensor
from models import BaseVAE 

class VectorQuantizerEMA(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, beta: float = 0.25, decay: float = 0.99, eps: float = 1e-5, noise_scale: float = 0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.noise_scale = noise_scale

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

        self.register_buffer('cluster_size', torch.zeros(self.vocab_size))
        self.register_buffer('embed_avg', self.embedding.weight.clone().detach())

    def forward(self, z_e: Tensor) -> Tuple[Tensor, Tensor, float]:
        if self.training and self.noise_scale > 0.0:
            z_e = z_e + torch.randn_like(z_e) * self.noise_scale

        z_e_transposed = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flat = z_e_transposed.reshape(-1, self.embed_dim)

        dist = torch.cdist(z_e_flat, self.embedding.weight, p=2.0)
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)

        unique_indices = torch.unique(encoding_indices)
        utilization = unique_indices.numel() / self.vocab_size

        if self.training:
            encodings = torch.zeros(encoding_indices.size(0), self.vocab_size, device=z_e.device)
            encodings.scatter_(1, encoding_indices, 1)

            # Modern PyTorch EMA update using torch.no_grad() instead of .data
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(
                    encodings.sum(0), alpha=1 - self.decay
                )
                n = self.cluster_size.sum()
                cluster_size_norm = (self.cluster_size + self.eps) / (n + self.vocab_size * self.eps) * n
                embed_sum = torch.matmul(encodings.t(), z_e_flat)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                self.embedding.weight.copy_(self.embed_avg / cluster_size_norm.unsqueeze(1))

        z_q = self.embedding(encoding_indices.squeeze(1)).reshape(z_e_transposed.shape)
        
        commitment_loss = F.mse_loss(z_q.detach(), z_e_transposed) * self.beta
        z_q = z_e_transposed + (z_q - z_e_transposed).detach()
        
        return z_q.permute(0, 3, 1, 2).contiguous(), commitment_loss, utilization

class ResidualLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.resblock(x)

class VQVAE(BaseVAE):
    def __init__(self, in_channels: int = 3, embed_dim: int = 64, vocab_size: int = 512,
                 hidden_dims: List[int] = None, beta: float = 0.25, noise_scale: float = 0.01, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [128, 256]

        enc_layers = []
        curr_channels = in_channels
        for h_dim in hidden_dims:
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(curr_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            curr_channels = h_dim

        enc_layers.append(nn.Conv2d(curr_channels, curr_channels, kernel_size=3, stride=1, padding=1))
        enc_layers.append(nn.LeakyReLU())

        for _ in range(6):
            enc_layers.append(ResidualLayer(curr_channels))

        enc_layers.append(nn.Conv2d(curr_channels, embed_dim, kernel_size=1, stride=1))
        self.encoder = nn.Sequential(*enc_layers)

        self.quantizer = VectorQuantizerEMA(vocab_size, embed_dim, beta, noise_scale=noise_scale)

        dec_layers = []
        dec_layers.append(nn.Conv2d(embed_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1))
        dec_layers.append(nn.LeakyReLU())

        for _ in range(6):
            dec_layers.append(ResidualLayer(hidden_dims[-1]))

        hidden_dims.reverse()
        curr_channels = hidden_dims[0]

        for i in range(len(hidden_dims) - 1):
            next_channels = hidden_dims[i + 1]
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(curr_channels, next_channels, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            curr_channels = next_channels

        dec_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(curr_channels, in_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z_q: Tensor) -> Tensor:
        return self.decoder(z_q)

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        z_e = self.encode(x)
        z_q, vq_loss, utilization = self.quantizer(z_e)
        x_recon = self.decode(z_q)
        return [x_recon, x, vq_loss, utilization]

    def loss_function(self, *args, **kwargs) -> dict:
        x_recon, x, vq_loss, utilization = args[0], args[1], args[2], args[3]
        
        recon_loss = F.smooth_l1_loss(x_recon, x)
        
        loss = recon_loss + vq_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recon_loss,
            'VQ_Loss': vq_loss,
            'Codebook_Utilization': utilization
        }

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]