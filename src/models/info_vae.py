from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List

Tensor = torch.Tensor

class BaseVAE(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: Tensor) -> List[Tensor]: ...
    @abstractmethod
    def decode(self, z: Tensor) -> Tensor: ...
    @abstractmethod
    def forward(self, x: Tensor, **kw) -> List[Tensor]: ...
    @abstractmethod
    def loss_function(self, *args, **kw) -> Dict[str, Tensor]: ...
    @abstractmethod
    def sample(self, n: int, device) -> Tensor: ...
    def generate(self, x: Tensor, **kw) -> Tensor:
        return self.forward(x)[0]

def _norm(channels: int) -> nn.Module:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)

def _maybe_sn(module: nn.Module, use_sn: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if use_sn else module

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = False) -> None:
        super().__init__()
        self.conv1 = _maybe_sn(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), use_sn)
        self.norm1 = _norm(out_ch)
        self.conv2 = _maybe_sn(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), use_sn)
        self.norm2 = _norm(out_ch)
        self.act   = nn.SiLU(inplace=True)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = False) -> None:
        super().__init__()
        self.down = nn.Sequential(
            _maybe_sn(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False), use_sn),
            _norm(out_ch),
            nn.SiLU(inplace=True),
        )
        self.res = ResBlock(out_ch, out_ch, use_sn=use_sn)

    def forward(self, x: Tensor) -> Tensor:
        return self.res(self.down(x))

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = False) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _maybe_sn(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), use_sn),
            _norm(out_ch),
            nn.SiLU(inplace=True),
        )
        self.res = ResBlock(out_ch, out_ch, use_sn=use_sn)

    def forward(self, x: Tensor) -> Tensor:
        return self.res(self.up(x))

class InfoVAE(BaseVAE):
    def __init__(
        self,
        in_channels:  int            = 3,
        latent_dim:   int            = 256,
        hidden_dims:  List[int]|None = None,
        alpha:        float          = 0.0,
        beta:         float          = 10.0,
        reg_weight:   float          = 50.0,
        kernel_type:  str            = "imq",
        latent_var:   float          = 2.0,
        input_size:   int            = 64,
        use_sn:       bool           = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.latent_dim  = latent_dim
        self.reg_weight  = reg_weight
        self.kernel_type = kernel_type
        self.z_var       = latent_var
        self.alpha       = alpha
        self.beta        = beta

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        n_downs = len(hidden_dims)
        factor  = 2 ** n_downs

        self.spatial_dim    = input_size // factor
        self.flattened_size = hidden_dims[-1] * self.spatial_dim ** 2

        enc_blocks = []
        c = in_channels
        for h in hidden_dims:
            enc_blocks.append(DownBlock(c, h, use_sn=use_sn))
            c = h
        self.encoder = nn.Sequential(*enc_blocks)

        self.fc_mu  = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)

        rev = list(reversed(hidden_dims))
        self._dec_first_ch = rev[0]
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        dec_blocks = []
        for i in range(len(rev) - 1):
            dec_blocks.append(UpBlock(rev[i], rev[i + 1], use_sn=use_sn))
        self.decoder = nn.Sequential(*dec_blocks)

        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(rev[-1], rev[-1], 3, padding=1, bias=False),
            _norm(rev[-1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(rev[-1], in_channels, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: Tensor) -> List[Tensor]:
        h = torch.flatten(self.encoder(x), start_dim=1)
        return [self.fc_mu(h), self.fc_var(h)]

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder_input(z)
        h = h.view(-1, self._dec_first_ch, self.spatial_dim, self.spatial_dim)
        return self.final_layer(self.decoder(h))

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, x: Tensor, **kw) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, z, mu, log_var]

    def loss_function(self, *args, **kwargs) -> Dict[str, Tensor]:
        recons, x, z, mu, log_var = args
        kld_weight      = kwargs["M_N"]
        perceptual_loss = kwargs.get("perceptual_loss", None)
        perceptual_w    = kwargs.get("perceptual_w", 0.1)

        recons_loss = F.l1_loss(recons, x)

        kld_loss = -0.5 * torch.mean(
            torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

        mmd_loss = self.compute_mmd(z)

        loss = (
            self.beta * recons_loss
            + (1.0 - self.alpha) * kld_weight * kld_loss
            + (self.alpha + self.reg_weight - 1.0) * mmd_loss
        )

        if perceptual_loss is not None:
            loss = loss + perceptual_w * perceptual_loss

        return {
            "loss":                loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD":                 kld_loss.detach(),
            "MMD":                 mmd_loss.detach(),
        }

    def _expand_pair(self, x1: Tensor, x2: Tensor):
        N, D = x1.size(0), x1.size(1)
        return (
            x1.unsqueeze(1).expand(N, N, D),
            x2.unsqueeze(0).expand(N, N, D),
        )

    def compute_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1e, x2e = self._expand_pair(x1, x2)
        return self._imq(x1e, x2e) if self.kernel_type == "imq" else self._rbf(x1e, x2e)

    def _rbf(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:
        sigma = 2.0 * x1.size(-1) * self.z_var
        return torch.exp(-(x1 - x2).pow(2).mean(-1) / (sigma + eps))

    def _imq(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:
        C = 2.0 * x1.size(-1) * self.z_var
        return C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

    def compute_mmd(self, z: Tensor) -> Tensor:
        prior = torch.randn_like(z)
        k_pp  = self.compute_kernel(prior, prior)
        k_zz  = self.compute_kernel(z, z)
        k_pz  = self.compute_kernel(prior, z)
        N     = z.size(0)
        mask  = ~torch.eye(N, dtype=torch.bool, device=z.device)
        return k_pp[mask].mean() + k_zz[mask].mean() - 2.0 * k_pz.mean()

    def sample(self, num_samples: int, device, **kw) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def get_reconstruction(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.forward(x)[0]

    def interpolate(self, x1: Tensor, x2: Tensor, num_steps: int = 10) -> Tensor:
        with torch.no_grad():
            x1 = x1.unsqueeze(0) if x1.dim() == 3 else x1
            x2 = x2.unsqueeze(0) if x2.dim() == 3 else x2
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)
            alphas  = torch.linspace(0, 1, num_steps, device=x1.device).unsqueeze(1)
            z_batch = (1.0 - alphas) * mu1 + alphas * mu2
            return self.decode(z_batch)