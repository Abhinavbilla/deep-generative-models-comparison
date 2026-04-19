import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

class VanillaVAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 128, hidden_dims: List = None) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        return [self.fc_mu(result), self.fc_var(result)]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        return self.final_layer(result)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']
        
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}

    def sample(self, num_samples: int, current_device: torch.device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
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
            alphas = torch.linspace(0, 1, num_steps, device=x1.device).unsqueeze(1)
            z_batch = (1.0 - alphas) * mu1 + alphas * mu2
            return self.decode(z_batch)