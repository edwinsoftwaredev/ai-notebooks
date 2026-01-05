from torch import nn
import torch

import wandb

from funcs import get_layers

class VariationalAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Notes:
        # - It is common to avoid BatchNorm when training VAEs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # latent space dimensionality
        self.latent_dim = config['latent_dim']

        self.kl_cost_annealing_time = config['kl_cost_annealing_time']

        # encoder / aproximate inference model / posterior / q(z | x) ~= p(z | x)
        self.encoder = nn.Sequential(*get_layers(config['encoder']))

        # decoder / generative model / likelihood / p(x | z)
        self.decoder = nn.Sequential(*get_layers(config['decoder']))

        enc_out_c, *_ = config['encoder_output_shape']

        # TODO: set i/o dynamically
        self.encoder_mu_head = nn.Conv2d(enc_out_c, self.latent_dim, kernel_size=4, stride=1, padding=0, bias=True)        # 512, 7, 1, 1
        self.encoder_log_var_head = nn.Conv2d(enc_out_c, self.latent_dim, kernel_size=4, stride=1, padding=0, bias=True)   # 512, 7, 1, 1

        self.decoder_mu_head = nn.ConvTranspose2d(3, 3, 3, 1, padding=1, bias=True)
        self.decoder_log_var_head = nn.ConvTranspose2d(3, 3, 3, 1, padding=1, bias=True)

        # this value might not get saved when reloading model
        self.steps = 0

    def encode(self, input):
        # (mu, log σ) = q(z|x)
        h = self.encoder(input)
        
        # mu head
        mu = self.encoder_mu_head(h)

        # log σ head
        log_var = self.encoder_log_var_head(h)

        return mu, log_var


    def decode(self, z):
        # (mu, log var) = p(x|z)
         
        h = self.decoder(z)

        # mu head
        mu = self.decoder_mu_head(h)
        
        # log var head
        log_var = self.decoder_log_var_head(h)

        return mu, log_var


    def forward(self, images, epoch = None):
        # ENCODE
        self.steps += 1
        
        encoder_mu, encoder_log_var = self.encode(images)
        
        encoder_log_var = torch.clamp(encoder_log_var, min=-10, max=10).to(device=self.device, dtype=torch.float32)
        
        # encoder_sd = torch.exp(encoder_log_var * 1/2) # standard deviation
        encoder_var = torch.exp(encoder_log_var) # variance

        # sampled/random noise
        eps = torch.normal(0, 1, size=encoder_mu.shape, device=self.device, dtype=torch.float32)

        # reparameterization
        z = encoder_mu + encoder_var * eps

        # DECODE
        decoder_mu, decoder_log_var = self.decode(z)
        decoder_mu_norm = torch.tanh(decoder_mu).to(device=self.device, dtype=torch.float32)
        
        # decoder_log_var = torch.clamp(decoder_log_var, min=-10, max=10).to(device=self.device, dtype=torch.float32)
        decoder_log_var = torch.ones(size=decoder_log_var.shape, device=self.device, dtype=torch.float32)
        
        decoder_var = torch.exp(decoder_log_var) + 1e-10 # variance + epsilon value in case variance is 0

        pi = torch.tensor(2*torch.pi, device=self.device, dtype=torch.float32)

        ##        ELBO components      ##
        ##      -- log q(z | x) --     ##
        log_qz_x = -(1/2)*torch.sum(torch.log(pi) + encoder_log_var + torch.pow(eps, 2), dim=1)
        
        ##       -- log p(z) --        ##
        log_pz = -(1/2)*torch.sum(torch.log(pi) + torch.pow(z, 2), dim=1)
        
        ##      -- log p(x | z) --     ##
        log_px_z = -(1/2)*torch.sum(torch.log(pi) + decoder_log_var + torch.pow((images - decoder_mu_norm), 2) / decoder_var, dim=(1,2,3))

        
        if epoch is not None:
            kl_cost_annealing = min(1, self.steps / self.kl_cost_annealing_time) # warm-up KL divergence for self.kl_cost_annealing_time
            
            wandb.log({
                'log_px_z': torch.mean(log_px_z).item(),
                'log_qz_x': torch.mean(log_qz_x).item(),
                'log_pz': torch.mean(log_pz).item(),
                'z': torch.mean(z).item(),
                'KL_div': torch.mean(log_qz_x - log_pz).item(),
                'KL_div_annealing': torch.mean(kl_cost_annealing * (log_qz_x - log_pz)).item(),
                'kl_weight': kl_cost_annealing,
                'elbo': -torch.mean(log_px_z - kl_cost_annealing * (log_qz_x - log_pz)).item(),
                'recon_loss': -torch.mean(log_px_z).item(),
                'encoder_mu': torch.mean(encoder_mu).item(),
                'encoder_log_var': torch.mean(encoder_log_var).item(),
                'decoder_mu': torch.mean(decoder_mu_norm).item(),
                'images': torch.mean(images).item(),
                'decoder_log_var': torch.mean(decoder_log_var).item(),
                'decoder_var': torch.mean(decoder_var).item(),
                'distance': torch.mean(torch.sum(torch.pow((images - decoder_mu_norm), 2) / decoder_var, dim=(1,2,3))).item(),
                'epoch': epoch 
            })
            
            return torch.mean(-(log_px_z - kl_cost_annealing * (log_qz_x - log_pz)))

        
        # minimization of the ELBO ~= -ELBO
        return torch.mean(-(log_px_z + (log_pz - log_qz_x)))