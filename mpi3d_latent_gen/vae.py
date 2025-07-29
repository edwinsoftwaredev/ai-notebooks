from torch import nn
import torch

import wandb

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
        self.encoder = nn.Sequential(*self._get_layers(config['encoder']))

        # decoder / generative model / likelihood / p(x | z)
        self.decoder = nn.Sequential(*self._get_layers(config['decoder']))

        encoder_output_shape = config['encoder_output_shape']
        enc_out_c, enc_out_h, enc_out_w = encoder_output_shape
        encoder_output_flatten_shape = enc_out_c * enc_out_h * enc_out_w 

        # TODO: set i/o dynamically
        self.encoder_mu_head = nn.Linear(encoder_output_flatten_shape, self.latent_dim, device=self.device, dtype=torch.float32)
        self.encoder_log_var_head = nn.Linear(encoder_output_flatten_shape, self.latent_dim, device=self.device, dtype=torch.float32)
        
        self.z_network = nn.Sequential(
            nn.Linear(self.latent_dim, encoder_output_flatten_shape, device=self.device, dtype=torch.float32),
            nn.LeakyReLU(),
            nn.Unflatten(1, encoder_output_shape)
        )

        self.decoder_mu_head = nn.ConvTranspose2d(3, 3, 3, 1, padding=1, device=self.device, dtype=torch.float32)
        self.decoder_log_var_head = nn.ConvTranspose2d(3, 3, 3, 1, padding=1, device=self.device, dtype=torch.float32)


    def _get_layers(self, kv):
        layers = []
        
        for l in kv.values():
            if 'conv2d' in l:
                conv = l['conv2d']

                layers.append(nn.Conv2d(conv['in'], conv['out'], conv['kernel'], padding=conv['padding'], stride=conv['stride'], bias=True))
                layers.append(nn.LeakyReLU())


            if 'convT2d' in l:
                convt = l['convT2d']
                layers.append(
                    nn.ConvTranspose2d(
                        convt['in'], 
                        convt['out'], 
                        convt['kernel'], 
                        convt['stride'], 
                        padding=convt['padding'], 
                        output_padding=convt['out_padding'], 
                        bias=True
                    )
                )

                # Note that the decoder has two separate heads, so
                # an activation func is added after each layer in config['decoder']
                layers.append(nn.LeakyReLU())


            if 'linear' in l:
                linear = l['linear']
                layers.append(nn.Linear(linear['in'], linear['out']))
                layers.append(nn.LeakyReLU())
                

            if 'dropout' in l:
                dropout = l['dropout']
                layers.append(nn.Dropout2d(p=dropout['prob']))


            if 'maxpool' in l:
                maxpool = l['maxpool']
                layers.append(nn.MaxPool2d(maxpool['kernel'], stride=maxpool['stride'], padding=maxpool['padding']))

        
        return layers
    

    def encode(self, input):
        # (mu, log σ) = q(z|x)
        h = self.encoder(input)
        
        h = h.view(h.size(0), -1)   # Flattening: [batch_size, flatten_dim]
        
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
        encoder_mu, encoder_log_var = self.encode(images)
        encoder_log_var = torch.clamp(encoder_log_var, min=-2, max=2).to(device=self.device, dtype=torch.float32)
        encoder_sd = torch.exp(encoder_log_var * 1/2) # standard deviation
        encoder_var = torch.exp(encoder_log_var) # variance

        # sampled/random noise
        eps = torch.normal(0, 1, size=encoder_mu.shape, device=self.device, dtype=torch.float32)

        # reparameterization
        z = encoder_mu + encoder_sd * eps

        # latent space
        z_network_output = self.z_network(z)

        # DECODE
        decoder_mu, decoder_log_var = self.decode(z_network_output)
        decoder_mu = torch.sigmoid(decoder_mu).to(device=self.device, dtype=torch.float32) # Normalize decoder_mu output on the same scale of images: [0, 1]
        decoder_log_var = torch.clamp(decoder_log_var, min=-2, max=2).to(device=self.device, dtype=torch.float32)
        decoder_var = torch.exp(decoder_log_var) + 1e-10 # variance + epsilon value in case variance is 0

        pi = torch.tensor(2*torch.pi, device=self.device, dtype=torch.float32)

        ##        ELBO components      ##
        ##      -- log q(z | x) --     ##
        log_qz_x = -(1/2)*torch.sum(torch.log(pi) + encoder_log_var + torch.pow(eps, 2), dim=1)
        
        ##       -- log p(z) --        ##
        log_pz = -(1/2)*torch.sum(torch.log(pi) + torch.pow(z, 2), dim=1)
        
        ##      -- log p(x | z) --     ##
        log_px_z = -(1/2)*torch.sum(torch.log(pi) + decoder_log_var + torch.pow((images - decoder_mu), 2) / decoder_var, dim=(1,2,3))

        ##   normalizes log_px_z (the log likelihood) to make it the avg of log_px_z per pixel   ##
        # pixel_count = images.shape[1] * images.shape[2] * images.shape[3]
        # log_px_z = log_px_z / pixel_count

        
        if epoch is not None:
            kl_cost_annealing = min(1.0, (epoch - 1) / self.kl_cost_annealing_time) # warm-up KL divergence for self.kl_cost_annealing_time
            
            wandb.log({
                'log_px_z': torch.mean(log_px_z).item(),
                'log_qz_x': torch.mean(log_qz_x).item(),
                'log_pz': torch.mean(log_pz).item(),
                'z': torch.mean(z).item(),
                'z_network_output': torch.mean(z_network_output).item(),
                'KL_div': torch.mean(log_qz_x - log_pz).item(),
                'KL_div_annealing': torch.mean(kl_cost_annealing * (log_qz_x - log_pz)).item(),
                'kl_weight': kl_cost_annealing,
                'elbo': -torch.mean(log_px_z - kl_cost_annealing * (log_qz_x - log_pz)).item(),
                'recon_loss': -torch.mean(log_px_z).item(),
                'encoder_mu': torch.mean(encoder_mu).item(),
                'encoder_log_var': torch.mean(encoder_log_var).item(),
                'decoder_mu': torch.mean(decoder_mu).item(),
                'decoder_log_var': torch.mean(decoder_log_var).item(),
                'epoch': epoch 
            })
            
            return torch.mean(-(log_px_z - kl_cost_annealing * (log_qz_x - log_pz)))

        
        # minimization of the ELBO ~= -ELBO
        return torch.mean(-(log_px_z + log_pz - log_qz_x))