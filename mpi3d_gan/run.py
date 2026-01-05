from torch import nn
import torch

from torch.utils.data import DataLoader

import wandb

class Run:
    def __init__(
            self, 
            generator: nn.Module, 
            discriminator: nn.Module, 
            generator_optim, 
            discriminator_optim, 
            latent_dim, 
            batch_size,
            k,
            grad_penalty_c
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optim = generator_optim
        self.discriminator_optim = discriminator_optim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.k = k
        self.grad_penalty_c = grad_penalty_c

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)
    
    def generator_loss(self, z, epoch=None):
        output = self.discriminator(self.generator(z))

        # Wasserstein Generator Loss: D(G(z))
        loss = -torch.mean(output)

        if epoch is not None:
            wandb.log({ 'gen_score': loss.item(), 'epoch': epoch })

        return loss
        
    def discriminator_loss(self, real, z, epoch=None):
        fake = self.generator(z).detach()

        output_real = self.discriminator(real)
        output_fake = self.discriminator(fake)

        loss_real = torch.mean(output_real)
        loss_fake = torch.mean(output_fake)

        epsilon = torch.rand(len(real), 1, 1, 1, device=self.device, dtype=torch.float32)
        random_interpolated_samples = epsilon * real + (1 - epsilon) * fake
        # Make autograd backpropagate to this input/leaf node  
        random_interpolated_samples.requires_grad_(True)
        
        d_out = self.discriminator(random_interpolated_samples)
        grads = torch.autograd.grad(
            outputs=d_out,
            inputs=random_interpolated_samples,
            grad_outputs=torch.ones_like(d_out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        grads = grads[0].view(len(real), -1)
        gp = self.grad_penalty_c * ((grads.norm(2, dim=1) - 1) ** 2).mean()

        # Wasserstein Critic Loss + GP: D(x) - D(G(z)) + gp
        loss = -(loss_real - loss_fake) + gp
        
        if epoch is not None:
            wandb.log({ 'critic_score': loss.item(), 'epoch': epoch })

        return loss
        
    def generator_backprop(self, loss):
        loss.backward()
        self.generator_optim.step()
        self.generator_optim.zero_grad(set_to_none=True)

    def discriminator_backprop(self, loss):
        loss.backward()
        self.discriminator_optim.step()
        self.discriminator_optim.zero_grad(set_to_none=True)

    def gan_train(self, dataloader: DataLoader, epoch):
        critic_score, gen_score = 0, 0

        dataloader_iter = iter(dataloader)

        while True:
            try:
                # CRITIC TRAINING
                self.generator.eval()
                self.discriminator.train()
                for i in range(self.k):
                    # Each iteration of the critic requires a new batch
                    real = next(dataloader_iter)
                    real = real.to(self.device)
                    z = torch.normal(0, 1, size=(len(real), self.latent_dim, 1, 1), device=self.device, dtype=torch.float32)
                    loss = self.discriminator_loss(real, z, epoch)
                    self.discriminator_backprop(loss)
                    critic_score += loss.item()

                # GENERATOR TRAINING
                self.generator.train()
                self.discriminator.eval()
                z = torch.normal(0, 1, size=(self.batch_size, self.latent_dim, 1, 1), device=self.device, dtype=torch.float32)
                loss = self.generator_loss(z, epoch)
                self.generator_backprop(loss)
                gen_score += loss.item()

            except StopIteration:
                break

        return ( critic_score, gen_score )

    def generator_test(self, n):
        gen_score = 0

        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():
            for _ in range(n):
                z = torch.normal(0, 1, size=(self.batch_size, self.latent_dim, 1, 1), device=self.device, dtype=torch.float32)

                loss = self.generator_loss(z)
                gen_score += loss.item()

        return gen_score