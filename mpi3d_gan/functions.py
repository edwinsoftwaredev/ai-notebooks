import datetime
import os
from sched import scheduler
import tempfile

from torch.utils.data import DataLoader
from torch import nn
import torch

from ray import tune

from datetime import datetime

from mpi3d_gan import demo
from mpi3d_gan.datasets import load_datasets
from mpi3d_gan.run import Run
from mpi3d_gan.gan import Generator, Discriminator

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")

import wandb

wandb.login(key=secret_value_0)

def raytune_load_checkpoint(
        generator: nn.Module, 
        discriminator: nn.Module, 
        generator_optim: torch.optim.RMSprop, 
        discriminator_optim: torch.optim.RMSprop
    ):
    if tune.get_checkpoint():
        loaded_checkpoint = tune.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            data = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            generator.load_state_dict(data["generator"])
            discriminator.load_state_dict(data["discriminator"])
            generator_optim.load_state_dict(data["generator_optim"])
            discriminator_optim.load_state_dict(data["discriminator_optim"])

        

def raytune_save_checkpoint(
        generator: nn.Module, 
        discriminator: nn.Module, 
        generator_optim: torch.optim.RMSprop, 
        discriminator_optim: torch.optim.RMSprop,
        train_metrics, 
        test_metrics
):
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')
        torch.save({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "discriminator_optim": discriminator_optim.state_dict(),
            "generator_optim": generator_optim.state_dict(),
        }, path)
        checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
        tune.report({**train_metrics, **test_metrics}, checkpoint=checkpoint)


def train_model(config):
    generator = Generator(config['model_config'])
    discriminator = Discriminator(config['model_config'])

    lr = config['lr']
    beta_1, beta_2 = config['beta_1'], config['beta_2']
    epochs = config['epochs']
    batch_size = config['batch_size']
    schdlr_patience = config['schdlr_patience']
    schdlr_factor = config['schdlr_factor']
    loss_fn = None
    
    generator_optim = torch.optim.Adam(generator.parameters(), lr, betas=(beta_1, beta_2))
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr, betas=(beta_1, beta_2))

    wandb.init(project='mpi3d_base_gan', group='experiment_1', config=config)
    
    raytune_load_checkpoint(generator, discriminator, generator_optim, discriminator_optim)

    train_sets, _, npartitions = load_datasets()

    run = Run(
        generator, 
        discriminator, 
        generator_optim, 
        discriminator_optim, 
        config['model_config']['latent_dim'],
        batch_size,
        config['model_config']['k'],
        config['model_config']['grad_penalty_c']
    )

    for t in range(1, epochs + 1):
        start = datetime.now()

        nbatches = 0

        for partition in range(npartitions):
            train_dataloader = DataLoader(
                train_sets[partition],
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                prefetch_factor=4,
                persistent_workers=True,
                pin_memory=True
            )

            nbatches += len(train_dataloader)
            critic_score, gen_score = run.gan_train(train_dataloader, t)

            critic_score /= len(train_dataloader)
            gen_score /= len(train_dataloader)

        # TEST: Generator
        n = nbatches
        n = int(n * 0.2)
        gen_score = run.generator_test(n)
        gen_score /= n

        wandb.log({ 'examples': [wandb.Image(demo(generator, config['model_config']['latent_dim'], True))], 'epoch': t })

        end = datetime.now()
        print(f'epoch {t}, time: {end - start}\n')


    # There is only 1 run
    raytune_save_checkpoint(
        generator,
        discriminator,
        generator_optim,
        discriminator_optim,
        { 'critic_score': critic_score, 'epoch': t },
        { 'test_gen_score': gen_score, 'epoch': t }
    )

    wandb.finish()