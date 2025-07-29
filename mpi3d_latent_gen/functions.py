import datetime
import os
import tempfile

from torch.utils.data import DataLoader
from torch import nn
import torch

from ray import tune

from datetime import datetime

from mpi3d_latent_gen import demo
from mpi3d_latent_gen.datasets import load_datasets
from mpi3d_latent_gen.run import Run
from mpi3d_latent_gen.vae import VariationalAutoencoder

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")

import wandb
wandb.login(key=secret_value_0)

torch.backends.cudnn.benchmark = True

def raytune_load_checkpoint(model: nn.Module, optim: torch.optim.Adam):
    if tune.get_checkpoint():
        loaded_checkpoint = tune.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optim_state = torch.load(os.path.join(loaded_checkpoint_dir, 'checkpoint.pt'))
            model.load_state_dict(model_state)
            optim.load_state_dict(optim_state)
        

def raytune_save_checkpoint(model: nn.Module, optim: torch.optim.Adam, train_metrics, test_metrics):
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')
        torch.save((model.state_dict(), optim.state_dict()), path)
        checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
        tune.report({**train_metrics, **test_metrics}, checkpoint=checkpoint)


def train_model(config):
    model = VariationalAutoencoder(config['model_config'])
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    schdlr_patience = config['schdlr_patience']
    schdlr_factor = config['schdlr_factor']
    
    loss_fn = None
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=schdlr_patience, factor=schdlr_factor)

    wandb.init(project='mpi3d_latent_gen', group='experiment_1', config=config)

    raytune_load_checkpoint(model, optimizer)

    train_sets, test_sets, npartitions = load_datasets()

    train_metrics = None
    test_metrics = None

    run = Run(model, loss_fn, optimizer)

    for t in range(1, epochs+1):
        train_state = { 'loss': 0, 'nbatches': 0, 'dataset_size': 0 }
        test_state = { 'loss': 0, 'nbatches': 0, 'dataset_size': 0 }

        start = datetime.now()

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

            train_metrics = run.train(train_dataloader, t)

            train_state['loss'] += train_metrics['loss']
            train_state['nbatches'] += len(train_dataloader)
            train_state['dataset_size'] += len(train_dataloader.dataset)


        for partition in range(npartitions):
            test_dataloader = DataLoader(
                test_sets[partition],
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                prefetch_factor=4,
                persistent_workers=True,
                pin_memory=True
            )

            test_metrics = run.test(test_dataloader)

            test_state['loss'] += test_metrics['loss']
            test_state['nbatches'] += len(test_dataloader)
            test_state['dataset_size'] += len(test_dataloader.dataset)

        
        end = datetime.now()

        print(f'epoch {t}, time: {end - start}\n')

        train_loss = train_state['loss'] / train_state['nbatches']
        test_loss = test_state['loss'] / test_state['nbatches']

        scheduler.step(test_loss)

        print(f'last lr: {scheduler.get_last_lr()}')
        
        wandb.log({ 'train_loss': train_loss, 'epoch': t })
        wandb.log({ 'test_loss': test_loss, 'epoch': t })
        wandb.log({ 'examples': [wandb.Image(demo(model, config['model_config']['latent_dim'], True))], 'epoch': t })
        


    raytune_save_checkpoint(
        model,
        optimizer,
        { 'train_loss': train_loss, 'epoch': t },
        { 'test_loss': test_loss, 'epoch': t }
    )

    wandb.finish()
