import os
import tempfile
from datetime import datetime

import torch
from torch import nn

from ray import tune

from torch.utils.data import DataLoader

from cifar10_classifier.cnn import CNN
from cifar10_classifier.datasets import load_datasets
from cifar10_classifier.run import Run

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
    model = CNN(config['model_config'])
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    weight_decay = config['weight_decay']
    schdlr_patience = config['schdlr_patience']
    schdlr_factor = config['schdlr_factor']

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=schdlr_patience, factor=schdlr_factor)

    wandb.init(project='cifar10', group='experiment_4', config=config)

    raytune_load_checkpoint(model, optimizer)

    train_sets, test_sets, npartitions = load_datasets()

    train_metrics = None
    test_metrics = None

    run = Run(model, loss_fn, optimizer)

    for t in range(1, epochs + 1):
        train_state = { 'correct': 0, 'loss': 0, 'nbatches': 0, 'dataset_size': 0 }
        test_state = { 'correct': 0, 'loss': 0, 'nbatches': 0, 'dataset_size': 0 }

        start = datetime.now()

        for partition in range(npartitions):
            train_dataloader = DataLoader(
                train_sets[partition],
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                prefetch_factor=2,
                persistent_workers=True,
                pin_memory=True
            )

            test_dataloader = DataLoader(
                test_sets[partition],
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                prefetch_factor=2,
                persistent_workers=True,
                pin_memory=True
            )

            train_metrics = run.train(train_dataloader)
            test_metrics = run.test(test_dataloader)

            train_state['correct'] += train_metrics['correct']
            train_state['loss'] += train_metrics['loss']
            train_state['nbatches'] += len(train_dataloader)
            train_state['dataset_size'] += len(train_dataloader.dataset)

            test_state['correct'] += test_metrics['correct']
            test_state['loss'] += test_metrics['loss']
            test_state['nbatches'] += len(test_dataloader)
            test_state['dataset_size'] += len(test_dataloader.dataset)

        
        end = datetime.now()

        print(f'epoch {t}, time: {end - start}\n')

        train_correct = train_state['correct'] / train_state['dataset_size']
        train_loss = train_state['loss'] / train_state['nbatches']

        test_correct = test_state['correct'] / test_state['dataset_size']
        test_loss = test_state['loss'] / test_state['nbatches']

        scheduler.step(test_loss)

        print(f'last lr: {scheduler.get_last_lr()}')

        wandb.log({ 'train_accuracy': train_correct, 'train_loss': train_loss, 'epoch': t })
        wandb.log({ 'test_accuracy': test_correct, 'test_loss': test_loss, 'epoch': t })


    raytune_save_checkpoint(
        model,
        optimizer,
        { 'train_accuracy': train_correct, 'train_loss': train_loss, 'epoch': t },
        { 'test_accuracy': test_correct, 'test_loss': test_loss, 'epoch': t }
    )

    wandb.finish()
