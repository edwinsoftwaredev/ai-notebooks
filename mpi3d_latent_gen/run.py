from torch import nn
import torch

from torch.utils.data import DataLoader

import wandb

class Run:
    def __init__(self, model: nn.Module, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler(self.device)
        self.model.to(self.device)


    def loss(self, X, epoch=None):
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            loss = self.model(X, epoch)
        
        return loss

    
    def backprop(self, loss, epoch):
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=6000000)

        wandb.log({
            'grad_norm': grad_norm,
            'lr': self.optimizer.param_groups[0]['lr'],
            'epoch': epoch 
        })
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    
    def train(self, dataloader: DataLoader, epoch):
        avg_loss = 0

        self.model.train()
        for X in dataloader:
            X = X.to(self.device)

            loss = self.loss(X, epoch)
            self.backprop(loss, epoch)

            avg_loss += loss.item()

        return { 'loss': avg_loss }

    
    def test(self, dataloader: DataLoader):
        avg_loss = 0

        self.model.eval()
        with torch.no_grad():
            for X in dataloader:
                X = X.to(self.device)

                loss = self.loss(X)

                avg_loss += loss.item()


        return { 'loss': avg_loss }
