from torch import nn
import torch

from torch.utils.data import DataLoader

class Run:
    def __init__(self, model: nn.Module, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler(self.device)
        self.model.to(self.device)


    def loss(self, X, y):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            output = self.model(X)
            loss = self.loss_fn(output, y)
        
        return output, loss


    def backprop(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
    

    def train(self, dataloader: DataLoader):
        avg_loss = 0

        self.model.train()
        for X in dataloader:
            X = X.to(self.device)
            y = X

            output, loss = self.loss(X, y)
            self.backprop(loss)

            avg_loss += loss.item()


        return { 'loss': avg_loss }
    

    def test(self, dataloader: DataLoader):
        avg_loss = 0

        self.model.eval()
        with torch.no_grad():
            for X in dataloader:
                X = X.to(self.device)
                y = X

                output, loss = self.loss(X, y)

                avg_loss += loss.item()


        return { 'loss': avg_loss }
        