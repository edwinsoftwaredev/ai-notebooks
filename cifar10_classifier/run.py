import torch
from torch import nn
from torch.utils.data import DataLoader

class Run:
    def __init__(self, model: nn.Module, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler(self.device.type)
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
    

    def y_pred(self, output):
        pred_probs = torch.nn.functional.softmax(output, dim=1)
        return pred_probs.argmax(1)
        

    def train(self, dataloader: DataLoader):
        correct = 0
        avg_loss = 0

        self.model.train()
        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)

            output, loss = self.loss(X, y)

            self.backprop(loss)

            # validate results            
            correct += (self.y_pred(output) == y).sum().item()
            avg_loss += loss.item()


        return { 'correct': correct, 'loss': avg_loss }
    

    def test(self, dataloader: DataLoader):
        correct = 0
        avg_loss = 0

        self.model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)

                output, loss = self.loss(X, y)

                # validate results
                correct += (self.y_pred(output) == y).sum().item()
                avg_loss += loss.item()


        return { 'correct': correct, 'loss': avg_loss }