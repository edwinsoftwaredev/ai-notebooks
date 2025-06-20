import os

import torch

from torch.utils.data import DataLoader
from torchvision.transforms import v2

import matplotlib.pyplot as plt

import heapq
from itertools import count
from operator import itemgetter

from PIL import Image
import io

import wandb

from cifar10.cnn import CNN
from cifar10.datasets import load_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float16).view(3, 1, 1)
mean = mean.to(device)
std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float16).view(3, 1, 1)
std = std.to(device)

def tensor_to_pil(tensor):
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor.float() * 255).to(torch.uint8)
    return v2.ToPILImage()(tensor)


def demo(result):
    model = CNN(result.config['model_config'])

    model.to(device)

    checkpoint_path = os.path.join(result.checkpoint.to_directory(), 'checkpoint.pt')
    model_state, _optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)

    _train_sets, test_sets, npartitions = load_datasets()

    best = [] # correct - high confidence min heap
    worst = [] # incorrect - high confidence min heap

    counter = count()

    for partition in range(npartitions):
        test_dataloader = DataLoader(
            test_sets[partition],
            batch_size=64,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True
        )


        with torch.no_grad():

            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
    
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    output = model(X)
                    pred_probs = torch.nn.functional.softmax(output, dim=1)
                    max_args = pred_probs.argmax(1)

                    for i in range(len(X)):
                        prob = max_args[i]
                        if prob == y[i]:
                            if len(best) == 20:
                                if best[0][0] < pred_probs[i][prob]:
                                    heapq.heappop(best)
                                    heapq.heappush(best, (pred_probs[i][prob], next(counter), (X[i], pred_probs[i], y[i])))

                            else:
                                heapq.heappush(best, (pred_probs[i][prob], next(counter), (X[i], pred_probs[i], y[i])))

                        else:
                            if len(worst) == 20:
                                if worst[0][0] < pred_probs[i][prob]:
                                    heapq.heappop(worst)
                                    heapq.heappush(worst, (pred_probs[i][prob], next(counter), (X[i], pred_probs[i], y[i])))
                            else:
                                heapq.heappush(worst, (pred_probs[i][prob], next(counter), (X[i], pred_probs[i], y[i])))
                                


    labels_map = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    cols, rows = 4, 20
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2 * rows))

    def plot(items, col):    
        i = 0
        while items:
            _, _, t = heapq.heappop(items)
            img, pred, label = t
            img = tensor_to_pil(img)
    
            pred_vals = []
            pred = [(i,p) for i,p in enumerate(pred)]
            pred.sort(key=itemgetter(1), reverse=True)
            for j in range(len(pred)):
                pred_vals.append(f"{labels_map[pred[j][0]]}: {pred[j][1]*100:.2f}%\n")
    
            axes[i, col].imshow(img)
            axes[i, col].axis('off')
        
            axes[i, col+1].axis('off') 
            axes[i, col+1].text(0, 0.5, f"{''.join(pred_vals)}\nlabel: {labels_map[label.item()]}", fontsize=6, va='center', wrap=True)
    
            i += 1


    plot(worst, 0)
    plot(best, 2)

    plt.tight_layout()
    plt.show()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')

    wandb.init(project='cifar10', group='experiment_4')
    wandb.log({"examples": [wandb.Image(Image.open(buf))]})
    wandb.finish()
