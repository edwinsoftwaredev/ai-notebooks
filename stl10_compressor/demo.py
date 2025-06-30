import heapq
from itertools import count
import os

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from PIL import Image
import io

import matplotlib.pyplot as plt

from stl10_compressor.autoencoder import ConvolutionalAutoencoder
from stl10_compressor.datasets import load_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_to_pil(tensor):
    tensor = tensor
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor.float() * 255).to(torch.uint8)
    return v2.ToPILImage()(tensor)


def plot_images(worst, best):
    cols, rows = 6, 20
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2 * rows))

    def plot(items, col):    
        i = 0
        while items:
            loss, _, t = heapq.heappop(items)
            input, output = t
            input = tensor_to_pil(input)
            output = tensor_to_pil(output)
    
            axes[i, col].imshow(input)
            axes[i, col].axis('off')
        
            axes[i, col+1].axis('off') 
            axes[i, col+1].imshow(output)

            axes[i, col+2].axis('off')
            axes[i, col+2].text(0, 0.5, f"loss: {abs(loss)}", fontsize=6, va='center', wrap=True)
    
            i += 1


    plot(worst, 0)
    plot(best, 3)

    plt.tight_layout()
    plt.show()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')

    return Image.open(buf)


def demo(result):
    model = ConvolutionalAutoencoder(result.config['model_config'])

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
            batch_size=512,
            shuffle=True,
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True
        )


        with torch.no_grad():

            for X in test_dataloader:
                X = X.to(device)
    
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    output = model(X)
                    # loss per sample
                    loss = torch.nn.functional.mse_loss(output, X, reduction="none")
                    loss = loss.mean(dim=(1, 2, 3)) 
                    
                    for i in range(len(X)):
                        if len(best) < 20:
                            heapq.heappush(best, (-loss[i], next(counter), (X[i], output[i])))
                        elif abs(best[0][0]) > loss[i]:
                            heapq.heappop(best)
                            heapq.heappush(best, (-loss[i], next(counter), (X[i], output[i])))

                        if len(worst) < 20:
                            heapq.heappush(worst, (-loss[i], next(counter), (X[i], output[i])))
                        elif abs(worst[0][0]) < loss[i]:
                            heapq.heappop(worst)
                            heapq.heappush(worst, (-loss[i], next(counter), (X[i], output[i])))


    return plot_images(worst, best)