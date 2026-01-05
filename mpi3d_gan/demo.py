import os

import torch
from torchvision.transforms import v2

from PIL import Image
import io

import matplotlib.pyplot as plt

from mpi3d_gan.gan import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_to_pil(tensor):
    # tensor values range: [-1, 1]
    tensor = (tensor + 1) / 2 # to -> [0, 1]
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    return v2.ToPILImage()(tensor)


def plot_images(images):
    cols, rows = 20, 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    
    def plot(items):    
        i = 0
        while items:
            image = items.pop()
            input = tensor_to_pil(image)
    
            axes[i].imshow(input)
            axes[i].axis('off')
    
            i += 1


    plot(images)

    plt.tight_layout()
    plt.show()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')

    return Image.open(buf)


def demo(result, latent_dim, model_flag = False):
    if not model_flag:
        model = Generator(result.config['model_config']['generator'])
    
        model.to(device)

        checkpoint_path = os.path.join(result.checkpoint.to_directory(), 'checkpoint.pt')
        data = torch.load(checkpoint_path)
        model.load_state_dict(data['generator'])
    
    else:
        model = result


    with torch.no_grad():
        z = torch.normal(0, 1, size=(20, latent_dim, 1, 1)).to(device=device, dtype=torch.float32)
        output = model(z)
    
    return plot_images(list(torch.unbind(output, dim=0)))