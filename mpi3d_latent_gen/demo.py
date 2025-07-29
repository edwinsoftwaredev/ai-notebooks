import os

import torch
from torchvision.transforms import v2

from PIL import Image
import io

import matplotlib.pyplot as plt

from mpi3d_latent_gen.vae import VariationalAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_to_pil(tensor):
    tensor = (tensor * 255).to(torch.uint8)
    return v2.ToPILImage()(tensor)


def plot_images(images):
    cols, rows = 1, 20
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
        model = VariationalAutoencoder(result.config['model_config'])
    
        model.to(device)
    
        checkpoint_path = os.path.join(result.checkpoint.to_directory(), 'checkpoint.pt')
        model_state, _optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
    
    else:
        model = result


    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            z = torch.normal(0, 1, size=(20, latent_dim)).to(device=device, dtype=torch.float32)
            z_network_output = model.z_network(z)
            mu, _log_var = model.decode(z_network_output)
            mu = torch.sigmoid(mu).to(device=device, dtype=torch.float32)

    
    return plot_images(list(torch.unbind(mu, dim=0)))