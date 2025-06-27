from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch

train_transforms = v2.Compose([
    v2.ToDtype(torch.float16, scale=True),
])

test_transforms = v2.Compose([
    v2.ToDtype(torch.float16, scale=True),
])

class ImageDataset(Dataset):
    def __init__(self, images, train_set):
        self.images = images
        self.transforms = train_transforms if train_set else test_transforms


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transforms(image)
        return image
