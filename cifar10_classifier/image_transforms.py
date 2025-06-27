import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 
import torch

train_transform = v2.Compose([
    v2.PILToTensor(),
    v2.RandomCrop(32, padding=4),
    v2.ColorJitter(.5,.3),
    v2.RandomEqualize(p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float16, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

test_transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float16, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])


def apply_transforms(img, train):
    # PIL image object
    img_pil = Image.open(io.BytesIO(img['bytes']))
    img_pil = img_pil.convert('RGB')
    return train_transform(img_pil) if train else test_transform(img_pil)


class ImageDataset(Dataset):
    def __init__(self, images, labels, train):
        self.images = images
        self.labels = labels
        self.train = train


    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):
        image = self.images.iloc[idx]
        label = self.labels[idx]

        image = apply_transforms(image, self.train)
        label = torch.tensor(label, dtype=torch.long)

        return image, label