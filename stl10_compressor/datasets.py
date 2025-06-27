import os
import dask
import requests
import tarfile

from pathlib import Path

import numpy as np

import dask.array as da
import torch

from stl10_compressor.image_dataset import ImageDataset

dir_path = "./content/stl10"

def _download_dataset():
    url = 'https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

    os.makedirs(dir_path, exist_ok=True)
    
    filename = f"{dir_path}/stl10.tar.gz"
    
    response = requests.get(url)

    with open(filename, 'wb') as f:
        f.write(response.content)


    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(dir_path)


    os.remove(filename)


def load_datasets():
    if not Path(dir_path).is_dir():
        _download_dataset()

    filename = f"{dir_path}/stl10_binary/unlabeled_X.bin"
    
    with open(filename, 'rb') as f:
        ndarray = np.fromfile(f, dtype=np.uint8)


    images = np.reshape(ndarray, (-1, 3, 96, 96))

    # Dask
    # dda = da.from_array(images, chunks=(50000, 3, 96, 96))

    # @dask.delayed
    # def to_dataset(partition, train_set):
    #     return ImageDataset(partition, train_set)
    
    
    # train_sets, test_sets = dda.random_split([0.8, 0.2], random_state=123, shuffle=True)

    # npartitions = train_sets.npartitions

    # train_sets = [to_dataset(partition) for partition in train_sets.to_delayed()]
    # test_sets = [to_dataset(partition) for partition in test_sets.to_delayed()]

    # dataset fits in memory
    # train_sets = [partition.compute() for partition in train_sets]
    # test_sets = [partition.compute() for partition in test_sets]

    images = torch.tensor(images)
    train_sets = [ImageDataset(images[:int(len(images)*0.8)], True)]
    test_sets = [ImageDataset(images[int(len(images)*0.8):], False)]

    return train_sets, test_sets, 1