import dask.dataframe as dd
from dask.delayed import delayed
import pyarrow as pa

from image_transforms import ImageDataset

from pathlib import Path

def _download_parquet():
    splits = {
        "train": "plain_text/train-00000-of-00001.parquet",
        "test": "plain_text/test-00000-of-00001.parquet",
    }

    schema = {
        'img': pa.struct([('bytes', pa.binary()), ('path', pa.string())]), 
        'label': pa.int64()
    }

    url = "hf://datasets/uoft-cs/cifar10/"
    ddf_train = dd.read_parquet(url + splits["train"])
    ddf_test = dd.read_parquet(url + splits["test"])
    ddf_train.to_parquet('./cifar10/content/cifar10/train', schema=schema)
    ddf_test.to_parquet('./cifar10/content/cifar10/test', schema=schema)


def load_datasets():
    if not Path('./cifar10/content/cifar10').is_dir():
        _download_parquet()


    ddf_train = dd.read_parquet('./cifar10/content/cifar10/train')
    ddf_test = dd.read_parquet('./cifar10/content/cifar10/test')

    @delayed
    def partitions_to_datasets(partition, train):
        return ImageDataset(partition["img"], partition["label"], train)


    train_datasets = [partitions_to_datasets(partition, True) for partition in ddf_train.to_delayed()]
    test_datasets = [partitions_to_datasets(partition, False) for partition in ddf_test.to_delayed()]

    # dataset fits in memory
    train_datasets = [partition.compute() for partition in train_datasets]
    test_datasets = [partition.compute() for partition in test_datasets]


    return train_datasets, test_datasets, ddf_train.npartitions