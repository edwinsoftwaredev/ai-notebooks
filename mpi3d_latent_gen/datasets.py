import os
import gdown
from pathlib import Path
import numpy as np

from mpi3d_latent_gen.image_dataset import ImageDataset

from dask import dataframe as dd
from dask import array as dda
from dask.delayed import delayed

from random import randint

dir_path = "./content/MPI3D"

import os
import gdown
from pathlib import Path
import numpy as np

from dask import dataframe as dd
from dask import array as dda
from dask.delayed import delayed

from random import randint

dir_path = "/kaggle/working/content/MPI3D"

# WARN: Load dataset outside Ray's context
def _download_dataset():
    if Path(dir_path).is_dir():
        return

    os.makedirs(dir_path, exist_ok=True)
    
    URL = f'https://drive.google.com/uc?id=1Tp8eTdHxgUMtsZv5uAoYAbJR1BOa_OQm'
    
    filename = f"{dir_path}/real3d_complicated_shapes_ordered.npz"
    
    gdown.download(URL, filename, quiet=False)

    data = np.load(filename, allow_pickle=False)
    data = data['images'] # [N, H, W, C]
    
    da = dda.from_array(data, chunks=(50000, 64, 64, 3))

    da = da.reshape((da.shape[0], -1))
    
    columns = [str(i) for i in range(da.shape[1])]
    
    ddf = dd.from_dask_array(da, columns=columns)
    
    def add_random_col(df):
        seed = randint(100, 200)
        rng = np.random.RandomState(seed)
        return df.assign(_rand=rng.rand(len(df)))
        
    # Assign a random column in each partition with a different seed
    ddf = ddf.map_partitions(add_random_col)

    ddf = ddf.shuffle(ddf.columns[-1], ignore_index=True)

    ddf = ddf.drop(columns=['_rand'])

    ddf.to_parquet(f"{dir_path}/", engine="pyarrow", write_index=False)

    os.remove(filename)


def load_datasets():

    @delayed
    def partitions_to_datasets(partition, train):
        partition = partition.to_numpy()
        partition = partition.reshape((partition.shape[0], 64, 64, 3))
        return ImageDataset(partition, train)
    

    ddf = dd.read_parquet(dir_path)

    ddf, _ = ddf.random_split([0.3, 0.7], random_state=124, shuffle=False)

    train_set, test_set = ddf.random_split([0.8, 0.2], random_state=123, shuffle=True)

    train_datasets = [partitions_to_datasets(partition, True) for partition in train_set.to_delayed()]
    test_datasets = [partitions_to_datasets(partition, False) for partition in test_set.to_delayed()]

    # dataset fits in memory
    train_datasets = [partition.compute() for partition in train_datasets]
    test_datasets = [partition.compute() for partition in test_datasets]

    return train_datasets, test_datasets, ddf.npartitions


'''

    object_color:       white=0, green=1, red=2, blue=3, brown=4, olive=5
    object_shape:       cone=0, cube=1, cylinder=2, hexagonal=3, pyramid=4, sphere=5
    object_size:	    small=0, large=1
    camera_height:	    top=0, center=1, bottom=2
    background_color:	purple=0, sea green=1, salmon=2
    horizontal_axis:	0,...,39
    vertical_axis:      0,...,39

    [6,6,2,3,3,40,40,64,64,3]

    6 * 6 * 2 * 3 * 3 * 40 * 40 = 1036800 images

    (1036800 / 6) = 172800

'''