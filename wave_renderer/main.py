import wandb
import dask.dataframe as ddf
from huggingface_hub import hf_hub_download

def _download_dataset():
    splits = {
        'test.clean': 'all/test.clean/0000.parquet',
        'test.other': 'all/test.other/0000.parquet',
        'train.clean.100': 'all/train.clean.100/*.parquet',
        'train.clean.360': 'all/train.clean.360/*.parquet',
        'train.other.500': 'all/train.other.500/*.parquet',
        'validation.clean': 'all/validation.clean/0000.parquet',
        'validation.other': 'all/validation.other/0000.parquet'
    }

    path = hf_hub_download(
        repo_id="openslr/librispeech_asr",
        filename=splits["test.clean"],
        repo_type="dataset",
        local_dir="./"
    )

    df = ddf.read_parquet(path)
    df = df.head(10)
    df.to_parquet(
        "./dataset",
        engine="pyarrow"
    )

def read_dataset():
    df = ddf.read_parquet("./dataset")
    print(df["audio"].loc[0].compute())

def plot_frame():
    import os
    from wave_renderer.decoder import get_frame
    import numpy as np

    secret_value_0 = os.getenv("WAVE_RENDERER_WANDB_API_KEY")
    wandb.login(key=secret_value_0)

    SAMPLE_RATE = 16000
    BLOCK_SIZE = 4096
    time_1 = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    
    frame = get_frame()
    
    wandb.init(
        project="wave_renderer",
        group="experiment_1",
        resume="allow",
    )

    table = wandb.Table(
        data=np.column_stack((time_1, frame)),
        columns=["time", "amplitude"]
    )

    wandb.log({
        "audio_1": wandb.plot.line(
            table,
            x="time",
            y="amplitude",
            title="Audio_1 wave",
        )
    })

plot_frame()
