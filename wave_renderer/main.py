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
    from wave_renderer.decoder import FLACDatasetIter
    import numpy as np

    secret_value_0 = os.getenv("WAVE_RENDERER_WANDB_API_KEY")
    wandb.login(key=secret_value_0)

    dataset_iter = FLACDatasetIter()
    audio_wave = dataset_iter.next_batch()[0]

    SAMPLE_RATE = 16000
    SAMPLES = audio_wave.size
    time_axis = np.arange(SAMPLES) / SAMPLE_RATE

    wandb.init(
        project="wave_renderer",
        group="experiment_1",
    )

    table = wandb.Table(
        data=np.column_stack((time_axis, audio_wave)),
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
