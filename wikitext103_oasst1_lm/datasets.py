from dask import dataframe as dd
from dask.delayed import delayed
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, IterableDataset

DIR_PATH = "/kaggle/working/content"


def _download_wikitext_parquets():
    splits = {
        "test": "wikitext-103-raw-v1/test-00000-of-00001.parquet",
        "train1": "wikitext-103-raw-v1/train-00000-of-00002.parquet",
        "train2": "wikitext-103-raw-v1/train-00001-of-00002.parquet",
        "validation": "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
    }

    paths = dict()
    for key, filename in splits.items():
        paths[key] = hf_hub_download(
            "Salesforce/wikitext",
            filename,
            repo_type="dataset",
            cache_dir=DIR_PATH,
        )

    return paths


def _download_oasst1_parquets():
    splits = {
        "train": "data/train-00000-of-00001-b42a775f407cee45.parquet",
        "validation": "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
    }

    paths = dict()
    for key, filename in splits.items():
        paths[key] = hf_hub_download(
            "OpenAssistant/oasst1", filename, repo_type="dataset", cache_dir=DIR_PATH
        )

    return paths


@delayed
def oasst1_load_part(partition):
    return partition[
        (partition["lang"] == "en") & (partition["tree_state"] == "ready_for_export")
    ][["text", "role", "message_tree_id", "message_id", "parent_id"]]


@delayed
def oasst1_make_conversation(partition):
    pass


@delayed
def wikitext_load_part(partition):
    return partition[partition["text"].str.len() > 0]


def load_tokenizer_datasets():
    wikitext_datasets = _download_wikitext_parquets()
    splits = ["train1", "train2"]
    wikitext_parts = [dd.read_parquet(wikitext_datasets[key]) for key in splits]
    wikitext_parts = [
        wikitext_load_part(partition)
        for df in wikitext_parts
        for partition in df.to_delayed()
    ]

    splits = ["train"]
    oasst1_datasets = _download_oasst1_parquets()
    oasst1_parts = [dd.read_parquet(oasst1_datasets[key]) for key in splits]
    oasst1_parts = [
        oasst1_load_part(partition)
        for df in oasst1_parts
        for partition in df.to_delayed()
    ]

    return wikitext_parts, oasst1_parts


def load_wikitext_datasets():
    # TODO: shuffle and remove empty sequences

    # part = (part != pad).to_numpy()[:, None, :]

    wikitext_datasets = _download_wikitext_parquets()

    train = (dd.read_parquet(wikitext_datasets[key]) for key in ["train1", "train2"])
    train = [part for df in train for part in df.to_delayed()]

    validation = (dd.read_parquet(wikitext_datasets[key]) for key in ["validation"])
    validation = [part for df in validation for part in df.to_delayed()]

    return train, validation


class WikitextDataset(Dataset):
    def __init__(self, pd_dfs):
        self.pd_series = [pd_df["text"] for pd_df in pd_dfs]
        self.total_len = sum(len(series) for series in self.pd_series)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        for series in self.pd_series:
            if idx < len(series):
                return series.iloc[idx]

            idx -= len(series)

        raise IndexError(f"Index {idx} out of range")


class WikitextIterDataset(IterableDataset):
    def __init__(self, parts):
        self.parts = parts

    def __iter__(self):
        for part in self.parts:
            df = part.compute()
            series = df["text"]
            for row in series:
                yield row
