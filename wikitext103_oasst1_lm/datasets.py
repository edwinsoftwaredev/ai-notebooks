import re
from typing import cast

import pandas as pd
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
def oasst1_make_qa(partition: pd.DataFrame):
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load("/kaggle/working/ai-notebooks/m.model")

    left = pd.pivot(
        partition, index=["parent_id", "message_id"], columns="role", values="text"
    )
    right = pd.pivot(
        cast(pd.DataFrame, partition[["message_id", "text", "role"]]),
        index="message_id",
        columns="role",
        values="text",
    )
    df = left.join(right, on="parent_id", how="inner", rsuffix="_r")
    df["prompter"] = cast(pd.Series, df["prompter"]).combine_first(df["prompter_r"])
    df.drop(columns=["assistant_r", "prompter_r"], inplace=True)
    df = df[df["assistant"].notna()]
    df.reset_index(drop=True, inplace=True)

    df["prompter"] = df["prompter"].apply(lambda x: tokenizer.Encode(x, out_type=int))  # pyright: ignore
    df["assistant"] = df["assistant"].apply(lambda x: tokenizer.Encode(x, out_type=int))  # pyright: ignore

    df = df[(df["prompter"].apply(len) + df["assistant"].apply(len)) <= 512]  # pyright: ignore

    return df


def clean_wikitext(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Fix hyphens
    text = text.replace(" @-@ ", "-")

    # Fix quantities
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")
    text = re.sub(r"([$€£]+)\s+([0-9])", r"\1\2", text)

    # Normalize quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    # Remove excess spaces
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces

    # possessive w's, w 's, w' s, w ' s, w 'w
    text = re.sub(r"(\w+)\s+('\w+)", r"\1\2", text)  # contraction
    text = re.sub(r"(\w+)\s+('\s*s)\s+", r"\1's ", text)
    text = re.sub(r"(\w+?s)\s+'\s+", r"\1' ", text)
    text = re.sub(r"(\w+')\s+s\s+", r"\1s ", text)

    # punctuation
    text = re.sub(r"\s+([,.!?;:])\s+", r"\1 ", text)  # remove space before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # "etc. , " and final period case
    text = re.sub(r"(\d)\s*:\s*(\d)", r"\1:\2", text)  # time
    text = re.sub(r"(\d)\s*%", r"\1%", text)

    # enclosings (this does not handle '')
    text = re.sub(r"\(\s*(.+?)\s*\)", r"(\1)", text)
    text = re.sub(r'"\s*(.+?)\s*"', r'"\1"', text)
    text = re.sub(r"\[\s*(.+?)\s*\]", r"[\1]", text)

    text = re.sub(r"(^|\s+)'\s+(.+?)\s+'(\s+|$)", r"\1'\2'\3", text)

    # Remove headers markups
    text = re.sub(r"=(\s*=)*(.+?)=(\s*=)*", r"\2", text)

    # Remove wiki markup
    text = text.replace(" / ", "/")
    text = re.sub(r"\[\d+\]", "", text)  # numeric references
    text = re.sub(r"<ref>.*?</ref>", "", text, flags=re.DOTALL)  # ref tags

    return text.strip()


@delayed
def wikitext_load_part(partition):
    partition["text"] = partition["text"].map(clean_wikitext)
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
    train = [wikitext_load_part(part) for df in train for part in df.to_delayed()]

    validation = (dd.read_parquet(wikitext_datasets[key]) for key in ["validation"])
    validation = [
        wikitext_load_part(part) for df in validation for part in df.to_delayed()
    ]

    return train, validation


def load_oasst1_datasets():
    oasst1_datasets = _download_oasst1_parquets()
    train = (dd.read_parquet(oasst1_datasets[key]) for key in ["train"])
    train = (oasst1_load_part(df) for ddf in train for df in ddf.to_delayed())
    train = [oasst1_make_qa(df) for df in train]

    validation = (dd.read_parquet(oasst1_datasets[key]) for key in ["validation"])
    validation = (oasst1_load_part(df) for ddf in validation for df in ddf.to_delayed())
    validation = [oasst1_make_qa(df) for df in validation]

    return train, validation


class OasstDataset(Dataset):
    def __init__(self, dfs):
        self.pd_series = [df for df in dfs]
        self.total_len = sum(len(series) for series in self.pd_series)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        for series in self.pd_series:
            if idx < len(series):
                return series.iloc[idx]

            idx -= len(series)

        raise IndexError(f"Index {idx} out of range")


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
