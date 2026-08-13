import torch
from torch.utils.data import DataLoader, Dataset


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def queries_to_embeddings(input, tokenizer, embedder):
    max_len = 512
    stride = int(max_len * 0.35)  # search and store chunks independently
    stride = 0
    inputs = tokenizer(
        input,
        is_split_into_words=False,
        max_length=max_len,
        stride=stride,
        padding="max_length",
        truncation=False,
        return_overflowing_tokens=False,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    device = next(embedder.parameters()).device
    with (
        torch.inference_mode(),
        torch.amp.autocast(device_type=device.type, dtype=torch.float16),  # pyright: ignore
    ):
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        outputs = embedder(**inputs)
        embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
        del outputs

    return embeddings.float().cpu().numpy()


class DocsDataset(Dataset):
    def __init__(self, docs):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]


def docs_to_embeddings(input, tokenizer, embedder):
    def collate_fn(batch, tokenizer):
        max_len = 512
        stride = int(max_len * 0.35)
        return tokenizer(
            batch,
            is_split_into_words=True,
            max_length=max_len,
            stride=stride,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    ds = DocsDataset(input)
    coll_fn = lambda batch: collate_fn(batch, tokenizer)
    dl = DataLoader(
        ds,
        8,
        num_workers=0,  # FIX:
        collate_fn=coll_fn,
        pin_memory=True,
        persistent_workers=False,
    )

    passages, input_ids = [], []
    device = next(embedder.parameters()).device
    with (
        torch.inference_mode(),
        torch.amp.autocast(device_type=device.type, dtype=torch.float16),  # pyright: ignore
    ):
        for i, inputs in enumerate(dl):
            inputs.pop("overflow_to_sample_mapping")
            ids = inputs["input_ids"]
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            outputs = embedder(**inputs)
            embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
            passages.append(embeddings.cpu().float())
            input_ids.append(ids)
            del outputs, inputs, embeddings

    passages = torch.cat(passages).numpy()
    input_ids = torch.cat(input_ids)

    return passages, input_ids
