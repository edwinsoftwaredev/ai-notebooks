from math import ceil, log2

import sentencepiece as spm
import torch
from torch.nn.functional import pad


class Batch:
    def __init__(self, enc_in, dec_in, target, pad_id=3):
        device = "cpu"  # MPDeviceLoader will move batches to TPU

        # encoder input
        self.enc_in = enc_in
        # (batch_size, seq_len)
        self.enc_pad_mask = enc_in == pad_id

        # decoder input
        seq_len = dec_in.size(1)
        self.dec_in = dec_in
        self.dec_pad_mask = dec_in == pad_id

        # dec_in_pad_mask_queries = (dec_in == pad_id).unsqueeze(-2)
        # dec_in_pad_mask_keys = (dec_in == pad_id).unsqueeze(-1)
        # dec_in_pad_mask = dec_in_pad_mask_queries | dec_in_pad_mask_keys

        """

            dec_in: (batch_size, seq_len)
            dec_in_pad_mask_queries: (batch_size, 1, seq_len)
            dec_in_pad_mask_keys: (batch_size, seq_len, 1)
            causal_mask: (1, seq_len, seq_len)

        """

        ones = torch.ones((seq_len, seq_len), device=device)
        # causal_mask = torch.triu(ones, diagonal=1).unsqueeze(0).bool()
        causal_mask = torch.triu(ones, diagonal=1).bool()

        # dynamic shape: torch / xla recompilation trigger
        # (batch_size, seq_len, seq_len)
        self.dec_causal_mask = causal_mask

        # target
        self.target = target
        self.ntokens = (target != pad_id).sum()


def collate_batch(batch, tokenizer: spm.SentencePieceProcessor):
    device = "cpu"  # MPDeviceLoader will move batches to TPU

    enc_in, dec_in, target = [], [], []
    # pad_len = 8
    # max_seq_len = (
    #     tokenizer.Encode(seq, out_type=int, add_bos=True, add_eos=True) for seq in batch
    # )
    # max_seq_len = max(max_seq_len, key=len)
    # max_seq_len = len(max_seq_len) + pad_len
    # bucket = 2 ** ceil(log2(max_seq_len))  # power of 2 buckets: [128, 256, 512, 1024]

    bucket = 512  # fixed

    for seq in batch:
        encoded_seq = tokenizer.Encode(seq, out_type=int, add_bos=False, add_eos=False)
        encoded_seq = encoded_seq[:bucket]
        k = torch.randint(1, len(encoded_seq) + 1, (1,)).item()

        # PREFIX
        enc_seq = encoded_seq[:k]
        enc_seq = pad(
            torch.tensor(enc_seq, dtype=torch.long, device=device),
            (0, bucket - len(enc_seq)),
            value=tokenizer.pad_id(),
        )
        enc_in.append(enc_seq)

        # SUFFIX
        dec_seq = [tokenizer.bos_id()] + encoded_seq[k:]
        dec_seq = pad(
            torch.tensor(dec_seq, dtype=torch.long, device=device),
            (0, bucket - len(dec_seq)),
            value=tokenizer.pad_id(),
        )
        dec_in.append(dec_seq)

        # TARGET
        tgt = encoded_seq[k:] + [tokenizer.eos_id()]
        tgt = pad(
            torch.tensor(tgt, dtype=torch.long, device=device),
            (0, bucket - len(tgt)),
            value=tokenizer.pad_id(),
        )
        target.append(tgt)

    enc_in = torch.stack(enc_in)
    dec_in = torch.stack(dec_in)
    target = torch.stack(target)

    return Batch(enc_in, dec_in, target, tokenizer.pad_id())
