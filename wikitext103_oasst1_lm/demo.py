import torch
import wandb
from kaggle_secrets import UserSecretsClient  # pyright: ignore
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader

from wikitext103_oasst1_lm.datasets import WikitextDataset, load_wikitext_datasets
from wikitext103_oasst1_lm.transformer import Transformer

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")

wandb.login(key=secret_value_0)


def collate_batch(batch, tokenizer: SentencePieceProcessor):
    device = "cuda"
    src = []
    for seq in batch:
        seq = tokenizer.Encode(seq, out_type=int, add_bos=False, add_eos=False)

        if not 5 < len(seq) < 512:
            continue

        seq = torch.tensor(
            seq,
            dtype=torch.long,
            device=device,
        )

        src.append(seq)

    return src


def collate_fn(tokenizer):

    def collate(batch):
        return collate_batch(batch, tokenizer)

    return collate


def top_k_filter(logits, k=50):
    logits = logits.clone()
    values, _ = torch.topk(logits, k)
    min_val = values[:, -1].unsqueeze(-1)
    logits[logits < min_val] = -float("inf")
    return logits


def top_p_sampling(logits, top_p=0.9, top_k=50, temperature=0.7):
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = probs.squeeze(0)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff index
    cutoff_idx = torch.searchsorted(cumulative_probs, top_p)

    # Build mask
    indices = torch.arange(sorted_probs.size(0), device=logits.device)
    cutoff_mask = indices > cutoff_idx

    sorted_probs[cutoff_mask] = 0.0

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices[next_token]

    return next_token.unsqueeze(0)


def repetition_penalty(logits, generated, penalty=1.2):
    logits = logits.clone()

    for token in set(generated.tolist()):
        if logits[0, token] > 0:
            logits[0, token] /= penalty
        else:
            logits[0, token] *= penalty

    return logits


def wikitext_demo(config, tokenizer: SentencePieceProcessor, run_id):
    wandb.init(
        project="wikitext103_oasst1_lm",
        group="experiment_1",
        config=config,
        resume="allow",
        id=run_id,
    )

    device = "cuda"
    model = Transformer(config["transformer"])
    checkpoint = torch.load("model_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    _, valid_set = load_wikitext_datasets()

    valid_ds = WikitextDataset((part.compute() for part in valid_set))
    valid_dl = DataLoader(valid_ds, 32, True, collate_fn=collate_fn(tokenizer))

    # wandb table
    data = []

    max_seq_len = 512
    for step, batch in enumerate(valid_dl):
        if step == 3:
            break

        for src in batch:
            # encoder input shape: (batch_size, seq_len)
            # last 50% tokens
            prefix = torch.cat(
                (
                    src[: -(len(src) // 2)],
                    torch.tensor([tokenizer.eos_id()], device=device, dtype=torch.long),
                )
            )
            prefix = prefix.unsqueeze(0)
            suffix = torch.tensor([tokenizer.bos_id()], device=device, dtype=torch.long)
            suffix = suffix.unsqueeze(0)

            enc_out = model.encode(prefix, None)

            with torch.no_grad():
                for _ in range(max_seq_len):
                    seq_len = suffix.size(-1)
                    ones = torch.ones((seq_len, seq_len), device=device)
                    causal_mask = torch.tril(ones, diagonal=0).bool()
                    causal_mask = causal_mask.unsqueeze(0)  # (1, seq_len, seq_len)

                    dec_out = model.decode(enc_out, suffix, None, causal_mask)[:, -1, :]
                    logits = model.generator(dec_out)  # (1, vocab_size)
                    # next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    logits = repetition_penalty(logits, suffix[0], 1.2)
                    next_token = top_p_sampling(logits)
                    suffix = torch.cat((suffix, next_token), dim=1)

                    if next_token.item() == tokenizer.eos_id():
                        break

            seq = torch.cat((prefix, suffix), dim=1)
            data.append(tokenizer.Decode([seq.squeeze(0).tolist(), src.tolist()]))

    wandb.log({"validation": wandb.Table(columns=["Output", "Target"], data=data)})

    wandb.finish()
