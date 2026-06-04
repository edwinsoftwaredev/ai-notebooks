import sentencepiece as spm
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import wandb
from kaggle_secrets import UserSecretsClient  # pyright: ignore
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch_xla import runtime as xr

from wikitext103_oasst1_lm.causal_transformer import Transformer
from wikitext103_oasst1_lm.collate_batch import oasst_collate_batch
from wikitext103_oasst1_lm.datasets import OasstDataset, load_oasst1_datasets
from wikitext103_oasst1_lm.run import Run

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")

wandb.login(key=secret_value_0)

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("/kaggle/working/ai-notebooks/m.model")


def collate_fn(batch):
    return oasst_collate_batch(batch, tokenizer)


def tpu_train(config):
    train_set, validation_set = load_oasst1_datasets()

    # Do not compute partitions when using WikitextIterDataset
    train_set = [part.compute() for part in train_set]
    validation_set = [part.compute() for part in validation_set]

    train_set = OasstDataset(train_set)
    validation_set = OasstDataset(validation_set)

    args = {"config": config, "datasets": (train_set, validation_set)}

    return torch_xla.launch(train, args=(args,), debug_single_process=True)


def train(index, args):
    config = args["config"]
    device = torch_xla.device()
    model = Transformer(config["transformer"])
    checkpoint = torch.load("model_checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(device)

    xm.broadcast_master_param(model)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config["base_lr"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["optim_eps"],
        weight_decay=1e-2,  # 1e-2 default
    )

    lr_scheduler_1 = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1e-6, end_factor=1.0, total_iters=config["schdlr_warmup"]
    )

    lr_scheduler_2 = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=config["max_steps"] - config["schdlr_warmup"],
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[lr_scheduler_1, lr_scheduler_2],
        milestones=[config["schdlr_warmup"]],
    )

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=config["lbl_smoothing"], ignore_index=tokenizer.pad_id()
    )

    if xm.is_master_ordinal(local=False):
        wandb.init(project="wikitext103_oasst1_lm", group="fine_tune", config=config)

    run = Run(
        model,
        optim,
        lr_scheduler,
        loss_fn,
        config["batch_size"],
        config["grad_acc_steps"],
    )

    train_set, validation_set = args["datasets"]

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=True,
        drop_last=True,
    )

    validation_sampler = DistributedSampler(
        validation_set,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=True,
        drop_last=True,
    )

    train_dl = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=0,
        drop_last=True,
        sampler=train_sampler,
    )

    validation_dl = DataLoader(
        validation_set,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=0,
        drop_last=True,
        sampler=validation_sampler,
    )

    train_dl = pl.MpDeviceLoader(train_dl, device=device)
    validation_dl = pl.MpDeviceLoader(validation_dl, device=device)

    for epoch in range(config["epochs"]):
        train_sampler.set_epoch(epoch)
        run.train(train_dl, epoch)
        validation_sampler.set_epoch(epoch)
        run.eval(validation_dl, epoch)

        if xm.is_master_ordinal(local=False):
            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, "fine_tuned_model_checkpoint.pt")

    if xm.is_master_ordinal(local=False):
        print(f"Process {index}: Training finished!")
        wandb.finish()
