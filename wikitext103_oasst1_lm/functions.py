import os
import tempfile

import sentencepiece as spm
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import wandb
from kaggle_secrets import UserSecretsClient  # pyright: ignore
from ray import train, tune
from ray.tune.search import sample
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch_xla import runtime as xr

from wikitext103_oasst1_lm.collate_batch import collate_batch
from wikitext103_oasst1_lm.datasets import (
    WikitextDataset,
    WikitextIterDataset,
    load_wikitext_datasets,
)
from wikitext103_oasst1_lm.run import Run
from wikitext103_oasst1_lm.transformer import Transformer

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")

wandb.login(key=secret_value_0)


# def raytune_load_checkpoint(
#     model: nn.Module, optim: torch.optim.Adam, schdlr: torch.optim.lr_scheduler.LambdaLR
# ):
#     if tune.get_checkpoint():
#         loaded_checkpoint = tune.get_checkpoint()
#         with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
#             data = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
#             model.load_state_dict(data["model"])
#             optim.load_state_dict(data["optim"])
#             schdlr.load_state_dict(data["schdlr"])
#
#
# def raytune_save_checkpoint(
#     model: nn.Module,
#     optim: torch.optim.Adam,
#     schdlr: torch.optim.lr_scheduler.LambdaLR,
#     train_metrics,
#     test_metric,
# ):
#     with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
#         path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
#         xm.save(
#             {
#                 "model": model.state_dict(),
#                 "optim": optim.state_dict(),
#                 "schdlr": schdlr.state_dict(),
#             },
#             path,
#         )
#         checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
#         tune.report({**train_metrics, **test_metric}, checkpoint=checkpoint)


tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("/kaggle/working/ai-notebooks/m.model")


def collate_fn(batch):
    return collate_batch(batch, tokenizer)


def tpu_train(config):

    train_set, validation_set = load_wikitext_datasets()

    # Do not compute partitions when using WikitextIterDataset
    train_set = [part.compute() for part in train_set]
    validation_set = [part.compute() for part in validation_set]

    train_set = WikitextDataset(train_set)
    validation_set = WikitextDataset(validation_set)

    args = {"config": config, "datasets": (train_set, validation_set)}

    return torch_xla.launch(train_model, args=(args,))


def train_model(index, args):

    config = args["config"]

    device = torch_xla.device()
    model = Transformer(config["transformer"])
    model.to(device)

    xm.broadcast_master_param(model)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config["base_lr"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["optim_eps"],
        weight_decay=1e-2,  # 1e-2 default
    )

    # def lrate(step_num, d_model, factor, warmup_steps):
    #     if step_num == 0:
    #         return 0

    #     return factor * (
    #         d_model ** (-0.5)
    #         * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    #     )

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer=optim,
    #     lr_lambda=lambda step_num: lrate(
    #         step_num,
    #         config["d_model"],
    #         factor=config["schdlr_factor"],
    #         warmup_steps=config["schdlr_warmup"],
    #     ),
    # )

    lr_scheduler_1 = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1e-3, end_factor=1.0, total_iters=config["schdlr_warmup"]
    )

    # substract warmup from max_steps (fix offset)
    lr_scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=config["max_steps"] - config["schdlr_warmup"], eta_min=0.0
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
        wandb.init(project="wikitext103_oasst1_lm", group="experiment_1", config=config)
        wandb.watch(model, log="gradients")

    # if xm.is_master_ordinal(local=False):
    #     raytune_load_checkpoint(model, optim, lr_scheduler)

    run = Run(model, optim, lr_scheduler, loss_fn, config["batch_size"])

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
            torch.save(cpu_state_dict, "model_checkpoint.pt")

        # if xm.is_master_ordinal(local=False):
        #     raytune_save_checkpoint(
        #         model,
        #         optim,
        #         lr_scheduler,
        #         {"train_loss": train_loss, "epoch": epoch},
        #         {"eval_loss": validation_loss, "epoch": epoch},
        #     )

    if xm.is_master_ordinal(local=False):
        print(f"Process {index}: Training finished!")
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state_dict, "model_checkpoint.pt")
        # xm.save(model.state_dict(), "model_checkpoint.pt")

    wandb.finish()
