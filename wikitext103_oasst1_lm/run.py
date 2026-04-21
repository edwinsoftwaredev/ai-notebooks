import torch
import torch_xla.core.xla_model as xm
import wandb
from torch import nn
from torch_xla import runtime as xr
from torch_xla.distributed.parallel_loader import MpDeviceLoader

from wikitext103_oasst1_lm.transformer import Transformer


def hook_forward(module_name, grads, hook_backward):
    def hook(module, args, output):
        output.register_hook(hook_backward(module_name, grads))

    return hook


def hook_backward(module_name, grads):
    def hook(grad):
        grads.append((module_name, grad))

    return hook


def get_all_layers(model: nn.Module, hook_forward, hook_backward):
    layers = dict()
    grads = []
    for name, layer in model.named_modules():
        # skip Sequential and/or wrapper modules
        if any(layer.children()) is False:
            layers[layer] = name
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return layers, grads


def get_grads(grads):
    layer_idx = []
    avg_grads = []
    for idx, (name, grad) in enumerate(grads):
        if grad is not None:
            avg_grad = grad.abs().mean()
            avg_grads.append(avg_grad)
            layer_idx.append(len(grads) - 1 - idx)
    return layer_idx, avg_grads


class Run:
    def __init__(
        self,
        model: Transformer,
        optimizer,
        scheduler: torch.optim.lr_scheduler.SequentialLR,
        loss_fn,
        batch_size,
        grad_acc_steps,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.grad_acc_steps = grad_acc_steps

        # if xm.is_master_ordinal(local=False):
        #     layers, grads = get_all_layers(self.model, hook_forward, hook_backward)
        #     self.layers = layers
        #     self.grads = grads

    def loss(self, x, target):
        logits = self.model.generator(self.model(*x))
        # source: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # target: (batch_size, seq_len) -> (batch_size * seq_len)
        return self.loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))

    def backprop(self, loss, step, is_last_batch):
        loss.backward()

        if (step + 1) % self.grad_acc_steps == 0 or is_last_batch:
            xm.optimizer_step(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

    def train(self, dataloader: MpDeviceLoader, epoch):
        def log_loss(step, loss, loss_type, lr):
            wandb.log(
                {
                    f"{loss_type}": loss.item() * self.grad_acc_steps,
                    "step": step,
                    "lr": lr[0],
                }
            )

        n = len(dataloader)
        remainder = n % self.grad_acc_steps

        self.model.train()
        for step, batch in enumerate(dataloader):
            x = (
                batch.enc_in,
                batch.dec_in,
                batch.enc_pad_mask,
                batch.dec_pad_mask,
                batch.dec_causal_mask,
            )

            is_last_batch = step + 1 == n
            divisor = (
                self.grad_acc_steps
                if not (is_last_batch and remainder != 0)
                else remainder
            )

            loss = self.loss(x, batch.target)
            loss = loss / divisor

            self.backprop(loss, step, is_last_batch)

            if (step + 1) % 100 == 0 and xm.is_master_ordinal(local=False):
                # These operations require TPU and CPU to communicate
                xm.add_step_closure(
                    log_loss,
                    (step, loss, "train_batch_loss", self.scheduler.get_last_lr()),
                )

    def eval(self, dataloader: MpDeviceLoader, epoch):
        def log_loss(step, loss, loss_type):
            wandb.log({f"{loss_type}": loss.item(), "step": step})

        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                x = (
                    batch.enc_in,
                    batch.dec_in,
                    batch.enc_pad_mask,
                    batch.dec_pad_mask,
                    batch.dec_causal_mask,
                )

                loss = self.loss(x, batch.target)

                if step % 100 == 0 and xm.is_master_ordinal(local=False):
                    xm.add_step_closure(log_loss, (step, loss, "eval_batch_loss"))
