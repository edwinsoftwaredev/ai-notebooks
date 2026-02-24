import torch
import torch_xla.core.xla_model as xm
import wandb
from torch_xla.distributed.parallel_loader import MpDeviceLoader

from wikitext103_oasst1_lm.transformer import Transformer


class Run:
    def __init__(
        self,
        model: Transformer,
        optimizer,
        scheduler: torch.optim.lr_scheduler.SequentialLR,
        loss_fn,
        batch_size,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.loss_fn = loss_fn

    def loss(self, x, target):
        logits = self.model.generator(self.model(*x))
        # source: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # target: (batch_size, seq_len) -> (batch_size * seq_len)
        return self.loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))

    def backprop(self, loss):
        loss.backward()
        # TODO: accumulate gradients before optimizer step
        xm.optimizer_step(self.optimizer)

    def train(self, dataloader: MpDeviceLoader, epoch):
        def log_loss(step, loss, loss_type, lr):
            wandb.log({f"{loss_type}": loss.item(), "step": step, "lr": lr[0]})

        self.model.train()
        for step, batch in enumerate(dataloader):
            self.optimizer.zero_grad(set_to_none=True)
            x = (batch.enc_in, batch.dec_in, batch.enc_in_mask, batch.dec_in_mask)
            loss = self.loss(x, batch.target)
            self.backprop(loss)
            self.scheduler.step()

            if step % 100 == 0 and xm.is_master_ordinal(local=False):
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
                x = (batch.enc_in, batch.dec_in, batch.enc_in_mask, batch.dec_in_mask)
                loss = self.loss(x, batch.target)

                if step % 100 == 0 and xm.is_master_ordinal(local=False):
                    xm.add_step_closure(log_loss, (step, loss, "eval_batch_loss"))
