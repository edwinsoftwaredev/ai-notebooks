import os

from torch_xla import step

from wikitext103_oasst1_lm.datasets import WikitextDataset, load_wikitext_datasets
from wikitext103_oasst1_lm.demo import wikitext_demo
import math

# Kaggle Config
if "TPU_PROCESS_ADDRESSES" in os.environ:
    os.environ.pop("TPU_PROCESS_ADDRESSES")

if "CLOUD_TPU_TASK_ID" in os.environ:
    os.environ.pop("CLOUD_TPU_TASK_ID")

if "LD_PRELOAD" in os.environ:
    os.environ.pop("LD_PRELOAD")

os.environ["PJRT_DEVICE"] = "TPU"
os.environ["PT_XLA_DEBUG"] = "0"
os.environ["XLA_USE_BF16"] = "0"

import ray
import sentencepiece as spm
from ray import tune
from ray.tune.schedulers import ASHAScheduler


sp = spm.SentencePieceProcessor()
sp.Load("/kaggle/working/ai-notebooks/m.model")

D_MODEL = 768
VOCAB_SIZE = sp.GetPieceSize()
N = 6
H = 12
DROPOUT = 0.1
D_FF = 3072  # Position-wise FFN params

GRAD_ACC_STEPS = 8
EPOCHS = 5
DS = 1.2e6
BATCH_SIZE = 8
DEVICE_COUNT = 8

DS_PER_DEVICE = DS // (BATCH_SIZE * DEVICE_COUNT)
STEPS = DS_PER_DEVICE * EPOCHS

config = {
    "model_config": {
        "transformer": {
            "d_model": D_MODEL,
            "vocab_size": VOCAB_SIZE,
            "encoder": {
                "N": N,
                "d_model": D_MODEL,
                "encoder_layer": {
                    "d_model": D_MODEL,
                    "dropout": DROPOUT,
                    "ffn": {"d_model": D_MODEL, "d_ff": D_FF, "dropout": DROPOUT},
                    "multihead_attn": {
                        "d_model": D_MODEL,
                        "h": H,
                        "dropout": DROPOUT,
                    },
                },
            },
            "decoder": {
                "N": N,
                "d_model": D_MODEL,
                "decoder_layer": {
                    "d_model": D_MODEL,
                    "masked_multihead_attn": {
                        "d_model": D_MODEL,
                        "h": H,
                        "dropout": DROPOUT,
                    },
                    "dropout": DROPOUT,
                    "multihead_attn": {
                        "d_model": D_MODEL,
                        "h": H,
                        "dropout": DROPOUT,
                    },
                    "ffn": {"d_model": D_MODEL, "d_ff": D_FF, "dropout": DROPOUT},
                },
            },
            "token_embedder": {"d_model": D_MODEL, "vocab_size": VOCAB_SIZE},
            "position_embedder": {
                "d_model": D_MODEL,
                "dropout": DROPOUT,
                "max_seq_len": 5000,
            },
        },
        "d_model": D_MODEL,
        "base_lr": 2.5e-4 * (DEVICE_COUNT**0.5),
        "beta1": 0.9,
        "beta2": 0.98,
        "optim_eps": 1e-6,
        "schdlr_factor": 1,
        "schdlr_warmup": (STEPS * 0.002) // GRAD_ACC_STEPS,
        "lbl_smoothing": 0.1,
        "grad_acc_steps": GRAD_ACC_STEPS,
        "max_steps": STEPS // GRAD_ACC_STEPS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
    "num_trials": 1,
}


def demo():
    wikitext_demo(config["model_config"], sp, "8qfphg91")


def train():
    from wikitext103_oasst1_lm.functions import tpu_train

    # ray.init(
    #     runtime_env={"working_dir": "/kaggle/working/ai-notebooks"},
    # )

    # scheduler = ASHAScheduler(
    #     time_attr="training_iteration",
    #     max_t=config["model_config"]["epochs"],
    #     grace_period=1,
    #     reduction_factor=2,
    # )

    # tuner = tune.Tuner(
    #     tune.with_resources(
    #         tune.with_parameters(tpu_train),
    #         resources={"CPU": 8, "TPU": 8, "accelerator_type:TPU-V5LITEPOD": 1},
    #     ),
    #     tune_config=tune.TuneConfig(
    #         metric="eval_loss",
    #         mode="min",
    #         scheduler=scheduler,
    #         num_samples=config["num_trials"],
    #     ),
    #     param_space=config["model_config"],
    #     run_config=tune.RunConfig(
    #         storage_path="/kaggle/working", name="wikitext103_oasst1_lm"
    #     ),
    # )

    tpu_train(config["model_config"])

    # results = tuner.fit()

    # best_result = results.get_best_result("eval_loss", "min")

    # print(f"Best trial config: {best_result.config}")
    # if best_result.metrics:
    #     print(
    #         f"Best trial final validation accuracy: {best_result.metrics['eval_loss']}"
    #     )

    # sentences = demo(best_result)
    # wandb.init(project="wikitext103_oasst1_lm", group="experiment_1")
    # wandb.log({"examples": [sentences]})
    # wandb.finish()
