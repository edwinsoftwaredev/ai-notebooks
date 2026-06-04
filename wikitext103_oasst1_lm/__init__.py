import os

# Kaggle Config
if "TPU_PROCESS_ADDRESSES" in os.environ:
    os.environ.pop("TPU_PROCESS_ADDRESSES")

if "CLOUD_TPU_TASK_ID" in os.environ:
    os.environ.pop("CLOUD_TPU_TASK_ID")

if "LD_PRELOAD" in os.environ:
    os.environ.pop("LD_PRELOAD")

os.environ["X_NUM_DEVICES"] = "8"
os.environ["TPU_NUM_DEVICES"] = "8"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["PT_XLA_DEBUG"] = "0"
os.environ["XLA_USE_BF16"] = "1"

import sentencepiece as spm

from wikitext103_oasst1_lm.demo import oasst_demo, wikitext_demo

sp = spm.SentencePieceProcessor()
sp.Load("/kaggle/working/ai-notebooks/m.model")

D_MODEL = 1024 + 256
VOCAB_SIZE = sp.GetPieceSize()
N = 24
H = 16
DROPOUT = 0.1
D_FF = 4 * D_MODEL  # Position-wise FFN params
GRAD_ACC_STEPS = 6
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
        "base_lr": 5e-4,
        "beta1": 0.9,
        "beta2": 0.98,
        "optim_eps": 1e-6,
        "schdlr_factor": 1,
        "schdlr_warmup": (STEPS * 0.05) // GRAD_ACC_STEPS,
        "lbl_smoothing": 0.075,
        "grad_acc_steps": GRAD_ACC_STEPS,
        "max_steps": STEPS // GRAD_ACC_STEPS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
    "num_trials": 1,
}


def demo():
    wikitext_demo(config["model_config"], sp, "3w6rgve0")


def task_demo():
    oasst_demo(config["model_config"], sp, "4fyjun9s")


def train():
    from wikitext103_oasst1_lm.functions import tpu_train

    tpu_train(config["model_config"])


def tune():
    from wikitext103_oasst1_lm.fine_tuning_functions import tpu_train

    DEVICE_COUNT = 1
    BATCH_SIZE = 8
    GRAD_ACC_STEPS = 4
    DS = 22500
    EPOCHS = 20
    DS_PER_DEVICE = DS // (BATCH_SIZE * DEVICE_COUNT)
    STEPS = DS_PER_DEVICE * EPOCHS

    config["model_config"]["base_lr"] = 1e-5
    config["model_config"]["schdlr_warmup"] = (STEPS * 0.25) // GRAD_ACC_STEPS
    config["model_config"]["max_steps"] = STEPS // GRAD_ACC_STEPS
    config["model_config"]["epochs"] = EPOCHS
    config["model_config"]["grad_acc_steps"] = GRAD_ACC_STEPS
    config["model_config"]["batch_size"] = BATCH_SIZE

    tpu_train(config["model_config"])
