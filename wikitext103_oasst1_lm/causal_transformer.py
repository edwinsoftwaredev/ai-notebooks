import math

from torch import nn

from wikitext103_oasst1_lm.causal_decoder_layer import DecoderLayer
from wikitext103_oasst1_lm.position_embedder import PositionalEncoding
from wikitext103_oasst1_lm.token_embedder import TokenEmbedder


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(config["decoder_layer"]) for _ in range(config["N"])]
        )

        # This layer norm normalizes the last residual addition
        # in the decoder_layer
        self.norm = nn.LayerNorm(config["d_model"], eps=1e-5)

    def forward(self, x, pad_mask, causal_mask):
        for layer in self.layers:
            x = layer(x, pad_mask, causal_mask)

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.N = config["decoder"]["N"]

        self.decoder = Decoder(config["decoder"])

        # Removed nn.LogSoftMax and output just logits (CrossEntropyLoss)
        self.generator = nn.Linear(
            in_features=config["d_model"],
            out_features=config["vocab_size"],
            bias=False,
        )

        self.decoder_input = nn.Sequential(
            TokenEmbedder(config["token_embedder"]),
            PositionalEncoding(config["position_embedder"]),
        )

        self.apply(self._init_params)

        # generator shares input embeddings weight: weight tying
        # self.generator.weight = self.decoder_input[0].embeddings.weight  # pyright: ignore

    def forward(self, x, pad_mask, causal_mask):
        return self.decoder(self.decoder_input(x), pad_mask, causal_mask)

    def _init_params(self, module):
        "GPT init"
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02

            if hasattr(module, "_is_residual"):
                "GPT-2 residual scaling"
                std *= 1.0 / math.sqrt(self.N)  # std *= 1.0 / math.sqrt(2 * self.N)

            module.weight.data.normal_(mean=0.0, std=std)

            # zero-out biases
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
