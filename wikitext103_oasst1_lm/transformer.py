from torch import nn

from wikitext103_oasst1_lm.decoder_layer import DecoderLayer
from wikitext103_oasst1_lm.encoder_layer import EncoderLayer
from wikitext103_oasst1_lm.position_embedder import PositionalEncoding
from wikitext103_oasst1_lm.token_embedder import TokenEmbedder


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(config["encoder_layer"]) for _ in range(config["N"])],
        )

        # This layer norm normalizes the last residual addition
        # in the encoder_layer
        self.norm = nn.LayerNorm(config["d_model"], eps=1e-6)

    def forward(self, x, padding_mask):
        for layer in self.layers:
            x = layer(x, padding_mask)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(config["decoder_layer"]) for _ in range(config["N"])]
        )

        # This layer norm normalizes the last residual addition
        # in the decoder_layer
        self.norm = nn.LayerNorm(config["d_model"], eps=1e-6)

    def forward(self, x, encoder_out, dec_in_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, dec_in_mask)

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config["encoder"])
        self.decoder = Decoder(config["decoder"])

        # Removed nn.LogSoftMax and output just logits (CrossEntropyLoss)
        self.generator = nn.Sequential(
            nn.Linear(config["d_model"], config["vocab_size"])
        )

        # No Weight Sharing
        self.encoder_input = nn.Sequential(
            TokenEmbedder(config["token_embedder"]),
            PositionalEncoding(config["position_embedder"]),
        )
        self.decoder_input = nn.Sequential(
            TokenEmbedder(config["token_embedder"]),
            PositionalEncoding(config["position_embedder"]),
        )

        self._initialize_parameters()

    def forward(self, enc_in, dec_in, enc_in_mask, dec_in_mask):
        return self.decode(self.encode(enc_in, enc_in_mask), dec_in, dec_in_mask)

    def encode(self, source, enc_in_mask):
        return self.encoder(self.encoder_input(source), enc_in_mask)

    def decode(self, enc_out, dec_in, dec_in_mask):
        return self.decoder(self.decoder_input(dec_in), enc_out, dec_in_mask)

    def _initialize_parameters(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.generator.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
