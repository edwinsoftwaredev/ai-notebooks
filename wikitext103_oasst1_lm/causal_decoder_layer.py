from torch import nn

from wikitext103_oasst1_lm.position_wise_ffn import PositionWiseFFN


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Masked Multi-Head Attention sublayer
        mha_conf = config["masked_multihead_attn"]
        self.masked_self_attn_ln = nn.LayerNorm(config["d_model"], eps=1e-5)
        self.masked_self_attn = nn.MultiheadAttention(
            embed_dim=mha_conf["d_model"],
            num_heads=mha_conf["h"],
            dropout=mha_conf["dropout"],
            batch_first=True,
        )
        # the last linear layer in the MHA: Wo
        self.masked_self_attn.out_proj._is_residual = True  # pyright: ignore

        # Position-wise FFN layer
        self.ffn_ln = nn.LayerNorm(config["d_model"], eps=1e-5)
        self.ffn = PositionWiseFFN(config["ffn"])
        self.ffn_dropout = nn.Dropout(p=config["dropout"])

    def forward(self, x, pad_mask, causal_mask):
        residual_connection = x

        # Pre-LN masked multihead attention
        x = self.masked_self_attn_ln(x)
        x = self.masked_self_attn(
            x,
            x,
            x,
            attn_mask=causal_mask,
            key_padding_mask=pad_mask,
            is_causal=True,
            need_weights=False,
        )
        x = x[0]

        x += residual_connection

        residual_connection = x

        # Pre-LN position wise FFN
        x = self.ffn_ln(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)

        # This last residual addition is not normalized
        x += residual_connection

        return x
