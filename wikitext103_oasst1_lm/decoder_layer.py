from torch import nn

from wikitext103_oasst1_lm.multihead_attention import MultiHeadAttention
from wikitext103_oasst1_lm.position_wise_ffn import PositionWiseFFN


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Masked Multi-Head Attention sublayer
        self.masked_self_attn_ln = nn.LayerNorm(config["d_model"], eps=1e-6)
        self.masked_self_attn = MultiHeadAttention(config["masked_multihead_attn"])
        self.masked_self_attn_dropout = nn.Dropout(p=config["dropout"])

        # Multi-Head Attention sublayer (cross-attention encoder-decoder)
        self.cross_attn_ln = nn.LayerNorm(config["d_model"], eps=1e-6)
        self.cross_attn = MultiHeadAttention(config["multihead_attn"])
        self.cross_attn_dropout = nn.Dropout(p=config["dropout"])

        # Position-Wise FFN sublayer
        self.ffn_ln = nn.LayerNorm(config["d_model"], eps=1e-6)
        self.ffn = PositionWiseFFN(config["ffn"])
        self.ffn_dropout = nn.Dropout(p=config["dropout"])

    def forward(self, x, encoder_out, causal_mask):
        residual_connection = x

        # Pre-LN masked multi-head attention
        x = self.masked_self_attn_ln(x)
        x = self.masked_self_attn(x, x, x, causal_mask)
        x = self.masked_self_attn_dropout(x)

        x += residual_connection

        residual_connection = x

        # Pre-LN multi-head attention
        x = self.cross_attn_ln(x)
        x = self.cross_attn(x, encoder_out, encoder_out, causal_mask)
        x = self.cross_attn_dropout(x)

        x += residual_connection

        residual_connection = x

        # Pre-LN position wise FFN
        x = self.ffn_ln(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)

        # This last residual addition is not normalized
        x += residual_connection

        return x
