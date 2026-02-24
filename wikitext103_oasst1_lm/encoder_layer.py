from torch import nn
from wikitext103_oasst1_lm.multihead_attention import MultiHeadAttention
from wikitext103_oasst1_lm.position_wise_ffn import PositionWiseFFN


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Multi-Head Attention sublayer
        self.self_attn_ln = nn.LayerNorm(config["d_model"], eps=1e-6)
        self.self_attn = MultiHeadAttention(config["multihead_attn"])
        self.self_attn_dropout = nn.Dropout(p=config["dropout"])

        # Position-wise FFN sublayer
        self.ffn_ln = nn.LayerNorm(config["d_model"], eps=1e-6)
        self.ffn = PositionWiseFFN(config["ffn"])
        self.ffn_dropout = nn.Dropout(p=config["dropout"])

    def forward(self, x, mask):
        residual_connection = x

        # Pre-LN multi-head attention
        x = self.self_attn_ln(x)
        x = self.self_attn(x, x, x, mask)
        x = self.self_attn_dropout(x)

        x += residual_connection

        residual_connection = x

        # Pre-LN position wise FFN
        x = self.ffn_ln(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)

        # This last residual addition is not normalized
        x += residual_connection

        return x
