from torch import nn
from attention import scaled_dot_product_attention
from funcs import get_layers


class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        attn_head_conf = config["attn_head"]

        # shape: (d_model, d_q)
        self.Wq = nn.Sequential(*get_layers(attn_head_conf["query_params"]))
        # shape: (d_model, d_k)
        self.Wk = nn.Sequential(*get_layers(attn_head_conf["key_params"]))
        # shape: (d_model, d_v)
        self.Wv = nn.Sequential(*get_layers(attn_head_conf["value_params"]))

        # TODO: initialize weights

    def forward(self, x):
        # input (token embeddings + positional encodings)
        # input shape: (batch_len, seq_len, d_model)

        # TODO: Add Mask

        # TODO: Do all linear projections in batch

        # out shape: (batch_len, seq_len, d_v)
        return scaled_dot_product_attention(self.Wq(x), self.Wk(x), self.Wv(x))
