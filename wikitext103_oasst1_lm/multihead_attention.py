from torch import nn

from wikitext103_oasst1_lm.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config["d_model"]
        self.h = config["h"]

        assert d_model % self.h == 0

        # d_model == d_k * h
        self.d_k = d_model // self.h

        # Wq, Wk, Wv, Wo
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(4)]
        )

        self.dropout = nn.Dropout(p=config["dropout"])

    def forward(self, Q, K, V, mask=None):
        # input shape Q==K==V: (batch_len, seq_len, d_model)
        # input (token embeddings + positional encodings)

        if mask is not None:
            # shape: (1, seq_len, seq_len) -> (1, 1(h), seq_len, seq_len)
            mask = mask.unsqueeze(1)

        # Each linear layer (Wq, Wk, Wv, Wo)
        # already contains/concatenates
        # all h heads (h, d_k). The output of each linear
        # layer is (batch_len, seq_len, d_model), then the output
        # is viewed(batch_len, seq_len, h, d_k) and
        # transposed(batch_len, h, seq_len, d_k)
        Q, K, V = [
            linear_layer(x).view(x.size(0), x.size(1), self.h, self.d_k).transpose(1, 2)
            for linear_layer, x in zip(self.linear_layers, (Q, K, V))
        ]

        # shape: (batch_len, h, seq_len, d_v)
        heads = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # shape: (batch_len, h, seq_len, d_v)
        # shape: (batch_len, seq_len, h, d_v)
        # shape: (batch_len, seq_len, d_model)
        # d_v == d_k
        x = heads.transpose(1, 2).contiguous().view(Q.size(0), -1, self.h * self.d_k)

        del Q
        del K
        del V

        # apply final layer
        # out shape: (batch_len, seq_len, d_model)
        return self.linear_layers[-1](x)
