from torch import nn


class PositionWiseFFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config["d_model"]
        d_ff = config["d_ff"]
        dropout = config["dropout"]

        self.out_proj = nn.Linear(d_ff, d_model)
        self.out_proj._is_residual = True  # pyright: ignore

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), self.out_proj
        )

    def forward(self, x):
        # input: (batch_len, seq_len, d_model)
        return self.ffn(x)
