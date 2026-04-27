from torch import nn


class PositionWiseFFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config["d_model"]
        d_ff = config["d_ff"]
        dropout = config["dropout"]

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # input: (batch_len, seq_len, d_model)
        return self.ffn(x)
