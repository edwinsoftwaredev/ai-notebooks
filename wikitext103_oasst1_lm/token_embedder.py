import math
from torch import nn


class TokenEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        self.vocab_size = config["vocab_size"]
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        # in: (batch, seq_len)
        # out: (batch, seq_len, d_model)
        return self.embeddings(x) * math.sqrt(self.d_model)
