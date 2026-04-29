import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    encodings: torch.Tensor  # type hint for dynamic attribute

    def __init__(self, config):
        super().__init__()

        d_model = config["d_model"]
        dropout = config["dropout"]
        max_seq_len = config["max_seq_len"]  # should be as large as the longest seq

        # Note that the input for the positional encoding
        # is the position of a token in the sequence
        # and not the token or its content.
        # In a set of sequences, tokens located at
        # ith position share the same positional encoding value,
        # this and the fixed space of sin/cos allow
        # precomputation of positional encodings.

        self.dropout = nn.Dropout(p=dropout)

        # shape: (max_seq_len, d_model)
        encodings = torch.zeros(max_seq_len, d_model)

        # a range: [0, max_seq_len), shape (1, max_seq_len)
        positions = torch.arange(max_seq_len)
        # to shape (max_seq_len, 1)
        positions = positions.unsqueeze(1)

        # Note that the division term in the freq could be the cause of overflow/underflow
        # Division term converted in log-space (change of base or scale)
        # Note that the negative sign already inverts the division term
        # and that vector only contains products of even ith-dimensions 2i
        div_term = torch.exp(
            torch.arange(0, d_model, step=2) * -(math.log(10000.0) / d_model)
        )

        # Outer Product (positions @ div_term) shape:
        # (max_seq_len, 1) @ (1, d_model / 2) = (max_seq_len, d_model / 2)
        frequencies = positions * div_term

        # sin for all positions in even ith-dimension   2i
        encodings[:, 0::2] = torch.sin(frequencies)

        # cos for all positions in odd ith-dimension    2i + 1
        encodings[:, 1::2] = torch.cos(frequencies)

        # make encodings match forward function input for add operation
        # (batch_len, seq_len, d_model) => (1, max_seq_len, d_model)
        # Note that seq_len != max_seq_len
        encodings = encodings.unsqueeze(0)

        # Registers tensor to Module (not Model)
        self.register_buffer("encodings", encodings)

    def forward(self, x):
        # batch token embeddings shape: (batch_len, seq_len, d_model)
        # positional encodings shape: (1, max_seq_len, d_model)

        # add “positional encodings” to the token embeddings
        x = x + self.encodings[:, : x.size(1)].requires_grad_(False)

        return self.dropout(x)
