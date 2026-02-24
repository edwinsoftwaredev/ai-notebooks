import math
import torch


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    # (batch_len, seq_len, d_q)
    # (batch_len, seq_len, d_k)
    # (batch_len, seq_len, d_v)

    d_k = K.size(-1)  # the last dim: -1
    scaling_factor = 1 / math.sqrt(d_k)

    # We compute the dot products of THE query with ALL keys.
    # We need to compute a contextual representation (attention)
    # for this token at layer k+1 of the transformer, drawing on
    # the representations (from layer k) of every prior token.
    # out shape: (batch_len, seq_len, seq_len)
    attn_score = torch.matmul(Q, K.transpose(-2, -1)) * scaling_factor

    if mask is not None:
        # decoder: ith-token(query) must not attend to jth-token(keys) where j > i
        # sets -inf to unattended tokens(keys)
        attn_score = attn_score.masked_fill(~mask, -math.inf)

    # Normalization
    attn_score = torch.softmax(attn_score, dim=-1)

    if dropout is not None:
        attn_score = dropout(attn_score)

    # out shape: (batch_len, h, seq_len, d_v)
    return torch.matmul(attn_score, V)
