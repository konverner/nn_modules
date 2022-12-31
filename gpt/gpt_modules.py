"""
PyTorch implementation GPT architecture from "Improving Language Understanding by Generative Pre-Training"
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
"""
import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, masked: bool = False):
        super().__init__()
        self.masked = masked
        if masked:
            self.register_buffer("mask", torch.tril(torch.ones(512, 512))
                                 .view(1, 1, 512, 512))

    def forward(self, q, k, v):
        """
        q : tensor [batch_size, heads, length, d_model//heads]
        k : tensor [batch_size, heads, length, d_model//heads]
        v : tensor [batch_size, heads, length, d_model//heads]
        """
        d_k = k.shape[-1]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if self.masked:
            scores = scores.masked_fill(self.mask[:, :, :q.shape[2], :q.shape[2]] == 0, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        output = scores @ v
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1, masked: bool = False):
        """
        params
        ---
        heads : number of heads
        d_model : model dimension (embeddings size)
        dropout : dropout probability
        masked : is attention masked or not
        """
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(masked=masked)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        """
        params
        ---
        q : tensor [batch_size, length, emb_dim]
        k : tensor [batch_size, length, emb_dim]
        v : tensor [batch_size, length, emb_dim]

        returns
        ---
        output : [batch_size, length, emb_dim]
        """
        bs = q.size(0)
        # perform linear operation, split into h heads and get [bs, h, length, d_k]
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        scores = self.attention(q, k, v)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.dropout(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.weight = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.weight * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1, d_ff: int = 2048):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.dp = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(heads, d_model, dropout, masked=True)
        self.ff = FeedForward(d_model, d_ff=d_ff)

    def forward(self, x):
        x = self.norm_1(x + self.dp(self.attn(x, x, x)))
        x = self.norm_2(x + self.dp(self.ff(x)))
        return x


class GPT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_layers: int,
                 heads: int,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_length: int = 128):
        """
        params
        ---
        vocab_size : now many unique tokens
        d_model : tokens embeddings size
        n_layers : how many decoder layers
        heads : how many heads in attention block
        d_ff : embeddings size in decoder's feed forward NN
        dropout : probability of dropout
        max_length : maximum length of token sequence
        """
        super().__init__()
        self.max_length = max_length
        self.embedder = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_length, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout, d_ff) for _ in range(n_layers)])
        self.ff = nn.Linear(d_model, vocab_size)
        self.sm = nn.Softmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, tokens):
        """
        tokens : [bs, length]
        """
        pos = torch.arange(0, tokens.shape[1], dtype=torch.long).unsqueeze(0)
        x = self.embedder(tokens) + self.pe(pos)
        for layer in self.layers:
            x = layer(x)
        logits = self.ff(x)
        probs = self.sm(logits)
        return probs
