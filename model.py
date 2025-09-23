import torch
import torch.nn as nn
import math

# Input Embedding Layer
class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# Positional Encoding Layer   
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create matrix of size (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # create a vector of size (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)    # even indices have sine
        pe[:, 1::2] = torch.cos(position * div_term)    # odd indices have cosine

        pe = pe.unsqueeze(0)  # add batch dimension

        self.register_buffer('pe', pe)  # register as buffer so it is not a parameter but gets saved in the state dict and reused
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
# Layer Normalization
class LayerNorm(nn.Module):

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Feed Forward Layer
class FeedForward(nn.Module):

    def __init__(self, d_model:int, dff:int, dropout:float=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
# Multi-Head Attention Layer
