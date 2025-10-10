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
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
    
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
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, h, seq_len, seq_len)
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, -1e9)  # mask certain tokens
        attn_scores = attn_scores.softmax(dim=-1)  # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attn_scores = dropout(attn_scores)
        
        output = torch.matmul(attn_scores, value)  # (batch_size, h, seq_len, d_k)
        return (output, attn_scores)

    def forward(self, q, k, v, mask):
        # matrix multiplication of Q, K, V (batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # split into h heads (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # compute attention
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # concatenate heads (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # final linear layer (batch_size, seq_len, d_model)
        return self.w_o(x)
    
# Residual Connection + Layer Normalization
class ResidualConnection(nn.Module):

    def __init__(self, features, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))# Although, the paper applies norm at end, after adding with sublayer. LayerNorm(x+ Sublayer(x))

# Encoder Block
class EncoderBlock(nn.Module):

    def __init__(self, features, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout), ResidualConnection(features, dropout)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
# Encoder
class Encoder(nn.Module):

    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Decoder Block
class DecoderBlock(nn.Module):

    def __init__(self, features, masked_self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.masked_self_attention = masked_self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout), ResidualConnection(features, dropout), ResidualConnection(features, dropout)])

    # src_mask is for the encoder-decoder attention, trg_mask is for the masked self-attention
    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.residual_connections[0](x, lambda x: self.masked_self_attention(x, x, x, trg_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x
    
# Decoder
class Decoder(nn.Module):

    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)

# Linear + Softmax Layer
class ProjectionLayer(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.proj(x)
    
# Full Transformer Model
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, trg_embed: InputEmbedding, src_pos: PositionalEncoding, trg_pos: PositionalEncoding, projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_pos(self.src_embed(src))
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg ,trg_mask):
        trg = self.trg_pos(self.trg_embed(trg))
        return self.decoder(trg, encoder_output, src_mask, trg_mask)
    
    def project(self, x):
        return self.projection(x)

def make_model(src_vocab_size: int, trg_vocab_size: int, src_seq_len: int, trg_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, dff: int=2048) -> Transformer:
    # create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    trg_embed = InputEmbedding(d_model, trg_vocab_size)

    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, dff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, dff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # create projection layer
    projection = ProjectionLayer(trg_vocab_size, d_model)

    transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection)

    # initialize parameters with xavier uniform (many papers use this initialization)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
