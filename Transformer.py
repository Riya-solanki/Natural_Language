import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, is_encoder: bool, bias=True):
        super().__init__(in_features, out_features, bias)
        self.is_encoder = is_encoder


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        context_size: int,
        d_model: int,
        dim_keys: int,
        dim_values: int,
        drop_prob: float,
        mask_attention: bool,
    ):
        super().__init__()
        self.dim_keys = dim_keys
        self.mask_attention = mask_attention

        self.queries = nn.Linear(d_model, dim_keys)
        self.keys = nn.Linear(d_model, dim_keys)
        self.values = nn.Linear(d_model, dim_values)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_prob)

        if mask_attention:
            # triangular causal mask for max context size
            mask = torch.tril(torch.ones(context_size, context_size, dtype=torch.bool))
            self.register_buffer("mask", mask)

    def forward(
        self,
        X_Q: torch.Tensor,  # (batch, q_len, d_model)
        X_KV: torch.Tensor,  # (batch, k_len, d_model)
        attention_mask: Optional[torch.FloatTensor] = None,  # (batch, k_len) with 1 valid, 0 pad
    ):
        # Project
        Q = self.queries(X_Q)  # (batch, q_len, dim_keys)
        K = self.keys(X_KV)  # (batch, k_len, dim_keys)

        # Raw scores: (batch, q_len, k_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.dim_keys)

        # Apply causal mask if present (mask is (context_size, context_size))
        q_len, k_len = scores.size(-2), scores.size(-1)
        if self.mask_attention:
            causal = self.mask[:q_len, :k_len].to(scores.device)  # (q_len, k_len)
            # expand to batch
            scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))

        # Apply padding attention mask (if provided). attention_mask: (batch, k_len)
        if attention_mask is not None:
            # broadcast to (batch, q_len, k_len)
            am = attention_mask.to(dtype=torch.bool).unsqueeze(1).expand(-1, q_len, -1)
            scores = scores.masked_fill(~am, float("-inf"))

        attn = self.softmax(scores)
        attn = self.dropout(attn)

        V = self.values(X_KV)  # (batch, k_len, dim_values)
        out = attn @ V  # (batch, q_len, dim_values)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        context_size: int,
        d_model: int,
        dim_keys: int,
        dim_values: int,
        drop_prob: float,
        mask_attention: bool,
        is_encoder: bool,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                ScaledDotProductAttention(
                    context_size=context_size,
                    d_model=d_model,
                    dim_keys=dim_keys,
                    dim_values=dim_values,
                    drop_prob=drop_prob,
                    mask_attention=mask_attention,
                )
                for _ in range(num_heads)
            ]
        )
        self.linear = ResidualScaledLinear(num_heads * dim_values, d_model, is_encoder)
        self.dropout = nn.Dropout(drop_prob)

    def forward(
        self, X_Q: torch.Tensor, X_KV: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None
    ):
        # run each head and concat along last dim
        head_outs = [h(X_Q, X_KV, attention_mask) for h in self.heads]  # list of (batch, q_len, dim_values)
        Ans = torch.cat(head_outs, dim=-1)  # (batch, q_len, num_heads*dim_values)
        Ans = self.dropout(self.linear(Ans))  # project back to d_model
        return Ans


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, drop_prob: float, is_encoder: bool):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = ResidualScaledLinear(d_ff, d_model, is_encoder)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, X: torch.Tensor):
        return self.dropout(self.linear2(self.relu(self.linear1(X))))


class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, context_size: int, d_model: int, drop_prob: float, d_ff: int):
        super().__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            context_size=context_size,
            d_model=d_model,
            dim_keys=d_model // num_heads,
            dim_values=d_model // num_heads,
            drop_prob=drop_prob,
            mask_attention=False,
            is_encoder=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, drop_prob, is_encoder=True)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor, pre_norm: bool):
        if pre_norm:
            x_norm = self.norm1(X)
            X = X + self.mha(x_norm, x_norm)  # self-attention
            X = X + self.ff(self.norm2(X))
        else:
            X = self.norm1(X + self.mha(X, X))
            X = self.norm2(X + self.ff(X))
        return X


class DecoderBlock(nn.Module):
    """
    Full decoder block with masked self-attention and encoder-decoder cross-attention
    Supports pre-norm and post-norm variants.
    """

    def __init__(self, num_heads: int, context_size: int, d_model: int, drop_prob: float, d_ff: int):
        super().__init__()
        self.masked_mha = MultiHeadAttention(
            num_heads=num_heads,
            context_size=context_size,
            d_model=d_model,
            dim_keys=d_model // num_heads,
            dim_values=d_model // num_heads,
            drop_prob=drop_prob,
            mask_attention=True,
            is_encoder=False,
        )

        self.cross_mha = MultiHeadAttention(
            num_heads=num_heads,
            context_size=context_size,
            d_model=d_model,
            dim_keys=d_model // num_heads,
            dim_values=d_model // num_heads,
            drop_prob=drop_prob,
            mask_attention=False,
            is_encoder=False,
        )

        # three norms for post-norm; pre-norm will use them similarly before modules
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, drop_prob, is_encoder=False)

    def forward(self, X: torch.Tensor, encoded_info: torch.Tensor, pre_norm: bool, attention_mask: Optional[torch.FloatTensor] = None):
        # attention_mask passed to masked_mha and cross_mha for padding masking
        if pre_norm:
            x_norm = self.norm1(X)
            X = X + self.masked_mha(x_norm, x_norm, attention_mask)
            x_norm2 = self.norm2(X)
            X = X + self.cross_mha(x_norm2, encoded_info, attention_mask)
            X = X + self.ff(self.norm3(X))
        else:
            X = self.norm1(X + self.masked_mha(X, X, attention_mask))
            X = self.norm2(X + self.cross_mha(X, encoded_info, attention_mask))
            X = self.norm3(X + self.ff(X))
        return X


class OnlyDecoderBlock(nn.Module):
    def __init__(self, num_heads: int, context_size: int, d_model: int, drop_prob: float, d_ff: int, mask_attention: bool):
        super().__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            context_size=context_size,
            d_model=d_model,
            dim_keys=d_model // num_heads,
            dim_values=d_model // num_heads,
            drop_prob=drop_prob,
            mask_attention=mask_attention,
            is_encoder=False,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, drop_prob, is_encoder=False)

    def forward(self, X: torch.Tensor, pre_norm: bool, attention_mask: Optional[torch.FloatTensor] = None):
        if pre_norm:
            x_norm = self.norm1(X)
            X = X + self.mha(x_norm, x_norm, attention_mask)
            X = X + self.ff(self.norm2(X))
        else:
            X = X + self.mha(X, X, attention_mask)
            X = self.norm1(X)
            X = X + self.ff(X)
            X = self.norm2(X)
        return X


class Transformer(nn.Module):
    """
    Encoder-decoder Transformer.
    forward(src_tokens, tgt_tokens, src_attention_mask=None, tgt_attention_mask=None)
    - src_tokens: (batch, src_len) integer token ids for encoder
    - tgt_tokens: (batch, tgt_len) integer token ids for decoder input (shifted right)
    - src_attention_mask: (batch, src_len) float tensor with 1 for real tokens, 0 for padding
    - tgt_attention_mask: (batch, tgt_len) float tensor with 1 for real tokens, 0 for padding
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        enc_context_size: int,
        dec_context_size: int,
        pos_enc_dropout: float,
        num_encoder_blocks: int,
        num_decoder_blocks: int,
        num_heads: int,
        drop_prob: float,
        d_feedfwd: int,
        pre_norm: bool,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pre_norm = pre_norm
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks

        # token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # positional embeddings (learned)
        self.enc_pos_encoding = nn.Embedding(enc_context_size, d_model)
        self.dec_pos_encoding = nn.Embedding(dec_context_size, d_model)
        nn.init.normal_(self.enc_pos_encoding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.dec_pos_encoding.weight, mean=0.0, std=0.01)

        # encoder / decoder stacks
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(num_heads, enc_context_size, d_model, drop_prob, d_ff=d_feedfwd)
                for _ in range(num_encoder_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(num_heads, dec_context_size, d_model, drop_prob, d_ff=d_feedfwd)
                for _ in range(num_decoder_blocks)
            ]
        )

        if pre_norm:
            self.enc_final_ln = nn.LayerNorm(d_model)
            self.dec_final_ln = nn.LayerNorm(d_model)

        # initialize linear weights (ResidualScaledLinear use different scales)
        self.apply(self.init_linear_weights)

        # final head
        self.head = nn.Linear(d_model, vocab_size)
        # tie weights for token embedding and head
        self.token_embedding.weight = self.head.weight
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def init_linear_weights(self, module):
        if isinstance(module, ResidualScaledLinear):
            # different scaling for encoder vs decoder residual linears
            if getattr(module, "is_encoder", False):
                std = 0.02 * ((2 * max(1, self.num_encoder_blocks)) ** -0.5)
            else:
                std = 0.02 * ((3 * max(1, self.num_decoder_blocks)) ** -0.5)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_attention_mask: Optional[torch.FloatTensor] = None,
        tgt_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        """
        src_tokens: (batch, src_len)
        tgt_tokens: (batch, tgt_len)
        attention_mask: (batch, seq_len) float tensor with 1 for real tokens, 0 for pads
        """
        # Encoder embeddings
        batch, src_len = src_tokens.shape
        enc_pos = torch.arange(0, src_len, device=src_tokens.device).unsqueeze(0).expand(batch, -1)
        enc_token_emb = self.token_embedding(src_tokens)  # (batch, src_len, d_model)
        enc_pos_emb = self.enc_pos_encoding(enc_pos)  # (batch, src_len, d_model)
        enc_input = enc_token_emb + enc_pos_emb

        # pass through encoder stack
        enc_out = enc_input
        for eb in self.encoder_blocks:
            enc_out = eb(enc_out, self.pre_norm)

        if self.pre_norm:
            enc_out = self.enc_final_ln(enc_out)

        # Decoder embeddings
        batch, tgt_len = tgt_tokens.shape
        dec_pos = torch.arange(0, tgt_len, device=tgt_tokens.device).unsqueeze(0).expand(batch, -1)
        dec_token_emb = self.token_embedding(tgt_tokens)
        dec_pos_emb = self.dec_pos_encoding(dec_pos)
        dec_input = dec_token_emb + dec_pos_emb

        # pass through decoder stack (masked self-attention + cross-attention)
        dec_out = dec_input
        for db in self.decoder_blocks:
            dec_out = db(dec_out, enc_out, self.pre_norm, attention_mask=tgt_attention_mask)

        if self.pre_norm:
            dec_out = self.dec_final_ln(dec_out)

        logits = self.head(dec_out)  # (batch, tgt_len, vocab_size)
        return logits


class Decoder(nn.Module):
    """
    Standalone decoder (only self-attention stack) that mirrors the user's original second Decoder class design.
    forward(tgt_tokens, attention_mask=None)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_size: int,
        pos_enc_dropout: float,
        num_decoder_blocks: int,
        num_heads: int,
        drop_prob: float,
        d_feedfwd: int,
        pre_norm: bool,
        mask_attention: bool,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pre_norm = pre_norm
        self.num_decoder_blocks = num_decoder_blocks

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        self.pos_encoding = nn.Embedding(context_size, d_model)
        nn.init.normal_(self.pos_encoding.weight, mean=0.0, std=0.01)

        self.decoder = nn.ModuleList(
            [
                OnlyDecoderBlock(num_heads, context_size, d_model, drop_prob, d_ff=d_feedfwd, mask_attention=mask_attention)
                for _ in range(num_decoder_blocks)
            ]
        )

        if pre_norm:
            self.dec_final_ln = nn.LayerNorm(d_model)

        self.apply(self.init_linear_weights)

        self.head = nn.Linear(d_model, vocab_size)
        # tie embeddings and head
        self.token_embedding.weight = self.head.weight
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def init_linear_weights(self, module):
        if isinstance(module, ResidualScaledLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 * ((2 * max(1, self.num_decoder_blocks)) ** -0.5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None):
        # X: (batch, seq_len)
        batch, seq_len = X.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=X.device).unsqueeze(0).expand(batch, -1)
        dec_pos_enc = self.pos_encoding(pos)
        dec_token_emb = self.token_embedding(X)
        X_emb = dec_token_emb + dec_pos_enc

        out = X_emb
        for db in self.decoder:
            out = db(out, self.pre_norm, attention_mask)

        if self.pre_norm:
            out = self.dec_final_ln(out)

        out = self.head(out)
        return out
