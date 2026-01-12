import inspect
from importlib import import_module
from typing import Callable, Optional, Union
from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn


class SelfMutualAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        assert attention_mask is None, "attention mask is not supported in this version of the processor"
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        src_q, tgt_q = rearrange(query, "(b n) h s d -> b n h s d", n = 2).chunk(2, dim = 1)
        src_q, tgt_q = src_q.squeeze(1), tgt_q.squeeze(1)

        src_k, tgt_k = rearrange(key, "(b n) h s d -> b n h s d", n = 2).chunk(2, dim = 1)
        src_k, tgt_k = src_k.squeeze(1), tgt_k.squeeze(1)

        src_v, tgt_v = rearrange(value, "(b n) h s d -> b n h s d", n = 2).chunk(2, dim = 1)
        src_v, tgt_v = src_v.squeeze(1), tgt_v.squeeze(1)

        hidden_states_src = F.scaled_dot_product_attention(
            src_q, src_k, src_v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_tgt = F.scaled_dot_product_attention(
            tgt_q, 
            torch.cat([tgt_k, src_k], dim=2), 
            torch.cat([tgt_v, src_v], dim=2), 
            attn_mask=attention_mask, 
            dropout_p=0.0,
            is_causal=False
        )

        hidden_states = torch.cat([hidden_states_src.unsqueeze(1), hidden_states_tgt.unsqueeze(1)], dim=1)
        hidden_states = rearrange(hidden_states, "b n h s d -> (b n) h s d")

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
