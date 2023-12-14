import math
import random
from rich.console import Console
from copy import deepcopy
from functools import partial
from typing import Dict, Optional, Union, List, Any
from collections import OrderedDict


from gymnasium import spaces
import clip
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms as T
from torch.nn.parameter import Parameter
from torch.distributions import Distribution, Normal, Categorical, kl_divergence


from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.arm.utils.efficientnet_lib import EfficientNet
from jarvis.arm.utils.transformers import GPT 



class PositionalEncoding(nn.Module): 
    "Implement the PE function." 
    
    def __init__(self, d_model, max_len=256): 

        super(PositionalEncoding, self).__init__() 
        # Compute the positional encodings once in log space. 
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = (
            torch.exp(torch.arange(0, d_model, 2) * -(matorch.log(10000.0) / d_model)) 
        )
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe) 
    
    def forward(self, x): 
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
        return x 

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        pos = torch.arange(T, device=x.device).repeat(B, 1)
        return x + self.pos_embed(pos)

class SelfAttentionNet(nn.Module):
    
    def __init__(
        self, 
        input_size: int, 
        num_heads: int, 
        num_layers: int = 1, 
        pos_enc: str = 'learnable',
        **kwargs
    ) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(input_size, num_heads, batch_first=True) for _ in range(num_layers)
        ])
        if pos_enc == 'learnable':
            self.pos_embed = LearnablePositionalEncoding(input_size)
        else:
            self.pos_embed = PositionalEncoding(input_size)
    
    def forward(self, qkv):
        qkv = self.pos_embed(qkv)
        for i in range(len(self.attentions)):
            qkv, _ = self.attentions[i](qkv, qkv, qkv)
        return qkv

class CrossAttentionLayer(nn.Module):
    
    def __init__(
        self, 
        input_size: int, 
        num_heads: int, 
        pos_enc: str = 'learnable',
        **kwargs
    ) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(input_size)
        self.ln_kv = nn.LayerNorm(input_size)
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True)

        if pos_enc == 'learnable':
            self.pos_embed = LearnablePositionalEncoding(input_size)
        else:
            self.pos_embed = PositionalEncoding(input_size)

    def forward(self, q, kv):
        q = self.pos_embed(q)
        kv = self.pos_embed(kv)
        q = self.ln_q(q)
        kv = self.ln_kv(kv)
        attn_output, attn_weights = self.attention(q, kv, kv)
        return attn_output

class PastObsFusion(nn.Module):
    
    def __init__(
        self, 
        hidsize: int, 
        num_past_obs: int = 3, 
        num_heads: int = 4, 
        **kwargs, 
    ):
        super().__init__()
        self.hidsize = hidsize
        self.num_past_obs = num_past_obs
        self.attention_net = SelfAttentionNet(input_size=hidsize, num_heads=num_heads, num_layers=2)
    
    def forward(self, imgs):
        '''
        args:
            imgs: (B, num_past_obs + T, C)
        '''
        B, seq, C = imgs.shape
        T = seq - self.num_past_obs
        imgs_clauses = []
        for i in range(self.num_past_obs+1):
            imgs_clauses += [imgs[:, i:i+T, :]]
        extended_imgs = torch.stack(imgs_clauses, dim=2) # (B, T, num_past_obs+1, C)
        x = extended_imgs.reshape(B*T, self.num_past_obs+1, C)
        
        x = self.attention_net(qkv=x)
        x = x[:, -1, :].reshape(B, T, C)
        return x


class BaseConditioningLayer(nn.Module):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, vision_feats: torch.Tensor, cond_feats: torch.Tensor, **kwargs):
        raise NotImplementedError
    

class SpXattnCondLayer(BaseConditioningLayer):
    
    def __init__(
        self, 
        hidsize: int, 
        num_heads: int = 8, 
        **kwargs, 
    ) -> None:
        
        super().__init__(**kwargs)
        self.cls_token = nn.Parameter(torch.randn(1, hidsize) * 1e-3)
        gpt_config = GPT.get_default_config()
        gpt_config.model_type = None
        gpt_config.n_embd = hidsize
        gpt_config.block_size = kwargs.get('block_size', 256)
        gpt_config.n_layer = kwargs.get('n_layer', 2)
        gpt_config.n_head = kwargs.get('num_heads', 8)
        self.transformer = GPT(gpt_config)

    def forward(self, vision_feats: torch.Tensor, cond_feats: torch.Tensor, **kwargs):
        '''
        Inject the condition information into vision feats via cross-attention. 
        Args: 
            vision_feats: (B, T, C, W, W)
            cond_feats: (B, N, C)
        Return:
            output_feats: (B, T, C)
        '''
        assert len(vision_feats.shape) == 5 and len(cond_feats.shape) == 3, \
            "Error: vision feats must have 5 dimensions and cond feats must have 3 dimensions."
        B, T, C, W, _ = vision_feats.shape
        N = cond_feats.shape[1]
        cond_feats = cond_feats.unsqueeze(1).repeat(1, T, 1, 1)
        cond_tokens = cond_feats.reshape(B*T, N, C)
        vision_tokens = vision_feats.reshape(B*T, C, -1).permute(0, 2, 1)
        cls_tokens = self.cls_token.repeat(B*T, 1).unsqueeze(1)
        token_sequence = torch.cat( [cls_tokens, cond_tokens, vision_tokens] , dim=1)
        output_feats = self.transformer(token_sequence)[:, 0, :]
        output_feats = output_feats.reshape(B, T, C)   
        return output_feats


class AddCondLayer(BaseConditioningLayer):
    
    def forward(self, vision_feats: torch.Tensor, cond_feats: torch.Tensor, **kwargs):
        '''
        Inject the condition information into vision feats via cross-attention. 
        args: 
            vision_feats: (B, T, C)
            cond_feats: (B, T, C) or (B, C)
        return:
            output_feats: (B, T, C)
        '''
        assert len(vision_feats.shape) == 3, \
            "AddConditioningLayer requires no additional dimension than B, T, C."

        B, T, C = vision_feats.shape
        
        if len(cond_feats.shape) == 2:
            cond_feats = cond_feats.unsqueeze(1).repeat(1, vision_feats.shape[1], 1)
        
        output_feats = vision_feats + cond_feats
        
        return output_feats

class IdentityCondLayer(BaseConditioningLayer):
    
    def forward(self, vision_feats: torch.Tensor, cond_feats: torch.Tensor, **kwargs):
        '''
        Inject the condition information into vision feats via cross-attention. 
        args: 
            vision_feats: (B, T, C)
            cond_feats: (B, T, C) or (B, C)
        return:
            output_feats: (B, T, C)
        '''
        assert len(vision_feats.shape) == 3, \
            "AddConditioningLayer requires no additional dimension than B, T, C."
        
        return vision_feats


def build_condition_fusion_layer(
    name: Optional[str] = None, **kwargs
):
    if name == 'spatial_xattn': 
        return SpXattnCondLayer(**kwargs)
    elif name == 'add':
        return AddCondLayer(**kwargs)
    elif name == 'identity':
        return IdentityCondLayer(**kwargs)
    else:
        raise None

if __name__ == '__main__':
    pass