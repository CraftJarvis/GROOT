from typing import Dict, Optional, Union, List, Any, Tuple
import random
import numpy as np
import torch
from torch import nn
import torch.distributions as td
from jarvis.arm.utils.transformers import GPT 

from jarvis.arm.models.fusions import CrossAttentionLayer


class TrajectoryEncoder(nn.Module):
    
    def __init__(
        self, 
        hidsize: int, 
        num_slots: int = 1, 
        spatial_pooling: bool = True, 
        **kwargs
    ) -> None:
        
        super().__init__()
        self.num_slots = num_slots
        self.temporal_slot = nn.Parameter(torch.randn(self.num_slots, hidsize) * 1e-3)
        gpt_config = GPT.get_default_config()
        gpt_config.model_type = None
        gpt_config.n_embd = hidsize
        gpt_config.block_size = kwargs.get('block_size', 256)
        gpt_config.n_layer = kwargs.get('n_layer', 8)
        gpt_config.n_head = kwargs.get('num_heads', 8)
        self.traj_encoder = GPT(gpt_config)
        self.encode_mu = nn.Sequential(nn.Linear(hidsize, hidsize))
        self.encode_var = nn.Sequential(nn.Linear(hidsize, hidsize))
        
        self.spatial_pooling = spatial_pooling
        if self.spatial_pooling:
            self.spatial_pooling_layer = CrossAttentionLayer(input_size=hidsize, num_heads=kwargs.get('num_heads', 8))
            self.spatial_slot = nn.Parameter(torch.randn(1, hidsize) * 1e-3)
        
        self.mask_token = nn.Parameter(torch.randn(1, hidsize) * 1e-3)

    def forward(self, vision_feats, **kwargs):
        
        B, T, C = vision_feats.shape[:3]
        if self.spatial_pooling:
            assert len(vision_feats.shape) == 5, \
                'Error: vision_feats should be (B, T, C, W, W) if spatial_pooling is True!'
            W = vision_feats.shape[-1]
            flat_vision_feats = vision_feats.reshape(B*T, C, W*W).permute(0, 2, 1)
            spatial_slot_embedding = self.spatial_slot.unsqueeze(0).expand(B*T, -1, -1)
            obs_tokens = self.spatial_pooling_layer(spatial_slot_embedding, flat_vision_feats)
            obs_tokens = obs_tokens.reshape(B, T, C)
        else:
            assert len(vision_feats.shape) == 3, \
                'Error: vision_feats should be (B, T, C) if spatial_pooling is False!'
            obs_tokens = vision_feats
        
        temporal_slot_tokens = self.temporal_slot.unsqueeze(0).expand(B, -1, -1) 
        
        token_sequence = torch.cat([temporal_slot_tokens, obs_tokens], dim=1) 
        traj_embeddings = self.traj_encoder(token_sequence) 
        
        traj_embeddings = traj_embeddings[:, :self.num_slots, :] 
        
        mu = self.encode_mu(traj_embeddings)
        log_var = self.encode_var(traj_embeddings)
        dists = (mu, log_var)
        
        return dists