import random
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from typing import (
    List, Dict, Optional, Callable, Any
)

from jarvis.arm.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from jarvis.arm.utils.vpt_lib.misc import transpose
from jarvis.arm.models.backbones import build_backbone
from jarvis.arm.models.encoders import build_condition_encoder, ActionEncoder
from jarvis.arm.models.fusions import build_condition_fusion_layer

def dist_sample(dist, deterministic=False):
    mu, log_var = dist # (B, T, C)
    if deterministic:
        return mu
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

class GrootPolicy(nn.Module):
    
    def __init__(  
        self,
        action_space: Optional[Any] = None,
        hidsize: int = 512,
        init_norm_kwargs: Dict = {},
        # Below are TransformerXL's arguments
        attention_mask_style: str = "clipped_causal",
        attention_heads: int = 8,
        attention_memory_size: int = 1024,
        use_pointwise_layer: bool = True,
        pointwise_ratio: int = 4,
        pointwise_use_activation: bool = False,
        n_recurrence_layers: int = 4,
        recurrence_is_residual: bool = True,
        timesteps: int = 128,
        # Below are custimized arguments
        backbone_kwargs: Dict = {},
        condition_encoder_kwargs: Optional[Dict] = None,
        action_encoder_kwargs: Optional[Dict] = None,
        condition_fusion_kwargs: Optional[Dict] = None,
        **unused_kwargs,
    ):
        super().__init__()

        self.hidsize = hidsize
        self.timesteps = timesteps
        self.resolution = backbone_kwargs.get("resolution", None)
        
        # Prepare necessary parameters. (required by VPT)
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True
        
        # Build backbone module. 
        backbone_kwargs = {**backbone_kwargs, **unused_kwargs}
        backbone_kwargs['hidsize'] = hidsize
        backbone_kwargs['init_norm_kwargs'] = init_norm_kwargs
        backbone_kwargs['dense_init_norm_kwargs'] = self.dense_init_norm_kwargs
        backbone_results = build_backbone(**backbone_kwargs)
        self.img_preprocess = backbone_results['preprocessing']
        self.img_process = backbone_results['obsprocessing']
        
        # Build condition encoder. 
        self.condition_encoder_kwargs = condition_encoder_kwargs
        if self.condition_encoder_kwargs:
            self.condition_encoder = build_condition_encoder(
                hidsize=hidsize,
                **self.condition_encoder_kwargs
            ) 
            # Whether to build auxiliary condition encoder. 
            self.aux_condition_encoder = build_condition_encoder(
                hidsize=hidsize,
                **self.condition_encoder_kwargs
            ) if self.condition_encoder_kwargs.get('enable_aux_encoder', False) else None
        else:
            self.condition_encoder = None
            self.aux_condition_encoder = None
        
        # Build condition fusion layer. 
        self.condition_fusion_kwargs = condition_fusion_kwargs
        self.condition_fusion_layer = build_condition_fusion_layer(
            hidsize=hidsize,
            **self.condition_fusion_kwargs, 
        ) if self.condition_fusion_kwargs else None
        
        # Build action encoder. 
        self.action_encoder_kwargs = action_encoder_kwargs
        self.action_encoder = ActionEncoder(
            num_channels=hidsize, 
            action_space=action_space, 
            **self.action_encoder_kwargs
        ) if self.action_encoder_kwargs else None
        
        # Build TransformerXL layer. 
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type="transformer",
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
            use_flamingo_xattn=(self.condition_fusion_layer is not None),
        ) 

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = torch.nn.LayerNorm(hidsize)
        self.cached_init_states = {}
    
    def output_latent_size(self):
        return self.hidsize

    def extract_vision_feats(self, img: torch.Tensor) -> torch.Tensor:
        if self.resolution is not None:
            if isinstance(self.resolution, int):
                assert img.shape[-3:-1] == (self.resolution, self.resolution), f"the observation resolution {img.shape[-3:-1]} does not match the agent resolution {self.resolution}"
            else:
                raise NotImplementedError
        B, T = img.shape[:2]
        x = self.img_preprocess(img)
        x = self.img_process(x)
        vision_feats = x.reshape((B, T) + x.shape[2:])
        return vision_feats

    def extract_condition_feats(self, vi_latent: torch.Tensor, **kwargs):
        
        if getattr(self, 'condition_encoder', None):
            condition_dists = self.condition_encoder(vision_feats=vi_latent, **kwargs)
            ce_latent = dist_sample(condition_dists)
        else:
            condition_dists, ce_latent = None, None
        
        if getattr(self, 'aux_condition_encoder', None):
            cut_point = random.randint(vi_latent.shape[1] // 2, vi_latent.shape[1])
            prefix_vi_latent = vi_latent[:, :cut_point]
            aux_condition_dists = self.aux_condition_encoder(vision_feats=prefix_vi_latent, **kwargs)
            aux_ce_latent = dist_sample(aux_condition_dists)
        else:
            aux_condition_dists, aux_ce_latent = None, None
        
        return {
            'condition_dists': condition_dists, 
            'ce_latent': ce_latent, 
            'aux_condition_dists': aux_condition_dists, 
            'aux_ce_latent': aux_ce_latent, 
        } 

    def encode_condition(self, img, infer=False, **kwargs):
        '''called by outside to encode condition from video or text'''
        # extract vision feats, extract condition dists
        vi_latent = self.extract_vision_feats(img)
        conditions = self.extract_condition_feats(vi_latent=vi_latent, **kwargs)
        if infer:
            if (cd := conditions.get('condition_dists', None)) is not None:
                conditions['ce_latent'] = cd[0]
            if (cd := conditions.get('aux_condition_dists', None)) is not None:
                conditions['aux_ce_latent'] = cd[0]
        return conditions

    def forward(self, obs: Dict, state_in, context, ice_latent=None):
        latents = {}
        
        vi_latent = self.extract_vision_feats(obs['img'])
        
        if ice_latent is not None:
            ce_latent = ice_latent
        else:
            conditions = self.extract_condition_feats(vi_latent=vi_latent, texts=obs.get('text', None))
            latents.update(conditions)
            ce_latent = conditions['ce_latent']
        
        # condition policy on ce_latent
        if getattr(self, 'condition_fusion_layer', None): 
            # ce_latent = self.ce_transform(ce_latent)
            vi_latent = self.condition_fusion_layer(vi_latent, ce_latent) 
        
        x = vi_latent
        
        # add previous action embedding
        if getattr(self, 'action_encoder', None):
            prev_action_embedding = self.action_encoder(obs["prev_action"])
            x = prev_action_embedding + x
        
        # pass into TransformerXL layer
        x, state_out = self.recurrent_layer(x, context["first"], state_in, ce_latent=ce_latent)
        
        tf_latent = x
        x = F.relu(x, inplace=False)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        
        # return intermediate latents for decision making and other auxiliary tasks. 
        latents.update({
            "vi_latent": vi_latent,
            "pi_latent": pi_latent,
            "vf_latent": vf_latent,
            "tf_latent": tf_latent,
        })
        
        return latents, state_out

    def initial_state(self, batchsize):
        if self.recurrent_layer:
            if batchsize not in self.cached_init_states:
                self.cached_init_states[batchsize] = self.recurrent_layer.initial_state(batchsize)
            return self.cached_init_states[batchsize]
        else:
            return None

    def is_conditioned(self):
        return self.condition_encoder is not None