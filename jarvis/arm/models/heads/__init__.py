
import random
from functools import partial
from typing import Dict, Optional, Union, List, Any
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from jarvis.arm.models.heads.regular_head import RegularHead, PriorRegularHead, PrefixRegularHead

class BaseHead(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, latents: Dict[str, torch.Tensor], **kwargs) -> Any:
        '''
        Predict auxiliary task results based on the latents. 
        The returned results will be feed to the loss function as the `pred` term. 
        '''
        raise NotImplementedError
    
    def loss(self, obs, pred, mask=None, **kwargs) -> Any:
        '''
        `obs` terms refers to the original info that sampled from the dataset. 
        `pred` terms refers to the predicted results from the forward function. 
        You are supposed to return metric dict in this function. 
        '''
        raise NotImplementedError

def make_regular_head(**kwargs) -> nn.Module:
    return RegularHead(**kwargs)

def make_prior_regular_head(**kwargs) -> nn.Module:
    return PriorRegularHead(**kwargs)

def make_prefix_regular_head(**kwargs) -> nn.Module:
    return PrefixRegularHead(**kwargs)


register_heads = {
    'regular_head': make_regular_head,
    'prior_regular_head': make_prior_regular_head, 
    'prefix_regular_head': make_prefix_regular_head,
}

def build_auxiliary_heads(auxiliary_head_kwargs, **parent_kwargs) -> Dict[str, nn.Module]:
    
    auxilary_heads_dict = {}
    
    for head, head_kwargs in auxiliary_head_kwargs.items():
        assert head in register_heads, \
            f"Unknown auxiliary head {head}, available: {register_heads.keys()}"
        if not head_kwargs['enable']:
            continue
        auxilary_heads_dict[head] = register_heads[head](**head_kwargs, **parent_kwargs)
    
    return auxilary_heads_dict

