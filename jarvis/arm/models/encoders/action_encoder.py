import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict, Optional, Union, List, Any
from collections import OrderedDict
from gymnasium import spaces

from jarvis.stark_tech.env_interface import MinecraftWrapper


ACTION_KEY_DIM = OrderedDict({
    'forward': {'type': 'one-hot', 'dim': 2}, 
    'back': {'type': 'one-hot', 'dim': 2}, 
    'left': {'type': 'one-hot', 'dim': 2}, 
    'right': {'type': 'one-hot', 'dim': 2}, 
    'jump': {'type': 'one-hot', 'dim': 2}, 
    'sneak': {'type': 'one-hot', 'dim': 2}, 
    'sprint': {'type': 'one-hot', 'dim': 2}, 
    'attack': {'type': 'one-hot', 'dim': 2},
    'use': {'type': 'one-hot', 'dim': 2}, 
    'drop': {'type': 'one-hot', 'dim': 2},
    'inventory': {'type': 'one-hot', 'dim': 2}, 
    'camera': {'type': 'real', 'dim': 2}, 
    'hotbar.1': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.2': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.3': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.4': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.5': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.6': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.7': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.8': {'type': 'one-hot', 'dim': 2}, 
    'hotbar.9': {'type': 'one-hot', 'dim': 2}, 
})

class ActionEncoder(nn.Module):
    
    def __init__(
        self, 
        num_channels: int = 512,
        intermediate_dim: int = 64,
        action_type: Union['decomposed', 'composed'] = 'decomposed', 
        action_space: Optional[spaces.Space] = None, 
    ) -> None:
        super().__init__()
        self.action_type = action_type
        self.action_space = action_space
        if self.action_type == 'decomposed': 
            module_dict = dict()
            for key, conf in ACTION_KEY_DIM.items():
                key = 'act_' + key.replace('.', '_')
                if conf['type'] == 'one-hot':
                    module_dict[key] = nn.Embedding(conf['dim'], intermediate_dim)
                elif conf['type'] == 'real':
                    module_dict[key] = nn.Linear(conf['dim'], intermediate_dim)
            self.embedding_layer = nn.ModuleDict(module_dict)
            
        elif self.action_type == 'composed':
            module_dict = dict()
            for key, space in action_space.items():
                module_dict[key] = nn.Embedding(space.nvec.item(), num_channels)
            self.embedding_layer = nn.ModuleDict(module_dict)
        
        else:
            raise NotImplementedError
        self.final_layer = nn.Linear(len(self.embedding_layer) * intermediate_dim, num_channels)
    
    def forward_key_act(self, key: str, act: torch.Tensor) -> torch.Tensor:
        key_embedding_layer = self.embedding_layer['act_'+key.replace('.', '_')]
        if isinstance(key_embedding_layer, nn.Embedding):
            return key_embedding_layer(act.long())
        elif isinstance(key_embedding_layer, nn.Linear):
            return key_embedding_layer(act.float())
    
    def forward(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        if self.action_type == 'decomposed':
            if len(action) != len(ACTION_KEY_DIM):
                # convert to decomposed action and launch to device
                npy_act = MinecraftWrapper.agent_action_to_env(action)
                device = next(self.parameters()).device
                action = {key: torch.from_numpy(act).to(device) for key, act in npy_act.items()}
            return self.final_layer(torch.cat([
                self.forward_key_act(key, action[key]) for key in ACTION_KEY_DIM.keys()
            ], dim=-1))
        elif self.action_type == 'composed':
            return self.final_layer(torch.cat([
                self.forward_key_act(key, action[key]) for key in self.action_space.keys()
            ], dim=-1))
