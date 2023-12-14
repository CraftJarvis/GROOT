
from typing import (
    List, Dict, Optional, Callable, Union, Tuple, Any
)

import typing
import os
import pickle
from rich.console import Console
from pathlib import Path
import hydra
import torch
from torch import nn

from jarvis.arm.utils.vpt_lib.action_head import make_action_head
from jarvis.arm.utils.vpt_lib.normalize_ewma import NormalizeEwma
from jarvis.arm.utils.vpt_lib.scaled_mse_head import ScaledMSEHead
from jarvis.arm.utils.vpt_lib.tree_util import tree_map
from jarvis.arm.models.policys.groot_policy import GrootPolicy
from jarvis.arm.models.heads import build_auxiliary_heads

from omegaconf import DictConfig, OmegaConf
import gymnasium.spaces.dict as dict_spaces

RELATIVE_POLICY_CONFIG_DIR = '../../configs/policy'

def _make_policy(policy_name: str, **kwargs):
    # GrootPolicy is compatible with VPT
    if policy_name in ['groot', 'vpt']:
        return MinecraftAgentPolicy(policy_cls=GrootPolicy, **kwargs)
    else:
        raise ValueError(f'Unknown policy name: {policy_name}')
    
def load_policy_cfg(cfg_name: str) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    config_path = Path(RELATIVE_POLICY_CONFIG_DIR) / f"{cfg_name}.yaml"
    hydra.initialize(config_path=str(config_path.parent), version_base='1.3')
    policy_cfg = hydra.compose(config_name=config_path.stem)
    OmegaConf.resolve(policy_cfg)
    return policy_cfg

def make_policy(policy_cfg: Union[DictConfig, str, Dict[str, Any]], action_space: dict_spaces.Dict):
    if isinstance(policy_cfg, str):
        policy_cfg = load_policy_cfg(policy_cfg)

    if not isinstance(policy_cfg, dict):
        assert isinstance(policy_cfg, DictConfig), f"policy_cfg must be a string, a DictConfig or a dict, got {type(policy_cfg)}"
        build_kwargs = typing.cast(Dict[str, Any], OmegaConf.to_container(policy_cfg, resolve=True))
    else:
        build_kwargs = policy_cfg
    
    policy_name = build_kwargs['policy_name']
    
    building_info = {}
    model_path = policy_cfg['from'].get('model', None)
    if model_path and Path(model_path).is_file(): 
        Console().log(f"Loading predefined model from {model_path}. ")
        agent_parameters = pickle.load(Path(model_path).open("rb"))
        policy_body_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    else:
        policy_body_kwargs = build_kwargs['policy_kwargs']
        pi_head_kwargs = build_kwargs['pi_head_kwargs']
    auxiliary_head_kwargs = build_kwargs.get('auxiliary_head_kwargs', {})
    
    policy_kwargs = dict(
        action_space=action_space, 
        policy_body_kwargs=policy_body_kwargs, 
        pi_head_kwargs=pi_head_kwargs, 
        auxiliary_head_kwargs=auxiliary_head_kwargs, 
    )
    policy = _make_policy(policy_name, **policy_kwargs)
    
    weights_path = build_kwargs['from'].get('weights', None)
    if weights_path:
        Console().log('Loaded pretrained weights from checkpoint {}'.format(weights_path))
        if Path(weights_path).is_dir():
            weights_path = os.path.join(weights_path, 'model')
        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        filter_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('agent.', '') #! should be removed!!!!
            k = k.replace('conditioning_fusion_layer', 'condition_fusion_layer') #! should be removed!!!!
            if k.startswith('policy.'):
                filter_state_dict[k.replace('policy.', '')] = v
            else:
                filter_state_dict[k] = v
        building_info['ckpt_parameters'] = filter_state_dict
        policy.load_state_dict(filter_state_dict, strict=False)

    return policy, building_info

class MinecraftAgentPolicy(nn.Module):
    
    def __init__(
        self, 
        policy_cls,
        action_space, 
        policy_body_kwargs, 
        pi_head_kwargs, 
        auxiliary_head_kwargs, 
    ):
        super().__init__()
        
        self.net = policy_cls(**policy_body_kwargs, action_space=action_space)

        self.action_space = action_space

        self.value_head = self.make_value_head(self.net.output_latent_size())
        self.pi_head = self.make_action_head(self.net.output_latent_size(), **pi_head_kwargs)
        self.auxiliary_heads = nn.ModuleDict(build_auxiliary_heads(
            auxiliary_head_kwargs=auxiliary_head_kwargs, 
            hidsize=policy_body_kwargs['hidsize'],
        ))

    def make_value_head(self, v_out_size: int, norm_type: str = "ewma", norm_kwargs: Optional[Dict] = None):
        return ScaledMSEHead(v_out_size, 1, norm_type=norm_type, norm_kwargs=norm_kwargs)

    def make_action_head(self, pi_out_size: int, **pi_head_opts):
        return make_action_head(self.action_space, pi_out_size, **pi_head_opts)

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        self.pi_head.reset_parameters()
        self.value_head.reset_parameters()

    def encode_condition(self, *args, **kwargs):
        return self.net.encode_condition(*args, **kwargs)

    def is_conditioned(self):
        return self.net.is_conditioned()

    def forward(
        self, 
        obs: Dict, 
        first: torch.Tensor, 
        state_in: List[torch.Tensor], 
        stage: str = 'train', 
        ice_latent: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        obs = obs.copy()
        
        latents, state_out = self.net(
            obs=obs, 
            state_in=state_in, 
            context={"first": first}, 
            ice_latent=ice_latent,
        )
        result = {
            'pi_logits': self.pi_head(latents['pi_latent']),
            'vpred': self.value_head(latents['vf_latent']),
            'obs_mask': latents.get('obs_mask', None),
        }
        
        for head, module in self.auxiliary_heads.items():
            result[head] = module(latents, stage=stage)
        
        return result, state_out, latents