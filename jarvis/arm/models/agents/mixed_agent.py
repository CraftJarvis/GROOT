from functools import partial
import re
import typing
from typing import (
    Dict, List, Tuple, Optional, Union, Any
)
from jarvis.arm.models.agents.base_agent import BaseAgent
from jarvis.arm.models.agents.conditioned_agent import ConditionedAgent
from ray.rllib.utils.typing import ModelConfigDict
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import LightningModule

import gymnasium.spaces.dict as dict_spaces

import torch

from jarvis.arm.utils.vpt_lib.action_head import ActionHead
from jarvis.stark_tech.env_interface import MinecraftWrapper

class TensorDict(dict):
    def __init__(self, data: Dict[str, torch.Tensor]) -> None:
        super().__init__(data)
        self.data = data
    
    def _op(self, other, op):
        result = {}
        for key, value in self.data.items():
            if key in other.data:
                result[key] = op(value, other.data[key])
        return TensorDict(result)
    
    def __add__(self, other):
        return self._op(other, lambda a, b: a + b)
    
    def __sub__(self, other):
        return self._op(other, lambda a, b: a - b)
    
    def __mul__(self, scalar: Union[int, float]):
        result = {}
        for key, value in self.data.items():
            result[key] = value * scalar
        return TensorDict(result)
    
    def __rmul__(self, other):
        return self * other

class MixedAgent(BaseAgent):
    
    def __init__(
        self, 
        obs_space: dict_spaces.Dict, 
        action_space: dict_spaces.Dict, 
        policy_configs: DictConfig,
    ) -> None:
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.policy_configs = policy_configs

        self.sub_agents: Dict[str, ConditionedAgent] = {}
        for sub_policy_key, sub_policy_cfg in self.policy_configs.items():
            sub_policy_key = typing.cast(str, sub_policy_key)
            self.sub_agents[sub_policy_key] = ConditionedAgent(
                obs_space=obs_space,
                action_space=action_space,
                policy_config=sub_policy_cfg
            )

        assert len(self.sub_agents) > 0, "No sub agents found!"
        
        self.ordered_sub_keys = list(self.sub_agents.keys())
        self.ordered_sub_keys.sort()
        self.video_paths = []

        self.registered_sub_agents = torch.nn.ModuleDict(self.sub_agents)

    @property
    def action_head(self) -> ActionHead:
        return self.sub_agents[self.ordered_sub_keys[0]].action_head
        
    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        assert batch_size == None, "evaluation does not support batch_size!"
        # add agent_count dimension
        states = [
            torch.stack(item)
            for item in zip(*[self.sub_agents[key].initial_state(batch_size) for key in self.ordered_sub_keys])
        ]
        return states

    def forward(self, 
                obs: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]],
                first: Optional[torch.Tensor],
                stage: str = 'rollout',
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        # state shape: (batch=1, agent_count, memory, hiddim)
        B, T = obs['img'].shape[:2]
        outputs = None
        new_states = []
        vpred = {}
        obs_conf = obs.pop('obs_conf')
        for i, key in enumerate(self.ordered_sub_keys):
            sub_inputs = obs.copy()
            assert key in obs_conf, f"sub_agent_key {key} not found!"
            sub_inputs['obs_conf'] = obs_conf[key]
            if 'scale' in sub_inputs['obs_conf']:
                scale = eval(sub_inputs['obs_conf']['scale'][0][0])
            else:
                scale = 1.0
            sub_state = [ tensor[:,i,...] for tensor in state_in ]
            result, new_state, latents = self.sub_agents[key](
                obs=sub_inputs,
                state_in=sub_state,
                first=first,
                stage=stage,
            )
            vpred[key] = result['vpred']
            output = TensorDict(result['pi_logits'])
            if outputs is None:
                outputs = output * scale
            else:
                outputs += output * scale
            new_states.append(new_state)
        new_state = list(map(partial(torch.stack, dim=1), zip(*new_states)))
        self.vpred = vpred
        
        return {"pi_logits": outputs.data}, new_state, {}