import functools
from jarvis.arm.models.agents.base_agent import BaseAgent
import torch
import re
import av
import numpy as np
import typing

from typing import Union, Dict, Optional, List, Tuple, Any

from jarvis.arm.utils.vpt_lib.action_head import ActionHead

from omegaconf import DictConfig
import gymnasium.spaces.dict as dict_spaces
from jarvis.arm.models.policys import make_policy, load_policy_cfg

def tree_get(obj, keys: List, default=None):
    try:
        for key in keys:
            if key in obj:
                obj = obj[key]
            else:
                return default
        return obj
    except:
        return default

class ConditionedAgent(BaseAgent):
    def __init__(
        self, 
        obs_space: dict_spaces.Dict, 
        action_space: dict_spaces.Dict, 
        policy_config: Union[DictConfig, str]
    ) -> None:
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.policy_config = policy_config

        if isinstance(self.policy_config, str):
            self.policy_config = load_policy_cfg(self.policy_config)
            
        self.policy, self.policy_building_info = make_policy(policy_cfg=self.policy_config, action_space=self.action_space)

        self.timesteps = tree_get(
            obj=self.policy_config, 
            keys=['policy_kwargs', 'timesteps'], 
            default=128
        )
        
        self.cached_init_states = {}
        
    def wrapped_forward(self, 
                        obs: Dict[str, Any], 
                        state_in: Optional[List[torch.Tensor]],
                        first: Optional[torch.Tensor] = None, 
                        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        '''Wrap state and first arguments if not specified. '''
        B, T, W, H = obs['img'].shape[:4]

        state_in = self.initial_state(B) if state_in is None else state_in
        
        if self.policy.is_conditioned() and 'obs_conf' in obs:
            ice_latent = self.load_input_condition(obs_conf=obs['obs_conf'], resolution=(W, H))
        else:
            ice_latent = None
        
        if first is None:
            first = (
                torch.from_numpy(np.array((False,)))
                .unsqueeze(0)
                .repeat(B, T)
                .to(self.device)
            )
        
        return self.policy(
            obs=obs, 
            first=first, 
            state_in=state_in, 
            ice_latent=ice_latent, 
            **kwargs
        )
    
    @functools.lru_cache(maxsize=None)
    def encode_video(
        self, ref_video: str, ref_mask: float, resolution: Tuple[int, int], 
    ) -> torch.Tensor:
        input_mask = torch.zeros(self.timesteps)
        if ref_mask < 1: 
            one_idx = torch.arange(0, self.timesteps, int(1 / (1-ref_mask)))
            input_mask[one_idx] = 1
        
        frames = []
        with av.open(ref_video, "r") as container:
            for fid, frame in enumerate(container.decode(video=0)):
                frame = frame.reformat(width=resolution[0], height=resolution[1]).to_ndarray(format="rgb24")
                frames.append(frame)
        
        segment = (
            torch.from_numpy(np.stack(frames[:self.timesteps], axis=0) )
            .unsqueeze(0)
            .to(self.device)
        )
        
        conditions = self.policy.encode_condition(img=segment, infer=True, input_mask=input_mask)
        ce_latent = conditions['ce_latent'].squeeze(0)
        print(
            "=======================================================\n"
            f"Ref video is from: {ref_video};\n"
            f"Num frames: {len(frames)}. \n"
            "=======================================================\n"
        )
        print(f"[📚] ce_latent shape: {ce_latent.shape} | mean: {ce_latent.mean().item(): .3f} | std: {ce_latent.std(): .3f}")
        return ce_latent

    def load_input_condition(self, obs_conf: Dict, resolution: Tuple[int, int]) -> torch.Tensor:
        '''Load the input condition specified by the obs_conf. '''
        assert 'ref_video' in obs_conf, 'ref_video should be specified in obs_conf. '
        num = len(obs_conf['ref_video'])
        ice_latent = []
        for i in range(num):
            ref_video = obs_conf['ref_video'][i][0]
            if 'ref_mask' in obs_conf:
                ref_mask = obs_conf['ref_mask'][i][0]
            else:
                ref_mask = 0.0
            ce_latent = self.encode_video(ref_video=ref_video, ref_mask=float(ref_mask), resolution=resolution)
            ice_latent.append(ce_latent)
        return torch.stack(ice_latent, dim=0)
    
    @property
    def action_head(self) -> ActionHead:
        return self.policy.pi_head

    @property
    def value_head(self) -> torch.nn.Module:
        return self.policy.value_head
    
    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.policy.initial_state(1)]
        else:
            if batch_size not in self.cached_init_states:
                self.cached_init_states[batch_size] = [t.to(self.device) for t in self.policy.initial_state(batch_size)]
            return self.cached_init_states[batch_size]

    def forward(self, 
                obs: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]],
                first: Optional[torch.Tensor] = None,
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        forward_result, state_out, latents = self.wrapped_forward(obs=obs, state_in=state_in, first=first, **kwargs)
        return forward_result, state_out, latents