from abc import ABC, abstractmethod
import torch
from jarvis.arm.utils.vpt_lib.action_head import ActionHead
from typing import Dict, List, Optional, Tuple, Any, Union
from omegaconf import DictConfig, OmegaConf
import numpy as np

def dict_map(fn, d):
    if isinstance(d, Dict) or isinstance(d, DictConfig):
        return {k: dict_map(fn, v) for k, v in d.items()}
    else:
        return fn(d)
    
class BaseAgent(torch.nn.Module, ABC):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def _batchify(self, elem):
        if isinstance(elem, (int, float)):
            elem = torch.tensor(elem, device=self.device)
        if isinstance(elem, np.ndarray):
            return torch.from_numpy(elem).unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, torch.Tensor):
            return elem.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, str):
            return [[elem]]
        else:
            raise NotImplementedError
    
    @abstractmethod
    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        pass

    @property
    @abstractmethod
    def action_head(self) -> ActionHead:
        pass

    @abstractmethod
    def forward(self, 
                obs: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]],
                first: Optional[torch.Tensor],
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        pass

    @torch.inference_mode()
    def get_action(self,
                   obs: Dict[str, Any],
                   state_in: Optional[List[torch.Tensor]],
                   first: Optional[torch.Tensor],
                   deterministic: bool = False,
                   input_shape: str = "BT*",
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        if input_shape == "*":
            obs = dict_map(self._batchify, obs)
            if state_in is not None:
                # add dummy batch dimension
                state_in = [state.unsqueeze(0) for state in state_in] 
        elif input_shape != "BT*":
            raise NotImplementedError
        
        result, state_out, latent_out = self.forward(obs, state_in, first=first, stage='rollout')
        pi_logits = result['pi_logits']
        action = self.action_head.sample(pi_logits, deterministic)
        
        if input_shape == "BT*":
            return action, state_out
        elif input_shape == "*":
            return dict_map(lambda tensor: tensor[0][0], action), [state[0] for state in state_out]
        else:
            raise NotImplementedError