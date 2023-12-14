
import torch
import torch.nn as nn 

from typing import Dict, Optional, Union, List, Any
from jarvis.arm.models.encoders.action_encoder import ActionEncoder
from jarvis.arm.models.encoders.trajectory_encoder import TrajectoryEncoder

def build_condition_encoder(
    name: Optional[str] = None, **kwargs, 
) -> Union[nn.Module, None]:
    if name == 'trajectory':
        return TrajectoryEncoder(**kwargs)
    else:
        return None

if __name__ == '__main__':
    pass
