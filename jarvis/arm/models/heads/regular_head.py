

import random
from typing import Dict, Optional, Union, List, Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Normal, Categorical, kl_divergence

class PriorRegularHead(nn.Module):
    
    def __init__(self, weight: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.weight = weight
    
    def forward(self, latents, **kwargs) -> Dict[str, torch.Tensor]:
        if 'condition_dists' not in latents:
            return {}
        return {
            'condition_dists': latents['condition_dists'], 
        }

    def loss(self, obs, pred, mask=None, **kwargs):
        prior_loss = 0.
        num_dists = len(pred['condition_dists'])
        for dist in pred['condition_dists']:
            prior_dist = Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale))
            prior_loss += kl_divergence(dist, prior_dist).mean()
        prior_loss /= num_dists
        return {
            'prior_loss': self.weight * prior_loss,
        }

class PrefixRegularHead(nn.Module):
    
    def __init__(self, weight: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.weight = weight
        
    def forward(self, latents, **kwargs) -> Dict[str, torch.Tensor]:
        if 'condition_dists' not in latents:
            return {}
        return {
            'aux_condition_dists': latents['aux_condition_dists'], 
            'condition_dists': latents['condition_dists'],
        }
    
    def loss(self, obs, pred, mask=None, **kwargs):
        prefix_loss = 0.
        num_dists = len(pred['aux_condition_dists'])
        for prefix_dist, dist in zip(pred['aux_condition_dists'], pred['condition_dists']):
            prefix_loss += kl_divergence(dist, prefix_dist).mean()
        prefix_loss /= num_dists
        return {
            'prefix_loss': self.weight * prefix_loss, 
            'original_prefix': prefix_loss, 
        }

class RegularHead(nn.Module):

    def __init__(
        self, 
        interval: int = 1, 
        num_samples: int = 1,
        weight: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.interval = interval
        self.num_samples = num_samples
        self.weight = weight
        
        self.all_pairs = []
        self.acc_marks = [0]
        for i in range(256):
            if i >= self.interval:
                for j in range(self.interval, i - self.interval + 1):
                    self.all_pairs += [(i, j)]
            self.acc_marks += [len(self.all_pairs)]

    def forward(self, latents, **kwargs) -> Dict[str, torch.Tensor]:
        
        if 'condition_dists' not in latents:
            return {}
        
        dists = latents['condition_dists']
        T = len(dists)
        
        samples = []
        for _ in range(self.num_samples):
            mark = self.acc_marks[T]
            samples += [random.choice(self.all_pairs[:mark])]
        
        return {
            'dists': dists,
            'samples': samples, 
        }
    
    
    def loss(self, obs, pred, mask=None, **kwargs):
        '''
        mask: (B, T)
        '''
        dists = pred['dists']
        samples = pred['samples']
        loss = []
        for i, j in samples:
            kl_div = kl_divergence(dists[i], dists[j]).mean(dim=-1)
            if mask is not None:
                pair_mask = mask[:, [i, j]].all(dim=-1).float()
                loss += [(kl_div * pair_mask).mean()]
            else:
                loss += [kl_div.mean()]
        
        if len(loss) > 0:
            loss = torch.stack(loss).mean()
        else:
            loss = torch.zeros(1).to(loss.device)
            Console().log('No valid samples for regular head.')
        
        return {
            'loss_kl': loss * self.weight
        }
