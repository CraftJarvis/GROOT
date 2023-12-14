
import math
import random
import functools
from copy import deepcopy
from functools import partial
from typing import Dict, Optional, Union, List, Any, Tuple
from collections import OrderedDict

import clip
import numpy as np
import torch
from torch import nn

import torchvision
from torchvision import transforms as T

from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.arm.utils.vpt_lib.impala_cnn import ImpalaCNN
from jarvis.arm.utils.sam_lib.image_encoder import ImageEncoderViT
from jarvis.arm.utils.efficientnet_lib import EfficientNet
from jarvis.arm.utils.vpt_lib.tree_util import tree_map
from jarvis.arm.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks

class ImgPreprocessing(nn.Module):
    """Normalize incoming images.

    :param img_statistics: remote path to npz file with a mean and std image. If specified
        normalize images using this.
    :param scale_img: If true and img_statistics not specified, scale incoming images by 1/255.
    """

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = nn.Parameter(torch.Tensor(img_statistics["mean"]), requires_grad=False)
            self.img_std = nn.Parameter(torch.Tensor(img_statistics["std"]), requires_grad=False)
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        # x = img.to(dtype=torch.float32)
        x = img
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        if x.dim() == 4:
            x = x.unsqueeze(1)
        return x


class ImgObsProcess(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img, **kwargs):
        return self.linear(self.cnn(img))

def general_preprocessor(
    image_tensor: torch.Tensor, 
    scale_input:float = 255.0, 
    normalize:bool = True, 
    channel_last:bool = False,
    image_shape:Optional[Tuple[int, ...]] = None
) -> torch.Tensor:
    if image_tensor.dtype == torch.uint8:
        image_tensor = image_tensor.to(torch.float32)

    if image_tensor.dim() == 4:
        image_tensor = image_tensor.unsqueeze(1)
    
    # shape is (B, T, C, H, W) or (B, T, H, W, C)
    if image_tensor.shape[-1] == 3:
        image_tensor = image_tensor.permute(0, 1, 4, 2, 3).contiguous()
    # shape is (B, T, C, H, W)

    if image_shape is not None:
        H, W, C = image_shape
        assert image_tensor.shape[-3] == C
        assert image_tensor.shape[-2:] == (H, W)
    
    transform_list = [
        T.Lambda(lambda x: x / scale_input), 
    ]
    
    if normalize:
        transform_list.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    
    transform = T.Compose(transform_list)
    processed_images = transform(image_tensor)

    if channel_last:
        processed_images = processed_images.permute(0, 1, 3, 4, 2).contiguous()
    
    return processed_images


class SpatialSoftmax(nn.Module):
    
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1) # NxCxHW
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True) # NCxHW
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True) # NCxHW
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

class CustomEfficientNet(nn.Module):
    
    def __init__(
        self, 
        version: str, 
        resolution: int = 224, 
        out_dim: int = 1024, 
        pooling: bool = False, 
        # cross_attention: bool = False,
        **kwargs, 
    ) -> None:
        super().__init__()
        self.version = version
        self.resoulution = resolution
        self.out_dim = out_dim
        
        self.model = EfficientNet.from_pretrained(version)
        # self.model = EfficientNet.from_name(version)
        
        if 'b0' in version:
            self.mid_dim = 1280
        elif 'b4' in version:
            self.mid_dim = 1792
        
        if resolution == 360:
            self.feat_reso = (11, 11)
        elif resolution == 224:
            self.feat_reso = (7, 7)
        elif resolution == 128:
            self.feat_reso = (4, 4)

        self.final_layer = nn.Conv2d(self.mid_dim, out_dim, 1)
        
        # assert not (pooling and cross_attention), "pooling and cross_attention cannot be True at the same time"
        if pooling:
            self.pooling_layer = nn.AdaptiveMaxPool2d(1)

    def forward(self, imgs, cond=None, **kwargs): 
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:])
        x = self.model.extract_features(x)
        x = self.final_layer(x)
        x = x.reshape((B, T) + x.shape[1:])
        
        if hasattr(self, 'pooling_layer'):
            x = x.reshape(B * T, *x.shape[2:])
            x = self.pooling_layer(x).squeeze(-1).squeeze(-1)
            x = x.reshape(B, T, -1)
        
        return x

class CustomResNet(nn.Module):
    
    def __init__(self, version: str = '50', out_dim: int = 1024, **kwargs):
        super().__init__()
        if version == '18':
            self.model = torchvision.models.resnet18(pretrained=True)
        elif version == '50':
            self.model = torchvision.models.resnet50(pretrained=True)
        elif version == '101':
            self.model = torchvision.models.resnet101(pretrained=True)
        
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.final_layer = nn.Linear(2048, out_dim)
    
    def forward(self, imgs, **kwargs):
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:])
        x = self.model(x)
        x = x.view(B * T, -1)
        x = self.final_layer(x)
        return x.reshape(B, T, -1)


class CustomCLIPv(nn.Module):
    
    def __init__(self, version: str = "ViT-B/32", out_dim: int = 1024, **kwargs):
        super().__init__()
        # first load into cpu, then move to cuda by the lightning
        clip_model, preprocess = clip.load(version, device='cpu')
        self.preprocess = preprocess
        self.vision_encoder = clip_model.visual
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.final_layer = nn.Linear(512, out_dim)
    
    @torch.no_grad()
    def forward(self, imgs, **kwargs):
        B, T = imgs.shape[:2]
        x = imgs.reshape(B * T, *imgs.shape[2:])
        x = self.vision_encoder(x)
        x = self.final_layer(x)
        return x.reshape(B, T, -1)


def build_backbone(name: str = 'IMPALA', **kwargs) -> Dict:
    
    result_modules = {}
    if name == 'IMPALA':
        first_conv_norm = False
        impala_kwargs = kwargs.get('impala_kwargs', {})
        init_norm_kwargs = kwargs.get('init_norm_kwargs', {})
        dense_init_norm_kwargs = kwargs.get('dense_init_norm_kwargs', {})
        result_modules['preprocessing'] = partial(
            general_preprocessor, 
            scale_input=255.0, 
            normalize=False, 
            channel_last=True,
            image_shape=kwargs.get('img_shape', {})
        )
        result_modules['obsprocessing'] = ImgObsProcess(
            cnn_outsize=256,
            output_size=kwargs['hidsize'],
            inshape=kwargs['img_shape'],
            chans=tuple(int(kwargs['impala_width'] * c) for c in kwargs['impala_chans']),
            nblock=2,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs, 
        )
        
    elif name == 'CLIPv':
        model = CustomCLIPv(
            out_dim=kwargs['hidsize'],
            **kwargs,
        )
        result_modules['preprocessing'] = partial(
            general_preprocessor, 
            scale_input=255.0, 
            normalize=True, 
            channel_last=False
        )
        result_modules['obsprocessing'] = model

    elif name == 'SAM':
        pass
    
    elif name == 'EFFICIENTNET':
        model = CustomEfficientNet(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
        result_modules['preprocessing'] = partial(
            general_preprocessor, 
            scale_input=255.0, 
            normalize=True, 
            channel_last=False
        )
        result_modules['obsprocessing'] = model
        
    elif name == 'RESNET':
        result_modules['preprocessing'] = partial(
            general_preprocessor, 
            scale_input=255.0, 
            normalize=True, 
            channel_last=False, 
            out_dim=kwargs['hidsize']
        )
        result_modules['obsprocessing'] = CustomResNet(
            version=kwargs['version'], 
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
    return result_modules

if __name__ == '__main__':
    pass 