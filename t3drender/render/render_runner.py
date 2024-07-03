import math
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes
from tqdm import trange

from t3drender.cameras import NewCamerasBase
from .renderers.base_renderer import BaseRenderer
from .lights import AmbientLights, BaseLights

osj = os.path.join


def render(renderer: Union[nn.Module, dict],
           meshes: Union[Meshes, None] = None,
           device: Union[str, torch.device] = 'cpu',
           cameras: Union[NewCamerasBase, CamerasBase, dict, None] = None,
           lights: Union[BaseLights, dict, None] = None,
           batch_size: int = 5,
           no_grad: bool = True,
           verbose: bool = True,
           **forward_params):
    
    if isinstance(renderer, BaseRenderer):
        renderer = renderer
    else:
        raise TypeError('Wrong input renderer type.')

    renderer = renderer.to(device)

    if isinstance(cameras, NewCamerasBase):
        cameras = cameras
    else:
        raise TypeError('Wrong input cameras type.')

    num_frames = len(meshes)
    if isinstance(lights, BaseLights):
        lights = lights
    elif lights is None:
        lights = AmbientLights(device=device).extend(num_frames)
    else:
        raise ValueError('Wrong light type.')

    if len(cameras) == 1:
        cameras = cameras.extend(num_frames)
    if len(lights) == 1:
        lights = lights.extend(num_frames)

    forward_params.update(lights=lights, cameras=cameras, meshes=meshes)

    batch_size = min(batch_size, num_frames)
    tensors = []
    for k in forward_params:
        if isinstance(forward_params[k], np.ndarray):
            forward_params.update(
                {k: torch.tensor(forward_params[k]).to(device)})
    if verbose:
        iter_func = trange
    else:
        iter_func = range
    for i in iter_func(math.ceil(num_frames / batch_size)):
        indexes = list(
            range(i * batch_size, min((i + 1) * batch_size, len(meshes))))
        foward_params_batch = {}

        for k in forward_params:
            if hasattr(forward_params[k], '__getitem__'):
                foward_params_batch[k] = forward_params[k][indexes].to(device)

        if no_grad:
            with torch.no_grad():
                images_batch = renderer(indexes=indexes, **foward_params_batch)

        else:
            images_batch = renderer(indexes=indexes, **foward_params_batch)
        tensors.append(images_batch)

    if isinstance(tensors[0], torch.Tensor):
        tensors = torch.cat(tensors)
    else:
        tensors = np.concatenate(tensors)
    return tensors
