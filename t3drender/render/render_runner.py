import math
import os
from typing import Union, List
import multiprocessing
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


def render_flow(renderer: Union[nn.Module, dict],
           meshes_source: Union[Meshes, None] = None,
            meshes_target: Union[Meshes, None] = None,
           device: Union[str, torch.device] = 'cpu',
           cameras: Union[NewCamerasBase, CamerasBase, dict, None] = None,
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

    num_frames = len(meshes_source)

    if len(cameras) == 1:
        cameras = cameras.extend(num_frames)

    forward_params.update(cameras=cameras, meshes_source=meshes_source, meshes_target=meshes_target)

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
            range(i * batch_size, min((i + 1) * batch_size, len(meshes_source))))
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


def parse_device(device: Union[str, int, List[str], List[int], torch.device, List[torch.device]]):
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, list) and isinstance(device[0], int):
        device = [torch.device(f'cuda:{d}') for d in device]
    elif isinstance(device, list) and isinstance(device[0], str):
        device = [torch.device(d) for d in device]
    elif isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    if not isinstance(device, list):
        device = [device]
    return device


def unpack_kwargs_and_call_worker(kwargs):
    return render(**kwargs)

def unpack_kwargs_and_call_worker_flow(kwargs):
    return render_flow(**kwargs)

def render_mp(renderer: Union[nn.Module, dict],
                meshes: Union[Meshes, None] = None,
                device: Union[str, torch.device] = 'cpu',
                cameras: Union[NewCamerasBase, CamerasBase, dict, None] = None,
                lights: Union[BaseLights, dict, None] = None,
                batch_size: int = 5,
                no_grad: bool = True,
                verbose: bool = True,):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    device = parse_device(device)
    if len(device) >1:
        num_frames = max(len(meshes), len(cameras), len(lights))
        if len(meshes) == 1:
            meshes = meshes.extend(num_frames)
        if len(cameras) == 1:
            cameras = cameras.extend(num_frames)
        if len(lights) == 1:
            lights = lights.extend(num_frames)

        num_each_gpu = math.ceil(num_frames / len(device))
        slice_indices = [list(range(i * num_each_gpu, min((i + 1) * num_each_gpu, num_frames))) for i in range(len(device))]
        args = [dict(renderer=renderer, meshes=meshes[slice_indices[gpu_id]], device=device[gpu_id], cameras=cameras[slice_indices[gpu_id]], lights=lights[slice_indices[gpu_id]], batch_size=batch_size, no_grad=no_grad, verbose=verbose) for gpu_id in range(len(device))]
        with multiprocessing.Pool(processes=len(device)) as pool:
            rendered_frames = pool.map(unpack_kwargs_and_call_worker, args)
        rendered_frames = [frame.cpu() for frame in rendered_frames]
        rendered_frames = torch.cat(rendered_frames)
    else:
        rendered_frames = render(renderer=renderer, meshes=meshes, lights=lights, batch_size=batch_size, no_grad=no_grad, verbose=verbose, cameras=cameras, device=device[0])
    return rendered_frames


def render_mp_flow(renderer: Union[nn.Module, dict],
                    meshes_source: Union[Meshes, None] = None,
                    meshes_target: Union[Meshes, None] = None,
                    device: Union[str, torch.device] = 'cpu',
                    cameras: Union[NewCamerasBase, CamerasBase, dict, None] = None,
                    batch_size: int = 5,
                    no_grad: bool = True,
                    verbose: bool = True,):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    device = parse_device(device)
    if len(device) >1:
        num_frames = max(len(meshes), len(cameras), len(lights))
        if len(meshes) == 1:
            meshes = meshes.extend(num_frames)
        if len(cameras) == 1:
            cameras = cameras.extend(num_frames)
        if len(lights) == 1:
            lights = lights.extend(num_frames)

        num_each_gpu = math.ceil(num_frames / len(device))
        slice_indices = [list(range(i * num_each_gpu, min((i + 1) * num_each_gpu, num_frames))) for i in range(len(device))]
        args = [dict(renderer=renderer, meshes_source=meshes_source[slice_indices[gpu_id]], meshes_target=meshes_target[slice_indices[gpu_id]], cameras=cameras[slice_indices[gpu_id]], device=device[gpu_id], batch_size=batch_size, no_grad=no_grad, verbose=verbose) for gpu_id in range(len(device))]
        with multiprocessing.Pool(processes=len(device)) as pool:
            rendered_frames = pool.map(unpack_kwargs_and_call_worker_flow, args)
        rendered_frames = [frame.cpu() for frame in rendered_frames]
        rendered_frames = torch.cat(rendered_frames)
    else:
        rendered_frames = render_flow(renderer=renderer, meshes_source=meshes_source, meshes_target=meshes_target,  batch_size=batch_size, no_grad=no_grad, verbose=verbose, cameras=cameras, device=device[0])
    return rendered_frames
