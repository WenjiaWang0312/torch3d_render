import os.path as osp
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    DirectionalLights,
    Materials,
    MeshRasterizer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader
)

from t3drender.cameras.cameras import NewCamerasBase
from .utils import normalize, rgb2bgr, tensor2array


class BaseRenderer(nn.Module):

    def __init__(self,
                 resolution: Tuple[int, int] = None,
                 device: Union[torch.device, str] = 'cpu',
                 **kwargs) -> None:
        """BaseRenderer for differentiable rendering and visualization.

        Args:
            resolution (Iterable[int]):
                (width, height) of the rendered images resolution.
            device (Union[torch.device, str], optional):
                You can pass a str or torch.device for cpu or gpu render.
                Defaults to 'cpu'.

        **kwargs is used for render setting.
        You can set up your render kwargs like:
            {
                'shader': {
                    'type': 'soft_phong'
                },
                'lights': {
                        'type': 'directional',
                        'direction': [[10.0, 10.0, 10.0]],
                        'ambient_color': [[0.5, 0.5, 0.5]],
                        'diffuse_color': [[0.5, 0.5, 0.5]],
                        'specular_color': [[0.5, 0.5, 0.5]],
                    },
                'materials': {
                        'ambient_color': [[1, 1, 1]],
                        'diffuse_color': [[0.5, 0.5, 0.5]],
                        'specular_color': [[0.5, 0.5, 0.5]],
                        'shininess': 60.0,
                    },
                'rasterizer': {
                    'bin_size': 0,
                    'blur_radius': 0.0,
                    'faces_per_pixel': 1,
                    'perspective_correct': False,
                },
                'blend_params': {'background_color': (1.0, 1.0, 1.0)},
            },
        You can change any parameter in the suitable range, please check
        configs/render/smpl.py.

        Returns:
            None
        """
        super().__init__()
        self.device = device
        self.resolution = resolution
        self._init_renderer(**kwargs)

    def _init_renderer(self,
                       rasterizer: Union[dict, nn.Module, None] = None,
                       shader: Union[dict, nn.Module, None] = None,
                       materials: Union[dict, Materials, None] = None,
                       lights: Union[dict, DirectionalLights, PointLights,
                                     AmbientLights, None] = None,
                       blend_params: Union[dict, BlendParams, None] = None,
                       **kwargs):
        """Initial renderer."""
        if isinstance(materials, dict):
            materials = Materials(**materials)
        elif materials is None:
            materials = Materials()
        
        assert isinstance(materials, Materials), f'Wrong type of materials: {type(materials)}.'
        
        if isinstance(lights,
                        (AmbientLights, PointLights, DirectionalLights)):
            self.lights = lights
        elif lights is None:
            self.lights = AmbientLights()

        assert isinstance(self.lights, (AmbientLights, PointLights,
                                        DirectionalLights)), f'Wrong type of lights: {type(self.lights)}.'

        if isinstance(blend_params, dict):
            blend_params = BlendParams(**blend_params)
        elif blend_params is None:
            blend_params = BlendParams()

        assert isinstance(blend_params, BlendParams), f'Wrong type of blend_params: {type(blend_params)}.'

        if isinstance(rasterizer, nn.Module):
            if self.resolution is not None:
                rasterizer.raster_settings.image_size = self.resolution
            self.rasterizer = rasterizer
        elif isinstance(rasterizer, dict):
            if self.resolution is not None:
                rasterizer['image_size'] = self.resolution
            raster_settings = RasterizationSettings(**rasterizer)
            self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        elif rasterizer is None:
            self.rasterizer = MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.resolution,
                    bin_size=0,
                    blur_radius=0,
                    faces_per_pixel=1,
                    perspective_correct=False))
        
        assert isinstance(self.rasterizer, MeshRasterizer), f'Wrong type of rasterizer: {type(self.rasterizer)}.'

        if self.resolution is None:
            self.resolution = self.rasterizer.raster_settings.image_size
        assert self.resolution is not None
        self.resolution = (int(self.resolution),
                           int(self.resolution)) if isinstance(
                               self.resolution,
                               (float, int)) else tuple(self.resolution)
        if isinstance(shader, nn.Module):
            self.shader = shader
        elif shader is None:
            self.shader = SoftPhongShader(device=self.device, lights=self.lights, materials=materials, blend_params=blend_params)
        else:
            raise TypeError(f'Wrong type of shader: {type(shader)}.')
        self.shader.materials = materials
        self = self.to(self.device)

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if getattr(self.rasterizer, 'cameras', None) is not None:
            self.rasterizer.cameras = self.rasterizer.cameras.to(device)

        if getattr(self.shader, 'cameras', None) is not None:
            self.shader.cameras = self.shader.cameras.to(device)
        if getattr(self.shader, 'materials', None) is not None:
            self.shader.materials = self.shader.materials.to(device)
        if getattr(self.shader, 'lights', None) is not None:
            self.shader.lights = self.shader.lights.to(device)
        return self

    def _update_resolution(self, cameras, **kwargs):
        if isinstance(cameras, (NewCamerasBase)):
            self.resolution = (int(cameras.resolution[0][0]),
                               int(cameras.resolution[0][1]))
        if kwargs.get('resolution', None) is not None:
            self.resolution = kwargs.get('resolution')
        self.rasterizer.raster_settings.image_size = self.resolution

    def forward(self):
        """"Should be called by each sub renderer class."""
        raise NotImplementedError()
