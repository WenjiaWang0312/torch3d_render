import warnings
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
)
from pytorch3d.structures import Meshes, Pointclouds

from t3drender.cameras import NewCamerasBase
from .base_renderer import BaseRenderer


class PointCloudRenderer(BaseRenderer):

    def __init__(self,
                 resolution: Tuple[int, int] = None,
                 device: Union[torch.device, str] = 'cpu',
                 radius: Optional[float] = None,
                 **kwargs) -> None:
        """Point cloud renderer.

        Args:
            resolution (Iterable[int]):
                (width, height) of the rendered images resolution.
            device (Union[torch.device, str], optional):
                You can pass a str or torch.device for cpu or gpu render.
                Defaults to 'cpu'.
            radius (float, optional): radius of points. Defaults to None.

        Returns:
            None
        """
        self.radius = radius
        super().__init__(resolution=resolution,
                         device=device,
                         **kwargs)

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if getattr(self.rasterizer, 'cameras', None) is not None:
            self.rasterizer.cameras = self.rasterizer.cameras.to(device)

        self.compositor = self.compositor.to(device)
        return self

    def _init_renderer(self, rasterizer=None, compositor=None, **kwargs):
        """Set render params."""

        if isinstance(rasterizer, nn.Module):
            rasterizer.raster_settings.image_size = self.resolution
            self.rasterizer = rasterizer
        elif isinstance(rasterizer, dict):
            rasterizer['image_size'] = self.resolution
            if self.radius is not None:
                rasterizer.update(radius=self.radius)
            raster_settings = PointsRasterizationSettings(**rasterizer)
            self.rasterizer = PointsRasterizer(raster_settings=raster_settings)
        elif rasterizer is None:
            self.rasterizer = PointsRasterizer(
                raster_settings=PointsRasterizationSettings(
                    radius=self.radius,
                    image_size=self.resolution,
                    points_per_pixel=10))
        else:
            raise TypeError(
                f'Wrong type of rasterizer: {type(self.rasterizer)}.')

        if isinstance(compositor, dict):
            self.compositor = AlphaCompositor(**compositor)
        elif isinstance(compositor, nn.Module):
            self.compositor = compositor
        elif compositor is None:
            self.compositor = AlphaCompositor()
        else:
            raise TypeError(
                f'Wrong type of compositor: {type(self.compositor)}.')
        self = self.to(self.device)

    def forward(
        self,
        pointclouds: Optional[Pointclouds] = None,
        vertices: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        verts_rgba: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        cameras: Optional[NewCamerasBase] = None,
        **kwargs,
    ) -> Union[None, torch.Tensor]:
        """Render pointclouds.

        Args:
            pointclouds (Optional[Pointclouds], optional): pytorch3d data
                structure. If not None, `vertices` and `verts_rgba` will
                be ignored.
                Defaults to None.
            vertices (Optional[Union[torch.Tensor, List[torch.Tensor]]],
                optional): coordinate tensor of points. Defaults to None.
            verts_rgba (Optional[Union[torch.Tensor, List[torch.Tensor]]],
                optional): color tensor of points. Defaults to None.

        Returns:
            Union[None, torch.Tensor]: Return tensor or None.
        """
        if pointclouds is None:
            assert vertices is not None
            if isinstance(vertices, torch.Tensor):
                if vertices.ndim == 2:
                    vertices = vertices[None]
            if isinstance(verts_rgba, torch.Tensor):
                if verts_rgba.ndim == 2:
                    verts_rgba = verts_rgba[None]
            pointclouds = Pointclouds(points=vertices, features=verts_rgba)
        else:
            if vertices is not None or verts_rgba is not None:
                warnings.warn(
                    'Redundant input, will ignore `vertices` and `verts_rgb`.')
        pointclouds = pointclouds.to(self.device)
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(pointclouds, cameras=cameras)
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        rendered_images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            pointclouds.features_packed().permute(1, 0),
            **kwargs,
        )
        rendered_images = rendered_images.permute(0, 2, 3, 1)

        return rendered_images
