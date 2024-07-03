from typing import Iterable, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes
from t3drender.cameras.cameras import NewCamerasBase
from t3drender.render.lights import BaseLights
from .base_renderer import BaseRenderer


class MeshRenderer(BaseRenderer):
    """Render RGBA image with the help of camera system."""
    shader_type = 'SoftPhongShader'

    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        device: Union[torch.device, str] = 'cpu',
        **kwargs,
    ) -> None:
        """Renderer for RGBA image of meshes.

        Args:
            resolution (Iterable[int]):
                (width, height) of the rendered images resolution.
            device (Union[torch.device, str], optional):
                You can pass a str or torch.device for cpu or gpu render.
                Defaults to 'cpu'.
        """
        super().__init__(resolution=resolution,
                         device=device,
                         **kwargs)

    def forward(self,
                meshes: Meshes,
                cameras: Optional[NewCamerasBase] = None,
                lights: Optional[BaseLights] = None,
                **kwargs) -> Union[torch.Tensor, None]:
        """Render Meshes.

        Args:
            meshes (Meshes): meshes to be rendered.
            cameras (Optional[NewCamerasBase], optional): cameras for render.
                Defaults to None.
            lights (Optional[BaseLights], optional): lights for render.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """

        meshes = meshes.to(self.device)
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)

        rendered_images = self.shader(
            fragments=fragments,
            meshes=meshes,
            cameras=cameras,
            lights=self.lights if lights is None else lights)

        return rendered_images
