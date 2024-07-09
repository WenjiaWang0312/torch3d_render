from typing import Iterable, Optional, Union

import torch
from pytorch3d.structures import Meshes
from t3drender.cameras.cameras import NewCamerasBase
from .base_renderer import BaseRenderer



class NormalRenderer(BaseRenderer):
    """Render normal map with the help of camera system."""
    shader_type = 'NormalShader'

    def __init__(
        self,
        resolution: Iterable[int] = None,
        device: Union[torch.device, str] = 'cpu',
        **kwargs,
    ) -> None:
        """Renderer for normal map of meshes.

        Args:
            resolution (Iterable[int]):
                (width, height) of the rendered images resolution.
            device (Union[torch.device, str], optional):
                You can pass a str or torch.device for cpu or gpu render.
                Defaults to 'cpu'.

        Returns:
            None
        """
        super().__init__(resolution=resolution,
                         device=device,
                         **kwargs)

    def forward(self,
                meshes: Optional[Meshes] = None,
                cameras: Optional[NewCamerasBase] = None,
                **kwargs):
        """Render Meshes.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Defaults to None.
            cameras (Optional[NewCamerasBase], optional): cameras for render.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        
        meshes = meshes.to(self.device)
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        normal_map = self.shader(fragments=fragments,
                                 meshes=meshes,
                                 cameras=cameras)

        return normal_map


class PolarNoramlRenderer(NormalRenderer):
    shader_type = 'PolarNormalShader'
