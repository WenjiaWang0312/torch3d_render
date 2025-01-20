from typing import Iterable, Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes
from t3drender.cameras.cameras import NewCamerasBase
from .base_renderer import BaseRenderer
from .utils import normalize


class DepthRenderer(BaseRenderer):
    """Render depth map with the help of camera system."""
    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        device: Union[torch.device, str] = 'cpu',
        return_fragments: bool = False,
        **kwargs,
    ) -> None:
        """Renderer for depth map of meshes.

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
        self.return_fragments = return_fragments

    def forward(self,
                meshes: Optional[Meshes] = None,
                cameras: Optional[NewCamerasBase] = None,
                **kwargs):
        """Render depth map.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Defaults to None.
            cameras (Optional[NewCamerasBase], optional): cameras for rendering.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        meshes = meshes.to(self.device)
        self._update_resolution(cameras, **kwargs)

        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        depth_map = self.shader(fragments=fragments,
                                meshes=meshes,
                                cameras=cameras)
        return depth_map

