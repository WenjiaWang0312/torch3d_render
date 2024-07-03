from typing import Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes

from t3drender.cameras import NewCamerasBase
from .base_renderer import BaseRenderer


class SilhouetteRenderer(BaseRenderer):
    """Silhouette renderer."""
    shader_type = 'SilhouetteShader'

    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        device: Union[torch.device, str] = 'cpu',
        **kwargs,
    ) -> None:
        """SilhouetteRenderer for neural rendering and visualization.

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
        """Render silhouette map.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Require the textures type is `TexturesClosest`.
                The color indicates the class index of the triangle.
                Defaults to None.
            cameras (Optional[NewCamerasBase], optional): cameras for render.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """

        meshes = meshes.to(self.device)
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        silhouette_map = self.shader(fragments=fragments,
                                     meshes=meshes,
                                     cameras=cameras)

        return silhouette_map
