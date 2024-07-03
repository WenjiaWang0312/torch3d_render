from typing import Optional, Tuple, Union

import torch
from pytorch3d.structures import Meshes
from t3drender.cameras import NewCamerasBase
from .base_renderer import BaseRenderer


class SegmentationRenderer(BaseRenderer):
    """Render segmentation map into a segmentation index tensor."""
    shader_type = 'SegmentationShader'

    def __init__(self,
                 resolution: Tuple[int, int] = None,
                 device: Union[torch.device, str] = 'cpu',
                 palette: torch.Tensor = None,
                 **kwargs) -> None:
        """Render vertex-color mesh into a segmentation map of a (B, H, W)
        tensor. For visualization, the output rgba image will be (B, H, W, 4),
        and the color palette comes from `get_different_colors`. The
        segmentation map is a tensor each pixel saves the classification index.
        Please make sure you have allocate each pixel a correct classification
        index by defining a textures of vertex color.

        [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.
        CrossEntropyLoss.html)

        Args:
            resolution (Iterable[int]):
                (width, height) of the rendered images resolution.
            device (Union[torch.device, str], optional):
                You can pass a str or torch.device for cpu or gpu render.
                Defaults to 'cpu'.
            num_class (int, optional): number of segmentation parts.
                Defaults to 1.

        Returns:
            None
        """
        super().__init__(resolution=resolution,
                         device=device,
                         **kwargs)
        self.palette = palette

    def forward(self,
                meshes: Meshes,
                cameras: Optional[NewCamerasBase] = None,
                **kwargs):
        """Render segmentation map.

        Args:
            meshes (Meshes): meshes to be rendered.
                Require the textures type is `TexturesClosest`.
                The color indicates the class index of the triangle.
            cameras (Optional[NewCamerasBase], optional): cameras for render.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for images.
                Defaults to None.
            backgrounds (Optional[torch.Tensor], optional): background images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """

        meshes = meshes.to(self.device)
        self._update_resolution(cameras, **kwargs)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        segmentation_map = self.shader(fragments=fragments,
                                       meshes=meshes,
                                       cameras=cameras)

        return segmentation_map
