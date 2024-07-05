import torch
from typing import Union, List
from t3drender.render.render_runner import render_mp
from t3drender.render.renderers import MeshRenderer, DepthRenderer, NormalRenderer, SilhouetteRenderer, SegmentationRenderer
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from t3drender.render.shaders import NormalShader, DepthShader
from t3drender.cameras import PerspectiveCameras, FoVOrthographicCameras
from t3drender.render.lights import PointLights
import multiprocessing


def render_rgb(meshes, device, resolution=(512, 512), fov=90, batch_size=30, verbose=False):
    K = torch.eye(3, 3)[None]
    h, w = resolution
    fov = torch.deg2rad(torch.tensor(fov,))
    focal_length = w / (2 * torch.tan(fov / 2))
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 0, 2] = w / 2
    K[:, 1, 2] = h / 2
    mesh_renderer = MeshRenderer(resolution=resolution, shader=SoftPhongShader())
    cameras = PerspectiveCameras(in_ndc=False, K=K, convention='opencv', resolution=resolution)
    lights = PointLights(location=[[0.0, 0.0, 0.0]])

    rendered_frames = render_mp(renderer=mesh_renderer, meshes=meshes, lights=lights, batch_size=batch_size, verbose=verbose, cameras=cameras, device=device)
    return rendered_frames


def render_depth_perspective(meshes, device, resolution=(512, 512), fov=90, batch_size=30, verbose=False):
    K = torch.eye(3, 3)[None]
    h, w = resolution
    fov = torch.deg2rad(torch.tensor(fov,))
    focal_length = w / (2 * torch.tan(fov / 2))
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 0, 2] = w / 2
    K[:, 1, 2] = h / 2
    depth_renderer = DepthRenderer(resolution=resolution, device=device, shader=DepthShader())
    cameras = PerspectiveCameras(device=device, in_ndc=False, K=K, convention='opencv', resolution=resolution)
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

    rendered_frames = render_mp(renderer=depth_renderer, meshes=meshes, lights=lights, batch_size=batch_size, verbose=verbose, cameras=cameras, device=device)
    return rendered_frames


def render_depth_orthographic(meshes, device, resolution=(512, 512), batch_size=30, verbose=False):
    depth_renderer = DepthRenderer(resolution=resolution, device=device, shader=DepthShader())
    cameras = FoVOrthographicCameras(device=device, max_x=1, max_y=1, min_x=-1, min_y=-1, resolution=resolution)

    rendered_frames = render_mp(renderer=depth_renderer, meshes=meshes, batch_size=batch_size, verbose=verbose, cameras=cameras, device=device)
    return rendered_frames


def render_normal(meshes, device, resolution=(512, 512), fov=90, batch_size=30, verbose=False):
    K = torch.eye(3, 3)[None]
    h, w = resolution
    fov = torch.deg2rad(torch.tensor(fov,))
    focal_length = w / (2 * torch.tan(fov / 2))
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 0, 2] = w / 2
    K[:, 1, 2] = h / 2
    normal_renderer = NormalRenderer(resolution=resolution, device=device, shader=NormalShader())
    cameras = PerspectiveCameras(device=device, in_ndc=False, K=K, convention='opencv', resolution=resolution)
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

    rendered_frames = render_mp(renderer=normal_renderer, meshes=meshes, lights=lights, batch_size=batch_size, verbose=verbose, cameras=cameras, device=device)
    return rendered_frames