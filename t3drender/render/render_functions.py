import torch
from typing import Union, List
from t3drender.render.render_runner import render_mp, render_mp_flow
from t3drender.render.renderers import MeshRenderer, DepthRenderer, NormalRenderer, OpticalFlowRenderer, SilhouetteRenderer, SegmentationRenderer
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from t3drender.render.shaders import NormalShader, DepthShader, OpticalFlowShader, SegmentationShader
from t3drender.cameras import PerspectiveCameras, FoVOrthographicCameras
from t3drender.render.lights import PointLights
import multiprocessing


def render_rgb(meshes, device, resolution=(512, 512), focal_length=512, cameras=None, batch_size=30, verbose=False):
    K = torch.eye(3, 3)[None]
    h, w = resolution
    mesh_renderer = MeshRenderer(resolution=resolution, shader=SoftPhongShader())

    if cameras is None:
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 0, 2] = w / 2
        K[:, 1, 2] = h / 2
        cameras = PerspectiveCameras(in_ndc=False, K=K, convention='opencv', resolution=resolution)
    lights = PointLights(location=[[0.0, 0.0, 0.0]])

    rendered_frames = render_mp(renderer=mesh_renderer, meshes=meshes, lights=lights, batch_size=batch_size, verbose=verbose, cameras=cameras, device=device)
    return rendered_frames

def render_segmentation(meshes, device, resolution=(512, 512), focal_length=512, cameras=None, batch_size=30, verbose=False):
    K = torch.eye(3, 3)[None]
    h, w = resolution
    mesh_renderer = SegmentationRenderer(resolution=resolution, shader=SegmentationShader())
    if cameras is None:
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 0, 2] = w / 2
        K[:, 1, 2] = h / 2
        cameras = PerspectiveCameras(in_ndc=False, K=K, convention='opencv', resolution=resolution)
    rendered_frames = render_mp(renderer=mesh_renderer, meshes=meshes,  batch_size=batch_size, verbose=verbose, cameras=cameras, device=device)
    return rendered_frames

def render_depth_perspective(meshes, device, resolution=(512, 512), focal_length=512, cameras=None, batch_size=30, no_grad=True, verbose=False):
    K = torch.eye(3, 3)[None]
    h, w = resolution
    depth_renderer = DepthRenderer(resolution=resolution, shader=DepthShader())

    if cameras is None:
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 0, 2] = w / 2
        K[:, 1, 2] = h / 2
        cameras = PerspectiveCameras(in_ndc=False, K=K, convention='opencv', resolution=resolution)
    lights = PointLights(location=[[0.0, 0.0, 0.0]])

    rendered_frames = render_mp(renderer=depth_renderer, meshes=meshes, lights=lights, batch_size=batch_size, verbose=verbose, cameras=cameras, device=device, no_grad=no_grad)
    return rendered_frames

def render_flow_perspective(meshes_source, meshes_target, device, resolution=(512, 512), focal_length=512, cameras=None, no_grad=True, batch_size=30, verbose=False):
    K = torch.eye(3, 3)[None]
    h, w = resolution
    flow_renderer = OpticalFlowRenderer(resolution=resolution, shader=OpticalFlowShader())

    if cameras is None:
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 0, 2] = w / 2
        K[:, 1, 2] = h / 2
        cameras = PerspectiveCameras(in_ndc=False, K=K, convention='opencv', resolution=resolution)
    rendered_frames = render_mp_flow(renderer=flow_renderer, meshes_source=meshes_source, meshes_target=meshes_target, batch_size=batch_size, verbose=verbose, cameras=cameras, device=device, no_grad=no_grad)
    return rendered_frames

def render_depth_orthographic(meshes, device, resolution=(512, 512), batch_size=30, verbose=False):
    depth_renderer = DepthRenderer(resolution=resolution, shader=DepthShader())
    cameras = FoVOrthographicCameras(max_x=1, max_y=1, min_x=-1, min_y=-1, resolution=resolution)

    rendered_frames = render_mp(renderer=depth_renderer, meshes=meshes, batch_size=batch_size, verbose=verbose, cameras=cameras, device=device)
    return rendered_frames

def render_normal(meshes, device, resolution=(512, 512), focal_length=512, cameras=None, batch_size=30, verbose=False):
    K = torch.eye(3, 3)[None]
    h, w = resolution
    normal_renderer = NormalRenderer(resolution=resolution, shader=NormalShader())
    if cameras is None:
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 0, 2] = w / 2
        K[:, 1, 2] = h / 2
        cameras = PerspectiveCameras(in_ndc=False, K=K, convention='opencv', resolution=resolution)
    lights = PointLights(location=[[0.0, 0.0, 0.0]])

    rendered_frames = render_mp(renderer=normal_renderer, meshes=meshes, lights=lights, batch_size=batch_size, verbose=verbose, cameras=cameras, device=device)
    return rendered_frames