import cv2
import numpy as np
import trimesh
import open3d
import torch
import warnings
from typing import List, Optional, Union
from packaging import version
from pytorch3d.io import IO
from pytorch3d.io.obj_io import save_obj, load_objs_as_meshes
from pytorch3d.renderer.mesh.textures import TexturesUV, TexturesVertex
from pytorch3d.structures import (Meshes, Pointclouds, join_meshes_as_scene, join_meshes_as_batch,
                                  list_to_padded, padded_to_list)
from pytorch3d.transforms import Rotate

from pytorch3d.utils.ico_sphere import ico_sphere
from pytorch3d.renderer.mesh import TexturesVertex


def load_plys_as_meshes(
    files: Union[str, List[str]],
    device: Optional[Union[torch.device, str]] = None,
    load_textures: bool = True,
) -> Meshes:
    writer = IO()
    meshes = []
    if not isinstance(files, list):
        files = [files]
    for idx in range(len(files)):
        assert files[idx].endswith('.ply'), 'Please input .ply files.'
        mesh = writer.load_mesh(
            path=files[idx], include_textures=load_textures, device=device)
        meshes.append(mesh)
    meshes = join_meshes_as_batch(meshes, include_textures=load_textures)
    return meshes


def save_meshes_as_plys(files: Union[str, List[str]],
                        meshes: Optional[Meshes] = None) -> None:
    """Save meshes as .ply files. Mainly for vertex color meshes.

    Args:
        files (List[str]): Output .ply file list.
        meshes (Meshes, optional): higher priority than
            (verts & faces & verts_rgb). Defaults to None.
        verts (torch.Tensor, optional): lower priority than meshes.
            Defaults to None.
        faces (torch.Tensor, optional): lower priority than meshes.
            Defaults to None.
        verts_rgb (torch.Tensor, optional): lower priority than meshes.
            Defaults to None.
    """
    assert files is not None
    if not isinstance(files, list):
        files = [files]
    assert len(files) >= len(meshes), 'Not enough output files.'
    writer = IO()
    for idx in range(len(meshes)):
        assert files[idx].endswith('.ply'), 'Please save as .ply files.'
        writer.save_mesh(
            meshes[idx], files[idx], colors_as_uint8=True, binary=False)


def save_meshes_as_objs(files: Union[str, List[str]], meshes: Meshes = None) -> None:
    """Save meshes as .obj files. Pytorch3D will not save vertex color for.

    .obj, please use `save_meshes_as_plys`.

    Args:
        files (List[str]): Output .obj file list.
        meshes (Meshes, optional):
            Defaults to None.
    """
    if not isinstance(files, list):
        files = [files]

    assert len(files) >= len(meshes), 'Not enough output files.'

    for idx in range(len(meshes)):
        if isinstance(meshes.textures, TexturesUV):
            verts_uvs = meshes.textures.verts_uvs_padded()[idx]
            faces_uvs = meshes.textures.faces_uvs_padded()[idx]
            texture_map = meshes.textures.maps_padded()[idx]
        else:
            verts_uvs = None
            faces_uvs = None
            texture_map = None
        save_obj(
            f=files[idx],
            verts=meshes.verts_padded()[idx],
            faces=meshes.faces_padded()[idx],
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            texture_map=texture_map)


def join_batch_meshes_as_scene(
    meshes: List[Meshes],
    include_textures: bool = True,
) -> Meshes:
    """Join `meshes` as a scene each batch. Only for Pytorch3D `meshes`. The
    Meshes must share the same batch size, and topology could be different.
    They must all be on the same device. If `include_textures` is true, the
    textures should be the same type, all be None is not accepted. If
    `include_textures` is False, textures are ignored. The return meshes will
    have no textures.

    Args:
        meshes (List[Meshes]): A `list` of `Meshes` with the same batches.
            Required.
        include_textures: (bool) whether to try to join the textures.

    Returns:
        New Meshes which has join different Meshes by each batch.
    """
    for mesh in meshes:
        mesh._verts_list = padded_to_list(mesh.verts_padded(),
                                          mesh.num_verts_per_mesh().tolist())
    num_scene_size = len(meshes)
    num_batch_size = len(meshes[0])
    for i in range(num_scene_size):
        assert len(
            meshes[i]
        ) == num_batch_size, 'Please make sure that the Meshes all have'
        'the same batch size.'
    meshes_all = []
    for j in range(num_batch_size):
        meshes_batch = []
        for i in range(num_scene_size):
            meshes_batch.append(meshes[i][j])
        meshes_all.append(join_meshes_as_scene(meshes_batch, include_textures))
    meshes_final = join_meshes_as_batch(meshes_all, include_textures)
    return meshes_final


def get_pointcloud_mesh(verts_padded, level=0, radius=0.01, colors=None):
    device = verts_padded.device
    B, N, _ = verts_padded.shape  # B, N, 3
    sphere = ico_sphere(level).to(device)
    n = sphere.verts_padded().shape[1]
    f = sphere.faces_padded().shape[1]
    verts = radius * sphere.verts_padded()[:, None].repeat(B, N, 1,
                                                           1)  #B, N, n, 3
    faces = sphere.faces_padded()[:, None].repeat(B, N, 1, 1)  #B, N, f, 3
    faces_offsets = torch.arange(0, N)[None, :, None, None].repeat(
        B, 1, 1, 1) * n  # B, N, 1, 1
    verts = verts + verts_padded[:, :, None]
    faces = faces + faces_offsets.to(device)
    textures = None
    if colors is not None:
        colors = colors.to(device)[:, :, None].repeat_interleave(n,
                                                                 2)  #B, N,n, 3

        colors = colors.view(B, N * n, 3)
        textures = TexturesVertex(colors)
    verts = verts.view(B, N * n, 3)
    faces = faces.view(B, N * f, 3)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return meshes




o3d = open3d
vec3d = o3d.utility.Vector3dVector
vec2d = o3d.utility.Vector2dVector
vec3i = o3d.utility.Vector3iVector
PointCloud_o3d = o3d.geometry.PointCloud
TriangleMesh = o3d.geometry.TriangleMesh

new_version = version.parse(o3d.__version__) > version.parse('0.9.0')

def texture_uv2vc_t3d(meshes: Meshes):
    device = meshes.device
    vert_uv = meshes.textures.verts_uvs_padded()
    batch_size = vert_uv.shape[0]
    verts_features = []
    num_verts = meshes.verts_padded().shape[1]
    for index in range(batch_size):
        face_uv = vert_uv[index][meshes.textures.faces_uvs_padded()
                                 [index].view(-1)]
        img = meshes.textures._maps_padded[index]
        width, height, _ = img.shape
        face_uv = face_uv * torch.Tensor([width, height]).long().to(device)
        face_uv[:, 0] = torch.clip(face_uv[:, 0], 0, width - 1)
        face_uv[:, 1] = torch.clip(face_uv[:, 1], 0, height - 1)
        face_uv = face_uv.long()
        faces = meshes.faces_padded()
        verts_rgb = torch.zeros(1, num_verts, 3).to(device)
        verts_rgb[:, faces.view(-1)] = img[height - face_uv[:, 1], face_uv[:,
                                                                           0]]
        verts_features.append(verts_rgb)
    verts_features = torch.cat(verts_features)
    meshes = meshes.clone()
    meshes.textures = TexturesVertex(verts_features)
    return meshes

def t3d_to_o3d_mesh(
    meshes: Meshes,
    include_textures: bool = True,
) -> List[TriangleMesh]:
    """
    Convert pytorch3d Meshes to open3d TriangleMesh.
    Since open3d 0.9.0.0 doesn't support batch meshes, we only feed single
    `Meshes` of batch N. Will return a list(N) of `TriangleMesh`.

    Args:
        meshes (Meshes): batched `Meshes`.
            Defaults to None.
        include_textures (bool, optional): whether contain textures.
            Defaults to False.
    Returns:
        List[TriangleMesh]: return a list of open3d `TriangleMesh`.
    """
    meshes_o3d = []
    if meshes is not None:
        vertices = meshes.verts_padded().clone().detach().cpu().numpy()
        faces = meshes.faces_padded().clone().detach().cpu().numpy()
        textures = meshes.textures.clone().detach()
        batch_size = len(meshes)
    else:
        raise ValueError('The input mesh is None. Please pass right inputs.')
    for index in range(batch_size):
        mesh_o3d = TriangleMesh(
            vertices=vec3d(vertices[index]), triangles=vec3i(faces[index]))
        if include_textures:
            if isinstance(textures, TexturesVertex):
                mesh_o3d.vertex_colors = vec3d(
                    textures.verts_features_padded()
                    [index].detach().cpu().numpy())
            elif isinstance(textures, TexturesUV):
                vert_uv = textures.verts_uvs_padded()[index]
                face_uv = vert_uv[textures.faces_uvs_padded()[index].view(-1)]

                img = textures._maps_padded.cpu().numpy()[index]
                img = (img * 255).astype(np.uint8)
                img = cv2.flip(img, 0)
                if new_version:
                    mesh_o3d.textures = [o3d.geometry.Image(img)]
                    mesh_o3d.triangle_uvs = vec2d(
                        face_uv.detach().cpu().numpy())
                else:
                    mesh_o3d.triangle_uvs = list(
                        face_uv.detach().cpu().numpy())
                    mesh_o3d.texture = o3d.geometry.Image(img)
            elif textures is None:
                warnings.warn('Cannot load textures from original mesh.')
        meshes_o3d.append(mesh_o3d)
    return meshes_o3d

def o3d_to_t3d_mesh(meshes: Optional[Union[List[TriangleMesh],
                                           TriangleMesh]] = None,
                    include_textures: bool = True) -> Meshes:
    """
    Convert open3d TriangleMesh to pytorch3d Meshes .
    Args:
        meshes (Optional[Union[List[TriangleMesh], TriangleMesh]], optional):
            [description]. Defaults to None.
        include_textures (bool, optional): [description]. Defaults to True.

    Returns:
        Meshes: [description]
    """

    if not isinstance(meshes, list):
        meshes = [meshes]
    vertices = [torch.Tensor(np.asarray(mesh.vertices)) for mesh in meshes]

    vertices = list_to_padded(vertices, pad_value=0.0)
    faces = [torch.Tensor(np.asarray(mesh.triangles)) for mesh in meshes]
    faces = list_to_padded(faces, pad_value=-1.0)
    if include_textures:
        has_vertex_colors = meshes[0].has_vertex_colors()
        if new_version:
            has_textures = meshes[0].has_textures()
        else:
            has_textures = meshes[0].has_texture()
        if has_vertex_colors:
            features = [
                torch.Tensor(np.asarray(mesh.vertex_colors)) for mesh in meshes
            ]

            features = list_to_padded(features, pad_value=0.0)
            textures = TexturesVertex(verts_features=features)
        elif has_textures:
            if new_version:
                maps = [
                    torch.Tensor(
                        np.asarray(mesh.textures[0]).astype(np.float32))
                    for mesh in meshes
                ]
            else:
                maps = [
                    torch.Tensor(np.asarray(mesh.texture).astype(np.float32))
                    for mesh in meshes
                ]
            maps = list_to_padded(maps, pad_size=0) / 255.0
            faces_uvs = []
            verts_uvs = []
            for mesh in meshes:
                faces_uv = np.asarray(mesh.triangles)
                verts_uv = np.zeros((vertices.shape[1], 2))
                verts_uv[faces_uv.reshape(-1)] = np.asarray(mesh.triangle_uvs)
                faces_uvs.append(torch.Tensor(faces_uv))
                verts_uvs.append(torch.Tensor(verts_uv))
            faces_uvs = list_to_padded(faces_uvs, pad_value=0.0)
            verts_uvs = list_to_padded(verts_uvs, pad_value=0.0)
            textures = TexturesUV(
                maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
        else:
            warnings.warn('Cannot load textures from original mesh.')
            textures = None
    else:
        textures = None
    meshes_t3d = Meshes(verts=vertices, faces=faces, textures=textures)
    return meshes_t3d


def get_checkerboard_plane(plane_width=4, num_boxes=9, center=True):

    pw = plane_width/num_boxes
    white = [220, 220, 220, 255]
    black = [35, 35, 35, 255]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            ground = trimesh.primitives.Box(
                center=[0, 0, -0.0001],
                extents=[pw, pw, 0.0002]
            )

            if center:
                c = c[0]+(pw/2)-(plane_width/2), c[1]+(pw/2)-(plane_width/2)
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.apply_translation([c[0], c[1], 0])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.visual.face_colors = black if ((i+j) % 2) == 0 else white
            meshes.append(ground)
    meshes = trimesh.util.concatenate(meshes)
    return meshes


def create_checkerboard_mesh(N, M, square_size, center):

    x_start = center[0] - (M / 2) * square_size
    y_start = center[1] - (N / 2) * square_size
    z = center[2]

    vertices = []
    faces = []
    colors = []

    for i in range(N):
        for j in range(M):
            x0 = x_start + j * square_size
            y0 = y_start + i * square_size
            x1 = x0 + square_size
            y1 = y0 + square_size

            v_idx = len(vertices)

            vertices.extend([
                [x0, y0, z],
                [x1, y0, z],
                [x1, y1, z],
                [x0, y1, z]
            ])

            faces.extend([
                [v_idx, v_idx + 1, v_idx + 2],
                [v_idx, v_idx + 2, v_idx + 3]
            ])

            color = [1.0, 1.0, 1.0] if (i + j) % 2 == 0 else [0.5, 0.5, 0.5]
            colors.extend([color] * 4)

    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)
    colors = torch.tensor(colors, dtype=torch.float32)

    textures = TexturesVertex(verts_features=colors.unsqueeze(0))

    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    
    return mesh