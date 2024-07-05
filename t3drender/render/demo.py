import torch
import numpy as np
from pytorch3d.utils import ico_sphere
from t3drender.render.render_functions import render_rgb
from pytorch3d.renderer.mesh.textures import TexturesVertex
import cv2

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # You need to konw where to put your meshes
    meshes = ico_sphere(5, device)
    moved_verts = meshes.verts_padded()
    moved_verts[..., 2] += 2
    meshes = meshes.update_padded(moved_verts)

    # The default camera setting does not set extrinsic matrix, for advanced camera setting you need to feed R, T matrix into cameras
    meshes.textures = TexturesVertex(torch.randn_like(meshes.verts_padded()))
    meshes = meshes.extend(100)
    image_tensors = render_rgb(meshes, device=[0, 1], resolution=(512, 512), fov=90, batch_size=5, verbose=True)
    from IPython import embed; embed()
    image = image_tensors[..., :3].detach().cpu().numpy() * 256
    image = image.astype(np.uint8)