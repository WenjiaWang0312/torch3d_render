import torch
import numpy as np
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, SoftPhongShader
from t3drender.render.shaders import NoLightShader, DepthShader, SegmentationShader
from t3drender.cameras import PerspectiveCameras
from t3drender.render.lights import PointLights
import cv2


class UnifiedRenderer:
    """Unified renderer: compute fragments once, apply multiple shaders"""
    
    def __init__(self, resolution=(512, 512), device='cpu'):
        self.resolution = resolution
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Setup rasterizer (for computing fragments)
        raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        
        # Initialize shaders and move to correct device
        self.rgb_shader = SoftPhongShader(device=self.device)  # RGB shader with lighting
        self.rgb_no_light_shader = NoLightShader().to(self.device)  # RGB shader without lighting
        self.depth_shader = DepthShader().to(self.device)
        self.seg_shader = SegmentationShader().to(self.device)
        
        # Setup lights and move to correct device
        self.lights = PointLights(location=[[0.0, 0.0, 0.0]], device=self.device)
    
    def __call__(self, meshes, cameras, seg_meshes=None, use_lighting_rgb=True):
        """
        Compute fragments once and apply multiple shaders
        
        Args:
            meshes: Mesh with RGB textures
            cameras: Camera object
            seg_meshes: Mesh with segmentation mask textures
            use_lighting_rgb: Whether to use lighting in RGB rendering
        
        Returns:
            dict: Dictionary containing all rendering results
        """
        results = {}
        
        # 1. Compute fragments only once
        print("Computing fragments...")
        meshes = meshes.to(self.device)
        cameras = cameras.to(self.device)
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        
        # 2. Apply different shaders to the same fragments
        
        # RGB rendering (with optional lighting)
        if use_lighting_rgb:
            print("Rendering RGB (with lighting)...")
            rgb_images = self.rgb_shader(
                fragments=fragments,
                meshes=meshes,
                cameras=cameras,
                lights=self.lights
            )
        else:
            print("Rendering RGB (without lighting)...")
            rgb_images = self.rgb_no_light_shader(
                fragments=fragments,
                meshes=meshes
            )
        results['rgb'] = rgb_images
        
        # Depth rendering (always rendered, independent of lighting)
        print("Rendering depth map...")
        depth_images = self.depth_shader(
            fragments=fragments,
            meshes=meshes,
            cameras=cameras
        )
        results['depth'] = depth_images
        
        # Instance mask rendering (always rendered if seg_meshes provided, independent of lighting)
        if seg_meshes is not None:
            print("Rendering instance mask...")
            # For segmentation, we need to recompute fragments (different mesh)
            seg_meshes = seg_meshes.to(self.device)
            seg_fragments = self.rasterizer(meshes_world=seg_meshes, cameras=cameras)
            
            mask_images = self.seg_shader(
                fragments=seg_fragments,
                meshes=seg_meshes
            )
            results['mask'] = mask_images
        
        return results


def create_scene():
    """Create test scene"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create two spheres
    meshes1 = ico_sphere(5, device)
    meshes2 = ico_sphere(4, device)
    
    # Set positions
    moved_verts1 = meshes1.verts_padded()
    moved_verts1[..., 0] -= 0.8
    moved_verts1[..., 2] += 2
    meshes1 = meshes1.update_padded(moved_verts1)
    
    moved_verts2 = meshes2.verts_padded()
    moved_verts2[..., 0] += 0.8
    moved_verts2[..., 2] += 2.5
    meshes2 = meshes2.update_padded(moved_verts2)
    
    # RGB textures
    red_texture = torch.ones_like(meshes1.verts_padded()) * torch.tensor([1.0, 0.0, 0.0], device=device)
    green_texture = torch.ones_like(meshes2.verts_padded()) * torch.tensor([0.0, 1.0, 0.0], device=device)
    meshes1.textures = TexturesVertex(red_texture)
    meshes2.textures = TexturesVertex(green_texture)
    
    # Segmentation mask textures
    id1_texture = torch.ones_like(meshes1.verts_padded()) * torch.tensor([1.0, 1.0, 1.0], device=device)
    id2_texture = torch.ones_like(meshes2.verts_padded()) * torch.tensor([2.0, 2.0, 2.0], device=device)
    meshes1_seg = meshes1.clone()
    meshes2_seg = meshes2.clone()
    meshes1_seg.textures = TexturesVertex(id1_texture)
    meshes2_seg.textures = TexturesVertex(id2_texture)
    
    # Combine scene
    from t3drender.mesh_utils import join_meshes_as_scene
    rgb_scene = join_meshes_as_scene([meshes1, meshes2])
    seg_scene = join_meshes_as_scene([meshes1_seg, meshes2_seg])
    
    return rgb_scene, seg_scene, device


def save_results(results_lit, results_no_lit):
    """Save rendering results"""
    print("Saving results...")
    
    # Save RGB with lighting
    if 'rgb' in results_lit:
        rgb_data = results_lit['rgb'][0, ..., :3].detach().cpu().numpy() * 255
        rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
        cv2.imwrite('unified_rgb_with_lighting.png', cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))
    
    # Save RGB without lighting
    if 'rgb' in results_no_lit:
        rgb_data = results_no_lit['rgb'][0, ..., :3].detach().cpu().numpy() * 255
        rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
        cv2.imwrite('unified_rgb_no_lighting.png', cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))
    
    # Save depth (only once, same for both)
    if 'depth' in results_lit:
        depth_data = results_lit['depth'][0, ..., 0].detach().cpu().numpy()
        depth_normalized = (depth_data / depth_data.max() * 255).astype(np.uint8)
        cv2.imwrite('unified_depth.png', depth_normalized)
    
    # Save mask (only once, same for both)
    if 'mask' in results_lit:
        mask_data = results_lit['mask'][0, ..., 0].detach().cpu().numpy().astype(np.uint8)
        cv2.imwrite('unified_mask.png', mask_data * 100)


def main():
    """Main function: demonstrate unified rendering"""
    # Create scene
    rgb_scene, seg_scene, device = create_scene()
    
    # Setup camera
    resolution = (512, 512)
    focal_length = 512
    h, w = resolution
    K = torch.eye(3, 3, device=device)[None]
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 0, 2] = w / 2
    K[:, 1, 2] = h / 2
    cameras = PerspectiveCameras(in_ndc=False, K=K, convention='opencv', resolution=resolution)
    
    # Create unified renderer
    renderer = UnifiedRenderer(resolution=resolution, device=device)
    
    # Render RGB with lighting + depth + instance mask
    print("=== Rendering RGB (with lighting) + depth + instance mask ===")
    results_lit = renderer(
        meshes=rgb_scene,
        cameras=cameras,
        seg_meshes=seg_scene,
        use_lighting_rgb=True
    )
    
    # Render RGB without lighting (depth and mask will be the same)
    print("\n=== Rendering RGB (without lighting) ===")
    results_no_lit = renderer(
        meshes=rgb_scene,
        cameras=cameras,
        seg_meshes=seg_scene,
        use_lighting_rgb=False
    )
    
    # Save results
    save_results(results_lit, results_no_lit)
    
    print("\nRendering completed!")
    print("Output files:")
    print("- unified_rgb_with_lighting.png: RGB image (with lighting)")
    print("- unified_rgb_no_lighting.png: RGB image (without lighting)")
    print("- unified_depth.png: Depth map")
    print("- unified_mask.png: Instance mask")
    
    return results_lit, results_no_lit


if __name__ == "__main__":
    results_lit, results_no_lit = main()