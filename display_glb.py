import open3d as o3d
import sys
import numpy as np
import trimesh
from PIL import Image

def project_texture_on_mesh(mesh, image_path, center_point=None, size_scale=1.0):
    """
    Project a 2D texture onto the mesh, centered at a specified point.
    For black and white transparent images, only project the black parts.
    
    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        The mesh to texture
    image_path : str
        Path to the texture image
    center_point : np.ndarray or None
        3D point to center the texture at. If None, uses mesh center
    size_scale : float
        Scale factor for the texture size (1.0 = default size)
    """
    # Load and process the image
    img = Image.open(image_path)
    
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Convert to numpy array and normalize
    img_np = np.array(img) / 255.0
    
    # Split into RGB and alpha channels
    rgb = img_np[:, :, :3]
    alpha = img_np[:, :, 3]
    
    # Calculate darkness of each pixel (inverse of brightness)
    # For grayscale images, all RGB channels are the same
    darkness = 1 - np.mean(rgb, axis=2)
    
    # Create a mask for black parts (dark + not transparent)
    black_mask = (darkness > 0.5) & (alpha > 0.5)
    
    # Get mesh vertices
    vertices = np.asarray(mesh.vertices)
    
    # If no center point specified, use mesh center
    if center_point is None:
        center_point = np.mean(vertices, axis=0)
    center_point = np.array(center_point)
    
    # Calculate the projection plane
    up_vector = np.array([0, 1, 0])
    front_vector = np.array([0, 0, -1])
    right_vector = np.cross(up_vector, front_vector)
    
    # Calculate vertex positions relative to center
    relative_positions = vertices - center_point
    
    # Project vertices onto the plane
    u = np.dot(relative_positions, right_vector)
    v = np.dot(relative_positions, up_vector)
    
    # Scale the projection to match image aspect ratio
    aspect_ratio = img_np.shape[1] / img_np.shape[0]
    u = u / (np.max(np.abs(u)) * 2) * aspect_ratio
    v = v / (np.max(np.abs(v)) * 2)
    
    # Apply size scaling
    u = u * size_scale
    v = v * size_scale
    
    # Convert to image coordinates
    u = (u + 0.5) * (img_np.shape[1] - 1)
    v = (1 - (v + 0.5)) * (img_np.shape[0] - 1)
    
    # Clip coordinates to image bounds
    u = np.clip(u, 0, img_np.shape[1] - 1)
    v = np.clip(v, 0, img_np.shape[0] - 1)
    
    # Convert to integers for indexing
    u_int = u.astype(int)
    v_int = v.astype(int)
    
    # Get the mask values for each vertex
    vertex_darkness = darkness[v_int, u_int]
    vertex_alpha = alpha[v_int, u_int]
    
    # Initialize colors array with existing colors or white
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
    else:
        colors = np.ones_like(vertices)
    
    # Apply distance-based fade out
    dist_to_center = np.linalg.norm(relative_positions, axis=1)
    max_dist = np.max(dist_to_center) * 0.5
    fade = np.clip(1 - dist_to_center / max_dist, 0, 1)
    
    # Create black color mask
    is_black = (vertex_darkness > 0.5) & (vertex_alpha > 0.5)
    
    # Apply black color only where the mask is True
    black_color = np.zeros(3)  # Pure black
    for i in range(len(vertices)):
        if is_black[i]:
            # Blend between original color and black based on fade and alpha
            blend_factor = fade[i] * vertex_alpha[i] * vertex_darkness[i]
            colors[i] = colors[i] * (1 - blend_factor) + black_color * blend_factor
    
    return colors

def load_and_display_glb(glb_path, image_path=None, center_point=None, size_scale=1.0):
    """
    Load and display a GLB file using Open3D, optionally with a projected texture
    """
    try:
        print(f"Loading GLB file: {glb_path}")
        
        # Load the GLB file using trimesh
        scene = trimesh.load(glb_path)
        
        if isinstance(scene, trimesh.Scene):
            print("Loaded a scene with multiple meshes")
            meshes = []
            for name, geometry in scene.geometry.items():
                vertices = np.array(geometry.vertices)
                faces = np.array(geometry.faces)
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                
                if image_path:
                    # Project texture onto this mesh
                    colors = project_texture_on_mesh(mesh, image_path, center_point, size_scale)
                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                elif hasattr(geometry.visual, 'vertex_colors'):
                    colors = geometry.visual.vertex_colors[:, :3] / 255.0
                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                
                mesh.compute_vertex_normals()
                meshes.append(mesh)
                
            print(f"Processed {len(meshes)} meshes")
            
        else:
            print("Loaded a single mesh")
            vertices = np.array(scene.vertices)
            faces = np.array(scene.faces)
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            if image_path:
                # Project texture onto the mesh
                colors = project_texture_on_mesh(mesh, image_path, center_point, size_scale)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            elif hasattr(scene.visual, 'vertex_colors'):
                colors = scene.visual.vertex_colors[:, :3] / 255.0
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            mesh.compute_vertex_normals()
            meshes = [mesh]
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add all meshes to the visualizer
        for mesh in meshes:
            vis.add_geometry(mesh)
        
        # Set up the camera view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
        opt.point_size = 1.0
        opt.show_coordinate_frame = True
        
        print("Displaying mesh. Press 'q' to exit.")
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python display_glb.py <glb_file> [image_file] [x y z] [scale]")
        print("  glb_file: Path to the GLB file")
        print("  image_file: Optional path to texture image")
        print("  x y z: Optional center coordinates for texture projection")
        print("  scale: Optional scale factor for texture size (default: 1.0)")
        sys.exit(1)
    
    glb_path = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Parse center point if provided
    center_point = None
    if len(sys.argv) > 5:
        try:
            x, y, z = map(float, sys.argv[3:6])
            center_point = np.array([x, y, z])
        except ValueError:
            print("Error: Center coordinates must be three floating-point numbers")
            sys.exit(1)
    
    # Parse scale if provided
    size_scale = 1.0
    if len(sys.argv) > 6:
        try:
            size_scale = float(sys.argv[6])
        except ValueError:
            print("Error: Scale must be a floating-point number")
            sys.exit(1)
    
    load_and_display_glb(glb_path, image_path, center_point, size_scale)

if __name__ == "__main__":
    main() 