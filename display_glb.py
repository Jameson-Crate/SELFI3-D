import open3d as o3d
import sys
import numpy as np
import trimesh
from PIL import Image

def project_texture_on_mesh(mesh, image_path, center_point=None, size_scale=1.0):
    img = Image.open(image_path)
    
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    img_np = np.array(img) / 255.0
    
    rgb = img_np[:, :, :3]
    alpha = img_np[:, :, 3]
    
    darkness = 1 - np.mean(rgb, axis=2)
    black_mask = (darkness > 0.5) & (alpha > 0.5)
    vertices = np.asarray(mesh.vertices)
    
    if center_point is None:
        center_point = np.mean(vertices, axis=0)
    center_point = np.array(center_point)
    
    up_vector = np.array([0, 1, 0])
    front_vector = np.array([0, 0, -1])
    right_vector = np.cross(up_vector, front_vector)
    
    relative_positions = vertices - center_point
    
    u = np.dot(relative_positions, right_vector)
    v = np.dot(relative_positions, up_vector)
    
    aspect_ratio = img_np.shape[1] / img_np.shape[0]
    u = u / (np.max(np.abs(u)) * 2) * aspect_ratio
    v = v / (np.max(np.abs(v)) * 2)
    
    u = u * size_scale
    v = v * size_scale
    
    u = (u + 0.5) * (img_np.shape[1] - 1)
    v = (1 - (v + 0.5)) * (img_np.shape[0] - 1)
    
    u = np.clip(u, 0, img_np.shape[1] - 1)
    v = np.clip(v, 0, img_np.shape[0] - 1)
    
    u_int = u.astype(int)
    v_int = v.astype(int)
    
    vertex_darkness = darkness[v_int, u_int]
    vertex_alpha = alpha[v_int, u_int]
    
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
    else:
        colors = np.ones_like(vertices)
    
    dist_to_center = np.linalg.norm(relative_positions, axis=1)
    max_dist = np.max(dist_to_center) * 0.5
    fade = np.clip(1 - dist_to_center / max_dist, 0, 1)
    
    is_black = (vertex_darkness > 0.5) & (vertex_alpha > 0.5)
    
    black_color = np.zeros(3)
    for i in range(len(vertices)):
        if is_black[i]:
            blend_factor = fade[i] * vertex_alpha[i] * vertex_darkness[i]
            colors[i] = colors[i] * (1 - blend_factor) + black_color * blend_factor
    return colors

def load_and_display_glb(glb_path, image_path=None, center_point=None, size_scale=1.0):
    try:
        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            meshes = []
            for name, geometry in scene.geometry.items():
                vertices = np.array(geometry.vertices)
                faces = np.array(geometry.faces)
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                
                if image_path:
                    colors = project_texture_on_mesh(mesh, image_path, center_point, size_scale)
                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                elif hasattr(geometry.visual, 'vertex_colors'):
                    colors = geometry.visual.vertex_colors[:, :3] / 255.0
                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                
                mesh.compute_vertex_normals()
                meshes.append(mesh)
                            
        else:
            vertices = np.array(scene.vertices)
            faces = np.array(scene.faces)
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            if image_path:
                colors = project_texture_on_mesh(mesh, image_path, center_point, size_scale)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            elif hasattr(scene.visual, 'vertex_colors'):
                colors = scene.visual.vertex_colors[:, :3] / 255.0
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            mesh.compute_vertex_normals()
            meshes = [mesh]
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        for mesh in meshes:
            vis.add_geometry(mesh)
        
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        opt.point_size = 1.0
        opt.show_coordinate_frame = True
        
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Not enough args!")
        sys.exit(1)
    
    glb_path = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    center_point = None
    if len(sys.argv) > 5:
        try:
            x, y, z = map(float, sys.argv[3:6])
            center_point = np.array([x, y, z])
        except ValueError:
            sys.exit(1)
    
    size_scale = 1.0
    if len(sys.argv) > 6:
        try:
            size_scale = float(sys.argv[6])
        except ValueError:
            print("Scale must be a floating-point number")
            sys.exit(1)
    
    load_and_display_glb(glb_path, image_path, center_point, size_scale)

if __name__ == "__main__":
    main() 
