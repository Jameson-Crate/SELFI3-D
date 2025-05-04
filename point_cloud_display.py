import numpy as np
import open3d as o3d
from PIL import Image
import sys
import os

def project_image_on_point_cloud(pcd, image_path, center_point=None, size_scale=1):
    img = Image.open(image_path)
    img_np = np.array(img) / 255.0
    rgb = img_np[:, :, :3]
    alpha = img_np[:, :, 3]

    darkness = 1 - np.mean(rgb, axis=2)
    points = np.asarray(pcd.points)
    
    if center_point is None:
        center_point = np.mean(points, axis=0)
    center_point = np.array(center_point)
    print("\nProjection Information:")
    print(f"Center point: [{center_point[0]:.3f}, {center_point[1]:.3f}, {center_point[2]:.3f}]")
    
    front_vector = np.array([0, 0, 1])  
    up_vector = np.array([0, 1, 0])     
    right_vector = np.cross(up_vector, front_vector)  
    
    projection_offset = 0.1 
    projection_center = center_point + front_vector * projection_offset
    print(f"Projection plane center: [{projection_center[0]:.3f}, {projection_center[1]:.3f}, {projection_center[2]:.3f}]")
    print(f"Projection offset: {projection_offset:.3f} units in front of face")
    print(f"Projection scale: {size_scale:.3f}")
    
    relative_positions = points - projection_center
    
    u = np.dot(relative_positions, right_vector)
    v = np.dot(relative_positions, up_vector)
    
    max_dim = max(np.max(np.abs(u)), np.max(np.abs(v)))
    
    scale_factor = 1.0 / (max_dim * 2 * size_scale)
    u = u * scale_factor
    v = v * scale_factor
    
    u = (u + 0.5) * (img_np.shape[1] - 1)
    v = (1 - (v + 0.5)) * (img_np.shape[0] - 1)
    
    u = np.clip(u, 0, img_np.shape[1] - 1)
    v = np.clip(v, 0, img_np.shape[0] - 1)
    
    u_int = u.astype(int)
    v_int = v.astype(int)
    
    point_darkness = darkness[v_int, u_int]
    point_alpha = alpha[v_int, u_int]
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points)
    
    dist_to_plane = np.abs(np.dot(relative_positions, front_vector))
    max_dist = np.max(dist_to_plane) * 0.3  
    fade = np.clip(1 - dist_to_plane / max_dist, 0, 1)
    
    is_black = (point_darkness > 0.5) & (point_alpha > 0.5)
    black_color = np.zeros(3) 
    for i in range(len(points)):
        if is_black[i]:
            blend_factor = fade[i] * point_alpha[i] * point_darkness[i]
            colors[i] = colors[i] * (1 - blend_factor) + black_color * blend_factor
    
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    return colored_pcd

def visualize_point_cloud(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    vis.add_geometry(pcd)
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    opt.point_size = 2.0
    
    print("Displaying point cloud. Press 'q' to exit.")
    vis.run()
    vis.destroy_window()

def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    image_path = sys.argv[2]
    
    center_point = None
    if len(sys.argv) > 5:
        try:
            x, y, z = map(float, sys.argv[3:6])
            center_point = np.array([x, y, z])
        except ValueError:
            print("Error: Center coordinates must be three floating-point numbers")
            sys.exit(1)
    
    size_scale = 0.05  
    if len(sys.argv) > 6:
        try:
            size_scale = float(sys.argv[6])
        except ValueError:
            print("Error: Scale must be a floating-point number")
            sys.exit(1)
    
    print("Loading point cloud")
    pcd = o3d.io.read_point_cloud(ply_path)

    print("Projecting image onto point cloud...")
    colored_pcd = project_image_on_point_cloud(pcd, image_path, center_point, size_scale)
    
    base_name = os.path.splitext(os.path.basename(ply_path))[0]
    output_path = f"{base_name}_tattooed.ply"
    
    o3d.io.write_point_cloud(output_path, colored_pcd)
    print("Saved processed point cloud")
    
    print("\nDisplaying result...")
    visualize_point_cloud(colored_pcd)

if __name__ == "__main__":
    main() 
