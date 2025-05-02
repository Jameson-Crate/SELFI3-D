import numpy as np
import open3d as o3d
from PIL import Image
import sys
import os

def project_image_on_point_cloud(pcd, image_path, center_point=None, size_scale=1):
    """Project a black and white transparent image onto the point cloud"""
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
    darkness = 1 - np.mean(rgb, axis=2)
    
    # Get point cloud data
    points = np.asarray(pcd.points)
    
    # If no center point specified, use point cloud center
    if center_point is None:
        center_point = np.mean(points, axis=0)
    center_point = np.array(center_point)
    
    # Calculate the projection plane
    up_vector = np.array([0, 1, 0])
    front_vector = np.array([0, 0, -1])
    right_vector = np.cross(up_vector, front_vector)
    
    # Calculate positions relative to center
    relative_positions = points - center_point
    
    # Project points onto the plane
    u = np.dot(relative_positions, right_vector)
    v = np.dot(relative_positions, up_vector)
    
    # Get the scale of the point cloud for normalization
    max_dim = max(np.max(np.abs(u)), np.max(np.abs(v)))
    
    # Scale both coordinates uniformly while maintaining aspect ratio
    # Divide by size_scale to make the projection smaller as scale increases
    scale_factor = 1.0 / (max_dim * 2 * size_scale)
    u = u * scale_factor
    v = v * scale_factor
    
    # Convert to image coordinates (now in range -0.5 to 0.5)
    u = (u + 0.5) * (img_np.shape[1] - 1)
    v = (1 - (v + 0.5)) * (img_np.shape[0] - 1)
    
    # Clip coordinates to image bounds
    u = np.clip(u, 0, img_np.shape[1] - 1)
    v = np.clip(v, 0, img_np.shape[0] - 1)
    
    # Convert to integers for indexing
    u_int = u.astype(int)
    v_int = v.astype(int)
    
    # Get the mask values for each point
    point_darkness = darkness[v_int, u_int]
    point_alpha = alpha[v_int, u_int]
    
    # Initialize colors array with existing colors or white
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points)
    
    # Apply distance-based fade out
    dist_to_center = np.linalg.norm(relative_positions, axis=1)
    max_dist = np.max(dist_to_center) * 0.5
    fade = np.clip(1 - dist_to_center / max_dist, 0, 1)
    
    # Create black color mask (dark + not transparent)
    is_black = (point_darkness > 0.5) & (point_alpha > 0.5)
    
    # Apply black color only where the mask is True
    black_color = np.zeros(3)  # Pure black
    for i in range(len(points)):
        if is_black[i]:
            # Blend between original color and black based on fade and alpha
            blend_factor = fade[i] * point_alpha[i] * point_darkness[i]
            colors[i] = colors[i] * (1 - blend_factor) + black_color * blend_factor
    
    # Create new colored point cloud
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return colored_pcd

def visualize_point_cloud(pcd):
    """Visualize a single point cloud"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometry
    vis.add_geometry(pcd)
    
    # Set default camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
    opt.point_size = 2.0  # Larger point size for better visibility
    
    print("Displaying point cloud. Press 'q' to exit.")
    vis.run()
    vis.destroy_window()

def main():
    if len(sys.argv) < 3:
        print("Usage: python point_cloud_display.py <ply_file> <image_file> [x y z] [scale]")
        print("  ply_file: Path to the PLY point cloud file")
        print("  image_file: Path to the transparent tattoo image")
        print("  x y z: Optional center coordinates for projection (default: point cloud center)")
        print("  scale: Optional scale factor for tattoo size (default: 0.05)")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    image_path = sys.argv[2]
    
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
    size_scale = 0.1  # Default to 5% of original size
    if len(sys.argv) > 6:
        try:
            size_scale = float(sys.argv[6])
        except ValueError:
            print("Error: Scale must be a floating-point number")
            sys.exit(1)
    
    try:
        # Load the point cloud
        print(f"Loading point cloud from {ply_path}...")
        pcd = o3d.io.read_point_cloud(ply_path)
        print(f"Loaded point cloud with {len(pcd.points)} points")
        
        # Project image onto point cloud
        print(f"Projecting image {image_path} onto point cloud...")
        colored_pcd = project_image_on_point_cloud(pcd, image_path, center_point, size_scale)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(ply_path))[0]
        output_path = f"{base_name}_tattooed.ply"
        
        # Save the processed point cloud
        o3d.io.write_point_cloud(output_path, colored_pcd)
        print(f"\nSaved processed point cloud to: {output_path}")
        
        # Visualize
        print("\nDisplaying result...")
        visualize_point_cloud(colored_pcd)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 