import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

def load_pointcloud(file_path):
    pcd = o3d.io.read_point_cloud(str(file_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors

def save_pointcloud(file_path, points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(file_path), pcd, write_ascii=False)

def remove_plane(points, colors=None, distance_threshold=0.01, ransac_n=3, 
                num_iterations=1000, plane_offset_threshold=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    a, b, c, d = plane_model
    norm = np.sqrt(a*a + b*b + c*c)
    distances = np.abs(np.dot(points, [a, b, c]) + d) / norm
    
    mask = distances > plane_offset_threshold
    filtered_points = points[mask]
    filtered_colors = colors[mask] if colors is not None else None
    return filtered_points, filtered_colors

def visualize_point_cloud(filtered_points, filtered_colors, original_points=None, original_colors=None):
    o3d.visualization.ViewControl.destroy_all_windows()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud", width=1280, height=720, visible=True)
    
    vis.clear_geometries()
    vis.get_render_option().load_from_json("")
    vis.get_view_control().load_from_json("")
    
    vc = vis.get_view_control()
    vc.set_zoom(0.7)
    vc.set_front([0, 0, -1])
    vc.set_lookat([0, 0, 0])
    vc.set_up([0, -1, 0])
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    opt.show_coordinate_frame = False
    opt.light_on = False
    
    if original_points is not None:
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(original_points)
        if original_colors is not None:
            original_pcd.colors = o3d.utility.Vector3dVector(original_colors)
        else:
            original_pcd.paint_uniform_color([1, 0, 0])
        vis.add_geometry(original_pcd)
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    else:
        filtered_pcd.paint_uniform_color([0, 1, 0])
    vis.add_geometry(filtered_pcd)
    
    vis.update_geometry(filtered_pcd)
    vis.poll_events()
    vis.update_renderer()
    
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Remove plane from point cloud')
    parser.add_argument('input_file')
    parser.add_argument('--output_file')
    parser.add_argument('--distance_threshold', type=float, default=0.01)
    parser.add_argument('--ransac_n', type=int, default=3)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--plane_offset_threshold', type=float, default=0.05)
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        print("Error: Input file does not exist")
        return
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_no_plane{input_path.suffix}")
    
    points, colors = load_pointcloud(input_path)
    print("Point cloud loaded")
    
    original_points = points.copy() if args.visualize else None
    original_colors = colors.copy() if colors is not None and args.visualize else None
    
    filtered_points, filtered_colors = remove_plane(
        points,
        colors,
        distance_threshold=args.distance_threshold,
        ransac_n=args.ransac_n,
        num_iterations=args.num_iterations,
        plane_offset_threshold=args.plane_offset_threshold
    )
        
    print("Saving filtered point cloud")
    save_pointcloud(output_path, filtered_points, filtered_colors)
    
    if args.visualize:
        print("Visualizing point cloud...")
        visualize_point_cloud(filtered_points, filtered_colors, original_points, original_colors)
    
if __name__ == "__main__":
    main()
