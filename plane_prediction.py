#!/usr/bin/env python3
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

def load_pointcloud(file_path):
    """Load a point cloud from a file."""
    print(f"Loading point cloud from {file_path}")
    
    # Get file extension
    ext = file_path.suffix.lower()
    
    # Load the point cloud based on file extension
    if ext == '.ply':
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    elif ext == '.pcd':
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    elif ext in ['.xyz', '.txt', '.csv']:
        data = np.loadtxt(str(file_path), delimiter=',')
        points = data[:, :3]
        colors = data[:, 3:6] / 255.0 if data.shape[1] >= 6 else None
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return points, colors

def save_pointcloud(file_path, points, colors=None):
    """Save a point cloud to a file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Get file extension
    ext = file_path.suffix.lower()
    
    # Save the point cloud based on file extension
    if ext == '.ply':
        o3d.io.write_point_cloud(str(file_path), pcd, write_ascii=False)
    elif ext == '.pcd':
        o3d.io.write_point_cloud(str(file_path), pcd)
    elif ext in ['.xyz', '.txt', '.csv']:
        data = np.hstack((points, colors * 255)) if colors is not None else points
        np.savetxt(str(file_path), data, delimiter=',')
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def remove_plane(points, colors=None, distance_threshold=0.01, ransac_n=3, 
                num_iterations=1000, plane_offset_threshold=0.05):
    """
    Remove points that belong to a plane and points within a threshold above/below the plane.
    
    Args:
        points: Nx3 array of point coordinates
        colors: Nx3 array of point colors (optional)
        distance_threshold: Maximum distance a point can be from the plane to be considered an inlier
        ransac_n: Number of points to sample for RANSAC
        num_iterations: Number of RANSAC iterations
        plane_offset_threshold: Threshold for removing points above and below the plane
        
    Returns:
        Filtered points and colors (if provided)
    """
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Segment plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    # Extract plane parameters (a, b, c, d) from ax + by + cz + d = 0
    a, b, c, d = plane_model
    
    # Calculate distance of each point to the plane
    # Distance = (ax + by + cz + d) / sqrt(a² + b² + c²)
    norm = np.sqrt(a*a + b*b + c*c)
    distances = np.abs(np.dot(points, [a, b, c]) + d) / norm
    
    # Create a mask for points that are not on the plane and not within threshold
    mask = distances > plane_offset_threshold
    
    # Filter points and colors
    filtered_points = points[mask]
    filtered_colors = colors[mask] if colors is not None else None
    
    return filtered_points, filtered_colors

def visualize_point_cloud(filtered_points, filtered_colors, original_points=None, original_colors=None):
    """Visualize the original and filtered point clouds."""
    if original_points is not None:
        # Create original point cloud
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(original_points)
        if original_colors is not None:
            original_pcd.colors = o3d.utility.Vector3dVector(original_colors)
        else:
            # Set to red if no colors
            original_pcd.paint_uniform_color([1, 0, 0])
    
    # Create filtered point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    else:
        # Set to green if no colors
        filtered_pcd.paint_uniform_color([0, 1, 0])
    
    # Visualize
    if original_points is not None:
        o3d.visualization.draw_geometries([original_pcd, filtered_pcd],
                                         window_name="Original (red) vs Filtered (green)")
    else:
        o3d.visualization.draw_geometries([filtered_pcd],
                                         window_name="Filtered Point Cloud")

def main():
    parser = argparse.ArgumentParser(description='Remove plane from point cloud')
    parser.add_argument('input_file', help='Input point cloud file')
    parser.add_argument('--output_file', help='Output point cloud file')
    parser.add_argument('--distance_threshold', type=float, default=0.01,
                        help='Maximum distance a point can be from the plane to be considered an inlier')
    parser.add_argument('--ransac_n', type=int, default=3,
                        help='Number of points to sample for RANSAC')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Number of RANSAC iterations')
    parser.add_argument('--plane_offset_threshold', type=float, default=0.05,
                        help='Threshold for removing points above and below the plane')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the point cloud before and after filtering')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_no_plane{input_path.suffix}")
    
    # Load point cloud
    points, colors = load_pointcloud(input_path)
    print(f"Original point cloud has {len(points)} points")
    
    # Store original points and colors for visualization
    original_points = points.copy() if args.visualize else None
    original_colors = colors.copy() if colors is not None and args.visualize else None
    
    # Remove plane
    filtered_points, filtered_colors = remove_plane(
        points,
        colors,
        distance_threshold=args.distance_threshold,
        ransac_n=args.ransac_n,
        num_iterations=args.num_iterations,
        plane_offset_threshold=args.plane_offset_threshold
    )
    
    print(f"Filtered point cloud has {len(filtered_points)} points")
    print(f"Removed {len(points) - len(filtered_points)} points on or near the plane")
    
    # Save filtered point cloud
    print(f"Saving filtered point cloud to {output_path}")
    save_pointcloud(output_path, filtered_points, filtered_colors)
    
    # Visualize if requested
    if args.visualize:
        print("Visualizing point cloud...")
        visualize_point_cloud(filtered_points, filtered_colors, original_points, original_colors)
    
    print("Done")

if __name__ == "__main__":
    main()
