#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import argparse
from pathlib import Path

def load_pointcloud(file_path):
    """
    Load a pointcloud from a file based on its extension.
    Supports .txt, .xyz, .csv, .ply, and .nvm files.
    
    Args:
        file_path (str or Path): Path to the pointcloud file
        
    Returns:
        tuple: (points, colors) where points is a numpy array of shape (N, 3)
               and colors is a numpy array of shape (N, 3) or None
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext in ['.txt', '.xyz', '.csv']:
        data = np.loadtxt(file_path, delimiter=' ')
        points = data[:, :3]  # Extract xyz coordinates
        
        # If there are additional columns (like RGB), keep them as colors
        colors = data[:, 3:6] if data.shape[1] >= 6 else None
        
        # Normalize colors if they are in range [0, 255]
        if colors is not None and np.max(colors) > 1.0:
            colors = colors / 255.0
            
    elif ext == '.ply':
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
    elif ext == '.nvm':
        points, colors = parse_nvm_file(file_path)
        
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported extensions are .txt, .xyz, .csv, .ply, and .nvm")
    
    return points, colors

def parse_nvm_file(nvm_file_path):
    """
    Parse a COLMAP .nvm file and extract the point cloud data.
    
    Args:
        nvm_file_path (str or Path): Path to the .nvm file
        
    Returns:
        tuple: (points, colors) where points is a numpy array of shape (N, 3)
               and colors is a numpy array of shape (N, 3)
    """
    with open(nvm_file_path, 'r') as f:
        lines = f.readlines()
    
    # Check if the file is in NVM format
    if not lines[0].strip().startswith('NVM_V3'):
        raise ValueError("The file does not appear to be in NVM_V3 format")
    
    # Find where the camera section ends and point data begins
    line_idx = 2  # Skip header and empty line
    
    # Handle empty lines or non-integer values
    while line_idx < len(lines) and not lines[line_idx].strip():
        line_idx += 1
    
    if line_idx >= len(lines):
        raise ValueError("File format error: unexpected end of file")
    
    try:
        num_cameras = int(lines[line_idx].strip())
    except ValueError:
        raise ValueError(f"Invalid number of cameras: '{lines[line_idx].strip()}'")
    
    line_idx += num_cameras + 1  # Skip camera data
    
    # Ensure we haven't gone past the end of the file
    if line_idx >= len(lines):
        raise ValueError("File format error: no point data found")
    
    # Get number of points, handling empty lines
    while line_idx < len(lines) and not lines[line_idx].strip():
        line_idx += 1
    
    if line_idx >= len(lines):
        raise ValueError("File format error: no point count found")
    
    try:
        num_points = int(lines[line_idx].strip())
    except ValueError:
        raise ValueError(f"Invalid number of points: '{lines[line_idx].strip()}'")
    
    line_idx += 1
    
    points = []
    colors = []
    
    # Parse each point
    for i in range(num_points):
        if line_idx + i >= len(lines):
            break
            
        line = lines[line_idx + i].strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 7:  # At minimum we need x, y, z, r, g, b
            continue
            
        try:
            # Extract position
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            
            # Extract color
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            
            points.append([x, y, z])
            colors.append([r/255.0, g/255.0, b/255.0])  # Normalize to [0,1]
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping invalid point data: {line}")
            continue
    
    return np.array(points), np.array(colors)

def remove_outliers(points, colors=None, method='statistical', nb_neighbors=20, std_ratio=2.0, radius=0.1, min_neighbors=16):
    """
    Remove outliers from a point cloud using either statistical or radius outlier removal.
    
    Args:
        points (np.ndarray): Array of 3D points of shape (N, 3)
        colors (np.ndarray, optional): Array of RGB colors for each point of shape (N, 3)
        method (str): Method to use for outlier removal ('statistical' or 'radius')
        nb_neighbors (int): Number of neighbors to use for statistical outlier removal
        std_ratio (float): Standard deviation ratio for statistical outlier removal
        radius (float): Radius to use for radius outlier removal
        min_neighbors (int): Minimum number of neighbors for radius outlier removal
        
    Returns:
        tuple: (filtered_points, filtered_colors) where filtered_points is a numpy array of shape (M, 3)
               and filtered_colors is a numpy array of shape (M, 3) or None
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Apply outlier removal
    if method == 'statistical':
        print(f"Applying statistical outlier removal (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
        filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elif method == 'radius':
        print(f"Applying radius outlier removal (radius={radius}, min_neighbors={min_neighbors})")
        filtered_pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    else:
        raise ValueError(f"Unknown outlier removal method: {method}. Use 'statistical' or 'radius'.")
    
    # Extract filtered points and colors
    filtered_points = np.asarray(filtered_pcd.points)
    filtered_colors = np.asarray(filtered_pcd.colors) if filtered_pcd.has_colors() else None
    
    return filtered_points, filtered_colors

def save_pointcloud(file_path, points, colors=None):
    """
    Save a pointcloud to a file based on its extension.
    
    Args:
        file_path (str or Path): Path to save the pointcloud
        points (np.ndarray): Array of 3D points of shape (N, 3)
        colors (np.ndarray, optional): Array of RGB colors for each point of shape (N, 3)
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext in ['.txt', '.xyz', '.csv']:
        if colors is not None:
            # If colors are in range [0, 1], scale to [0, 255]
            if np.max(colors) <= 1.0:
                colors_to_save = colors * 255
            else:
                colors_to_save = colors
            
            # Combine points and colors
            data = np.hstack((points, colors_to_save))
        else:
            data = points
        
        np.savetxt(file_path, data, delimiter=' ')
    
    elif ext == '.ply':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(str(file_path), pcd)
    
    else:
        raise ValueError(f"Unsupported output file extension: {ext}. Supported extensions are .txt, .xyz, .csv, and .ply")

def visualize_point_cloud(points, colors=None, original_points=None, original_colors=None):
    """
    Visualize the point cloud using Open3D.
    
    Args:
        points (np.ndarray): Array of 3D points
        colors (np.ndarray, optional): Array of RGB colors for each point
        original_points (np.ndarray, optional): Array of original 3D points for comparison
        original_colors (np.ndarray, optional): Array of original RGB colors for comparison
    """
    geometries = []
    
    # Create filtered point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Use default color (blue) for filtered points
        pcd.paint_uniform_color([0, 0, 1])
    
    geometries.append(pcd)
    
    # Create original point cloud if provided
    if original_points is not None:
        orig_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(original_points)
        
        if original_colors is not None:
            orig_pcd.colors = o3d.utility.Vector3dVector(original_colors)
        else:
            # Use default color (red) for original points
            orig_pcd.paint_uniform_color([1, 0, 0])
        
        # Only show points that were removed
        if len(points) < len(original_points):
            # Find indices of removed points
            removed_indices = []
            for i, p in enumerate(original_points):
                if not np.any(np.all(points == p, axis=1)):
                    removed_indices.append(i)
            
            removed_points = original_points[removed_indices]
            removed_pcd = o3d.geometry.PointCloud()
            removed_pcd.points = o3d.utility.Vector3dVector(removed_points)
            removed_pcd.paint_uniform_color([1, 0, 0])  # Red for removed points
            geometries.append(removed_pcd)
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coordinate_frame)
    
    # Visualize
    o3d.visualization.draw_geometries(geometries)

def main():
    parser = argparse.ArgumentParser(description='Remove outliers from a point cloud')
    parser.add_argument('input_file', help='Path to input point cloud file (.txt, .xyz, .csv, .ply, or .nvm)')
    parser.add_argument('--output_file', help='Path to output point cloud file. If not specified, will use input_file_filtered[.ext]')
    parser.add_argument('--method', choices=['statistical', 'radius'], default='statistical', 
                        help='Method for outlier removal: statistical or radius (default: statistical)')
    parser.add_argument('--nb_neighbors', type=int, default=20, 
                        help='Number of neighbors to consider for statistical outlier removal (default: 20)')
    parser.add_argument('--std_ratio', type=float, default=2.0, 
                        help='Standard deviation ratio for statistical outlier removal (default: 2.0)')
    parser.add_argument('--radius', type=float, default=0.1, 
                        help='Radius for radius outlier removal (default: 0.1)')
    parser.add_argument('--min_neighbors', type=int, default=16, 
                        help='Minimum number of neighbors for radius outlier removal (default: 16)')
    parser.add_argument('--visualize', action='store_true', help='Visualize the point cloud before and after filtering')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_filtered{input_path.suffix}")
    
    print(f"Loading point cloud from {input_path}")
    points, colors = load_pointcloud(input_path)
    
    print(f"Original point cloud has {len(points)} points")
    
    # Store original points and colors for visualization
    original_points = points.copy() if args.visualize else None
    original_colors = colors.copy() if colors is not None and args.visualize else None
    
    # Remove outliers
    filtered_points, filtered_colors = remove_outliers(
        points, 
        colors, 
        method=args.method,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        radius=args.radius,
        min_neighbors=args.min_neighbors
    )
    
    print(f"Filtered point cloud has {len(filtered_points)} points")
    print(f"Removed {len(points) - len(filtered_points)} outlier points")
    
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
