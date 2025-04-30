import numpy as np
import open3d as o3d
import argparse
import os

def parse_nvm_file(nvm_file_path):
    """
    Parse a COLMAP .nvm file and extract the point cloud data.
    
    Args:
        nvm_file_path (str): Path to the .nvm file
        
    Returns:
        np.ndarray: Array of 3D points
        np.ndarray: Array of RGB colors for each point
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

def visualize_point_cloud(points, colors):
    """
    Visualize the point cloud using Open3D.
    
    Args:
        points (np.ndarray): Array of 3D points
        colors (np.ndarray): Array of RGB colors for each point
    """
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, coordinate_frame])

def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud from COLMAP .nvm file')
    parser.add_argument('--nvm_file', type=str, required=True, help='Path to the .nvm file')
    args = parser.parse_args()
    
    if not os.path.exists(args.nvm_file):
        print(f"Error: File {args.nvm_file} does not exist")
        return
    
    try:
        points, colors = parse_nvm_file(args.nvm_file)
        if len(points) == 0:
            print("No points found in the NVM file")
            return
            
        print(f"Loaded {len(points)} points from {args.nvm_file}")
        visualize_point_cloud(points, colors)
    except Exception as e:
        print(f"Error processing the NVM file: {e}")

if __name__ == "__main__":
    main()
