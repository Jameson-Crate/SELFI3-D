#make me a python script that takes in a point cloud and a set of points and returns a poisson surface reconstruction of the point cloud

import open3d as o3d
import numpy as np


def poisson_surface_reconstruction(points, normals=None, depth=8, scale=1.1, linear_fit=False, n_threads=-1):
    """
    Perform Poisson surface reconstruction on a point cloud.
    
    Parameters
    ----------
    points : numpy.ndarray
        Nx3 array of point coordinates.
    normals : numpy.ndarray, optional
        Nx3 array of point normals. If None, normals will be estimated.
    depth : int, default=8
        Maximum depth of the octree used for reconstruction.
    scale : float, default=1.1
        Scale factor to apply to the octree.
    linear_fit : bool, default=False
        If True, use linear interpolation to fit the implicit function.
    n_threads : int, default=-1
        Number of threads to use. -1 means use all available threads.
    
    Returns
    -------
    mesh : open3d.geometry.TriangleMesh
        The reconstructed mesh.
    """
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals if not provided
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit, n_threads=n_threads
    )
    
    return mesh


def save_mesh(mesh, output_path):
    """
    Save a mesh to a file.
    
    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The mesh to save.
    output_path : str
        Path to save the mesh to.
    
    Returns
    -------
    bool
        True if successful.
    """
    return o3d.io.write_triangle_mesh(output_path, mesh)


def load_point_cloud(input_path):
    """
    Load a point cloud from a file.
    
    Parameters
    ----------
    input_path : str
        Path to the point cloud file.
    
    Returns
    -------
    points : numpy.ndarray
        Nx3 array of point coordinates.
    normals : numpy.ndarray or None
        Nx3 array of point normals if available, None otherwise.
    """
    # Try to load the point cloud
    try:
        pcd = o3d.io.read_point_cloud(input_path)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        return points, normals
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return None, None


def transfer_colors(mesh, pcd, k=3):
    """
    Transfer colors from point cloud to mesh using nearest neighbors.
    
    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The mesh to color.
    pcd : open3d.geometry.PointCloud
        The point cloud with colors.
    k : int, default=3
        Number of nearest neighbors to consider for color averaging.
    
    Returns
    -------
    mesh : open3d.geometry.TriangleMesh
        The colored mesh.
    """
    if not pcd.has_colors():
        print("Point cloud has no colors to transfer")
        return mesh
    
    # Build KD-tree for the point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # Get mesh vertices
    mesh_vertices = np.asarray(mesh.vertices)
    
    # Initialize vertex colors
    vertex_colors = np.zeros((len(mesh_vertices), 3))
    
    # For each vertex in the mesh, find the nearest points in the point cloud
    for i, vertex in enumerate(mesh_vertices):
        # Find k nearest neighbors
        k_actual, idx, _ = pcd_tree.search_knn_vector_3d(vertex, k)
        
        # Get colors of the nearest points
        nearest_colors = np.asarray(pcd.colors)[idx[:k_actual]]
        
        # Average the colors
        vertex_colors[i] = np.mean(nearest_colors, axis=0)
    
    # Set the mesh colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    return mesh


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Poisson Surface Reconstruction")
    parser.add_argument("input", help="Input point cloud file")
    parser.add_argument("output", help="Output mesh file")
    parser.add_argument("--depth", type=int, default=8, help="Maximum depth of the octree")
    parser.add_argument("--scale", type=float, default=1.1, help="Scale factor")
    parser.add_argument("--linear", action="store_true", help="Use linear fit")
    parser.add_argument("--threads", type=int, default=-1, help="Number of threads")
    parser.add_argument("--density_threshold", type=float, default=0.0, help="Density threshold for removing faces")
    parser.add_argument("--color", action="store_true", help="Transfer colors from point cloud to mesh")
    parser.add_argument("--color_k", type=int, default=3, help="Number of nearest neighbors for color transfer")
    
    args = parser.parse_args()
    
    # Load point cloud
    points, normals = load_point_cloud(args.input)
    
    if points is not None:
        print(f"Loaded {len(points)} points")
        
        # Create point cloud object for later use
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Perform reconstruction
        mesh = poisson_surface_reconstruction(
            points, normals, 
            depth=args.depth, 
            scale=args.scale, 
            linear_fit=args.linear, 
            n_threads=args.threads
        )
        
        # Remove low-density vertices if threshold is provided
        if args.density_threshold > 0 and hasattr(mesh, 'vertex_density'):
            densities = np.asarray(mesh.vertex_density)
            vertices_to_remove = densities < args.density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Transfer colors if requested
        if args.color:
            # Check if point cloud has colors
            try:
                colors = np.asarray(pcd.colors)
                if len(colors) == 0:
                    # Try to extract colors from points if they exist
                    if points.shape[1] >= 6:  # x,y,z,r,g,b format
                        color_data = points[:, 3:6]
                        # Normalize if needed
                        if color_data.max() > 1:
                            color_data = color_data / 255.0
                        pcd.colors = o3d.utility.Vector3dVector(color_data)
                
                mesh = transfer_colors(mesh, pcd, k=args.color_k)
                print("Transferred colors from point cloud to mesh")
            except Exception as e:
                print(f"Failed to transfer colors: {e}")
        
        # Save the result
        if save_mesh(mesh, args.output):
            print(f"Mesh saved to {args.output}")
        else:
            print(f"Failed to save mesh to {args.output}")
    else:
        print("Failed to load point cloud")
