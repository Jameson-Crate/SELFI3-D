#make me a python script that takes in a point cloud and a set of points and returns a poisson surface reconstruction of the point cloud

import open3d as o3d
import numpy as np

def poisson_surface_reconstruction(points, normals=None, depth=8, scale=1.1, linear_fit=False, n_threads=-1):
    pcd = o3d.geometry.PointCloud() #create a point cloud object
    pcd.points = o3d.utility.Vector3dVector(points) #set the points of the point cloud to the points in the point cloud
    
    # Estimate normals if not provided - this doesnt work so made another function below
    if normals is None:
        pcd.estimate_normals() #built in function to estimate normals
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals) #set the normals of the point cloud to the normals in the point cloud 
    
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit, n_threads=n_threads
    ) #built in function to create a mesh from a point cloud
    
    return mesh


def save_mesh(mesh, output_path):
    return o3d.io.write_triangle_mesh(output_path, mesh)


def load_point_cloud(input_path):
   
    # Try to load the point cloud
    try:
        pcd = o3d.io.read_point_cloud(input_path) #built in function to read a point cloud from a file
        points = np.asarray(pcd.points) #convert the points of the point cloud to a numpy array
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None #convert the normals of the point cloud to a numpy array if the point cloud has normals
        return points, normals
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return None, None


def transfer_colors(mesh, pcd, k=3):
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


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0): #THIS FUNCTION DOES NOT WORK
    print(f"Removing outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl


def estimate_normals(points, k=30, orient_with_viewpoint=True, viewpoint=None):
   
    print(f"Estimating normals (k={k})...")
    
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals - FIX: use knn instead of k
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)) #this is a better function to do
    
    # Orient normals
    if orient_with_viewpoint:
        if viewpoint is None:
            viewpoint = [0, 0, 0]  
        pcd.orient_normals_towards_camera_location(viewpoint)
    
    # Try to make normals consistent
    try:
        pcd.orient_normals_consistent_tangent_plane(k=k)
    except Exception as e:
        print(f"Warning: Could not orient normals consistently: {e}")
    
    return np.asarray(pcd.normals)


def save_point_cloud_with_normals(file_path, points, normals, colors=None): #this is what will be put into cloud compare
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return o3d.io.write_point_cloud(file_path, pcd)


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description="Point Cloud Processing with Normal Estimation")
    parser.add_argument("input", help="Input point cloud file")
    parser.add_argument("output", help="Output point cloud file with normals")
    parser.add_argument("--estimate_normals", action="store_true", help="Estimate normals for the point cloud")
    parser.add_argument("--normal_k", type=int, default=30, help="Number of neighbors for normal estimation")
    parser.add_argument("--clean_points", action="store_true", help="Remove outliers from point cloud before processing")
    parser.add_argument("--outlier_std", type=float, default=2.0, help="Standard deviation multiplier for outlier removal")
    parser.add_argument("--outlier_nb", type=int, default=20, help="Number of neighbors to consider for outlier removal")
    parser.add_argument("--poisson", action="store_true", help="Perform Poisson reconstruction")
    parser.add_argument("--depth", type=int, default=8, help="Maximum depth of the octree")
    parser.add_argument("--scale", type=float, default=1.1, help="Scale factor")
    parser.add_argument("--linear", action="store_true", help="Use linear fit")
    parser.add_argument("--threads", type=int, default=-1, help="Number of threads")
    parser.add_argument("--density_threshold", type=float, default=0.0, help="Density threshold for removing faces")
    parser.add_argument("--color", action="store_true", help="Transfer colors from point cloud to mesh")
    parser.add_argument("--color_k", type=int, default=3, help="Number of nearest neighbors for color transfer")
    
    args = parser.parse_args()
    
  
    points, colors = load_point_cloud(args.input)
    
    if points is not None:
        print(f"Loaded {len(points)} points")
        

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Clean point cloud if requested
        if args.clean_points:
            pcd = remove_outliers(pcd, nb_neighbors=args.outlier_nb, std_ratio=args.outlier_std)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            print(f"After outlier removal: {len(points)} points")
        
        # Estimate normals if requested
        normals = None
        if args.estimate_normals:
            normals = estimate_normals(points, k=args.normal_k)
            print(f"Estimated normals for {len(normals)} points")
        
        # Save point cloud with normals
        output_path = args.output
        if args.poisson:
            # If we're doing Poisson reconstruction, save the point cloud with normals to a temporary file
            normals_path = args.output.replace('.ply', '_with_normals.ply')
            save_point_cloud_with_normals(normals_path, points, normals, colors)
            print(f"Saved point cloud with normals to {normals_path}")
            
            # Perform reconstruction
            mesh = poisson_surface_reconstruction(
                points, normals, 
                depth=args.depth, 
                scale=args.scale, 
                linear_fit=args.linear, 
                n_threads=args.threads
            )
            
            # Remove low-density vertices if threshold is provided
   
            
            # Transfer colors if requested
            if args.color:
                # Check if point cloud has colors
                try:
                    if pcd.has_colors(): 
                        mesh = transfer_colors(mesh, pcd, k=args.color_k)
                        print("Transferred colors from point cloud to mesh")
                except Exception as e:
                    print(f"Failed to transfer colors: {e}")
            
            # Save the result
            if save_mesh(mesh, output_path):
                print(f"Mesh saved to {output_path}")
            else:
                print(f"Failed to save mesh to {output_path}")
        else:
            # Just save the point cloud with normals
            if save_point_cloud_with_normals(output_path, points, normals, colors):
                print(f"Point cloud with normals saved to {output_path}")
            else:
                print(f"Failed to save point cloud to {output_path}")
    else:
        print("Failed to load point cloud")
