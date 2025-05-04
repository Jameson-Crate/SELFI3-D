import open3d as o3d
import numpy as np

def poisson_surface_reconstruction(points, normals=None, depth=8, scale=1.1, linear_fit=False, n_threads=-1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=linear_fit, n_threads=n_threads) 
    return mesh


def save_mesh(mesh, output_path):
    return o3d.io.write_triangle_mesh(output_path, mesh)

def load_point_cloud(input_path):
   
    try:
        pcd = o3d.io.read_point_cloud(input_path)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        return points, normals
    except Exception as e:
        print("Error loading point cloud")
        return None, None


def transfer_colors(mesh, pcd, k=3):
    if not pcd.has_colors():
        print("Point cloud has no colors to transfer")
        return mesh
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_vertices = np.asarray(mesh.vertices)
    
    vertex_colors = np.zeros((len(mesh_vertices), 3))
    for i, vertex in enumerate(mesh_vertices):
        k_actual, idx, _ = pcd_tree.search_knn_vector_3d(vertex, k)
        nearest_colors = np.asarray(pcd.colors)[idx[:k_actual]]
        vertex_colors[i] = np.mean(nearest_colors, axis=0)
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def estimate_normals(points, k=30, orient_with_viewpoint=True, viewpoint=None):    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    
    if orient_with_viewpoint:
        if viewpoint is None:
            viewpoint = [0, 0, 0]  
        pcd.orient_normals_towards_camera_location(viewpoint)
    
    try:
        pcd.orient_normals_consistent_tangent_plane(k=k)
    except Exception as e:
        print(f"Warning: Could not orient normals consistently: {e}")
    
    return np.asarray(pcd.normals)


def save_point_cloud_with_normals(file_path, points, normals, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return o3d.io.write_point_cloud(file_path, pcd)


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description="Point Cloud Processing with Normal Estimation")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--estimate_normals", action="store_true")
    parser.add_argument("--normal_k", type=int, default=30)
    parser.add_argument("--clean_points", action="store_true")
    parser.add_argument("--poisson")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.1)
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--density_threshold", type=float, default=0.0)
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--color_k", type=int, default=3)
    
    args = parser.parse_args()
    points, colors = load_point_cloud(args.input)
    
    if points is not None:
        print(f"Loaded {len(points)} points")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        
        normals = None
        if args.estimate_normals:
            normals = estimate_normals(points, k=args.normal_k)
        
        output_path = args.output
        if args.poisson:
            normals_path = args.output.replace('.ply', '_with_normals.ply')
            save_point_cloud_with_normals(normals_path, points, normals, colors)
            
            mesh = poisson_surface_reconstruction(
                points, normals, 
                depth=args.depth, 
                scale=args.scale, 
                linear_fit=args.linear, 
                n_threads=args.threads
            )
            
            if args.color:
                try:
                    if pcd.has_colors(): 
                        mesh = transfer_colors(mesh, pcd, k=args.color_k)
                        print("Transferred colors from point cloud to mesh")
                except Exception as e:
                    print(f"Failed to transfer colors: {e}")
            
            if save_mesh(mesh, output_path):
                print("Mesh saved")
            else:
                print("Failed to save mesh")
        else:
            if save_point_cloud_with_normals(output_path, points, normals, colors):
                print("Point cloud with normals saved")
            else:
                print(f"Failed to save point cloud")
    else:
        print("Failed to load point cloud")
