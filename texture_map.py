import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import KDTree
from tqdm import tqdm
import copy

def inverse_texture_mapping(pcd, mesh, max_distance=None, num_neighbors=3):
    mesh_vertices = np.asarray(mesh.vertices)
    cloud_points = np.asarray(pcd.points)
    cloud_colors = np.asarray(pcd.colors)
    
    pcd_tree = KDTree(cloud_points)
    vertex_colors = np.zeros((len(mesh_vertices), 3))
    
    if max_distance is None:
        mesh_bbox = mesh.get_axis_aligned_bounding_box()
        mesh_extent = np.linalg.norm(mesh_bbox.get_max_bound() - mesh_bbox.get_min_bound())
        max_distance = mesh_extent * 0.05
    
    
    for i, vertex in tqdm(enumerate(mesh_vertices), total=len(mesh_vertices), desc="Processing vertices"):
        distances, indices = pcd_tree.query(vertex, k=num_neighbors)
        
        weights = np.zeros(len(indices))
        valid_indices = []
        
        for j, (idx, dist) in enumerate(zip(indices, distances)):
            if dist <= max_distance:
                weights[j] = 1.0 / (dist + 1e-10)
                valid_indices.append(j)
        
        if valid_indices:
            weights = weights[valid_indices] / np.sum(weights[valid_indices])
            for w, idx in zip(weights, indices[valid_indices]):
                vertex_colors[i] += cloud_colors[idx] * w
    
    colored_mesh = o3d.geometry.TriangleMesh()
    colored_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    colored_mesh.triangles = mesh.triangles
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    if mesh.has_vertex_normals():
        colored_mesh.vertex_normals = mesh.vertex_normals
    if mesh.has_triangle_normals():
        colored_mesh.triangle_normals = mesh.triangle_normals
    
    return colored_mesh


def transform_coordinates_for_format(mesh, target_format):
    target_format = target_format.lower()
    if target_format not in ['.obj', '.glb', '.gltf']:
        return mesh
    
    transformed_mesh = copy.deepcopy(mesh)
    vertices = np.asarray(transformed_mesh.vertices).copy()
    
    y_temp = vertices[:, 1].copy()
    vertices[:, 1] = vertices[:, 2]
    vertices[:, 2] = -y_temp
    
    transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    if transformed_mesh.has_vertex_normals():
        normals = np.asarray(transformed_mesh.vertex_normals).copy()
        y_temp = normals[:, 1].copy()
        normals[:, 1] = normals[:, 2]
        normals[:, 2] = -y_temp
        transformed_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    
    if transformed_mesh.has_triangle_normals():
        tri_normals = np.asarray(transformed_mesh.triangle_normals).copy()
        y_temp = tri_normals[:, 1].copy()
        tri_normals[:, 1] = tri_normals[:, 2]
        tri_normals[:, 2] = -y_temp
        transformed_mesh.triangle_normals = o3d.utility.Vector3dVector(tri_normals)
    
    return transformed_mesh


def main():
    parser = argparse.ArgumentParser(description='Map texture from a point cloud to a mesh')
    parser.add_argument('pointcloud')
    parser.add_argument('mesh')
    parser.add_argument('--output', '-o')
    parser.add_argument('--max_distance', type=float, default=None)
    parser.add_argument('--neighbors', type=int, default=3)
    parser.add_argument('--upsample', type=float, default=0)
    parser.add_argument('--fix-rotation', action='store_true')
    
    args = parser.parse_args()
    
    try:
        pcd = o3d.io.read_point_cloud(str(args.pointcloud))
    
        if len(pcd.points) == 0:
            raise ValueError(f"Failed to load point cloud")
        
        if not pcd.has_colors():
            raise ValueError(f"Point cloud has no color information")
        
        mesh = o3d.io.read_triangle_mesh(str(args.mesh))
        
        if len(mesh.vertices) == 0:
            raise ValueError(f"Failed to load mesh")
        
        if args.upsample > 0:
            target_vertices = int(len(pcd.points) * args.upsample)
            current_vertices = len(mesh.vertices)
            
            if target_vertices > current_vertices:                
                iterations = 0
                estimated_vertices = current_vertices
                while estimated_vertices < target_vertices:
                    estimated_vertices *= 4
                    iterations += 1
                
                if iterations > 0 and estimated_vertices / 4 >= target_vertices:
                    iterations -= 1
                    estimated_vertices /= 4
                
                if iterations > 0:
                    mesh = mesh.subdivide_midpoint(number_of_iterations=iterations)
        
        colored_mesh = inverse_texture_mapping(pcd, mesh, args.max_distance, args.neighbors)
        
        if args.output:
            output_path = args.output
        else:
            input_mesh_path = Path(args.mesh)
            output_path = str(input_mesh_path.with_name(f"{input_mesh_path.stem}_textured{input_mesh_path.suffix}"))
        
        file_ext = Path(output_path).suffix.lower()
        mesh_to_save = colored_mesh
        
        if file_ext.lower() in ['.obj', '.glb', '.gltf'] and args.fix_rotation:
            mesh_to_save = transform_coordinates_for_format(colored_mesh, file_ext)
        
        print(f"Saving textured mesh to {output_path}")
        success = o3d.io.write_triangle_mesh(output_path, mesh_to_save)
        
        if not success:
            if file_ext != '.ply':
                fallback_path = str(Path(output_path).with_suffix('.ply'))
                o3d.io.write_triangle_mesh(fallback_path, colored_mesh)
            return 1
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
