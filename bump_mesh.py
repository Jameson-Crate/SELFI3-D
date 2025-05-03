import numpy as np
import open3d as o3d
import noise
from pathlib import Path

input_mesh = "ak_mesh.obj"
output_mesh = "ak_bumped.obj"
freq = 5.0
octaves = 4
persistence = 0.5
lacunarity = 2.0
scale = 0.01
recompute_normals = True

mesh = o3d.io.read_triangle_mesh(input_mesh)
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()
verts = np.asarray(mesh.vertices)
norms = np.asarray(mesh.vertex_normals)

displaced = verts.copy()
for i, v in enumerate(verts):
    h = noise.pnoise3(
        v[0] * freq,
        v[1] * freq,
        v[2] * freq,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        repeatx=1024,
        repeaty=1024,
        repeatz=1024,
        base=0
    )
    height = (h + 1.0) * 0.5
    displaced[i] += norms[i] * (height * scale)

out = o3d.geometry.TriangleMesh()
out.vertices = o3d.utility.Vector3dVector(displaced)
out.triangles = mesh.triangles
if mesh.has_vertex_colors():
    out.vertex_colors = mesh.vertex_colors
if recompute_normals:
    out.compute_vertex_normals()
o3d.io.write_triangle_mesh(output_mesh, out)
