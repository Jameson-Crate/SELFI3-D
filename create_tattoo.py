import numpy as np
import open3d as o3d
import skimage.io as skio

pcd = o3d.io.read_point_cloud('data/outputs/ak_mast3r_centered.ply')

tattoo = skio.imread('data/ak_old/seal.png')
tattoo = np.rot90(tattoo, 3)
skio.imshow(tattoo)
tattoo = tattoo / 255

pcd_arr = np.hstack([pcd.points, pcd.colors])

x_scale = 8
z_scale = 8
max_x, max_y, max_z = np.max(pcd_arr[:, :3], axis=0) / x_scale
min_x, min_y, min_z  = np.min(pcd_arr[:, :3], axis=0) / z_scale
range_x = max_x - min_x
range_y = max_y - min_y
range_z = max_z - min_z

th, tw = tattoo.shape[:2]
x_shift = 7
z_shift = 110
for i in range(len(pcd_arr)):
    pcd_x = np.round(tw * ((pcd_arr[i][0] - min_x) / range_x) - (x_shift * x_scale)).astype(int)
    pcd_z = np.round(th * ((pcd_arr[i][2] - min_z) / range_z) - (z_shift * z_scale)).astype(int)
    if (0 <= pcd_x < tw) and (0 <= pcd_z < tw) and any(tattoo[pcd_x, pcd_z] < 1):
        pcd_arr[i][3:] = tattoo[pcd_x, pcd_z]

points = o3d.utility.Vector3dVector(pcd_arr[:, :3])
colors = o3d.utility.Vector3dVector(pcd_arr[:, 3:])
tattoo_pcd = o3d.geometry.PointCloud(points)
tattoo_pcd.colors = colors
o3d.io.write_point_cloud('ak_tattoo.ply', tattoo_pcd)