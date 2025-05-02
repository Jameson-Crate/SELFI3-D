#!/usr/bin/env python3
import argparse
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.texture import TextureVisuals

def ensure_uv(mesh):
    has_uv = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
    if not has_uv:
        verts = mesh.vertices[:, :2]
        mn, mx = verts.min(axis=0), verts.max(axis=0)
        span = mx - mn
        span[span == 0] = 1.0
        uv = (verts - mn) / span
        mesh.visual = TextureVisuals(uv=uv)
    return mesh

def bake_overlay(mesh_path, overlay_png, output_obj, atlas_size=2048):
    # 1) Load mesh
    mesh = trimesh.load(mesh_path, process=False)

    # 2) Ensure UVs
    mesh = ensure_uv(mesh)

    # 3) Create blank white atlas
    atlas = Image.new("RGBA", (atlas_size, atlas_size), (255, 255, 255, 255))

    # 4) Load and composite overlay
    decal = Image.open(overlay_png).convert("RGBA")
    decal = decal.resize((atlas_size, atlas_size), resample=Image.Resampling.LANCZOS)
    atlas.alpha_composite(decal)

    # 5) Assign atlas to mesh material
    mesh.visual.material.image = np.asarray(atlas)
    atlas_name = output_obj.replace('.obj', '.png').replace('.ply', '.png')
    mesh.visual.material.image_name = atlas_name
    # 6) Export OBJ/MTL + atlas PNG
    mesh.export(output_obj)
    atlas.save(atlas_name)
    print(f"Exported textured mesh → {output_obj}")
    print(f"Exported atlas           → {atlas_name}")

if __name__=="__main__":
    p = argparse.ArgumentParser("Bake an RGBA overlay into a mesh’s texture atlas")
    p.add_argument("mesh",    help="Input mesh (OBJ/PLY/GLB)")
    p.add_argument("overlay", help="RGBA PNG overlay")
    p.add_argument("out_obj", help="Output .obj/.ply (with new atlas)")
    p.add_argument("--size",  type=int, default=2048, help="Atlas size in px")
    args = p.parse_args()

    bake_overlay(args.mesh, args.overlay, args.out_obj, atlas_size=args.size)
