"""
face_tattoo_app.py

Gradio front‑end prototype for interactive 3D face‑tattoo preview.

Usage:
  python face_tattoo_app.py

Requirements:
  pip install gradio pillow numpy
  # For full pipeline (optional):
  pip install torch torchvision pytorch3d trimesh open3d

This script currently stubs‑out the heavy lifting (3 D face reconstruction
and texture baking). Replace the marked TODO sections with calls to your
MAST3R pipeline and your chosen texturing/rendering stack.
"""

import gradio as gr
from PIL import Image
from datetime import datetime
import os
import trimesh
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Configuration
DEFAULT_TATTOO_PATH = "tattoo.png"  # Changed from jpg to png
DEFAULT_TEXTURE_SIZE = (1024, 1024)
DEFAULT_BASE_SCALE = 0.2  # Base size is 20% of texture size

# Default UI values
DEFAULT_SCALE = 0.1  # 10% of the base size
DEFAULT_U = 0.5     # Center horizontally
DEFAULT_V = 0.7     # Move up from center (0.5 is center, higher values move up)

# -----------------------------------------------------------------------------
# ── Pipeline placeholders ─────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def generate_mesh(image_files):
    """Generate an .obj mesh from several face images (stub).

    Parameters
    ----------
    image_files : list[str | Path]
        Paths to user‑uploaded face images (front, profile, etc.).

    Returns
    -------
    tuple[str, str]
        (1) path to the mesh for preview, (2) the same path passed through a
        gr.State so later steps can access it.
    """
    # TODO: integrate your MAST3R + post‑processing here
<<<<<<< Updated upstream
    mesh_path = "ak_mast3r_no_plane.obj"
    with open(mesh_path, "w", encoding="utf‑8") as f:
        f.write(f"# dummy OBJ generated {datetime.utcnow().isoformat()}\n")
=======
    mesh_path = "test3.glb"
    # with open(mesh_path, "w", encoding="utf‑8") as f:
    #     f.write(f"# dummy OBJ generated {datetime.utcnow().isoformat()}\n")
>>>>>>> Stashed changes
    return mesh_path, mesh_path  # (mesh for <Model3D/>, mesh for State)


def generate_uv_coordinates(mesh):
    """Generate UV coordinates using spherical projection.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh
        
    Returns
    -------
    numpy.ndarray
        UV coordinates for each vertex
    """
    # Get vertices
    vertices = mesh.vertices - mesh.centroid
    
    # Normalize to unit sphere
    radii = np.linalg.norm(vertices, axis=1)
    vertices = vertices / np.maximum(radii[:, np.newaxis], 1e-10)
    
    # Convert to spherical coordinates
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])
    theta = np.arccos(np.clip(vertices[:, 2], -1.0, 1.0))
    
    # Convert to UV coordinates
    u = (phi + np.pi) / (2 * np.pi)
    v = theta / np.pi
    
    return np.column_stack((u, v))


def map_tattoo(mesh_path, tattoo_img, u, v, scale):
    """Project a 2‑D tattoo onto the 3‑D face mesh.

    Parameters
    ----------
    mesh_path : str
        Path to the GLB mesh file.
    tattoo_img : PIL.Image | ndarray | str | None
        The uploaded tattoo image path.
    u, v : float
        Normalised UV coordinates (0‑1) chosen by the user.
    scale : float > 0
        Relative scale factor.

    Returns
    -------
    str
        Path to the textured mesh in OBJ format.
    """
    try:
        # Validate mesh path
        if mesh_path is None:
            raise ValueError("No mesh file selected. Please generate a mesh first.")
            
        # Handle tattoo image path
        tattoo_path = tattoo_img if tattoo_img else DEFAULT_TATTOO_PATH
        if not os.path.exists(tattoo_path):
            raise ValueError(f"Tattoo image not found at: {tattoo_path}")
            
        # Validate input file exists
        mesh_path = os.path.abspath(mesh_path)
        if not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file not found at: {mesh_path}")
            
        print(f"Loading mesh from: {mesh_path}")
            
        # Load the GLB file
        try:
            loaded = trimesh.load(mesh_path, file_type='glb', process=False, maintain_order=True)
            print("Successfully loaded GLB file")
        except Exception as e:
            print(f"Failed to load GLB file: {str(e)}")
            raise ValueError(f"Failed to load mesh file: {str(e)}")
        
        # Handle both single meshes and scenes
        if isinstance(loaded, trimesh.Scene):
            print("Loaded file is a scene")
            if len(loaded.geometry) == 0:
                raise ValueError("No meshes found in the GLB file")
            mesh = next(iter(loaded.geometry.values()))
            print(f"Found {len(loaded.geometry)} meshes in scene")
        else:
            print("Loaded file is a single mesh")
            mesh = loaded
            
        if mesh is None:
            raise ValueError("Failed to load mesh: no valid mesh found")
            
        print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Load the original texture if it exists
        if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
            base_texture = mesh.visual.material.image
            if base_texture.mode != 'RGBA':
                base_texture = base_texture.convert('RGBA')
            print("Using existing texture from mesh")
        else:
            base_texture = Image.new('RGBA', DEFAULT_TEXTURE_SIZE, (255, 255, 255, 255))
            print("Created new white texture")
        
        # Load and process the tattoo image
        print(f"Loading tattoo from: {tattoo_path}")
        try:
            tattoo = Image.open(tattoo_path)
            print(f"Tattoo image size: {tattoo.size}")
            print(f"Tattoo image mode: {tattoo.mode}")
        except Exception as e:
            raise ValueError(f"Failed to load tattoo image: {str(e)}")
        
        # Convert tattoo to RGBA if it isn't already
        if tattoo.mode != 'RGBA':
            tattoo = tattoo.convert('RGBA')
            print(f"Converted tattoo to RGBA mode")
            
        # Print base texture information
        print(f"Base texture size: {base_texture.size}")
        print(f"Base texture mode: {base_texture.mode}")
        
        # Calculate tattoo size based on scale
        base_size = min(base_texture.size) * DEFAULT_BASE_SCALE
        tattoo_size = int(base_size * scale)
        tattoo = tattoo.resize((tattoo_size, tattoo_size), Image.Resampling.LANCZOS)
        
        print(f"Tattoo will be placed at coordinates ({u}, {v}) with size {tattoo_size}")
        
        # Calculate position for the tattoo
        x = int((u - 0.5) * base_texture.size[0] + base_texture.size[0] // 2 - tattoo_size // 2)
        y = int((v - 0.5) * base_texture.size[1] + base_texture.size[1] // 2 - tattoo_size // 2)
        
        # Create a new layer for the tattoo
        tattoo_layer = Image.new('RGBA', base_texture.size, (0, 0, 0, 0))
        tattoo_layer.paste(tattoo, (x, y), tattoo)
        
        # Composite the tattoo onto the base texture
        base_texture = Image.alpha_composite(base_texture, tattoo_layer)
        
        # Save the final texture image
        texture_path = os.path.abspath("texture.png")
        base_texture.save(texture_path, "PNG")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(mesh_path)
        if not output_dir:
            output_dir = "."
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        output_path = os.path.abspath(os.path.join(output_dir, f"{mesh_name}_tattooed.obj"))
        print(f"Saving textured mesh to: {output_path}")
        
        # Write OBJ file manually
        with open(output_path, 'w') as f:
            f.write("mtllib material.mtl\n")
            f.write("usemtl material0\n")
            
            # Write vertices
            for v in mesh.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                
            # Write texture coordinates
            if hasattr(mesh.visual, 'uv'):
                uvs = mesh.visual.uv
            else:
                # Generate UV coordinates using spherical projection
                uvs = generate_uv_coordinates(mesh)
            
            for uv in uvs:
                f.write(f"vt {uv[0]} {1-uv[1]}\n")
                
            # Write faces (1-based indexing in OBJ)
            for face in mesh.faces:
                f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
                
        # Write MTL file with absolute paths
        mtl_path = os.path.abspath(os.path.join(output_dir, "material.mtl"))
        with open(mtl_path, 'w') as f:
            f.write("newmtl material0\n")
            f.write("Ka 0.200000 0.200000 0.200000\n")  # Reduced ambient light
            f.write("Kd 0.800000 0.800000 0.800000\n")  # Slightly reduced diffuse
            f.write("Ks 0.000000 0.000000 0.000000\n")  # No specular
            f.write(f"map_Kd {texture_path}\n")  # Using absolute path
            f.write("d 1.0\n")  # Full opacity
            
        print("Successfully saved textured mesh at: ", output_path)
        return output_path
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        error_msg = str(e)
        # Return a more user-friendly error message to the Gradio interface
        raise gr.Error(f"Failed to process mesh or tattoo: {error_msg}")


# -----------------------------------------------------------------------------
# ── Gradio UI ─────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            "# 🎨 3D Face‑Tattoo Preview\n"
            "Upload a few clear photos, generate your 3‑D face, then position and"
            " size a tattoo texture in real‑time."
        )

        # Status message for feedback
        status_msg = gr.Markdown("")

        # ── Step 1: face images & mesh reconstruction ─────────────────────────
        with gr.Row():
            face_imgs = gr.File(
                label="Face photos (front + profile)", 
                file_count="multiple", 
                type="filepath"
            )
            gen_btn = gr.Button("Generate face mesh", variant="primary")
        mesh_view = gr.Model3D(label="Mesh preview", clear_color=[0.9, 0.9, 0.9, 1.0])
        mesh_state = gr.State()

        def generate_with_status(image_files):
            status_msg.value = "Generating 3D mesh from photos..."
            try:
                mesh_path, state = generate_mesh(image_files)
                status_msg.value = "Mesh generated successfully!"
                return [mesh_path, state]
            except Exception as e:
                status_msg.value = f"Error generating mesh: {str(e)}"
                raise gr.Error(str(e))

        gen_btn.click(
            fn=generate_with_status, 
            inputs=face_imgs, 
            outputs=[mesh_view, mesh_state]
        )

        # ── Step 2: tattoo image & placement ──────────────────────────────────
        gr.Markdown("## Tattoo placement")
        with gr.Row():
            tattoo_img = gr.Image(
                label="Tattoo image", 
                type="filepath",
                image_mode="RGBA"
            )
            with gr.Column():
                u_slider = gr.Slider(0, 1, value=DEFAULT_U, step=0.01, label="Horizontal (U)")
                v_slider = gr.Slider(0, 1, value=DEFAULT_V, step=0.01, label="Vertical (V)")
                s_slider = gr.Slider(0.1, 2.0, value=DEFAULT_SCALE, step=0.05, label="Scale")
                apply_btn = gr.Button("Apply tattoo", variant="primary")

        textured_mesh = gr.Model3D(
            label="Result with tattoo",
            clear_color=[0.9, 0.9, 0.9, 1.0]
        )

        def apply_with_status(mesh_state, tattoo_img, u, v, s):
            if not mesh_state:
                raise gr.Error("Please generate a face mesh first")
            status_msg.value = "Applying tattoo to mesh..."
            try:
                result = map_tattoo(mesh_state, tattoo_img, u, v, s)
                status_msg.value = "Tattoo applied successfully!"
                return result
            except Exception as e:
                status_msg.value = f"Error applying tattoo: {str(e)}"
                raise gr.Error(str(e))

        apply_btn.click(
            fn=apply_with_status,
            inputs=[mesh_state, tattoo_img, u_slider, v_slider, s_slider],
            outputs=textured_mesh,
        )

        gr.Markdown(
            "---\n"
            "### Tips:\n"
            "1. For best results, use PNG images with transparency for tattoos\n"
            "2. The preview may take a few seconds to update\n"
            "3. If the preview doesn't update, try adjusting the sliders slightly"
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.queue()  # Enable queuing for better handling of concurrent users
    demo.launch(share=False)  # Disable sharing for security