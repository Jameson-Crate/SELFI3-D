"""
face_tattoo_app.py

Gradio front‑end prototype for interactive 3D face‑tattoo preview.

Usage:
  python face_tattoo_app.py

Requirements:
  pip install gradio pillow numpy
  # For full pipeline (optional):
  pip install torch torchvision pytorch3d trimesh

This script currently stubs‑out the heavy lifting (3 D face reconstruction
and texture baking). Replace the marked TODO sections with calls to your
MAST3R pipeline and your chosen texturing/rendering stack.
"""

import gradio as gr
from PIL import Image
from datetime import datetime
import os

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
    mesh_path = "ak_mast3r_no_plane.obj"
    with open(mesh_path, "w", encoding="utf‑8") as f:
        f.write(f"# dummy OBJ generated {datetime.utcnow().isoformat()}\n")
    return mesh_path, mesh_path  # (mesh for <Model3D/>, mesh for State)


def map_tattoo(mesh_path, tattoo_img, u, v, scale):
    """Project a 2‑D tattoo onto the 3‑D face mesh (stub).

    Parameters
    ----------
    mesh_path : str
        Path (from gr.State) to the .obj mesh.
    tattoo_img : PIL.Image | ndarray | str
        The uploaded tattoo image.
    u, v : float
        Normalised UV coordinates (0‑1) chosen by the user.
    scale : float > 0
        Relative scale factor.

    Returns
    -------
    str
        Path to the textured mesh (or the original mesh for now).
    """
    # TODO:
    #   1. read UV map from mesh
    #   2. composite tattoo_img into texture atlas at (u,v) with given scale
    #   3. bake / export a new mesh (OBJ/GLB) with updated material
    # For the prototype we simply echo the existing mesh.
    return mesh_path


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

        # ── Step 1: face images & mesh reconstruction ─────────────────────────
        with gr.Row():
            face_imgs = gr.File(
                label="Face photos (front + profile)", file_count="multiple", type="filepath"
            )
            gen_btn = gr.Button("Generate face mesh")
        mesh_view = gr.Model3D(label="Mesh preview")
        mesh_state = gr.State()

        gen_btn.click(
            fn=generate_mesh, inputs=face_imgs, outputs=[mesh_view, mesh_state]
        )

        # ── Step 2: tattoo image & placement ──────────────────────────────────
        gr.Markdown("## Tattoo placement")
        with gr.Row():
            tattoo_img = gr.Image(label="Tattoo image", type="filepath")
            with gr.Column():
                u_slider = gr.Slider(0, 1, value=0.5, step=0.01, label="Horizontal (U)")
                v_slider = gr.Slider(0, 1, value=0.5, step=0.01, label="Vertical (V)")
                s_slider = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Scale")
                apply_btn = gr.Button("Apply tattoo")

        textured_mesh = gr.Model3D(label="Result with tattoo")

        apply_btn.click(
            fn=map_tattoo,
            inputs=[mesh_state, tattoo_img, u_slider, v_slider, s_slider],
            outputs=textured_mesh,
        )

        gr.Markdown(
            "---\n👆 Replace the two stub functions with your production pipeline and"
            " you have an end‑to‑end MVP."
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()