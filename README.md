# SELFI3-D: Facial Mesh Reconstruction and Tattoo Visualization

SELFI3-D is a pipeline for facial mesh reconstruction and tattoo visualization from sparse input views. This project helps users visualize what they would look like with face tattoos before making permanent decisions.

## Overview

Using a combination of gaussian splatting, poisson reconstruction, and texture mapping, SELFI3-D creates realistic 3D facial reconstructions from a small set of input images. The system then allows users to apply custom tattoo designs to their facial model with realistic texture mapping.

## Features

- 3D facial reconstruction from sparse image views
- High-quality texture mapping of facial features
- Tattoo visualization on reconstructed face models
- Realistic rendering with proper lighting and shading

## Installation

### Prerequisites

- Python 3.8+
- Required packages (install via pip):
  ```
  pip install -r requirements.txt
  ```

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/jameson-crate/SELFI3-D.git
   cd SELFI3-D
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your input images in the `input/` directory
2. Run the reconstruction pipeline:
   ```
   python reconstruct.py --input input/images/ --output output/mesh/
   ```
3. Apply a tattoo design:
   ```
   python apply_tattoo.py --mesh output/mesh/face.obj --tattoo designs/tattoo.png --output output/final/
   ```

## Team Members

- Jameson Crate
- Michelle Chen
- Valerie Li
- Akshaan Ahuja

## Project Website

For more information, visit our [project website](https://jameson-crate.github.io/SELFI3-D/).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
