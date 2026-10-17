# Stereo3D

Stereo3D is a Python-based pipeline for reconstructing 3D surface models from autostereograms (Single Image Random Dot Stereograms and Single Image Stereograms).

## Overview

Autostereograms encode depth by modulating the horizontal repetition distance of a pattern across horizontal scanlines:

$$S(x, y) = S_0 - d(x, y)$$

where:
- $S_0$ is the base repetition period corresponding to the background plane.
- $d(x, y)$ is the horizontal disparity proportional to surface depth.

The pipeline automates the end-to-end recovery of 3D geometry:
1. **Base Period Estimation**: Identifying the fundamental repeat period $S_0$ using row-wise normalized autocorrelation and sum of absolute differences (SAD).
2. **Disparity and Depth Recovery**: Symmetric scanline correspondence matching to extract a dense depth map.
3. **3D Reconstruction**: Projecting the depth map into a 3D polygonal mesh or point cloud.

# THIS IS UNDER DEVELOPMENT!!!

## Usage

### 1. Depth Map Extraction
Place your autostereogram image in the `input/` folder, then pass the filename as an argument:

```powershell
python main.py image_name.png
```

You can also provide a direct path:
```powershell
python main.py path/to/image.png
```

Generated outputs in `output/`:
- `<name>_depth.png`: Standard 8-bit grayscale depth map.
- `<name>_depth_16bit.png`: 16-bit uint16 grayscale depth map (for 3D displacement).
- `<name>_depth_color.png`: High-contrast colormapped depth visualization.
- `<name>_depth.npy`: Raw floating-point depth array.
- `<name>_analysis.png`: Diagnostic plot showing the input image, difference map, and depth map.
- `<name>_mesh.obj`: 3D polygonal mesh in Wavefront OBJ format with vertex colors, normals, and material definitions.
- `<name>_mesh.ply`: 3D mesh in Stanford binary PLY format with embedded RGB vertex colors.

When run with `--postprocess` (or `postprocess`), additional clean outputs are generated:
- `<name>_depth_postprocessed.png`: Denoised 8-bit depth map with unrecoverable side bars cropped.
- `<name>_depth_postprocessed_16bit.png`: Denoised 16-bit uint16 depth map (optimal for clean 3D displacement).
- `<name>_depth_postprocessed_color.png`: Denoised Turbo colorized depth map.
- `<name>_depth_postprocessed.npy`: Denoised floating-point array.
- `<name>_postprocess_mask.png`: Binary mask showing all flagged and inpainted noise spikes/splotches.
- `<name>_postprocess_analysis.png`: 3-panel diagnostic plot comparing Raw Depth vs Noise Mask vs Clean Depth.
- `<name>_mesh_postprocessed.obj` / `.ply`: 3D meshes reconstructed from the clean post-processed depth map.

Optional flags:
- `--postprocess`: Activate the post-processing pipeline (drops side bars, multi-scale spike inpainting, splotch filtering, and bilateral smoothing).
- `--no-drop-margins`: Do not crop the unrecoverable side margins during post-processing.
- `--min-island-area <int>`: Minimum pixel area for isolated elevated islands (splotches below this are inpainted; default: `300`).
- `--inpaint-method {navier_stokes,telea}`: Inpainting algorithm (default: `navier_stokes`).
- `--period <int>`: Manually specify base period $S_0$ in pixels.
- `--max-disparity <int>`: Maximum disparity in pixels.
- `--view-mode {parallel,cross}`: Set viewing mode (default: `parallel`).
- `--mesh-step <int>`: Subsampling step for 3D mesh grid (default: `2`, `1` for 1:1 pixel resolution).
- `--z-scale <float>`: Height displacement scale factor.
- `--mesh-color {rgb,depth,uniform}`: Vertex coloring strategy (default: `rgb`).
- `--view`: Open interactive 3D viewer immediately after reconstruction.
- `--view-mode-3d {untextured,material,depth}`: Initial 3D shading mode (default: `untextured`).
- `--viewer-engine {pbr,classic}`: 3D engine ('pbr' for Global Illumination or 'classic' for OpenGL).
- `--no-mesh`: Disable 3D mesh generation.

### 2. Interactive 3D Object Viewer
View the generated 3D object directly without external software:

```powershell
python view_3d.py [model_name_or_path]
```

By default, the viewer starts in **Untextured Shaded Clay mode** with hardware-accelerated **Global Illumination** (Image-Based Lighting) and directional sunlight with soft shadows, making surface contours and relief immediately clear without visual camouflage from repeating pattern textures.

Options:
- `--mode {untextured,material,depth}`: Starting display mode (default: `untextured`).
- `--engine {pbr,classic}`: Render engine ('pbr' for Filament Global Illumination, 'classic' for OpenGL).

Interactive Controls:
- **Swap Material / Untextured**: Click `[Swap: Material / Untextured]` in the PBR sidebar or press `[M]` / `[T]` in classic mode.
- **Toggle Depth Heatmap**: Click `[Toggle Depth Heatmap]` in the sidebar or press `[C]`.
- **Rotate / Orbit**: Left click and drag.
- **Pan / Translate**: Right click and drag (or Shift + Left drag).
- **Zoom**: Scroll wheel (or Ctrl + Left drag).
- **Settings Sidebar**: Use the right-hand panel to adjust Sunlight intensity, sun angle, and Global Illumination (IBL) environment reflections.
- **`[W]`**: Toggle wireframe overlay.
- **`[R]`**: Reset camera view.
- **`[Q]` / `[ESC]`**: Exit viewer.

### 3. Synthetic Generation & Testing
Generate synthetic stereograms and validate period estimation, depth recovery, and 3D mesh generation against ground-truth geometry:

```powershell
python generate_and_test.py
```
Options:
- `--shape {circle,box}`: Shape of hidden 3D object (default: `circle`).
- `--period <int>`: Ground-truth base period (default: `120`).
- `--max-disparity <int>`: Maximum disparity (default: `24`).
