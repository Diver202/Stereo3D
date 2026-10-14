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

Optional flags:
- `--period <int>`: Manually specify base period $S_0$ in pixels.
- `--max-disparity <int>`: Maximum disparity in pixels.
- `--view-mode {parallel,cross}`: Set viewing mode (default: `parallel`).

### 2. Synthetic Generation & Testing
Generate synthetic stereograms and validate period estimation and depth recovery against ground-truth geometry:

```powershell
python generate_and_test.py
```
Options:
- `--shape {circle,box}`: Shape of hidden 3D object (default: `circle`).
- `--period <int>`: Ground-truth base period (default: `120`).
- `--max-disparity <int>`: Maximum disparity (default: `24`).
