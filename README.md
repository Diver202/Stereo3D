# Stereo3D

Stereo3D is a Python-based pipeline for reconstructing 3D surface models from autostereograms (Single Image Random Dot Stereograms and Single Image Stereograms).

## Overview

Autostereograms encode depth by modulating the horizontal repetition distance of a pattern across horizontal scanlines:

$$S(x, y) = S_0 - d(x, y)$$

where:
- $S_0$ is the base repetition period corresponding to the background plane.
- $d(x, y)$ is the horizontal disparity proportional to surface depth.

The objective of this project is to automate the end-to-end recovery of 3D geometry from autostereogram images:
1. **Base Period Estimation**: Identifying the fundamental repeat period $S_0$ using row-wise normalized autocorrelation and sum of absolute differences (SAD).
2. **Disparity and Depth Recovery**: Solving scanline correspondence matching to extract a dense depth map.
3. **3D Reconstruction**: Filtering depth noise and projecting the depth map into a 3D polygonal mesh or point cloud.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

# THIS IS UNDER DEVELOPMENT!!!
```
