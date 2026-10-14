from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from .datastructures import StereogramImage


@dataclass
class MeshResult:
    """
    Data structure representing a 3D surface mesh reconstructed from an autostereogram depth map.

    Attributes:
        vertices: Array of shape (N, 3) containing 3D vertex coordinates (X, Y, Z).
        triangles: Array of shape (M, 3) containing 0-indexed triangle vertex indices.
        normals: Array of shape (N, 3) containing unit normal vectors at each vertex.
        vertex_colors: Array of shape (N, 3) with float RGB values in [0.0, 1.0].
        uvs: Array of shape (N, 2) with normalized texture coordinates in [0.0, 1.0].
        metadata: Dictionary with configuration and generation parameters.
    """

    vertices: np.ndarray  # Shape: (N, 3), float32
    triangles: np.ndarray  # Shape: (M, 3), int32
    normals: Optional[np.ndarray] = None  # Shape: (N, 3), float32
    vertex_colors: Optional[np.ndarray] = None  # Shape: (N, 3), float32 [0..1]
    uvs: Optional[np.ndarray] = None  # Shape: (N, 2), float32 [0..1]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must have shape (N, 3), got {self.vertices.shape}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"Triangles must have shape (M, 3), got {self.triangles.shape}")
        if self.normals is not None and self.normals.shape != self.vertices.shape:
            raise ValueError("Normals shape must match vertices shape")
        if self.vertex_colors is not None and self.vertex_colors.shape != self.vertices.shape:
            raise ValueError("Vertex colors shape must match vertices shape")
        if self.uvs is not None and (self.uvs.ndim != 2 or self.uvs.shape[0] != self.vertices.shape[0]):
            raise ValueError("UVs must have shape (N, 2) matching vertex count")

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_triangles(self) -> int:
        return self.triangles.shape[0]

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) bounding box."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def to_open3d(
        self,
        include_texture: bool = False,
        texture_image: Optional[np.ndarray] = None,
        include_uvs: bool = True,
    ):
        """
        Convert to an Open3D TriangleMesh.
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for to_open3d(). Run: pip install open3d")

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles.astype(np.int32))

        if self.normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(self.normals.astype(np.float64))
        else:
            mesh.compute_vertex_normals()

        if self.vertex_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors.astype(np.float64))

        if include_uvs and self.uvs is not None:
            tri_uvs = np.empty((len(self.triangles) * 3, 2), dtype=np.float64)
            tri_uvs[0::3] = self.uvs[self.triangles[:, 0]]
            tri_uvs[1::3] = self.uvs[self.triangles[:, 1]]
            tri_uvs[2::3] = self.uvs[self.triangles[:, 2]]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)

        if include_texture and texture_image is not None:
            if texture_image.dtype != np.uint8:
                texture_image = (np.clip(texture_image, 0.0, 1.0) * 255).astype(np.uint8)
            o3d_img = o3d.geometry.Image(texture_image)
            mesh.textures = [o3d_img]

        return mesh

    def to_trimesh(self):
        """
        Convert to a Trimesh object.
        """
        if not HAS_TRIMESH:
            raise ImportError("Trimesh is required for to_trimesh(). Run: pip install trimesh")

        vc = None
        if self.vertex_colors is not None:
            vc = (np.clip(self.vertex_colors, 0.0, 1.0) * 255).astype(np.uint8)

        return trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.triangles,
            vertex_normals=self.normals,
            vertex_colors=vc,
            visual=trimesh.visual.TextureVisuals(uv=self.uvs) if self.uvs is not None else None,
            process=False,
        )

    def save_obj(
        self,
        filepath: Union[str, Path],
        texture_image: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Export mesh to Wavefront OBJ format.

        If Open3D is available, utilizes C++ accelerated writer with vertex colors,
        normals, and face definitions. If texture_image is provided, exports associated
        MTL material and PNG texture map.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_OPEN3D:
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            mesh = self.to_open3d(
                include_texture=(texture_image is not None),
                texture_image=texture_image,
                include_uvs=True,
            )
            o3d.io.write_triangle_mesh(str(path), mesh)
            return path

        # Pure NumPy / Python fallback exporter
        return self._write_obj_fallback(path)

    def _write_obj_fallback(self, filepath: Path) -> Path:
        """Fallback OBJ writer when Open3D is unavailable."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Stereo3D Reconstructed Mesh\n")
            f.write(f"# Vertices: {self.num_vertices}\n")
            f.write(f"# Triangles: {self.num_triangles}\n")

            has_colors = self.vertex_colors is not None
            has_normals = self.normals is not None
            has_uvs = self.uvs is not None

            # Vertices and vertex colors
            for i in range(self.num_vertices):
                v = self.vertices[i]
                if has_colors:
                    c = self.vertex_colors[i]
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
                else:
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

            # Normals
            if has_normals:
                for i in range(self.num_vertices):
                    n = self.normals[i]
                    f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")

            # UV coordinates
            if has_uvs:
                for i in range(self.num_vertices):
                    u = self.uvs[i]
                    f.write(f"vt {u[0]:.4f} {u[1]:.4f}\n")

            # Faces (1-indexed)
            if has_normals and has_uvs:
                for t in self.triangles:
                    f.write(f"f {t[0]+1}/{t[0]+1}/{t[0]+1} {t[1]+1}/{t[1]+1}/{t[1]+1} {t[2]+1}/{t[2]+1}/{t[2]+1}\n")
            elif has_normals:
                for t in self.triangles:
                    f.write(f"f {t[0]+1}//{t[0]+1} {t[1]+1}//{t[1]+1} {t[2]+1}//{t[2]+1}\n")
            else:
                for t in self.triangles:
                    f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

        return filepath

    def save_ply(self, filepath: Union[str, Path], binary: bool = True) -> Path:
        """
        Export mesh to Stanford PLY format.

        Supports binary export for minimal file size and fast loading.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_OPEN3D:
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            mesh = self.to_open3d(include_uvs=False)
            o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=not binary)
            return path

        if HAS_TRIMESH:
            tm = self.to_trimesh()
            tm.export(str(path), file_type="ply", encoding="binary" if binary else "ascii")
            return path

        raise RuntimeError("Open3D or Trimesh required for PLY export.")


class MeshGenerator:
    """
    Converts 2D depth maps extracted from autostereograms into full 3D surface meshes.

    Performs regular grid triangulations with custom displacement scaling,
    normal computation, and texture / colormap sampling.
    """

    def __init__(
        self,
        z_scale: Optional[float] = None,
        step: int = 2,
        color_mode: str = "rgb",
        colormap: str = "turbo",
        smooth_normals: bool = True,
        invert_z: bool = False,
    ):
        """
        Args:
            z_scale: Height displacement scaling factor. If None, defaults to 20% of max dimension.
            step: Subsampling step for grid decimation (step=1 uses every pixel, step=2 samples every 2nd pixel).
            color_mode: Coloring strategy: 'rgb' (stereogram colors), 'depth' (colormap), or 'uniform'.
            colormap: Matplotlib colormap used when color_mode is 'depth'.
            smooth_normals: Compute smooth area-weighted vertex normals.
            invert_z: Invert displacement direction (concave vs convex).
        """
        if step < 1:
            raise ValueError(f"Step must be an integer >= 1, got {step}")

        self.z_scale = z_scale
        self.step = step
        self.color_mode = color_mode.lower()
        self.colormap = colormap
        self.smooth_normals = smooth_normals
        self.invert_z = invert_z

    def generate(
        self,
        depth: np.ndarray,
        stereogram: Optional[StereogramImage] = None,
    ) -> MeshResult:
        """
        Generate 3D mesh from depth map and optional stereogram image.

        Args:
            depth: 2D array of shape (H, W) with normalized depth values in [0.0, 1.0].
            stereogram: StereogramImage instance for RGB texture sampling.

        Returns:
            MeshResult instance with vertices, triangles, normals, colors, and UVs.
        """
        if depth.ndim != 2:
            raise ValueError(f"Depth array must have 2 dimensions (H, W), got {depth.shape}")

        height, width = depth.shape

        # Grid subsampling step
        eff_step = max(1, self.step)

        # Calculate extrusion scaling
        z_scale = self.z_scale
        if z_scale is None:
            z_scale = 0.20 * max(width, height)

        # 1. Subsample depth map
        depth_sub = depth[::eff_step, ::eff_step].astype(np.float32)
        hs, ws = depth_sub.shape

        # 2. Compute vertex coordinates (centered at origin, Y points up)
        xs = (np.arange(ws, dtype=np.float32) * eff_step) - (width - 1) / 2.0
        ys = ((height - 1) - np.arange(hs, dtype=np.float32) * eff_step) - (height - 1) / 2.0
        grid_x, grid_y = np.meshgrid(xs, ys)

        grid_z = depth_sub * float(z_scale)
        if self.invert_z:
            grid_z = -grid_z

        vertices = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).astype(np.float32)

        # 3. Vectorized quad triangulation (CCW winding facing camera +Z)
        r_grid, c_grid = np.meshgrid(
            np.arange(hs - 1, dtype=np.int32),
            np.arange(ws - 1, dtype=np.int32),
            indexing="ij",
        )
        v0 = (r_grid * ws + c_grid).ravel()
        v1 = ((r_grid + 1) * ws + c_grid).ravel()
        v2 = ((r_grid + 1) * ws + (c_grid + 1)).ravel()
        v3 = (r_grid * ws + (c_grid + 1)).ravel()

        t1 = np.column_stack([v0, v1, v2])
        t2 = np.column_stack([v0, v2, v3])

        triangles = np.empty((len(v0) * 2, 3), dtype=np.int32)
        triangles[0::2] = t1
        triangles[1::2] = t2

        # 4. Normalized UV texture coordinates ([0, 1] x [0, 1])
        u_1d = (np.arange(ws, dtype=np.float32) * eff_step) / max(1.0, float(width - 1))
        v_1d = 1.0 - (np.arange(hs, dtype=np.float32) * eff_step) / max(1.0, float(height - 1))
        grid_u, grid_v = np.meshgrid(u_1d, v_1d)
        uvs = np.column_stack([grid_u.ravel(), grid_v.ravel()]).astype(np.float32)

        # 5. Vertex colors
        if self.color_mode == "rgb" and stereogram is not None:
            rgb_sub = stereogram.rgb[::eff_step, ::eff_step].astype(np.float32) / 255.0
            vertex_colors = rgb_sub.reshape(-1, 3).astype(np.float32)
        elif self.color_mode == "depth" or (self.color_mode == "rgb" and stereogram is None):
            cmap = plt.get_cmap(self.colormap)
            vertex_colors = cmap(depth_sub.ravel())[:, :3].astype(np.float32)
        else:
            # Uniform neutral studio clay
            vertex_colors = np.full((len(vertices), 3), 0.85, dtype=np.float32)

        # 6. Smooth Normals
        normals = None
        if self.smooth_normals:
            normals = self._compute_normals(vertices, triangles)

        meta = {
            "image_shape": (height, width),
            "grid_shape": (hs, ws),
            "step": eff_step,
            "z_scale": z_scale,
            "color_mode": self.color_mode,
            "invert_z": self.invert_z,
        }

        return MeshResult(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
            vertex_colors=vertex_colors,
            uvs=uvs,
            metadata=meta,
        )

    @staticmethod
    def _compute_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Compute smooth vertex normals via face normal accumulation."""
        e1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
        e2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
        face_normals = np.cross(e1, e2)

        normals = np.zeros_like(vertices, dtype=np.float32)
        np.add.at(normals, triangles[:, 0], face_normals)
        np.add.at(normals, triangles[:, 1], face_normals)
        np.add.at(normals, triangles[:, 2], face_normals)

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(norms, 1e-8)
        return normals
