from __future__ import annotations

from pathlib import Path

import numpy as np

from fem3d.mesh import TetMesh


def render_deformed_surface_png(
    path: str | Path,
    mesh: TetMesh,
    displacement: np.ndarray,
    cell_values: np.ndarray,
    displacement_scale: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output = Path(path)
    deformed = mesh.nodes + displacement_scale * displacement
    faces, owners = mesh.boundary_faces_with_owners()
    polygons = [deformed[face] for face in faces]
    values = np.asarray(cell_values, dtype=float)[owners]
    collection = Poly3DCollection(polygons, linewidths=0.25, edgecolors="0.25")
    collection.set_array(values)
    collection.set_cmap("viridis")
    collection.set_clim(float(values.min()), float(values.max()))

    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(collection)
    _set_equal_3d_axes(ax, deformed)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    colorbar = fig.colorbar(collection, ax=ax, shrink=0.7, pad=0.02)
    colorbar.set_label("von Mises stress")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_convergence_png(
    path: str | Path,
    h: np.ndarray,
    l2: np.ndarray,
    h1: np.ndarray,
    l2_rate: float | None,
    h1_rate: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.loglog(h, l2, "o-", label=_slope_label("L2", l2_rate))
    ax.loglog(h, h1, "s-", label=_slope_label("H1 seminorm", h1_rate))
    ax.invert_xaxis()
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.set_xlabel("mesh size h")
    ax.set_ylabel("error norm")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _slope_label(name: str, rate: float | None) -> str:
    return name if rate is None else f"{name} slope {rate:.2f}"


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float((maxs - mins).max())
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
