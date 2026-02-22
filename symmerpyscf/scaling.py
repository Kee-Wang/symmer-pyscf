"""Geometry scaling utilities for bond-length scans.

Provides pure functions for scaling molecular geometries, generating
adaptive alpha grids, and detecting convergence in energy curves.
"""

from typing import List, Tuple, Optional, Dict

import numpy as np


def _parse_xyz_string(xyz: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Parse xyz string into geometry list.

    Format: "H 0.0 0.0 0.0\\nH 0.0 0.0 0.74"
    Returns: [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))]
    """
    geometry = []
    for line in xyz.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) != 4:
            raise ValueError(
                f"Expected 4 fields (element x y z), got {len(parts)}: '{line}'"
            )
        element = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        geometry.append((element, (x, y, z)))
    return geometry


def scale_geometry(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    alpha: float,
) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Scale all inter-atomic distances by *alpha* relative to the centroid.

    Args:
        geometry: List of (atom, (x, y, z)) tuples.
        alpha: Scaling factor (1.0 = equilibrium geometry).

    Returns:
        New geometry with distances from centroid scaled by alpha.
    """
    coords = np.array([c for _, c in geometry])
    centroid = coords.mean(axis=0)
    scaled = centroid + alpha * (coords - centroid)
    return [(el, tuple(row)) for el, row in zip([e for e, _ in geometry], scaled)]


def generate_scaling_grid(
    alpha_min: float = 0.5,
    alpha_max: float = 3.0,
    dense_min: float = 0.8,
    dense_max: float = 2.0,
    dense_step: float = 0.05,
    sparse_step: float = 0.25,
) -> np.ndarray:
    """Generate an adaptive scaling grid.

    Dense spacing in [dense_min, dense_max] (the bond-breaking region),
    sparse spacing outside (repulsive wall and dissociation tail).
    Endpoints are always included.

    Args:
        alpha_min: Smallest alpha value.
        alpha_max: Largest alpha value.
        dense_min: Start of the dense region.
        dense_max: End of the dense region.
        dense_step: Step size in the dense region.
        sparse_step: Step size in the sparse regions.

    Returns:
        Sorted, deduplicated array of alpha values.
    """
    points = set()

    # Sparse region below dense_min
    a = alpha_min
    while a < dense_min - 1e-10:
        points.add(round(a, 6))
        a += sparse_step
    # Dense region
    a = dense_min
    while a <= dense_max + 1e-10:
        points.add(round(a, 6))
        a += dense_step
    # Sparse region above dense_max
    a = dense_max + sparse_step
    while a <= alpha_max + 1e-10:
        points.add(round(a, 6))
        a += sparse_step
    # Ensure endpoints
    points.add(round(alpha_min, 6))
    points.add(round(alpha_max, 6))

    return np.array(sorted(points))


def should_stop_scanning(
    energies: Dict[float, Optional[float]],
    current_alpha: float,
    threshold: float = 1e-4,
) -> bool:
    """Check if the FCI energy curve has flattened (dissociation limit reached).

    Only activates after alpha > 2.0. Checks if |delta_E_FCI| < threshold
    per 0.1 step in alpha over the last few points.

    Args:
        energies: Dict mapping alpha -> FCI energy (None if FCI failed).
        current_alpha: The alpha value just computed.
        threshold: Energy change threshold per 0.1 alpha step.

    Returns:
        True if scanning should stop.
    """
    if current_alpha <= 2.0:
        return False

    # Need at least 2 valid consecutive points to compare
    sorted_alphas = sorted(a for a, e in energies.items() if e is not None)
    if len(sorted_alphas) < 2:
        return False

    # Check the last two valid points
    a_prev, a_curr = sorted_alphas[-2], sorted_alphas[-1]
    e_prev, e_curr = energies[a_prev], energies[a_curr]
    delta_alpha = a_curr - a_prev
    if delta_alpha < 1e-10:
        return False

    # Normalize to per-0.1-alpha-step
    delta_e_per_step = abs(e_curr - e_prev) / delta_alpha * 0.1
    return delta_e_per_step < threshold
