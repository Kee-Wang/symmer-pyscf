"""Bond-scaling Hamiltonian database pipeline.

Generates Symmer-format Hamiltonian JSON files across bond-scaling
trajectories (alpha = 0.5 -> 3.0) for stress-testing quantum computing
algorithms on static correlation problems.
"""

import csv
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from .molecule import generate_symmer_data


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MoleculeRecord:
    """A single molecule parsed from the CSV database."""
    id: str
    species: str
    formula: str
    name: str
    n_atoms: int
    charge: int
    multiplicity: int
    n_electrons: int
    geometry: List[Tuple[str, Tuple[float, float, float]]]
    reference_energy: Optional[float] = None
    n_qubits_sto3g: Optional[int] = None

    @property
    def is_single_atom(self) -> bool:
        return self.n_atoms == 1


@dataclass
class ScalingResult:
    """Result of a single alpha-point calculation."""
    status: str  # "success", "partial", "failed"
    alpha: float
    molecule_id: str
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    output_file: Optional[str] = None


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_molecule_csv(csv_path: str) -> List[MoleculeRecord]:
    """Parse a CSV file of molecules into MoleculeRecord objects.

    Expected CSV columns:
        id, species, formula, name, n_atoms, charge, multiplicity,
        n_electrons, xyz, reference_energy (optional), sto-3g (optional)

    The 'xyz' column should contain newline-separated lines of
    "Element x y z" (e.g. "H 0.0 0.0 0.0\\nH 0.0 0.0 0.74").

    Returns:
        List of MoleculeRecord objects.
    """
    records = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # row 1 = header
            try:
                geometry = _parse_xyz_string(row['xyz'].strip())
                n_qubits = _parse_qubit_count(row.get('sto-3g', ''))
                ref_energy_str = row.get('reference_energy', '').strip()
                ref_energy = float(ref_energy_str) if ref_energy_str else None

                record = MoleculeRecord(
                    id=row['id'].strip(),
                    species=row['species'].strip(),
                    formula=row['formula'].strip(),
                    name=row['name'].strip(),
                    n_atoms=int(row['n_atoms']),
                    charge=int(row['charge']),
                    multiplicity=int(row['multiplicity']),
                    n_electrons=int(row['n_electrons']),
                    geometry=geometry,
                    reference_energy=ref_energy,
                    n_qubits_sto3g=n_qubits,
                )
                records.append(record)
            except Exception as e:
                tb = traceback.format_exc()
                print(f'WARNING [AUDIT]: Skipping CSV row {row_num}\n'
                      f'  Row data: {row}\n'
                      f'  Error: {type(e).__name__}: {e}\n'
                      f'  Traceback:\n{tb}')
                continue
    return records


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


def _parse_qubit_count(value: str) -> Optional[int]:
    """Parse qubit count from sto-3g column, stripping any emoji prefix."""
    value = value.strip()
    if not value:
        return None
    # Strip leading non-digit characters (emoji, spaces, etc.)
    digits = ''.join(c for c in value if c.isdigit())
    return int(digits) if digits else None


# ---------------------------------------------------------------------------
# Geometry scaling
# ---------------------------------------------------------------------------

def scale_geometry(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    alpha: float,
) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Uniformly scale all atomic coordinates by factor *alpha*.

    Args:
        geometry: List of (atom, (x, y, z)) tuples.
        alpha: Scaling factor (1.0 = equilibrium geometry).

    Returns:
        New geometry with all coordinates multiplied by alpha.
    """
    return [
        (element, (x * alpha, y * alpha, z * alpha))
        for element, (x, y, z) in geometry
    ]


# ---------------------------------------------------------------------------
# Adaptive grid
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-point runner (outer error layer)
# ---------------------------------------------------------------------------

def run_single_point(
    molecule: MoleculeRecord,
    alpha: float,
    basis: str = "sto-3g",
    verbose: bool = False,
) -> Tuple[ScalingResult, Optional[Dict[str, Any]]]:
    """Run a single alpha-point calculation for a molecule.

    Wraps ``generate_symmer_data`` with blanket error handling so that
    catastrophic failures produce an error report instead of crashing
    the pipeline.

    Args:
        molecule: Parsed molecule record from the CSV database.
        alpha: Bond-scaling factor (1.0 = equilibrium geometry).
        basis: Basis set name (default ``"sto-3g"``).
        verbose: Print calculation progress.

    Returns:
        Tuple of (ScalingResult, symmer_data or None).
    """
    t0 = time.time()
    scaled_geom = scale_geometry(molecule.geometry, alpha)

    try:
        mol_info, symmer_data = generate_symmer_data(
            geometry=scaled_geom,
            basis=basis,
            charge=molecule.charge,
            multiplicity=molecule.multiplicity,
            verbose=verbose,
        )

        # Determine status from what succeeded
        errors_in_data = symmer_data.get('_errors', {})
        calc_props = symmer_data.get('calculated_properties', {})

        # Check which solvers produced None energy
        failed_solvers = [
            method for method, props in calc_props.items()
            if props.get('energy') is None
        ]

        warnings = []
        if errors_in_data:
            warnings.extend(
                f'{k}: {v}' for k, v in errors_in_data.items()
            )
        if failed_solvers:
            warnings.append(f'Solvers with None energy: {failed_solvers}')

        if failed_solvers or errors_in_data:
            status = "partial"
        else:
            status = "success"

        # Attach scaling metadata to the output
        symmer_data['scaling_metadata'] = {
            'alpha': alpha,
            'molecule_id': molecule.id,
            'molecule_name': molecule.name,
            'formula': molecule.formula,
            'basis': basis,
        }
        symmer_data['status'] = status

        elapsed = time.time() - t0
        result = ScalingResult(
            status=status,
            alpha=alpha,
            molecule_id=molecule.id,
            elapsed_seconds=elapsed,
            warnings=warnings,
        )
        return result, symmer_data

    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(f'ERROR [AUDIT]: Catastrophic failure for {molecule.id} at alpha={alpha}\n'
              f'  Error: {type(e).__name__}: {e}\n'
              f'  Traceback:\n{tb}')

        error_data = {
            'status': 'failed',
            'scaling_metadata': {
                'alpha': alpha,
                'molecule_id': molecule.id,
                'molecule_name': molecule.name,
                'formula': molecule.formula,
                'basis': basis,
            },
            '_errors': {
                'catastrophic': f'{type(e).__name__}: {e}',
                'traceback': tb,
            },
        }
        result = ScalingResult(
            status="failed",
            alpha=alpha,
            molecule_id=molecule.id,
            elapsed_seconds=elapsed,
            error=f'{type(e).__name__}: {e}',
            error_traceback=tb,
        )
        return result, error_data


# ---------------------------------------------------------------------------
# Molecule scan
# ---------------------------------------------------------------------------

def run_molecule_scan(
    molecule: MoleculeRecord,
    output_dir: str,
    grid: Optional[np.ndarray] = None,
    basis: str = "sto-3g",
    verbose: bool = False,
    adaptive_stop: bool = True,
    flat_naming: bool = False,
) -> Dict[str, Any]:
    """Run a full scaling scan for a single molecule.

    Args:
        molecule: MoleculeRecord to scan.
        output_dir: Base output directory. Files go under output_dir/<molecule.id>/
            unless flat_naming is True.
        grid: Alpha values to scan. If None, uses generate_scaling_grid().
        basis: Basis set.
        verbose: Verbose output from PySCF.
        adaptive_stop: If True, stop scanning when energy curve flattens.
        flat_naming: If True, save files as <output_dir>/<id>_alpha_<value>.json
            instead of <output_dir>/<id>/alpha_<value>.json.

    Returns:
        Summary dict with per-alpha results.
    """
    if grid is None:
        grid = generate_scaling_grid()

    # Single atoms: only compute alpha=1.0
    if molecule.is_single_atom:
        grid = np.array([1.0])

    if flat_naming:
        mol_dir = output_dir
    else:
        mol_dir = os.path.join(output_dir, molecule.id)
    os.makedirs(mol_dir, exist_ok=True)

    fci_energies: Dict[float, Optional[float]] = {}
    results = []

    for alpha in grid:
        # Resumability: skip if output JSON already exists
        if flat_naming:
            out_file = os.path.join(mol_dir, f'{molecule.id}_alpha_{alpha:.3f}.json')
        else:
            out_file = os.path.join(mol_dir, f'alpha_{alpha:.3f}.json')
        if os.path.exists(out_file):
            print(f'  [SKIP] {molecule.id} alpha={alpha:.3f} — already exists')
            # Try to load existing FCI energy for adaptive stopping
            try:
                with open(out_file, 'r') as f:
                    existing = json.load(f)
                fci_e = existing.get('calculated_properties', {}).get('FCI', {}).get('energy')
                fci_energies[alpha] = fci_e
            except Exception:
                pass
            results.append({'alpha': alpha, 'status': 'skipped'})
            continue

        print(f'  [RUN] {molecule.id} alpha={alpha:.3f} ...', end=' ', flush=True)
        scaling_result, symmer_data = run_single_point(
            molecule, alpha, basis=basis, verbose=verbose,
        )
        scaling_result.output_file = out_file

        # Save immediately
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(symmer_data, f, indent=4)

        # Track FCI energy for adaptive stopping
        fci_e = None
        if symmer_data and 'calculated_properties' in symmer_data:
            fci_e = symmer_data['calculated_properties'].get('FCI', {}).get('energy')
        fci_energies[alpha] = fci_e

        status_msg = scaling_result.status
        if scaling_result.warnings:
            status_msg += f' (warnings: {len(scaling_result.warnings)})'
        print(f'{status_msg} [{scaling_result.elapsed_seconds:.1f}s]')

        if scaling_result.warnings:
            for w in scaling_result.warnings:
                print(f'    WARNING [AUDIT]: {w}')

        results.append({
            'alpha': alpha,
            'status': scaling_result.status,
            'elapsed_seconds': scaling_result.elapsed_seconds,
            'warnings': scaling_result.warnings,
            'error': scaling_result.error,
        })

        # Adaptive stopping
        if adaptive_stop and should_stop_scanning(fci_energies, alpha):
            print(f'  [STOP] Energy curve flattened at alpha={alpha:.3f}, '
                  f'stopping scan for {molecule.id}')
            break

    return {
        'molecule_id': molecule.id,
        'formula': molecule.formula,
        'n_atoms': molecule.n_atoms,
        'n_points_computed': sum(1 for r in results if r['status'] != 'skipped'),
        'n_points_skipped': sum(1 for r in results if r['status'] == 'skipped'),
        'n_success': sum(1 for r in results if r['status'] == 'success'),
        'n_partial': sum(1 for r in results if r['status'] == 'partial'),
        'n_failed': sum(1 for r in results if r['status'] == 'failed'),
        'results': results,
    }


# ---------------------------------------------------------------------------
# Database pipeline
# ---------------------------------------------------------------------------

def run_database_pipeline(
    csv_path: str,
    output_dir: str,
    molecule_ids: Optional[List[str]] = None,
    skip_single_atoms: bool = False,
    basis: str = "sto-3g",
    verbose: bool = False,
    adaptive_stop: bool = True,
    grid: Optional[np.ndarray] = None,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Top-level orchestrator: parse CSV, loop molecules, save summary.

    Args:
        csv_path: Path to molecule CSV database.
        output_dir: Base output directory.
        molecule_ids: If provided, only process these molecule IDs.
        skip_single_atoms: If True, skip single-atom molecules entirely.
        basis: Basis set.
        verbose: Verbose PySCF output.
        adaptive_stop: Enable adaptive stopping per molecule.
        grid: Custom alpha grid. If None, uses generate_scaling_grid().
        log_file: Optional path for a log file (not yet implemented).

    Returns:
        Pipeline summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse CSV
    print(f'Parsing CSV: {csv_path}')
    records = parse_molecule_csv(csv_path)
    print(f'  Found {len(records)} molecules')

    # Filter
    if molecule_ids is not None:
        id_set = set(molecule_ids)
        records = [r for r in records if r.id in id_set]
        print(f'  Filtered to {len(records)} molecules by ID')
    if skip_single_atoms:
        records = [r for r in records if not r.is_single_atom]
        print(f'  After removing single atoms: {len(records)} molecules')

    # Run scans
    pipeline_t0 = time.time()
    molecule_summaries = []

    for i, mol in enumerate(records):
        print(f'\n[{i+1}/{len(records)}] {mol.id} ({mol.formula}, '
              f'{mol.n_atoms} atoms, charge={mol.charge}, '
              f'mult={mol.multiplicity})')
        summary = run_molecule_scan(
            mol, output_dir, grid=grid, basis=basis,
            verbose=verbose, adaptive_stop=adaptive_stop,
        )
        molecule_summaries.append(summary)

    pipeline_elapsed = time.time() - pipeline_t0

    # Pipeline summary
    pipeline_summary = {
        'csv_path': csv_path,
        'output_dir': output_dir,
        'basis': basis,
        'n_molecules': len(records),
        'total_elapsed_seconds': pipeline_elapsed,
        'molecule_summaries': molecule_summaries,
    }

    summary_file = os.path.join(output_dir, 'pipeline_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline_summary, f, indent=2)
    print(f'\nPipeline summary saved to {summary_file}')

    # Print overview
    total_success = sum(s['n_success'] for s in molecule_summaries)
    total_partial = sum(s['n_partial'] for s in molecule_summaries)
    total_failed = sum(s['n_failed'] for s in molecule_summaries)
    print(f'\nPipeline complete: {total_success} success, '
          f'{total_partial} partial, {total_failed} failed '
          f'({pipeline_elapsed:.1f}s total)')

    return pipeline_summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for the bond-scaling pipeline.

    Parses CLI arguments and delegates to ``run_database_pipeline``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Bond-scaling Hamiltonian database pipeline'
    )
    parser.add_argument('csv_path', help='Path to molecule CSV database')
    parser.add_argument('output_dir', help='Output directory for JSON files')
    parser.add_argument('--basis', default='sto-3g', help='Basis set (default: sto-3g)')
    parser.add_argument('--molecule-ids', nargs='+', default=None,
                        help='Only process these molecule IDs')
    parser.add_argument('--skip-single-atoms', action='store_true',
                        help='Skip single-atom molecules')
    parser.add_argument('--no-adaptive-stop', action='store_true',
                        help='Disable adaptive stopping')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose PySCF output')
    parser.add_argument('--alpha-min', type=float, default=0.5)
    parser.add_argument('--alpha-max', type=float, default=3.0)
    parser.add_argument('--dense-step', type=float, default=0.05)
    parser.add_argument('--sparse-step', type=float, default=0.25)

    args = parser.parse_args()

    grid = generate_scaling_grid(
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        dense_step=args.dense_step,
        sparse_step=args.sparse_step,
    )

    run_database_pipeline(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        molecule_ids=args.molecule_ids,
        skip_single_atoms=args.skip_single_atoms,
        basis=args.basis,
        verbose=args.verbose,
        adaptive_stop=not args.no_adaptive_stop,
        grid=grid,
    )


if __name__ == '__main__':
    main()
