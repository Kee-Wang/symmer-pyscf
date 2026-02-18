#!/usr/bin/env python
"""Standalone end-to-end test of the bond-scaling pipeline.

Runs 10 molecules from the CSV database through the scaling pipeline,
generates energy-vs-alpha plots, and produces a markdown report.

Usage:
    python scaling_test/run_test.py
"""

import json
import os
import time
from glob import glob

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from symmerpyscf.scaling import (
    parse_molecule_csv,
    generate_scaling_grid,
    run_molecule_scan,
)

# ── Configuration ─────────────────────────────────────────────────────────

CSV_PATH = (
    "/Users/qwang17/Library/CloudStorage/OneDrive-Tufts/research/"
    "9-quantum-molecule-zoo/pipeline/outputs/3_screening_usable_18q.csv"
)
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
HAM_DIR = os.path.join(BASE_DIR, "hamiltonians")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
REPORT_PATH = os.path.join(BASE_DIR, "report.md")
LOG_PATH = os.path.join(BASE_DIR, "pipeline_log.json")

TARGET_IDS = [
    # Batch 1: 8-10 qubits, 4-6 electrons
    "B+_singlet_Kh",
    "BH2+_singlet_C2v",
    "BH_singlet_Coov",
    "B_doublet_Kh",
    "BeH+_singlet_Coov",
    # Batch 2: 10-18 qubits, 10-18 electrons (stress test)
    "H2O_singlet_C2v",      # 10q, 10e
    "CH4_singlet_Td",       # 14q, 10e
    "N2_singlet_Dooh",      # 15q, 14e — triple bond
    "CO_singlet_Coov",      # 16q, 14e — hard correlation
    "H2S_singlet_C2v",      # 18q, 18e — dataset ceiling
]

# ~10-point grid: dense in [0.8, 2.0], sparse outside
GRID = generate_scaling_grid(
    alpha_min=0.5, alpha_max=3.0,
    dense_step=0.2, sparse_step=0.5,
)

# Energy methods and their plot styles
METHODS = ["HF", "MP2", "CISD", "CCSD", "FCI"]
METHOD_STYLES = {
    "HF":   {"color": "#1f77b4", "marker": "o",  "label": "HF"},
    "MP2":  {"color": "#ff7f0e", "marker": "s",  "label": "MP2"},
    "CISD": {"color": "#2ca02c", "marker": "^",  "label": "CISD"},
    "CCSD": {"color": "#d62728", "marker": "D",  "label": "CCSD"},
    "FCI":  {"color": "#9467bd", "marker": "v",  "label": "FCI"},
}


# ── Phase 1: Setup ────────────────────────────────────────────────────────

def setup():
    """Parse CSV, filter to target molecules, create directories."""
    os.makedirs(HAM_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"Grid ({len(GRID)} points): {GRID}")

    records = parse_molecule_csv(CSV_PATH)
    id_set = set(TARGET_IDS)
    targets = [r for r in records if r.id in id_set]

    # Preserve requested order
    id_to_rec = {r.id: r for r in targets}
    ordered = [id_to_rec[mid] for mid in TARGET_IDS if mid in id_to_rec]

    missing = id_set - {r.id for r in targets}
    if missing:
        print(f"WARNING: IDs not found in CSV: {missing}")

    print(f"Selected {len(ordered)} molecules:")
    for m in ordered:
        atom_tag = " (single atom)" if m.is_single_atom else ""
        q_str = f"  q={m.n_qubits_sto3g}" if m.n_qubits_sto3g else ""
        print(f"  {m.id}  {m.formula}  charge={m.charge} mult={m.multiplicity}"
              f"  e={m.n_electrons}{q_str}{atom_tag}")

    return ordered


# ── Phase 2: Compute ──────────────────────────────────────────────────────

def compute(molecules):
    """Run scaling scans for all molecules."""
    timing = {}
    summaries = {}

    for i, mol in enumerate(molecules):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(molecules)}] {mol.id} ({mol.formula})")
        print(f"{'='*60}")

        t0 = time.time()
        summary = run_molecule_scan(
            mol, HAM_DIR, grid=GRID, flat_naming=True, adaptive_stop=True,
        )
        elapsed = time.time() - t0

        timing[mol.id] = elapsed
        summaries[mol.id] = summary
        print(f"  Total: {summary['n_points_computed']} computed, "
              f"{summary['n_points_skipped']} skipped, "
              f"{summary['n_failed']} failed  [{elapsed:.1f}s]")

    return summaries, timing


# ── Phase 3: Plot ─────────────────────────────────────────────────────────

def load_molecule_data(mol_id):
    """Load all JSON files for a molecule, return sorted (alpha, data) list."""
    pattern = os.path.join(HAM_DIR, f"{mol_id}_alpha_*.json")
    files = sorted(glob(pattern))
    points = []
    for fpath in files:
        with open(fpath, 'r') as f:
            data = json.load(f)
        alpha = data.get('scaling_metadata', {}).get('alpha')
        if alpha is None:
            # Fallback: parse from filename
            base = os.path.basename(fpath)
            alpha = float(base.split('_alpha_')[1].replace('.json', ''))
        points.append((alpha, data))
    points.sort(key=lambda x: x[0])
    return points


def plot_line(mol_id, points):
    """Line plot: E vs alpha for multi-atom molecules."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHODS:
        alphas = []
        energies = []
        for alpha, data in points:
            e = data.get('calculated_properties', {}).get(method, {}).get('energy')
            if e is not None:
                alphas.append(alpha)
                energies.append(e)
        if energies:
            style = METHOD_STYLES[method]
            ax.plot(alphas, energies, color=style["color"],
                    marker=style["marker"], label=style["label"],
                    linewidth=1.5, markersize=5)

    ax.set_xlabel("Bond-scaling factor (alpha)", fontsize=12)
    ax.set_ylabel("Energy (Hartree)", fontsize=12)
    ax.set_title(f"{mol_id}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(PLOT_DIR, f"{mol_id}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out}")
    return out


def plot_bar(mol_id, points):
    """Horizontal bar chart for single-atom molecules (alpha=1.0 only)."""
    if not points:
        print(f"  No data for {mol_id}, skipping plot")
        return None

    _, data = points[0]  # single point
    props = data.get('calculated_properties', {})

    methods_present = []
    energies_present = []
    colors_present = []
    for method in METHODS:
        e = props.get(method, {}).get('energy')
        if e is not None:
            methods_present.append(method)
            energies_present.append(e)
            colors_present.append(METHOD_STYLES[method]["color"])

    if not methods_present:
        print(f"  No valid energies for {mol_id}, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(7, 3))
    y_pos = range(len(methods_present))
    ax.barh(y_pos, energies_present, color=colors_present, height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods_present, fontsize=11)
    ax.set_xlabel("Energy (Hartree)", fontsize=12)
    ax.set_title(f"{mol_id} (single atom, alpha=1.0)", fontsize=13)

    # Add value labels
    for i, (m, e) in enumerate(zip(methods_present, energies_present)):
        ax.text(e, i, f"  {e:.6f}", va='center', fontsize=9)

    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()

    out = os.path.join(PLOT_DIR, f"{mol_id}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out}")
    return out


def make_plots(molecules):
    """Generate plots for all molecules."""
    plot_paths = {}
    for mol in molecules:
        points = load_molecule_data(mol.id)
        if mol.is_single_atom:
            path = plot_bar(mol.id, points)
        else:
            path = plot_line(mol.id, points)
        plot_paths[mol.id] = path
    return plot_paths


# ── Phase 4: Report ───────────────────────────────────────────────────────

def generate_report(molecules, summaries, timing, plot_paths):
    """Generate a markdown report with tables and embedded images."""
    lines = []
    lines.append("# Bond-Scaling Pipeline Test Report\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Grid**: {len(GRID)} points, alpha = "
                 f"[{GRID[0]:.1f}, ..., {GRID[-1]:.1f}]\n")
    lines.append(f"**Basis**: sto-3g\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| # | ID | Formula | Atoms | Charge | Mult | e- | Qubits | "
                 "Computed | Skipped | Failed | Time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, mol in enumerate(molecules):
        s = summaries.get(mol.id, {})
        t = timing.get(mol.id, 0)
        q = mol.n_qubits_sto3g or "?"
        lines.append(
            f"| {i+1} | `{mol.id}` | {mol.formula} | {mol.n_atoms} | "
            f"{mol.charge} | {mol.multiplicity} | {mol.n_electrons} | {q} | "
            f"{s.get('n_points_computed', '?')} | "
            f"{s.get('n_points_skipped', '?')} | "
            f"{s.get('n_failed', '?')} | {t:.1f} |"
        )
    lines.append("")

    # Per-molecule sections
    for mol in molecules:
        lines.append(f"---\n")
        lines.append(f"## {mol.id}\n")
        lines.append(f"- **Formula**: {mol.formula}")
        lines.append(f"- **Atoms**: {mol.n_atoms}")
        lines.append(f"- **Charge**: {mol.charge}")
        lines.append(f"- **Multiplicity**: {mol.multiplicity}")
        lines.append(f"- **Electrons**: {mol.n_electrons}")
        lines.append(f"- **Qubits (STO-3G)**: {mol.n_qubits_sto3g or '?'}")
        lines.append(f"- **Single atom**: {'Yes' if mol.is_single_atom else 'No'}")
        lines.append("")

        # Energy table
        points = load_molecule_data(mol.id)
        if points:
            lines.append("### Energies (Hartree)\n")
            lines.append("| Alpha | HF | MP2 | CISD | CCSD | FCI | Errors |")
            lines.append("|---|---|---|---|---|---|---|")
            for alpha, data in points:
                props = data.get('calculated_properties', {})
                errors = data.get('_errors', {})
                row = [f"{alpha:.3f}"]
                for method in METHODS:
                    e = props.get(method, {}).get('energy')
                    row.append(f"{e:.6f}" if e is not None else "—")
                err_str = ", ".join(errors.keys()) if errors else "—"
                row.append(err_str)
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        # Embedded plot (relative path, URL-encoded for markdown viewers)
        ppath = plot_paths.get(mol.id)
        if ppath:
            from urllib.parse import quote
            rel_path = os.path.relpath(ppath, BASE_DIR)
            rel_path = quote(rel_path, safe='/')
            lines.append(f"![{mol.id}]({rel_path})\n")

    # Write report
    report_text = "\n".join(lines)
    with open(REPORT_PATH, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved: {REPORT_PATH}")


def save_log(summaries, timing):
    """Save timing and summary data as JSON."""
    log = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "grid": GRID.tolist(),
        "timing": timing,
        "summaries": summaries,
    }
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Log saved: {LOG_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Bond-Scaling Pipeline — End-to-End Test")
    print("=" * 60)

    t_total = time.time()

    # Phase 1
    print("\n── Phase 1: Setup ──")
    molecules = setup()

    # Phase 2
    print("\n── Phase 2: Compute ──")
    summaries, timing = compute(molecules)

    # Phase 3
    print("\n── Phase 3: Plot ──")
    plot_paths = make_plots(molecules)

    # Phase 4
    print("\n── Phase 4: Report ──")
    generate_report(molecules, summaries, timing, plot_paths)
    save_log(summaries, timing)

    total = time.time() - t_total
    print(f"\nAll done! Total elapsed: {total:.1f}s")
    print(f"  Hamiltonians: {HAM_DIR}")
    print(f"  Plots: {PLOT_DIR}")
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
