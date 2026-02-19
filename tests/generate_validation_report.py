#!/usr/bin/env python3
"""Generate a human-auditable markdown validation report.

Compares symmerpyscf-generated Hamiltonians against symmer reference data
for all molecules in the molecule zoo, producing:

  - tests/validation_output/report.md            Markdown lab report
  - tests/validation_output/generated/*.json      Generated Hamiltonian files
  - tests/validation_output/figures/*.png          Plots
  - tests/validation_output/comparison_data.json   Machine-readable summary

Usage:
    python tests/generate_validation_report.py                 # Tier 1 only
    python tests/generate_validation_report.py --tiers 1 2     # Tiers 1 & 2
    python tests/generate_validation_report.py --tiers 1 2 3 4 # All tiers
"""

import argparse
import datetime
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports work when invoked as
# ``python tests/generate_validation_report.py`` from the repo root.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from test_molzoo_validation import (  # noqa: E402
    ALL_MOLECULES,
    SYMMER_SOURCE_DIR,
    _load_ref,
    _parse_ref_geometry,  # used ONLY for geometry precision comparison
    _KNOWN_FCI_ISSUES,
)
from symmerpyscf import generate_symmer_data  # noqa: E402
from symmerpyscf.scaling import _parse_xyz_string  # noqa: E402

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
ENERGY_TOL = 1e-6    # Hartree — matches PySCF conv_tol (molecule.py:69)
COEFF_TOL = 1e-5     # coefficient difference — one order above energy tol
FIDELITY_TOL = 1e-8  # Hilbert-Schmidt fidelity: 1 - F must be below this

# ---------------------------------------------------------------------------
# PySCF symmetry_subgroup mapping
# ---------------------------------------------------------------------------
# PySCF only accepts Abelian point groups (plus Coov/Dooh which it maps
# internally to C2v/D2h).  For non-Abelian groups, we pass None and let
# PySCF auto-detect the Abelian subgroup from the geometry.
_PYSCF_ACCEPTED_SUBGROUPS = frozenset({
    "C1", "Ci", "C2", "Cs", "C2v", "C2h", "D2", "D2h", "Coov", "Dooh",
})


def _to_pyscf_subgroup(point_group):
    """Map a molzoo point group to a PySCF-compatible symmetry_subgroup.

    Returns the group name if PySCF accepts it directly, otherwise None
    (letting PySCF auto-detect the largest Abelian subgroup).
    """
    if point_group in _PYSCF_ACCEPTED_SUBGROUPS:
        return point_group
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_complex(v):
    """Normalise mixed float / [real, imag] to complex."""
    if isinstance(v, list):
        return complex(v[0], v[1])
    return complex(v)


def _to_serialisable(v):
    """Convert a Hamiltonian value to [real, imag] list for JSON."""
    c = _to_complex(v)
    return [c.real, c.imag]


def _safe_energy(props, method):
    """Extract energy from calculated_properties, returning None on absence."""
    entry = props.get(method)
    if entry is None:
        return None
    e = entry.get("energy")
    if e is None:
        return None
    if isinstance(e, float) and np.isnan(e):
        return None
    return float(e)


def _safe_converged(props, method):
    """Extract convergence flag."""
    entry = props.get(method)
    if entry is None:
        return False
    return bool(entry.get("converged", False))


def _energy_record(ref_val, new_val):
    """Build a ref/new/diff dict for one energy method."""
    diff = None
    if ref_val is not None and new_val is not None:
        diff = new_val - ref_val
    return {"ref": ref_val, "new": new_val, "diff": diff}


def _energy_verdict(energy_rec, molzoo_id, method):
    """Per-method verdict for the energy table."""
    ref, new, diff = energy_rec.get("ref"), energy_rec.get("new"), energy_rec.get("diff")
    if ref is None:
        return "no ref"
    if new is None:
        return "no data"
    if diff is not None and abs(diff) < ENERGY_TOL:
        return "PASS"
    if molzoo_id in _KNOWN_FCI_ISSUES and method in ("FCI", "MP2"):
        return "WARN (known)"
    return "FAIL"


def _compare_geometries(geom_molzoo, geom_ref):
    """Compare two geometries and return max coordinate difference.

    Both geometries are lists of (element, (x, y, z)) tuples.
    Returns (max_coord_diff, per_atom_diffs) or (None, []) if incompatible.
    """
    if len(geom_molzoo) != len(geom_ref):
        return None, []

    per_atom = []
    max_diff = 0.0
    for (elem_m, (xm, ym, zm)), (elem_r, (xr, yr, zr)) in zip(geom_molzoo, geom_ref):
        dx = abs(xm - xr)
        dy = abs(ym - yr)
        dz = abs(zm - zr)
        atom_max = max(dx, dy, dz)
        per_atom.append({"element": elem_m, "dx": dx, "dy": dy, "dz": dz, "max": atom_max})
        max_diff = max(max_diff, atom_max)

    return max_diff, per_atom


def _compare_hamiltonians(new_h, ref_ham):
    """Compare two Hamiltonian dicts using a principled layered approach.

    Layer 1: Hilbert-Schmidt fidelity F = (sum ai*bi)^2 / (sum ai^2 * sum bi^2)
      - F = 1.0 means operators identical up to global sign.

    Layer 2: Magnitude comparison max ||ai| - |bi|| for orbital phase diagnosis.
      - If magnitudes match but signs differ, it's an orbital phase convention.

    Layer 3: Raw coefficient comparison max |ai - bi|.

    For Hermitian Hamiltonians, coefficients should be real; we use Re(coeff).

    Returns dict with all metrics.
    """
    all_keys = sorted(set(new_h.keys()) | set(ref_ham.keys()))
    if not all_keys:
        return {
            "ham_hs_fidelity": 1.0,
            "ham_raw_overlap": 1.0,
            "ham_max_coeff_diff": 0.0,
            "ham_max_abs_coeff_diff": 0.0,
            "ham_mean_abs_coeff_diff": 0.0,
            "n_mismatched_keys": 0,
        }

    # Extract real parts (Hermitian -> real coefficients in Pauli basis)
    new_vals = np.array([_to_complex(new_h.get(k, 0)).real for k in all_keys])
    ref_vals = np.array([_to_complex(ref_ham.get(k, 0)).real for k in all_keys])

    # Layer 1: Hilbert-Schmidt fidelity
    dot = np.dot(new_vals, ref_vals)
    norm_new = np.dot(new_vals, new_vals)
    norm_ref = np.dot(ref_vals, ref_vals)
    denom = norm_new * norm_ref
    if denom > 0:
        hs_fidelity = dot**2 / denom
        raw_overlap = dot / np.sqrt(denom)
    else:
        hs_fidelity = 1.0 if (norm_new == 0 and norm_ref == 0) else 0.0
        raw_overlap = 1.0 if (norm_new == 0 and norm_ref == 0) else 0.0

    # Layer 2: Magnitude comparison (for orbital phase diagnosis)
    abs_new = np.abs(new_vals)
    abs_ref = np.abs(ref_vals)
    abs_diffs = np.abs(abs_new - abs_ref)
    max_abs_coeff_diff = float(np.max(abs_diffs))
    mean_abs_coeff_diff = float(np.mean(abs_diffs))

    # Layer 3: Raw coefficient comparison
    raw_diffs = np.abs(new_vals - ref_vals)
    max_coeff_diff = float(np.max(raw_diffs))

    # Count mismatched keys
    mismatched = sum(
        1 for k in all_keys
        if (k not in new_h) or (k not in ref_ham)
    )

    return {
        "ham_hs_fidelity": float(hs_fidelity),
        "ham_raw_overlap": float(raw_overlap),
        "ham_max_coeff_diff": max_coeff_diff,
        "ham_max_abs_coeff_diff": max_abs_coeff_diff,
        "ham_mean_abs_coeff_diff": mean_abs_coeff_diff,
        "n_mismatched_keys": mismatched,
    }


# ---------------------------------------------------------------------------
# Per-molecule processing
# ---------------------------------------------------------------------------

def process_molecule(mol_entry, generated_dir):
    """Run one molecule and return a detailed comparison record."""
    import molzoo

    molzoo_id, json_filename, basis, tier = mol_entry
    record = {
        "molzoo_id": molzoo_id,
        "json_filename": json_filename,
        "basis": basis,
        "tier": tier,
    }

    # Load reference (for comparison ONLY — never as computation input)
    try:
        ref = _load_ref(json_filename)
    except FileNotFoundError:
        record["status"] = "SKIP"
        record["notes"] = f"Reference file not found: {json_filename}"
        return record

    ref_data = ref["data"]
    ref_ham = ref.get("hamiltonian", {})

    # Get molzoo molecule (authoritative source for ALL computation inputs)
    mols = {m.id: m for m in molzoo.load_source("symmer")}
    mol = mols.get(molzoo_id)
    if mol is None:
        record["status"] = "SKIP"
        record["notes"] = f"molzoo_id '{molzoo_id}' not found in molzoo"
        return record

    # All computation inputs from molzoo
    geometry = _parse_xyz_string(mol.xyz)
    charge = mol.charge
    multiplicity = mol.multiplicity
    point_group = mol.point_group

    # Map point group to PySCF-compatible Abelian subgroup
    pyscf_subgroup = _to_pyscf_subgroup(point_group)

    record["charge"] = charge
    record["multiplicity"] = multiplicity
    record["point_group_input"] = point_group
    record["symmetry_subgroup_used"] = pyscf_subgroup

    # Geometry precision analysis: compare molzoo vs reference JSON
    try:
        ref_geometry = _parse_ref_geometry(ref_data["geometry"])
        geom_max_diff, _geom_per_atom = _compare_geometries(geometry, ref_geometry)
        record["geom_max_coord_diff"] = geom_max_diff
    except Exception as e:
        record["geom_max_coord_diff"] = None
        record["geom_parse_error"] = str(e)

    # Run generation with all molzoo inputs
    t0 = time.perf_counter()
    try:
        _mol_info, new_data = generate_symmer_data(
            geometry=geometry,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            symmetry_subgroup=pyscf_subgroup,
        )
    except Exception as e:
        record["status"] = "FAIL"
        record["notes"] = f"generate_symmer_data raised {type(e).__name__}: {e}"
        record["elapsed_seconds"] = time.perf_counter() - t0
        return record
    elapsed = time.perf_counter() - t0
    record["elapsed_seconds"] = elapsed

    # Basic metadata
    record["n_qubits"] = new_data["n_qubits"]
    record["n_electrons"] = new_data["n_particles"]["total"]
    record["point_group_detected"] = new_data.get("point_group", {}).get("topgroup", "?")

    # Save generated Hamiltonian JSON
    save_path = os.path.join(generated_dir, f"{molzoo_id}.json")
    serialisable_data = dict(new_data)
    serialisable_data["H"] = {k: _to_serialisable(v) for k, v in new_data["H"].items()}
    with open(save_path, "w") as f:
        json.dump(serialisable_data, f, indent=2, default=str)

    # Energy comparisons
    ref_props = ref_data.get("calculated_properties", {})
    new_props = new_data.get("calculated_properties", {})
    for method in ("HF", "MP2", "CCSD", "FCI"):
        record[method] = _energy_record(
            _safe_energy(ref_props, method),
            _safe_energy(new_props, method),
        )
        record[f"{method.lower()}_converged"] = _safe_converged(new_props, method)

    # Hamiltonian comparison (principled layered approach)
    new_h = new_data["H"]
    record["n_terms_ref"] = len(ref_ham)
    record["n_terms_new"] = len(new_h)
    record["n_terms_match"] = len(ref_ham) == len(new_h)

    ham_metrics = _compare_hamiltonians(new_h, ref_ham)
    record.update(ham_metrics)

    # Determine verdict using layered logic
    status = "PASS"
    notes_parts = []

    if molzoo_id in _KNOWN_FCI_ISSUES:
        notes_parts.append("Known FCI convergence issue")

    # Annotate missing data
    for method in ("HF", "MP2", "CCSD", "FCI"):
        e = record[method]
        if e["ref"] is None:
            notes_parts.append(f"{method}: no reference energy in symmer JSON")
        elif e["new"] is None:
            notes_parts.append(f"{method}: computation produced no energy (nan or absent)")

    # Check energies
    energies_match = True
    for method in ("HF", "MP2", "CCSD", "FCI"):
        diff = record[method].get("diff")
        if diff is not None and abs(diff) >= ENERGY_TOL:
            if molzoo_id in _KNOWN_FCI_ISSUES and method in ("FCI", "MP2"):
                status = max(status, "WARN", key=["PASS", "WARN", "FAIL"].index)
                notes_parts.append(f"{method} diff={diff:.2e} (known issue)")
            else:
                energies_match = False
                status = "FAIL"
                notes_parts.append(f"{method} diff={diff:.2e} exceeds {ENERGY_TOL}")

    # Check Hamiltonian — layered verdict
    F = ham_metrics["ham_hs_fidelity"]
    max_abs_diff = ham_metrics["ham_max_abs_coeff_diff"]

    if 1.0 - F < FIDELITY_TOL:
        # Layer 1 PASS: operators identical up to global sign
        pass  # status stays as whatever energy checks set it to
    elif max_abs_diff < COEFF_TOL and energies_match:
        # Layer 2: magnitudes match but signs differ -> orbital phase convention
        # For BH2+: spatial orbital 2 (B 2py bonding MO) has opposite phase
        # in reference vs current PySCF.  This flips 308/1086 Pauli term signs
        # (0.02% of ||H||^2), giving F = (1-2f)^2 = 0.9992.
        # See tests/investigate_warnings.py for the full analysis.
        status = max(status, "WARN", key=["PASS", "WARN", "FAIL"].index)
        notes_parts.append(
            f"Fidelity={F:.8f} (orbital phase convention vs reference; "
            f"Hamiltonians physically equivalent); "
            f"max||a|-|b||={max_abs_diff:.2e}"
        )
    else:
        # Genuine difference
        status = "FAIL"
        notes_parts.append(
            f"Fidelity={F:.8f}; max|coeff_diff|={ham_metrics['ham_max_coeff_diff']:.2e}; "
            f"max||a|-|b||={max_abs_diff:.2e}"
        )

    if not record["n_terms_match"]:
        if 1.0 - F < FIDELITY_TOL and energies_match:
            # Term count differs but fidelity ~1.0 and energies match:
            # extra/missing terms are below numerical noise threshold
            status = max(status, "WARN", key=["PASS", "WARN", "FAIL"].index)
            notes_parts.append(
                f"term count mismatch: new={record['n_terms_new']} "
                f"ref={record['n_terms_ref']} (fidelity ~1.0, numerical noise)"
            )
        else:
            status = "FAIL"
            notes_parts.append(
                f"term count mismatch: new={record['n_terms_new']} "
                f"ref={record['n_terms_ref']}"
            )

    record["status"] = status
    record["notes"] = "; ".join(notes_parts) if notes_parts else ""
    return record


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_plots(records, figures_dir):
    """Create all matplotlib figures and save to figures_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter to records that have data
    valid = [r for r in records if "n_qubits" in r]
    if not valid:
        return

    # ---- 1. Energy differences bar chart ----
    methods = ["HF", "MP2", "FCI"]
    labels = [r["molzoo_id"] for r in valid]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 6))
    for i, method in enumerate(methods):
        diffs = []
        for r in valid:
            d = r.get(method, {}).get("diff")
            diffs.append(abs(d) if d is not None else 0.0)
        ax.bar(x + i * width, diffs, width, label=method, alpha=0.8)
    ax.axhline(y=ENERGY_TOL, color="r", linestyle="--", linewidth=1, label=f"tol={ENERGY_TOL}")
    ax.set_yscale("log")
    ax.set_ylabel("|ΔE| (Ha)")
    ax.set_title("Energy Differences by Molecule and Method")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "energy_differences.png"), dpi=150)
    plt.close(fig)

    # ---- 3. Hilbert-Schmidt fidelity deviation per molecule ----
    labels_f = [r["molzoo_id"] for r in valid if "ham_hs_fidelity" in r]
    fidelities = [r["ham_hs_fidelity"] for r in valid if "ham_hs_fidelity" in r]

    if labels_f:
        fig, ax = plt.subplots(figsize=(max(10, len(labels_f) * 0.5), 5))
        colors = ["green" if f > 1 - FIDELITY_TOL else "orange" if f > 0.99 else "red"
                  for f in fidelities]
        # Plot 1-F on log scale for resolution near F=1
        one_minus_f = [max(1 - f, 1e-16) for f in fidelities]
        ax.bar(range(len(labels_f)), one_minus_f, color=colors, alpha=0.8)
        ax.axhline(y=FIDELITY_TOL, color="r", linestyle="--", linewidth=1,
                   label=f"tol={FIDELITY_TOL}")
        ax.set_yscale("log")
        ax.set_ylabel("1 - F (Hilbert-Schmidt)")
        ax.set_title("Hamiltonian Fidelity Deviation (lower = better)")
        ax.set_xticks(range(len(labels_f)))
        ax.set_xticklabels(labels_f, rotation=90, fontsize=7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "hamiltonian_fidelity.png"), dpi=150)
        plt.close(fig)

    # ---- 4. Hamiltonian max coeff diff ----
    labels_h = [r["molzoo_id"] for r in valid if "ham_max_coeff_diff" in r]
    max_diffs = [r["ham_max_coeff_diff"] for r in valid if "ham_max_coeff_diff" in r]

    if labels_h:
        fig, ax = plt.subplots(figsize=(max(10, len(labels_h) * 0.5), 5))
        colors = ["green" if d < COEFF_TOL else "red" for d in max_diffs]
        ax.bar(range(len(labels_h)), max_diffs, color=colors, alpha=0.8)
        ax.axhline(y=COEFF_TOL, color="r", linestyle="--", linewidth=1, label=f"tol={COEFF_TOL}")
        ax.set_yscale("log")
        ax.set_ylabel("max |Δcoeff|")
        ax.set_title("Hamiltonian Max Coefficient Difference (Raw)")
        ax.set_xticks(range(len(labels_h)))
        ax.set_xticklabels(labels_h, rotation=90, fontsize=7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "hamiltonian_max_diff.png"), dpi=150)
        plt.close(fig)

    # ---- 5. Timing vs qubits ----
    nq = [r["n_qubits"] for r in valid if "elapsed_seconds" in r]
    times = [r["elapsed_seconds"] for r in valid if "elapsed_seconds" in r]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(nq, times, s=40, alpha=0.8, edgecolors="k", linewidths=0.5)
    for r in valid:
        if "elapsed_seconds" in r:
            ax.annotate(r["molzoo_id"], (r["n_qubits"], r["elapsed_seconds"]),
                        fontsize=5, alpha=0.6, rotation=30)
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Generation time (s)")
    ax.set_title("Hamiltonian Generation Time vs System Size")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "timing_vs_qubits.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _fmt_energy_diff(diff):
    """Format an energy difference for the summary table."""
    if diff is None:
        return "---"
    return f"{diff:.2e}"


def _fmt_float(val, fmt=".2e"):
    """Format a float or return dash."""
    if val is None:
        return "---"
    return f"{val:{fmt}}"


def _status_icon(status):
    """Return a color-coded status string for markdown."""
    icons = {"PASS": "\U0001f7e2", "WARN": "\U0001f7e1", "FAIL": "\U0001f534", "SKIP": "\u26aa"}
    icon = icons.get(status, "")
    return f"{icon} **{status}**"


def _md_anchor(heading_text):
    """Build a GitHub-flavored markdown anchor from a heading string.

    GitHub lowercases, strips non-alphanumeric chars (except hyphens,
    underscores, spaces), and converts spaces to hyphens.
    """
    anchor = heading_text.lower()
    anchor = "".join(c for c in anchor if c.isalnum() or c in (" ", "-", "_"))
    return anchor.replace(" ", "-")


def _get_version(module_name):
    """Safely get version of an installed package."""
    try:
        import importlib.metadata
        return importlib.metadata.version(module_name)
    except Exception:
        try:
            mod = __import__(module_name)
            return getattr(mod, "__version__", "?")
        except Exception:
            return "?"


def write_report(records, output_dir, tiers_run):
    """Write the full markdown report to output_dir/report.md."""
    report_path = os.path.join(output_dir, "report.md")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    valid = [r for r in records if "n_qubits" in r]
    n_pass = sum(1 for r in records if r.get("status") == "PASS")
    n_warn = sum(1 for r in records if r.get("status") == "WARN")
    n_fail = sum(1 for r in records if r.get("status") == "FAIL")
    n_skip = sum(1 for r in records if r.get("status") == "SKIP")

    lines = []
    w = lines.append  # shorthand

    # --- Header ---
    w("# Validation Lab Report: symmerpyscf vs symmer Reference Hamiltonians\n")
    w(f"**Generated:** {now}  ")
    w(f"**Tiers:** {', '.join(str(t) for t in sorted(tiers_run))}  ")
    w(f"**Molecules tested:** {len(records)}  ")
    w(f"**Results:** {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL, {n_skip} SKIP\n")

    # Environment
    w("## Environment\n")
    w(f"- Python: {sys.version.split()[0]}")
    w(f"- Platform: {platform.platform()}")
    w(f"- PySCF: {_get_version('pyscf')}")
    w(f"- symmer: {_get_version('symmer')}")
    w(f"- openfermion: {_get_version('openfermion')}")
    w(f"- symmerpyscf: {_get_version('symmerpyscf')}")
    w(f"- numpy: {np.__version__}")
    w("")

    # --- Purpose ---
    w("## Purpose\n")
    w("This report validates that `symmerpyscf.generate_symmer_data()` reproduces")
    w("the original symmer reference Hamiltonians from the Quantum Molecule Zoo.")
    w("It provides transparent, human-auditable numerical comparisons for every")
    w("molecule, so that discrepancies can be identified and investigated.\n")

    # --- Methodology ---
    w("## Methodology\n")

    w("### Data Flow\n")
    w("All computation inputs come from **molzoo** (the authoritative molecule database).")
    w("The symmer reference JSON is used **only for comparison**, never as computation input.\n")
    w("```")
    w('molzoo (inputs)                    symmer JSON (comparison only)')
    w('---------------------              -----------------------------------')
    w('mol.xyz          -> geometry        ref["hamiltonian"]  -> compare H')
    w('mol.reference_basis -> basis        ref["data"]["calculated_properties"] -> compare energies')
    w('mol.charge       -> charge          ref["data"]["n_qubits"] -> compare qubit count')
    w("mol.multiplicity -> multiplicity")
    w("mol.point_group  -> symmetry_subgroup (mapped to Abelian subgroup)")
    w("```\n")

    w("### Why molzoo geometry (not reference JSON)?\n")
    w("molzoo is the authoritative molecule database; the reference JSON is the comparison target.")
    w("Using reference JSON as input would be circular --- testing whether we reproduce X by using X")
    w("as input. The Geometry Precision section below shows the actual coordinate differences")
    w("and their impact.\n")

    w("### Why `symmetry_subgroup` from `mol.point_group`?\n")
    w("PySCF auto-detects symmetry from coordinates, but molzoo's 6-decimal truncation can break")
    w("detection of high-symmetry molecules (e.g., Dooh detected as C1). Passing the known")
    w("point_group ensures PySCF uses the correct symmetry group. This is metadata from molzoo,")
    w("not from the reference JSON.\n")
    w("PySCF only accepts Abelian subgroups (C1, Ci, C2, Cs, C2v, C2h, D2, D2h) plus the")
    w("infinite groups Coov and Dooh (mapped internally to C2v/D2h). For non-Abelian full")
    w("point groups (D3h, D4h, Td, Kh, etc.), we pass `None` and let PySCF auto-detect")
    w("the largest Abelian subgroup from the geometry.\n")

    w("### Why `multiplicity` from molzoo?\n")
    w("Determines spin state (`spin = multiplicity - 1`) for PySCF's UHF/ROHF method selection.")
    w("Must match the physical system being studied.\n")

    w("### Hamiltonian Comparison: Hilbert-Schmidt Fidelity\n")
    w("For two Hamiltonians H1 = sum(ai * Pi), H2 = sum(bi * Pi) in the Pauli basis:\n")
    w("```")
    w("F = (sum(ai * bi))^2 / (sum(ai^2) * sum(bi^2))")
    w("```\n")
    w("This is the operator analog of |<psi1|psi2>|^2 for states. Pauli operators form an")
    w("orthonormal basis under Tr(Pi * Pj) = 2^n * delta_ij, so the coefficient vectors fully")
    w("characterize the operator and their normalized inner product gives the fidelity.\n")
    w("**Interpretation:**")
    w("- F = 1.0: operators identical up to global sign. **PASS.**")
    w("- F < 1.0 but all ||ai| - |bi|| < tol and energies match: orbital phase convention")
    w("  difference (Hamiltonians related by diagonal unitary). **WARN.**")
    w("- Otherwise: genuine coefficient difference. **FAIL.**\n")
    w("Unlike the naive |coeff| comparison, this correctly distinguishes global phase")
    w("(physically irrelevant) from relative sign flips (physically meaningful).\n")

    w("### Tolerance Justification\n")
    w(f"- **Energy tolerance ({ENERGY_TOL} Ha):** PySCF convergence threshold is set to")
    w("  1e-6 (`pyscf_molecule.conv_tol = 1e-6` in molecule.py:69). Differences at this")
    w("  level are numerical noise from iterative solvers.")
    w(f"- **Coefficient tolerance ({COEFF_TOL}):** One order of magnitude above energy")
    w("  tolerance to account for accumulated numerical error in the integral -> Hamiltonian")
    w("  -> JW pipeline.")
    w(f"- **Fidelity tolerance ({FIDELITY_TOL}):** Several orders below coefficient tolerance;")
    w("  F is a normalized inner product, so numerical noise at 1e-10 in coefficients")
    w("  translates to ~1e-12 in F.\n")

    # Evidence: actual distributions from this run
    energy_diffs_all = []
    coeff_diffs_all = []
    fidelity_devs_all = []
    for r in valid:
        for method in ("HF", "MP2", "CCSD", "FCI"):
            d = r.get(method, {}).get("diff")
            if d is not None:
                energy_diffs_all.append(abs(d))
        if "ham_max_coeff_diff" in r:
            coeff_diffs_all.append(r["ham_max_coeff_diff"])
        if "ham_hs_fidelity" in r:
            fidelity_devs_all.append(1 - r["ham_hs_fidelity"])

    w("### Evidence from This Run\n")
    if energy_diffs_all:
        w(f"- **Energy diffs:** min={min(energy_diffs_all):.2e}, "
          f"max={max(energy_diffs_all):.2e}, "
          f"median={np.median(energy_diffs_all):.2e} (tol={ENERGY_TOL})")
    if coeff_diffs_all:
        w(f"- **Max coeff diffs:** min={min(coeff_diffs_all):.2e}, "
          f"max={max(coeff_diffs_all):.2e}, "
          f"median={np.median(coeff_diffs_all):.2e} (tol={COEFF_TOL})")
    if fidelity_devs_all:
        w(f"- **Fidelity deviations (1-F):** min={min(fidelity_devs_all):.2e}, "
          f"max={max(fidelity_devs_all):.2e}, "
          f"median={np.median(fidelity_devs_all):.2e} (tol={FIDELITY_TOL})")
    w("")

    # --- Figures (placed early — readers want visual overview first) ---
    w("## Figures\n")

    w("### Energy Differences by Molecule\n")
    w("![Energy Differences](figures/energy_differences.png)\n")
    w(f"Absolute energy differences per molecule and method. "
      f"Red dashed line = {ENERGY_TOL} Ha tolerance.\n")

    w("### Hamiltonian Fidelity Deviation\n")
    w("![Hamiltonian Fidelity](figures/hamiltonian_fidelity.png)\n")
    w(f"1-F (Hilbert-Schmidt fidelity deviation) per molecule. "
      f"Green = below {FIDELITY_TOL} tolerance.\n")

    w("### Hamiltonian Max Coefficient Difference\n")
    w("![Hamiltonian Max Diff](figures/hamiltonian_max_diff.png)\n")
    w(f"Max \\|Δcoeff\\| per molecule (raw coefficient diff). "
      f"Green = below {COEFF_TOL} tolerance.\n")

    w("### Generation Timing vs System Size\n")
    w("![Timing vs Qubits](figures/timing_vs_qubits.png)\n")
    w("Hamiltonian generation wall-clock time vs number of qubits.\n")

    # --- Geometry Precision ---
    w("## Geometry Precision\n")
    w("Comparison of coordinate precision between molzoo (`mol.xyz`, 6 decimal places)")
    w("and the symmer reference JSON (full precision). This section documents whether")
    w("the precision difference in geometry inputs affects the computed results.\n")

    geom_diffs = [(r["molzoo_id"], r.get("geom_max_coord_diff"))
                  for r in valid if r.get("geom_max_coord_diff") is not None]
    if geom_diffs:
        w("| Molecule | max \\|Δcoord\\| (Angstrom) | Impact on Fidelity | Impact on ΔE_HF |")
        w("|---|---|---|---|")
        for mid, gd in geom_diffs:
            r = next(r for r in valid if r["molzoo_id"] == mid)
            fid = r.get("ham_hs_fidelity")
            hf_diff = r.get("HF", {}).get("diff")
            fid_str = f"{fid:.10f}" if fid is not None else "---"
            hf_str = f"{hf_diff:.2e}" if hf_diff is not None else "---"
            w(f"| {mid} | {gd:.2e} | F={fid_str} | {hf_str} |")
        w("")

        all_gd = [gd for _, gd in geom_diffs]
        w(f"**Summary:** max coordinate difference across all molecules: {max(all_gd):.2e} Angstrom")
        if max(all_gd) < 1e-6:
            w("(< 1e-6 Angstrom --- below PySCF's numerical precision for geometry)\n")
        elif max(all_gd) < 1e-4:
            w("(< 1e-4 Angstrom --- small but potentially detectable in integrals)\n")
        else:
            w("(>= 1e-4 Angstrom --- may cause measurable differences in results)\n")
    else:
        w("No geometry comparison data available.\n")

    # --- Summary Table ---
    w('<a id="summary-table"></a>\n')
    w("## Summary Table\n")
    w("| Molecule | nq | e- | Basis | Fidelity | dE_HF | dE_FCI "
      "| max\\|dcoeff\\| | #terms | time(s) | Status |")
    w("|---|---:|---:|---|---|---|---|---|---:|---:|---|")
    for r in records:
        mid = r["molzoo_id"]
        link = f"[{mid}](#{_md_anchor(mid)})"
        if "n_qubits" not in r:
            w(f"| {link} | --- | --- | {r.get('basis','')} "
              f"| --- | --- | --- | --- | --- | --- | {_status_icon(r['status'])} |")
            continue
        fid = r.get("ham_hs_fidelity")
        fid_str = f"{fid:.8f}" if fid is not None else "---"
        w(
            f"| {link} "
            f"| {r['n_qubits']} "
            f"| {r['n_electrons']} "
            f"| {r['basis']} "
            f"| {fid_str} "
            f"| {_fmt_energy_diff(r['HF'].get('diff'))} "
            f"| {_fmt_energy_diff(r['FCI'].get('diff'))} "
            f"| {_fmt_float(r.get('ham_max_coeff_diff'))} "
            f"| {r.get('n_terms_new', '---')} "
            f"| {r.get('elapsed_seconds', 0):.1f} "
            f"| {_status_icon(r['status'])} |"
        )
    w("")

    # --- Detailed per-molecule ---
    w("## Detailed Results\n")
    for r in records:
        mid = r["molzoo_id"]
        w(f"### {mid}\n")
        w("[↑ Back to Summary Table](#summary-table)\n")
        if r.get("status") == "SKIP":
            w(f"**Skipped:** {r.get('notes', '')}\n")
            continue

        if "n_qubits" not in r:
            w(f"**Failed during generation:** {r.get('notes', '')}\n")
            continue

        # Inputs section (all from molzoo)
        w("**Inputs (all from molzoo):**\n")
        w(f"- **geometry:** `mol.xyz` (6 decimal places)")
        w(f"- **basis:** `mol.reference_basis` = `\"{r['basis']}\"`")
        w(f"- **charge:** `mol.charge` = `{r.get('charge', '?')}`")
        mult = r.get('multiplicity', '?')
        spin = mult - 1 if isinstance(mult, int) else '?'
        w(f"- **multiplicity:** `mol.multiplicity` = `{mult}` (spin = {spin})")
        pg_input = r.get('point_group_input', '?')
        pg_used = r.get('symmetry_subgroup_used', '?')
        if pg_used is None:
            pg_used_str = "None (auto-detect)"
        else:
            pg_used_str = f'"{pg_used}"'
        w(f"- **symmetry_subgroup:** `mol.point_group` = `\"{pg_input}\"` "
          f"-> PySCF: `{pg_used_str}`")
        w("")

        w(f"- **Qubits:** {r['n_qubits']}  **Electrons:** {r['n_electrons']}")
        w(f"- **Point group (detected):** {r.get('point_group_detected', '?')}")
        w(f"- **Geometry precision:** max |Δcoord| = "
          f"{_fmt_float(r.get('geom_max_coord_diff'))} Angstrom")
        w(f"- **Generation time:** {r.get('elapsed_seconds', 0):.2f}s")
        w(f"- **Status:** **{r['status']}**")
        if r.get("notes"):
            w(f"- **Notes:** {r['notes']}")
        w("")

        # Energy table
        w("| Method | Reference (Ha) | New (Ha) | d (Ha) | Verdict |")
        w("|---|---|---|---|---|")
        for method in ("HF", "MP2", "CCSD", "FCI"):
            e = r.get(method, {})
            verdict = _energy_verdict(e, mid, method)
            w(
                f"| {method} "
                f"| {_fmt_float(e.get('ref'), '.10f')} "
                f"| {_fmt_float(e.get('new'), '.10f')} "
                f"| {_fmt_energy_diff(e.get('diff'))} "
                f"| {verdict} |"
            )
        w("")

        # PySCF convergence metadata
        conv_parts = []
        for method in ("HF", "MP2", "CCSD", "FCI"):
            conv = r.get(f"{method.lower()}_converged", "?")
            e = r.get(method, {})
            if e.get("new") is None and conv:
                conv_parts.append(f"{method}=True (energy=nan)")
            else:
                conv_parts.append(f"{method}={conv}")
        w(f"- **PySCF convergence:** {', '.join(conv_parts)}")
        w("")

        # Hamiltonian stats
        w(f"- **Hamiltonian terms:** ref={r.get('n_terms_ref', '?')}, "
          f"new={r.get('n_terms_new', '?')}, match={r.get('n_terms_match', '?')}")
        fid = r.get("ham_hs_fidelity")
        w(f"- **Hilbert-Schmidt fidelity:** "
          f"{f'{fid:.10f}' if fid is not None else '---'}")
        w(f"- **Raw overlap (signed):** "
          f"{_fmt_float(r.get('ham_raw_overlap'), '.10f')}")
        w(f"- **Max |Δcoeff| (raw):** "
          f"{_fmt_float(r.get('ham_max_coeff_diff'))}")
        w(f"- **Max ||a|-|b|| (magnitude):** "
          f"{_fmt_float(r.get('ham_max_abs_coeff_diff'))}")
        w(f"- **Mean ||a|-|b|| (magnitude):** "
          f"{_fmt_float(r.get('ham_mean_abs_coeff_diff'))}")
        w(f"- **Mismatched Pauli keys:** {r.get('n_mismatched_keys', '?')}")
        w("")

    # --- Known Issues ---
    w("## Known Issues\n")
    if _KNOWN_FCI_ISSUES:
        w(f"- **Known FCI/MP2 issues:** {', '.join(sorted(_KNOWN_FCI_ISSUES))}")
    else:
        w("- **Known FCI/MP2 issues:** None (all resolved)")
    w("")
    w("### Resolved: HN (NH) degenerate pi orbitals\n")
    w("Previously, HN returned wrong FCI root and NaN MP2 due to exact orbital")
    w("degeneracy at the HOMO/LUMO boundary (orbitals 3,4 at -0.342 Ha).\n")
    w("**Fix applied in `molecule.py`:**")
    w("- **FCI:** `fix_spin_(ss=0)` constrains the solver to the singlet sector,")
    w("  avoiding the triplet Ms=0 state. Verified via `<S^2>` with automatic retry")
    w("  at higher penalty (shift=1.0) if needed.")
    w("- **MP2:** When MP2 diverges (NaN) due to zero HOMO-LUMO gap from")
    w("  symmetry-adapted SCF, the pipeline re-runs SCF without symmetry to break")
    w("  the degeneracy, then computes MP2.")
    w("- See `tests/investigate_warnings.py` for the original diagnostic analysis.")
    w("")
    w("- **Known orbital phase issue:** BH2+_singlet_C2v_sto3g")
    w("  - **Root cause:** Spatial orbital 2 (B 2py bonding MO, occupied) has")
    w("    opposite phase convention in the reference vs current PySCF. Orbitals")
    w("    2 and 6 share the same irreducible representation in C2v; exactly one")
    w("    is phase-flipped in the reference (indistinguishable from Hamiltonian alone).")
    w("  - **Evidence:** The single-orbital phase flip hypothesis on orbital 2")
    w("    correctly predicts all 308/1086 sign-flipped Pauli terms (1086/1086 correct).")
    w("  - **Fidelity math:** The flipped terms carry f = 0.0206% of ||H||^2.")
    w("    A sign flip reverses the dot-product contribution: overlap = 1 - 2f = 0.99959.")
    w("    F = (1-2f)^2 = 0.99918, matching observation to 5e-10.")
    w("  - **Impact:** None. Magnitudes match (max ||a|-|b|| = 2.6e-6), so the")
    w("    Hamiltonians have identical eigenvalues and are physically equivalent.")
    w("  - See `tests/investigate_warnings.py` for the full orbital-level analysis.")
    w("")

    # --- Appendix ---
    w("## Appendix\n")
    w("### Reproducing This Report\n")
    w("```bash")
    tiers_str = " ".join(str(t) for t in sorted(tiers_run))
    w(f"python tests/generate_validation_report.py --tiers {tiers_str}")
    w("```\n")
    w("### File Locations\n")
    w(f"- **Report:** `{os.path.join(output_dir, 'report.md')}`")
    w(f"- **Generated Hamiltonians:** `{os.path.join(output_dir, 'generated/')}`")
    w(f"- **Figures:** `{os.path.join(output_dir, 'figures/')}`")
    w(f"- **Machine-readable data:** `{os.path.join(output_dir, 'comparison_data.json')}`")
    w(f"- **Reference data:** `{SYMMER_SOURCE_DIR}`")
    w("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate validation lab report comparing symmerpyscf against symmer references."
    )
    parser.add_argument(
        "--tiers", nargs="+", type=int, default=[1],
        help="Which tiers to run (default: 1). E.g. --tiers 1 2 3 4",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(_SCRIPT_DIR, "validation_output"),
        help="Output directory (default: tests/validation_output/)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    generated_dir = os.path.join(output_dir, "generated")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    tiers = set(args.tiers)
    molecules = [m for m in ALL_MOLECULES if m[3] in tiers]

    print(f"=== Validation Report Generator ===")
    print(f"Tiers: {sorted(tiers)}")
    print(f"Molecules: {len(molecules)}")
    print(f"Output: {output_dir}")
    print()

    records = []
    for i, mol_entry in enumerate(molecules, 1):
        molzoo_id = mol_entry[0]
        print(f"[{i}/{len(molecules)}] {molzoo_id} ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            record = process_molecule(mol_entry, generated_dir)
        except Exception as e:
            tb_str = traceback.format_exc()
            record = {
                "molzoo_id": molzoo_id,
                "json_filename": mol_entry[1],
                "basis": mol_entry[2],
                "tier": mol_entry[3],
                "status": "FAIL",
                "notes": f"Unhandled exception: {type(e).__name__}: {e}\n{tb_str}",
                "elapsed_seconds": time.perf_counter() - t0,
            }
        elapsed = record.get("elapsed_seconds", time.perf_counter() - t0)
        print(f"{record.get('status', '?')} ({elapsed:.1f}s)")
        records.append(record)

    # Save machine-readable data
    comparison_path = os.path.join(output_dir, "comparison_data.json")
    with open(comparison_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\nSaved comparison data: {comparison_path}")

    # Generate plots
    print("Generating plots ... ", end="", flush=True)
    try:
        generate_plots(records, figures_dir)
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")

    # Write markdown report
    report_path = write_report(records, output_dir, tiers)
    print(f"Report written: {report_path}")

    # Print summary
    n_pass = sum(1 for r in records if r.get("status") == "PASS")
    n_warn = sum(1 for r in records if r.get("status") == "WARN")
    n_fail = sum(1 for r in records if r.get("status") == "FAIL")
    n_skip = sum(1 for r in records if r.get("status") == "SKIP")
    print(f"\n=== Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL, {n_skip} SKIP ===")

    # Exit with error code if any failures
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
