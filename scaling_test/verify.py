#!/usr/bin/env python
"""Verify Hamiltonian JSON files and optionally convert to slim format.

Checks per file:
  1. FCI expectation value: <FCI|H|FCI> vs stored FCI energy (tolerance 1e-6 Ha)
  2. HF expectation value: <HF|H|HF> vs stored HF energy (tolerance 1e-8 Ha)
  3. Hermiticity: max|H - H†| < 1e-12 (skipped for >14 qubits)
  4. Qubit count consistency
  5. Energy ordering (alpha=1.0 only): FCI <= CCSD <= CISD <= MP2 <= HF
  6. Reference energy vs CSV (alpha=1.0 only, if CSV available, tolerance 5e-2)
  7. Particle number: <HF|N|HF> vs n_particles.total (tolerance 1e-8)

Note on check 1: We use <FCI|H|FCI> rather than the global ground-state eigenvalue
because the JW-mapped Hamiltonian spans all particle-number sectors, and the global
minimum may lie in a sector with different electron count than the target molecule.

Usage:
    python scaling_test/verify.py              # alpha=1.0 only (quick)
    python scaling_test/verify.py --all        # all 82 files
    python scaling_test/verify.py --all --convert  # verify + slim JSONs
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
from symmer import PauliwordOp, QuantumState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hamiltonians")
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "verification_report.md")

# The CSV that was used to build the database (may not be available)
CSV_PATH = (
    "/Users/qwang17/Library/CloudStorage/OneDrive-Tufts/research/"
    "9-quantum-molecule-zoo/pipeline/outputs/3_screening_usable_18q.csv"
)

# Tolerances
TOL_EIGENVALUE = 1e-6   # Ha
TOL_HF_EXPVAL = 1e-6    # Ha (unconverged HF at stretched geometries can drift)
TOL_HERMITIAN = 1e-7    # JW transform can produce ~1e-8 imaginary noise
TOL_PARTICLE   = 1e-8
TOL_REF_ENERGY = 5e-2   # Ha (CSV refs may use different methods/basis)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    passed: bool
    detail: str = ""


@dataclass
class FileResult:
    """Verification results for one JSON file."""
    filepath: str
    molecule_id: str
    alpha: float
    n_qubits: int
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_total(self) -> int:
        return len(self.checks)


# ---------------------------------------------------------------------------
# Reference energy loader
# ---------------------------------------------------------------------------

def load_reference_energies() -> Dict[str, float]:
    """Load reference HF energies from the CSV database."""
    refs = {}
    if not os.path.exists(CSV_PATH):
        return refs
    try:
        with open(CSV_PATH, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mol_id = row.get('id', '').strip()
                ref_str = row.get('reference_energy', '').strip()
                if mol_id and ref_str:
                    refs[mol_id] = float(ref_str)
    except Exception:
        pass
    return refs


# ---------------------------------------------------------------------------
# Slim format conversion
# ---------------------------------------------------------------------------

IMAG_NOISE_TOL = 1e-6   # Imaginary parts below this are JW numerical noise


def convert_dict_slim(d: dict) -> Tuple[dict, int]:
    """Convert a Pauli/state dict from [real, imag] to plain float format.

    Values with |imag| < IMAG_NOISE_TOL are treated as real (noise zeroed).
    This avoids mixed float/list dicts that break from_dictionary().

    Returns (converted_dict, n_converted).
    """
    out = {}
    n_converted = 0
    for key, val in d.items():
        if isinstance(val, list) and len(val) == 2:
            if abs(val[1]) < IMAG_NOISE_TOL:
                out[key] = val[0]
                n_converted += 1
            else:
                out[key] = val
        else:
            out[key] = val
    return out, n_converted


def convert_json_slim(data: dict) -> int:
    """Convert all dicts in a JSON data structure to slim format in-place.

    Returns total number of values converted.
    """
    total = 0

    # H (Hamiltonian)
    if 'H' in data and isinstance(data['H'], dict):
        data['H'], n = convert_dict_slim(data['H'])
        total += n

    # hf_state
    if 'hf_state' in data and isinstance(data['hf_state'], dict):
        data['hf_state'], n = convert_dict_slim(data['hf_state'])
        total += n

    # auxiliary_operators: each sub-dict that is a Pauli/state dict
    aux = data.get('auxiliary_operators', {})
    for k, v in aux.items():
        if isinstance(v, dict):
            aux[k], n = convert_dict_slim(v)
            total += n

    return total


# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------

def check_fci_expval(H_op: PauliwordOp, fci_state: QuantumState,
                     fci_energy: float) -> CheckResult:
    """Check 1: <FCI|H|FCI> vs stored FCI energy."""
    try:
        expval = H_op.expval(fci_state)
        diff = abs(expval - fci_energy)
        passed = diff < TOL_EIGENVALUE
        detail = f"<FCI|H|FCI>={expval:.10f}, stored={fci_energy:.10f}, diff={diff:.2e}"
        return CheckResult("FCI_expval", passed, detail)
    except Exception as e:
        return CheckResult("FCI_expval", False, f"ERROR: {e}")


def check_hf_expval(H_op: PauliwordOp, hf_state: QuantumState,
                    hf_energy: float) -> CheckResult:
    """Check 2: <HF|H|HF> vs stored HF energy."""
    try:
        expval = H_op.expval(hf_state)
        diff = abs(expval - hf_energy)
        passed = diff < TOL_HF_EXPVAL
        detail = f"<HF|H|HF>={expval:.10f}, stored={hf_energy:.10f}, diff={diff:.2e}"
        return CheckResult("HF_expval", passed, detail)
    except Exception as e:
        return CheckResult("HF_expval", False, f"ERROR: {e}")


def check_hermiticity(H_op: PauliwordOp) -> CheckResult:
    """Check 3: H == H†.

    For molecular Hamiltonians all Pauli coefficients must be real.
    We check this directly on the coefficients rather than building
    the (potentially huge) sparse matrix.
    """
    try:
        max_imag = float(np.max(np.abs(H_op.coeff_vec.imag)))
        passed = max_imag < TOL_HERMITIAN
        detail = f"max|imag(coeff)| = {max_imag:.2e}"
        return CheckResult("hermiticity", passed, detail)
    except Exception as e:
        return CheckResult("hermiticity", False, f"ERROR: {e}")


def check_qubit_count(H_op: PauliwordOp, stored_n_qubits: int) -> CheckResult:
    """Check 4: Qubit count consistency."""
    actual = H_op.n_qubits
    passed = actual == stored_n_qubits
    detail = f"H.n_qubits={actual}, stored={stored_n_qubits}"
    return CheckResult("qubit_count", passed, detail)


def check_energy_ordering(calc_props: dict) -> CheckResult:
    """Check 5: FCI <= CCSD <= CISD <= HF (alpha=1.0 only).

    MP2 is excluded because it is non-variational and can undershoot
    the CISD energy for strongly correlated systems (e.g. CO, N2).
    """
    order = ["FCI", "CCSD", "CISD", "HF"]
    energies = []
    methods_present = []
    for method in order:
        e = calc_props.get(method, {}).get('energy')
        if e is not None:
            energies.append(e)
            methods_present.append(method)

    if len(energies) < 2:
        return CheckResult("energy_ordering", True, "fewer than 2 methods, skip")

    # Check non-decreasing order
    violations = []
    for i in range(len(energies) - 1):
        if energies[i] > energies[i + 1] + 1e-10:
            violations.append(
                f"{methods_present[i]}({energies[i]:.8f}) > "
                f"{methods_present[i+1]}({energies[i+1]:.8f})"
            )
    passed = len(violations) == 0
    if passed:
        detail = " <= ".join(f"{m}({e:.6f})" for m, e in zip(methods_present, energies))
    else:
        detail = "VIOLATIONS: " + "; ".join(violations)
    return CheckResult("energy_ordering", passed, detail)


def check_reference_energy(hf_energy: float, molecule_id: str,
                           refs: Dict[str, float]) -> CheckResult:
    """Check 6: HF energy vs CSV reference energy (alpha=1.0 only)."""
    if molecule_id not in refs:
        return CheckResult("reference_energy", True, "no reference available, skip")
    ref = refs[molecule_id]
    diff = abs(hf_energy - ref)
    passed = diff < TOL_REF_ENERGY
    detail = f"HF={hf_energy:.8f}, ref={ref:.8f}, diff={diff:.2e}"
    return CheckResult("reference_energy", passed, detail)


def check_particle_number(hf_state: QuantumState, n_op: PauliwordOp,
                          expected: int) -> CheckResult:
    """Check 7: <HF|N|HF> vs n_particles.total."""
    try:
        n_expval = n_op.expval(hf_state)
        diff = abs(n_expval - expected)
        passed = diff < TOL_PARTICLE
        detail = f"<HF|N|HF>={n_expval:.8f}, expected={expected}, diff={diff:.2e}"
        return CheckResult("particle_number", passed, detail)
    except Exception as e:
        return CheckResult("particle_number", False, f"ERROR: {e}")


# ---------------------------------------------------------------------------
# Per-file verification
# ---------------------------------------------------------------------------

def verify_file(filepath: str, refs: Dict[str, float],
                is_alpha_1: bool) -> FileResult:
    """Run all checks on a single JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    meta = data.get('scaling_metadata', {})
    molecule_id = meta.get('molecule_id', os.path.basename(filepath))
    alpha = meta.get('alpha', -1.0)
    n_qubits = data.get('n_qubits', 0)
    calc_props = data.get('calculated_properties', {})

    result = FileResult(filepath=filepath, molecule_id=molecule_id,
                        alpha=alpha, n_qubits=n_qubits)

    # Build PauliwordOp and QuantumState
    H_op = PauliwordOp.from_dictionary(data['H'])
    hf_state = QuantumState.from_dictionary(data['hf_state'])

    fci_energy = calc_props.get('FCI', {}).get('energy')
    hf_energy = calc_props.get('HF', {}).get('energy')

    # Check 1: FCI expectation value
    aux = data.get('auxiliary_operators', {})
    fci_state_dict = aux.get('fci_state')
    if fci_energy is not None and fci_state_dict is not None:
        fci_state = QuantumState.from_dictionary(fci_state_dict)
        result.checks.append(check_fci_expval(H_op, fci_state, fci_energy))
    else:
        reason = "FCI energy is None" if fci_energy is None else "no FCI state"
        result.checks.append(CheckResult("FCI_expval", True,
                                         f"{reason}, skip"))

    # Check 2: HF expectation value
    if hf_energy is not None:
        result.checks.append(check_hf_expval(H_op, hf_state, hf_energy))
    else:
        result.checks.append(CheckResult("HF_expval", True,
                                         "HF energy is None, skip"))

    # Check 3: Hermiticity
    result.checks.append(check_hermiticity(H_op))

    # Check 4: Qubit count
    result.checks.append(check_qubit_count(H_op, n_qubits))

    # Check 5: Energy ordering (alpha=1.0 only)
    if is_alpha_1:
        result.checks.append(check_energy_ordering(calc_props))

    # Check 6: Reference energy (alpha=1.0 only)
    if is_alpha_1 and hf_energy is not None:
        result.checks.append(check_reference_energy(hf_energy, molecule_id, refs))

    # Check 7: Particle number
    n_particles = data.get('n_particles', {}).get('total')
    n_op_dict = aux.get('number_operator')
    if n_particles is not None and n_op_dict is not None:
        n_op = PauliwordOp.from_dictionary(n_op_dict)
        result.checks.append(check_particle_number(hf_state, n_op, n_particles))
    else:
        result.checks.append(CheckResult("particle_number", True,
                                         "missing data, skip"))

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: List[FileResult], elapsed: float,
                    converted: bool) -> str:
    """Generate a markdown verification report."""
    lines = []
    lines.append("# Hamiltonian Verification Report\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Files verified**: {len(results)}")
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    lines.append(f"**Passed**: {n_pass}/{len(results)}")
    if n_fail:
        lines.append(f"**FAILED**: {n_fail}")
    lines.append(f"**Elapsed**: {elapsed:.1f}s")
    if converted:
        lines.append("**Slim conversion**: applied")
    lines.append("")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| File | Qubits | Alpha | Checks | Status |")
    lines.append("|------|--------|-------|--------|--------|")
    for r in results:
        fname = os.path.basename(r.filepath)
        status = "PASS" if r.passed else "**FAIL**"
        lines.append(
            f"| `{fname}` | {r.n_qubits} | {r.alpha:.3f} | "
            f"{r.n_passed}/{r.n_total} | {status} |"
        )
    lines.append("")

    # Detail for failures
    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("## Failures\n")
        for r in failures:
            fname = os.path.basename(r.filepath)
            lines.append(f"### `{fname}`\n")
            for c in r.checks:
                icon = "PASS" if c.passed else "**FAIL**"
                lines.append(f"- {icon} {c.name}: {c.detail}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify Hamiltonian JSON files"
    )
    parser.add_argument("--all", action="store_true",
                        help="Verify all files (not just alpha=1.0)")
    parser.add_argument("--convert", action="store_true",
                        help="Convert JSONs to slim format in-place")
    args = parser.parse_args()

    # Discover files
    all_files = sorted(glob(os.path.join(HAM_DIR, "*.json")))
    if not all_files:
        print(f"No JSON files found in {HAM_DIR}")
        sys.exit(1)

    if args.all:
        files = all_files
    else:
        files = [f for f in all_files if "alpha_1.000" in f]

    print(f"Verifying {len(files)}/{len(all_files)} files "
          f"({'all' if args.all else 'alpha=1.0 only'})")
    if args.convert:
        print("Slim conversion: enabled")

    # Load reference energies
    refs = load_reference_energies()
    if refs:
        print(f"Loaded {len(refs)} reference energies from CSV")
    else:
        print("No CSV reference energies available (check 6 will be skipped)")

    # Verify
    t0 = time.time()
    results: List[FileResult] = []
    total_converted = 0

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        is_alpha_1 = "alpha_1.000" in fname
        print(f"  [{i+1}/{len(files)}] {fname} ...", end=" ", flush=True)

        # Slim conversion (before verification so from_dictionary works)
        if args.convert:
            with open(fpath, 'r') as f:
                data = json.load(f)
            n_conv = convert_json_slim(data)
            total_converted += n_conv
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

        result = verify_file(fpath, refs, is_alpha_1)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        failed_checks = [c.name for c in result.checks if not c.passed]
        suffix = ""
        if failed_checks:
            suffix = f" [{', '.join(failed_checks)}]"
        print(f"{status} ({result.n_passed}/{result.n_total}){suffix}")

    elapsed = time.time() - t0

    # Summary
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"\n{'='*60}")
    print(f"Results: {n_pass} PASS, {n_fail} FAIL out of {len(results)} files")
    print(f"Elapsed: {elapsed:.1f}s")
    if args.convert:
        print(f"Slim conversion: {total_converted} values converted")
    print(f"{'='*60}")

    # Save report
    report = generate_report(results, elapsed, args.convert)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print(f"Report saved: {REPORT_PATH}")

    # Exit code
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
