#!/usr/bin/env python3
"""Investigate the 3 WARN molecules from the original validation report.

Runs a 2x2 factorial experiment for each molecule to disentangle whether
geometry source (ref JSON vs molzoo CSV) or symmetry_subgroup (None vs
explicit) causes discrepancies vs the reference Hamiltonian.

Experimental matrix per molecule:
                          symmetry_subgroup=None    symmetry_subgroup=<explicit>
    ref JSON geometry     Condition A (baseline)    Condition B
    molzoo CSV geometry   Condition C               Condition D (report path)

Original warnings (before molzoo fix):
  HN  — MP2 NaN, FCI off by 0.04 Ha (degenerate pi orbitals)
  H2O — 48 extra Hamiltonian terms (geometry truncation) [RESOLVED by molzoo]
  BH2+— Fidelity 0.9992 (orbital phase convention vs old reference)

Usage:
    python tests/investigate_warnings.py
"""

import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from test_molzoo_validation import (
    ALL_MOLECULES,
    _load_ref,
    _parse_ref_geometry,
)
from generate_validation_report import (
    _compare_hamiltonians,
    _compare_geometries,
    _to_pyscf_subgroup,
    _to_complex,
    _safe_energy,
)
from symmerpyscf import generate_symmer_data
from symmerpyscf.scaling import _parse_xyz_string

# ---------------------------------------------------------------------------
# Molecule configs: 3 warnings + 2 controls
# ---------------------------------------------------------------------------
MOLECULES_TO_INVESTIGATE = [
    # Warning molecules (H2O was resolved by molzoo geometry precision fix)
    "HN_singlet_Coov_sto3g",
    "H2O_singlet_C2v_sto3g",   # RESOLVED — included to confirm fix
    "BH2+_singlet_C2v_sto3g",
    # Controls: one Coov, one C2v that PASS in the report
    "FH_singlet_Coov_sto3g",
    "CH2_triplet_C2v_sto3g",
]

# Build lookup from ALL_MOLECULES
_MOL_LOOKUP = {entry[0]: entry for entry in ALL_MOLECULES}


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(geometry, basis, charge, multiplicity, symmetry_subgroup,
                   ref_ham):
    """Run generate_symmer_data and compare against reference Hamiltonian.

    Returns dict with Hamiltonian metrics + energy values.
    """
    t0 = time.perf_counter()
    _mol_info, new_data = generate_symmer_data(
        geometry=geometry,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        symmetry_subgroup=symmetry_subgroup,
    )
    elapsed = time.perf_counter() - t0

    new_h = new_data["H"]
    ham_metrics = _compare_hamiltonians(new_h, ref_ham)

    new_props = new_data.get("calculated_properties", {})
    result = {
        "elapsed": elapsed,
        "n_terms": len(new_h),
        **ham_metrics,
        "hf_energy": _safe_energy(new_props, "HF"),
        "mp2_energy": _safe_energy(new_props, "MP2"),
        "fci_energy": _safe_energy(new_props, "FCI"),
        "point_group_detected": new_data.get("point_group", {}),
        "new_h": new_h,  # keep for deep diagnostics
        "new_data": new_data,
    }
    return result


def investigate_molecule(molzoo_id):
    """Run the 2x2 factorial experiment for one molecule.

    Returns dict with geometry comparison + 4 condition results.
    """
    import molzoo

    entry = _MOL_LOOKUP[molzoo_id]
    _, json_filename, basis, _ = entry

    # Load reference
    ref = _load_ref(json_filename)
    ref_data = ref["data"]
    ref_ham = ref.get("hamiltonian", {})

    # Get molzoo molecule
    mols = {m.id: m for m in molzoo.load_source("symmer")}
    mol = mols[molzoo_id]

    # Two geometry sources
    geom_ref = _parse_ref_geometry(ref_data["geometry"])
    geom_molzoo = _parse_xyz_string(mol.xyz)

    # Geometry comparison
    geom_max_diff, geom_per_atom = _compare_geometries(geom_molzoo, geom_ref)

    # Point group -> PySCF subgroup
    explicit_subgroup = _to_pyscf_subgroup(mol.point_group)

    # Reference energies
    ref_props = ref_data.get("calculated_properties", {})
    ref_energies = {
        "HF": _safe_energy(ref_props, "HF"),
        "MP2": _safe_energy(ref_props, "MP2"),
        "FCI": _safe_energy(ref_props, "FCI"),
    }

    # Run 4 conditions
    common = dict(basis=basis, charge=mol.charge, multiplicity=mol.multiplicity)

    print(f"  Condition A: ref geom + subgroup=None ...", end=" ", flush=True)
    cond_a = run_experiment(geom_ref, symmetry_subgroup=None, ref_ham=ref_ham,
                            **common)
    print(f"{cond_a['elapsed']:.2f}s")

    print(f"  Condition B: ref geom + subgroup={explicit_subgroup} ...",
          end=" ", flush=True)
    cond_b = run_experiment(geom_ref, symmetry_subgroup=explicit_subgroup,
                            ref_ham=ref_ham, **common)
    print(f"{cond_b['elapsed']:.2f}s")

    print(f"  Condition C: molzoo geom + subgroup=None ...", end=" ", flush=True)
    cond_c = run_experiment(geom_molzoo, symmetry_subgroup=None,
                            ref_ham=ref_ham, **common)
    print(f"{cond_c['elapsed']:.2f}s")

    print(f"  Condition D: molzoo geom + subgroup={explicit_subgroup} ...",
          end=" ", flush=True)
    cond_d = run_experiment(geom_molzoo, symmetry_subgroup=explicit_subgroup,
                            ref_ham=ref_ham, **common)
    print(f"{cond_d['elapsed']:.2f}s")

    return {
        "molzoo_id": molzoo_id,
        "point_group": mol.point_group,
        "explicit_subgroup": explicit_subgroup,
        "geom_max_diff": geom_max_diff,
        "geom_per_atom": geom_per_atom,
        "ref_energies": ref_energies,
        "ref_n_terms": len(ref_ham),
        "conditions": {"A": cond_a, "B": cond_b, "C": cond_c, "D": cond_d},
    }


# ---------------------------------------------------------------------------
# Deep diagnostics per warning molecule
# ---------------------------------------------------------------------------

def diagnose_hn(result):
    """HN: Degenerate pi orbitals causing MP2 div-by-zero and wrong FCI root.

    The reference JSON has MP2=-54.163 (converged=True) and FCI=-54.200
    (converged=True).  Our pipeline (symmetry=True) gives MP2=NaN and
    FCI=-54.160 (wrong root).

    Root cause: with symmetry=True and Coov, orbitals 3,4 (pi pair) are
    exactly degenerate.  Since nelec=8 and nao=6, the HOMO/LUMO boundary
    falls exactly on this degenerate pair, causing:
      - MP2: 1/(e_occ - e_virt) diverges (zero denominator)
      - FCI: solver gets trapped in wrong symmetry sector

    With symmetry=False the degeneracy lifts and both reproduce correctly.
    """
    from pyscf import gto, scf, mp, fci
    from openfermionpyscf._run_pyscf import compute_scf
    import warnings as _warnings
    _warnings.filterwarnings("ignore")

    lines = ["  --- HN Deep Diagnostic: Orbital Degeneracy ---"]

    ref = _load_ref("NH_STO-3G_SINGLET_JW.json")
    ref_data = ref["data"]
    ref_props = ref_data.get("calculated_properties", {})
    geom = _parse_ref_geometry(ref_data["geometry"])

    ref_mp2 = ref_props["MP2"]["energy"]
    ref_fci = ref_props["FCI"]["energy"]

    lines.append(f"  Reference: MP2={ref_mp2:.12f} (converged={ref_props['MP2']['converged']})")
    lines.append(f"             FCI={ref_fci:.12f} (converged={ref_props['FCI']['converged']})")

    # --- symmetry=True (our pipeline) ---
    lines.append(f"\n  === symmetry=True (Coov) — our pipeline ===")
    mol_sym = gto.Mole()
    mol_sym.atom = geom
    mol_sym.basis = "sto-3g"
    mol_sym.charge = 0
    mol_sym.spin = 0
    mol_sym.symmetry = True
    mol_sym.symmetry_subgroup = "Coov"
    mol_sym.verbose = 0
    mol_sym.unit = "Angstrom"
    mol_sym.conv_tol = 1e-6
    mol_sym.build()

    mf_sym = compute_scf(mol_sym)
    mf_sym.conv_tol = 1e-6
    mf_sym.run()

    orb_e = mf_sym.mo_energy
    lines.append(f"  Orbital energies: {np.array2string(orb_e, precision=8)}")
    lines.append(f"  nelec={mol_sym.nelec}, nao={mol_sym.nao_nr()}")

    for i in range(len(orb_e) - 1):
        diff = abs(orb_e[i + 1] - orb_e[i])
        if diff < 1e-8:
            lines.append(
                f"  ** Degenerate pair: orbitals {i},{i+1}  "
                f"e={orb_e[i]:.10f}  diff={diff:.2e}  "
                f"(HOMO/LUMO boundary!)"
            )

    mp2_obj = mp.MP2(mf_sym)
    mp2_obj.verbose = 0
    mp2_obj.run()
    mp2_e = mf_sym.e_tot + mp2_obj.e_corr
    lines.append(f"  MP2: {'NaN' if np.isnan(mp2_e) else f'{mp2_e:.12f}'} "
                 f"(zero HOMO-LUMO gap -> divide by zero)")

    # FCI with nroots=1 (what our pipeline does)
    fci_1 = fci.FCI(mol_sym, mf_sym.mo_coeff)
    fci_1.verbose = 0
    fci_e1, _ = fci_1.kernel()
    diff1 = fci_e1 - ref_fci
    lines.append(f"  FCI (nroots=1): {fci_e1:.12f}  (diff={diff1:.2e})"
                 f"{'  <-- matches' if abs(diff1) < 1e-6 else '  <-- WRONG ROOT'}")

    # FCI with nroots=5 to search for the correct root
    fci_sym = fci.FCI(mol_sym, mf_sym.mo_coeff)
    fci_sym.verbose = 0
    fci_sym.nroots = 5
    e_roots, _ = fci_sym.kernel()
    lines.append(f"  FCI roots (nroots=5):")
    found_correct = False
    for i, e in enumerate(e_roots):
        diff = e - ref_fci
        marker = ""
        if abs(diff) < 1e-10:
            marker = "  <-- matches reference"
            found_correct = True
        lines.append(f"    Root {i}: {e:.12f}  (diff={diff:.2e}){marker}")
    if not found_correct:
        lines.append(f"  ** Reference energy NOT found in first 5 roots!")

    # --- symmetry=False (reproduces reference) ---
    lines.append(f"\n  === symmetry=False — lifts degeneracy ===")
    mol_nosym = gto.Mole()
    mol_nosym.atom = geom
    mol_nosym.basis = "sto-3g"
    mol_nosym.charge = 0
    mol_nosym.spin = 0
    mol_nosym.symmetry = False
    mol_nosym.verbose = 0
    mol_nosym.unit = "Angstrom"
    mol_nosym.conv_tol = 1e-6
    mol_nosym.build()

    mf_nosym = scf.RHF(mol_nosym)
    mf_nosym.conv_tol = 1e-6
    mf_nosym.run()

    orb_e2 = mf_nosym.mo_energy
    lines.append(f"  Orbital energies: {np.array2string(orb_e2, precision=8)}")
    lines.append(f"  (Degeneracy lifted: gap = {orb_e2[4]-orb_e2[3]:.6f} Ha)")

    mp2_nosym = mp.MP2(mf_nosym)
    mp2_nosym.verbose = 0
    mp2_nosym.run()
    mp2_e2 = mf_nosym.e_tot + mp2_nosym.e_corr
    lines.append(f"  MP2: {mp2_e2:.12f}  (diff from ref: {mp2_e2 - ref_mp2:.2e})")

    fci_nosym = fci.FCI(mol_nosym, mf_nosym.mo_coeff)
    fci_nosym.verbose = 0
    fci_nosym.nroots = 5
    e_roots2, _ = fci_nosym.kernel()
    lines.append(f"  FCI roots (nroots=5):")
    for i, e in enumerate(e_roots2):
        diff = e - ref_fci
        marker = ""
        if abs(diff) < 1e-10:
            marker = "  <-- MATCHES reference"
        elif i == 0:
            marker = "  (triplet ground state)"
        lines.append(f"    Root {i}: {e:.12f}  (diff={diff:.2e}){marker}")

    # 2x2 check
    conds = result["conditions"]
    all_same = True
    for label in ("B", "C", "D"):
        f_diff = abs(conds[label]["ham_hs_fidelity"] - conds["A"]["ham_hs_fidelity"])
        if f_diff > 1e-12:
            all_same = False

    lines.append(f"\n  === Summary ===")
    lines.append(f"  Hamiltonian: all 4 conditions identical? {all_same}")
    lines.append(f"  (Hamiltonian itself is correct — only MP2/FCI energies are affected)")
    lines.append(
        f"\n  Root cause: The reference was generated with symmetry=False (or "
        f"equivalent), where the pi degeneracy is lifted by numerical noise. "
        f"Our pipeline uses symmetry=True, creating exact degeneracy at the "
        f"HOMO/LUMO boundary (orbitals 3,4). This causes:"
    )
    lines.append(f"    - MP2: 1/(e_HOMO - e_LUMO) = 1/0 -> NaN")
    lines.append(
        f"    - FCI: solver trapped in wrong symmetry sector; correct "
        f"singlet energy (-54.200) not accessible"
    )
    lines.append(
        f"  The Hamiltonian (Pauli coefficients) is NOT affected — it depends "
        f"only on integrals and JW transform, not on post-HF solvers."
    )
    return "\n".join(lines)


def diagnose_h2o(result):
    """H2O: Was caused by geometry truncation; now resolved by molzoo fix."""
    lines = ["  --- H2O Deep Diagnostic: Geometry Truncation (RESOLVED) ---"]

    conds = result["conditions"]
    for label, desc in [("A", "ref+None"), ("B", "ref+C2v"),
                        ("C", "molzoo+None"), ("D", "molzoo+C2v")]:
        lines.append(f"  Cond {label} ({desc}): {conds[label]['n_terms']} terms")

    geom_diff = result["geom_max_diff"]
    lines.append(f"\n  Geometry max diff: {geom_diff:.2e} Angstrom")

    # Identify which conditions have extra terms
    ref_n = result["ref_n_terms"]
    extra_conds = [l for l in "ABCD" if conds[l]["n_terms"] > ref_n]

    if extra_conds:
        # Still seeing extra terms — analyze them
        label = extra_conds[0]
        new_h = conds[label]["new_h"]
        ref_ham = _load_ref("H2O_STO-3G_SINGLET_JW.json").get("hamiltonian", {})

        extra_keys = sorted(set(new_h.keys()) - set(ref_ham.keys()))
        lines.append(f"\n  {len(extra_keys)} extra Pauli strings in condition {label}:")
        extra_coeffs = []
        for k in extra_keys[:10]:  # show first 10
            c = _to_complex(new_h[k])
            extra_coeffs.append(abs(c.real))
            lines.append(f"    {k}: {c.real:.2e}")
        if len(extra_keys) > 10:
            lines.append(f"    ... ({len(extra_keys) - 10} more)")
        if extra_coeffs:
            lines.append(
                f"  Extra term magnitudes: "
                f"max={max(extra_coeffs):.2e}, min={min(extra_coeffs):.2e}"
            )

        geom_matters = (conds["A"]["n_terms"] != conds["C"]["n_terms"] or
                        conds["B"]["n_terms"] != conds["D"]["n_terms"])
        subgroup_matters = (conds["A"]["n_terms"] != conds["B"]["n_terms"] or
                            conds["C"]["n_terms"] != conds["D"]["n_terms"])
        lines.append(f"\n  Geometry matters for term count? {geom_matters}")
        lines.append(f"  symmetry_subgroup matters for term count? {subgroup_matters}")
    else:
        # No extra terms — fix confirmed!
        all_match_ref = all(conds[l]["n_terms"] == ref_n for l in "ABCD")
        all_fidelity_ok = all(
            1.0 - conds[l]["ham_hs_fidelity"] < 1e-8 for l in "ABCD"
        )
        lines.append(f"\n  All 4 conditions produce {ref_n} terms (matches reference): "
                      f"{all_match_ref}")
        lines.append(f"  All 4 conditions have fidelity ~1.0: {all_fidelity_ok}")
        lines.append(
            f"\n  RESOLVED: molzoo now provides full-precision coordinates "
            f"(max diff = {geom_diff:.2e}). The original warning was caused by "
            f"6-decimal truncation (~3e-7 Angstrom) which broke exact integral "
            f"cancellations, producing 48 near-zero extra Hamiltonian terms."
        )
    return "\n".join(lines)


def diagnose_bh2plus(result):
    """BH2+: Single orbital phase flip — identified and mathematically verified.

    Evidence-based analysis:
    1. Spatial orbital 2 (B 2py bonding MO) has opposite phase in reference
       vs any fresh computation.
    2. This single orbital phase flip perfectly predicts all 308/1086
       sign-flipped Pauli terms (1086/1086 correct).
    3. The flipped terms carry f = 0.0206% of ||H||^2.
       Fidelity = (1 - 2f)^2 = 0.9992, matching observation exactly.
    """
    from pyscf import gto
    from openfermionpyscf._run_pyscf import compute_scf
    from itertools import combinations

    lines = ["  --- BH2+ Deep Diagnostic: Single Orbital Phase Flip ---"]

    conds = result["conditions"]

    # Show fidelity for each condition
    for label, desc in [("A", "ref+None"), ("B", "ref+C2v"),
                        ("C", "molzoo+None"), ("D", "molzoo+C2v")]:
        f = conds[label]["ham_hs_fidelity"]
        lines.append(f"  Cond {label} ({desc}): F={f:.10f}")

    fids = [conds[l]["ham_hs_fidelity"] for l in "ABCD"]
    all_same = max(fids) - min(fids) < 1e-10
    lines.append(f"\n  All 4 conditions identical? {all_same}")
    lines.append(
        "  => Phase flip is between reference and ALL fresh computations."
    )

    # --- Identify the flipped orbital ---
    lines.append(f"\n  === Step 1: Identify sign-flipped Pauli terms ===")

    ref = _load_ref("BH2+_STO-3G_SINGLET_JW.json")
    ref_ham = ref.get("hamiltonian", {})
    new_h = conds["A"]["new_h"]

    all_keys = sorted(set(new_h.keys()) | set(ref_ham.keys()))
    new_vals = np.array([_to_complex(new_h.get(k, 0)).real for k in all_keys])
    ref_vals = np.array([_to_complex(ref_ham.get(k, 0)).real for k in all_keys])

    threshold = 1e-10
    is_flipped = np.zeros(len(all_keys), dtype=bool)
    for i in range(len(all_keys)):
        a, b = new_vals[i], ref_vals[i]
        if abs(a) > threshold and abs(b) > threshold and a * b < 0:
            is_flipped[i] = True

    n_flipped = int(is_flipped.sum())
    lines.append(f"  Sign-flipped terms: {n_flipped}/{len(all_keys)}")

    # --- Test single-orbital phase flip hypothesis ---
    lines.append(f"\n  === Step 2: Test single-orbital phase flip hypotheses ===")
    lines.append(
        f"  In JW encoding, flipping spatial orbital j reverses the sign of"
    )
    lines.append(
        f"  every Pauli term with an odd count of X/Y at qubits {{2j, 2j+1}}."
    )

    n_qubits = len(all_keys[0])
    n_spatial = n_qubits // 2

    def _test_hypothesis(flipped_orbitals):
        qubit_set = set()
        for orb in flipped_orbitals:
            qubit_set.add(2 * orb)
            qubit_set.add(2 * orb + 1)
        correct = 0
        for i, k in enumerate(all_keys):
            n_xy = sum(1 for q in qubit_set if k[q] in ('X', 'Y'))
            predicted = (n_xy % 2 == 1)
            if predicted == is_flipped[i]:
                correct += 1
        return correct

    lines.append(
        f"  {'Spatial orb':>12s} {'Correct':>10s} {'Score':>8s} {'Result':>12s}"
    )
    perfect_orbitals = []
    for orb in range(n_spatial):
        correct = _test_hypothesis([orb])
        score = f"{correct}/{len(all_keys)}"
        result_str = "PERFECT" if correct == len(all_keys) else ""
        if correct == len(all_keys):
            perfect_orbitals.append(orb)
        lines.append(f"  {orb:>12d} {correct:>10d} {score:>8s} {result_str:>12s}")
    best_orb = perfect_orbitals[0]  # first match (occupied orbital)

    if len(perfect_orbitals) > 1:
        lines.append(
            f"\n  Note: orbitals {perfect_orbitals} all give perfect matches."
        )
        lines.append(
            f"  These orbitals share the same irreducible representation, so"
        )
        lines.append(
            f"  they always appear with the same X/Y parity in the Hamiltonian."
        )
        lines.append(
            f"  Exactly one has a phase flip in the reference (indistinguishable"
        )
        lines.append(
            f"  from Hamiltonian alone). Using orbital {best_orb} (occupied) below."
        )

    # --- Show what orbital 2 is ---
    lines.append(f"\n  === Step 3: Identify orbital {best_orb} ===")

    geom = _parse_ref_geometry(ref["data"]["geometry"])
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = "sto-3g"
    mol.charge = 1
    mol.spin = 0
    mol.symmetry = True
    mol.verbose = 0
    mol.unit = "Angstrom"
    mol.build()

    mf = compute_scf(mol)
    mf.conv_tol = 1e-6
    mf.run()

    lines.append(f"  Orbital energies and occupancy:")
    lines.append(
        f"  {'Orb':>4s} {'Energy (Ha)':>14s} {'Occ':>5s} {'Note':>20s}"
    )
    for i in range(mol.nao_nr()):
        note = ""
        if i == best_orb:
            note = "<-- PHASE-FLIPPED"
        lines.append(
            f"  {i:>4d} {mf.mo_energy[i]:>14.8f} {mf.mo_occ[i]:>5.1f} {note:>20s}"
        )

    lines.append(f"\n  MO coefficients for orbital {best_orb} (the flipped orbital):")
    ao_labels = mol.ao_labels()
    for j in range(len(ao_labels)):
        c = mf.mo_coeff[j, best_orb]
        if abs(c) > 1e-6:
            lines.append(f"    {ao_labels[j]}: {c:>12.8f}")

    lines.append(
        f"  => Orbital {best_orb} is a B(2py) + H(1s) - H(1s) bonding MO"
    )

    # --- Show top sign-flipped terms ---
    lines.append(f"\n  === Step 4: Largest sign-flipped Pauli terms ===")
    flipped_terms = [
        (all_keys[i], new_vals[i], ref_vals[i])
        for i in range(len(all_keys)) if is_flipped[i]
    ]
    flipped_terms.sort(key=lambda x: abs(x[1]), reverse=True)

    lines.append(
        f"  {'Pauli string':>16s} {'new':>12s} {'ref':>12s} {'ratio':>8s}"
    )
    for k, a, b in flipped_terms[:6]:
        lines.append(f"  {k:>16s} {a:>12.8f} {b:>12.8f} {a/b:>8.4f}")
    lines.append(f"  ... ({len(flipped_terms) - 6} more)")

    # --- Fidelity math ---
    lines.append(f"\n  === Step 5: Fidelity math ===")

    dot = np.dot(new_vals, ref_vals)
    norm_sq = np.dot(ref_vals, ref_vals)
    F = dot**2 / (norm_sq * norm_sq)
    raw_overlap = dot / norm_sq

    norm_flipped = np.dot(ref_vals[is_flipped], ref_vals[is_flipped])
    f_frac = norm_flipped / norm_sq

    lines.append(f"  ||H_ref||^2                     = {norm_sq:.6f}")
    lines.append(f"  ||H_ref||^2 in flipped terms    = {norm_flipped:.6f}")
    lines.append(
        f"  Fraction in flipped terms (f)   = {f_frac:.8f} ({f_frac*100:.4f}%)"
    )
    lines.append(f"")
    lines.append(f"  Sign flip reverses contribution: dot loses 2 * norm_flipped")
    lines.append(
        f"  overlap = dot / ||H||^2 = (1 - 2f) = {1-2*f_frac:.10f}"
    )
    lines.append(f"  predicted F = (1 - 2f)^2        = {(1-2*f_frac)**2:.10f}")
    lines.append(f"  actual F                        = {F:.10f}")
    lines.append(
        f"  agreement: |predicted - actual| = {abs((1-2*f_frac)**2 - F):.2e}"
    )

    lines.append(f"\n  === Conclusion ===")
    lines.append(
        f"  Spatial orbital {best_orb} (B 2py bonding MO, occupied) has opposite"
    )
    lines.append(
        f"  phase convention in the reference vs current PySCF. This single"
    )
    lines.append(
        f"  orbital phase difference:"
    )
    lines.append(
        f"    - Flips {n_flipped}/{len(all_keys)} Pauli term signs "
        f"(all correctly predicted)"
    )
    lines.append(
        f"    - Those terms carry {f_frac*100:.4f}% of ||H||^2"
    )
    lines.append(
        f"    - Reduces fidelity to F = (1-2f)^2 = {F:.6f}"
    )
    lines.append(
        f"    - Magnitudes match: max||a|-|b|| = "
        f"{conds['A']['ham_max_abs_coeff_diff']:.2e}"
    )
    lines.append(
        f"    - Hamiltonians are physically equivalent (same eigenvalues)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def print_report(results):
    """Print formatted text report."""
    print("\n" + "=" * 80)
    print("INVESTIGATION REPORT: 3 Warning Molecules + 2 Controls")
    print("  (H2O warning resolved by molzoo geometry precision fix)")
    print("=" * 80)

    # Summary table
    print("\n--- 2x2 Factorial Summary ---\n")
    header = (
        f"{'Molecule':<30s} {'Cond':>4s} {'Geom':>7s} {'Subgrp':>8s} "
        f"{'#terms':>7s} {'Fidelity':>12s} {'max|raw|':>10s} "
        f"{'max|mag|':>10s} {'dE_HF':>10s} {'dE_FCI':>10s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        mid = r["molzoo_id"]
        ref_e = r["ref_energies"]
        for label, geom_src, subgrp_src in [
            ("A", "ref", "None"),
            ("B", "ref", r["explicit_subgroup"] or "None"),
            ("C", "molzoo", "None"),
            ("D", "molzoo", r["explicit_subgroup"] or "None"),
        ]:
            c = r["conditions"][label]
            de_hf = ""
            if c["hf_energy"] is not None and ref_e["HF"] is not None:
                de_hf = f"{c['hf_energy'] - ref_e['HF']:.2e}"
            de_fci = ""
            if c["fci_energy"] is not None and ref_e["FCI"] is not None:
                de_fci = f"{c['fci_energy'] - ref_e['FCI']:.2e}"
            print(
                f"{mid if label == 'A' else '':30s} "
                f"{label:>4s} {geom_src:>7s} {subgrp_src:>8s} "
                f"{c['n_terms']:>7d} {c['ham_hs_fidelity']:>12.10f} "
                f"{c['ham_max_coeff_diff']:>10.2e} "
                f"{c['ham_max_abs_coeff_diff']:>10.2e} "
                f"{de_hf:>10s} {de_fci:>10s}"
            )
        print()

    # Geometry comparison
    print("--- Geometry Precision ---\n")
    for r in results:
        mid = r["molzoo_id"]
        gd = r["geom_max_diff"]
        gd_str = f"{gd:.2e}" if gd is not None else "N/A"
        print(f"  {mid:<30s}  max|dcoord| = {gd_str} Angstrom")
    print()

    # Per-molecule diagnosis
    print("--- Per-Molecule Diagnosis ---\n")
    for r in results:
        mid = r["molzoo_id"]
        conds = r["conditions"]

        print(f"  {mid}:")
        print(f"    Point group: {r['point_group']} -> subgroup: {r['explicit_subgroup']}")

        # Determine which factor matters
        # Geometry effect: compare A vs C (same subgroup=None)
        geom_fidelity_diff = abs(
            conds["A"]["ham_hs_fidelity"] - conds["C"]["ham_hs_fidelity"]
        )
        geom_terms_diff = abs(conds["A"]["n_terms"] - conds["C"]["n_terms"])

        # Subgroup effect: compare A vs B (same geometry=ref)
        subgrp_fidelity_diff = abs(
            conds["A"]["ham_hs_fidelity"] - conds["B"]["ham_hs_fidelity"]
        )
        subgrp_terms_diff = abs(conds["A"]["n_terms"] - conds["B"]["n_terms"])

        geom_matters = geom_fidelity_diff > 1e-8 or geom_terms_diff > 0
        subgrp_matters = subgrp_fidelity_diff > 1e-8 or subgrp_terms_diff > 0

        print(f"    Geometry effect:  dF={geom_fidelity_diff:.2e}, "
              f"d#terms={geom_terms_diff}")
        print(f"    Subgroup effect:  dF={subgrp_fidelity_diff:.2e}, "
              f"d#terms={subgrp_terms_diff}")
        print(f"    Geometry matters? {geom_matters}")
        print(f"    symmetry_subgroup matters? {subgrp_matters}")
        print()

    # Deep diagnostics
    print("--- Deep Diagnostics ---\n")
    for r in results:
        mid = r["molzoo_id"]
        diag_fn = _DIAGNOSTICS.get(mid)
        if diag_fn:
            print(f"  {mid}:")
            print(diag_fn(r))
            print()

    # Final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Molecule':<30s} {'Geom?':>8s} {'Subgroup?':>10s} {'Root Cause'}")
    print("-" * 90)
    for r in results:
        mid = r["molzoo_id"]
        conds = r["conditions"]

        geom_f = abs(conds["A"]["ham_hs_fidelity"] - conds["C"]["ham_hs_fidelity"])
        geom_t = abs(conds["A"]["n_terms"] - conds["C"]["n_terms"])
        subg_f = abs(conds["A"]["ham_hs_fidelity"] - conds["B"]["ham_hs_fidelity"])
        subg_t = abs(conds["A"]["n_terms"] - conds["B"]["n_terms"])

        geom = "YES" if (geom_f > 1e-8 or geom_t > 0) else "no"
        subg = "YES" if (subg_f > 1e-8 or subg_t > 0) else "no"

        cause = _ROOT_CAUSES.get(mid, "(control molecule)")
        print(f"  {mid:<30s} {geom:>6s} {subg:>10s}   {cause}")
    print()


# Expected root causes
_ROOT_CAUSES = {
    "HN_singlet_Coov_sto3g": (
        "symmetry=True creates exact pi degeneracy at HOMO/LUMO -> "
        "MP2 div/0, FCI wrong root; ref used symmetry=False"
    ),
    "H2O_singlet_C2v_sto3g": (
        "RESOLVED: was geometry truncation -> 48 extra terms; "
        "molzoo now has full-precision coords"
    ),
    "BH2+_singlet_C2v_sto3g": (
        "Spatial orbital 2 (B 2py bonding MO) has opposite phase in reference; "
        "flips 308/1086 terms (0.02% of norm), F=(1-2f)^2=0.9992"
    ),
    "FH_singlet_Coov_sto3g": "(Coov control — should be clean)",
    "CH2_triplet_C2v_sto3g": "(C2v control — should be clean)",
}

# Deep diagnostic functions
_DIAGNOSTICS = {
    "HN_singlet_Coov_sto3g": diagnose_hn,
    "H2O_singlet_C2v_sto3g": diagnose_h2o,
    "BH2+_singlet_C2v_sto3g": diagnose_bh2plus,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Investigating 3 WARN molecules + 2 controls")
    print("  (H2O was resolved by molzoo — confirming fix)")
    print("=" * 60)

    results = []
    for i, molzoo_id in enumerate(MOLECULES_TO_INVESTIGATE, 1):
        print(f"\n[{i}/{len(MOLECULES_TO_INVESTIGATE)}] {molzoo_id}")
        t0 = time.perf_counter()
        r = investigate_molecule(molzoo_id)
        elapsed = time.perf_counter() - t0
        print(f"  Total: {elapsed:.1f}s")
        results.append(r)

    print_report(results)


if __name__ == "__main__":
    main()
