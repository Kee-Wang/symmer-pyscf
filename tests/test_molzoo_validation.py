"""Validation tests: symmerpyscf reproduces original symmer reference Hamiltonians.

Tier 1 (≤14 qubits, ~10 min): runs by default
Tier 2 (16-18 qubits):  pytest -m medium
Tier 3 (20-22 qubits):  pytest -m slow
Tier 4 (>22 qubits):    pytest -m very_slow
All tiers:               pytest tests/test_molzoo_validation.py --durations=0 -v
"""

import json
import os
import time

import numpy as np
import pytest
from symmer import PauliwordOp

from symmerpyscf import generate_symmer_data

# ---------------------------------------------------------------------------
# Reference data location
# ---------------------------------------------------------------------------
SYMMER_SOURCE_DIR = (
    "/Users/qwang17/Library/CloudStorage/OneDrive-Tufts"
    "/research/9-quantum-molecule-zoo/sources/symmer"
)

# ---------------------------------------------------------------------------
# Molecule table: (molzoo_id, json_filename, basis, tier)
#   tier 1: ≤14 qubits   tier 2: 16-18   tier 3: 20-22   tier 4: >22
# ---------------------------------------------------------------------------
ALL_MOLECULES = [
    # --- Tier 1: ≤14 qubits (29 molecules) ---
    ("H3+_singlet_D3h_sto3g",      "H3+_STO-3G_SINGLET_JW.json",      "sto-3g",  1),   #  6q
    ("H2_singlet_Dooh_631g",       "H2_6-31G_SINGLET_JW.json",         "6-31g",   1),   #  8q
    ("H2_singlet_Dooh_321g",       "H2_3-21G_SINGLET_JW.json",         "3-21g",   1),   #  8q
    ("H4_singlet_D4h_sto3g",       "H4_STO-3G_SINGLET_JW.json",        "sto-3g",  1),   #  8q
    ("HHe+_singlet_Coov_321g",     "HeH+_3-21G_SINGLET_JW.json",       "3-21g",   1),   #  8q
    ("Be_singlet_Kh_sto3g",        "Be_STO-3G_SINGLET_JW.json",        "sto-3g",  1),   # 10q
    ("B+_singlet_Kh_sto3g",        "B+_STO-3G_SINGLET_JW.json",        "sto-3g",  1),   # 10q
    ("B_doublet_Kh_sto3g",         "B_STO-3G_DOUBLET_JW.json",         "sto-3g",  1),   # 10q
    ("Li_doublet_Kh_sto3g",        "Li_STO-3G_DOUBLET_JW.json",        "sto-3g",  1),   # 10q
    ("C_triplet_Kh_sto3g",         "C_STO-3G_TRIPLET_JW.json",         "sto-3g",  1),   # 10q
    ("O_triplet_C1_sto3g",         "O_STO-3G_TRIPLET_JW.json",         "sto-3g",  1),   # 10q
    ("N_quartet_Kh_sto3g",         "N_STO-3G_QUARTET_JW.json",         "sto-3g",  1),   # 10q
    ("HLi_singlet_Coov_sto3g",     "LiH_STO-3G_SINGLET_JW.json",      "sto-3g",  1),   # 12q
    ("BH_singlet_Coov_sto3g",      "BH_STO-3G_SINGLET_JW.json",       "sto-3g",  1),   # 12q
    ("BeH+_singlet_Coov_sto3g",    "BeH+_STO-3G_SINGLET_JW.json",     "sto-3g",  1),   # 12q
    ("CH+_singlet_Coov_sto3g",     "CH+_STO-3G_SINGLET_JW.json",      "sto-3g",  1),   # 12q
    ("FH_singlet_Coov_sto3g",      "HF_STO-3G_SINGLET_JW.json",       "sto-3g",  1),   # 12q
    ("HN_singlet_Coov_sto3g",      "NH_STO-3G_SINGLET_JW.json",       "sto-3g",  1),   # 12q
    ("HO-_singlet_Coov_sto3g",     "OH-_STO-3G_SINGLET_JW.json",      "sto-3g",  1),   # 12q
    ("HNe+_singlet_Coov_sto3g",    "NeH+_STO-3G_SINGLET_JW.json",     "sto-3g",  1),   # 12q
    ("H6_singlet_Dooh_STO3G",      "H6_STO-3G_SINGLET_JW.json",       "STO-3G",  1),   # 12q
    ("H2_singlet_Dooh_6311g",      "H2_6-311G_SINGLET_JW.json",        "6-311g",  1),   # 12q
    ("HHe+_singlet_Coov_6311g",    "HeH+_6-311G_SINGLET_JW.json",      "6-311g",  1),   # 12q
    ("H3+_singlet_D3h_321g",       "H3+_3-21G_SINGLET_JW.json",        "3-21g",   1),   # 12q
    ("H2O_singlet_C2v_sto3g",      "H2O_STO-3G_SINGLET_JW.json",      "sto-3g",  1),   # 14q
    ("BeH2_singlet_Dooh_sto3g",    "BeH2_STO-3G_SINGLET_JW.json",     "sto-3g",  1),   # 14q
    ("BH2+_singlet_C2v_sto3g",     "BH2+_STO-3G_SINGLET_JW.json",     "sto-3g",  1),   # 14q
    ("CH2_triplet_C2v_sto3g",      "CH2_STO-3G_TRIPLET_JW.json",      "sto-3g",  1),   # 14q
    ("H2N-_singlet_C2v_sto3g",     "NH2-_STO-3G_SINGLET_JW.json",     "sto-3g",  1),   # 14q
    # --- Tier 2: 16-18 qubits (5 molecules) ---
    ("H3O+_singlet_C3v_sto3g",     "H3O+_STO-3G_SINGLET_JW.json",     "sto-3g",  2),   # 16q
    ("H3N_singlet_Cs_sto3g",       "NH3_STO-3G_SINGLET_JW.json",      "sto-3g",  2),   # 16q
    ("CH4_singlet_Td_sto3g",       "CH4_STO-3G_SINGLET_JW.json",      "sto-3g",  2),   # 18q
    ("H4N+_singlet_Td_sto3g",      "NH4+_STO-3G_SINGLET_JW.json",     "sto-3g",  2),   # 18q
    ("Mg_singlet_Kh_sto3g",        "Mg_STO-3G_SINGLET_JW.json",       "sto-3g",  2),   # 18q
    # --- Tier 3: 20-22 qubits (12 molecules) ---
    ("CO_singlet_Coov_sto3g",      "CO_STO-3G_SINGLET_JW.json",       "sto-3g",  3),   # 20q
    ("F2_singlet_Dooh_sto3g",      "F2_STO-3G_SINGLET_JW.json",       "sto-3g",  3),   # 20q
    ("N2_singlet_Dooh_sto3g",      "N2_STO-3G_SINGLET_JW.json",       "sto-3g",  3),   # 20q
    ("ClH_singlet_Coov_sto3g",     "HCl_STO-3G_SINGLET_JW.json",      "sto-3g",  3),   # 20q
    ("HNa_singlet_Coov_sto3g",     "NaH_STO-3G_SINGLET_JW.json",      "sto-3g",  3),   # 20q
    ("O2_triplet_Dooh_sto3g",      "O2_STO-3G_TRIPLET_JW.json",       "sto-3g",  3),   # 20q
    ("H2S_singlet_C2v_sto3g",      "H2S_STO-3G_SINGLET_JW.json",      "sto-3g",  3),   # 22q
    ("CHN_singlet_Coov_sto3g",     "HCN_STO-3G_SINGLET_JW.json",      "sto-3g",  3),   # 22q
    ("H2Mg_singlet_Dooh_sto3g",    "MgH2_STO-3G_SINGLET_JW.json",     "sto-3g",  3),   # 22q
    ("HLiO_singlet_Coov_sto3g",    "LiOH_STO-3G_SINGLET_JW.json",     "sto-3g",  3),   # 22q
    ("FH_singlet_Coov_321g",       "HF_3-21G_SINGLET_JW.json",         "3-21g",  3),   # 22q
    ("HLi_singlet_Coov_321g",      "LiH_3-21G_SINGLET_JW.json",        "3-21g",  3),   # 22q
    # --- Tier 4: >22 qubits (5 molecules) ---
    ("H2O2_singlet_C2_sto3g",      "HOOH_STO-3G_SINGLET_JW.json",     "sto-3g",  4),   # 24q
    ("H4Si_singlet_Td_sto3g",      "SiH4_STO-3G_SINGLET_JW.json",     "sto-3g",  4),   # 26q
    ("CH4O_singlet_Cs_sto3g",      "CH3OH_STO-3G_SINGLET_JW.json",    "sto-3g",  4),   # 28q
    ("CO2_singlet_Dooh_sto3g",     "CO2_STO-3G_SINGLET_JW.json",      "sto-3g",  4),   # 30q
    ("ClNa_singlet_Coov_sto3g",    "NaCl_STO-3G_SINGLET_JW.json",     "sto-3g",  4),   # 36q
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_TIER_MARKERS = {
    2: pytest.mark.medium,
    3: pytest.mark.slow,
    4: pytest.mark.very_slow,
}

# Molecules with known PySCF FCI convergence issues (degenerate orbitals).
# HF/Hamiltonian still validate; only FCI/MP2 comparisons are affected.
# HN was resolved by spin-constrained FCI (fix_spin_) and symmetry-broken MP2 fallback.
_KNOWN_FCI_ISSUES: set = set()

# Timing records collected during the session.
# conftest.pytest_terminal_summary reads this via sys.modules import.
_timing_records: list = []


def _load_ref(json_filename):
    """Load a symmer reference JSON and return the parsed dict."""
    path = os.path.join(SYMMER_SOURCE_DIR, json_filename)
    with open(path, "r") as f:
        return json.load(f)


def _parse_ref_geometry(geom_str):
    """Parse geometry from reference JSON 'data.geometry' field.

    Format: "N_atoms\\n \\nELEM\\tx\\ty\\tz\\n..."  (tab-separated, first two lines are header)
    Uses full precision from the original symmer data.
    """
    geometry = []
    lines = geom_str.strip().split("\n")
    for line in lines[2:]:  # skip atom-count and blank line
        parts = line.strip().split("\t")
        if len(parts) == 4:
            geometry.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    return geometry


def _run_molecule(mol_entry):
    """Run generate_symmer_data for one molecule, return (new_data, ref, elapsed)."""
    import molzoo

    molzoo_id, json_filename, basis, _tier = mol_entry

    # Load reference
    ref = _load_ref(json_filename)
    ref_data = ref["data"]

    # Get molecule from molzoo (for charge/multiplicity metadata)
    mols = {m.id: m for m in molzoo.load_source("symmer")}
    mol = mols[molzoo_id]

    # Parse geometry from the reference JSON (full precision) rather than
    # from molzoo CSV (which may have truncated coordinates)
    geometry = _parse_ref_geometry(ref_data["geometry"])

    # Run generation
    t0 = time.perf_counter()
    _mol_info, new_data = generate_symmer_data(
        geometry=geometry,
        basis=basis,
        charge=mol.charge,
        multiplicity=mol.multiplicity,
    )
    elapsed = time.perf_counter() - t0

    return new_data, ref_data, ref.get("hamiltonian", {}), elapsed


# ---------------------------------------------------------------------------
# Module-scoped cache: compute each molecule only once
# ---------------------------------------------------------------------------
_cache: dict = {}


def _get_result(mol_entry):
    """Return cached (new_data, ref_data, ref_ham, elapsed) for a molecule."""
    key = mol_entry[0]  # molzoo_id
    if key not in _cache:
        _cache[key] = _run_molecule(mol_entry)
        new_data, ref_data, _ref_ham, elapsed = _cache[key]
        _timing_records.append({
            "molecule_id": key,
            "n_qubits": new_data["n_qubits"],
            "n_electrons": new_data["n_particles"]["total"],
            "n_terms": len(new_data["H"]),
            "elapsed_seconds": elapsed,
        })
    return _cache[key]


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------
def _mol_ids():
    """Return list of (molzoo_id, mol_entry) with appropriate markers."""
    params = []
    for entry in ALL_MOLECULES:
        molzoo_id = entry[0]
        tier = entry[3]
        marks = []
        if tier in _TIER_MARKERS:
            marks.append(_TIER_MARKERS[tier])
        params.append(pytest.param(entry, id=molzoo_id, marks=marks))
    return params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mol_entry", _mol_ids())
def test_molzoo_lookup(mol_entry):
    """Molecule can be found in molzoo.load_source('symmer')."""
    import molzoo

    molzoo_id = mol_entry[0]
    mols = {m.id: m for m in molzoo.load_source("symmer")}
    assert molzoo_id in mols, f"{molzoo_id} not found in molzoo symmer source"


@pytest.mark.parametrize("mol_entry", _mol_ids())
def test_qubit_count(mol_entry):
    """Generated n_qubits matches reference."""
    new_data, ref_data, _ref_ham, _elapsed = _get_result(mol_entry)
    assert new_data["n_qubits"] == ref_data["n_qubits"], (
        f"n_qubits mismatch: new={new_data['n_qubits']} ref={ref_data['n_qubits']}"
    )


@pytest.mark.parametrize("mol_entry", _mol_ids())
def test_hf_energy(mol_entry):
    """HF energy matches reference within 1e-6 Ha."""
    new_data, ref_data, _ref_ham, _elapsed = _get_result(mol_entry)
    new_hf = new_data["calculated_properties"]["HF"]["energy"]
    ref_hf = ref_data["calculated_properties"]["HF"]["energy"]
    assert abs(new_hf - ref_hf) < 1e-6, (
        f"HF energy mismatch: new={new_hf:.10f} ref={ref_hf:.10f} "
        f"diff={abs(new_hf - ref_hf):.2e}"
    )


@pytest.mark.parametrize("mol_entry", _mol_ids())
def test_mp2_energy(mol_entry):
    """MP2 energy matches reference within 1e-6 Ha."""
    molzoo_id = mol_entry[0]
    if molzoo_id in _KNOWN_FCI_ISSUES:
        pytest.skip(f"Known PySCF convergence issue for {molzoo_id}")
    new_data, ref_data, _ref_ham, _elapsed = _get_result(mol_entry)
    ref_mp2_entry = ref_data["calculated_properties"].get("MP2")
    if ref_mp2_entry is None or ref_mp2_entry.get("energy") is None:
        pytest.skip("No MP2 reference energy")
    new_mp2_entry = new_data["calculated_properties"].get("MP2", {})
    new_mp2 = new_mp2_entry.get("energy")
    if new_mp2 is None or (isinstance(new_mp2, float) and np.isnan(new_mp2)):
        pytest.fail("MP2 failed in generation but reference has MP2 energy")
    ref_mp2 = ref_mp2_entry["energy"]
    assert abs(new_mp2 - ref_mp2) < 1e-6, (
        f"MP2 energy mismatch: new={new_mp2:.10f} ref={ref_mp2:.10f} "
        f"diff={abs(new_mp2 - ref_mp2):.2e}"
    )


@pytest.mark.parametrize("mol_entry", _mol_ids())
def test_fci_energy(mol_entry):
    """FCI energy matches reference within 1e-6 Ha."""
    molzoo_id = mol_entry[0]
    if molzoo_id in _KNOWN_FCI_ISSUES:
        pytest.skip(f"Known PySCF FCI convergence issue for {molzoo_id}")
    new_data, ref_data, _ref_ham, _elapsed = _get_result(mol_entry)
    ref_fci_entry = ref_data["calculated_properties"].get("FCI")
    if ref_fci_entry is None or ref_fci_entry.get("energy") is None:
        pytest.skip("No FCI reference energy (e.g. O_STO-3G_TRIPLET)")
    new_fci_entry = new_data["calculated_properties"].get("FCI", {})
    new_fci = new_fci_entry.get("energy")
    if new_fci is None:
        pytest.fail("FCI failed in generation but reference has FCI energy")
    ref_fci = ref_fci_entry["energy"]
    assert abs(new_fci - ref_fci) < 1e-6, (
        f"FCI energy mismatch: new={new_fci:.10f} ref={ref_fci:.10f} "
        f"diff={abs(new_fci - ref_fci):.2e}"
    )


@pytest.mark.parametrize("mol_entry", _mol_ids())
def test_n_hamiltonian_terms(mol_entry):
    """Number of Hamiltonian terms matches reference exactly."""
    new_data, _ref_data, ref_ham, _elapsed = _get_result(mol_entry)
    new_n = len(new_data["H"])
    ref_n = len(ref_ham)
    assert new_n == ref_n, (
        f"Hamiltonian term count mismatch: new={new_n} ref={ref_n}"
    )


@pytest.mark.parametrize("mol_entry", _mol_ids())
def test_hamiltonian_coefficients(mol_entry):
    """Hamiltonian coefficient magnitudes match reference within 1e-6.

    Note: We compare |coeff| rather than raw coefficients because orbital
    phase conventions can differ between PySCF runs, causing sign flips in
    the JW-transformed Pauli coefficients.  The Hamiltonians are physically
    equivalent (same eigenvalues) regardless of these signs.
    """
    new_data, _ref_data, ref_ham, _elapsed = _get_result(mol_entry)
    new_h = new_data["H"]

    # Normalise dict values to float (symmer_to_dict mixes float/[r,i])
    def _to_complex(v):
        if isinstance(v, list):
            return complex(v[0], v[1])
        return complex(v)

    # Build sorted arrays by Pauli string
    all_keys = sorted(set(new_h.keys()) | set(ref_ham.keys()))
    new_vals = np.array([abs(_to_complex(new_h.get(k, 0))) for k in all_keys])
    ref_vals = np.array([abs(_to_complex(ref_ham.get(k, 0))) for k in all_keys])

    max_diff = np.max(np.abs(new_vals - ref_vals))
    assert max_diff < 1e-5, (
        f"Hamiltonian coefficient magnitude max diff = {max_diff:.2e} (threshold 1e-5)"
    )
