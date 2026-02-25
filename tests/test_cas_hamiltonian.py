"""Tests for CAS Hamiltonian generation."""

import json
import os
import tempfile

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from symmer import PauliwordOp, QuantumState
from symmerpyscf import generate_cas_qubit_hamiltonian

# ── Reference data ──────────────────────────────────────────────────────────

H2_GEOMETRY = [("H", (0, 0, 0)), ("H", (0, 0, 0.735))]
H2_BASIS = "sto-3g"
H2_FCI_ENERGY = -1.137306035753  # Exact FCI = CAS(2,2) for H2/STO-3G

N2_GEOMETRY = [("N", (0, 0, 0)), ("N", (0, 0, 1.1))]
N2_BASIS = "sto-3g"

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def h2_cas_full():
    """H2/STO-3G CAS(2,2) at R=0.735 A — returns (cas_result, symmer_data)."""
    return generate_cas_qubit_hamiltonian(H2_GEOMETRY, H2_BASIS, ncas=2, nelecas=2)


@pytest.fixture(scope="module")
def h2_cas(h2_cas_full):
    """H2 cas_result dict (backward-compat convenience)."""
    return h2_cas_full[0]


@pytest.fixture(scope="module")
def h2_symmer_data(h2_cas_full):
    """H2 symmer_data dict."""
    return h2_cas_full[1]


@pytest.fixture(scope="module")
def n2_cas_results():
    """N2/STO-3G CASCI results for multiple active spaces at R=1.1 A."""
    results = {}
    for ncas, nelecas in [(4, 2), (6, 6), (10, 14)]:
        cas_result, _ = generate_cas_qubit_hamiltonian(
            N2_GEOMETRY, N2_BASIS, ncas, nelecas
        )
        results[(ncas, nelecas)] = cas_result
    return results


# ── Helper ──────────────────────────────────────────────────────────────────

def _min_eigenvalue(H_cas):
    """Compute minimum eigenvalue of a PauliwordOp."""
    mat = H_cas.to_sparse_matrix
    if mat.shape[0] <= 64:
        evals = np.linalg.eigvalsh(mat.toarray())
        return float(evals[0])
    return float(eigsh(mat, k=1, which="SA", return_eigenvectors=False)[0])


# ── Fast tests (H2 only, ~2s) ──────────────────────────────────────────────

EXPECTED_KEYS = {
    "H_cas", "H_fermion", "e_core", "cas_ground_state",
    "e_casci", "e_fci", "e_hf", "n_qubits", "ncas", "nelecas",
}


def test_return_type(h2_cas_full):
    """generate_cas_qubit_hamiltonian returns a (cas_result, symmer_data) tuple."""
    assert isinstance(h2_cas_full, tuple)
    assert len(h2_cas_full) == 2
    assert isinstance(h2_cas_full[0], dict)
    assert isinstance(h2_cas_full[1], dict)


def test_return_keys(h2_cas):
    """All expected keys are present in the cas_result dict."""
    assert set(h2_cas.keys()) == EXPECTED_KEYS


def test_h2_reference_energy(h2_cas):
    """H2/STO-3G CAS(2,2) matches known FCI energy to 1e-6."""
    assert abs(h2_cas["e_casci"] - H2_FCI_ENERGY) < 1e-6


def test_h2_hermiticity(h2_cas):
    """H_cas is Hermitian — all Pauli coefficients are real (H2)."""
    assert np.allclose(h2_cas["H_cas"].coeff_vec.imag, 0, atol=1e-10)


def test_h2_eigenvalue_consistency(h2_cas):
    """min(eig(H_cas)) == e_casci to 1e-8 (H2, e_core is included in H_cas)."""
    e_min = _min_eigenvalue(h2_cas["H_cas"])
    assert abs(e_min - h2_cas["e_casci"]) < 1e-8


def test_h2_energy_ordering(h2_cas):
    """e_fci <= e_casci <= e_hf (H2)."""
    assert h2_cas["e_fci"] <= h2_cas["e_casci"] + 1e-10
    assert h2_cas["e_casci"] <= h2_cas["e_hf"] + 1e-10


def test_h2_state_normalization(h2_cas):
    """cas_ground_state has unit norm (H2)."""
    state_vec = h2_cas["cas_ground_state"].to_sparse_matrix.toarray().flatten()
    assert abs(np.linalg.norm(state_vec) - 1.0) < 1e-10


def test_h2_state_dimension(h2_cas):
    """cas_ground_state lives in 2^(2*ncas) dimensional space (H2)."""
    state_vec = h2_cas["cas_ground_state"].to_sparse_matrix.toarray().flatten()
    assert len(state_vec) == 2 ** (2 * h2_cas["ncas"])


def test_mp2_natorbs_flag():
    """use_mp2_natorbs=False runs without error."""
    result, _ = generate_cas_qubit_hamiltonian(
        H2_GEOMETRY, H2_BASIS, ncas=2, nelecas=2, use_mp2_natorbs=False
    )
    assert "H_cas" in result
    # Self-consistency: min(eig(H_cas)) == e_casci
    e_min = _min_eigenvalue(result["H_cas"])
    assert abs(e_min - result["e_casci"]) < 1e-8


def test_nelecas_tuple_vs_int():
    """nelecas=6 and nelecas=(3,3) give identical results for CAS(6,6)."""
    result_int, _ = generate_cas_qubit_hamiltonian(
        N2_GEOMETRY, N2_BASIS, ncas=6, nelecas=6
    )
    result_tuple, _ = generate_cas_qubit_hamiltonian(
        N2_GEOMETRY, N2_BASIS, ncas=6, nelecas=(3, 3)
    )
    assert abs(result_int["e_casci"] - result_tuple["e_casci"]) < 1e-10


# ── Symmer-data format tests (H2, fast) ────────────────────────────────────

EXPECTED_SYMMER_KEYS = {
    "H", "H_second_quantized", "qubit_encoding", "geometry", "basis",
    "charge", "spin", "hf_array", "hf_state", "n_particles", "n_qubits",
    "calculated_properties", "auxiliary_operators", "cas_metadata",
}

EXPECTED_AUX_KEYS = {
    "number_operator", "N_alpha", "N_beta", "S^2_operator", "fci_state",
    "number_operator_second_quantized", "N_alpha_second_quantized",
    "N_beta_second_quantized", "S^2_operator_second_quantized",
}


def test_symmer_data_keys(h2_symmer_data):
    """symmer_data contains all expected top-level keys."""
    assert set(h2_symmer_data.keys()) == EXPECTED_SYMMER_KEYS


def test_symmer_data_aux_keys(h2_symmer_data):
    """auxiliary_operators contains all expected keys."""
    assert set(h2_symmer_data["auxiliary_operators"].keys()) == EXPECTED_AUX_KEYS


def test_symmer_data_n_qubits(h2_symmer_data):
    """n_qubits in symmer_data matches H2 CAS(2,2) = 4."""
    assert h2_symmer_data["n_qubits"] == 4


def test_symmer_data_hf_array(h2_symmer_data):
    """HF array is correct for H2 CAS(2,2): 2 electrons in 4 qubits."""
    assert h2_symmer_data["hf_array"] == [1, 1, 0, 0]


def test_symmer_data_n_particles(h2_symmer_data):
    """Particle counts are correct for H2 CAS(2,2)."""
    np_ = h2_symmer_data["n_particles"]
    assert np_["total"] == 2
    assert np_["alpha"] == 1
    assert np_["beta"] == 1


def test_symmer_data_cas_metadata(h2_symmer_data):
    """CAS metadata is correct."""
    meta = h2_symmer_data["cas_metadata"]
    assert meta["ncas"] == 2
    assert meta["nelecas"] == [1, 1]
    assert isinstance(meta["e_core"], float)
    assert meta["use_mp2_natorbs"] is True


def test_symmer_data_h_roundtrip(h2_symmer_data):
    """H dict round-trips through PauliwordOp.from_dictionary."""
    H_reloaded = PauliwordOp.from_dictionary(h2_symmer_data["H"])
    assert H_reloaded.n_qubits == h2_symmer_data["n_qubits"]
    assert H_reloaded.n_terms > 0


def test_symmer_data_hf_state_roundtrip(h2_symmer_data):
    """HF state dict round-trips through QuantumState.from_dictionary."""
    hf = QuantumState.from_dictionary(h2_symmer_data["hf_state"])
    vec = hf.to_sparse_matrix.toarray().flatten()
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-10


def test_save_file_json_roundtrip():
    """save_file writes valid JSON that round-trips."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        _, symmer_data = generate_cas_qubit_hamiltonian(
            H2_GEOMETRY, H2_BASIS, ncas=2, nelecas=2, save_file=tmp_path
        )
        with open(tmp_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Key structure preserved
        assert set(loaded.keys()) == EXPECTED_SYMMER_KEYS
        # H can be reloaded
        H_reloaded = PauliwordOp.from_dictionary(loaded["H"])
        assert H_reloaded.n_qubits == loaded["n_qubits"]
        # Eigenvalue matches
        e_min = _min_eigenvalue(H_reloaded)
        assert abs(e_min - loaded["calculated_properties"]["CASCI"]["energy"]) < 1e-8
    finally:
        os.unlink(tmp_path)


# ── Slow tests (N2 multi-active-space, ~7 min) ─────────────────────────────

@pytest.mark.slow
def test_n_qubits(n2_cas_results):
    """n_qubits == 2 * ncas for each active space."""
    for (ncas, _), result in n2_cas_results.items():
        assert result["n_qubits"] == 2 * ncas


@pytest.mark.slow
def test_hamiltonian_hermiticity(n2_cas_results):
    """H_cas is Hermitian — all Pauli coefficients are real."""
    for result in n2_cas_results.values():
        assert np.allclose(result["H_cas"].coeff_vec.imag, 0, atol=1e-10)


@pytest.mark.slow
def test_state_normalization(n2_cas_results):
    """cas_ground_state has unit norm."""
    for result in n2_cas_results.values():
        state_vec = result["cas_ground_state"].to_sparse_matrix.toarray().flatten()
        assert abs(np.linalg.norm(state_vec) - 1.0) < 1e-10


@pytest.mark.slow
def test_state_dimension(n2_cas_results):
    """cas_ground_state lives in 2^(2*ncas) dimensional space."""
    for (ncas, _), result in n2_cas_results.items():
        state_vec = result["cas_ground_state"].to_sparse_matrix.toarray().flatten()
        assert len(state_vec) == 2 ** (2 * ncas)


@pytest.mark.slow
def test_eigenvalue_consistency(n2_cas_results):
    """min(eig(H_cas)) == e_casci to 1e-8 (e_core included in H_cas)."""
    checked = 0
    for (ncas, nelecas), result in n2_cas_results.items():
        if result["n_qubits"] > 16:
            continue  # encoding validated by smaller active spaces; sparse matrix too large
        e_min = _min_eigenvalue(result["H_cas"])
        assert abs(e_min - result["e_casci"]) < 1e-8, (
            f"CAS({ncas},{nelecas}): {e_min} vs {result['e_casci']}"
        )
        checked += 1
    assert checked > 0, "No CAS configs were small enough to check — review n_qubits threshold"


@pytest.mark.slow
def test_variational_principle(n2_cas_results):
    """e_casci >= e_fci for every active space."""
    for (ncas, nelecas), result in n2_cas_results.items():
        assert result["e_casci"] >= result["e_fci"] - 1e-10, (
            f"CAS({ncas},{nelecas}): e_casci={result['e_casci']} < e_fci={result['e_fci']}"
        )


@pytest.mark.slow
def test_energy_ordering(n2_cas_results):
    """e_fci <= e_casci <= e_hf for every active space."""
    for (ncas, nelecas), result in n2_cas_results.items():
        assert result["e_fci"] <= result["e_casci"] + 1e-10
        assert result["e_casci"] <= result["e_hf"] + 1e-10


@pytest.mark.slow
def test_full_space_recovers_fci(n2_cas_results):
    """CAS(10,14) for N2/STO-3G recovers FCI to 1e-6."""
    result = n2_cas_results[(10, 14)]
    assert abs(result["e_casci"] - result["e_fci"]) < 1e-6


@pytest.mark.slow
def test_monotonicity(n2_cas_results):
    """Larger active space gives lower (or equal) CASCI energy."""
    e_4_2 = n2_cas_results[(4, 2)]["e_casci"]
    e_6_6 = n2_cas_results[(6, 6)]["e_casci"]
    e_10_14 = n2_cas_results[(10, 14)]["e_casci"]
    assert e_4_2 >= e_6_6 - 1e-10
    assert e_6_6 >= e_10_14 - 1e-10


@pytest.mark.slow
def test_multiple_bond_lengths():
    """N2 at 1.5 A and 2.0 A — self-consistency checks."""
    for bond_length in [1.5, 2.0]:
        geom = [("N", (0, 0, 0)), ("N", (0, 0, bond_length))]
        result, _ = generate_cas_qubit_hamiltonian(geom, N2_BASIS, ncas=6, nelecas=6)

        # Self-consistency: min(eig(H_cas)) == e_casci
        e_min = _min_eigenvalue(result["H_cas"])
        assert abs(e_min - result["e_casci"]) < 1e-8, (
            f"R={bond_length}: {e_min} vs {result['e_casci']}"
        )

        # Energy ordering
        assert result["e_fci"] <= result["e_casci"] + 1e-10
        assert result["e_casci"] <= result["e_hf"] + 1e-10
