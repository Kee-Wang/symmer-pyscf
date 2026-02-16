"""Tests for CAS Hamiltonian generation."""

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from symmerpyscf import generate_cas_qubit_hamiltonian

# ── Reference data ──────────────────────────────────────────────────────────

H2_GEOMETRY = [("H", (0, 0, 0)), ("H", (0, 0, 0.735))]
H2_BASIS = "sto-3g"
H2_FCI_ENERGY = -1.137306035753  # Exact FCI = CAS(2,2) for H2/STO-3G

N2_GEOMETRY = [("N", (0, 0, 0)), ("N", (0, 0, 1.1))]
N2_BASIS = "sto-3g"

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def h2_cas():
    """H2/STO-3G CAS(2,2) at R=0.735 A."""
    return generate_cas_qubit_hamiltonian(H2_GEOMETRY, H2_BASIS, ncas=2, nelecas=2)


@pytest.fixture(scope="module")
def n2_cas_results():
    """N2/STO-3G CASCI results for multiple active spaces at R=1.1 A."""
    results = {}
    for ncas, nelecas in [(4, 2), (6, 6), (10, 14)]:
        results[(ncas, nelecas)] = generate_cas_qubit_hamiltonian(
            N2_GEOMETRY, N2_BASIS, ncas, nelecas
        )
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


def test_return_keys(h2_cas):
    """All expected keys are present in the returned dict."""
    assert set(h2_cas.keys()) == EXPECTED_KEYS


def test_h2_reference_energy(h2_cas):
    """H2/STO-3G CAS(2,2) matches known FCI energy to 1e-6."""
    assert abs(h2_cas["e_casci"] - H2_FCI_ENERGY) < 1e-6


def test_h2_hermiticity(h2_cas):
    """H_cas sparse matrix is Hermitian (H2)."""
    mat = h2_cas["H_cas"].to_sparse_matrix
    diff = mat - mat.conj().T
    assert abs(diff).max() < 1e-10


def test_h2_eigenvalue_consistency(h2_cas):
    """min(eig(H_cas)) + e_core == e_casci to 1e-8 (H2)."""
    e_min = _min_eigenvalue(h2_cas["H_cas"])
    total = e_min + h2_cas["e_core"]
    assert abs(total - h2_cas["e_casci"]) < 1e-8


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
    result = generate_cas_qubit_hamiltonian(
        H2_GEOMETRY, H2_BASIS, ncas=2, nelecas=2, use_mp2_natorbs=False
    )
    assert "H_cas" in result
    # Self-consistency still holds
    e_min = _min_eigenvalue(result["H_cas"])
    assert abs(e_min + result["e_core"] - result["e_casci"]) < 1e-8


def test_nelecas_tuple_vs_int():
    """nelecas=6 and nelecas=(3,3) give identical results for CAS(6,6)."""
    result_int = generate_cas_qubit_hamiltonian(
        N2_GEOMETRY, N2_BASIS, ncas=6, nelecas=6
    )
    result_tuple = generate_cas_qubit_hamiltonian(
        N2_GEOMETRY, N2_BASIS, ncas=6, nelecas=(3, 3)
    )
    assert abs(result_int["e_casci"] - result_tuple["e_casci"]) < 1e-10


# ── Slow tests (N2 multi-active-space, ~7 min) ─────────────────────────────

@pytest.mark.slow
def test_n_qubits(n2_cas_results):
    """n_qubits == 2 * ncas for each active space."""
    for (ncas, _), result in n2_cas_results.items():
        assert result["n_qubits"] == 2 * ncas


@pytest.mark.slow
def test_hamiltonian_hermiticity(n2_cas_results):
    """H_cas sparse matrix is Hermitian."""
    for result in n2_cas_results.values():
        mat = result["H_cas"].to_sparse_matrix
        diff = mat - mat.conj().T
        assert abs(diff).max() < 1e-10


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
    """min(eig(H_cas)) + e_core == e_casci to 1e-8."""
    for (ncas, nelecas), result in n2_cas_results.items():
        e_min = _min_eigenvalue(result["H_cas"])
        total = e_min + result["e_core"]
        assert abs(total - result["e_casci"]) < 1e-8, (
            f"CAS({ncas},{nelecas}): {total} vs {result['e_casci']}"
        )


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
        result = generate_cas_qubit_hamiltonian(geom, N2_BASIS, ncas=6, nelecas=6)

        # Self-consistency: min(eig) + e_core == e_casci
        e_min = _min_eigenvalue(result["H_cas"])
        assert abs(e_min + result["e_core"] - result["e_casci"]) < 1e-8, (
            f"R={bond_length}: {e_min + result['e_core']} vs {result['e_casci']}"
        )

        # Energy ordering
        assert result["e_fci"] <= result["e_casci"] + 1e-10
        assert result["e_casci"] <= result["e_hf"] + 1e-10
