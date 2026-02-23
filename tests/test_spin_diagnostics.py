"""Tests for S² diagnostics in mol_info_to_H_cs."""

import numpy as np
import pytest

from symmer import PauliwordOp, QuantumState
from symmerpyscf import initialize_molecule, mol_info_to_H_cs


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lih_mol_info():
    """LiH/STO-3G at R=1.6 A — returns mol_info dict."""
    mol_info, _, _, _ = initialize_molecule(bondlength=1.6, molecule="LiH")
    return mol_info


@pytest.fixture(scope="module")
def lih_cs_result(lih_mol_info):
    """mol_info_to_H_cs for LiH/STO-3G with 4 CS qubits."""
    return mol_info_to_H_cs(lih_mol_info, n_cs_qubits=4)


# ── Tests ───────────────────────────────────────────────────────────────────

def test_s2_cs_returned(lih_cs_result):
    """S2_cs is in the return dict and is a PauliwordOp."""
    assert 'S2_cs' in lih_cs_result
    assert isinstance(lih_cs_result['S2_cs'], PauliwordOp)


def test_cs_s2_returned(lih_cs_result):
    """cs_s2 is in the return dict and is a float."""
    assert 'cs_s2' in lih_cs_result
    assert isinstance(lih_cs_result['cs_s2'], float)


def test_cs_state_returned(lih_cs_result):
    """cs_state is in the return dict and is a QuantumState."""
    assert 'cs_state' in lih_cs_result
    assert isinstance(lih_cs_result['cs_state'], QuantumState)


def test_lih_singlet_s2(lih_cs_result):
    """LiH ground state is a singlet: <S²> should be near 0."""
    assert lih_cs_result['cs_s2'] < 0.1, (
        f"LiH ground state <S²> = {lih_cs_result['cs_s2']:.4f}, expected ~0 (singlet)"
    )


def test_s2_operator_hermitian(lih_cs_result):
    """S2_cs sparse matrix is Hermitian."""
    mat = lih_cs_result['S2_cs'].to_sparse_matrix
    diff = mat - mat.conj().T
    assert abs(diff).max() < 1e-10


def test_s2_nonnegative(lih_cs_result):
    """<S²> >= 0 (S² is a positive semi-definite operator)."""
    assert lih_cs_result['cs_s2'] >= -1e-10


def test_cs_state_consistency(lih_cs_result):
    """cs_state expectation value of H_cs matches cs_energy."""
    H_cs = lih_cs_result['H_cs']
    cs_state = lih_cs_result['cs_state']
    expval = H_cs.expval(cs_state).real
    assert np.isclose(expval, lih_cs_result['cs_energy'], atol=1e-8)


def test_return_dict_keys(lih_cs_result):
    """All expected keys are present in the return dict."""
    expected = {
        'H_cs', 'Na_CS', 'Nb_CS', 'CCSD_generator_CS', 'S2_cs',
        'hf_cs', 'cs_state', 'cs_energy', 'cs_s2', 'fci_energy',
        'beta', 'n_terms_hamiltonian', 'n_terms_ccsd_generator', 'n_qubits_cs',
    }
    assert set(lih_cs_result.keys()) == expected
