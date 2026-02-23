"""Tests for QSCI solver."""

import numpy as np
import pytest
from symmer import PauliwordOp, QuantumState

from symmerpyscf import qsci_symmer_with_prob_hist


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def h2_hamiltonian():
    """Minimal 2-qubit Hamiltonian: H = -0.5 ZI - 0.5 IZ + 0.25 ZZ.

    Eigenvalues: {-1.25, -0.25, -0.25, 0.75}
    Ground state: |00> with energy -1.25
    """
    return PauliwordOp.from_dictionary({
        'ZI': -0.5,
        'IZ': -0.5,
        'ZZ': 0.25,
        'II': 0.25,
    })


@pytest.fixture
def heisenberg_2q():
    """2-qubit Heisenberg: H = XX + YY + ZZ.

    Eigenvalues: {-3, 1, 1, 1}
    Ground state: singlet (|01> - |10>)/sqrt(2) with energy -3.
    """
    return PauliwordOp.from_dictionary({
        'XX': 1.0,
        'YY': 1.0,
        'ZZ': 1.0,
    })


# ── Tests ───────────────────────────────────────────────────────────────────

def test_single_bitstring(h2_hamiltonian):
    """With one bitstring, QSCI returns its diagonal element."""
    hist = {'00': 100}
    energy, state = qsci_symmer_with_prob_hist(h2_hamiltonian, hist)
    # <00|H|00> = -0.5 - 0.5 + 0.25 + 0.25 = -0.5
    # Actually compute it: ZI|00> = +1, IZ|00> = +1, ZZ|00> = +1
    # E = -0.5*1 + -0.5*1 + 0.25*1 + 0.25*1 = -0.5
    expected = h2_hamiltonian.expval(QuantumState.from_dictionary({'00': 1})).real
    assert np.isclose(energy, expected, atol=1e-10)


def test_full_basis_recovers_exact(h2_hamiltonian):
    """With all 4 bitstrings, QSCI recovers the exact ground state energy."""
    hist = {'00': 25, '01': 25, '10': 25, '11': 25}
    energy, state = qsci_symmer_with_prob_hist(h2_hamiltonian, hist)
    # Eigenvalues of this diagonal Hamiltonian: {-0.5, 0, 0, 1.5}
    assert np.isclose(energy, -0.5, atol=1e-10)


def test_heisenberg_subspace(heisenberg_2q):
    """Subspace {|01>, |10>} contains the singlet ground state at E=-3."""
    hist = {'01': 50, '10': 50}
    energy, state = qsci_symmer_with_prob_hist(heisenberg_2q, hist)
    assert np.isclose(energy, -3.0, atol=1e-10)


def test_heisenberg_wrong_subspace(heisenberg_2q):
    """Subspace {|00>, |11>} gives triplet energy E=1, not ground state."""
    hist = {'00': 50, '11': 50}
    energy, state = qsci_symmer_with_prob_hist(heisenberg_2q, hist)
    assert np.isclose(energy, 1.0, atol=1e-10)


def test_returned_state_is_eigenstate(heisenberg_2q):
    """The returned state satisfies H|psi> = E|psi>."""
    hist = {'01': 50, '10': 50}
    energy, state = qsci_symmer_with_prob_hist(heisenberg_2q, hist)
    expval = heisenberg_2q.expval(state).real
    assert np.isclose(expval, energy, atol=1e-10)


def test_check_output_flag(h2_hamiltonian):
    """check_output=False skips the assertion."""
    hist = {'00': 100}
    energy, state = qsci_symmer_with_prob_hist(
        h2_hamiltonian, hist, check_output=False
    )
    assert isinstance(energy, (float, np.floating))
