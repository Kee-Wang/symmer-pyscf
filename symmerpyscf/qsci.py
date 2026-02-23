"""QSCI (Quantum Selected Configuration Interaction) solver."""

from typing import Dict, Tuple

import numpy as np
import scipy as sp
from symmer.operators.base import PauliwordOp, QuantumState


def qsci_symmer_with_prob_hist(
    symmer_hamiltonian: PauliwordOp,
    probability_histogram: Dict[str, int],
    check_output: bool = True,
) -> Tuple[float, QuantumState]:
    """Compute QSCI energy from a Z-basis probability histogram.

    Builds a subspace from the measured bitstrings, projects the Hamiltonian
    into that subspace, and diagonalizes to find the ground state.

    Parameters
    ----------
    symmer_hamiltonian : PauliwordOp
        Pauli Hamiltonian.
    probability_histogram : dict[str, int | float]
        Bitstring -> count dictionary from Z-basis measurements.
        Assumes bad bitstrings (wrong particle number) have already been removed.
    check_output : bool
        If True, verify the QSCI energy via expectation value.

    Returns
    -------
    qsci_energy : float
        Ground-state energy in the measured subspace.
    qsci_state : QuantumState
        Corresponding eigenstate.
    """
    q_state = QuantumState.from_dictionary(probability_histogram).cleanup()
    q_state.state_op.coeff_vec = np.ones(len(q_state.state_op.coeff_vec), dtype=int)

    H_sub = np.zeros((q_state.n_terms, q_state.n_terms), dtype=float)
    for idx1, bra in enumerate(q_state):
        for idx2, ket in enumerate(q_state):
            H_sub[idx1, idx2] = (bra.dagger * symmer_hamiltonian * ket).real

    # Diagonalize
    if H_sub.shape == (1, 1):
        eigvals = np.array([H_sub[0, 0]])
        eigvecs = np.array([[1.0]])
    else:
        H_sparse = sp.sparse.csr_array(H_sub)
        eigvals, eigvecs = sp.sparse.linalg.eigsh(H_sparse, which='SA', k=1)

    min_idx = np.argmin(eigvals)
    qsci_energy = eigvals[min_idx]
    qsci_state = q_state.copy()
    qsci_state.state_op.coeff_vec = eigvecs[:, min_idx]

    if check_output:
        assert np.isclose(
            symmer_hamiltonian.expval(qsci_state).real, qsci_energy
        ), "energy not correct"

    return qsci_energy, qsci_state
