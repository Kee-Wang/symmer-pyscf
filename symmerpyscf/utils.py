"""Utility functions for symmer-pyscf package."""

from typing import Dict, Any, Union
import numpy as np
import scipy.sparse as sp
from openfermion.ops import FermionOperator
from symmer import QuantumState, PauliwordOp


def symmer_to_dict(state: Union[QuantumState, PauliwordOp]) -> Dict[str, Any]:
    """
    Convert Symmer QuantumState or PauliwordOp to JSON-serializable dictionary.

    Args:
        state: QuantumState or PauliwordOp object

    Returns:
        Dictionary with string keys. Values are plain floats when the
        imaginary part is zero, or [real, imag] lists otherwise.

    Example:
        >>> from symmer import QuantumState
        >>> state = QuantumState.from_dictionary({'00': 1.0})
        >>> state_dict = symmer_to_dict(state)
        >>> print(state_dict)
        {'00': 1.0}
    """
    state_dict = {}
    for key, val in state.sort().to_dictionary.items():
        if val.imag == 0:
            state_dict[key] = float(val.real)
        else:
            state_dict[key] = [float(val.real), float(val.imag)]

    return state_dict


def t1_t2_to_fermionic_operator(
        t1: np.ndarray,
        t2: np.ndarray,
        n_occ: int,
        n_virt: int
) -> FermionOperator:
    """
    Convert CCSD T1 and T2 amplitudes to fermionic operator.

    Args:
        t1: Single excitation amplitudes (n_occ x n_virt)
        t2: Double excitation amplitudes (n_occ x n_occ x n_virt x n_virt)
        n_occ: Number of occupied orbitals
        n_virt: Number of virtual orbitals

    Returns:
        FermionOperator representing the cluster operator

    Example:
        >>> import numpy as np
        >>> t1 = np.random.rand(2, 2)
        >>> t2 = np.random.rand(2, 2, 2, 2)
        >>> ccsd_op = t1_t2_to_fermionic_operator(t1, t2, n_occ=2, n_virt=2)
    """
    op = FermionOperator()

    # T1: single excitations (occupied -> virtual)
    for i in range(n_occ):
        for a in range(n_virt):
            amp = t1[i, a]
            # Create electron in virtual orbital a+n_occ, annihilate in occupied i
            op += FermionOperator(((a + n_occ, 1), (i, 0)), amp)

    # T2: double excitations
    for i in range(n_occ):
        for j in range(n_occ):
            for a in range(n_virt):
                for b in range(n_virt):
                    amp = t2[i, j, a, b]
                    # Create electrons in virtuals a,b, annihilate in occupied i,j
                    op += FermionOperator(
                        ((a + n_occ, 1), (b + n_occ, 1), (j, 0), (i, 0)),
                        coefficient=amp / 4.0  # Symmetry factor
                    )

    return op


# ---------------------------------------------------------------------------
# Hilbert-Schmidt inner product, norm, fidelity, and distance
# ---------------------------------------------------------------------------

def _to_sparse(A):
    """Convert A to a scipy sparse matrix if it is a PauliwordOp; pass through otherwise."""
    if isinstance(A, PauliwordOp):
        return A.to_sparse_matrix
    return A


def _both_pauli(A, B) -> bool:
    return isinstance(A, PauliwordOp) and isinstance(B, PauliwordOp)


def _pauli_coeff_inner_product(A: PauliwordOp, B: PauliwordOp) -> complex:
    """Tr(A† B) = 2^n Σ_P conj(α_P) β_P via Pauli orthogonality. O(k)."""
    if A.n_qubits != B.n_qubits:
        raise ValueError(f"Qubit count mismatch: {A.n_qubits} vs {B.n_qubits}")
    d_a = A.to_dictionary
    d_b = B.to_dictionary
    common = set(d_a) & set(d_b)
    if not common:
        return 0j
    coeff_sum = sum(np.conj(d_a[k]) * d_b[k] for k in common)
    return complex((2 ** A.n_qubits) * coeff_sum)


def hs_inner_product(A, B) -> complex:
    """
    Compute the Hilbert-Schmidt inner product Tr(A† B).

    Args:
        A: PauliwordOp, scipy sparse matrix, or numpy ndarray.
        B: PauliwordOp, scipy sparse matrix, or numpy ndarray.

    Returns:
        Complex value Tr(A† B).

    Notes:
        When both A and B are PauliwordOp, uses O(k) Pauli coefficient
        arithmetic instead of constructing 2^n × 2^n matrices.
    """
    if _both_pauli(A, B):
        return _pauli_coeff_inner_product(A, B)

    A_mat = _to_sparse(A)
    B_mat = _to_sparse(B)

    product = A_mat.conj().T @ B_mat

    if sp.issparse(product):
        return product.diagonal().sum()
    return np.trace(product)


def hs_norm(A) -> float:
    """
    Compute the Hilbert-Schmidt norm √Tr(A† A) (= Frobenius norm for matrices).

    Args:
        A: PauliwordOp, scipy sparse matrix, or numpy ndarray.

    Returns:
        Non-negative real value.
    """
    return np.sqrt(np.real(hs_inner_product(A, A)))


def hs_fidelity(A, B) -> float:
    """
    Compute the Hilbert-Schmidt fidelity |Tr(A† B)|² / (Tr(A† A) · Tr(B† B)).

    The result lies in [0, 1]; 1.0 means A and B are identical up to a global
    scalar factor.  Returns 0.0 if either operator is zero.

    Args:
        A: PauliwordOp, scipy sparse matrix, or numpy ndarray.
        B: PauliwordOp, scipy sparse matrix, or numpy ndarray.

    Returns:
        Float in [0, 1].
    """
    norm_A_sq = np.real(hs_inner_product(A, A))
    norm_B_sq = np.real(hs_inner_product(B, B))

    if norm_A_sq == 0 or norm_B_sq == 0:
        return 0.0

    ip = hs_inner_product(A, B)
    return float(np.abs(ip) ** 2 / (norm_A_sq * norm_B_sq))


def hs_infidelity(A, B) -> float:
    """
    Compute 1 − hs_fidelity(A, B).

    The result lies in [0, 1]; 0.0 means A and B are identical up to a global
    scalar factor.

    Args:
        A: PauliwordOp, scipy sparse matrix, or numpy ndarray.
        B: PauliwordOp, scipy sparse matrix, or numpy ndarray.

    Returns:
        Float in [0, 1].
    """
    return 1.0 - hs_fidelity(A, B)


def hs_distance(A, B) -> float:
    """
    Compute the Hilbert-Schmidt distance ‖A − B‖_HS = √Tr((A−B)†(A−B)).

    Args:
        A: PauliwordOp, scipy sparse matrix, or numpy ndarray.
        B: PauliwordOp, scipy sparse matrix, or numpy ndarray.

    Returns:
        Non-negative real value.

    Notes:
        When both A and B are PauliwordOp, uses O(k) Pauli coefficient
        arithmetic instead of constructing 2^n × 2^n matrices.
    """
    if _both_pauli(A, B):
        if A.n_qubits != B.n_qubits:
            raise ValueError(f"Qubit count mismatch: {A.n_qubits} vs {B.n_qubits}")
        d_a = A.to_dictionary
        d_b = B.to_dictionary
        all_keys = set(d_a) | set(d_b)
        sum_sq = sum(abs(d_a.get(k, 0) - d_b.get(k, 0)) ** 2 for k in all_keys)
        return float(np.sqrt((2 ** A.n_qubits) * sum_sq))

    A_mat = _to_sparse(A)
    B_mat = _to_sparse(B)
    diff = A_mat - B_mat
    if sp.issparse(diff):
        return float(np.sqrt(np.real(diff.conj().multiply(diff).sum())))
    return float(np.linalg.norm(diff, 'fro'))
