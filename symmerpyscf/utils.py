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


def reverse_bits(x: int, bit_length: int) -> int:
    """
    Reverse the bits of an integer.

    Args:
        x: Integer to reverse
        bit_length: Number of bits to consider

    Returns:
        Integer with reversed bits

    Example:
        >>> reverse_bits(0b1010, 4)  # Returns 0b0101 = 5
        5
    """
    result = 0
    for i in range(bit_length):
        if (x >> i) & 1:
            result |= 1 << (bit_length - 1 - i)
    return result


def compare_energies(energy_dict: Dict[str, float], reference_key: str = 'FCI') -> Dict[str, float]:
    """
    Compare energies relative to a reference energy.

    Args:
        energy_dict: Dictionary of method names to energies
        reference_key: Key for reference energy

    Returns:
        Dictionary of energy differences

    Example:
        >>> energies = {'HF': -1.0, 'CCSD': -1.1, 'FCI': -1.15}
        >>> errors = compare_energies(energies, reference_key='FCI')
        >>> print(errors)
        {'HF': 0.15, 'CCSD': 0.05, 'FCI': 0.0}
    """
    if reference_key not in energy_dict:
        raise ValueError(f"Reference key '{reference_key}' not found in energy_dict")

    reference_energy = energy_dict[reference_key]

    return {
        method: energy - reference_energy
        for method, energy in energy_dict.items()
    }


def print_energy_summary(
        energy_data: Dict[str, Dict[str, Any]],
        reference: str = 'FCI'
) -> None:
    """
    Print a formatted summary of energies and errors.

    Args:
        energy_data: Dictionary with structure {method: {'energy': float, 'converged': bool}}
        reference: Reference method for computing errors

    Example:
        >>> energy_data = {
        ...     'HF': {'energy': -1.0, 'converged': True},
        ...     'FCI': {'energy': -1.15, 'converged': True}
        ... }
        >>> print_energy_summary(energy_data)
    """
    print("=" * 60)
    print("Energy Summary")
    print("=" * 60)
    print(f"{'Method':<10} {'Energy (Ha)':<15} {'Error (Ha)':<15} {'Converged'}")
    print("-" * 60)

    ref_energy = energy_data[reference]['energy']

    for method, data in energy_data.items():
        energy = data['energy']
        converged = data.get('converged', 'N/A')
        error = energy - ref_energy if method != reference else 0.0

        print(f"{method:<10} {energy:<15.8f} {error:<15.8e} {converged}")

    print("=" * 60)


def pauli_string_complexity(operator: PauliwordOp) -> Dict[str, Any]:
    """
    Analyze the complexity of a PauliwordOp.

    Args:
        operator: PauliwordOp to analyze

    Returns:
        Dictionary with complexity metrics

    Example:
        >>> from symmer import PauliwordOp
        >>> op = PauliwordOp.from_dictionary({'XXYY': 1.0, 'ZZZZ': 0.5})
        >>> metrics = pauli_string_complexity(op)
        >>> print(metrics['n_terms'])
        2
    """
    pauli_dict = operator.to_dictionary

    all_weights = [
        sum(1 for p in pauli_str if p != 'I')
        for pauli_str in pauli_dict.keys()
    ]

    weight_distribution = {}
    for w in all_weights:
        weight_distribution[w] = weight_distribution.get(w, 0) + 1

    return {
        'n_terms': operator.n_terms,
        'n_qubits': operator.n_qubits,
        'weight_distribution': weight_distribution,
        'max_weight': max(all_weights) if all_weights else 0,
        'mean_weight': np.mean(all_weights) if all_weights else 0,
    }


# ---------------------------------------------------------------------------
# Hilbert-Schmidt inner product, norm, fidelity, and distance
# ---------------------------------------------------------------------------

def _to_sparse(A):
    """Convert A to a scipy sparse matrix if it is a PauliwordOp; pass through otherwise."""
    if isinstance(A, PauliwordOp):
        return A.to_sparse_matrix
    return A


def hs_inner_product(A, B) -> complex:
    """
    Compute the Hilbert-Schmidt inner product Tr(A† B).

    Args:
        A: PauliwordOp, scipy sparse matrix, or numpy ndarray.
        B: PauliwordOp, scipy sparse matrix, or numpy ndarray.

    Returns:
        Complex value Tr(A† B).
    """
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


def hs_distance(A, B) -> float:
    """
    Compute the Hilbert-Schmidt distance ‖A − B‖_HS = √Tr((A−B)†(A−B)).

    Args:
        A: PauliwordOp, scipy sparse matrix, or numpy ndarray.
        B: PauliwordOp, scipy sparse matrix, or numpy ndarray.

    Returns:
        Non-negative real value.
    """
    A_mat = _to_sparse(A)
    B_mat = _to_sparse(B)
    diff = A_mat - B_mat
    if sp.issparse(diff):
        return float(np.sqrt(np.real(diff.conj().multiply(diff).sum())))
    return float(np.linalg.norm(diff, 'fro'))
