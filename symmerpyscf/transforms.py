"""Generalized fermionic transformations for quantum chemistry."""

import numpy as np
import os
import time
import multiprocessing
from typing import Optional, Union, List
from sympy import Matrix

from openfermion.ops.operators import FermionOperator, QubitOperator
from openfermion.utils.operator_utils import count_qubits
from symmer import QuantumState


def random_invertible_binary_matrix(
        n: int,
        beta: Optional[Union[str, np.ndarray]] = None,
        max_non_zero: Optional[int] = None,
        check_invertible: bool = True
) -> np.ndarray:
    """
    Generate a random invertible binary matrix over GF(2).

    Args:
        n: Matrix dimension (n x n)
        beta: If string, returns predefined transformation ('Jordan-Wigner', 'Bravyi-Kitaev')
              If array, returns the array itself
        max_non_zero: Maximum number of non-zero elements (optional)
        check_invertible: Whether to check matrix invertibility

    Returns:
        n x n binary matrix that is invertible over GF(2)

    Example:
        >>> beta = random_invertible_binary_matrix(4, beta='Jordan-Wigner')
        >>> # Returns 4x4 identity matrix
    """
    if beta is not None:
        return _select_beta(beta, n)

    ct = 0
    while True:
        if max_non_zero is not None:
            # Create sparse matrix with at most max_non_zero ones
            matrix = np.zeros((n, n), dtype=np.int8)
            ones_count = 0
            while ones_count < max_non_zero:
                i, j = np.random.randint(0, n), np.random.randint(0, n)
                if matrix[i, j] == 0:
                    matrix[i, j] = 1
                    ones_count += 1
        else:
            # Full random matrix
            seed = os.getpid() + time.time_ns()
            rng = np.random.default_rng(seed)
            matrix = rng.integers(0, 2, size=(n, n), dtype=np.int8)

        if not check_invertible:
            return matrix

        # Check invertibility: determinant must be odd over GF(2)
        if Matrix(matrix).det() % 2 != 0:
            return matrix

        ct += 1


def generalized_transformation(
        operator: FermionOperator,
        beta: Optional[Union[str, np.ndarray]] = None,
        n_qubits: Optional[int] = None
) -> QubitOperator:
    """
    Apply generalized fermionic transformation to convert fermions to qubits.

    This implements a general class of transformations parameterized by a binary
    matrix beta. Special cases include Jordan-Wigner (beta = identity) and
    Bravyi-Kitaev transformations.

    Args:
        operator: FermionOperator to transform
        beta: Transformation matrix or string ('Jordan-Wigner', 'Bravyi-Kitaev')
        n_qubits: Number of qubits (inferred if not specified)

    Returns:
        QubitOperator: Transformed operator in qubit basis

    Raises:
        ValueError: If invalid number of qubits or beta matrix not invertible
        TypeError: If operator is not a FermionOperator

    Example:
        >>> from openfermion import FermionOperator
        >>> hop = FermionOperator('0^ 1')
        >>> beta = random_invertible_binary_matrix(2, beta='Jordan-Wigner')
        >>> qubit_op = generalized_transformation(hop, beta)
    """
    if n_qubits is None:
        if beta is not None and hasattr(beta, 'shape'):
            n_qubits = beta.shape[0]
        else:
            n_qubits = count_qubits(operator)

    beta = _select_beta(beta, n_qubits)

    if isinstance(operator, FermionOperator):
        return _bravyi_kitaev_fermion_operator(operator, beta)

    raise TypeError(
        f"Couldn't apply Generalized Transform to object of type {type(operator)}"
    )


def generalized_transformation_product_state(
        product_state: np.ndarray,
        beta: Optional[Union[str, np.ndarray]] = None,
        n_qubits: Optional[int] = None
) -> np.ndarray:
    """
    Transform a product state (occupation number vector) via beta matrix.

    Args:
        product_state: Binary occupation number vector
        beta: Transformation matrix
        n_qubits: Number of qubits

    Returns:
        Transformed binary state vector

    Example:
        >>> state = np.array([1, 1, 0, 0])  # Two electrons in first two orbitals
        >>> beta = random_invertible_binary_matrix(4, beta='Jordan-Wigner')
        >>> transformed = generalized_transformation_product_state(state, beta)
    """
    if n_qubits is None:
        n_qubits = len(product_state)
    beta = _select_beta(beta, n_qubits)

    return np.mod(beta @ product_state, 2)


def generalized_transformation_symmer_jw_state(
        jw_state: QuantumState,
        beta: Optional[Union[str, np.ndarray]] = None,
        least_significant_bit: bool = False,
        n_qubits: Optional[int] = None
) -> QuantumState:
    """
    Transform a Symmer QuantumState from Jordan-Wigner basis to generalized basis.

    Args:
        jw_state: QuantumState in Jordan-Wigner encoding
        beta: Transformation matrix
        least_significant_bit: Bit ordering convention
        n_qubits: Number of qubits

    Returns:
        QuantumState in transformed basis

    Example:
        >>> from symmer import QuantumState
        >>> jw_state = QuantumState.from_dictionary({'1100': 1.0})
        >>> beta = random_invertible_binary_matrix(4, beta='Bravyi-Kitaev')
        >>> gt_state = generalized_transformation_symmer_jw_state(jw_state, beta)
    """
    jw_state_dict = jw_state.to_dictionary
    gt_state_dict = {}

    for product_state, coeff in jw_state_dict.items():
        product_state_ = np.array(list(product_state), dtype=np.int8)[::-1]

        product_state_gt = generalized_transformation_product_state(
            product_state=product_state_,
            beta=beta,
            n_qubits=n_qubits
        )

        if not least_significant_bit:
            product_state_gt = product_state_gt[::-1]

        binary_str = "".join(product_state_gt.astype(str))
        gt_state_dict[binary_str] = coeff

    gt_state = QuantumState.from_dictionary(gt_state_dict)
    return gt_state


# ============================================================================
# Internal helper functions
# ============================================================================

def _select_beta(beta: Optional[Union[str, np.ndarray]], n_qubits: int) -> np.ndarray:
    """Select or generate beta transformation matrix."""
    # Handle None case
    if beta is None:
        return np.eye(n_qubits, dtype=np.int8)

    # Handle string specifications
    if isinstance(beta, str):
        if beta == 'Jordan-Wigner':
            return np.eye(n_qubits, dtype=np.int8)
        elif beta == 'Bravyi-Kitaev':
            return _bravyi_kitaev_sub_matrix(n_qubits)
        elif beta == 'Bravyi-Kitaev-plus-i':
            base_beta = _bravyi_kitaev_sub_matrix(n_qubits)
            betas = [base_beta]
            for i in range(base_beta.shape[0]):
                for j in range(i + 1, base_beta.shape[0]):
                    beta_new = base_beta.copy()
                    beta_new[i, j] = np.mod(base_beta[i, j] + 1, 2)
                    betas.append(beta_new)
            return betas
        else:
            raise ValueError(f"Invalid Beta string: {beta}")

    # Handle numpy array - return as is
    if isinstance(beta, np.ndarray):
        return beta

    # Fallback: convert to array and return
    return np.array(beta, dtype=np.int8)


def _bravyi_kitaev_sub_matrix(n_qubits: int) -> np.ndarray:
    """Generate Bravyi-Kitaev transformation matrix."""
    log_qubit_number = int(np.ceil(np.sqrt(n_qubits - 0.5)))
    full_beta_matrix = _bravyi_kitaev_matrix_full_size(log_qubit_number)
    size_beta_matrix = 2 ** log_qubit_number

    return full_beta_matrix[
        size_beta_matrix - n_qubits:,
        size_beta_matrix - n_qubits:
    ]


def _bravyi_kitaev_matrix_full_size(log_qubit_number: int) -> np.ndarray:
    """Recursively build full-size Bravyi-Kitaev matrix."""
    if log_qubit_number == 0:
        return np.array([[1]], dtype=np.int8)

    beta_sub = _bravyi_kitaev_matrix_full_size(log_qubit_number - 1)
    sub_mat_size = 2 ** (log_qubit_number - 1)

    filling = np.zeros((sub_mat_size, sub_mat_size), dtype=np.int8)
    filling[0, :] = 1

    zeros = np.zeros((sub_mat_size, sub_mat_size), dtype=np.int8)

    return np.block([[beta_sub, filling], [zeros, beta_sub]])


def _update_set(index: int, beta: np.ndarray) -> set:
    """Return update set (Kee's notation) for given index."""
    n_qubits = beta.shape[0]
    update_set = n_qubits - np.nonzero(beta[:, n_qubits - index - 1])[0] - 1
    return set(update_set.tolist())


def _remainder_set(index: int, beta_inv: np.ndarray) -> set:
    """Return remainder set (Kee's notation) for given index."""
    n_qubits = beta_inv.shape[0]
    parity = np.triu(np.ones(beta_inv.shape, dtype=np.int8))
    remainder_matrix = np.mod(parity @ beta_inv, 2).astype(np.int8)

    remainder_set = n_qubits - np.nonzero(remainder_matrix[n_qubits - index - 1, :])[0] - 1
    return set(remainder_set.tolist())


def _parity_set(index: int, beta_inv: np.ndarray) -> set:
    """Return parity set (Kee's notation) for given index."""
    n_qubits = beta_inv.shape[0]
    parity = np.triu(np.ones(beta_inv.shape, dtype=np.int8))
    parity_matrix = np.mod(parity @ beta_inv - beta_inv, 2).astype(np.int8)

    parity_set = n_qubits - np.nonzero(parity_matrix[n_qubits - index - 1, :])[0] - 1
    return set(parity_set.tolist())


def _generalized_transformation_table(beta: np.ndarray) -> dict:
    """Pre-compute transformation table for all ladder operators."""
    try:
        beta_inv = Matrix(beta).inv_mod(2)
    except:
        raise ValueError("Input Beta Matrix is not invertible")

    n_qubit = beta.shape[0]
    table = {}

    for index in range(n_qubit):
        update_set = _update_set(index, beta)
        remainder_set = _remainder_set(index, beta_inv)
        parity_set = _parity_set(index, beta_inv)

        transformed_operator = QubitOperator(
            [(i, 'X') for i in update_set] + [(i, 'Z') for i in parity_set],
            0.5
        )

        transformed_majorana_difference = QubitOperator(
            [(i, 'X') for i in update_set] + [(i, 'Z') for i in remainder_set],
            0.5
        )

        # action=1: creation, action=0: annihilation
        table[(index, 1)] = transformed_operator + transformed_majorana_difference
        table[(index, 0)] = transformed_operator - transformed_majorana_difference

    return table


def _transform_ladder_operator(
        ladder_operator: tuple,
        transformation_table: dict
) -> QubitOperator:
    """Transform a single ladder operator using pre-computed table."""
    index, action = ladder_operator
    return transformation_table[(index, action)]


def _transform_operator_term(
        term: list,
        coefficient: float,
        transformation_table: dict
) -> QubitOperator:
    """Transform a single term in the FermionOperator."""
    transformed_ladder_ops = (
        _transform_ladder_operator(ladder_operator, transformation_table)
        for ladder_operator in term
    )
    return _inline_product(
        factors=transformed_ladder_ops,
        seed=QubitOperator((), coefficient)
    )


def _transform_wrapper(args):
    """Wrapper for multiprocessing."""
    term, coefficient, transformation_table = args
    return _transform_operator_term(term, coefficient, transformation_table)


def _bravyi_kitaev_fermion_operator(
        operator: FermionOperator,
        beta: np.ndarray
) -> QubitOperator:
    """Transform FermionOperator using generalized transformation with parallelization."""
    n_qubits = beta.shape[0]
    N = count_qubits(operator)

    if n_qubits is None:
        n_qubits = N
    if n_qubits < N:
        raise ValueError('Invalid number of qubits specified.')

    transformation_table = _generalized_transformation_table(beta)

    # Prepare arguments for parallel processing
    args_list = [
        (term, operator.terms[term], transformation_table)
        for term in operator.terms
    ]

    # Use multiprocessing for large operators
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        transformed_terms = pool.map(_transform_wrapper, args_list)

    return _inline_sum(summands=transformed_terms, seed=QubitOperator())


def _inline_sum(summands, seed):
    """Compute sum using __iadd__ operator."""
    for r in summands:
        seed += r
    return seed


def _inline_product(factors, seed):
    """Compute product using __imul__ operator."""
    for r in factors:
        seed *= r
    return seed