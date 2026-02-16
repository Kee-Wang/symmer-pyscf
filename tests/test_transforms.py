"""Tests for symmerpyscf.transforms — generalized fermion-to-qubit mappings.

Compares generalized_transformation() against OpenFermion's jordan_wigner()
and bravyi_kitaev() reference implementations.
"""

import numpy as np
import pytest
from sympy import Matrix

from openfermion import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from openfermion.linalg import eigenspectrum, get_sparse_operator
from openfermion.utils import count_qubits

from symmerpyscf.transforms import (
    generalized_transformation,
    generalized_transformation_product_state,
    random_invertible_binary_matrix,
    _select_beta,
    _bravyi_kitaev_sub_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qubit_op_to_matrix(op: QubitOperator, n_qubits: int) -> np.ndarray:
    """Convert QubitOperator to dense complex128 matrix for numerical comparison."""
    return get_sparse_operator(op, n_qubits=n_qubits).toarray().astype(np.complex128)


def _assert_operators_equal(op1: QubitOperator, op2: QubitOperator, n_qubits: int):
    """Assert two QubitOperators are equal via their matrix representations."""
    mat1 = _qubit_op_to_matrix(op1, n_qubits)
    mat2 = _qubit_op_to_matrix(op2, n_qubits)
    np.testing.assert_allclose(mat1, mat2, atol=1e-12,
                               err_msg=f"Operators differ:\n  got:      {op1}\n  expected: {op2}")


# ---------------------------------------------------------------------------
# H2 Hamiltonian (STO-3G, 4 qubits) — standard benchmark
# ---------------------------------------------------------------------------

def _h2_fermion_hamiltonian() -> FermionOperator:
    """Minimal H2 Hamiltonian in STO-3G basis (4 spin-orbitals).

    Coefficients from standard quantum chemistry references.
    """
    h = FermionOperator()
    # One-body terms  (h_pq a†_p a_q)
    h1 = np.array([
        [-1.2563390730032498, 0.0],
        [0.0, -0.47189600728114245],
    ])
    for p in range(2):
        for q in range(2):
            if abs(h1[p, q]) > 1e-15:
                # alpha
                h += FermionOperator(f'{2*p}^ {2*q}', h1[p, q])
                # beta
                h += FermionOperator(f'{2*p+1}^ {2*q+1}', h1[p, q])

    # Two-body terms (0.5 * h_pqrs a†_p a†_q a_s a_r) — chemist notation
    h2 = 0.67460573424698263
    # (00|00)
    h += FermionOperator('0^ 0^ 0 0', 0.0)  # Pauli exclusion
    # (00|11) = (11|00) = h2
    h += 0.5 * h2 * FermionOperator('0^ 1^ 1 0')
    h += 0.5 * h2 * FermionOperator('1^ 0^ 0 1')
    # Cross-spin two-electron integrals
    h += 0.5 * h2 * FermionOperator('0^ 2^ 2 0')
    h += 0.5 * h2 * FermionOperator('2^ 0^ 0 2')
    h += 0.5 * h2 * FermionOperator('1^ 3^ 3 1')
    h += 0.5 * h2 * FermionOperator('3^ 1^ 1 3')
    h += 0.5 * h2 * FermionOperator('0^ 3^ 3 0')
    h += 0.5 * h2 * FermionOperator('3^ 0^ 0 3')
    h += 0.5 * h2 * FermionOperator('1^ 2^ 2 1')
    h += 0.5 * h2 * FermionOperator('2^ 1^ 1 2')
    # (11|11)
    h += 0.5 * 0.69757345816655 * FermionOperator('2^ 3^ 3 2')
    h += 0.5 * 0.69757345816655 * FermionOperator('3^ 2^ 2 3')
    # (01|10) exchange
    h += 0.5 * 0.66347023791446527 * FermionOperator('0^ 2^ 0 2')
    h += 0.5 * 0.66347023791446527 * FermionOperator('2^ 0^ 2 0')
    h += 0.5 * 0.66347023791446527 * FermionOperator('1^ 3^ 1 3')
    h += 0.5 * 0.66347023791446527 * FermionOperator('3^ 1^ 3 1')
    # nuclear repulsion
    h += FermionOperator('', 0.7137539936876182)
    return h


# ---------------------------------------------------------------------------
# 1. JW single creation / annihilation operators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_jw_single_creation(j):
    """a†_j via generalized_transformation(JW) matches openfermion.jordan_wigner."""
    n = 4
    op = FermionOperator(f'{j}^')
    beta = random_invertible_binary_matrix(n, beta='Jordan-Wigner')
    got = generalized_transformation(op, beta, n_qubits=n)
    expected = jordan_wigner(op)
    _assert_operators_equal(got, expected, n)


@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_jw_single_annihilation(j):
    """a_j via generalized_transformation(JW) matches openfermion.jordan_wigner."""
    n = 4
    op = FermionOperator(f'{j}')
    beta = random_invertible_binary_matrix(n, beta='Jordan-Wigner')
    got = generalized_transformation(op, beta, n_qubits=n)
    expected = jordan_wigner(op)
    _assert_operators_equal(got, expected, n)


# ---------------------------------------------------------------------------
# 2. BK single creation / annihilation operators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_bk_single_creation(j):
    """a†_j via generalized_transformation(BK) matches openfermion.bravyi_kitaev."""
    n = 4
    op = FermionOperator(f'{j}^')
    beta = random_invertible_binary_matrix(n, beta='Bravyi-Kitaev')
    got = generalized_transformation(op, beta, n_qubits=n)
    expected = bravyi_kitaev(op, n_qubits=n)
    _assert_operators_equal(got, expected, n)


@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_bk_single_annihilation(j):
    """a_j via generalized_transformation(BK) matches openfermion.bravyi_kitaev."""
    n = 4
    op = FermionOperator(f'{j}')
    beta = random_invertible_binary_matrix(n, beta='Bravyi-Kitaev')
    got = generalized_transformation(op, beta, n_qubits=n)
    expected = bravyi_kitaev(op, n_qubits=n)
    _assert_operators_equal(got, expected, n)


# ---------------------------------------------------------------------------
# 3. JW hopping term
# ---------------------------------------------------------------------------

def test_jw_hopping_term():
    """a†_0 a_1 + h.c. via JW matches OpenFermion."""
    n = 4
    hop = FermionOperator('0^ 1') + FermionOperator('1^ 0')
    beta = random_invertible_binary_matrix(n, beta='Jordan-Wigner')
    got = generalized_transformation(hop, beta, n_qubits=n)
    expected = jordan_wigner(hop)
    _assert_operators_equal(got, expected, n)


# ---------------------------------------------------------------------------
# 4. BK hopping term
# ---------------------------------------------------------------------------

def test_bk_hopping_term():
    """a†_0 a_1 + h.c. via BK matches OpenFermion."""
    n = 4
    hop = FermionOperator('0^ 1') + FermionOperator('1^ 0')
    beta = random_invertible_binary_matrix(n, beta='Bravyi-Kitaev')
    got = generalized_transformation(hop, beta, n_qubits=n)
    expected = bravyi_kitaev(hop, n_qubits=n)
    _assert_operators_equal(got, expected, n)


# ---------------------------------------------------------------------------
# 5. JW number operator
# ---------------------------------------------------------------------------

def test_jw_number_operator():
    """n_0 = a†_0 a_0 via JW matches OpenFermion."""
    n = 4
    num = FermionOperator('0^ 0')
    beta = random_invertible_binary_matrix(n, beta='Jordan-Wigner')
    got = generalized_transformation(num, beta, n_qubits=n)
    expected = jordan_wigner(num)
    _assert_operators_equal(got, expected, n)


# ---------------------------------------------------------------------------
# 6-7. H2 Hamiltonian eigenvalues
# ---------------------------------------------------------------------------

def test_jw_h2_hamiltonian_eigenvalues():
    """Full H2 eigenspectrum via JW matches OpenFermion jordan_wigner."""
    h_ferm = _h2_fermion_hamiltonian()
    n = count_qubits(h_ferm)
    beta = random_invertible_binary_matrix(n, beta='Jordan-Wigner')

    got_op = generalized_transformation(h_ferm, beta, n_qubits=n)
    expected_op = jordan_wigner(h_ferm)

    got_eigs = np.sort(eigenspectrum(got_op))
    expected_eigs = np.sort(eigenspectrum(expected_op))

    np.testing.assert_allclose(got_eigs, expected_eigs, atol=1e-10,
                               err_msg="JW H2 eigenvalue mismatch")


def test_bk_h2_hamiltonian_eigenvalues():
    """Full H2 eigenspectrum via BK matches OpenFermion bravyi_kitaev."""
    h_ferm = _h2_fermion_hamiltonian()
    n = count_qubits(h_ferm)
    beta = random_invertible_binary_matrix(n, beta='Bravyi-Kitaev')

    got_op = generalized_transformation(h_ferm, beta, n_qubits=n)
    expected_op = bravyi_kitaev(h_ferm, n_qubits=n)

    got_eigs = np.sort(eigenspectrum(got_op))
    expected_eigs = np.sort(eigenspectrum(expected_op))

    np.testing.assert_allclose(got_eigs, expected_eigs, atol=1e-10,
                               err_msg="BK H2 eigenvalue mismatch")


# ---------------------------------------------------------------------------
# 8. BK non-power-of-2 qubit counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 5, 6, 7, 9, 10])
def test_bk_non_power_of_2(n):
    """BK transform for non-power-of-2 n matches OpenFermion for a†_0 a_1 + h.c."""
    hop = FermionOperator('0^ 1') + FermionOperator('1^ 0')
    beta = random_invertible_binary_matrix(n, beta='Bravyi-Kitaev')
    got = generalized_transformation(hop, beta, n_qubits=n)
    expected = bravyi_kitaev(hop, n_qubits=n)
    _assert_operators_equal(got, expected, n)


# ---------------------------------------------------------------------------
# 9. BK matrix shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", list(range(1, 21)))
def test_bk_matrix_shape(n):
    """_bravyi_kitaev_sub_matrix(n) has shape (n, n)."""
    mat = _bravyi_kitaev_sub_matrix(n)
    assert mat.shape == (n, n), f"Expected ({n}, {n}), got {mat.shape}"


# ---------------------------------------------------------------------------
# 10. BK matrix invertible over GF(2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", list(range(1, 21)))
def test_bk_matrix_invertible(n):
    """BK matrix has odd determinant (invertible over GF(2))."""
    mat = _bravyi_kitaev_sub_matrix(n)
    det = int(round(float(Matrix(mat).det())))
    assert det % 2 != 0, f"BK matrix for n={n} is singular over GF(2) (det={det})"


# ---------------------------------------------------------------------------
# 11. JW beta is identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 4, 8, 10])
def test_jw_is_identity(n):
    """_select_beta('Jordan-Wigner', n) returns the identity matrix."""
    beta = _select_beta('Jordan-Wigner', n)
    np.testing.assert_array_equal(beta, np.eye(n, dtype=np.int8))


# ---------------------------------------------------------------------------
# 12. Product state JW is identity mapping
# ---------------------------------------------------------------------------

def test_product_state_jw():
    """generalized_transformation_product_state with JW leaves state unchanged."""
    state = np.array([1, 1, 0, 0], dtype=np.int8)
    beta = random_invertible_binary_matrix(4, beta='Jordan-Wigner')
    result = generalized_transformation_product_state(state, beta)
    np.testing.assert_array_equal(result, state)


# ---------------------------------------------------------------------------
# 13. Product state BK matches beta @ state mod 2
# ---------------------------------------------------------------------------

def test_product_state_bk():
    """BK product state equals beta @ state mod 2."""
    state = np.array([1, 1, 0, 0], dtype=np.int8)
    n = len(state)
    beta = random_invertible_binary_matrix(n, beta='Bravyi-Kitaev')
    result = generalized_transformation_product_state(state, beta)
    expected = np.mod(beta @ state, 2)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 14. Anticommutation relations: {a_i, a†_j} = delta_ij
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transform_name", ['Jordan-Wigner', 'Bravyi-Kitaev'])
def test_anticommutation_relations(transform_name):
    """Verify {a_i, a†_j} = delta_ij for all i, j after transformation."""
    n = 4
    beta = random_invertible_binary_matrix(n, beta=transform_name)

    # Transform all ladder operators
    creation_ops = []
    annihilation_ops = []
    for j in range(n):
        creation_ops.append(
            generalized_transformation(FermionOperator(f'{j}^'), beta, n_qubits=n)
        )
        annihilation_ops.append(
            generalized_transformation(FermionOperator(f'{j}'), beta, n_qubits=n)
        )

    dim = 2 ** n
    for i in range(n):
        ai_mat = _qubit_op_to_matrix(annihilation_ops[i], n)
        for j in range(n):
            adj_mat = _qubit_op_to_matrix(creation_ops[j], n)
            anticomm = ai_mat @ adj_mat + adj_mat @ ai_mat

            if i == j:
                np.testing.assert_allclose(
                    anticomm, np.eye(dim), atol=1e-12,
                    err_msg=f"{transform_name}: {{a_{i}, a†_{j}}} != I"
                )
            else:
                np.testing.assert_allclose(
                    anticomm, np.zeros((dim, dim)), atol=1e-12,
                    err_msg=f"{transform_name}: {{a_{i}, a†_{j}}} != 0"
                )


# ---------------------------------------------------------------------------
# 15. Random invertible matrix is actually invertible over GF(2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_random_matrix_invertible(n):
    """random_invertible_binary_matrix produces GF(2)-invertible matrices."""
    for _ in range(3):  # Test a few random samples
        mat = random_invertible_binary_matrix(n)
        det = int(round(float(Matrix(mat).det())))
        assert det % 2 != 0, f"Random matrix n={n} not invertible (det={det})"
        assert mat.shape == (n, n)
        assert set(np.unique(mat)).issubset({0, 1})
