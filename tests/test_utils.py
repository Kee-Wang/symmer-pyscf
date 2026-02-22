"""Tests for Hilbert-Schmidt utility functions in symmerpyscf.utils."""

import numpy as np
import scipy.sparse as sp
import pytest
from symmer import PauliwordOp

from symmerpyscf.utils import (
    hs_inner_product,
    hs_norm,
    hs_fidelity,
    hs_infidelity,
    hs_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pauli_matrices():
    """Return the four 2x2 Pauli matrices as dense arrays."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


# ---------------------------------------------------------------------------
# hs_inner_product
# ---------------------------------------------------------------------------

class TestHSInnerProduct:
    def test_identity_2x2(self):
        I = np.eye(2, dtype=complex)
        # Tr(I† I) = Tr(I) = 2
        assert hs_inner_product(I, I) == pytest.approx(2.0)

    def test_pauli_orthogonality(self):
        """Pauli matrices are orthogonal under HS inner product: Tr(σ_i† σ_j) = 2 δ_ij."""
        paulis = _pauli_matrices()
        for i, Pi in enumerate(paulis):
            for j, Pj in enumerate(paulis):
                expected = 2.0 if i == j else 0.0
                assert hs_inner_product(Pi, Pj) == pytest.approx(expected, abs=1e-12)

    def test_sparse_input(self):
        A = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=complex))
        B = sp.csr_matrix(np.array([[5, 6], [7, 8]], dtype=complex))
        # Tr(A† B) = 1*5 + 3*7 + 2*6 + 4*8 = 5 + 21 + 12 + 32 = 70
        # A† = [[1,3],[2,4]]*, here real so conj is identity
        # (A†B)_00 = 1*5 + 3*7 = 26,  (A†B)_11 = 2*6 + 4*8 = 44
        # Tr = 26 + 44 = 70
        assert hs_inner_product(A, B) == pytest.approx(70.0)

    def test_complex_matrix(self):
        A = np.array([[1, 1j], [0, 1]], dtype=complex)
        # Tr(A† A) = |1|^2 + |0|^2 + |1j|^2 + |1|^2 = 1 + 0 + 1 + 1 = 3
        assert hs_inner_product(A, A) == pytest.approx(3.0)

    def test_pauliwordop_input(self):
        op = PauliwordOp.from_dictionary({'II': 1.0})
        # II is 4x4 identity; Tr(I† I) = Tr(I_4) = 4
        assert hs_inner_product(op, op) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# hs_norm
# ---------------------------------------------------------------------------

class TestHSNorm:
    def test_identity(self):
        I = np.eye(3, dtype=complex)
        assert hs_norm(I) == pytest.approx(np.sqrt(3.0))

    def test_pauli_norm(self):
        _, X, _, _ = _pauli_matrices()
        # ||X||_HS = sqrt(Tr(X†X)) = sqrt(2)
        assert hs_norm(X) == pytest.approx(np.sqrt(2.0))

    def test_zero_matrix(self):
        Z = np.zeros((2, 2), dtype=complex)
        assert hs_norm(Z) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# hs_fidelity
# ---------------------------------------------------------------------------

class TestHSFidelity:
    def test_identical_matrices(self):
        A = np.array([[1, 2], [3, 4]], dtype=complex)
        assert hs_fidelity(A, A) == pytest.approx(1.0)

    def test_scalar_multiple(self):
        A = np.array([[1, 0], [0, -1]], dtype=complex)  # Z
        B = 3.0 * A
        assert hs_fidelity(A, B) == pytest.approx(1.0)

    def test_orthogonal_operators(self):
        _, X, _, Z = _pauli_matrices()
        assert hs_fidelity(X, Z) == pytest.approx(0.0)

    def test_zero_operator(self):
        A = np.zeros((2, 2), dtype=complex)
        B = np.eye(2, dtype=complex)
        assert hs_fidelity(A, B) == pytest.approx(0.0)
        assert hs_fidelity(B, A) == pytest.approx(0.0)
        assert hs_fidelity(A, A) == pytest.approx(0.0)

    def test_range_0_to_1(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        B = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        f = hs_fidelity(A, B)
        assert 0.0 <= f <= 1.0

    def test_pauliwordop_self_fidelity(self):
        op = PauliwordOp.from_dictionary({'XX': 1.0, 'ZZ': 0.5})
        assert hs_fidelity(op, op) == pytest.approx(1.0)

    def test_mixed_types(self):
        """PauliwordOp vs sparse matrix of the same operator should give fidelity 1."""
        op = PauliwordOp.from_dictionary({'XY': 1.0, 'ZI': 0.3})
        sparse_mat = op.to_sparse_matrix
        assert hs_fidelity(op, sparse_mat) == pytest.approx(1.0)

    def test_4x4_hand_computed(self):
        """Verify with a hand-computable 4x4 case.

        A = diag(1, 1, 0, 0),  B = diag(1, 0, 1, 0)
        Tr(A†B) = 1, Tr(A†A) = 2, Tr(B†B) = 2
        Fidelity = |1|² / (2·2) = 0.25
        """
        A = np.diag([1.0, 1, 0, 0]).astype(complex)
        B = np.diag([1.0, 0, 1, 0]).astype(complex)
        assert hs_fidelity(A, B) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# hs_distance
# ---------------------------------------------------------------------------

class TestHSDistance:
    def test_same_operator(self):
        A = np.array([[1, 2], [3, 4]], dtype=complex)
        assert hs_distance(A, A) == pytest.approx(0.0)

    def test_known_distance(self):
        """||I - Z||_HS for 2x2: I-Z = diag(0,2), norm = 2."""
        I, _, _, Z = _pauli_matrices()
        assert hs_distance(I, Z) == pytest.approx(2.0)

    def test_sparse_distance(self):
        A = sp.eye(2, dtype=complex)
        B = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
        # same as I-Z above
        assert hs_distance(A, B) == pytest.approx(2.0)

    def test_triangle_inequality(self):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        B = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        C = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        assert hs_distance(A, C) <= hs_distance(A, B) + hs_distance(B, C) + 1e-12

    def test_sparse_complex_distance(self):
        """Sparse path must handle complex off-diagonal entries correctly."""
        A = sp.csr_matrix(np.array([[1, 1j], [0, 1]], dtype=complex))
        B = sp.csr_matrix(np.zeros((2, 2), dtype=complex))
        # ||A||_F = sqrt(|1|^2 + |1j|^2 + |0|^2 + |1|^2) = sqrt(3)
        assert hs_distance(A, B) == pytest.approx(np.sqrt(3.0))

    def test_pauliwordop_distance(self):
        op1 = PauliwordOp.from_dictionary({'ZZ': 1.0})
        op2 = PauliwordOp.from_dictionary({'ZZ': 1.0, 'XX': 0.5})
        # distance should be ||0.5 * XX||_HS = 0.5 * ||XX||_HS = 0.5 * sqrt(4) = 1.0
        # XX is 4x4 with 4 non-zero entries of magnitude 1, so ||XX||_HS = sqrt(4) = 2
        assert hs_distance(op1, op2) == pytest.approx(0.5 * 2.0)


# ---------------------------------------------------------------------------
# Cross-validation: Pauli coefficient path vs sparse matrix path
# ---------------------------------------------------------------------------

def _multi_term_ops():
    """Return two multi-term PauliwordOps for cross-validation."""
    A = PauliwordOp.from_dictionary({'XY': 0.3, 'ZI': -0.7, 'IX': 0.5j})
    B = PauliwordOp.from_dictionary({'ZI': 1.0, 'IX': 0.2, 'YY': -0.4})
    return A, B


class TestPauliFastPathCrossValidation:
    """Verify that the Pauli coefficient fast path matches the sparse matrix path."""

    def test_inner_product_matches_sparse(self):
        A, B = _multi_term_ops()
        pauli_result = hs_inner_product(A, B)
        sparse_result = hs_inner_product(A.to_sparse_matrix, B.to_sparse_matrix)
        assert pauli_result == pytest.approx(sparse_result, abs=1e-12)

    def test_norm_matches_sparse(self):
        A, _ = _multi_term_ops()
        pauli_result = hs_norm(A)
        sparse_result = hs_norm(A.to_sparse_matrix)
        assert pauli_result == pytest.approx(sparse_result, abs=1e-12)

    def test_fidelity_matches_sparse(self):
        A, B = _multi_term_ops()
        pauli_result = hs_fidelity(A, B)
        sparse_result = hs_fidelity(A.to_sparse_matrix, B.to_sparse_matrix)
        assert pauli_result == pytest.approx(sparse_result, abs=1e-12)

    def test_distance_matches_sparse(self):
        A, B = _multi_term_ops()
        pauli_result = hs_distance(A, B)
        sparse_result = hs_distance(A.to_sparse_matrix, B.to_sparse_matrix)
        assert pauli_result == pytest.approx(sparse_result, abs=1e-12)

    def test_inner_product_self_matches_sparse(self):
        A, _ = _multi_term_ops()
        pauli_result = hs_inner_product(A, A)
        sparse_result = hs_inner_product(A.to_sparse_matrix, A.to_sparse_matrix)
        assert pauli_result == pytest.approx(sparse_result, abs=1e-12)

    def test_distance_disjoint_keys(self):
        """Operators with no common Pauli terms."""
        A = PauliwordOp.from_dictionary({'XI': 1.0})
        B = PauliwordOp.from_dictionary({'IZ': 1.0})
        pauli_result = hs_distance(A, B)
        sparse_result = hs_distance(A.to_sparse_matrix, B.to_sparse_matrix)
        assert pauli_result == pytest.approx(sparse_result, abs=1e-12)


# ---------------------------------------------------------------------------
# hs_infidelity
# ---------------------------------------------------------------------------

class TestHSInfidelity:
    def test_identical_operators(self):
        A = PauliwordOp.from_dictionary({'XX': 1.0, 'ZZ': 0.5})
        assert hs_infidelity(A, A) == pytest.approx(0.0)

    def test_orthogonal_operators(self):
        A = PauliwordOp.from_dictionary({'XI': 1.0})
        B = PauliwordOp.from_dictionary({'IZ': 1.0})
        assert hs_infidelity(A, B) == pytest.approx(1.0)

    def test_complement_of_fidelity(self):
        A, B = _multi_term_ops()
        assert hs_infidelity(A, B) == pytest.approx(1.0 - hs_fidelity(A, B))

    def test_zero_operator(self):
        A = PauliwordOp.from_dictionary({'II': 0.0})
        B = PauliwordOp.from_dictionary({'ZI': 1.0})
        # fidelity is 0 when either operator is zero → infidelity = 1
        assert hs_infidelity(A, B) == pytest.approx(1.0)

    def test_dense_matrix_input(self):
        A = np.array([[1, 2], [3, 4]], dtype=complex)
        B = 2.0 * A
        assert hs_infidelity(A, B) == pytest.approx(0.0)
