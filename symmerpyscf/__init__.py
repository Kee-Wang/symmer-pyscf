"""
symmer-pyscf: PySCF integration for Symmer quantum chemistry package

This package provides seamless integration between PySCF and Symmer,
enabling quantum chemistry calculations with generalized fermionic transformations.
"""

__version__ = "0.1.0"

from .molecule import generate_symmer_data, initialize_molecule, get_geometry
from .transforms import (
    generalized_transformation,
    generalized_transformation_product_state,
    generalized_transformation_symmer_jw_state,
    random_invertible_binary_matrix,
    bravyi_kitaev_single_perturbations,
)
from .cas_hamiltonian import generate_cas_qubit_hamiltonian
from .contextual import mol_info_to_H_cs
from .utils import symmer_to_dict

__all__ = [
    "generate_symmer_data",
    "initialize_molecule",
    "generalized_transformation",
    "get_geometry",
    "generalized_transformation_product_state",
    "generalized_transformation_symmer_jw_state",
    "random_invertible_binary_matrix",
    "bravyi_kitaev_single_perturbations",
    "generate_cas_qubit_hamiltonian",
    "mol_info_to_H_cs",
    "symmer_to_dict",
]
