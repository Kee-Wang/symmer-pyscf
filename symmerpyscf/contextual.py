"""Contextual subspace VQE methods for molecular systems."""

from typing import Dict, Optional, Any
import numpy as np

from symmer import PauliwordOp
from symmer.projection import QubitTapering, ContextualSubspace
from symmer.utils import exact_gs_energy

from .transforms import (
    generalized_transformation,
    generalized_transformation_symmer_jw_state,
    random_invertible_binary_matrix
)


def mol_info_to_H_cs(
        mol_info: Dict[str, Any],
        n_cs_qubits: int,
        beta: Optional[np.ndarray] = None,
        manual_stabilizer: Optional[list] = None,
        use_ccsd_state: bool = False,
        noncontextual_strategy: str = 'SingleSweep_magnitude',
        unitary_partitioning_method: str = 'LCU'
) -> Dict[str, Any]:
    """
    Convert molecular information to contextual subspace representation.

    This function performs the complete workflow:
    1. Apply generalized fermionic transformation (beta)
    2. Perform qubit tapering using symmetries
    3. Project onto contextual subspace
    4. Compute ground state energy

    Args:
        mol_info: Dictionary containing molecular data with keys:
            - H_second_quantized: Fermionic Hamiltonian
            - fci_state: Full CI state
            - hf_state: Hartree-Fock state
            - ccsd_state: CCSD state
            - number_alpha: Alpha particle number operator
            - number_beta: Beta particle number operator
            - CCSD_generator: CCSD cluster operator
            - n_particles: Total number of particles
            - n_qubits_full: Total number of qubits
            - fci_energy: FCI energy for reference
        n_cs_qubits: Target number of qubits in contextual subspace
        beta: Transformation matrix (generated randomly if None)
        manual_stabilizer: Optional manual stabilizer specification
        use_ccsd_state: Use CCSD state instead of FCI for stabilizer selection
        noncontextual_strategy: Strategy for contextual subspace construction
        unitary_partitioning_method: Method for unitary partitioning

    Returns:
        Dictionary containing:
            - H_cs: Contextual subspace Hamiltonian
            - Na_CS: Alpha number operator in CS
            - Nb_CS: Beta number operator in CS
            - CCSD_generator_CS: CCSD generator in CS
            - hf_cs: Hartree-Fock state in CS
            - cs_energy: Ground state energy in CS
            - fci_energy: Reference FCI energy
            - beta: Transformation matrix used

    Example:
        >>> data_cs = mol_info_to_H_cs(mol_info, n_cs_qubits=2, beta=None)
        >>> print(f"CS Energy: {data_cs['cs_energy']:.6f}")
        >>> print(f"Error vs FCI: {data_cs['cs_energy'] - data_cs['fci_energy']:.6e}")
    """
    # Generate random invertible transformation if not provided
    if beta is None:
        beta = random_invertible_binary_matrix(n=mol_info['n_qubits_full'])

    # Extract molecular data
    H = mol_info['H_second_quantized']
    fci_state = mol_info['fci_state']
    hf_state = mol_info['hf_state']
    ccsd_state = mol_info['ccsd_state']
    number_alpha = mol_info['number_alpha']
    number_beta = mol_info['number_beta']
    CCSD_generator = mol_info['CCSD_generator']
    n_particles = mol_info['n_particles']

    # Apply generalized transformation to states
    hf_state = generalized_transformation_symmer_jw_state(
        jw_state=hf_state, beta=beta
    )
    fci_state = generalized_transformation_symmer_jw_state(
        jw_state=fci_state, beta=beta
    )
    ccsd_state = generalized_transformation_symmer_jw_state(
        jw_state=ccsd_state, beta=beta
    )

    # Apply generalized transformation to operators
    H = PauliwordOp.from_openfermion(
        generalized_transformation(H, beta=beta),
        n_qubits=beta.shape[0]
    )
    CCSD_generator = PauliwordOp.from_openfermion(
        generalized_transformation(CCSD_generator, beta=beta),
        n_qubits=beta.shape[0]
    )
    number_alpha = PauliwordOp.from_openfermion(
        generalized_transformation(number_alpha, beta=beta),
        n_qubits=beta.shape[0]
    )
    number_beta = PauliwordOp.from_openfermion(
        generalized_transformation(number_beta, beta=beta),
        n_qubits=beta.shape[0]
    )

    # Perform qubit tapering
    taper_obj = QubitTapering(H)
    H_tap = taper_obj.taper_it(ref_state=hf_state)

    N_alpha_tap = taper_obj.taper_it(aux_operator=number_alpha)
    N_beta_tap = taper_obj.taper_it(aux_operator=number_beta)
    ccsd_generator_tap = taper_obj.taper_it(aux_operator=CCSD_generator)

    # Project states onto tapered space
    tap_ccsd_state = taper_obj.project_state(ccsd_state)
    tap_fci_state = taper_obj.project_state(fci_state)

    # Setup contextual subspace VQE
    cs_vqe = ContextualSubspace(
        H_tap,
        noncontextual_strategy=noncontextual_strategy,
        unitary_partitioning_method=unitary_partitioning_method
    )

    # Apply manual stabilizers if provided
    if manual_stabilizer is not None:
        cs_vqe.manual_stabilizers(manual_stabilizer)

    # Update stabilizers based on reference state
    reference_state = tap_ccsd_state if use_ccsd_state else tap_fci_state
    cs_vqe.update_stabilizers(
        n_qubits=n_cs_qubits,
        strategy='aux_preserving',
        aux_operator=reference_state.state_op
    )

    # Project onto contextual subspace
    H_cs = cs_vqe.project_onto_subspace()
    Na_CS = cs_vqe.project_onto_subspace(N_alpha_tap)
    Nb_CS = cs_vqe.project_onto_subspace(N_beta_tap)
    CCSD_generator_CS = cs_vqe.project_onto_subspace(ccsd_generator_tap)

    # Compute ground state energy in contextual subspace
    cs_energy, cs_state = exact_gs_energy(
        H_cs.to_sparse_matrix.real,
        n_particles=n_particles,
        number_operator=(Na_CS + Nb_CS)
    )

    # Prepare HF state in CS (normalized and positive coefficients)
    hf_cs = cs_state.sort()[0].normalize
    hf_cs.state_op.coeff_vec = np.abs(hf_cs.state_op.coeff_vec)

    return {
        'H_cs': H_cs,
        'Na_CS': Na_CS,
        'Nb_CS': Nb_CS,
        'CCSD_generator_CS': CCSD_generator_CS,
        'hf_cs': hf_cs,
        'cs_energy': cs_energy,
        'fci_energy': mol_info['fci_energy'],
        'beta': beta,
        'n_terms_hamiltonian': H_cs.n_terms,
        'n_terms_ccsd_generator': CCSD_generator_CS.n_terms,
        'n_qubits_cs': H_cs.n_qubits
    }
