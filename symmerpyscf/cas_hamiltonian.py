"""CAS (Complete Active Space) qubit Hamiltonian generation."""

import numpy as np
from pyscf import gto, scf, fci, mcscf, mp, ao2mo
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import get_fermion_operator
import openfermion as of
from symmer import PauliwordOp

from .molecule import _convert_fci_state


def generate_cas_qubit_hamiltonian(
    geometry,
    basis,
    ncas,
    nelecas,
    use_mp2_natorbs=True,
):
    """Generate a qubit Hamiltonian for a Complete Active Space.

    Runs PySCF RHF -> (optional) MP2 natural orbitals -> CASCI, then extracts
    the effective CAS integrals and converts to a Jordan-Wigner qubit
    Hamiltonian via openfermion.

    Parameters
    ----------
    geometry : list[tuple[str, tuple[float, float, float]]]
        Molecular geometry as [(atom, (x, y, z)), ...].
    basis : str
        Basis set name (e.g., "sto-3g", "6-31g").
    ncas : int
        Number of active spatial orbitals.
    nelecas : int or tuple[int, int]
        Number of active electrons. If int, split evenly for singlet.
        If tuple, (n_alpha, n_beta).
    use_mp2_natorbs : bool, optional
        Use MP2 natural orbitals as the CAS orbital basis (default True).

    Returns
    -------
    dict
        H_cas : PauliwordOp
            Qubit Hamiltonian on 2*ncas qubits (does NOT include e_core).
        H_fermion : FermionOperator
            Fermionic Hamiltonian in the active space.
        e_core : float
            Frozen-core energy. Add to any eigenvalue of H_cas to get total energy.
        cas_ground_state : QuantumState
            CASCI ground state on 2*ncas qubits.
        e_casci : float
            CASCI total energy (= smallest eigenvalue of H_cas + e_core).
        e_fci : float
            Full-space FCI energy (for reference/validation).
        e_hf : float
            Hartree-Fock energy (for reference/validation).
        n_qubits : int
            Number of qubits (= 2 * ncas).
        ncas : int
            Number of active spatial orbitals.
        nelecas : tuple[int, int]
            Active electrons as (n_alpha, n_beta).

    Raises
    ------
    RuntimeError
        If RHF does not converge.
    """
    # Step 1: Build molecule and run RHF
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.symmetry = False
    mol.verbose = 0
    mol.unit = "Angstrom"
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    if not mf.converged:
        raise RuntimeError("RHF did not converge.")
    e_hf = float(mf.e_tot)

    # Step 2: Full-space FCI for reference
    fci_solver = fci.FCI(mf)
    fci_solver.verbose = 0
    e_fci = float(fci_solver.kernel()[0])

    # Step 3: CASCI with optional MP2 natural orbitals
    mc = mcscf.CASCI(mf, ncas, nelecas)
    if use_mp2_natorbs:
        mp2_obj = mp.MP2(mf)
        mp2_obj.verbose = 0
        mp2_obj.run()
        _, natorbs = mcscf.addons.make_natural_orbitals(mp2_obj)
        mc.mo_coeff = natorbs
    mc.verbose = 0
    e_casci, _, ci_vec, _, _ = mc.kernel()
    e_casci = float(e_casci)

    # Step 4: Extract CAS effective integrals
    h1eff, e_core = mc.get_h1eff()
    e_core = float(e_core)
    h2eff_compact = mc.get_h2eff()
    h2eff_4d = ao2mo.restore(1, h2eff_compact, ncas)

    # Step 5: Convert integral convention (PySCF chemist -> openfermion)
    h2_of = h2eff_4d.transpose(0, 2, 3, 1)

    # Step 6: Build InteractionOperator
    one_body_SO, two_body_SO = spinorb_from_spatial(h1eff, h2_of)
    interaction_op = InteractionOperator(
        constant=0.0,
        one_body_tensor=one_body_SO,
        two_body_tensor=0.5 * two_body_SO,
    )

    # Step 7: Convert to qubit Hamiltonian
    n_qubits = 2 * ncas
    fermion_op = get_fermion_operator(interaction_op)
    qubit_op = of.jordan_wigner(fermion_op)
    H_cas = PauliwordOp.from_openfermion(qubit_op, n_qubits=n_qubits)

    # Step 8: Convert CASCI ground state
    nelec_a, nelec_b = mc.nelecas
    cas_ground_state = _convert_fci_state(ci_vec, norb=ncas, n_alpha=nelec_a, n_beta=nelec_b)

    return {
        "H_cas": H_cas,
        "H_fermion": fermion_op,
        "e_core": e_core,
        "cas_ground_state": cas_ground_state,
        "e_casci": e_casci,
        "e_fci": e_fci,
        "e_hf": e_hf,
        "n_qubits": n_qubits,
        "ncas": ncas,
        "nelecas": (nelec_a, nelec_b),
    }
