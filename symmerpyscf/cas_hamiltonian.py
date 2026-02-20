"""CAS (Complete Active Space) qubit Hamiltonian generation."""

import json
from pyscf import gto, scf, fci, mcscf, mp, ao2mo
from pyscf.fci import spin_op as fci_spin_op
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops import FermionOperator
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import get_fermion_operator
import openfermion as of
from symmer import PauliwordOp

from .molecule import _convert_fci_state
from .utils import symmer_to_dict


def _generate_cas_auxiliary_operators(n_qubits):
    """Generate number and spin operators for CAS active space.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= 2 * ncas).

    Returns
    -------
    dict
        Symmer PauliwordOp and openfermion FermionOperator forms of
        number operators (total, alpha, beta) and S^2.
    """
    of_number_operator = FermionOperator()
    of_number_operator_alpha = FermionOperator()
    of_number_operator_beta = FermionOperator()

    for i in range(n_qubits):
        if i % 2 == 0:
            of_number_operator_alpha += of.number_operator(n_qubits, mode=i)
        else:
            of_number_operator_beta += of.number_operator(n_qubits, mode=i)
        of_number_operator += of.number_operator(n_qubits, mode=i)

    symmer_number_operator = PauliwordOp.from_openfermion(
        of.jordan_wigner(of_number_operator), n_qubits=n_qubits
    )
    symmer_number_operator_alpha = PauliwordOp.from_openfermion(
        of.jordan_wigner(of_number_operator_alpha), n_qubits=n_qubits
    )
    symmer_number_operator_beta = PauliwordOp.from_openfermion(
        of.jordan_wigner(of_number_operator_beta), n_qubits=n_qubits
    )

    of_s2_operator = of.hamiltonians.s_squared_operator(n_qubits // 2)
    symmer_s2 = PauliwordOp.from_openfermion(
        of.jordan_wigner(of_s2_operator), n_qubits=n_qubits
    )

    return {
        'number_operator': symmer_number_operator,
        'number_operator_of': of_number_operator,
        'N_alpha': symmer_number_operator_alpha,
        'N_alpha_of': of_number_operator_alpha,
        'N_beta': symmer_number_operator_beta,
        'N_beta_of': of_number_operator_beta,
        'S2': symmer_s2,
        'S2_of': of_s2_operator,
    }


def generate_cas_qubit_hamiltonian(
    geometry,
    basis,
    ncas,
    nelecas,
    multiplicity=1,
    charge=0,
    use_mp2_natorbs=True,
    save_file=None,
):
    """Generate a qubit Hamiltonian for a Complete Active Space.

    Runs PySCF RHF -> (optional) MP2 natural orbitals -> CASCI, then extracts
    the effective CAS integrals and converts to a Jordan-Wigner qubit
    Hamiltonian via openfermion.  The CASCI solver is spin-constrained via
    ``fix_spin_()`` to ensure the correct spin sector is targeted.

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
    multiplicity : int, optional
        Spin multiplicity 2S+1 (default 1 = singlet). Used to constrain
        the CASCI solver to the correct spin sector via fix_spin_().
    charge : int, optional
        Molecular charge (default 0).
    use_mp2_natorbs : bool, optional
        Use MP2 natural orbitals as the CAS orbital basis (default True).
    save_file : str, optional
        Path to save the symmer_data dict as JSON (default None).

    Returns
    -------
    tuple[dict, dict]
        cas_result : dict
            H_cas : PauliwordOp
                Qubit Hamiltonian on 2*ncas qubits (includes e_core).
            H_fermion : FermionOperator
                Fermionic Hamiltonian in the active space.
            e_core : float
                Frozen-core energy (already included in H_cas).
            cas_ground_state : QuantumState
                CASCI ground state on 2*ncas qubits.
            e_casci : float
                CASCI total energy (= smallest eigenvalue of H_cas).
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
        symmer_data : dict
            Symmer-compatible JSON schema dictionary with Hamiltonian,
            auxiliary operators, HF state, and CAS metadata.

    Raises
    ------
    RuntimeError
        If RHF does not converge.
    """
    # Step 1: Build molecule and run RHF
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1
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

    # Constrain CASCI to the correct spin sector
    s = (multiplicity - 1) / 2.0
    target_ss = s * (s + 1)
    fci.addons.fix_spin_(mc.fcisolver, shift=0.2, ss=target_ss)

    e_casci, _, ci_vec, _, _ = mc.kernel()
    e_casci = float(e_casci)

    # Verify <S^2> of the CASCI solution
    nelec_a, nelec_b = mc.nelecas
    ss_val, mult_val = fci_spin_op.spin_square(ci_vec, ncas, (nelec_a, nelec_b))
    cas_spin_squared = float(ss_val)
    cas_multiplicity = float(mult_val)
    spin_ok = abs(ss_val - target_ss) < 0.1

    if not spin_ok:
        # Retry with stronger penalty
        print(f'WARNING [AUDIT]: CASCI spin verification failed: <S^2>={ss_val:.4f} '
              f'(expected {target_ss:.1f}). Retrying with shift=1.0.')
        mc2 = mcscf.CASCI(mf, ncas, nelecas)
        if use_mp2_natorbs:
            mc2.mo_coeff = natorbs
        mc2.verbose = 0
        fci.addons.fix_spin_(mc2.fcisolver, shift=1.0, ss=target_ss)
        e_casci2, _, ci_vec2, _, _ = mc2.kernel()
        ss_val2, mult_val2 = fci_spin_op.spin_square(ci_vec2, ncas, (nelec_a, nelec_b))

        if abs(ss_val2 - target_ss) < 0.1:
            e_casci = float(e_casci2)
            ci_vec = ci_vec2
            cas_spin_squared = float(ss_val2)
            cas_multiplicity = float(mult_val2)
        else:
            print(f'WARNING [AUDIT]: CASCI spin constraint failed even with shift=1.0: '
                  f'<S^2>={ss_val2:.4f}. Using best result.')

    # Step 4: Extract CAS effective integrals
    h1eff, e_core = mc.get_h1eff()
    e_core = float(e_core)
    h2eff_compact = mc.get_h2eff()
    h2eff_4d = ao2mo.restore(1, h2eff_compact, ncas)

    # Step 5: Convert integral convention (PySCF chemist -> openfermion)
    h2_of = h2eff_4d.transpose(0, 2, 3, 1)

    # Step 6: Build InteractionOperator (include e_core so eigenvalues = total energy)
    one_body_SO, two_body_SO = spinorb_from_spatial(h1eff, h2_of)
    interaction_op = InteractionOperator(
        constant=e_core,
        one_body_tensor=one_body_SO,
        two_body_tensor=0.5 * two_body_SO,
    )

    # Step 7: Convert to qubit Hamiltonian
    n_qubits = 2 * ncas
    fermion_op = get_fermion_operator(interaction_op)
    qubit_op = of.jordan_wigner(fermion_op)
    H_cas = PauliwordOp.from_openfermion(qubit_op, n_qubits=n_qubits)

    # Step 8: Convert CASCI ground state (uses spin-verified ci_vec)
    cas_ground_state = _convert_fci_state(ci_vec, norb=ncas, n_alpha=nelec_a, n_beta=nelec_b)

    # Step 9: Build CAS HF state (interleaved alpha/beta convention)
    nelecas_total = nelec_a + nelec_b
    hf_array = [0] * n_qubits
    hf_array[0:nelecas_total] = [1] * nelecas_total

    # Step 10: Build auxiliary operators for CAS space
    operators = _generate_cas_auxiliary_operators(n_qubits)

    # Step 11: Build symmer-compatible data dict
    symmer_data = {
        "H": symmer_to_dict(H_cas),
        "H_second_quantized": str(fermion_op),
        "qubit_encoding": "JW",
        "geometry": [[atom, *coords] for atom, coords in geometry],
        "basis": basis,
        "charge": mol.charge,
        "spin": mol.spin,
        "hf_array": hf_array,
        "hf_state": {"".join(str(b) for b in hf_array): [1.0, 0.0]},
        "n_particles": {
            "total": nelecas_total,
            "alpha": nelec_a,
            "beta": nelec_b,
        },
        "n_qubits": n_qubits,
        "calculated_properties": {
            "HF": {"energy": e_hf, "converged": True},
            "FCI": {"energy": e_fci, "converged": True},
            "CASCI": {
                "energy": e_casci,
                "converged": True,
                "spin_squared": cas_spin_squared,
                "multiplicity": cas_multiplicity,
                "spin_constrained": True,
            },
        },
        "auxiliary_operators": {
            "number_operator": symmer_to_dict(operators['number_operator']),
            "N_alpha": symmer_to_dict(operators['N_alpha']),
            "N_beta": symmer_to_dict(operators['N_beta']),
            "S^2_operator": symmer_to_dict(operators['S2']),
            "fci_state": symmer_to_dict(cas_ground_state),
            "number_operator_second_quantized": str(operators['number_operator_of']),
            "N_alpha_second_quantized": str(operators['N_alpha_of']),
            "N_beta_second_quantized": str(operators['N_beta_of']),
            "S^2_operator_second_quantized": str(operators['S2_of']),
        },
        "cas_metadata": {
            "ncas": ncas,
            "nelecas": [nelec_a, nelec_b],
            "e_core": e_core,
            "use_mp2_natorbs": use_mp2_natorbs,
        },
    }

    if save_file is not None:
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(symmer_data, f, indent=4)

    cas_result = {
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

    return cas_result, symmer_data
