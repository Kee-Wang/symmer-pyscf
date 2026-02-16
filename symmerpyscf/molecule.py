"""Core molecular data generation and initialization functions."""

import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from openfermion.chem import MolecularData
from openfermion.ops import FermionOperator
from openfermion.transforms import get_fermion_operator
from openfermionpyscf import PyscfMolecularData
from openfermionpyscf._run_pyscf import compute_scf, compute_integrals

from pyscf import gto, fci, mp, cc
import pyscf.ci
from pyscf.fci import cistring

import pennylane as qml
import openfermion as of
from symmer import QuantumState, PauliwordOp

from .utils import symmer_to_dict, t1_t2_to_fermionic_operator


def generate_symmer_data(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    save_file: Optional[str] = None,
    symmetry_subgroup: Optional[str] = None,
    symmetry: bool = True,
    verbose: bool = False,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate Symmer-compatible quantum chemistry data from molecular geometry.

    Args:
        geometry: List of (atom, (x, y, z)) tuples defining molecular structure
        save_file: Optional path to save JSON output
        symmetry_subgroup: Optional symmetry subgroup for PySCF
        symmetry: Enable symmetry in calculations
        verbose: Print detailed output
        basis: Basis set (default: "sto-3g")
        charge: Molecular charge
        multiplicity: Spin multiplicity

    Returns:
        mol_info: Dictionary with key molecular data for further processing
        symmer_data: Complete dataset in Symmer format

    Example:
        >>> geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))]
        >>> mol_info, data = generate_symmer_data(geometry, basis="sto-3g")
    """
    # Initialize molecule
    description = ''
    molecule = MolecularData(geometry, basis, multiplicity, charge, description)

    # Setup PySCF molecule
    pyscf_molecule = gto.Mole()
    pyscf_molecule.atom = molecule.geometry
    pyscf_molecule.basis = molecule.basis
    pyscf_molecule.charge = molecule.charge
    pyscf_molecule.spin = molecule.multiplicity - 1
    pyscf_molecule.symmetry = symmetry
    pyscf_molecule.symmetry_subgroup = symmetry_subgroup
    pyscf_molecule.verbose = 0
    pyscf_molecule.unit = 'Angstrom'
    pyscf_molecule.conv_tol = 1e-6
    pyscf_molecule.build()

    # Basic molecular properties
    molecule.n_orbitals = int(pyscf_molecule.nao_nr())
    molecule.n_qubits = 2 * molecule.n_orbitals
    molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())

    # Run SCF
    pyscf_scf = compute_scf(pyscf_molecule)
    pyscf_scf.conv_tol = 1e-6
    mf = pyscf_scf.run()
    molecule.hf_energy = float(pyscf_scf.e_tot)
    molecule.n_electrons = pyscf_molecule.nelec[0] + pyscf_molecule.nelec[1]

    if verbose:
        print(f'Hartree-Fock energy for {molecule.name} '
              f'({molecule.n_electrons} electrons) is {molecule.hf_energy}')

    # Store PySCF data
    molecule._pyscf_data = pyscf_data = {}
    pyscf_data['mol'] = pyscf_molecule
    pyscf_data['scf'] = pyscf_scf

    # Run post-HF methods
    pyscf_mp2 = _run_mp2(pyscf_scf, molecule, pyscf_data, verbose, multiplicity)
    pyscf_ccsd = _run_ccsd(pyscf_scf, molecule, pyscf_data, verbose)
    pyscf_cisd = _run_cisd(pyscf_scf, molecule, pyscf_data, verbose)
    pyscf_fci, fcivec = _run_fci(pyscf_molecule, pyscf_scf, molecule, pyscf_data, verbose)

    # Get integrals and orbital data
    molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
    molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)
    one_body_integrals, two_body_integrals = compute_integrals(pyscf_molecule, pyscf_scf)
    molecule.one_body_integrals = one_body_integrals
    molecule.two_body_integrals = two_body_integrals
    molecule.overlap_integrals = pyscf_scf.get_ovlp()

    # Convert FCI state to Symmer format
    norb = mf.mo_coeff.shape[1]
    n_alpha, n_beta = pyscf_molecule.nelec
    qml_fci_state = _convert_fci_state(fcivec, norb, n_alpha, n_beta)

    # Generate CCSD operator in second quantization
    t1 = pyscf.cc.addons.spatial2spin(pyscf_ccsd.t1)
    t2 = pyscf.cc.addons.spatial2spin(pyscf_ccsd.t2)
    ccsd_2nd = t1_t2_to_fermionic_operator(
        t1, t2,
        pyscf_molecule.nelec[0] * 2,
        (pyscf_molecule.nao_nr() - pyscf_molecule.nelec[0]) * 2
    )
    symmer_ccsd_generator = PauliwordOp.from_openfermion(of.jordan_wigner(ccsd_2nd))

    # Generate auxiliary operators
    operators = _generate_auxiliary_operators(molecule)

    # Generate molecular Hamiltonian
    pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
    pyscf_molecular_data.__dict__.update(molecule.__dict__)
    second_quantized_ham = get_fermion_operator(pyscf_molecular_data.get_molecular_hamiltonian())
    symmer_ham = PauliwordOp.from_openfermion(of.jordan_wigner(second_quantized_ham))

    # Import states
    qml_ccsd_state = qml.qchem.import_state(pyscf_ccsd).reshape(-1, 1)
    symmer_ccsd_state = QuantumState.from_array(qml_ccsd_state)

    qml_cisd_state = qml.qchem.import_state(pyscf_cisd).reshape(-1, 1)
    symmer_cisd_state = QuantumState.from_array(qml_cisd_state)

    # HF state
    hf_state = [0] * molecule.n_qubits
    hf_state[0:molecule.n_electrons] = [1] * molecule.n_electrons

    # Compile symmer_data
    symmer_data = _compile_symmer_data(
        molecule, pyscf_molecule, pyscf_scf, pyscf_mp2, pyscf_cisd,
        pyscf_ccsd, pyscf_fci, symmer_ham, second_quantized_ham,
        hf_state, operators, symmer_ccsd_generator, ccsd_2nd,
        symmer_ccsd_state, symmer_cisd_state, qml_fci_state
    )

    # Save if requested
    if save_file is not None:
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(symmer_data, f, indent=4)

    # Prepare mol_info for workflow
    mol_info = {
        'H_second_quantized': of.FermionOperator(symmer_data['H_second_quantized']),
        'hf_state': QuantumState.from_dictionary(symmer_data['hf_state']),
        'fci_state': QuantumState.from_dictionary(symmer_data['auxiliary_operators']['fci_state']),
        'ccsd_state': QuantumState.from_dictionary(symmer_data['auxiliary_operators']['ccsd_state']),
        'number_alpha': of.FermionOperator(symmer_data['auxiliary_operators']['N_alpha_second_quantized']),
        'number_beta': of.FermionOperator(symmer_data['auxiliary_operators']['N_beta_second_quantized']),
        'CCSD_generator': of.FermionOperator(symmer_data['auxiliary_operators']['CCSD_operator_second_quantized']),
        'n_qubits_full': symmer_data['n_qubits'],
        'n_particles': symmer_data['n_particles']['total'],
        'fci_energy': symmer_data['calculated_properties']['FCI']['energy']
    }

    return mol_info, symmer_data


def initialize_molecule(
    bondlength: Optional[float] = None,
    molecule: Optional[str] = None,
    geometry: Optional[List[Tuple[str, Tuple[float, float, float]]]] = None,
    basis: str = "sto-3g",
    charge: int = 0,
    outdir: Optional[str] = None,
    **kwargs
) -> Tuple[Dict, Dict, Dict, Optional[str]]:
    """
    Initialize a molecule with given bond length or custom geometry and generate all quantum data.

    Args:
        bondlength: Bond length in Angstroms (used with molecule parameter)
        molecule: Molecule type (e.g., "H2", "LiH", "HeH+")
        geometry: Custom geometry as list of (atom, (x, y, z)) tuples.
                  If provided, bondlength and molecule are ignored.
        basis: Basis set
        charge: Molecular charge
        outdir: Optional output directory for saving data
        **kwargs: Additional arguments passed to generate_symmer_data

    Returns:
        mol_info: Molecular information for processing
        pyscf_data: PySCF calculation metadata
        energy_data: Dictionary of energies from different methods
        filename: Path to saved JSON file (if outdir specified)

    Example:
        >>> # Using predefined molecule
        >>> mol_info, pyscf_data, energy_data, _ = initialize_molecule(
        ...     bondlength=0.74, molecule="H2"
        ... )

        >>> # Using custom geometry
        >>> geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))]
        >>> mol_info, pyscf_data, energy_data, _ = initialize_molecule(
        ...     geometry=geometry, basis="sto-3g"
        ... )

        >>> # Using helper function for geometry
        >>> from symmerpyscf.molecule import get_geometry
        >>> geometry = get_geometry("H2", 0.74)
        >>> mol_info, pyscf_data, energy_data, _ = initialize_molecule(
        ...     geometry=geometry
        ... )
    """
    # Determine geometry
    if geometry is None:
        if molecule is None or bondlength is None:
            raise ValueError(
                "Either provide 'geometry' directly, or both 'molecule' and 'bondlength'"
            )
        geometry = get_geometry(molecule, bondlength)
        mol_name = molecule
    else:
        # Use custom geometry
        mol_name = "molecule"
        if molecule is not None:
            mol_name = molecule
        if bondlength is None:
            bondlength = 0.0  # placeholder for filename

    # Setup output filename
    filename = None
    if outdir is not None:
        import os
        os.makedirs(outdir, exist_ok=True)
        filename = os.path.join(outdir, f"{mol_name}_{bondlength:.3f}_{basis}.json")

    # Generate data
    mol_info, symmer_data = generate_symmer_data(
        geometry=geometry,
        save_file=filename,
        basis=basis,
        charge=charge,
        **kwargs
    )

    # Extract PySCF metadata
    pyscf_data = {
        'n_qubits': symmer_data['n_qubits'],
        'n_particles': symmer_data['n_particles'],
        'point_group': symmer_data['point_group'],
        'hf_method': symmer_data['hf_method']
    }

    # Extract energy data
    energy_data = symmer_data['calculated_properties'].copy()

    return mol_info, pyscf_data, energy_data, filename


def get_geometry(molecule: str, bondlength: float) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Generate molecular geometry for common molecules."""
    geometries = {
        "H2": [
            ('H', (0.0, 0.0, 0.0)),
            ('H', (0.0, 0.0, bondlength))
        ],
        "LiH": [
            ('Li', (0.0, 0.0, 0.0)),
            ('H', (0.0, 0.0, bondlength))
        ],
        "HeH+": [
            ('He', (0.0, 0.0, 0.0)),
            ('H', (0.0, 0.0, bondlength))
        ],
    }

    if molecule not in geometries:
        raise ValueError(f"Molecule {molecule} not supported. Available: {list(geometries.keys())}")

    return geometries[molecule]


def _run_mp2(pyscf_scf, molecule, pyscf_data, verbose, multiplicity):
    """Run MP2 calculation."""
    if multiplicity != 1:
        if verbose:
            print("WARNING: RO-MP2 is not available in PySCF.")
        molecule.mp2_energy = None
        return None

    pyscf_mp2 = mp.MP2(pyscf_scf)
    pyscf_mp2.verbose = 0
    pyscf_mp2.run()
    molecule.mp2_energy = pyscf_scf.e_tot + pyscf_mp2.e_corr
    pyscf_data['mp2'] = pyscf_mp2

    if verbose:
        print(f'MP2 energy for {molecule.name} '
              f'({molecule.n_electrons} electrons) is {molecule.mp2_energy}')

    return pyscf_mp2


def _run_ccsd(pyscf_scf, molecule, pyscf_data, verbose):
    """Run CCSD calculation."""
    pyscf_ccsd = cc.CCSD(pyscf_scf)
    pyscf_ccsd.verbose = 0
    pyscf_ccsd.run()
    molecule.ccsd_energy = pyscf_ccsd.e_tot
    pyscf_data['ccsd'] = pyscf_ccsd

    if verbose:
        print(f'CCSD energy for {molecule.name} '
              f'({molecule.n_electrons} electrons) is {molecule.ccsd_energy}')

    return pyscf_ccsd


def _run_cisd(pyscf_scf, molecule, pyscf_data, verbose):
    """Run CISD calculation."""
    pyscf_cisd = pyscf.ci.CISD(pyscf_scf)
    pyscf_cisd.verbose = 0
    pyscf_cisd.run()
    molecule.cisd_energy = pyscf_cisd.e_tot
    pyscf_data['cisd'] = pyscf_cisd

    if verbose:
        print(f'CISD energy for {molecule.name} '
              f'({molecule.n_electrons} electrons) is {molecule.cisd_energy}')

    return pyscf_cisd


def _run_fci(pyscf_molecule, pyscf_scf, molecule, pyscf_data, verbose):
    """Run FCI calculation."""
    pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
    pyscf_fci.verbose = 0
    molecule.fci_energy, fcivec = pyscf_fci.kernel()
    pyscf_data['fci'] = pyscf_fci

    if verbose:
        print(f'FCI energy for {molecule.name} '
              f'({molecule.n_electrons} electrons) is {molecule.fci_energy}')

    return pyscf_fci, fcivec


def _convert_fci_state(fcivec, norb, n_alpha, n_beta):
    """Convert FCI state from PySCF to Symmer format."""
    alpha_strings = cistring.make_strings(range(norb), n_alpha)
    beta_strings = cistring.make_strings(range(norb), n_beta)

    fcimat_dict = {}
    for i, alpha in enumerate(alpha_strings):
        for j, beta in enumerate(beta_strings):
            fcimat_dict[(alpha, beta)] = fcivec[i, j]

    fcimat_dict_signed = qml.qchem.convert._sign_chem_to_phys(fcimat_dict, norb)
    qml_fci_state = qml.qchem.convert._wfdict_to_statevector(fcimat_dict_signed, norb)
    return QuantumState.from_array(qml_fci_state.reshape([-1, 1]))


def _generate_auxiliary_operators(molecule):
    """Generate number and spin operators."""
    of_number_operator = FermionOperator()
    of_number_operator_alpha = FermionOperator()
    of_number_operator_beta = FermionOperator()

    for i in range(molecule.n_qubits):
        if i % 2 == 0:
            of_number_operator_alpha += of.number_operator(molecule.n_qubits, mode=i)
        else:
            of_number_operator_beta += of.number_operator(molecule.n_qubits, mode=i)
        of_number_operator += of.number_operator(molecule.n_qubits, mode=i)

    symmer_number_operator = PauliwordOp.from_openfermion(
        of.jordan_wigner(of_number_operator), n_qubits=molecule.n_qubits
    )
    symmer_number_operator_alpha = PauliwordOp.from_openfermion(
        of.jordan_wigner(of_number_operator_alpha), n_qubits=molecule.n_qubits
    )
    symmer_number_operator_beta = PauliwordOp.from_openfermion(
        of.jordan_wigner(of_number_operator_beta), n_qubits=molecule.n_qubits
    )

    of_s2_operator = of.hamiltonians.s_squared_operator(molecule.n_qubits // 2)
    symmer_s2 = PauliwordOp.from_openfermion(of.jordan_wigner(of_s2_operator))

    return {
        'number_operator': symmer_number_operator,
        'number_operator_of': of_number_operator,
        'N_alpha': symmer_number_operator_alpha,
        'N_alpha_of': of_number_operator_alpha,
        'N_beta': symmer_number_operator_beta,
        'N_beta_of': of_number_operator_beta,
        'S2': symmer_s2,
        'S2_of': of_s2_operator
    }


def _compile_symmer_data(molecule, pyscf_molecule, pyscf_scf, pyscf_mp2, pyscf_cisd,
                         pyscf_ccsd, pyscf_fci, symmer_ham, second_quantized_ham,
                         hf_state, operators, symmer_ccsd_generator, ccsd_2nd,
                         symmer_ccsd_state, symmer_cisd_state, qml_fci_state):
    """Compile all data into Symmer format dictionary."""
    symmer_data = {}

    # Hamiltonian
    symmer_data['H'] = symmer_to_dict(symmer_ham)
    symmer_data['H_second_quantized'] = str(second_quantized_ham)
    symmer_data['qubit_encoding'] = "JW"

    # Molecular info
    symmer_data['unit'] = pyscf_molecule.unit
    try:
        symmer_data['geometry'] = [[atom[0], *atom[1]] for atom in molecule.geometry]
    except (TypeError, IndexError):
        symmer_data['geometry'] = molecule.geometry

    symmer_data['basis'] = molecule.basis
    symmer_data['charge'] = molecule.charge
    symmer_data['spin'] = pyscf_molecule.spin

    # States
    symmer_data['hf_array'] = hf_state
    symmer_data['hf_state'] = {"".join([str(i) for i in hf_state]): [1.0, 0.0]}

    # Metadata
    symmer_data['hf_method'] = f'{pyscf_scf.__module__}.{pyscf_scf.__class__.__name__}'
    symmer_data['n_particles'] = {
        "total": molecule.n_electrons,
        "alpha": pyscf_molecule.nelec[0],
        "beta": pyscf_molecule.nelec[1]
    }
    symmer_data['n_qubits'] = molecule.n_qubits
    symmer_data['convergence_threshold'] = pyscf_molecule.conv_tol
    symmer_data['point_group'] = {
        'groupname': pyscf_molecule.groupname,
        'topgroup': pyscf_molecule.topgroup
    }

    # Energies
    mp2_energy = molecule.mp2_energy if molecule.mp2_energy is not None else float('nan')
    symmer_data['calculated_properties'] = {
        "HF": {"energy": float(molecule.hf_energy), "converged": bool(pyscf_scf.converged)},
        "MP2": {"energy": float(mp2_energy), "converged": pyscf_mp2 is not None},
        "CISD": {"energy": float(molecule.cisd_energy), "converged": bool(pyscf_cisd.converged)},
        "CCSD": {"energy": float(molecule.ccsd_energy), "converged": bool(pyscf_ccsd.converged)},
        "FCI": {"energy": float(molecule.fci_energy), "converged": bool(pyscf_fci.converged)}
    }

    # Auxiliary operators
    symmer_data['auxiliary_operators'] = {}
    symmer_data['auxiliary_operators']['number_operator'] = symmer_to_dict(operators['number_operator'])
    symmer_data['auxiliary_operators']['N_alpha'] = symmer_to_dict(operators['N_alpha'])
    symmer_data['auxiliary_operators']['N_beta'] = symmer_to_dict(operators['N_beta'])
    symmer_data['auxiliary_operators']['S^2_operator'] = symmer_to_dict(operators['S2'])
    symmer_data['auxiliary_operators']['CCSD_operator'] = symmer_to_dict(symmer_ccsd_generator)

    # Second quantized operators
    symmer_data['auxiliary_operators']['number_operator_second_quantized'] = str(operators['number_operator_of'])
    symmer_data['auxiliary_operators']['N_alpha_second_quantized'] = str(operators['N_alpha_of'])
    symmer_data['auxiliary_operators']['N_beta_second_quantized'] = str(operators['N_beta_of'])
    symmer_data['auxiliary_operators']['S^2_operator_second_quantized'] = str(operators['S2_of'])
    symmer_data['auxiliary_operators']['CCSD_operator_second_quantized'] = str(ccsd_2nd)

    # States
    symmer_data['auxiliary_operators']['ccsd_state'] = symmer_to_dict(symmer_ccsd_state)
    symmer_data['auxiliary_operators']['cisd_state'] = symmer_to_dict(symmer_cisd_state)
    symmer_data['auxiliary_operators']['fci_state'] = symmer_to_dict(qml_fci_state)

    return symmer_data
