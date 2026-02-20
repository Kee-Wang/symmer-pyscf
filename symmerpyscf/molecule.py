"""Core molecular data generation and initialization functions."""

import json
import traceback
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

    # Detect orbital degeneracy at HOMO/LUMO boundary
    degeneracy_info = _detect_orbital_degeneracy(pyscf_scf.mo_energy, pyscf_molecule.nelec)

    # Run post-HF methods
    pyscf_mp2, mp2_warnings = _run_mp2(pyscf_scf, molecule, pyscf_data, verbose,
                                        multiplicity, degeneracy_info)
    pyscf_ccsd = _run_ccsd(pyscf_scf, molecule, pyscf_data, verbose)
    pyscf_cisd = _run_cisd(pyscf_scf, molecule, pyscf_data, verbose)
    pyscf_fci, fcivec, fci_warnings = _run_fci(pyscf_molecule, pyscf_scf, molecule,
                                                pyscf_data, verbose, multiplicity,
                                                degeneracy_info)

    # Get integrals and orbital data
    molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
    molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)
    one_body_integrals, two_body_integrals = compute_integrals(pyscf_molecule, pyscf_scf)
    molecule.one_body_integrals = one_body_integrals
    molecule.two_body_integrals = two_body_integrals
    molecule.overlap_integrals = pyscf_scf.get_ovlp()

    # Track errors for audit trail
    _errors = {}

    # Convert FCI state to Symmer format
    # Skip for large systems: state vector has 2^n_qubits entries.
    # At 28 qubits the FCI state alone produces ~112 MB JSON; keep <= 20
    # to stay under ~5 MB per file.
    _MAX_QUBITS_FOR_STATE = 20
    qml_fci_state = None
    if pyscf_fci is not None and fcivec is not None:
        if molecule.n_qubits > _MAX_QUBITS_FOR_STATE:
            print(f'WARNING [AUDIT]: Skipping FCI state conversion '
                  f'(n_qubits={molecule.n_qubits} > {_MAX_QUBITS_FOR_STATE})')
        else:
            try:
                norb = mf.mo_coeff.shape[1]
                n_alpha, n_beta = pyscf_molecule.nelec
                qml_fci_state = _convert_fci_state(fcivec, norb, n_alpha, n_beta)
            except Exception as e:
                tb = traceback.format_exc()
                print(f'WARNING [AUDIT]: FCI state conversion failed\n'
                      f'  Error: {type(e).__name__}: {e}\n'
                      f'  Traceback:\n{tb}')
                _errors['fci_state_conversion'] = f'{type(e).__name__}: {e}'

    # Generate CCSD operator in second quantization
    symmer_ccsd_generator = None
    ccsd_2nd = None
    if pyscf_ccsd is not None:
        try:
            t1 = pyscf.cc.addons.spatial2spin(pyscf_ccsd.t1)
            t2 = pyscf.cc.addons.spatial2spin(pyscf_ccsd.t2)
            ccsd_2nd = t1_t2_to_fermionic_operator(
                t1, t2,
                pyscf_molecule.nelec[0] * 2,
                (pyscf_molecule.nao_nr() - pyscf_molecule.nelec[0]) * 2
            )
            symmer_ccsd_generator = PauliwordOp.from_openfermion(of.jordan_wigner(ccsd_2nd))
        except Exception as e:
            tb = traceback.format_exc()
            print(f'WARNING [AUDIT]: CCSD operator construction failed\n'
                  f'  Error: {type(e).__name__}: {e}\n'
                  f'  Traceback:\n{tb}')
            _errors['ccsd_operator'] = f'{type(e).__name__}: {e}'

    # Generate auxiliary operators
    operators = _generate_auxiliary_operators(molecule)

    # Generate molecular Hamiltonian
    pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
    pyscf_molecular_data.__dict__.update(molecule.__dict__)
    second_quantized_ham = get_fermion_operator(pyscf_molecular_data.get_molecular_hamiltonian())
    symmer_ham = PauliwordOp.from_openfermion(of.jordan_wigner(second_quantized_ham))

    # Import CCSD state
    symmer_ccsd_state = None
    if pyscf_ccsd is not None:
        if molecule.n_qubits > _MAX_QUBITS_FOR_STATE:
            print(f'WARNING [AUDIT]: Skipping CCSD state import '
                  f'(n_qubits={molecule.n_qubits} > {_MAX_QUBITS_FOR_STATE})')
        else:
            try:
                qml_ccsd_state = qml.qchem.import_state(pyscf_ccsd).reshape(-1, 1)
                symmer_ccsd_state = QuantumState.from_array(qml_ccsd_state)
            except Exception as e:
                tb = traceback.format_exc()
                print(f'WARNING [AUDIT]: CCSD state import failed\n'
                      f'  Error: {type(e).__name__}: {e}\n'
                      f'  Traceback:\n{tb}')
                _errors['ccsd_state_import'] = f'{type(e).__name__}: {e}'

    # Import CISD state
    symmer_cisd_state = None
    if pyscf_cisd is not None:
        if molecule.n_qubits > _MAX_QUBITS_FOR_STATE:
            print(f'WARNING [AUDIT]: Skipping CISD state import '
                  f'(n_qubits={molecule.n_qubits} > {_MAX_QUBITS_FOR_STATE})')
        else:
            try:
                qml_cisd_state = qml.qchem.import_state(pyscf_cisd).reshape(-1, 1)
                symmer_cisd_state = QuantumState.from_array(qml_cisd_state)
            except Exception as e:
                tb = traceback.format_exc()
                print(f'WARNING [AUDIT]: CISD state import failed\n'
                      f'  Error: {type(e).__name__}: {e}\n'
                      f'  Traceback:\n{tb}')
                _errors['cisd_state_import'] = f'{type(e).__name__}: {e}'

    # HF state
    hf_state = [0] * molecule.n_qubits
    hf_state[0:molecule.n_electrons] = [1] * molecule.n_electrons

    # Compile symmer_data
    symmer_data = _compile_symmer_data(
        molecule, pyscf_molecule, pyscf_scf, pyscf_mp2, pyscf_cisd,
        pyscf_ccsd, pyscf_fci, symmer_ham, second_quantized_ham,
        hf_state, operators, symmer_ccsd_generator, ccsd_2nd,
        symmer_ccsd_state, symmer_cisd_state, qml_fci_state,
        degeneracy_info=degeneracy_info,
        mp2_warnings=mp2_warnings,
        fci_warnings=fci_warnings,
    )

    # Attach error audit trail to output
    if _errors:
        symmer_data['_errors'] = _errors

    # Save if requested
    if save_file is not None:
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(symmer_data, f, indent=4)

    # Prepare mol_info for workflow — always include core fields,
    # conditionally include fields that depend on post-HF results
    mol_info = {
        'H_second_quantized': of.FermionOperator(symmer_data['H_second_quantized']),
        'hf_state': QuantumState.from_dictionary(symmer_data['hf_state']),
        'n_qubits_full': symmer_data['n_qubits'],
        'n_particles': symmer_data['n_particles']['total'],
        'number_alpha': of.FermionOperator(symmer_data['auxiliary_operators']['N_alpha_second_quantized']),
        'number_beta': of.FermionOperator(symmer_data['auxiliary_operators']['N_beta_second_quantized']),
    }

    # Optional fields — only present if the solver succeeded
    fci_props = symmer_data['calculated_properties'].get('FCI', {})
    if fci_props.get('energy') is not None:
        mol_info['fci_energy'] = fci_props['energy']
    fci_state_data = symmer_data['auxiliary_operators'].get('fci_state')
    if fci_state_data:
        mol_info['fci_state'] = QuantumState.from_dictionary(fci_state_data)
    ccsd_state_data = symmer_data['auxiliary_operators'].get('ccsd_state')
    if ccsd_state_data:
        mol_info['ccsd_state'] = QuantumState.from_dictionary(ccsd_state_data)
    ccsd_op_data = symmer_data['auxiliary_operators'].get('CCSD_operator_second_quantized')
    if ccsd_op_data:
        mol_info['CCSD_generator'] = of.FermionOperator(ccsd_op_data)

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
    """Generate molecular geometry for common molecules.

    Args:
        molecule: Molecule identifier ("H2", "LiH", or "HeH+").
        bondlength: Bond length in Angstroms.

    Returns:
        List of (atom, (x, y, z)) tuples.

    Raises:
        ValueError: If molecule is not in the supported set.
    """
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


def _detect_orbital_degeneracy(mo_energy, nelec, threshold=1e-8):
    """Detect orbital degeneracy at the HOMO/LUMO boundary.

    Returns dict with:
      - 'degenerate': bool — whether HOMO-LUMO gap is below threshold
      - 'gap': float — HOMO-LUMO gap in Hartree
      - 'degenerate_pairs': list of (i, j) pairs within threshold
    """
    n_occ = (nelec[0] + nelec[1]) // 2  # for RHF
    if n_occ > 0 and n_occ < len(mo_energy):
        gap = mo_energy[n_occ] - mo_energy[n_occ - 1]
    else:
        gap = float('inf')
    degenerate_pairs = []
    for i in range(max(0, n_occ - 2), min(len(mo_energy) - 1, n_occ + 2)):
        if abs(mo_energy[i + 1] - mo_energy[i]) < threshold:
            degenerate_pairs.append((i, i + 1))
    return {
        'degenerate': bool(gap < threshold),
        'gap': float(gap),
        'degenerate_pairs': degenerate_pairs,
    }


def _run_mp2(pyscf_scf, molecule, pyscf_data, verbose, multiplicity,
             degeneracy_info=None):
    """Run MP2 calculation with degeneracy-aware fallback. Returns (solver, warnings_list)."""
    from pyscf import scf
    warnings_list = []

    try:
        pyscf_mp2 = mp.MP2(pyscf_scf)
        pyscf_mp2.verbose = 0
        pyscf_mp2.run()
        e_mp2 = pyscf_scf.e_tot + pyscf_mp2.e_corr

        if np.isnan(e_mp2) and degeneracy_info and degeneracy_info['degenerate']:
            # MP2 diverged due to exact orbital degeneracy from symmetry-adapted SCF.
            # Re-run SCF without symmetry to break the degeneracy, then compute MP2.
            msg = (f"MP2 diverged (NaN) due to orbital degeneracy "
                   f"(HOMO-LUMO gap = {degeneracy_info['gap']:.2e} Ha). "
                   f"Retrying with symmetry-broken SCF.")
            print(f"WARNING [AUDIT]: {msg}")
            warnings_list.append(msg)

            mol_nosym = pyscf_scf.mol.copy()
            mol_nosym.symmetry = False
            mol_nosym.build(0, 0)
            mf_nosym = scf.RHF(mol_nosym)
            mf_nosym.conv_tol = pyscf_scf.conv_tol
            mf_nosym.verbose = 0
            mf_nosym.run()

            pyscf_mp2 = mp.MP2(mf_nosym)
            pyscf_mp2.verbose = 0
            pyscf_mp2.run()
            e_mp2 = mf_nosym.e_tot + pyscf_mp2.e_corr

            if not np.isnan(e_mp2):
                msg2 = (f"MP2 converged with symmetry-broken SCF: "
                        f"{e_mp2:.12f}")
                print(f"  {msg2}")
                warnings_list.append(msg2)

        molecule.mp2_energy = e_mp2
        pyscf_data['mp2'] = pyscf_mp2

        if verbose:
            print(f'MP2 energy for {molecule.name} '
                  f'({molecule.n_electrons} electrons) is {molecule.mp2_energy}')

        return pyscf_mp2, warnings_list
    except Exception as e:
        tb = traceback.format_exc()
        print(f'WARNING [AUDIT]: MP2 failed for {molecule.name}\n'
              f'  Error: {type(e).__name__}: {e}\n'
              f'  Traceback:\n{tb}')
        molecule.mp2_energy = None
        return None, warnings_list


def _run_ccsd(pyscf_scf, molecule, pyscf_data, verbose):
    """Run CCSD calculation. Returns None on failure with full error audit."""
    try:
        pyscf_ccsd = cc.CCSD(pyscf_scf)
        pyscf_ccsd.verbose = 0
        pyscf_ccsd.run()
        molecule.ccsd_energy = pyscf_ccsd.e_tot
        pyscf_data['ccsd'] = pyscf_ccsd

        if verbose:
            print(f'CCSD energy for {molecule.name} '
                  f'({molecule.n_electrons} electrons) is {molecule.ccsd_energy}')

        return pyscf_ccsd
    except Exception as e:
        tb = traceback.format_exc()
        print(f'WARNING [AUDIT]: CCSD failed for {molecule.name}\n'
              f'  Error: {type(e).__name__}: {e}\n'
              f'  Traceback:\n{tb}')
        molecule.ccsd_energy = None
        return None


def _run_cisd(pyscf_scf, molecule, pyscf_data, verbose):
    """Run CISD calculation. Returns None on failure with full error audit."""
    try:
        pyscf_cisd = pyscf.ci.CISD(pyscf_scf)
        pyscf_cisd.verbose = 0
        pyscf_cisd.run()
        molecule.cisd_energy = pyscf_cisd.e_tot
        pyscf_data['cisd'] = pyscf_cisd

        if verbose:
            print(f'CISD energy for {molecule.name} '
                  f'({molecule.n_electrons} electrons) is {molecule.cisd_energy}')

        return pyscf_cisd
    except Exception as e:
        tb = traceback.format_exc()
        print(f'WARNING [AUDIT]: CISD failed for {molecule.name}\n'
              f'  Error: {type(e).__name__}: {e}\n'
              f'  Traceback:\n{tb}')
        molecule.cisd_energy = None
        return None


def _run_fci(pyscf_molecule, pyscf_scf, molecule, pyscf_data, verbose,
             multiplicity=1, degeneracy_info=None):
    """Run spin-constrained FCI with verification and symmetry-broken fallback.

    Strategy:
      1. If orbital degeneracy detected at HOMO/LUMO boundary, go directly to
         symmetry-broken FCI (symmetry-adapted basis can restrict the FCI
         configuration space and miss the correct root, even when spin is correct).
      2. Otherwise, run FCI with fix_spin_(shift=0.2) on symmetry-adapted orbitals.
      3. Verify <S^2> matches target spin.  If not, retry with shift=1.0.

    Returns (solver, fcivec, warnings_list). On failure returns (None, None, warnings_list).
    """
    from pyscf import scf as pyscf_scf_mod
    from pyscf.fci import addons as fci_addons, spin_op
    warnings_list = []
    fci_spin_squared = None
    fci_multiplicity = None

    try:
        S = (multiplicity - 1) / 2
        target_ss = S * (S + 1)

        # If orbital degeneracy detected, use symmetry-broken SCF directly.
        # With symmetry=True, degenerate orbitals at the HOMO/LUMO boundary
        # can restrict the FCI configuration space to the wrong root (e.g.,
        # HN finds a singlet 0.04 Ha above the true ground state singlet).
        if degeneracy_info and degeneracy_info['degenerate']:
            msg = (f"Orbital degeneracy detected at HOMO/LUMO boundary "
                   f"(gap={degeneracy_info['gap']:.2e} Ha). "
                   f"Using symmetry-broken SCF for FCI to avoid "
                   f"restricted configuration space.")
            print(f"WARNING [AUDIT]: {msg}")
            warnings_list.append(msg)

            mol_nosym = pyscf_molecule.copy()
            mol_nosym.symmetry = False
            mol_nosym.build(0, 0)
            mf_nosym = pyscf_scf_mod.RHF(mol_nosym)
            mf_nosym.conv_tol = pyscf_scf.conv_tol
            mf_nosym.verbose = 0
            mf_nosym.run()

            norb = mf_nosym.mo_coeff.shape[1]
            nelec = mol_nosym.nelec

            pyscf_fci = fci.FCI(mol_nosym, mf_nosym.mo_coeff)
            pyscf_fci.verbose = 0
            fci_addons.fix_spin_(pyscf_fci, shift=0.2, ss=target_ss)
            fci_energy, fcivec = pyscf_fci.kernel()

            ss_val, mult_val = spin_op.spin_square(fcivec, norb, nelec)
            fci_spin_squared = float(ss_val)
            fci_multiplicity = float(mult_val)
            warnings_list.append(
                f"FCI with symmetry-broken SCF: E={fci_energy:.12f}, "
                f"<S^2>={ss_val:.4f}")
        else:
            norb = pyscf_scf.mo_coeff.shape[1]
            nelec = pyscf_molecule.nelec

            pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
            pyscf_fci.verbose = 0
            fci_addons.fix_spin_(pyscf_fci, shift=0.2, ss=target_ss)

            fci_energy, fcivec = pyscf_fci.kernel()

            # Verify spin of the result
            ss_val, mult_val = spin_op.spin_square(fcivec, norb, nelec)
            fci_spin_squared = float(ss_val)
            fci_multiplicity = float(mult_val)
            spin_ok = abs(ss_val - target_ss) < 0.1

            if not spin_ok:
                msg = (f"FCI spin verification failed: <S^2>={ss_val:.4f} "
                       f"(expected {target_ss:.1f}), mult={mult_val:.2f}. "
                       f"Retrying with stronger penalty (shift=1.0).")
                print(f"WARNING [AUDIT]: {msg}")
                warnings_list.append(msg)

                pyscf_fci2 = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
                pyscf_fci2.verbose = 0
                fci_addons.fix_spin_(pyscf_fci2, shift=1.0, ss=target_ss)
                fci_energy2, fcivec2 = pyscf_fci2.kernel()
                ss_val2, mult_val2 = spin_op.spin_square(fcivec2, norb, nelec)

                if abs(ss_val2 - target_ss) < 0.1:
                    fci_energy, fcivec = fci_energy2, fcivec2
                    pyscf_fci = pyscf_fci2
                    fci_spin_squared = float(ss_val2)
                    fci_multiplicity = float(mult_val2)
                    warnings_list.append(
                        f"FCI converged with shift=1.0: E={fci_energy:.12f}, "
                        f"<S^2>={ss_val2:.4f}"
                    )
                else:
                    msg2 = (f"FCI spin constraint failed even with shift=1.0: "
                            f"<S^2>={ss_val2:.4f}")
                    print(f"WARNING [AUDIT]: {msg2}")
                    warnings_list.append(msg2)

        if degeneracy_info and degeneracy_info['degenerate']:
            msg = (f"Note: orbital degeneracy detected "
                   f"(gap={degeneracy_info['gap']:.2e} Ha). "
                   f"FCI used fix_spin_(ss={target_ss}) to target "
                   f"multiplicity={multiplicity}.")
            warnings_list.append(msg)

        molecule.fci_energy = float(fci_energy)
        pyscf_data['fci'] = pyscf_fci
        pyscf_data['fci_spin_squared'] = fci_spin_squared
        pyscf_data['fci_multiplicity'] = fci_multiplicity

        if verbose:
            print(f'FCI energy for {molecule.name} '
                  f'({molecule.n_electrons} electrons) is {molecule.fci_energy}'
                  f' (<S^2>={fci_spin_squared:.4f})')

        return pyscf_fci, fcivec, warnings_list
    except Exception as e:
        tb = traceback.format_exc()
        print(f'WARNING [AUDIT]: FCI failed for {molecule.name}\n'
              f'  Error: {type(e).__name__}: {e}\n'
              f'  Traceback:\n{tb}')
        molecule.fci_energy = None
        return None, None, warnings_list


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
                         symmer_ccsd_state, symmer_cisd_state, qml_fci_state,
                         degeneracy_info=None, mp2_warnings=None, fci_warnings=None):
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

    # Energies — None means solver failed; record that explicitly for audit
    def _energy_entry(energy, solver_obj):
        if energy is None:
            return {"energy": None, "converged": False}
        if solver_obj is not None and hasattr(solver_obj, 'converged'):
            converged = bool(solver_obj.converged)
        else:
            converged = solver_obj is not None
        return {"energy": float(energy), "converged": converged}

    # Build MP2 entry with warnings
    mp2_entry = _energy_entry(molecule.mp2_energy, pyscf_mp2)
    if mp2_warnings:
        mp2_entry['warnings'] = mp2_warnings

    # Build FCI entry with spin info and warnings
    fci_entry = _energy_entry(molecule.fci_energy, pyscf_fci)
    fci_entry['spin_constrained'] = True  # always using fix_spin_ now
    if molecule._pyscf_data.get('fci_spin_squared') is not None:
        fci_entry['spin_squared'] = molecule._pyscf_data['fci_spin_squared']
    if molecule._pyscf_data.get('fci_multiplicity') is not None:
        fci_entry['multiplicity'] = molecule._pyscf_data['fci_multiplicity']
    if fci_warnings:
        fci_entry['warnings'] = fci_warnings

    symmer_data['calculated_properties'] = {
        "HF": {"energy": float(molecule.hf_energy), "converged": bool(pyscf_scf.converged)},
        "MP2": mp2_entry,
        "CISD": _energy_entry(molecule.cisd_energy, pyscf_cisd),
        "CCSD": _energy_entry(molecule.ccsd_energy, pyscf_ccsd),
        "FCI": fci_entry,
    }

    # Orbital degeneracy metadata
    if degeneracy_info is not None:
        symmer_data['orbital_degeneracy'] = {
            'homo_lumo_gap': degeneracy_info['gap'],
            'degenerate': degeneracy_info['degenerate'],
            'degenerate_pairs': degeneracy_info['degenerate_pairs'],
        }

    # Auxiliary operators — always include number/spin operators (they depend only on HF)
    symmer_data['auxiliary_operators'] = {}
    symmer_data['auxiliary_operators']['number_operator'] = symmer_to_dict(operators['number_operator'])
    symmer_data['auxiliary_operators']['N_alpha'] = symmer_to_dict(operators['N_alpha'])
    symmer_data['auxiliary_operators']['N_beta'] = symmer_to_dict(operators['N_beta'])
    symmer_data['auxiliary_operators']['S^2_operator'] = symmer_to_dict(operators['S2'])

    # Second quantized operators (always available — depend on HF only)
    symmer_data['auxiliary_operators']['number_operator_second_quantized'] = str(operators['number_operator_of'])
    symmer_data['auxiliary_operators']['N_alpha_second_quantized'] = str(operators['N_alpha_of'])
    symmer_data['auxiliary_operators']['N_beta_second_quantized'] = str(operators['N_beta_of'])
    symmer_data['auxiliary_operators']['S^2_operator_second_quantized'] = str(operators['S2_of'])

    # CCSD-dependent fields — only if CCSD succeeded
    if symmer_ccsd_generator is not None:
        symmer_data['auxiliary_operators']['CCSD_operator'] = symmer_to_dict(symmer_ccsd_generator)
    if ccsd_2nd is not None:
        symmer_data['auxiliary_operators']['CCSD_operator_second_quantized'] = str(ccsd_2nd)

    # States — only if available
    if symmer_ccsd_state is not None:
        symmer_data['auxiliary_operators']['ccsd_state'] = symmer_to_dict(symmer_ccsd_state)
    if symmer_cisd_state is not None:
        symmer_data['auxiliary_operators']['cisd_state'] = symmer_to_dict(symmer_cisd_state)
    if qml_fci_state is not None:
        symmer_data['auxiliary_operators']['fci_state'] = symmer_to_dict(qml_fci_state)

    return symmer_data
