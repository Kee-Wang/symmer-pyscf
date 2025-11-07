from symmerpyscf import (
    initialize_molecule,
    get_geometry,
    random_invertible_binary_matrix,
    mol_info_to_H_cs
)

# Set parameters
bondlength = 0.74  # Angstroms
molecule = "LiH"
geometry = get_geometry(molecule=molecule, bondlength=bondlength)

print(geometry)

basis = "sto-3g"
outdir = "./output"  # Optional: save data to file

print('Output file will be saved in', outdir)

# Initialize molecule and run quantum chemistry calculations
mol_info, pyscf_data, energy_data, filename = initialize_molecule(
    molecule=molecule,
    bondlength=bondlength,
    geometry=geometry,
    basis=basis,
    outdir=outdir,
    verbose=True
)

print(f"\nNumber of qubits: {pyscf_data['n_qubits']}")
print(f"Number of electrons: {pyscf_data['n_particles']['total']}")
print(f"Point group: {pyscf_data['point_group']['groupname']}")

# Option 1: Jordan-Wigner transformation
beta_jw = random_invertible_binary_matrix(
    n=pyscf_data['n_qubits'],
    beta='Jordan-Wigner'
)

print("Jordan-Wigner beta matrix:")
print(beta_jw)

# Option 2: Bravyi-Kitaev transformation
beta_bk = random_invertible_binary_matrix(
    n=pyscf_data['n_qubits'],
    beta='Bravyi-Kitaev'
)

print("\nBravyi-Kitaev beta matrix:")
print(beta_bk)

# Option 3: Random invertible matrix
beta_random = random_invertible_binary_matrix(
    n=pyscf_data['n_qubits']
)

print("\nRandom beta matrix:")
print(beta_random)

# Set contextual subspace parameters
n_cs_qubit = 6  # Target number of qubits

# Use Jordan-Wigner transformation
beta = beta_jw

# Compute baseline NCS energy (1 qubit contextual subspace)
data_ncs = mol_info_to_H_cs(
    mol_info,
    n_cs_qubit=1,
    beta=beta
)

ncs_energy = data_ncs['cs_energy']
fci_energy = data_ncs['fci_energy']

print(f"NCS Energy (1 qubit): {ncs_energy:.8f} Ha")
print(f"FCI Energy: {fci_energy:.8f} Ha")
print(f"Error vs FCI: {ncs_energy - fci_energy:.8e} Ha\n")

# Compute with target number of qubits
data_cs = mol_info_to_H_cs(
    mol_info,
    n_cs_qubit=n_cs_qubit,
    beta=beta
)

print(f"CS Energy ({n_cs_qubit} qubits): {data_cs['cs_energy']:.8f} Ha")
print(f"Error vs FCI: {data_cs['cs_energy'] - fci_energy:.8e} Ha")
print(f"Number of Hamiltonian terms: {data_cs['n_terms_hamiltonian']}")
print(f"Number of CCSD generator terms: {data_cs['n_terms_ccsd_generator']}")

transformations = {
    'Jordan-Wigner': beta_jw,
    'Bravyi-Kitaev': beta_bk,
    'Random': beta_random
}

results = {}
datas = []
for name, beta_matrix in transformations.items():
    data = mol_info_to_H_cs(
        mol_info,
        n_cs_qubit=n_cs_qubit,
        beta=beta_matrix
    )
    results[name] = data
    datas.append(data)

    print(f"\n{name} Transformation:")
    print(f"  CS Energy: {data['cs_energy']:.8f} Ha")
    print(f"  Error: {data['cs_energy'] - fci_energy:.8e} Ha")
    print(f"  Hamiltonian terms: {data['n_terms_hamiltonian']}")
    print(f"  HF contextual state {data['hf_cs']}")

import json

# Load saved data (if outdir was specified)
if filename:
    with open(filename, 'r') as f:
        chem_data = json.load(f)
    print('Loaded data from:', filename)
    print("Loaded data keys:")
    print(list(chem_data.keys()))

    print('\nSubkey under ["calculated_properties"]\n')
    print(list(chem_data['calculated_properties'].keys()))
    print('\nSubkey under ["auxiliary_operators"]\n')
    print(list(chem_data['auxiliary_operators'].keys()))

    # Access specific data
    print(f"\nBasis set: {chem_data['basis']}")
    print(f"Number of qubits: {chem_data['n_qubits']}")
    print(f"FCI energy: {chem_data['calculated_properties']['FCI']['energy']:.8f} Ha")

from symmer import PauliwordOp, QuantumState

H = PauliwordOp.from_dictionary(chem_data['H'])

hf_state = QuantumState.from_dictionary(chem_data['hf_state'])
cisd_state = QuantumState.from_dictionary(chem_data['auxiliary_operators']['cisd_state'])
ccsd_state = QuantumState.from_dictionary(chem_data['auxiliary_operators']['ccsd_state'])
fci_state = QuantumState.from_dictionary(chem_data['auxiliary_operators']['fci_state'])

number_alpha = PauliwordOp.from_dictionary(chem_data['auxiliary_operators']['N_alpha'])
number_beta = PauliwordOp.from_dictionary(chem_data['auxiliary_operators']['N_beta'])
S2_op = PauliwordOp.from_dictionary(chem_data['auxiliary_operators']['S^2_operator'])
fci_energy = chem_data['calculated_properties']['FCI']['energy']

print('test energy difference:', H.expval(fci_state) - fci_energy)