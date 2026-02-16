"""
symmer-pyscf example: full-space and CAS Hamiltonian workflows.

Demonstrates:
  1. Full-space Hamiltonian generation (LiH/STO-3G)
  2. Fermionic transformation comparison (JW, BK, random)
  3. Contextual subspace energy calculation
  4. Loading/inspecting saved molecular JSON data
  5. CAS Hamiltonian generation (H2 and N2)
"""

import json
import numpy as np

from symmerpyscf import (
    initialize_molecule,
    get_geometry,
    random_invertible_binary_matrix,
    mol_info_to_H_cs,
    generate_cas_qubit_hamiltonian,
)

# ── 1. Full-space workflow: LiH ─────────────────────────────────────────────

print("=" * 60)
print("1. Full-space Hamiltonian: LiH / STO-3G")
print("=" * 60)

bondlength = 0.74  # Angstroms
molecule = "LiH"
geometry = get_geometry(molecule=molecule, bondlength=bondlength)
basis = "sto-3g"
outdir = "./output"

mol_info, pyscf_data, energy_data, filename = initialize_molecule(
    molecule=molecule,
    bondlength=bondlength,
    geometry=geometry,
    basis=basis,
    outdir=outdir,
    verbose=True,
)

print(f"\nNumber of qubits: {pyscf_data['n_qubits']}")
print(f"Number of electrons: {pyscf_data['n_particles']['total']}")
print(f"Point group: {pyscf_data['point_group']['groupname']}")

# ── 2. Fermionic transformations ────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. Fermionic Transformation Matrices")
print("=" * 60)

beta_jw = random_invertible_binary_matrix(n=pyscf_data['n_qubits'], beta='Jordan-Wigner')
beta_bk = random_invertible_binary_matrix(n=pyscf_data['n_qubits'], beta='Bravyi-Kitaev')
beta_random = random_invertible_binary_matrix(n=pyscf_data['n_qubits'])

print("Jordan-Wigner beta matrix:")
print(beta_jw)
print("\nBravyi-Kitaev beta matrix:")
print(beta_bk)
print("\nRandom beta matrix:")
print(beta_random)

# ── 3. Contextual subspace calculations ─────────────────────────────────────

print("\n" + "=" * 60)
print("3. Contextual Subspace Energies")
print("=" * 60)

n_cs_qubits = 6
beta = beta_jw

# Baseline NCS energy (1 qubit)
data_ncs = mol_info_to_H_cs(mol_info, n_cs_qubits=1, beta=beta)
ncs_energy = data_ncs['cs_energy']
fci_energy = data_ncs['fci_energy']

print(f"NCS Energy (1 qubit):  {ncs_energy:.8f} Ha")
print(f"FCI Energy:            {fci_energy:.8f} Ha")
print(f"Error vs FCI:          {ncs_energy - fci_energy:.8e} Ha\n")

# Compare transformations at target qubit count
transformations = {
    'Jordan-Wigner': beta_jw,
    'Bravyi-Kitaev': beta_bk,
    'Random': beta_random,
}

for name, beta_matrix in transformations.items():
    data = mol_info_to_H_cs(mol_info, n_cs_qubits=n_cs_qubits, beta=beta_matrix)
    print(f"{name:20s}  CS Energy: {data['cs_energy']:.8f} Ha  "
          f"Error: {data['cs_energy'] - fci_energy:.2e} Ha  "
          f"Terms: {data['n_terms_hamiltonian']}")

# ── 4. Load and inspect saved JSON ──────────────────────────────────────────

print("\n" + "=" * 60)
print("4. Inspecting Saved Molecular Data")
print("=" * 60)

if filename:
    with open(filename, 'r') as f:
        chem_data = json.load(f)
    print(f"Loaded: {filename}")
    print(f"Top-level keys: {list(chem_data.keys())}")
    print(f"Energy methods: {list(chem_data['calculated_properties'].keys())}")

    from symmer import PauliwordOp, QuantumState

    H = PauliwordOp.from_dictionary(chem_data['H'])
    fci_state = QuantumState.from_dictionary(
        chem_data['auxiliary_operators']['fci_state']
    )
    fci_energy_loaded = chem_data['calculated_properties']['FCI']['energy']
    print(f"<FCI|H|FCI> - E_fci = {H.expval(fci_state) - fci_energy_loaded:.2e}")

# ── 5. CAS Hamiltonian workflow ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. CAS Hamiltonian Generation")
print("=" * 60)

# H2 / STO-3G  (full space = CAS(2,2) = 4 qubits)
h2_geom = [("H", (0, 0, 0)), ("H", (0, 0, 0.735))]
result = generate_cas_qubit_hamiltonian(h2_geom, "sto-3g", ncas=2, nelecas=2)

print(f"\nH2 / STO-3G  CAS(2,2)")
print(f"  Qubits:       {result['n_qubits']}")
print(f"  E(HF):        {result['e_hf']:.8f} Ha")
print(f"  E(CASCI):     {result['e_casci']:.8f} Ha")
print(f"  E(FCI):       {result['e_fci']:.8f} Ha")

# Verify self-consistency: min(eig(H_cas)) + e_core == e_casci
evals = np.linalg.eigvalsh(result['H_cas'].to_sparse_matrix.toarray())
e_total = evals[0] + result['e_core']
print(f"  min(eig)+e_core: {e_total:.8f} Ha  (diff: {abs(e_total - result['e_casci']):.2e})")

# N2 / STO-3G  (multiple active spaces)
n2_geom = [("N", (0, 0, 0)), ("N", (0, 0, 1.1))]
print(f"\nN2 / STO-3G  at R=1.1 A")
print(f"  {'CAS':12s} {'Qubits':>6s} {'E(CASCI)':>16s} {'E(CASCI)-E(FCI)':>16s}")

for ncas, nelecas in [(4, 2), (6, 6), (10, 14)]:
    r = generate_cas_qubit_hamiltonian(n2_geom, "sto-3g", ncas=ncas, nelecas=nelecas)
    label = f"CAS({ncas},{nelecas})"
    print(f"  {label:12s} {r['n_qubits']:6d} {r['e_casci']:16.8f} {r['e_casci'] - r['e_fci']:16.2e}")

print("\nDone.")
