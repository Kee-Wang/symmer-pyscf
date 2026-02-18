# symmer-pyscf

Required main package: [Symmer](https://github.com/qmatter-labs/symmer/tree/main).

## Features

1. **Full-space Hamiltonian generation**: Generate Symmer-compatible Pauli Hamiltonians and states from molecular configuration (geometry, symmetry, spin, etc.) via `generate_symmer_data`.
2. **Generalized transformation**: The fermion-to-qubit mapping can be Jordan-Wigner, Bravyi-Kitaev, or **arbitrary** invertible binary matrices.
3. **CAS (Complete Active Space) Hamiltonian**: Build qubit Hamiltonians restricted to a complete active space via `generate_cas_qubit_hamiltonian`, with optional MP2 natural orbital selection. Returns a Symmer-compatible JSON dict alongside the raw result dict, and supports `save_file` for JSON export.
4. **Contextual subspace reduction**: Compute contextual subspace Hamiltonians with reduced qubit count via `mol_info_to_H_cs`.
5. **Bond-scaling pipeline**: Systematically generate Hamiltonians across bond-scaling trajectories for benchmarking via `run_database_pipeline`.

## Installation

Download the package locally and install:

```bash
conda create --name symmer-pyscf python=3.11
conda activate symmer-pyscf
pip install .
```

## Quick Start

```python
from symmerpyscf import generate_symmer_data, generate_cas_qubit_hamiltonian

# Full-space Hamiltonian
geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]
mol_info, symmer_data = generate_symmer_data(geometry, basis="sto-3g")

# CAS Hamiltonian (fewer qubits)
cas_result, cas_data = generate_cas_qubit_hamiltonian(
    geometry, "sto-3g", ncas=2, nelecas=2, save_file="h2_cas.json"
)
```

Both functions return `(result_dict, symmer_data)` where `symmer_data` is a JSON-serializable dictionary compatible with Symmer's `PauliwordOp.from_dictionary` and `QuantumState.from_dictionary`.

## Examples

Run the combined Python example:

```bash
python examples/example_python_combined.py
```

Or explore the Jupyter notebooks in `examples/`:
- `example_notebook.ipynb` — full workflow tutorial
- `cas_tutorial.ipynb` — CAS Hamiltonian tutorial

## Tests

Run the test suite:

```bash
pytest tests/ -m "not slow" --durations=0 -v
```

Include slow N2 multi-active-space tests (~7 min):

```bash
pytest tests/ --durations=0 -v
```

## Citation

For generalized transformation cite:
> Wang, Qingfeng, et al. "Resource-optimized fermionic local-hamiltonian simulation on a quantum computer for quantum chemistry." Quantum 5 (2021): 509.
>
> Steudtner, Mark, and Stephanie Wehner. "Fermion-to-qubit mappings with varying resource requirements for quantum simulation." New Journal of Physics 20.6 (2018): 063010.
