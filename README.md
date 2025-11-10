# symmer-pyscf

Required main package: [Symmer](https://github.com/qmatter-labs/symmer/tree/main).

## Features:
1. Generate Symmer-compatible Pauli Hamiltonian and states with **user-provided** molecular configuration (geometry, symmetry, spin etc.).
2. Generalized transformation: The Fermion $\mapsto$ qubit transformation could be Jordan-Wigner, Bravyi-Kitaev or **arbitrary** invertible binary matrices.





## Installation

Simply download package locally and 

` conda create --name symmer-pyscf python=3.10`

`conda activate symmer-pyscf`

`pip install .` (under root directory)



You can test either by running

`python examples/example_python_combined.py` 

or check `/examples/example_notebook.ipynb`


## Citation
For generalized transformation cite:
> Wang, Qingfeng, et al. "Resource-optimized fermionic local-hamiltonian simulation on a quantum computer for quantum chemistry." Quantum 5 (2021): 509.
>
> Steudtner, Mark, and Stephanie Wehner. "Fermion-to-qubit mappings with varying resource requirements for quantum simulation." New Journal of Physics 20.6 (2018): 063010.
