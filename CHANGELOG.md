# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **FCI for degenerate systems**: FCI now uses `fix_spin_()` to constrain
  the solver to the correct spin sector, with a three-stage fallback:
  (1) fix_spin_ with shift=0.2, (2) retry with shift=1.0, (3) re-run SCF
  without symmetry to break the degeneracy and retry FCI. Previously,
  systems with orbital degeneracy at the HOMO/LUMO boundary (e.g., HN with
  degenerate pi orbitals) returned the wrong root because the symmetry-adapted
  Hilbert space excluded the correct spin sector.

- **MP2 for degenerate systems**: When MP2 diverges (NaN) due to exact
  orbital degeneracy from symmetry-adapted SCF, the pipeline now re-runs
  SCF without symmetry to break the degeneracy, then computes MP2.

- **Large-system OOM**: State vector conversion (FCI, CCSD, CISD) now skips
  systems with >30 qubits to avoid out-of-memory (2^n entries). Previously,
  ClNa (36 qubits) was killed by the OS during state conversion.

### Added
- Orbital degeneracy detection at HOMO/LUMO boundary
- FCI spin metadata in output JSON (`spin_squared`, `multiplicity`, `spin_constrained`)
- Warning audit trail in MP2 and FCI output entries
- `orbital_degeneracy` field in output JSON
- Validation suite: 51 molecules across 4 tiers (6–36 qubits), comparing
  energies and Hamiltonian coefficients against symmer reference JSONs
