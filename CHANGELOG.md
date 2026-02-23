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
  orbital degeneracy from symmetry-adapted SCF, the code now re-runs
  SCF without symmetry to break the degeneracy, then computes MP2.

- **Large-system OOM**: State vector conversion (FCI, CCSD, CISD) now skips
  systems with >20 qubits to avoid out-of-memory (2^n entries). Previously,
  ClNa (36 qubits) was killed by the OS during state conversion.

### Added
- **S² spin diagnostics**: `mol_info_to_H_cs` now builds, tapers, and projects
  the S² operator through the contextual-subspace pipeline, returning `S2_cs`,
  `cs_state`, and `cs_s2` (expectation value ⟨S²⟩ of the ground state)
- **QSCI solver**: New `qsci_symmer_with_prob_hist` function implementing
  quantum-selected configuration interaction from Z-basis measurement histograms
- Orbital degeneracy detection at HOMO/LUMO boundary
- FCI spin metadata in output JSON (`spin_squared`, `multiplicity`, `spin_constrained`)
- Warning audit trail in MP2 and FCI output entries
- `orbital_degeneracy` field in output JSON

### Removed
- Validation suite (51 molecules across 4 tiers) archived to `3-Symmer-Hamiltonian`
- Pipeline orchestration (`MoleculeRecord`, `ScalingResult`, `parse_molecule_csv`,
  `run_single_point`, `run_molecule_scan`, `run_database_pipeline`, CLI entry point)
  archived to `3-Symmer-Hamiltonian`
