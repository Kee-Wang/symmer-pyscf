# Changelog

## [Unreleased]

### Added

- O(k) Pauli coefficient fast path for `hs_inner_product`, `hs_norm`,
  `hs_fidelity`, and `hs_distance` when both inputs are `PauliwordOp`.
  Avoids constructing 2^n × 2^n matrices by exploiting Pauli trace
  orthogonality.
- `hs_infidelity(A, B)` — returns `1 − hs_fidelity(A, B)`, a convenient
  error metric in [0, 1] where 0 means identical up to a scalar.
- `include_state_vectors` parameter to `generate_symmer_data()` — set to
  `False` to exclude state vectors (FCI, CCSD, CISD) from the JSON output
  and reduce file size. Defaults to `True` (existing behavior). State
  vectors remain available in `mol_info` regardless of this flag.
