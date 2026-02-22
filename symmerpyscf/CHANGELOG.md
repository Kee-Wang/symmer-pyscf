# Changelog

## [Unreleased]

### Added

- `include_state_vectors` parameter to `generate_symmer_data()` — set to
  `False` to exclude state vectors (FCI, CCSD, CISD) from the JSON output
  and reduce file size. Defaults to `True` (existing behavior). State
  vectors remain available in `mol_info` regardless of this flag.
