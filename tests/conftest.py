"""Pytest configuration for symmerpyscf tests."""

import sys


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests that use expensive N2 fixtures (~7 min)"
    )
    config.addinivalue_line(
        "markers", "medium: marks tests with 16-18 qubit molecules (~15 min)"
    )
    config.addinivalue_line(
        "markers", "very_slow: marks tests with >22 qubit molecules (hours+)"
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print per-molecule timing summary from validation tests."""
    # pytest may register the module under various names depending on how
    # it was invoked (e.g. "test_molzoo_validation",
    # "tests.test_molzoo_validation", or a full dotted path).
    mod = None
    for name, m in sys.modules.items():
        if name.endswith("test_molzoo_validation"):
            mod = m
            break
    if mod is None:
        return
    records = getattr(mod, "_timing_records", [])
    if not records:
        return
    terminalreporter.write_sep("=", "Molecule Timing Report")
    header = (
        f"{'Molecule':<40s} {'nq':>4s} {'e-':>4s} "
        f"{'terms':>8s} {'time(s)':>10s} {'10α est.':>10s}"
    )
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * len(header))
    for rec in sorted(records, key=lambda r: r["n_qubits"]):
        est_10a = rec["elapsed_seconds"] * 10
        terminalreporter.write_line(
            f"{rec['molecule_id']:<40s} "
            f"{rec['n_qubits']:>4d} "
            f"{rec['n_electrons']:>4d} "
            f"{rec['n_terms']:>8d} "
            f"{rec['elapsed_seconds']:>10.1f} "
            f"{est_10a:>10.1f}"
        )
    total = sum(r["elapsed_seconds"] for r in records)
    terminalreporter.write_line("-" * len(header))
    terminalreporter.write_line(
        f"{'Total':<40s} {'':>4s} {'':>4s} "
        f"{'':>8s} {total:>10.1f} {total * 10:>10.1f}"
    )
