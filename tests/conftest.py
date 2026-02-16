"""Pytest configuration for symmerpyscf tests."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests that use expensive N2 fixtures (~7 min)"
    )
