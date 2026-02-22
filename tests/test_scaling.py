"""Tests for geometry scaling utilities."""

import numpy as np
import pytest

from symmerpyscf.scaling import (
    _parse_xyz_string,
    scale_geometry,
    generate_scaling_grid,
    should_stop_scanning,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def h2_geometry():
    return [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))]


# ===========================================================================
# Unit tests
# ===========================================================================

class TestParseXyzString:
    def test_single_atom(self):
        result = _parse_xyz_string('Be 0.0 0.0 0.0')
        assert result == [('Be', (0.0, 0.0, 0.0))]

    def test_two_atoms(self):
        result = _parse_xyz_string('H 0.0 0.0 0.0\nH 0.0 0.0 0.74')
        assert result == [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))]

    def test_three_atoms(self):
        xyz = 'O 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587'
        result = _parse_xyz_string(xyz)
        assert len(result) == 3
        assert result[0][0] == 'O'
        assert result[1][0] == 'H'

    def test_bad_format_raises(self):
        with pytest.raises(ValueError, match='Expected 4 fields'):
            _parse_xyz_string('H 0.0 0.0')  # missing z

    def test_whitespace_handling(self):
        result = _parse_xyz_string('  H  0.0  0.0  0.0  \n  H  0.0  0.0  0.74  ')
        assert len(result) == 2


class TestScaleGeometry:
    def test_identity(self, h2_geometry):
        result = scale_geometry(h2_geometry, 1.0)
        assert result == h2_geometry

    def test_double(self, h2_geometry):
        result = scale_geometry(h2_geometry, 2.0)
        assert result[0] == ('H', (0.0, 0.0, 0.0))
        assert result[1][1][2] == pytest.approx(1.48)

    def test_half(self, h2_geometry):
        result = scale_geometry(h2_geometry, 0.5)
        assert result[1][1][2] == pytest.approx(0.37)

    def test_preserves_elements(self, h2_geometry):
        result = scale_geometry(h2_geometry, 3.0)
        assert result[0][0] == 'H'
        assert result[1][0] == 'H'


class TestGenerateScalingGrid:
    def test_default_grid(self):
        grid = generate_scaling_grid()
        assert grid[0] == pytest.approx(0.5)
        assert grid[-1] == pytest.approx(3.0)
        assert len(grid) > 20  # Should be ~31 points

    def test_endpoints_included(self):
        grid = generate_scaling_grid(alpha_min=0.3, alpha_max=4.0)
        assert 0.3 in grid
        assert 4.0 in grid

    def test_dense_region_spacing(self):
        grid = generate_scaling_grid()
        # Points in [0.8, 2.0] should be spaced at ~0.05
        dense = grid[(grid >= 0.8) & (grid <= 2.0)]
        diffs = np.diff(dense)
        assert np.all(diffs <= 0.051)

    def test_sorted_and_unique(self):
        grid = generate_scaling_grid()
        assert np.all(np.diff(grid) > 0)  # strictly increasing

    def test_custom_steps(self):
        grid = generate_scaling_grid(dense_step=0.1, sparse_step=0.5)
        assert len(grid) > 10  # reasonable size


class TestShouldStopScanning:
    def test_before_threshold_alpha(self):
        energies = {1.0: -1.0, 1.5: -0.9, 2.0: -0.85}
        assert should_stop_scanning(energies, 2.0) is False

    def test_flat_curve_stops(self):
        energies = {
            1.0: -1.0,
            1.5: -0.9,
            2.0: -0.85,
            2.25: -0.850001,
            2.5: -0.850002,
        }
        assert should_stop_scanning(energies, 2.5) is True

    def test_steep_curve_continues(self):
        energies = {
            1.0: -1.0,
            2.0: -0.85,
            2.25: -0.80,
            2.5: -0.75,
        }
        assert should_stop_scanning(energies, 2.5) is False

    def test_insufficient_data(self):
        energies = {2.5: -0.85}
        assert should_stop_scanning(energies, 2.5) is False

    def test_none_energies_ignored(self):
        energies = {2.0: None, 2.25: None, 2.5: -0.85}
        assert should_stop_scanning(energies, 2.5) is False
