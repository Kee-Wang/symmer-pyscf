"""Tests for the bond-scaling pipeline.

Unit tests (no PySCF) cover: CSV parsing, geometry scaling, adaptive grid,
adaptive stopping. Integration tests (require PySCF) cover the full pipeline.
"""

import csv
import json

import numpy as np
import pytest

from symmerpyscf.scaling import (
    MoleculeRecord,
    ScalingResult,
    parse_molecule_csv,
    _parse_xyz_string,
    _parse_qubit_count,
    scale_geometry,
    generate_scaling_grid,
    should_stop_scanning,
    run_single_point,
    run_molecule_scan,
    run_database_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def h2_geometry():
    return [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))]


@pytest.fixture
def h2_record():
    return MoleculeRecord(
        id='H2_singlet_Dooh',
        species='H2',
        formula='H2',
        name='Hydrogen',
        n_atoms=2,
        charge=0,
        multiplicity=1,
        n_electrons=2,
        geometry=[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))],
    )


@pytest.fixture
def be_record():
    """Single-atom molecule for testing."""
    return MoleculeRecord(
        id='Be_singlet_Kh',
        species='Be',
        formula='Be',
        name='Beryllium',
        n_atoms=1,
        charge=0,
        multiplicity=1,
        n_electrons=4,
        geometry=[('Be', (0.0, 0.0, 0.0))],
    )


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal CSV file for testing."""
    csv_file = tmp_path / 'molecules.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'id', 'species', 'formula', 'name', 'n_atoms',
            'charge', 'multiplicity', 'n_electrons', 'xyz',
            'reference_energy', 'sto-3g',
        ])
        # H2
        writer.writerow([
            'H2_singlet_Dooh', 'H2', 'H2', 'Hydrogen', 2,
            0, 1, 2, 'H 0.0 0.0 0.0\nH 0.0 0.0 0.74',
            '-1.137', '4',
        ])
        # Be (single atom)
        writer.writerow([
            'Be_singlet_Kh', 'Be', 'Be', 'Beryllium', 1,
            0, 1, 4, 'Be 0.0 0.0 0.0',
            '', '',
        ])
        # LiH
        writer.writerow([
            'LiH_singlet_Coov', 'LiH', 'LiH', 'Lithium hydride', 2,
            0, 1, 4, 'Li 0.0 0.0 0.0\nH 0.0 0.0 1.595',
            '-7.882', '12',
        ])
    return str(csv_file)


# ===========================================================================
# Unit tests — no PySCF needed
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


class TestParseQubitCount:
    def test_plain_number(self):
        assert _parse_qubit_count('4') == 4

    def test_emoji_prefix(self):
        assert _parse_qubit_count('\u2705 12') == 12

    def test_empty(self):
        assert _parse_qubit_count('') is None

    def test_no_digits(self):
        assert _parse_qubit_count('\u274c') is None


class TestParseMoleculeCsv:
    def test_basic_parse(self, sample_csv):
        records = parse_molecule_csv(sample_csv)
        assert len(records) == 3

        h2 = records[0]
        assert h2.id == 'H2_singlet_Dooh'
        assert h2.formula == 'H2'
        assert h2.n_atoms == 2
        assert h2.charge == 0
        assert h2.multiplicity == 1
        assert h2.n_electrons == 2
        assert len(h2.geometry) == 2
        assert h2.reference_energy == pytest.approx(-1.137)
        assert h2.n_qubits_sto3g == 4

    def test_single_atom_detected(self, sample_csv):
        records = parse_molecule_csv(sample_csv)
        be = records[1]
        assert be.is_single_atom is True
        assert be.reference_energy is None
        assert be.n_qubits_sto3g is None

    def test_geometry_parsed(self, sample_csv):
        records = parse_molecule_csv(sample_csv)
        lih = records[2]
        assert lih.geometry[0] == ('Li', (0.0, 0.0, 0.0))
        assert lih.geometry[1][0] == 'H'
        assert lih.geometry[1][1][2] == pytest.approx(1.595)


class TestMoleculeRecord:
    def test_is_single_atom(self, be_record):
        assert be_record.is_single_atom is True

    def test_is_not_single_atom(self, h2_record):
        assert h2_record.is_single_atom is False


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


# ===========================================================================
# Integration tests — require PySCF
# ===========================================================================

class TestRunSinglePoint:
    def test_h2_equilibrium(self, h2_record):
        """H2 at alpha=1.0 should succeed with all solvers."""
        result, data = run_single_point(h2_record, alpha=1.0)
        assert result.status == 'success'
        assert result.alpha == 1.0
        assert result.molecule_id == 'H2_singlet_Dooh'
        assert result.elapsed_seconds > 0

        # Check that all energies are present
        props = data['calculated_properties']
        assert props['HF']['energy'] is not None
        assert props['FCI']['energy'] is not None
        assert props['CCSD']['energy'] is not None

        # Check scaling metadata is attached
        assert data['scaling_metadata']['alpha'] == 1.0
        assert data['scaling_metadata']['molecule_id'] == 'H2_singlet_Dooh'

        # FCI should be lower than HF
        assert props['FCI']['energy'] < props['HF']['energy']


class TestRunMoleculeScan:
    def test_h2_small_grid(self, h2_record, tmp_path):
        """H2 with 3 alpha values — tests the full scan loop."""
        grid = np.array([0.8, 1.0, 1.5])
        summary = run_molecule_scan(
            h2_record, str(tmp_path), grid=grid,
        )
        assert summary['molecule_id'] == 'H2_singlet_Dooh'
        assert summary['n_points_computed'] == 3
        assert summary['n_failed'] == 0

        # Check output files exist
        mol_dir = tmp_path / 'H2_singlet_Dooh'
        assert (mol_dir / 'alpha_0.800.json').exists()
        assert (mol_dir / 'alpha_1.000.json').exists()
        assert (mol_dir / 'alpha_1.500.json').exists()

    def test_single_atom_only_alpha_1(self, be_record, tmp_path):
        """Single atoms should only compute alpha=1.0."""
        summary = run_molecule_scan(
            be_record, str(tmp_path),
        )
        assert summary['n_points_computed'] == 1

        mol_dir = tmp_path / 'Be_singlet_Kh'
        assert (mol_dir / 'alpha_1.000.json').exists()

    def test_resumability(self, h2_record, tmp_path):
        """Running twice should skip already-computed points."""
        grid = np.array([1.0])
        run_molecule_scan(h2_record, str(tmp_path), grid=grid)
        summary2 = run_molecule_scan(h2_record, str(tmp_path), grid=grid)
        assert summary2['n_points_skipped'] == 1
        assert summary2['n_points_computed'] == 0


class TestRunDatabasePipeline:
    def test_small_pipeline(self, sample_csv, tmp_path):
        """Run pipeline on H2 only from the CSV."""
        summary = run_database_pipeline(
            csv_path=sample_csv,
            output_dir=str(tmp_path),
            molecule_ids=['H2_singlet_Dooh'],
            grid=np.array([1.0]),
        )
        assert summary['n_molecules'] == 1
        assert (tmp_path / 'pipeline_summary.json').exists()
        assert (tmp_path / 'H2_singlet_Dooh' / 'alpha_1.000.json').exists()

    def test_skip_single_atoms(self, sample_csv, tmp_path):
        """Verify skip_single_atoms filters correctly."""
        summary = run_database_pipeline(
            csv_path=sample_csv,
            output_dir=str(tmp_path),
            skip_single_atoms=True,
            grid=np.array([1.0]),
        )
        # Should have H2 and LiH but not Be
        assert summary['n_molecules'] == 2
        assert not (tmp_path / 'Be_singlet_Kh').exists()
