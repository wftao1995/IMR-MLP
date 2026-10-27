"""Fixed-cell molecular NEB uses Cartesian coordinates without a padding cell."""
import numpy as np
import pytest
from ase import Atoms as ASEAtoms
from imrmlp import Atoms, NEB, Potential
from test_cli_error_paths import run_command


@pytest.mark.parametrize('from_ase', [False, True])
@pytest.mark.parametrize('optimizer', ['fire', 'lbfgs'])
@pytest.mark.parametrize('budget', [1, 3])
@pytest.mark.parametrize('init_method', ['linear', 'idpp'])
def test_molecular_neb_matches_padding_and_preserves_endpoints(from_ase, optimizer, budget, init_method):
    potential = Potential.from_settings({'pottype': 'lj612', 'ntypes': 1,
                                         '0-0': '1 1', 'cut:0-0': 3})
    initial_positions = [[0., 0., 0.], [1.3, 0., 0.]]
    final_positions = [[0., 0., 0.], [.8, 1.2, .1]]
    if from_ase:
        initial = Atoms.from_ase(ASEAtoms('H2', positions=initial_positions))
        final = Atoms.from_ase(ASEAtoms('H2', positions=final_positions))
    else:
        initial = Atoms(numbers=[1, 1], elements=['H'], positions=initial_positions,
                        cell=np.zeros((3, 3)), pbc=False)
        final = Atoms(numbers=[1, 1], elements=['H'], positions=final_positions,
                      cell=np.zeros((3, 3)), pbc=False)
    # ASE imports retain atomic-number IDs until an explicit potential mapping.
    initial.chemmap = [0, 0]
    final.chemmap = [0, 0]
    saved = initial.copy(), final.copy()
    padded_initial, padded_final = initial.copy(), final.copy()
    padded_initial.cell = np.eye(3) * 20
    padded_final.cell = np.eye(3) * 20
    driver = NEB(potential, n_images=3, optimizer=optimizer, max_iter=budget,
                 max_move=.03, neb_ftol=1e-6, init_method=init_method)
    actual = driver.run(initial, final)
    expected = driver.run(padded_initial, padded_final)
    assert actual.stop_reason == expected.stop_reason
    assert actual.converged == expected.converged
    assert actual.n_iter == expected.n_iter
    assert actual.max_neb_force == pytest.approx(expected.max_neb_force, abs=1e-8)
    assert len(actual.images) == len(expected.images) == 3
    for image, padded in zip(actual.images, expected.images):
        np.testing.assert_array_equal(image.cell, np.zeros((3, 3)))
        np.testing.assert_array_equal(image.pbc, [False, False, False])
        np.testing.assert_allclose(image.positions, padded.positions, atol=1e-8, rtol=0)
        assert np.isfinite(image.energy)
        recomputed = image.copy()
        potential.calculate(recomputed)
        assert image.energy == pytest.approx(recomputed.energy, abs=1e-10)
    for endpoint, source in zip((initial, final), saved):
        np.testing.assert_array_equal(endpoint.positions, source.positions)
        np.testing.assert_array_equal(endpoint.cell, source.cell)
        np.testing.assert_array_equal(endpoint.pbc, source.pbc)


def test_cell_free_neb_cli_retains_extxyz_geometry_and_cartesian_image_outputs(tmp_path):
    results = []
    for padded in (False, True):
        work = tmp_path / ('padded' if padded else 'molecule')
        work.mkdir()
        (work / 'index.map').write_text('H\n')
        cell = '20 0 0 0 20 0 0 0 20' if padded else '0 0 0 0 0 0 0 0 0'
        for image, position in ((0, [1.3, 0., 0.]), (2, [.8, 1.2, .1])):
            (work / f'image{image}.xyz').write_text(
                f'2\nLattice="{cell}" pbc="F F F" Properties=species:S:1:pos:R:3\n'
                'H 0 0 0\nH ' + ' '.join(map(str, position)) + '\n')
        (work / 'neb.ini').write_text('\n'.join([
            'pot:pottype lj612', 'pot:ntypes 1', 'pot:0-0 1 1', 'pot:cut:0-0 3',
            'nimgs 3', 'input-format extxyz', 'driver fire', 'max_total_iter 3',
            'maxiteration 3', 'max_move .03', 'neb:gtol 1e-6',
            'image:0 image0.xyz', 'image:2 image2.xyz', 'neb:climb false',
        ]) + '\n')
        completed = run_command('imrmlp', ['neb', '--settings', 'neb.ini', '--num_threads', '1'], work)
        assert completed.returncode in (0, 2), completed.stdout + completed.stderr
        lines = (work / 'mep.xyz').read_text().splitlines()
        assert len(lines) == 12
        assert all('pbc="F F F"' in lines[4 * i + 1] for i in range(3))
        assert 'nan' not in '\n'.join(lines).lower()
        positions = np.array([list(map(float, lines[4 * i + j].split()[1:4]))
                              for i in range(3) for j in (2, 3)])
        for i in range(3):
            image_output = (work / f'nebimage_{i}_final.vasp').read_text()
            assert 'Cartesian' in image_output
            assert 'nan' not in image_output.lower()
        rows = [line.split() for line in (work / 'neb.dat').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
        irc = np.array([float(row[1]) for row in rows])
        assert np.all(np.isfinite(irc))
        assert irc[-1] > 0
        results.append((completed.returncode, positions, irc))
    assert results[0][0] == results[1][0]
    np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(results[0][2], results[1][2], atol=1e-6, rtol=0)
