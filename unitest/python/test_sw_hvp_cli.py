"""The actual calc_fc matrix-free SW path agrees with total-energy differences."""
import itertools
import numpy as np
import pytest
from test_cli_error_paths import run_command


def energy(positions):
    # This fixture has identical interactions for both species. Enumerate
    # physical unordered pairs and the three centered triplets independently.
    value = 0.
    for i, j in itertools.combinations(range(3), 2):
        r = np.linalg.norm(positions[j]-positions[i])
        value += 2 * (.7*(.8/r)**4 - .8/r) * np.exp(.8/(r-4.5*.8))
    for center in range(3):
        j, k = [i for i in range(3) if i != center]
        x, y = positions[j]-positions[center], positions[k]-positions[center]
        r, s = np.linalg.norm(x), np.linalg.norm(y)
        cosine = np.dot(x, y)/(r*s)
        value += .9*1.3/3 * np.exp(.9*.75/(r-4.8*.75) + .9*.75/(s-4.8*.75)) * (cosine+.2)**2
    return value


def energy_hessian(positions, step):
    flat = positions.flatten()
    center = energy(positions)
    result = np.zeros((9, 9))
    for i in range(9):
        ei = np.eye(9)[i]*step
        result[i, i] = (energy((flat+ei).reshape(3, 3))-2*center
                        + energy((flat-ei).reshape(3, 3)))/step**2
        for j in range(i):
            ej = np.eye(9)[j]*step
            result[i, j] = result[j, i] = (
                energy((flat+ei+ej).reshape(3, 3))-energy((flat+ei-ej).reshape(3, 3))
                + energy((flat-ei-ej).reshape(3, 3))-energy((flat-ei+ej).reshape(3, 3)))/(4*step**2)
    return result


@pytest.mark.parametrize('method', ['analytic', 'auto'])
def test_sw_lowest_modes_match_two_step_independent_energy_oracle(tmp_path, method):
    rows = [f'{i} {j} 2body 1 .8 2 .7 4 1 4.5 3.4'
            for i, j in itertools.combinations_with_replacement(['H', 'He'], 2)]
    rows += [f'{center} {j} {k} .9 .75 4.8 1.3 .9 -.2 3.25'
             for center in ['H', 'He']
             for j, k in itertools.combinations_with_replacement(['H', 'He'], 2)]
    (tmp_path/'model.sw').write_text('\n'.join(rows)+'\n')
    (tmp_path/'index.map').write_text('H 1 2\nHe 1 5\n')
    positions = np.array([[0., 0., 0.], [1.3, .2, .1], [.4, 1.6, .3]])
    (tmp_path/'input.xyz').write_text(
        '3\nLattice="20 0 0 0 20 0 0 0 20" pbc="F F F" '
        'Properties=species:S:1:pos:R:3\nH 0 0 0\nHe 1.3 .2 .1\nH .4 1.6 .3\n')
    result = run_command('imrmlp', ['calc_fc', '--potfile=model.sw', '--potential-type=sw',
        '--input=input.xyz', '--input-format=extxyz', '--index=index.map',
        f'--hessian-method={method}', '--lowest-modes=2', '--lowest-mode-tol=1e-10',
        '--num_threads=1'], tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'Hessian-vector products' in result.stdout
    assert not (tmp_path/'FORCE_CONSTANTS').exists()
    values = np.loadtxt(tmp_path/'EIGENVALS', skiprows=1)[:, 0]
    roots = np.repeat(np.sqrt([2., 5., 2.]), 3)
    translations = np.zeros((9, 3))
    for i in range(3):
        translations[3*i:3*i+3] = np.eye(3)*roots[3*i]
    internal = np.linalg.qr(translations, mode='complete')[0][:, 3:]
    expected = []
    for step in [1e-4, 5e-5]:
        hessian = energy_hessian(positions, step)
        dynamical = hessian / np.outer(roots, roots)
        expected.append(np.linalg.eigvalsh(internal.T@dynamical@internal)[:2])
        np.testing.assert_allclose(values, expected[-1], atol=2e-6, rtol=0)
    np.testing.assert_allclose(expected[0], expected[1], atol=2e-6, rtol=0)
