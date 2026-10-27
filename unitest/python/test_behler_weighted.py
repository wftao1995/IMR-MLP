"""Current imrmlp weighted ACSF contracts, including periodic derivative folding."""
import numpy as np
import pytest

from imrmlp import Atoms, Descriptor


def params(weighted=True):
    result = {
        'behler:dispcount': '5', 'behler:dispselect': '1 2 3 4 5',
        'behler:cutoff': '5', 'behler:eta': '.2', 'behler:rshift': '.3',
        'behler:kppa': '.7', 'behler:zeta': '2', 'behler:lambda': '1',
        'behler:weighted': str(weighted).lower(),
    }
    if weighted:
        result['behler:weights'] = '1 8'
    return result


def atoms(periodic=False):
    return Atoms(numbers=[1, 8, 1], elements=['H', 'O'],
                 positions=[[0., 0., 0.], [1.2, .3, .1], [-.4, 1.6, .2]],
                 cell=np.diag([3.1, 4.7, 8.]), pbc=[periodic, False, False])


def reference(center_positions, types, weighted=True, weights=(1., 8.), shift=.3):
    """Independent ordered-neighbor reference, excluding source center."""
    result = np.zeros(5)
    radii = np.linalg.norm(center_positions, axis=1)
    fc = lambda r: .5 * (np.cos(np.pi * r / 5.) + 1.) if r < 5. else 0.
    for i, (r, p) in enumerate(zip(radii, center_positions)):
        w = weights[types[i]] if weighted else 1.
        result[0] += w * fc(r)
        result[1] += w * np.exp(-.2 * (r-shift)**2) * fc(r)
        result[2] += w * np.cos(.7*r) * fc(r)
        for j, (t, q) in enumerate(zip(radii, center_positions)):
            if i == j or (not weighted and j <= i):
                continue
            wj = weights[types[j]] if weighted else 1.
            s = shift if weighted else 0.
            d = np.linalg.norm(p-q)
            angular = .5 * (1. + np.dot(p, q)/(r*t))**2
            radial = (r-s)**2 + (t-s)**2
            result[3] += w*wj*angular*np.exp(-.2*(radial+(d-s)**2))*fc(r)*fc(t)*fc(d)
            result[4] += w*wj*angular*np.exp(-.2*radial)*fc(r)*fc(t)
    return result


@pytest.mark.parametrize('weighted', [False, True])
def test_behler_weighted_hand_values_and_center_independence(weighted):
    a = atoms()
    d = Descriptor('behler', params(weighted))
    result = d.compute(a)[0]
    np.testing.assert_allclose(result, reference(a.positions[1:], a.chemmap[1:], weighted), atol=1e-13)
    changed = a.copy()
    changed.chemmap = [1, 1, 0]
    np.testing.assert_array_equal(d.compute(changed)[0], result)
    assert d.feature_size == 5


def test_behler_elements_explicit_override_and_type_remapping():
    a = atoms()
    p = params()
    p.pop('behler:weights')
    p['behler:elements'] = 'H O'
    d = Descriptor('behler', p)
    np.testing.assert_allclose(d.compute(a), Descriptor('behler', params()).compute(a), atol=1e-13)
    p['behler:weights'] = '2 3'
    custom = Descriptor('behler', p)
    np.testing.assert_allclose(custom.compute(a)[0], reference(a.positions[1:], a.chemmap[1:], weights=(2., 3.)))
    p['behler:weights'] = '3 2'
    p['behler:elements'] = 'O H'
    remapped = a.copy()
    remapped.chemmap = [1-t for t in a.chemmap]
    np.testing.assert_allclose(Descriptor('behler', p).compute(remapped), custom.compute(a), atol=1e-13)


def test_behler_parameter_rows_match_separate_features():
    a = atoms()
    p = params()
    rows = [(2., .1, .0, .3, 0., -1.), (3., .2, .1, .4, 1., 1.),
            (4., .3, .2, .5, 2., -1.), (5., .4, .3, .6, 3., 1.),
            (6., .5, .4, .7, 4., -1.)]
    for col, key in enumerate(('rcs', 'etas', 'rshifts', 'kppas', 'zetas', 'lambdas')):
        p['behler:'+key] = ' '.join(str(row[col]) for row in rows)
    combined = Descriptor('behler', p).compute(a)
    for f, row in enumerate(rows):
        single = params()
        single.update({'behler:dispcount': '1', 'behler:dispselect': str(f+1)})
        for key, val in zip(('cutoff', 'eta', 'rshift', 'kppa', 'zeta', 'lambda'), row):
            single['behler:'+key] = str(val)
        np.testing.assert_allclose(combined[:, f], Descriptor('behler', single).compute(a)[:, 0], atol=1e-12)


@pytest.mark.parametrize('periodic', [False, True])
def test_behler_weighted_position_mean_and_strain_jacobians(periodic):
    a = atoms(periodic)
    d = Descriptor('behler', params())
    value, indices, jac = d.compute_with_gradient(a, 0)
    mean, mean_jac = d.compute_averaged_with_gradient(a)
    strain_value, strain_jac = d.compute_with_strain_gradient(a, 0)
    mean_strain_value, mean_strain_jac = d.compute_averaged_with_strain_gradient(a)
    np.testing.assert_allclose(value, strain_value)
    np.testing.assert_allclose(mean, mean_strain_value)
    for step in (1e-4, 1e-5):
        for i in range(len(a)):
            for axis in range(3):
                plus, minus = a.copy(), a.copy()
                pp, pm = a.positions.copy(), a.positions.copy()
                pp[i, axis] += step
                pm[i, axis] -= step
                plus.positions, minus.positions = pp, pm
                fd = (d.compute(plus)[0]-d.compute(minus)[0])/(2*step)
                np.testing.assert_allclose(jac[:, np.flatnonzero(indices == i)[0], axis], fd, rtol=2e-5, atol=3e-6)
                fd_mean = (d.compute_averaged(plus)-d.compute_averaged(minus))/(2*step)
                np.testing.assert_allclose(mean_jac[:, i, axis], fd_mean, rtol=2e-5, atol=3e-6)
        for x in range(3):
            for y in range(3):
                plus, minus = a.copy(), a.copy()
                strain = np.zeros((3, 3))
                strain[x, y] = step
                plus.positions = a.positions @ (np.eye(3)+strain).T
                minus.positions = a.positions @ (np.eye(3)-strain).T
                plus.cell = a.cell @ (np.eye(3)+strain).T
                minus.cell = a.cell @ (np.eye(3)-strain).T
                fd = (d.compute(plus)[0]-d.compute(minus)[0])/(2*step)
                np.testing.assert_allclose(strain_jac[:, x, y], fd, rtol=3e-5, atol=4e-6)
                fd_mean = (d.compute_averaged(plus)-d.compute_averaged(minus))/(2*step)
                np.testing.assert_allclose(mean_strain_jac[:, x, y], fd_mean, rtol=3e-5, atol=4e-6)


@pytest.mark.parametrize('updates', [
    {'behler:weights': ''}, {'behler:weights': 'nan 1'},
    {'behler:elements': 'H'}, {'behler:rcs': '2 3'},
    {'behler:etas': '-1 -1 -1 -1 -1'}, {'behler:elements': 'H Unknown'},
])
def test_behler_weighted_rejects_invalid_configuration(updates):
    p = params()
    p.update(updates)
    with pytest.raises(Exception):
        Descriptor('behler', p)


def test_behler_weighted_checks_type_table_on_all_python_paths():
    a = atoms()
    a.chemmap = [0, 2, 0]
    d = Descriptor('behler', params())
    for call in (lambda: d.compute(a), lambda: d.compute_averaged(a),
                 lambda: d.compute_with_gradient(a, 0),
                 lambda: d.compute_averaged_with_gradient(a),
                 lambda: d.compute_with_strain_gradient(a, 0)):
        with pytest.raises(Exception, match='species|weights'):
            call()


def test_behler_weighted_real_cli_matches_current_python(tmp_path):
    import re
    from test_cli_error_paths import run_command

    a = atoms()
    p = params()
    p['behler:rcs'] = '3 4 5 4 5'
    p['behler:etas'] = '.1 .2 .3 .4 .5'
    p['behler:elements'] = 'H O'
    (tmp_path/'index.map').write_text('H 0\nO 1\n')
    rows = ['3', 'pbc="F F F" Properties=species:S:1:pos:R:3']
    for symbol, xyz in zip(('H', 'O', 'H'), a.positions):
        rows.append(symbol+' '+' '.join(map(str, xyz)))
    (tmp_path/'input.xyz').write_text('\n'.join(rows)+'\n')
    (tmp_path/'settings.ini').write_text(''.join(f'{k} {v}\n' for k, v in p.items()))
    result = run_command('imrmlp', [
        'calc_descriptors', '--dtype', 'behler', '--input', 'input.xyz',
        '--input-format', 'extxyz', '--settings', 'settings.ini',
        '--output', 'output.xyz', '--local-name', 'wacsf_local',
        '--global-name', 'wacsf_global', '--lossless', '1', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    lines = (tmp_path/'output.xyz').read_text().splitlines()
    match = re.search(r'\bwacsf_global="([^"]*)"', lines[1])
    assert match, lines[1]
    global_values = np.fromstring(match.group(1), sep=' ')
    properties = re.search(r'\bProperties=([^\s]+)', lines[1]).group(1).split(':')
    offset, span = 0, None
    for i in range(0, len(properties), 3):
        count = int(properties[i+2])
        if properties[i] == 'wacsf_local':
            span = slice(offset, offset+count)
        offset += count
    assert span is not None, lines[1]
    local = np.array([[float(t) for t in row.split()[span]] for row in lines[2:5]])
    d = Descriptor('behler', p)
    np.testing.assert_allclose(local, d.compute(a), rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(global_values, d.compute_averaged(a), rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize('weighted', [False, True])
def test_behler_g4_zero_shift_has_smooth_coincident_neighbor_limit(weighted):
    a = Atoms(numbers=[1, 1, 8], elements=['H', 'O'],
              positions=[[0., 0., 0.], [1., 0., 0.], [1., 0., 0.]],
              cell=np.eye(3)*20., pbc=False)
    p = params(weighted)
    p.update({'behler:dispcount': '1', 'behler:dispselect': '4', 'behler:rshift': '0'})
    d = Descriptor('behler', p)
    value, indices, jac = d.compute_with_gradient(a, 0)
    fc = .5*(np.cos(np.pi/5.)+1.)
    scale = 16. if weighted else 1.
    np.testing.assert_allclose(value, [scale*2.*np.exp(-.4)*fc*fc], atol=1e-12)
    assert np.isfinite(jac).all()
    for step in (1e-4, 1e-5):
        for atom in (1, 2):
            for axis in range(3):
                plus, minus = a.copy(), a.copy()
                pp, pm = a.positions.copy(), a.positions.copy()
                pp[atom, axis] += step
                pm[atom, axis] -= step
                plus.positions, minus.positions = pp, pm
                vp = d.compute_with_gradient(plus, 0)[0]
                vm = d.compute_with_gradient(minus, 0)[0]
                np.testing.assert_allclose(jac[:, np.flatnonzero(indices == atom)[0], axis],
                                           (vp-vm)/(2*step), atol=3e-6)
    if weighted:
        p['behler:rshift'] = '.3'
        with pytest.raises(Exception, match='cusp'):
            Descriptor('behler', p).compute_with_gradient(a, 0)
