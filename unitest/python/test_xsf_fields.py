"""XSF values use logical C order; literals independently specify x-fast files."""
import io
import numpy as np
import pytest
from imrmlp.xsf_fields import (GridLayout, ScalarField, DataGridBlock,
                             read_xsf_fields, write_xsf_fields)

GRID = """BEGIN_BLOCK_DATAGRID_3D
block#1
BEGIN_DATAGRID_3D_rho#1
2 3 4
1 2 3
2 .5 0
0 3 .25
.5 0 4
0 100 10 110 20 120
1 101 11 111 21 121
2 102 12 112 22 122
3 103 13 113 23 123
END_DATAGRID_3D
END_BLOCK_DATAGRID_3D
"""
PLANE = """BEGIN_BLOCK_DATAGRID_2D
plane
BEGIN_DATAGRID_2D_phi
2 3
0 0 1
2 1 0
0 3 1
0 100 10 110 20 120
END_DATAGRID_2D
END_BLOCK_DATAGRID_2D
"""


def oracle():
    return np.fromfunction(lambda i, j, k: 100*i + 10*j + k, (2, 3, 4))


def test_literal_geometry_prefix_multiblock_and_boundaries():
    blocks = read_xsf_fields(io.StringIO("ATOMS\nH 0 0 0\n" + GRID + "# comment\n" + PLANE))
    assert len(blocks) == 2
    b = blocks[0]
    assert b.identifier == "block#1" and b.fields[0].identifier == "rho#1"
    np.testing.assert_array_equal(b.fields[0].values, oracle())
    assert b.fields[0].values.flags.c_contiguous
    assert b.fields[0].values.flags.owndata
    np.testing.assert_array_equal(b.layout.coordinate((1, 2, 3)), [3.5, 5.5, 7.25])
    np.testing.assert_array_equal(blocks[1].fields[0].values, [[0,10,20],[100,110,120]])
    np.testing.assert_array_equal(blocks[1].layout.coordinate((1,2)), [2,4,2])


def test_write_oracle_and_vector_view():
    layout = GridLayout((2,3,4), [1,2,3], [[2,.5,0],[0,3,.25],[.5,0,4]])
    fields = [ScalarField(name, oracle() + offset) for name, offset in zip(("vx", "vy", "vz"), (0,1,2))]
    b = DataGridBlock("velocity", layout, fields)
    view = b.vector_view(("vx", "vy", "vz"))
    assert view.components[0] is fields[0]
    fields[0].values[0,0,0] = 42
    assert view.values[0,0,0,0] == 42
    fields[0].values[0,0,0] = 0
    out = io.StringIO()
    write_xsf_fields(out, [b])
    assert "0 100 10 110 20 120\n1 101 11 111 21 121\n" in out.getvalue()
    again = read_xsf_fields(io.StringIO(out.getvalue()))[0]
    assert [f.identifier for f in again.fields] == ["vx", "vy", "vz"]
    np.testing.assert_array_equal(again.fields[1].values, oracle()+1)
    with pytest.raises(ValueError):
        b.vector_view(("vx", "missing", "vz"))


@pytest.mark.parametrize("suffix", [".xsf", ".xsf.gz"])
def test_paths_compressed_and_failed_validation_preserve_file(tmp_path, suffix):
    path = tmp_path / ("field" + suffix)
    b = read_xsf_fields(io.StringIO(GRID))[0]
    write_xsf_fields(path, [b])
    np.testing.assert_array_equal(read_xsf_fields(path)[0].fields[0].values, oracle())
    b.fields[0].values[0,0,0] = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        write_xsf_fields(path, [b])
    np.testing.assert_array_equal(read_xsf_fields(path)[0].fields[0].values, oracle())


def test_singletons_and_validation():
    layout = GridLayout((1,3), [1,2,3], [[100,200,300], [0,4,2]])
    np.testing.assert_array_equal(layout.coordinate((0,2)), [1,6,5])
    with pytest.raises(ValueError):
        layout.coordinate((1,2))
    with pytest.raises(ValueError):
        DataGridBlock("bad", layout, [ScalarField("f", np.zeros((2,3)))])
    with pytest.raises((RuntimeError, ValueError)):
        GridLayout((0,3), [0,0,0], [[1,0,0],[0,1,0]])
    out = io.StringIO()
    with pytest.raises(RuntimeError):
        write_xsf_fields(out, [])
    assert out.getvalue() == ""
    with pytest.raises(TypeError):
        read_xsf_fields(io.BytesIO(GRID.encode()))


@pytest.mark.parametrize("text", ["", "ANIMSTEPS 1\n"+GRID,
    GRID.replace("2 3 4", "-2 3 4"), GRID.replace("2 3 4", "18446744073709551615 3 4"),
    GRID.replace("0 100", "nan 100"), GRID.replace("23 123", "23"),
    GRID.replace("23 123", "23 123 999"), GRID.replace("END_DATAGRID_3D", "END_DATAGRID_2D"),
    GRID.replace("END_BLOCK_DATAGRID_3D", "END_BLOCK_DATAGRID_2D")])
def test_malformed(text):
    with pytest.raises(RuntimeError):
        read_xsf_fields(io.StringIO(text))


def test_nonseekable_text_stream():
    class ReadOnly:
        def read(self):
            return GRID
    np.testing.assert_array_equal(read_xsf_fields(ReadOnly())[0].fields[0].values, oracle())
