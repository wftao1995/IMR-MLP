"""Content detection is bounded, explicit and independent of file extensions."""
import gzip
import pytest
import imrmlp
from imrmlp._imrmlp import _GPIO


XYZ = '1\nProperties=species:S:1:pos:R:3 energy=2\nHe 1 2 3\n'


@pytest.mark.parametrize("compressed", [False, True])
def test_content_detection_and_explicit_import(tmp_path, compressed):
    source = tmp_path / ("misnamed.data.gz" if compressed else "misnamed.data")
    if compressed:
        with gzip.open(source, "wt") as out:
            out.write(XYZ)
    else:
        source.write_text(XYZ)
    assert imrmlp.detect_format(source) == {"format": "extxyz", "candidates": ["extxyz"]}
    for reader in [imrmlp.read_one, imrmlp.read, imrmlp.iread, imrmlp.open_reader]:
        result = reader(source, detect_content=True)
        if reader is imrmlp.read:
            atom = result[0]
        elif reader is imrmlp.read_one:
            atom = result
        else:
            try:
                atom = next(result)
            finally:
                result.close()
        assert atom.energy == 2
        assert atom.numbers.tolist() == [2]
    # Native sessions replay the inspected prefix without a second read/seek.
    native = _GPIO().open_reader(str(source), detect_content=True)
    assert native.format == "extxyz"
    assert native.read_next().energy == 2
    assert native.read_next() is None


def test_unknown_does_not_fall_back_to_misleading_name(tmp_path):
    source = tmp_path / "wrong.xyz"
    source.write_text("not a structure\n")
    assert imrmlp.detect_format(source) == {"format": "", "candidates": []}
    with pytest.raises(ValueError, match="Unknown"):
        imrmlp.read(source, detect_content=True)
    with pytest.raises(ValueError, match="detect"):
        _GPIO().open_reader(str(source), detect_content=True)


def test_conflicting_signatures_require_explicit_format(tmp_path):
    source = tmp_path / "ambiguous.xyz"
    source.write_text('1\nProperties=species:S:1:pos:R:3\nH 1 2 3\nATOMS\n')
    detected = imrmlp.detect_format(source)
    assert detected["format"] == ""
    assert set(detected["candidates"]) == {"xsf", "extxyz"}
    with pytest.raises(ValueError, match="Ambiguous"):
        imrmlp.read_one(source, detect_content=True)
    assert imrmlp.read_one(source, format="extxyz", detect_content=True).natoms == 1


def test_prefix_limit_and_unsupported_signatures(tmp_path):
    source = tmp_path / "plain"
    source.write_text(XYZ)
    assert imrmlp.detect_format(source, limit=2)["format"] == ""
    for invalid in [0, 65537]:
        with pytest.raises(ValueError, match="limit"):
            imrmlp.detect_format(source, limit=invalid)
    for text in ['BEGIN_BLOCK_BANDGRID_3D\n',
                 'BEGIN_BLOCK_DATAGRID_3D\n',
                 'title\n1.0\n1 0 0\n0 1 0\n0 0 1\nH\n1\nDirect\n0 0 0\n']:
        source.write_text(text)
        assert imrmlp.detect_format(source)["format"] == ""


def test_xml_is_detected_without_adding_stream_reader(tmp_path):
    source = tmp_path / "unknown"
    source.write_text('<?xml version="1.0"?>\n<modeling>\n</modeling>\n')
    assert imrmlp.detect_format(source)["format"] == "vasprun"
    with pytest.raises(ValueError, match="file-based"):
        imrmlp.read_one(source, detect_content=True)
    with pytest.raises(ValueError, match="streaming readers"):
        _GPIO().open_reader(str(source), detect_content=True)
