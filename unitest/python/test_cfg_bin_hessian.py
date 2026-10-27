"""Dense Hessian persistence in existing named Feature records."""
import gzip
import struct
import numpy as np
import pytest
pytest.importorskip('imrmlp._imrmlp')
import imrmlp

MATRIX=np.arange(36,dtype=float).reshape(6,6)-17

def fixture(format,payload,duplicate=False):
    if format=='cfg':
        body=('BEGIN_CFG\nSize\n2\nSupercell\n8 0 0\n0 9 0\n0 0 10\n'
              'AtomData: id type cartes_x cartes_y cartes_z\n1 1 1 2 3\n2 8 4 5 6\n'
              'Feature hessian '+payload+'\n')
        return (body+('Feature hessian '+payload+'\n' if duplicate else '')+'END_CFG\n').encode()
    def string(text):
        value=text.encode(); return struct.pack('=i',len(value))+value
    feature=b'F'+string('hessian')+string(payload)
    return (b'GPMLBIN2'+struct.pack('=Ii',3,2)+b'l'+struct.pack('=9d',8,0,0,0,9,0,0,0,10)
            +b'p'+struct.pack('=6d',1,2,3,4,5,6)+b't'+struct.pack('=2i',1,8)
            +feature+(feature if duplicate else b'')+b'#')

@pytest.mark.parametrize('format',['cfg','bin'])
def test_read_independent_dense_feature_and_raw_write_order(tmp_path,format):
    payload=' '.join(str(int(v)) for v in MATRIX.ravel())
    path=tmp_path/('independent.'+format); path.write_bytes(fixture(format,payload))
    a=imrmlp.read_one(path,format=format)
    np.testing.assert_array_equal(a.hessian,MATRIX)
    out=tmp_path/('out.'+format); imrmlp.write(out,a,format=format)
    text=out.read_bytes()
    if format=='cfg': assert ('Feature hessian '+payload+'\n').encode() in text
    else:
        assert text[:8]==b'GPMLBIN2' and struct.unpack_from('=I',text,8)[0]==3
        at=text.index(b'hessian')+7; size=struct.unpack_from('=i',text,at)[0]
        assert text[at+4:at+4+size].decode()==payload

@pytest.mark.parametrize('format',['cfg','mlpcfg','bin','mlpbin'])
@pytest.mark.parametrize('suffix',['','.gz'])
def test_dense_hessian_bounded_write_read_and_owned_frames(tmp_path,format,suffix):
    a=imrmlp.Atoms(numbers=[1,8],positions=[[1,2,3],[4,5,6]],cell=np.eye(3)*10)
    a.set_hessian(MATRIX+.125)
    directions=np.arange(12,dtype=float).reshape(6,2)/7
    a.set_hvp(directions,1.3*directions)
    b=a.copy(); b.set_hessian(-MATRIX+.25)
    target=tmp_path/('trajectory.'+format+suffix)
    seen=[]
    def generate():
        for frame in [a,b]: seen.append(len(seen)); yield frame
    imrmlp.write_iter(target,generate(),format=format)
    assert seen==[0,1]
    iterator=imrmlp.iread(target,format=format)
    first=next(iterator); second=next(iterator)
    with pytest.raises(StopIteration): next(iterator)
    np.testing.assert_array_equal(first.hessian,MATRIX+.125)
    np.testing.assert_array_equal(second.hessian,-MATRIX+.25)
    np.testing.assert_array_equal(first.hvp_targets,1.3*directions)
    np.testing.assert_array_equal(a.hessian,MATRIX+.125)
    second.clear_hessian(); np.testing.assert_array_equal(first.hessian,MATRIX+.125)

@pytest.mark.parametrize('format',['cfg','bin'])
@pytest.mark.parametrize('payload',['user acoustic notes','1 2 3','nan 0 0 0 1 0 0 0 1'])
def test_legacy_arbitrary_hessian_feature_keeps_metadata_contract(tmp_path,format,payload):
    path=tmp_path/('legacy.'+format); path.write_bytes(fixture(format,payload))
    a=imrmlp.read_one(path,format=format); assert a.hessian.shape==(0,0)
    out=tmp_path/('legacy-out.'+format); imrmlp.write(out,a,format=format)
    assert payload.encode() in out.read_bytes()
    # Canonical labels take precedence while the source legacy object is usable.
    a.set_hessian(MATRIX); imrmlp.write(out,a,format=format)
    assert out.read_bytes().count(b'hessian') == 1
    np.testing.assert_array_equal(imrmlp.read_one(out,format=format).hessian,MATRIX)

@pytest.mark.parametrize('format',['cfg','bin'])
def test_duplicate_canonical_feature_is_late_stream_error(tmp_path,format):
    payload=' '.join(str(v) for v in MATRIX.ravel())
    path=tmp_path/('bad-later.'+format)
    path.write_bytes(fixture(format,payload)+fixture(format,payload,True))
    iterator=imrmlp.iread(path,format=format); first=next(iterator)
    with pytest.raises((RuntimeError,ValueError),match='Duplicate'): next(iterator)
    np.testing.assert_array_equal(first.hessian,MATRIX)
