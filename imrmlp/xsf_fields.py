"""Independent static XSF 2D/3D scalar grids, including boundary samples.

Values have logical shape ``(nx, ny[, nz])`` in C order. File I/O explicitly
converts XSF's x-fast order. Coordinates and spans are in Angstroms; scalar
units are caller defined and never inferred or converted. A singleton axis
samples its origin (zero contribution from that axis's span).

``read_xsf_fields`` extracts only fields from a static geometry+fields file.
AXSF and BXSF are unsupported. Text file objects are accepted; paths also
support gzip/zstd through the native compressed-stream implementation.
"""
from dataclasses import dataclass
import operator
import os
import numpy as np
from . import _imrmlp

__all__ = ["GridLayout", "ScalarField", "DataGridBlock", "VectorFieldView",
           "read_xsf_fields", "write_xsf_fields"]


def _identifier(value):
    if not isinstance(value, str) or not value or value.startswith("#") or any(c.isspace() for c in value):
        raise ValueError("identifier must be a nonempty single word, not a comment")
    return value


@dataclass
class GridLayout:
    """Grid shape, origin and one 3-vector span per logical axis."""
    shape: tuple
    origin: np.ndarray
    spanning_vectors: np.ndarray

    def __post_init__(self):
        self.shape = tuple(operator.index(n) for n in self.shape)
        if len(self.shape) not in (2, 3) or any(n <= 0 for n in self.shape):
            raise ValueError("shape must have two or three positive integer axes")
        self.origin = np.array(self.origin, dtype=float, copy=True, order="C")
        self.spanning_vectors = np.array(self.spanning_vectors, dtype=float, copy=True, order="C")
        if self.origin.shape != (3,) or self.spanning_vectors.shape != (len(self.shape), 3):
            raise ValueError("origin must have shape (3,) and spanning_vectors (dimension, 3)")
        self._native().validate()

    @property
    def dimension(self):
        return len(self.shape)

    def _native(self):
        result = _imrmlp._XsfGridLayout()
        result.shape = self.shape
        result.origin = self.origin.tolist()
        result.spanning_vectors = self.spanning_vectors.tolist()
        return result

    def coordinate(self, indices):
        """Return origin + sum(i/(n-1)*span); singleton axes contribute zero."""
        indices = tuple(operator.index(i) for i in indices)
        if len(indices) != self.dimension or any(i < 0 or i >= n for i, n in zip(indices, self.shape)):
            raise ValueError("coordinate index out of range or wrong dimension")
        return np.array(self._native().coordinate(indices))

    def same_as(self, other):
        return (isinstance(other, GridLayout) and self.shape == other.shape
                and np.array_equal(self.origin, other.origin)
                and np.array_equal(self.spanning_vectors, other.spanning_vectors))


@dataclass
class ScalarField:
    """A named scalar component with an owning C-contiguous ndarray."""
    identifier: str
    values: np.ndarray

    def __post_init__(self):
        _identifier(self.identifier)
        self.values = np.array(self.values, dtype=float, order="C", copy=True)
        if self.values.ndim not in (2, 3) or not np.all(np.isfinite(self.values)):
            raise ValueError("scalar values must be a finite 2D or 3D array")


@dataclass
class DataGridBlock:
    """An ordered list of scalar fields sharing exactly one layout."""
    identifier: str
    layout: GridLayout
    fields: list

    def __post_init__(self):
        self.fields = list(self.fields)
        self._native().validate()

    def _native(self):
        _identifier(self.identifier)
        if not isinstance(self.layout, GridLayout):
            raise TypeError("layout must be a GridLayout")
        result = _imrmlp._XsfDataGridBlock()
        result.identifier = self.identifier
        result.layout = self.layout._native()
        fields = []
        for field in self.fields:
            if not isinstance(field, ScalarField):
                raise TypeError("fields must contain ScalarField objects")
            _identifier(field.identifier)
            if field.values.shape != self.layout.shape:
                raise ValueError("field shape must equal layout shape")
            native_field = _imrmlp._XsfScalarField()
            native_field.identifier = field.identifier
            native_field.values = field.values.ravel(order="C").tolist()
            fields.append(native_field)
        result.fields = fields
        return result

    def vector_view(self, components):
        """Select three explicit component names; no vector record is written."""
        components = tuple(components)
        if len(components) != 3:
            raise ValueError("a vector view requires three component identifiers")
        selected = []
        for name in components:
            matches = [field for field in self.fields if field.identifier == name]
            if len(matches) != 1:
                raise ValueError("each component name must identify exactly one scalar field")
            selected.append(matches[0])
        return VectorFieldView(self.layout, tuple(selected))


@dataclass(frozen=True)
class VectorFieldView:
    """References to three scalar components with a common layout.

    ``components`` references the original fields; ``values`` returns a fresh
    array with a final Cartesian component axis. XSF stores the three named
    scalar fields, with no invented vector syntax.
    """
    layout: GridLayout
    components: tuple

    def __post_init__(self):
        if not isinstance(self.layout, GridLayout) or len(self.components) != 3:
            raise ValueError("a vector view requires one layout and three scalar fields")
        if any(not isinstance(f, ScalarField) or f.values.shape != self.layout.shape for f in self.components):
            raise ValueError("vector components must have the common layout shape")

    @property
    def values(self):
        return np.stack([field.values for field in self.components], axis=-1)


def _from_native(block):
    layout = GridLayout(tuple(block.layout.shape), block.layout.origin, block.layout.spanning_vectors)
    fields = [ScalarField(f.identifier, np.array(f.values).reshape(layout.shape, order="C")) for f in block.fields]
    return DataGridBlock(block.identifier, layout, fields)


def read_xsf_fields(source):
    """Read fields from a path or readable text stream; discard geometry."""
    if hasattr(source, "read"):
        text = source.read()
        if not isinstance(text, str):
            raise TypeError("XSF file objects must be opened in text mode")
        blocks = _imrmlp._parse_xsf_fields(text)
    else:
        blocks = _imrmlp._read_xsf_fields(os.fsdecode(os.fspath(source)))
    return [_from_native(block) for block in blocks]


def write_xsf_fields(destination, blocks):
    """Write validated blocks to a path or writable text stream.

    A stream may already contain static XSF geometry. Validation completes
    before any field bytes are emitted. Existing paths are not truncated for
    invalid blocks. Output keeps every boundary point and scalar component.
    """
    native = [block._native() for block in blocks]
    # Native formatter validates the whole sequence before stream writes too.
    if hasattr(destination, "write"):
        text = _imrmlp._format_xsf_fields(native)
        count = destination.write(text)
        if count is not None and count != len(text):
            raise OSError("incomplete XSF stream write")
    else:
        _imrmlp._write_xsf_fields(os.fsdecode(os.fspath(destination)), native)
