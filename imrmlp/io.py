# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Structure I/O — read and write atomic structures.

Wraps the C++ GPIO class. Public constants use its registered format names:
    mlpcfg, bin, extxyz, poscar, outcar, xdatcar, vasprun, vaspabn,
    lmptrj, lmpdat, yaml, cif, xsf, gro, cp2kxyz, n2p2,
    vtk, dlpoly, msi, xsd, car

OUTCAR, VASPRUN, LMPYAML, GRODATA, MSI, XSD and CAR are read-only;
VTK and DLPOLY4 are write-only. VASPRUN uses file-based ``read``; streaming reads for supported trajectory formats.

The old Python spellings ``mlpbin``, ``lmpyaml``, ``grodata``, ``n2p2dat`` and
``dlpoly4`` remain accepted as explicit format arguments. They select the same
native codecs and do not define new file formats or extensions.

Format auto-detection from file extension is automatic.
"""

from __future__ import annotations
from typing import Iterable, Iterator, List, Union, Optional
from pathlib import Path

from ._imrmlp import _GPIO, _GPIOStreamReader, _detect_format, canonicalize_atoms as _canonicalize_raw
from .atoms import Atoms
from .units import _source_units


# Format constants for explicit format specification
MLPCFG = "mlpcfg"
MLIPBIN = "bin"
EXTXYZ = "extxyz"
POSCAR = "poscar"
OUTCAR = "outcar"
XDATCAR = "xdatcar"
VASPRUN = "vasprun"
VASPABN = "vaspabn"
LMPTRJ = "lmptrj"
LMPDAT = "lmpdat"
LMPYAML = "yaml"
CIF = "cif"
XSF = "xsf"
GRODATA = "gro"
CP2KXYZ = "cp2kxyz"
N2P2DAT = "n2p2"
VTK = "vtk"
DLPOLY4 = "dlpoly"
MSI = "msi"
XSD = "xsd"
CAR = "car"


_FORMAT_ALIASES = {
    "mlpbin": MLIPBIN, "lmpyaml": LMPYAML, "grodata": GRODATA,
    "n2p2dat": N2P2DAT, "dlpoly4": DLPOLY4,
}
# Include native aliases so explicit names and auto-detection reject the same
# unsupported directions before opening or truncating a file. Tests compare
# these sets against GPIO's registration and dispatch tables.
_READ_ONLY_FORMATS = frozenset({
    "outcar", "OUTCAR", "xml", "vasprun", "vasprun.xml", "yaml", "gro",
    "msi", "xsd", "car", "poscars",
})
_WRITE_ONLY_FORMATS = frozenset({"vtk", "dlpoly", "DLPOLY"})
_READ_ALL_ONLY_FORMATS = frozenset({"xml", "vasprun", "vasprun.xml"})


def _checked_format(filename: Union[str, Path], format: str, operation: str) -> str:
    name = _FORMAT_ALIASES.get(format, format) if format else _GPIO.guess_format(str(filename))
    if operation == "write" and name in _READ_ONLY_FORMATS:
        raise ValueError(f"Format {name!r} does not support writing")
    if operation != "write" and name in _WRITE_ONLY_FORMATS:
        raise ValueError(f"Format {name!r} does not support reading")
    if operation == "read_one" and name in _READ_ALL_ONLY_FORMATS:
        raise ValueError(f"Format {name!r} requires the file-based reader; use read()")
    return name


def detect_format(filename: Union[str, Path], *, limit: int = 65536) -> dict:
    """Inspect up to 64 KiB of decoded content without using the filename.

    Return ``{"format": name, "candidates": [...]}``. Unknown content has no
    candidates; conflicting signatures have several candidates and no format.
    This is explicit content detection, separate from ``guess_format``.
    """
    return _detect_format(str(filename), limit)


def _input_format(filename, format, operation, detect_content):
    if detect_content and not format:
        found = detect_format(filename)
        if not found["format"]:
            reason = "Ambiguous" if found["candidates"] else "Unknown"
            raise ValueError(f"{reason} input content; specify format explicitly")
        format = found["format"]
    return _checked_format(filename, format, operation)


def _input_units(format, input_units):
    if input_units is None:
        return None
    if format not in {"extxyz", "xyz"}:
        raise ValueError("Input unit overrides are supported only for EXTXYZ")
    return _source_units(input_units)


_SESSION_READ_FORMATS = frozenset({
    "extxyz", "xyz", "cfg", "mlpcfg", "bin", "mlpbin", "n2p2", "n2p2dat",
    "xsf", "axsf", "outcar", "OUTCAR", "xdatcar", "xdatcar6",
})


class _ReaderSession:
    def __init__(self, native):
        self._native = native

    def __iter__(self):
        return self

    def __next__(self):
        raw = self._native.read_next()
        if raw is None:
            raise StopIteration
        return Atoms(_impl=raw)

    def close(self):
        self._native.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_reader(filename: Union[str, Path], format: str = "", *,
                detect_content: bool = False, input_units=None):
    """Open an independent bounded reader; use as an iterator or context manager.

    Supports EXTXYZ, CFG, BIN, N2P2, XSF, OUTCAR and XDATCAR/6. Each session
    owns its parser and stream; ``next`` and ``close`` serialize within one
    session. Separate sessions can run concurrently. Explicit ``format`` wins
    over content detection. Unit overrides apply only to EXTXYZ.
    """
    format = _input_format(filename, format, "read", detect_content)
    units = _input_units(format, input_units)
    io = _GPIO()
    return _ReaderSession(io.open_reader(str(filename), format, input_units=units))


class _WriterSession:
    def __init__(self, native, elements):
        self._native = native
        self._elements = elements

    def write(self, atoms):
        if atoms._elements != self._elements:
            raise ValueError("Atoms element mapping must match the writer's elements")
        self._native.write(atoms._impl)

    def close(self):
        self._native.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_writer(filename: Union[str, Path], format: str = "", *,
                elements=None, append: bool = False):
    """Open an independent lazy writer with checked close/finalization.

    Supports EXTXYZ, CFG, BIN, N2P2 and XDATCAR/6. ``elements`` specifies the
    same explicit element map as Atoms; None uses atomic-number types. Append
    is available only for EXTXYZ, CFG and BIN. No file is created until a frame
    is written. A late failure can leave the complete written prefix.
    """
    format = _checked_format(filename, format, "write")
    mapping = None if elements is None else list(elements)
    io = _GPIO()
    io.set_elements(mapping or [])
    io.append = append
    return _WriterSession(io.open_writer(str(filename), format), mapping)


def read(filename: Union[str, Path], format: str = "", *,
         detect_content: bool = False, input_units=None) -> List[Atoms]:
    """Read all structures from a file.

    GRO bulk reading retains only the first frame; use ``iread`` for a GRO
    trajectory.

    Args:
        filename: Path to the structure file.
        format: File format string. Auto-detected from extension if empty.
        detect_content: Explicit bounded content detection when format is empty.
        input_units: Explicit EXTXYZ source preset or base-unit mapping.

    Returns:
        List of Atoms objects (may be empty if file has no structures).

    Raises:
        ValueError: If the format only supports writing.

    Examples
    --------
    >>> structures = imrmlp.read("training_set.cfg")
    >>> for s in structures:
    ...     print(s.natoms, s.energy)
    """
    format = _input_format(filename, format, "read", detect_content)
    units = _input_units(format, input_units)
    io = _GPIO()
    raw_atoms = io.read(str(filename), format)
    if units is not None:
        raw_atoms = [_canonicalize_raw(a, **units) for a in raw_atoms]
    return [Atoms(_impl=a) for a in raw_atoms]


def read_one(filename: Union[str, Path], format: str = "", *,
             detect_content: bool = False, input_units=None) -> Atoms:
    """Read a single structure from a file.

    trajectory decodes only the first physical record through its filename backend;
    later frames are neither loaded nor validated. Native trajectory support is
    required. External gzip wrapping is unsupported for this backend.

    Args:
        filename: Path to the structure file.
        format: File format string. Auto-detected if empty.
        detect_content: Explicit bounded content detection when format is empty.
        input_units: Explicit EXTXYZ source preset or base-unit mapping.

    Returns:
        Atoms object.

    Raises:
        ValueError: If the format cannot be read by this entry point.
        RuntimeError: If the file cannot be read or contains no structure.

    Examples
    --------
    >>> atoms = imrmlp.read_one("POSCAR")
    """
    format = _input_format(filename, format, "read_one", detect_content)
    units = _input_units(format, input_units)
    io = _GPIO()
    raw = io.read_one(str(filename), format)
    if units is not None:
        raw = _canonicalize_raw(raw, **units)
    return Atoms(_impl=raw)


def iread(filename: Union[str, Path], format: str = "", *,
          detect_content: bool = False, input_units=None) -> Iterator[Atoms]:
    """Yield independent frames without loading the whole trajectory.

    Supported formats are ``extxyz``/``xyz``, ``cfg``/``mlpcfg``,
    ``bin``/``mlpbin``, ``n2p2``/``n2p2dat``, ``cp2kxyz``, ``gro``/``grodata``, ``xdatcar``,
    ``xdatcar6``, ``outcar``, ``xsf``/``axsf``.
    trajectory reads one frame at a time from the named file. Internal trajectory4
    compression is supported; external gzip wrapping is not decompressed.
    For other formats, optional gzip decompression follows the native
    reader. The file opens on the first iteration; malformed started frames and decoding failures raise
    when consumed. Complete zero-atom frames in EXTXYZ, CFG, BIN, N2P2,
    CP2KXYZ and XSF are yielded normally.

    CFG/BIN retain the file's integer chemmap types, as with ``read``; they do
    not infer elements or renumber species. Text CFG uses its native full-PBC
    convention; BIN retains its version's PBC metadata. Native framing rules
    apply, including CFG scanning for the next BEGIN_CFG marker.

    N2P2 accepts periodic and lattice-free molecular blocks. Its default element map gives atomic
    numbers; Bohr/Hartree inputs convert to Angstrom, eV and eV/Angstrom.
    Atomic charges and site energies are retained; a total charge is kept
    only when explicit. Comment text is ignored. For names such as
    ``input.data``, pass ``format="n2p2"`` explicitly.

    XSF accepts single structures or ANIMSTEPS animations with sequential indexed
    PRIMCOORD/ATOMS sections and a constant atom count. Periodic animations use
    one shared cell or indexed PRIMVEC cells for every frame; optional CONVVEC
    never replaces PRIMVEC. Coordinates are Angstrom and forces convert from
    Hartree/Angstrom to eV/Angstrom. Molecular ATOMS frames have FFF/zero cell.
    Full iteration checks frame counts and unread tails; early close does not.
    Grid sections remain unsupported. A molecular boundary requires reading the
    next ATOMS delimiter or EOF; periodic frames yield after their counted rows.

    CP2KXYZ reads a count, comment and element/xyz rows. Coordinates are
    Cartesian Angstrom values and the default element map gives atomic
    numbers. The comment supplies no cell, energy or time metadata; energy
    text in the comment is ignored, with no Hartree or Bohr conversion.

    GRO accepts standard fixed-width rows and the native whitespace subset,
    with a positive atom count and a box. Coordinates and box lengths convert
    from nm to Angstrom; the default element map gives atomic numbers. Complete
    standard fixed-width velocity columns in every atom row convert from nm/ps
    to Angstrom/ps. Zero velocities remain present; other suffixes and incomplete
    frame velocities keep geometry-only behavior. The title is ignored.
    Consecutive residue groups retain native molecule fields,
    without inferring topology. Unlike bulk ``read``, this yields every frame.

    XDATCAR shares the initial cell and species across frames; XDATCAR6 has
    a header per frame. Direct coordinates convert to Cartesian Angstrom,
    with default species mapped to atomic numbers. VASP headers require
    a positive total atom count. A fixed-cell header without configurations yields
    no frames; XDATCAR6 requires a complete initial frame.

    OUTCAR uses the native VASP 5/6 reader, including its existing MLFF
    layout. Each yielded ionic step has a cell, positions, forces and energy.
    Ordinary DFT steps must meet the native electronic convergence rule;
    unfinished or unconverged steps raise when consumed. A statistics tail
    after the last complete step is accepted. This does not add other MLFF
    layouts or require optional stress and magnetization output.

    ``detect_content=True`` requests bounded signature detection when format
    is empty; explicit format takes priority. ``input_units`` explicitly
    overrides EXTXYZ base units only; native defaults remain unchanged.

    The reader stores one frame plus bounded I/O buffers. Use ``close()`` when
    stopping early, or ``contextlib.closing`` for an explicit scope. Closing
    does not consume or validate an unread suffix. Full iteration validates
    the compressed trailer as well as the frames for gzip stream formats.

    Examples
    --------
    >>> from contextlib import closing
    >>> with closing(imrmlp.iread("trajectory.extxyz.gz")) as frames:
    ...     for atoms in frames:
    ...         print(atoms.natoms)
    """
    format = _input_format(filename, format, "read", detect_content)
    units = _input_units(format, input_units)
    if format in _SESSION_READ_FORMATS:
        reader = _GPIO().open_reader(str(filename), format, input_units=units)
    else:
        reader = _GPIOStreamReader(str(filename), format)
    try:
        while True:
            raw = reader.read_next()
            if raw is None:
                return
            yield Atoms(_impl=raw)
    finally:
        reader.close()


def write(filename: Union[str, Path],
          atoms: Union[Atoms, List[Atoms]],
          format: str = "", *, append: bool = False):
    """Write structure(s) to a file.

    A list written with format ``axsf`` or an ``.axsf`` filename produces one
    standard ANIMSTEPS animation. Frames must have the same atom count and PBC;
    fixed and changing periodic cells and molecular FFF frames are supported.
    Coordinates and optional forces are retained; energy/stress/other metadata
    are omitted, and molecular padding cells read back as zero. Every selected
    frame is validated before the target is opened. Ordinary ``xsf`` filenames
    retain separate single-geometry files; a single Atoms uses the single writer.
    Animation output requires a known list; ``write_iter`` is unsupported.

    Args:
        filename: Output file path.
        atoms: Single Atoms or list of Atoms to write.
        format: File format string. Auto-detected if empty.
        append: Extend an existing same-schema trajectory file (default False).

    Raises:
        ValueError: If the format only supports reading, or a single Atoms object
            is passed for a read-only format. The output is not opened.

    Examples
    --------
    >>> imrmlp.write("output.cfg", atoms)
    >>> imrmlp.write("trajectory.extxyz", [atoms1, atoms2, atoms3])
    """
    format = _checked_format(filename, format, "write")
    if append:
        raise ValueError("append is not supported; write a complete trajectory")
    io = _GPIO()
    if isinstance(atoms, Atoms):
        io.set_elements(atoms._elements or [])
        io.write_one(atoms._impl, str(filename), format)
    else:
        elements = atoms[0]._elements if atoms else None
        if any(a._elements != elements for a in atoms):
            raise ValueError("Structures written together must use the same element mapping")
        io.set_elements(elements or [])
        io.write([a._impl for a in atoms], str(filename), format)


def write_iter(filename: Union[str, Path],
               structures: Iterable[Atoms],
               format: str = "", *, append: bool = False):
    """Write EXTXYZ, CFG, BIN, XDATCAR, XDATCAR6, CP2KXYZ, N2P2 or trajectory from an iterable.

    Format detection, aliases and optional gzip compression follow
    ``write``. All frames must use the same element mapping. Native state is
    copied one frame at a time, so fractional inputs remain unchanged.
    XDATCAR fixes the cell and species counts from its first frame; later
    frames may reorder atoms. XDATCAR6 writes a complete header per frame and
    permits changing cells and atom counts. Both group positions by species
    and store geometry only.
    CP2KXYZ writes count, comment and element/Cartesian Angstrom coordinates
    in input atom order, allowing variable atom counts. Cell and E/F/stress
    labels are omitted. Native comments ``i = 0, time = 0`` are placeholders,
    not physical step/time metadata. Native default coordinate precision is
    retained; this is not a lossless format.

    N2P2 writes fully periodic or nonperiodic training blocks in Bohr/Hartree units. Complete forces,
    charges and site energies are retained; forces must be complete. Absent
    atomic charge/site-energy columns become zero.
    Total energy and explicit total_charge are optional. Stress, velocities,
    arbitrary metadata and topology are omitted; Hessian/HVP data are rejected.
    Molecular blocks omit lattice rows and read back with FFF/zero cell; any
    nonperiodic padding cell is omitted. Partial PBC is unsupported. The
    atomic-energy column is reserved/unused by standard n2p2 applications.

    trajectory uses the filename backend and requires native trajectory support. Its
    first frame fixes the atom count and uniformly present label and
    attribute schema. The same cell, units and label restrictions as the list
    writer apply. Existing targets are rejected by default; outer gzip wrapping is
    not applied. Each frame is validated before writing; a late error may leave
    the valid prefix. List writing still validates all frames before creation.

    An empty iterable does not create or truncate the target; a zero-atom Atoms
    is a complete frame except for trajectory, which requires nonempty frames.
    Errors close the output and may leave a written prefix.
    Successful return includes checked output and compression finalization.
    ``append=True`` is restricted to trajectory and requires an existing nonempty
    unlimited-frame file with the native writer's exact variable/type/dimension,
    units and fill layout. Atom count, PBC and present labels/attributes must
    match. Packed/external schemas and incomplete final records are rejected;
    existing compression is retained. Schema and list-input rejection preserves
    the target; bounded input errors can leave a complete appended prefix.
    I/O or close failures can leave a partial trailing record: append is not
    atomic, and concurrent file mutation is unsupported. No history is loaded;
    only the final existing frame is checked. Empty iterables remain a no-op.
    The writer retains one current frame (plus the
    first frame for trajectory) and bounded I/O buffers;
    objects retained by the caller still occupy memory.
    """
    format = _checked_format(filename, format, "write")
    if append:
        raise ValueError("append=True supports only trajectory")
    io = _GPIO()
    io.append = append

    def raw_frames():
        first = True
        elements = None
        for atoms in structures:
            current = None if atoms._elements is None else list(atoms._elements)
            if first:
                elements = current
                io.set_elements(elements or [])
                first = False
            elif current != elements:
                raise ValueError("Structures written together must use the same element mapping")
            yield atoms._impl

    io.write_iter(raw_frames(), str(filename), format)


def write_one(filename: Union[str, Path],
              atoms: Atoms,
              format: str = ""):
    """Write a single structure to a file.

    Args:
        filename: Output file path.
        atoms: Atoms object to write.
        format: File format string. Auto-detected if empty.
    """
    write(filename, atoms, format)


def guess_format(filename: Union[str, Path]) -> str:
    """Auto-detect file format from filename/extension.

    Args:
        filename: File path.

    Returns:
        Native format name (e.g., "poscar", "extxyz", "mlpcfg"). This does not
        imply support for both reading and writing; see the module's direction
        restrictions. An unrecognized filename returns an empty string.
    """
    return _GPIO.guess_format(str(filename))
