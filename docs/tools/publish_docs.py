"""Replace a validated directory without a missing/partially copied live tree.

The supported build platform is Linux. renameat2(RENAME_EXCHANGE) keeps the
old tree at the staging path until its TemporaryDirectory context cleans it up.
"""
import ctypes
import os
from pathlib import Path


def publish_directory(staging: Path, destination: Path):
    if not destination.exists():
        staging.rename(destination)
        return
    rename = getattr(ctypes.CDLL(None, use_errno=True), 'renameat2', None)
    if rename is None:
        raise RuntimeError('Atomic documentation publication requires Linux renameat2')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 2):
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))
