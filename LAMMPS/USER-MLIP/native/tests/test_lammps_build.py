#!/usr/bin/env python3
"""Configure-only backend/ABI checks; actual execution uses run_regression.py.

Synthetic flag files exercise CUDA/HIP selection without claiming device
compilation or execution. Pass a real upstream tree and existing IMR-MLP library.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class BuildContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="native-lmp-config-")
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)

    def configure(self, backend="CPU", flags="-DIMRMLP_MPI", extra=(), source=None, error=None):
        stamp = self.work / "flags"
        stamp.write_text(flags)
        build = self.work / "build"
        result = subprocess.run([
            "cmake", "-S", str(ENTRY), "-B", str(build),
            f"-DLAMMPS_SOURCE_DIR={source or UPSTREAM}",
            f"-DIMRMLP_NATIVE_BACKEND={backend}", f"-DIMRMLP_LIBRARY={LIBRARY}",
            f"-DIMRMLP_FLAGS_FILE={stamp}", "-DBUILD_TESTING=OFF",
            "-DPKG_PLUGIN=OFF", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", *extra,
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        if error:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(error, result.stdout)
        else:
            self.assertEqual(result.returncode, 0, result.stdout)
        return build

    def test_backend_matrix_registers_all_styles_without_plugin_or_gpu_compiler(self):
        for backend in ("CPU", "CUDA", "HIP"):
            with self.subTest(backend=backend):
                self.work = Path(self.directory.name) / backend
                self.work.mkdir()
                flags = "-DIMRMLP_MPI"
                if backend != "CPU":
                    flags += f" -DIMRMLP_GPU -DIMRMLP_{backend}"
                build = self.configure(backend, flags)
                pair = (build / "upstream/styles/style_pair.h").read_text()
                compute = (build / "upstream/styles/style_compute.h").read_text()
                self.assertIn("pair_mlip_native.h", pair)
                self.assertIn("compute_mlip_native.h", compute)
                commands = json.loads((build / "compile_commands.json").read_text())
                native = [x for x in commands if Path(x["file"]).name in
                          ("pair_mlip_native.cpp", "compute_mlip_native.cpp")]
                self.assertEqual(len(native), 2)
                self.assertTrue(all("-DIMRMLP_MPI" in x["command"] for x in native))
                self.assertFalse(any("native_plugin.cpp" in x["file"] for x in commands))
                self.assertFalse(any("-DIMRMLP_MPI" in x["command"] for x in commands if x not in native))
                cache = (build / "CMakeCache.txt").read_text()
                self.assertNotIn("CMAKE_CUDA_COMPILER:", cache)
                self.assertNotIn("CMAKE_HIP_COMPILER:", cache)
                self.assertIn("PKG_PLUGIN:BOOL=OFF", cache)

    def test_reject_serial_lammps(self):
        self.configure(extra=["-DBUILD_MPI=OFF"], error="requires BUILD_MPI=ON")

    def test_reject_serial_gpml(self):
        self.configure(flags="-DIMRMLP_CUDA -DIMRMLP_GPU", error="not the serial MPI stubs")

    def test_reject_backend_mismatch(self):
        self.configure("HIP", error="conflicts with CPU IMRMLP_FLAGS_FILE")

    def test_reject_invalid_backend(self):
        self.configure("invalid", error="IMRMLP_NATIVE_BACKEND must be CPU, CUDA, or HIP")

    def test_reject_inconsistent_gpu_flags(self):
        self.configure(flags="-DIMRMLP_MPI -DIMRMLP_GPU", error="Inconsistent GPU backend")

    def test_reject_missing_gpml_library(self):
        self.configure(extra=[f"-DIMRMLP_LIBRARY={self.work / 'missing.so'}"], error="Build the matching MPI IMR-MLP")

    def test_reject_missing_upstream(self):
        self.configure(source=self.work / "missing", error="LAMMPS_SOURCE_DIR must identify")

    def test_reject_duplicate_native_style(self):
        # A small upstream fixture reaches the registration guard without
        # modifying or installing files into the user's real LAMMPS tree.
        upstream = self.work / "upstream"
        (upstream / "src").mkdir(parents=True)
        (upstream / "cmake").mkdir()
        (upstream / "src/lammps.h").touch()
        duplicate = upstream / "src/duplicate.h"
        duplicate.write_text("PairStyle(mlip/native,OldPair)\n")
        (upstream / "cmake/CMakeLists.txt").write_text('''
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/empty.cpp" "int main(){return 0;}")
add_library(lammps STATIC "${CMAKE_CURRENT_BINARY_DIR}/empty.cpp")
add_executable(lmp "${CMAKE_CURRENT_BINARY_DIR}/empty.cpp")
function(AddStyleHeader)
endfunction()
function(GenerateStyleHeader)
endfunction()
set_property(GLOBAL PROPERTY PAIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/duplicate.h")
''')
        self.configure(source=upstream, error="Duplicate native style")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps-source", type=Path, required=True)
    parser.add_argument("--imrmlp-library", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    UPSTREAM, LIBRARY = options.lammps_source.resolve(), options.imrmlp_library.resolve()
    ENTRY = Path(__file__).resolve().parents[1] / "lammps"
    unittest.main(argv=[__file__, *remaining])
