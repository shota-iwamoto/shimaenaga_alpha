"""Setup script for Shimaenaga Python package."""
import os
import sys
import shutil
import sysconfig
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


def _has_pybind11():
    try:
        import pybind11  # noqa: F401
        return True
    except ImportError:
        return False


class CMakeBuild(build_ext):
    """Build C++ extension via CMake (pybind11 path)."""

    def build_extension(self, ext):
        src_dir = Path(__file__).parent.absolute()
        build_dir = src_dir / "build"
        # Drop stale CMake state. cibuildwheel builds every CPython version
        # against the *same* copied source tree on Linux, so a CMakeCache.txt
        # left behind by the previous interpreter would pin the wrong Python
        # (and a now-deleted pybind11 overlay), making the reconfigure a
        # silent no-op that never creates the _shimaenaga target ("No rule to
        # make target _shimaenaga").
        #
        # Only CMakeCache.txt/CMakeFiles though -- NOT the whole build/ dir:
        # setuptools' build_py has already copied the pure-Python package into
        # build/lib.*/ by the time build_ext runs, and wiping the directory
        # here would ship a wheel with the compiled module but no shimaenaga
        # package inside.
        cache = build_dir / "CMakeCache.txt"
        if cache.exists():
            cache.unlink()
        cmake_files = build_dir / "CMakeFiles"
        if cmake_files.exists():
            shutil.rmtree(cmake_files)
        build_dir.mkdir(exist_ok=True)

        cmake_args = [
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DSHIMAENAGA_BUILD_TESTS=OFF",
            "-DSHIMAENAGA_BUILD_PYTHON=ON",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
        ]

        subprocess.check_call(
            ["cmake", str(src_dir)] + cmake_args, cwd=str(build_dir)
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "_shimaenaga",
             "--config", "Release", "-j4"],
            cwd=str(build_dir),
        )

        # CMakeLists.txt always drops the compiled module under python/ (so
        # a plain `cmake --build` still leaves `import shimaenaga` working
        # straight from the repo). Copy it to where setuptools expects the
        # built extension so bdist_wheel actually bundles it. Match the exact
        # suffix for the running interpreter -- a glob would also pick up
        # stale builds for other Python versions left over in python/.
        module_name = f"_shimaenaga{sysconfig.get_config_var('EXT_SUFFIX')}"
        # python/Release|Debug fallbacks: multi-config generators (Visual
        # Studio) may append the config name to the output directory.
        candidates = [
            src_dir / "python" / module_name,
            src_dir / "python" / "Release" / module_name,
            src_dir / "python" / "Debug" / module_name,
        ]
        built = next((p for p in candidates if p.exists()), None)
        if built is None:
            raise RuntimeError(
                "Expected pybind11 module not found; looked in: "
                + ", ".join(str(p) for p in candidates))
        dest = Path(self.get_ext_fullpath(ext.name))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, dest)


class CMakeBuildCore(build_py):
    """Build only the shared C core library (ctypes path, no pybind11)."""

    def run(self):
        super().run()
        src_dir = Path(__file__).parent.absolute()
        build_dir = src_dir / "build"
        build_dir.mkdir(exist_ok=True)

        cmake_args = [
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DSHIMAENAGA_BUILD_TESTS=OFF",
            "-DSHIMAENAGA_BUILD_PYTHON=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        try:
            subprocess.check_call(
                ["cmake", str(src_dir)] + cmake_args, cwd=str(build_dir)
            )
            subprocess.check_call(
                ["cmake", "--build", ".", "--target", "shimaenaga_core",
                 "--config", "Release", "-j4"],
                cwd=str(build_dir),
            )
        except subprocess.CalledProcessError as e:
            print(f"WARNING: C++ build failed ({e}). "
                  "Ensure cmake and a C++17 compiler are available.")
            return

        # Bundle the shared core next to _ctypes_backend.py so _find_lib()
        # can locate it inside an installed wheel (no system install needed).
        ext = ".dylib" if sys.platform == "darwin" else (
            ".dll" if sys.platform == "win32" else ".so"
        )
        built = next(build_dir.rglob(f"*shimaenaga_core{ext}"), None)
        if built is not None:
            dest_dir = Path(self.build_lib) / "shimaenaga"
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built, dest_dir / built.name)


if _has_pybind11():
    ext_modules = [Extension("_shimaenaga", sources=[])]
    cmdclass = {"build_ext": CMakeBuild}
else:
    ext_modules = []
    cmdclass = {"build_py": CMakeBuildCore}


setup(
    name="shimaenaga",
    version="1.3.0",
    description="Shimaenaga: Attentive Histogram GBDT with sample-level token attention",
    long_description=Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else "",
    # Without this PyPI/twine parse the Markdown README as reStructuredText
    # and reject the upload on "syntax errors in markup".
    long_description_content_type="text/markdown",
    author="Shimaenaga Authors",
    license="MIT",
    license_files=["LICENSE", "THIRD_PARTY_NOTICES.md"],
    packages=find_packages("python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
    extras_require={
        "sklearn": ["scikit-learn>=1.0"],
        "dev": ["pytest", "scikit-learn", "pandas"],
    },
    zip_safe=False,
)
