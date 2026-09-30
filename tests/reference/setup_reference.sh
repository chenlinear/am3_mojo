#!/usr/bin/env bash
# Build the original C++ AM3 v1.0.1 (commit 04d63d9, gitlab.desy.de/am3/am3) as
# the reference for tests/reference and the benchmark:
#   work/build/serial/libpython   the unmodified original (serial build)
#   work/build/oobfix/libpython   the same with its two out-of-bounds reads
#                                  fixed (oob_fix.patch)
# Dependencies (GSL 2.8, Eigen 3.4.0 and, on macOS, LLVM libomp 19.1.7) are
# downloaded and built into work/deps; cmake, ninja and pybind11 come from the
# project's Python environment (`uv sync`). Every step is skipped if its
# result already exists, so the script can be re-run.
#
# Environment variables:
#   AM3_REFERENCE_DIR   work directory (default: tests/reference/work)
#   AM3_SOURCE_DIR      use an existing AM3 source tree instead of cloning
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="${AM3_REFERENCE_DIR:-$HERE/work}"
COMMIT=04d63d9cfe2f0dc9725dfe9c5dfc46cbf50eab68
PY="$ROOT/.venv/bin/python"
export PATH="$ROOT/.venv/bin:$PATH"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
DEPS="$WORK/deps"
PREFIX="$DEPS/install"
mkdir -p "$WORK" "$DEPS"

say() { printf '\n==> %s\n' "$*"; }
fetch() { [ -f "$2" ] || curl -fL --retry 3 -o "$2" "$1"; }
[ -x "$PY" ] || { echo "Python environment not found: run 'uv sync' first"; exit 1; }
command -v cmake >/dev/null && command -v ninja >/dev/null || { echo "cmake/ninja missing: run 'uv sync'"; exit 1; }

# --- 1. AM3 source ------------------------------------------------------------
SRC="${AM3_SOURCE_DIR:-$WORK/am3}"
if [ ! -f "$SRC/CMakeLists.txt" ]; then
    say "cloning AM3 ($COMMIT)"
    git clone -q https://gitlab.desy.de/am3/am3.git "$SRC"
    git -C "$SRC" checkout -q "$COMMIT"
fi

# --- 2. dependencies ----------------------------------------------------------
if [ ! -f "$PREFIX/lib/libgsl.a" ]; then
    say "building GSL 2.8"
    fetch https://ftp.gnu.org/gnu/gsl/gsl-2.8.tar.gz "$DEPS/gsl-2.8.tar.gz"
    tar -xzf "$DEPS/gsl-2.8.tar.gz" -C "$DEPS"
    (cd "$DEPS/gsl-2.8" && CFLAGS="-O2 -fPIC" ./configure --prefix="$PREFIX" --disable-shared --enable-static >/dev/null \
        && make -j"$JOBS" >/dev/null && make install >/dev/null)
fi
if [ ! -d "$PREFIX/share/eigen3/cmake" ]; then
    say "installing Eigen 3.4.0"
    fetch https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz "$DEPS/eigen-3.4.0.tar.gz"
    tar -xzf "$DEPS/eigen-3.4.0.tar.gz" -C "$DEPS"
    cmake -S "$DEPS/eigen-3.4.0" -B "$DEPS/eigen-build" -G Ninja -DCMAKE_INSTALL_PREFIX="$PREFIX" >/dev/null
    cmake --install "$DEPS/eigen-build" >/dev/null
fi
# AM3's CMakeLists requires OpenMP; the reference is built serially (the
# OpenMP flags only point at the headers, so the pragmas are ignored).
if [ "$(uname)" = Darwin ]; then
    if [ ! -f "$PREFIX/lib/libomp.dylib" ]; then
        say "building LLVM libomp 19.1.7"
        R=https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7
        fetch "$R/openmp-19.1.7.src.tar.xz" "$DEPS/openmp-19.1.7.src.tar.xz"
        fetch "$R/cmake-19.1.7.src.tar.xz" "$DEPS/cmake-19.1.7.src.tar.xz"
        mkdir -p "$DEPS/llvm"
        tar -xJf "$DEPS/openmp-19.1.7.src.tar.xz" -C "$DEPS/llvm" && rm -rf "$DEPS/llvm/openmp" && mv "$DEPS/llvm/openmp-19.1.7.src" "$DEPS/llvm/openmp"
        tar -xJf "$DEPS/cmake-19.1.7.src.tar.xz" -C "$DEPS/llvm" && rm -rf "$DEPS/llvm/cmake" && mv "$DEPS/llvm/cmake-19.1.7.src" "$DEPS/llvm/cmake"
        cmake -S "$DEPS/llvm/openmp" -B "$DEPS/omp-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$PREFIX" -DLIBOMP_INSTALL_ALIASES=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF >/dev/null
        cmake --build "$DEPS/omp-build" >/dev/null && cmake --install "$DEPS/omp-build" >/dev/null
    fi
    OMP_ARGS=(-DOpenMP_CXX_FLAGS="-I$PREFIX/include" -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY="$PREFIX/lib/libomp.dylib")
else
    GOMP="$(c++ -print-file-name=libgomp.so)"
    OMP_ARGS=(-DOpenMP_CXX_FLAGS="-DAM3_SERIAL_REFERENCE" -DOpenMP_CXX_LIB_NAMES=gomp -DOpenMP_gomp_LIBRARY="$GOMP")
fi

# --- 3. AM3 builds ------------------------------------------------------------
PYBIND="$("$PY" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
build() {  # build <name> <source dir>
    local out="$WORK/build/$1"
    if ls "$out/libpython/"am3*.so >/dev/null 2>&1; then return; fi
    say "building AM3 ($1)"
    cmake -S "$2" -B "$out" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="$PREFIX/lib" -DCMAKE_PREFIX_PATH="$PREFIX" \
        -DEigen3_DIR="$PREFIX/share/eigen3/cmake" -DGSL_ROOT_DIR="$PREFIX" "${OMP_ARGS[@]}" \
        -DPython_EXECUTABLE="$PY" -Dpybind11_DIR="$PYBIND" >/dev/null
    cmake --build "$out" >/dev/null
    # the module loads libomp through @rpath; keep a copy next to it
    if [ "$(uname)" = Darwin ]; then cp "$PREFIX/lib/libomp.dylib" "$out/libpython/"; fi
}
build serial "$SRC"
if [ ! -d "$WORK/src-oobfix" ]; then
    cp -R "$SRC" "$WORK/src-oobfix" && rm -rf "$WORK/src-oobfix/.git"
    patch -s -d "$WORK/src-oobfix" -p1 < "$HERE/oob_fix.patch"
fi
build oobfix "$WORK/src-oobfix"

"$PY" - "$WORK" <<'PYEOF'
import os, subprocess, sys
ok = True
for b in ('serial', 'oobfix'):
    d = os.path.join(sys.argv[1], 'build', b, 'libpython')
    r = subprocess.run([sys.executable, '-c', f'import sys; sys.path.insert(0, {d!r}); import am3; am3.AM3()'],
                       capture_output=True, text=True)
    ok &= r.returncode == 0
    print(f'reference {b:7s}', 'OK' if r.returncode == 0 else 'FAILED\n' + r.stderr[-2000:])
sys.exit(0 if ok else 1)
PYEOF
