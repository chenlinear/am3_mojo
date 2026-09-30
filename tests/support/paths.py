"""Locations of the implementations under test and of the outputs.

Implementations (each is imported in its own process):
  mojo        am3_mojo, built by `make build` into build/
  cpp         the original C++ AM3 (reference), built by `make reference`
  cpp-oobfix  the original with its two out-of-bounds reads fixed
              (tests/reference/oob_fix.patch), also built by `make reference`
"""
import importlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.join(ROOT, 'tests')
SUPPORT = os.path.join(TESTS, 'support')
RESULTS = os.path.join(ROOT, 'results')
REFERENCE_WORK = os.environ.get('AM3_REFERENCE_DIR', os.path.join(TESTS, 'reference', 'work'))

IMPLEMENTATIONS = {
    'mojo': (os.path.join(ROOT, 'build'), 'am3_mojo'),
    'cpp': (os.path.join(REFERENCE_WORK, 'build', 'serial', 'libpython'), 'am3'),
    'cpp-oobfix': (os.path.join(REFERENCE_WORK, 'build', 'oobfix', 'libpython'), 'am3'),
}
LABELS = {'mojo': 'am3_mojo', 'cpp': 'C++ AM3', 'cpp-oobfix': 'C++ AM3 (out-of-bounds reads fixed)'}


def available(impl):
    """True if the implementation's Python module has been built."""
    path, name = IMPLEMENTATIONS[impl]
    return os.path.isdir(path) and any(f.startswith(name + '.') and f.endswith('.so') for f in os.listdir(path))


def load(impl):
    """Import and return the module of `impl` (one implementation per process)."""
    path, name = IMPLEMENTATIONS[impl]
    if not available(impl):
        raise RuntimeError(f'{LABELS[impl]} is not built ({path}); see README.md')
    sys.path.insert(0, path)
    return importlib.import_module(name)


def results_dir(*parts):
    d = os.path.join(RESULTS, *parts)
    os.makedirs(d, exist_ok=True)
    return d
