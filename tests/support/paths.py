"""Locations of the implementations under test, of the outputs and of the derivation modules.

Implementations (each is imported in its own process):
  mojo        am3_mojo, built by `make build` into build/ (or the directory
              AM3_MOJO_BUILD names, e.g. build-asserts/ for `make test-asserts`)
  mojo-am3    the same module with the processes am3_mojo adds switched off
              (EXTENSION_SWITCHES): AM3's physics, for every comparison with
              the C++ code
  cpp         the original C++ AM3 (reference), built by `make reference`
  cpp-fixed   the original with am3_mojo's fixes of AM3 bugs
              (tests/reference/am3_fixes.patch), also built by `make reference`:
              the like-for-like reference of the comparisons
"""
import importlib, os, sys, sysconfig

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.join(ROOT, 'tests')
SUPPORT = os.path.join(TESTS, 'support')
DERIVATIONS = os.path.join(TESTS, 'literature_checks', 'derivations')
RESULTS = os.path.join(ROOT, 'results')
REFERENCE_WORK = os.environ.get('AM3_REFERENCE_DIR', os.path.join(TESTS, 'reference', 'work'))

MOJO_BUILD = os.path.join(ROOT, os.environ.get('AM3_MOJO_BUILD', 'build'))
# raw outputs of a non-default build go to their own cache directories
# (results/runs-build-asserts/, ...), apart from the ones the figures read
CACHE_SUFFIX = '' if 'AM3_MOJO_BUILD' not in os.environ else '-' + os.path.basename(os.path.normpath(MOJO_BUILD))
# Mojo's assertion level for the harnesses the tests compile (`all` under
# `make test-asserts`)
MOJO_ASSERT = os.environ.get('AM3_MOJO_ASSERT', 'none')

IMPLEMENTATIONS = {
    'mojo': (MOJO_BUILD, 'am3_mojo'),
    'mojo-am3': (MOJO_BUILD, 'am3_mojo'),
    'cpp': (os.path.join(REFERENCE_WORK, 'build', 'serial', 'libpython'), 'am3'),
    'cpp-fixed': (os.path.join(REFERENCE_WORK, 'build', 'fixed', 'libpython'), 'am3'),
}
LABELS = {'mojo': 'am3_mojo', 'mojo-am3': 'am3_mojo (AM3 processes only)', 'cpp': 'C++ AM3',
          'cpp-fixed': 'C++ AM3 (with the fixes of am3_fixes.patch)'}

# Master switches of the processes am3_mojo adds to AM3 (all on by default; off in `mojo-am3`)
EXTENSION_SWITCHES = ['set_process_pair_annihilation', 'set_process_triplet_pair_production',
                      'set_process_neutron_decay', 'set_process_kaons']


def _module_files(impl):
    """The implementation's module files that this Python can import: untagged (`am3_mojo.so`, as Mojo builds it)
    or tagged for this Python (e.g. `am3.cpython-314-darwin.so`, as pybind11 builds the C++ reference). A module of
    another Python version does not count."""
    path, name = IMPLEMENTATIONS[impl]
    if not os.path.isdir(path):
        return []
    names = {name + '.so', name + sysconfig.get_config_var('EXT_SUFFIX')}
    return [os.path.join(path, f) for f in os.listdir(path) if f in names]


def available(impl):
    """True if the implementation's Python module has been built for this Python."""
    return bool(_module_files(impl))


def module_mtime(impl):
    """Modification time of the implementation's built Python module (0 if not built)."""
    return max((os.path.getmtime(f) for f in _module_files(impl)), default=0.0)


def load(impl):
    """Import and return the module of `impl` (one implementation per process)."""
    path, name = IMPLEMENTATIONS[impl]
    if not available(impl):
        raise RuntimeError(f'{LABELS[impl]} is not built for this Python ({path}); see README.md')
    sys.path.insert(0, path)
    return importlib.import_module(name)


def new_am3(impl):
    """A fresh AM3 object of `impl`, configured as the implementation's name
    says (for `mojo-am3`: the added processes switched off)."""
    obj = load(impl).AM3()
    if impl == 'mojo-am3':
        for setter in EXTENSION_SWITCHES:
            getattr(obj, setter)(False)
    return obj


def load_derivation(name):
    """The module `name` of tests/literature_checks/derivations/ (the README's equations and values as functions),
    its directory put on sys.path after the importing script's own."""
    if DERIVATIONS not in sys.path:
        sys.path.insert(1, DERIVATIONS)
    return importlib.import_module(name)


def use_tool(name=''):
    """Puts tools/<name>/ (or tools/ itself) on sys.path, after the importing script's own directory."""
    path = os.path.join(ROOT, 'tools', name)
    if path not in sys.path:
        sys.path.insert(1, path)


def results_dir(*parts):
    d = os.path.join(RESULTS, *parts)
    os.makedirs(d, exist_ok=True)
    return d
