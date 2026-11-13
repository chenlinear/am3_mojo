"""Implementations (each imported in its own process):
mojo        am3_mojo, built by `make build` into build/ (or AM3_MOJO_BUILD,
            for example build-asserts/ for `make test-asserts`)
mojo-am3    the same module with the processes am3_mojo adds switched off
            (EXTENSION_SWITCHES): AM3's physics, for every comparison with the
            C++ code
cpp         the original C++ AM3, built by `make reference`
cpp-fixed   the original with am3_mojo's fixes of AM3 bugs and its PDG 2026
            constants (tests/reference/am3_fixes.patch), also built by
            `make reference`: the like-for-like reference
"""

import filecmp, importlib, os, sys, sysconfig

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TESTS = os.path.join(ROOT, "tests")
SUPPORT = os.path.join(TESTS, "support")
# the folders of the derivation modules, as verify.FOLDERS of
# tests/literature_checks/common/verify.py
DERIVATIONS = [
    os.path.join(TESTS, "literature_checks", f)
    for f in (
        "common",
        "annihilation",
        "triplet",
        "neutron_decay",
        "kaons",
        "charm",
    )
]
REFERENCE_WORK = os.environ.get(
    "AM3_REFERENCE_DIR", os.path.join(TESTS, "reference", "work")
)

MOJO_BUILD = os.path.join(ROOT, os.environ.get("AM3_MOJO_BUILD", "build"))
# A variant module (AM3_MOJO_BUILD) needs its own sources in AM3_MOJO_SRC for
# the harness-based tests to test it.
MOJO_SRC = os.path.join(ROOT, os.environ.get("AM3_MOJO_SRC", "src"))
# A build in another folder than build/ caches its raw outputs apart from those
# that the figures read (results/runs-build-asserts/, ...). AM3_MOJO_BUILD=build
# names the default build.
CACHE_SUFFIX = (
    ""
    if os.path.realpath(MOJO_BUILD)
    == os.path.realpath(os.path.join(ROOT, "build"))
    else "-" + os.path.basename(os.path.normpath(MOJO_BUILD))
)
MOJO_ASSERT = os.environ.get("AM3_MOJO_ASSERT", "none")

IMPLEMENTATIONS = {
    "mojo": (MOJO_BUILD, "am3_mojo"),
    "mojo-am3": (MOJO_BUILD, "am3_mojo"),
    "cpp": (
        os.path.join(REFERENCE_WORK, "build", "serial", "libpython"),
        "am3",
    ),
    "cpp-fixed": (
        os.path.join(REFERENCE_WORK, "build", "fixed", "libpython"),
        "am3",
    ),
}
LABELS = {
    "mojo": "am3_mojo",
    "mojo-am3": "am3_mojo (AM3 processes only)",
    "cpp": "C++ AM3",
    "cpp-fixed": "C++ AM3 (with the fixes of am3_fixes.patch)",
}

EXTENSION_SWITCHES = [
    "set_process_pair_annihilation",
    "set_process_triplet_pair_production",
    "set_process_neutron_decay",
    "set_process_kaons",
    "set_process_charm",
]


def _module_files(impl):
    """The module files are untagged (`am3_mojo.so`, as Mojo builds it) or
    tagged for this Python (for example `am3.cpython-314-darwin.so`, as pybind11
    builds the C++ reference)."""
    path, name = IMPLEMENTATIONS[impl]
    if not os.path.isdir(path):
        return []
    names = {name + ".so", name + sysconfig.get_config_var("EXT_SUFFIX")}
    return [os.path.join(path, f) for f in os.listdir(path) if f in names]


def available(impl):
    return bool(_module_files(impl))


def built_from_another_patch():
    """tests/reference/setup_reference.sh keeps the patch that it applied as
    work/src-fixed/.patch. A build without that copy is taken as it is."""
    applied = os.path.join(REFERENCE_WORK, "src-fixed", ".patch")
    patch = os.path.join(TESTS, "reference", "am3_fixes.patch")
    return os.path.isfile(applied) and not filecmp.cmp(
        patch, applied, shallow=False
    )


def module_mtime(impl):
    return max((os.path.getmtime(f) for f in _module_files(impl)), default=0.0)


def module_fingerprint(impl):
    """Path, size and modification time tell two builds apart, also when
    directories have one name or a module is copied back with its old time."""
    return "\n".join(
        f"{os.path.realpath(f)} {os.path.getsize(f)} {os.stat(f).st_mtime_ns}"
        for f in sorted(_module_files(impl))
    )


def load(impl):
    """One implementation per process."""
    path, name = IMPLEMENTATIONS[impl]
    if not available(impl):
        raise RuntimeError(
            f"{LABELS[impl]} is not built for this Python "
            f"({path}); see README.md"
        )
    sys.path.insert(0, path)
    return importlib.import_module(name)


def new_am3(impl):
    obj = load(impl).AM3()
    if impl == "mojo-am3":
        for setter in EXTENSION_SWITCHES:
            getattr(obj, setter)(False)
    return obj


def load_derivation(name):
    """The folders DERIVATIONS are put on sys.path after the importing script's
    own directory."""
    for path in reversed(DERIVATIONS):
        if path not in sys.path:
            sys.path.insert(1, path)
    return importlib.import_module(name)


def use_tool(name):
    """Puts tools/<name>/ on sys.path, after the importing script's own
    directory."""
    path = os.path.join(ROOT, "tools", name)
    if path not in sys.path:
        sys.path.insert(1, path)


def published_dir(*parts):
    """The default build (build/) writes the published tables and figures. A
    variant (AM3_MOJO_BUILD) stops instead of replacing them."""
    if CACHE_SUFFIX:
        raise SystemExit(
            f"AM3_MOJO_BUILD={os.environ['AM3_MOJO_BUILD']}: results/"
            f"{'/'.join(parts)} holds the default build's results"
        )
    return results_dir(*parts)


def results_dir(*parts):
    d = os.path.join(ROOT, "results", *parts)
    os.makedirs(d, exist_ok=True)
    return d
