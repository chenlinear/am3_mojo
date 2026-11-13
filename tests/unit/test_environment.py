"""Importing am3_mojo must not change the process environment.

Mojo's runtime sets PYTHONEXECUTABLE, PYTHONPATH and MOJO_PYTHON_LIBRARY at the
C level while its Python interop initializes. PyInit_am3_mojo restores them, so
Python subprocesses and multiprocessing workers started after the import still
run in the same virtual environment.
"""

import json, os, subprocess, sys, textwrap

from support.paths import ROOT, IMPLEMENTATIONS

MOJO = IMPLEMENTATIONS["mojo"][0]
KEYS = ["PYTHONEXECUTABLE", "PYTHONPATH", "MOJO_PYTHON_LIBRARY"]

# run as a script in a fresh interpreter, so that the import happens inside the
# test
PROBE = textwrap.dedent("""
    import ctypes, json, multiprocessing as mp, subprocess, sys
    libc = ctypes.CDLL(None); libc.getenv.restype = ctypes.c_char_p
    def env():
        return {k: (libc.getenv(k.encode()) or b'<unset>').decode() for k in %r}
    def work(x):
        import numpy
        return float(numpy.sqrt(x))
    if __name__ == '__main__':
        before = env()
        sys.path.insert(0, %r)
        import am3_mojo
        a = am3_mojo.AM3()
        a.set_process_escape(True)   # bool conversion
        # numpy conversion
        a.set_injection_rate_photons(a.get_egrid_photons() * 0.0)
        after = env()
        child = subprocess.run(
            [sys.executable, '-c', 'import numpy, sys; print(sys.prefix)'],
            capture_output=True, text=True)
        with mp.get_context('spawn').Pool(2) as pool:
            pooled = pool.map(work, [4.0, 9.0])
        print(json.dumps(dict(before=before, after=after,
                              child_rc=child.returncode,
                              child_prefix=child.stdout.strip(),
                              prefix=sys.prefix, pooled=pooled)))
""") % (KEYS, MOJO)


def probe(tmp_path, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in KEYS} | (
        extra_env or {}
    )
    # a file: spawned workers re-import __main__ from it
    script = tmp_path / "probe.py"
    script.write_text(PROBE)
    r = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_import_leaves_environment_and_venv_unchanged(tmp_path):
    """With none of KEYS set, they stay unset after the import. A Python
    subprocess and a spawned pool started after the import run in the same
    virtual environment."""
    d = probe(tmp_path)
    assert d["after"] == d["before"]
    assert all(v == "<unset>" for v in d["after"].values())
    assert d["child_rc"] == 0
    assert d["child_prefix"] == d["prefix"]
    assert d["pooled"] == [2.0, 3.0]


def test_user_pythonpath_is_preserved(tmp_path):
    d = probe(tmp_path, {"PYTHONPATH": "/tmp/am3-user-path"})
    assert d["after"]["PYTHONPATH"] == "/tmp/am3-user-path"
    assert d["after"] == d["before"]
