"""The Python API of am3_mojo.

* every method of the AM3 v1.0.1 Python interface exists, with its signature
  and docstring (tools/am3_api.json, introspected from the original module);
* on/off switches are Python booleans: the getters return True/False, the
  setters take True/False (0/1 and numpy booleans are accepted too, anything
  else is an error), docstrings and the printed switch table say True/False.
"""
import json, os, re, subprocess, sys
import numpy as np
import pytest

from support.paths import ROOT, IMPLEMENTATIONS, available, load
from support.scenarios import dump_state

am3_mojo = load('mojo') if available('mojo') else None
API = json.load(open(os.path.join(ROOT, 'tools', 'am3_api.json')))
SIG = {n: d.split('\n')[0] for n, d in API}
BOOL_GETTERS = sorted([n for n, s in SIG.items() if n.startswith('get_') and s.endswith('-> bool')]
                      + ['get_verbosity_level'])
BOOL_SETTERS = sorted(n for n, s in SIG.items() if n.startswith('set_') and 'arg0: bool' in s)
PAIRS = [s for s in BOOL_SETTERS if 'get_' + s[4:] in BOOL_GETTERS]
# Quirk of the original kept on purpose (see README): these setters store
# int(value) + 1 into a C++ bool, so the switch is stuck at True and
# set(False) has no effect. The original module behaves identically.
STUCK_TRUE = {'set_optimize_compton_emission_grid', 'set_optimize_compton_target_grid',
              'set_optimize_bethe_heitler_outgoing_pairs_grid',
              'set_optimize_photopion_target_photon_grid'}


def expect(setter, value):
    return True if setter in STUCK_TRUE else value


def test_module_attributes():
    assert am3_mojo.__version__ == '0.1.0'
    assert am3_mojo.AM3_API_VERSION == '1.0.1'
    assert am3_mojo.AM3.__module__ == 'am3_mojo'


@pytest.mark.parametrize('name', sorted(SIG))
def test_method_signature_and_docstring(name):
    """Every method of the AM3 interface exists, with the original signature
    (class name am3_mojo.AM3; switch methods use bool; see gen_bindings.py)."""
    doc = getattr(am3_mojo.AM3, name).__doc__
    first = doc.split('\n')[0]
    expected = SIG[name].replace('am3.AM3', 'am3_mojo.AM3')
    if name == 'get_verbosity_level':
        expected = expected.replace('-> float', '-> bool')
    if 'Overloaded function' in SIG[name] or '(*args' in SIG[name]:
        return
    assert first == expected


def test_switch_lists_are_complete():
    assert len(BOOL_GETTERS) == 59 and len(BOOL_SETTERS) == 59
    assert len(PAIRS) == 59  # every switch setter has a matching getter


@pytest.mark.parametrize('name', BOOL_GETTERS)
def test_getter_returns_bool(name):
    v = getattr(am3_mojo.AM3(), name)()
    assert type(v) is bool, f'{name}() returned {v!r} ({type(v).__name__})'


@pytest.mark.parametrize('setter', PAIRS)
def test_setter_roundtrip_false_true(setter):
    a = am3_mojo.AM3()
    get = getattr(a, 'get_' + setter[4:])
    getattr(a, setter)(False)
    assert get() is expect(setter, False)
    getattr(a, setter)(True)
    assert get() is True
    getattr(a, setter)(False)
    assert get() is expect(setter, False)


@pytest.mark.parametrize('value, expected', [(0, False), (1, True), (np.bool_(True), True),
                                             (np.bool_(False), False), (np.int64(1), True)])
def test_setters_accept_integers_0_1_and_numpy(value, expected):
    a = am3_mojo.AM3()
    for setter in PAIRS:
        getattr(a, setter)(value)
        assert getattr(a, 'get_' + setter[4:])() is expect(setter, expected), setter


@pytest.mark.parametrize('bad', ['x', 'True', 0.5, 1.0, 2, -1, None, [1]])
def test_setters_reject_non_booleans(bad):
    a = am3_mojo.AM3()
    for setter in BOOL_SETTERS:
        with pytest.raises(Exception, match=f'{setter}\\(\\): expected True or False'):
            getattr(a, setter)(bad)


@pytest.mark.parametrize('name', sorted(set(BOOL_GETTERS) | set(BOOL_SETTERS)))
def test_docstrings_use_true_false(name):
    doc = getattr(am3_mojo.AM3, name).__doc__
    assert '-> bool' in doc.split('\n')[0] or 'arg0: bool' in doc.split('\n')[0]
    for phrase in ('Set to 0', 'set to 0', '/ 1 (', '0=', '1=', 'Default:0', 'Default: 0',
                   'Setting to zero'):
        assert phrase not in doc, f'{name}: {phrase!r} still in docstring'


def test_simulation_identical_with_true_false_and_1_0():
    """Switches set with True/False or with 1/0 give bitwise the same run."""
    def run(on, off):
        a = am3_mojo.AM3()
        a.set_verbosity_level(off)
        a.update_energy_grid(1e-6, 1e2, 1e17)
        a.set_process_hadronic(off)
        a.set_process_parse_sed(on)
        a.set_process_annihilation(on)
        a.set_process_merge_positrons_into_electrons(on)
        a.init_kernels()
        a.set_mag_field(0.1)
        a.set_escape_timescale(3e5)
        a.set_solver_time_step(3e3)
        a.set_powerlaw_injection_parameters_electrons(4e48, 1e43, 5e7, 5e7, 5e11, 2.2, 2.2, 1.0)
        for _ in range(20):
            a.evolve_step()
        return dump_state(a)

    x, y = run(True, False), run(1, 0)
    assert x.keys() == y.keys()
    assert [n for n in x if not np.array_equal(x[n], y[n], equal_nan=True)] == []
    assert np.max(x['get_photons']) > 0


def test_print_process_switches_shows_true_false():
    """The switch table is printed by Mojo straight to the process stdout, so it
    is captured from a subprocess."""
    code = ('import sys; sys.path.insert(0, %r); import am3_mojo; a = am3_mojo.AM3(); '
            'a.set_process_escape(False); a.set_process_pp(True); a.print_process_switches()'
            % IMPLEMENTATIONS['mojo'][0])
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         check=True).stdout
    assert 'escape: False' in out
    assert 'proton p-p: True' in out
    assert 'parse sed components: True' in out
    body = out.split('\n', 1)[1]  # skip the header '... (at step: 0)'
    values = re.findall(r'(?:: |\) )(True|False|[01])\b', body)
    assert len(values) >= 40 and set(values) <= {'True', 'False'}, out
