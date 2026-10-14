"""The Python API of am3_mojo.

* every method of the AM3 v1.2.0 Python interface exists, with its signature and docstring (tools/am3_api.json,
  introspected from the original module), and so does every method am3_mojo adds (tools/am3_mojo_api.json);
* the version attributes __version__ and AM3_API_VERSION;
* on/off switches are Python booleans: getters return True/False and setters take nothing else (not even 0/1 or
  numpy booleans); docstrings and the printed switch table say True/False; the added processes are on by default.

Each test checks every method and lists all that fail.
"""
import json, os, re, subprocess, sys
import numpy as np

from support.paths import ROOT, EXTENSION_SWITCHES, IMPLEMENTATIONS, available, load

am3_mojo = load('mojo') if available('mojo') else None
API = json.load(open(os.path.join(ROOT, 'tools', 'am3_api.json')))
DOC = dict(API)
SIG = {n: d.split('\n')[0] for n, d in API}
BOOL_GETTERS = sorted([n for n, s in SIG.items() if n.startswith('get_') and s.endswith('-> bool')]
                      + ['get_verbosity_level'])
BOOL_SETTERS = sorted(n for n, s in SIG.items() if n.startswith('set_') and 'arg0: bool' in s)
PAIRS = [s for s in BOOL_SETTERS if 'get_' + s[4:] in BOOL_GETTERS]
# the methods am3_mojo adds, and the switches of the added processes
EXT_SIG = {n: d.split('\n')[0] for n, d in json.load(open(os.path.join(ROOT, 'tools', 'am3_mojo_api.json')))}
EXT_PAIRS = sorted(n for n in EXT_SIG if n.startswith('set_process_'))
ALL_PAIRS = PAIRS + EXT_PAIRS
# Quirk of the original kept on purpose (tests/reference/README.md, "Kept as in AM3"): these setters store
# int(value) + 1 into a C++ bool, so the switch is stuck at True and set(False) has no effect.
STUCK_TRUE = {'set_optimize_compton_emission_grid', 'set_optimize_compton_target_grid',
              'set_optimize_bethe_heitler_outgoing_pairs_grid',
              'set_optimize_photopion_target_photon_grid'}
NON_BOOLEANS = [0, 1, np.bool_(True), np.bool_(False), np.int64(1), 'x', 'True', 0.5, 1.0, 2, -1, None, [1]]


def expect(setter, value):
    return True if setter in STUCK_TRUE else value


def doc_of(name):
    method = getattr(am3_mojo.AM3, name, None)
    return (method.__doc__ or '') if method is not None else ''


def test_module_attributes():
    assert am3_mojo.__version__ == '0.2.0'
    assert am3_mojo.AM3_API_VERSION == '1.0.1'
    assert am3_mojo.AM3.__module__ == 'am3_mojo'


def test_method_signatures_and_docstrings():
    """Every method of the AM3 interface exists, with the original signature
    (class name am3_mojo.AM3; switch methods use bool; see gen_bindings.py),
    and every method am3_mojo adds, with its own."""
    wrong = []
    for name, sig in SIG.items():
        doc = doc_of(name)
        expected = sig.replace('am3.AM3', 'am3_mojo.AM3')
        if name == 'get_verbosity_level':
            expected = expected.replace('-> float', '-> bool')
        if '(*args' in sig:
            # pybind11 overloads: the whole docstring lists every overload
            if doc != DOC[name].replace('am3.AM3', 'am3_mojo.AM3'):
                wrong.append(name)
        elif doc.split('\n')[0] != expected:
            wrong.append(name)
    wrong += [name for name, sig in EXT_SIG.items()
              if doc_of(name).split('\n')[0] != sig.replace('am3.AM3', 'am3_mojo.AM3')]
    assert wrong == []


def test_switch_lists_are_complete():
    assert len(BOOL_GETTERS) == 59 and len(BOOL_SETTERS) == 59
    assert len(PAIRS) == 59  # every switch setter has a matching getter
    assert len(EXT_PAIRS) == 15 and all('get_' + s[4:] in EXT_SIG for s in EXT_PAIRS)
    # `mojo-am3` turns off master switches of the added processes
    assert set(EXTENSION_SWITCHES) <= set(EXT_PAIRS)


def test_getters_return_bool():
    a = am3_mojo.AM3()
    wrong = []
    for name in BOOL_GETTERS + ['get_' + s[4:] for s in EXT_PAIRS]:
        v = getattr(a, name)()
        if type(v) is not bool:
            wrong.append(f'{name}() returned {v!r} ({type(v).__name__})')
    assert wrong == []


def test_added_process_switches_on_by_default():
    a = am3_mojo.AM3()
    assert [s for s in EXT_PAIRS if getattr(a, 'get_' + s[4:])() is not True] == []


def test_setters_round_trip_false_true():
    a = am3_mojo.AM3()
    wrong = []
    for setter in ALL_PAIRS:
        get, put = getattr(a, 'get_' + setter[4:]), getattr(a, setter)
        seen = []
        for value in (False, True, False):
            put(value)
            seen.append(get())
        if seen != [expect(setter, False), True, expect(setter, False)]:
            wrong.append(f'{setter}: {seen}')
    assert wrong == []


def test_setters_reject_non_booleans():
    """Only True and False are accepted: not the integers 0 and 1, and not
    numpy booleans."""
    a = am3_mojo.AM3()
    wrong = []
    for setter in BOOL_SETTERS + EXT_PAIRS:
        for bad in NON_BOOLEANS:
            try:
                getattr(a, setter)(bad)
                wrong.append(f'{setter}({bad!r}) accepted')
            except Exception as e:
                if f'{setter}(): expected True or False' not in str(e):
                    wrong.append(f'{setter}({bad!r}): {e}')
    assert wrong == []


def test_rejected_value_leaves_switch_unchanged():
    a = am3_mojo.AM3()
    wrong = []
    for setter in ALL_PAIRS:
        get = getattr(a, 'get_' + setter[4:])
        before = get()
        try:
            getattr(a, setter)(int(not before))
            wrong.append(f'{setter}({int(not before)}) accepted')
        except Exception:
            pass
        if get() is not before:
            wrong.append(f'{setter}: {before} -> {get()}')
    assert wrong == []


def test_docstrings_use_true_false():
    wrong = []
    names = set(BOOL_GETTERS) | set(BOOL_SETTERS) | set(EXT_PAIRS) | {'get_' + s[4:] for s in EXT_PAIRS}
    for name in sorted(names):
        doc = doc_of(name)
        first = doc.split('\n')[0]
        if '-> bool' not in first and 'arg0: bool' not in first:
            wrong.append(f'{name}: {first}')
        wrong += [f'{name}: {phrase!r} in docstring'
                  for phrase in ('Set to 0', 'set to 0', '/ 1 (', '0=', '1=', 'Default:0', 'Default: 0',
                                 'Setting to zero') if phrase in doc]
    assert wrong == []


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
