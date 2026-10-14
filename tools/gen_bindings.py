"""Generate src/am3_mojo_module.mojo, the Python extension module `am3_mojo`.

Its class `AM3` has the methods, argument types and docstrings of the AM3
v1.2.0 Python interface (tools/am3_api.json, introspected from the original
pybind11 module), plus the methods for the processes am3_mojo adds
(tools/am3_mojo_api.json, same format).

usage: python tools/gen_bindings.py
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE = 'am3_mojo'
VERSION = '0.2.0'           # am3_mojo release
API_VERSION = '1.0.1'       # AM3 interface it implements
api = json.load(open(os.path.join(ROOT, 'tools', 'am3_api.json')))
ext = json.load(open(os.path.join(ROOT, 'tools', 'am3_mojo_api.json')))
clash = {n for n, _ in api} & {n for n, _ in ext}
if clash:
    raise SystemExit(f'am3_mojo_api.json redefines AM3 methods: {sorted(clash)}')
api = api + ext
# signatures/docstrings refer to the class as am3_mojo.AM3
api = [(n, d.replace('am3.AM3', f'{MODULE}.AM3')) for n, d in api]

def mojo_str(s):
    s = s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    return '"' + s + '"'

F = 'typing.SupportsFloat | typing.SupportsIndex'

# On/off switches are Python bools: getters return True/False, and setters
# reject anything else (0/1, numpy booleans, ...). get_verbosity_level returns a
# float in the original and a bool here. bool_doc rewrites their docstrings
# from 0/1 to False/True.
BOOL_RET_OVERRIDE = {'get_verbosity_level'}


def bool_doc(doc):
    lines = doc.split('\n')
    for i, l in enumerate(lines[1:], 1):
        l = re.sub(r'\bset to 0/1 \(off/on\)', 'set to False/True (off/on)', l)
        l = re.sub(r'\b0 \(([^()]*(?:\([^()]*\)[^()]*)*)\) / 1 \(', r'False (\1) / True (', l)
        l = re.sub(r'\b0=([^,]*), 1=', r'False=\1, True=', l)
        l = l.replace('(1=yes, 0=no)', '(True=yes, False=no)')
        l = re.sub(r'Default:\s*0$', 'Default: False', l)
        l = l.replace('Setting to zero turns off', 'Setting to False turns off')
        lines[i] = l
    return '\n'.join(lines)


# the profiler also has slots for the stages of the processes am3_mojo adds
PROFILED = ('36 sub-steps profiled in the evolution of one step',
            "43 sub-steps profiled in the evolution of one step (AM3's 36, then 7 of "
            'the processes am3_mojo adds)')

I = 'typing.SupportsInt | typing.SupportsIndex'
funcs = []
regs = []
for name, doc in api:
    if name in ('get_profiled_times', 'get_profiled_time_labels'):
        assert PROFILED[0] in doc, name
        doc = doc.replace(*PROFILED)
    if name in BOOL_RET_OVERRIDE:
        doc = doc.replace(f'{name}(self: {MODULE}.AM3) -> float', f'{name}(self: {MODULE}.AM3) -> bool', 1)
    first = doc.strip().split('\n')[0]
    if first.endswith('-> bool') or 'arg0: bool' in first:
        doc = bool_doc(doc)
    if 'Overloaded function' in doc or first.startswith(name + '(*args'):
        # pybind11 overload set of two identical definitions: take the first signature
        m = re.search(r'1\. ' + name + r'\(self: ' + re.escape(MODULE) + r'\.AM3([^\n]*)', doc)
        sig = '(' + m.group(1)
        doc_clean = doc
    else:
        sig = re.sub(r'^\w+\(self: ' + re.escape(MODULE) + r'\.AM3', '(', first)
    args_part, ret = sig.rsplit(' -> ', 1)
    args = [a.strip() for a in args_part.strip('()').lstrip(',').split(', arg') if a.strip()]
    types = [a.split(': ', 1)[1] for a in args]
    pnames = [f'a{k}' for k in range(len(types))]
    conv = []
    for p, t in zip(pnames, types):
        if t == 'bool': conv.append(f'_to_bool({p}, "{name}")')
        elif t == F: conv.append(f'_to_f64({p})')
        elif t == I: conv.append(f'_to_int({p})')
        elif 'ArrayLike' in t: conv.append(f'_to_vec({p})')
        else: raise SystemExit(f'unknown arg type {t} in {name}')
    call = f'self_[].{name}(' + ', '.join(conv) + ')'
    if ret == 'None':
        body = f'    {call}\n    return PythonObject(None)'
    elif ret == 'bool':
        body = f'    return PythonObject({call})'
    elif ret in ('float', 'int'):
        body = f'    return PythonObject({call})'
    elif ret.startswith('numpy.typing.NDArray'):
        body = f'    return _to_numpy({call})'
    elif ret.startswith('tuple['):
        body = f'    var t = {call}\n    return Python.tuple(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7])'
    elif ret == 'list[float]':
        body = f'    return _to_pylist_f({call})'
    elif ret == 'list[str]':
        body = f'    return _to_pylist_s({call})'
    else:
        raise SystemExit(f'unknown ret {ret} in {name}')
    params = ''.join(f', {p}: PythonObject' for p in pnames)
    funcs.append(f'def _b_{name}(self_: Pointer[AM3, MutUnsafeAnyOrigin]{params}) raises -> PythonObject:\n{body}\n')
    regs.append(f'        _ = t.def_method[_b_{name}]("{name}", docstring={mojo_str(doc)})')

head = '''"""Python extension module `am3_mojo`.

Exposes class `AM3` with the methods, argument conventions and docstrings of
the AM3 v1.2.0 Python interface.
Generated by tools/gen_bindings.py - do not edit by hand.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.python.numpy import copy_to_numpy_array
from std.os import abort, getenv, setenv, unsetenv
from std.runtime import initialize_runtime
from am3_mojo.api.am3class import AM3
from am3_mojo.utils.buffers import Vec


def _to_bool(o: PythonObject, name: StaticString) raises -> Bool:
    """On/off switch argument: only the Python booleans True and False."""
    var b = Python.import_module("builtins")
    if Bool(py=b.isinstance(o, b.bool)):
        return Bool(py=o)
    raise Error(String(name, "(): expected True or False, got ", String(b.repr(o))))


# Mojo's runtime sets these at the C level (visible to child processes but not
# in os.environ) when its Python interop initialises during import. Left in
# place, PYTHONEXECUTABLE=/usr/bin/python3 makes every Python subprocess started
# afterwards (subprocess with sys.executable, multiprocessing workers) run
# outside the virtual environment. PyInit_am3_mojo restores the previous values.
comptime _N_ENV = 3
comptime _UNSET = "\x01<unset>"


def _env_key(i: Int) -> String:
    if i == 0:
        return "PYTHONEXECUTABLE"
    elif i == 1:
        return "PYTHONPATH"
    return "MOJO_PYTHON_LIBRARY"


def _save_env() -> List[String]:
    var saved = List[String]()
    for i in range(_N_ENV):
        saved.append(getenv(_env_key(i), _UNSET))
    return saved^


def _restore_env(saved: List[String]):
    for i in range(_N_ENV):
        if saved[i] == _UNSET:
            _ = unsetenv(_env_key(i))
        else:
            _ = setenv(_env_key(i), saved[i], True)


def _to_f64(o: PythonObject) raises -> Float64:
    return Float64(py=o)


def _to_int(o: PythonObject) raises -> Int:
    # integers only (as AM3's pybind11 module): no floats, no strings
    return Int(py=Python.import_module("operator").index(o))


def _to_vec(o: PythonObject) raises -> Vec:
    var np = Python.import_module("numpy")
    if Int(py=np.ndim(o)) != 1:  # (ascontiguousarray makes scalars 1-d)
        raise Error("Number of dimensions must be one and/ or array size is wrong")
    var arr = np.ascontiguousarray(o, dtype=np.float64)
    var n = Int(py=arr.size)
    var v = Vec(length=n, fill=0.0)
    if n > 0:
        var src = arr.ctypes.data.unsafe_get_as_pointer[DType.float64]()
        for i in range(n):
            v[i] = src[unsafe_offset=i]
    _ = arr  # the numpy array must outlive the reads through its pointer
    return v^


def _to_numpy(v: Vec) raises -> PythonObject:
    return copy_to_numpy_array(Span(v))


def _to_pylist_f(v: Vec) raises -> PythonObject:
    var l = Python.list()
    for i in range(len(v)):
        l.append(PythonObject(v[i]))
    return l


def _to_pylist_s(v: List[String]) raises -> PythonObject:
    var l = Python.list()
    for i in range(len(v)):
        l.append(PythonObject(v[i]))
    return l


'''
tail = '''

@export
def PyInit_am3_mojo() abi("C") -> PythonObject:
    var env = _save_env()
    initialize_runtime()  # Mojo worker pool used by the parallel sections
    try:
        var m = PythonModuleBuilder("am3_mojo")
        ref t = m.add_type[AM3]("am3_mojo.AM3")
        _ = t.def_init_defaultable[AM3]()
%s
        var mod = m.finalize()
        # registered as "am3_mojo.AM3" so that the type's __module__ is
        # "am3_mojo"; expose it under its plain name
        var cls = mod.__getattr__("am3_mojo.AM3")
        mod.__setattr__("AM3", cls)
        _ = Python.import_module("builtins").delattr(mod, "am3_mojo.AM3")
        mod.__setattr__("__version__", PythonObject("%s"))
        mod.__setattr__("AM3_API_VERSION", PythonObject("%s"))
        mod.__setattr__(
            "__doc__",
            PythonObject("am3_mojo: AM3 lepto-hadronic radiation code in Mojo"),
        )
        _restore_env(env)
        return mod
    except e:
        abort(String("error creating Python Mojo module:", e))
''' % ('\n'.join(regs), VERSION, API_VERSION)
open(os.path.join(ROOT, 'src', 'am3_mojo_module.mojo'), 'w').write(head + '\n\n'.join(funcs) + tail)
print(len(funcs), 'bindings')
