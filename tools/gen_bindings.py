"""Generate src/am3_mojo_module.mojo, the Python extension module `am3_mojo`.

Its class `AM3` has the methods, argument types and docstrings of the AM3 v1.2.0
Python interface (tools/am3_api.json). That file was introspected from the
original pybind11 module and has am3_mojo's notes on get_electron_neutrinos and
get_muon_neutrinos. The class also has the methods for the processes that
am3_mojo adds (tools/am3_mojo_api.json, same format). Like the rest of the Mojo
code, the module is then formatted with `mojo format`. The script calls the
`mojo` next to the Python that runs it (.venv/bin/mojo).

The generated module records a digest of the generator and of the two API
descriptions. With `--check`, the script exits with status 1 if the module was
generated from other ones (tools/mojo_build_backend.py runs it before
compiling).

usage: .venv/bin/python tools/gen_bindings.py [--check]
"""

import argparse, hashlib, json, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "src", "am3_mojo_module.mojo")


def inputs_digest():
    h = hashlib.sha256()
    for f in ("gen_bindings.py", "am3_api.json", "am3_mojo_api.json"):
        with open(os.path.join(ROOT, "tools", f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


INPUTS_LINE = "# inputs sha256 "
ap = argparse.ArgumentParser(
    description="Generate src/am3_mojo_module.mojo from tools/am3_api.json and"
    " tools/am3_mojo_api.json."
)
ap.add_argument(
    "--check",
    action="store_true",
    help="write nothing; exit with status 1 if the module was generated from"
    " another generator or other API descriptions",
)
if ap.parse_args().check:
    with open(OUT) as f:
        sys.exit(0 if INPUTS_LINE + inputs_digest() + "\n" in f.read() else 1)
MODULE = "am3_mojo"
VERSION = "0.2.0"  # am3_mojo release
API_VERSION = "1.0.1"  # AM3 interface it implements
api = json.load(open(os.path.join(ROOT, "tools", "am3_api.json")))
ext = json.load(open(os.path.join(ROOT, "tools", "am3_mojo_api.json")))
clash = {n for n, _ in api} & {n for n, _ in ext}
if clash:
    raise SystemExit(
        f"am3_mojo_api.json redefines AM3 methods: {sorted(clash)}"
    )
api = api + ext
api = [(n, d.replace("am3.AM3", f"{MODULE}.AM3")) for n, d in api]


def mojo_str(s):
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return '"' + s + '"'


F = "typing.SupportsFloat | typing.SupportsIndex"

# On/off switches are Python bools: getters return True/False, and setters
# reject anything else (for example 0/1 or numpy booleans). get_verbosity_level
# returns a float in the original and a bool here.
BOOL_RET_OVERRIDE = {"get_verbosity_level"}


def bool_doc(doc):
    lines = doc.split("\n")
    for i, l in enumerate(lines[1:], 1):
        l = re.sub(r"\bset to 0/1 \(off/on\)", "set to False/True (off/on)", l)
        l = re.sub(
            r"\b0 \(([^()]*(?:\([^()]*\)[^()]*)*)\) / 1 \(",
            r"False (\1) / True (",
            l,
        )
        l = re.sub(r"\b0=([^,]*), 1=", r"False=\1, True=", l)
        l = l.replace("(1=yes, 0=no)", "(True=yes, False=no)")
        l = re.sub(r"Default:\s*0$", "Default: False", l)
        l = l.replace("Setting to zero turns off", "Setting to False turns off")
        lines[i] = l
    return "\n".join(lines)


# the profiler also has slots for the stages of the processes am3_mojo adds
PROFILED = (
    "36 sub-steps profiled in the evolution of one step",
    "47 sub-steps profiled in the evolution of one step "
    "(AM3's 36, then 11 of "
    "the processes am3_mojo adds)",
)

# AM3's docstring has the two treatments the wrong way round (ProtonProton.h has
# them right). The value 0 applies the direct kernel, and 1 sends the pi0 into
# the cascade.
PI0_CASCADE = (
    "Set to 0 (use same treatment as for photopion) / 1 (use Kaafexhiu et al."
    " 2014 treatment with direct kernel from protons)",
    "Set to 0 (use Kafexhiu et al. 2014 treatment with direct kernel from"
    " protons) / 1 (use same treatment as for photopion)",
)

# AM3's docstrings give 100 as the default of the matrix threshold, which both
# codes set to 10 (SimulationManager.cc)
THRESHOLD_MATRIX = ("Default:  100\n", "Default:  10\n")

# AM3's docstring gives the time step 0.01 of the escape timescale. Both codes
# set 1e-3 of the constructor's, 1e6 s (RunParams.cc), and keep it when the
# escape timescale changes.
TIME_STEP = (
    "Default: 0.01 * escape_timescale\n",
    "Default: 1e3 (1e-3 * the default escape_timescale; it does not follow"
    " set_escape_timescale)\n",
)

# AM3's docstring says that the minimum IC photon energy is used only with
# set_lep_optimizer_on and is 1e-3 eV by default. Both codes always use it, by
# default at point 100 of the photon grid (SimulationManager.cc).
COMPTON_EMIN = (
    "This parameter is only used when set_lep_optimizer_on is turned on.\n"
    "Default: 1e-3 eV\n",
    "It is used whether or not set_lep_optimizer_on is turned on, which sets it"
    " to 1e-3 eV.\nDefault: 7.8e-3 eV\n",
)

# AM3's docstring prints a backslash and an n instead of a line break
PP_DENSITY = ("[cm-3].\\nDefault", "[cm-3].\nDefault")

# AM3's docstrings name six of the eight values that the power-law getters
# return, those of the setters (AM3.cc)
POWERLAW = (
    "luminosity, minimum energy, break energy, maximum energy, spectral index"
    " below the break, and spectral index above the break.",
    "region volume [cm^3], injection luminosity [erg/s], minimum energy [eV],"
    " break energy [eV], maximum energy [eV], spectral index below the break,"
    " spectral index above the break, and exponential cutoff steepness.",
)

# evolve_step raises for photo-pion production switched on after an
# init_kernels without it, because init_kernels builds its kernels
PHOTOPION = (
    "Master switch for photopion production.\n",
    "Master switch for photopion production. Set it before `init_kernels`:"
    " switched on after an `init_kernels` without it, `evolve_step`"
    " raises.\n",
)

# AM3's docstrings say that the e- and e+ are not updated while the positrons
# are merged. Both codes split the merged pairs after every merged step
# (SimulationManager.cc).
MERGED = (
    "If `get_process_merge_positrons_into_electrons() == 1`, this array will"
    " not be updated; the densities of electrons and positrons can only be"
    " checked via get_pairs().\n",
    "If `get_process_merge_positrons_into_electrons() == 1`, the merged pairs"
    " of each bin are split into electrons and positrons in the ratio of the"
    " bin's electrons and positrons before the merge (all electrons where"
    " there were none).\n",
)

# AM3's docstrings say that the maximum target-photon energies of inverse
# Compton and Bethe-Heitler pair production are used only with
# set_lep/had_optimizer_on. They say the same of the minimum proton energy of
# Bethe-Heitler pair production and give 1 MeV and 1 TeV as the defaults. Both
# codes always use these limits, by default at m_e c^2 and at point 70 of the
# hadron grid, 1.1 TeV (SimulationManager.cc).
COMPTON_EMAX = (
    "This parameter is only used when set_lep_optimizer_on is turned on.\n"
    "Default: 1MeV (approx. m_e c^2)\n",
    "It is used whether or not set_lep_optimizer_on is turned on, which sets it"
    " to 1 MeV.\nDefault: 511 keV (m_e c^2)\n",
)
BH_PROTONS_MIN = (
    "This parameter is only used when set_had_optimizer_on is tuned on.\n"
    "Default: 1 TeV\n",
    "It is used whether or not set_had_optimizer_on is turned on, which sets it"
    " to 1 TeV.\nDefault: 1.1 TeV\n",
)
# its first overload describes another parameter, of the gamma-gamma kernel
BH_PHOTON_MAX = (
    "optimisation of full kernel of injection of lepton pairs due to gamma"
    " gamma pair annihilation: factor above threshold energy where to stop"
    " kernel\nThis parameter is only used when set_had_optimizer_on is tuned"
    " on.\nDefault: 1e6\n",
    "For fast version of Bethe-Heitler e-production: max. target photon"
    " energy\nIt is used whether or not set_had_optimizer_on is turned on,"
    " which sets it to 1 MeV.\nDefault: 511 keV (m_e c^2)\n",
)
BH_PHOTON_MAX_2 = (
    "This parameter is only used when set_had_optimizer_on is tuned on.\n"
    "Default: 1 MeV (approx. m_e c^2)\n",
    "It is used whether or not set_had_optimizer_on is turned on, which sets it"
    " to 1 MeV.\nDefault: 511 keV (m_e c^2)\n",
)
PHOTOPION_MAX = (
    "Default: 1 MeV (approx. m_e c^2)\n",
    "Default: 511 keV (m_e c^2)\n",
)

# AM3 stores the grid switches in C++ bools, which read as True whatever is
# set (AM3.cc). So AM3's every-2nd-grid-point code is unreachable, and am3_mojo
# always uses the full grid.
GRID = (
    "the switch always reads as True (AM3 stores it in a C++ bool) and the full"
    " grid is used: AM3's every-2nd-grid-point option is unreachable.\n"
)
COMPTON_EMISSION_GRID = (
    "0=full photon out grid, 1=every 2nd photon grid point + interpolation.\n"
    "This parameter is only used when set_lep_optimizer_on is turned on.\n",
    GRID,
)
COMPTON_TARGET_GRID = (
    "0=full target grid, 1=every 2nd target grid point.\nThis parameter is"
    " only used when set_lep_optimizer_on is turned on.\n",
    GRID,
)
PHOTOPION_GRID = (
    " 0=full target grid, 1=every 2nd target grid point\n",
    " " + GRID,
)
BH_PAIRS_GRID = (
    "0=full electrons and positrons out grid, 1=every 2nd electrons and"
    " positrons grid point + interpolation\nThis parameter is only used when"
    " set_had_optimizer_on is tuned on.\n",
    GRID,
)

# AM3's docstrings do not say that init_kernels ends with clear_am3 (both
# codes). The escape, injection and extra-cooling rates set before init_kernels
# are lost, and a fresh object has no escape until set_escape_timescale is
# called.
INIT_KERNELS = (
    "on the sizes of the energy grids.\n",
    "on the sizes of the energy grids.\nIt ends with clear_am3(): set the"
    " escape timescale, the injection and the escape and extra-cooling arrays"
    " after it.\n",
)
ESCAPE_DEFAULT = (
    "Default: 1e6.\n",
    "Default: 1e6. The escape rates follow it from this call on: a fresh"
    " object, init_kernels() and clear_am3() leave them zero.\n",
)

# AM3's docstring names only the densities and the injection rates. clear_am3
# also zeroes the escape and extra-cooling arrays and the rates that the
# processes computed (both codes), and am3_mojo resets the estimated maximum
# energies.
CLEAR_AM3 = (
    "Set all current particle densities and injection rates to zero.\n",
    "Set to zero all current particle densities, the injection, escape and"
    " extra-cooling arrays and the rates the processes computed, and reset the"
    " estimated maximum energies to those of a fresh object.\n",
)

# With parse_sed off, the steps evolve only the totals. If parse_sed is switched
# on after such a step, am3_mojo starts the components from zero. (AM3 evolves
# them on from the values that they kept, which then no longer add up to the
# totals.)
PARSE_SED = (
    "while for efficiency it should be turned off.\n",
    "while for efficiency it should be turned off.\nSteps with it off evolve"
    " only the totals; switched on after such a step, the components start"
    " from zero and count the particles from the next step on.\n",
)

# the two methods take no argument: they switch a selection on, and no call
# switches it off again
OPTIMIZER_LEP = (
    "Include or exclude leptonic optimization routines.\nSetting to 1 turns"
    " on the necessary selection of switches for optimization of leptonic"
    " processes.\nSet to 0 (off) / 1 (on)\n",
    "Turn on the selection of switches and limits that optimizes the leptonic"
    " processes.\nThe method takes no argument: to go back, set the switches"
    " and limits one by one.\n",
)
OPTIMIZER_HAD = (
    "Include or exclude hadronic optimization routines. Setting to 1 turns on"
    " the necessary selection of switches for optimization of hadronic"
    " processes.\nSet to 0 (off) / 1 (on)\n",
    "Turn on the selection of switches and limits that optimizes the hadronic"
    " processes.\nThe method takes no argument: to go back, set the switches"
    " and limits one by one.\n",
)
# The approximation puts the pairs of an absorbed photon in 15 equal shares on
# 14 lepton bins. Eight shares go from half the photon energy up, and seven go
# from there down. AM3's verbose output calls it "8bin".
EIGHT_BIN = ("14-bin approx.", "eight-bin approx.")
# a volume
VOLUME = ("region volume [cm-3]", "region volume [cm3]")
# the getter divides the electrons by their own source: with positrons tracked
# apart, by the electrons' half of a pair source (both codes)
PAIR_EMISSION = (
    " [s]\n",
    " [s]\nWith positrons tracked apart: the electrons over the electrons'"
    " part of the source.\n",
)
PAIR_EMISSION_DECAY = (
    " [s]\n",
    " [s]\nWith positrons tracked apart: the electrons over the electrons'"
    " part of the source.\nNeeds parse_sed on: with it off, this source is"
    " part of the total only and the timescale is infinite.\n",
)
# the decay getters follow the decay switches, as the loss getters do
DECAY_OFF = (
    " [s]\n",
    " [s]\nInfinite (1e200 s) while the decay is switched off.\n",
)

# AM3's docstring gives no range. am3_mojo raises outside the range, and with
# pp on, init_kernels raises when the pp transition energy lies above the
# hadron grid.
GRID_RANGE = (
    "max. energy for all particles [eV]\n",
    "max. energy for all particles [eV]\nRange: 1e-100 eV <= min. photon"
    " energy < 511 keV, 1e-100 eV <= min. neutrino energy < 1 GeV and 10 GeV"
    " <= max. energy < 1e300 eV. With pp collisions on, init_kernels() also"
    " needs the pp transition energy (set_pp_transition_energy_to_delta_approx,"
    " by default 100 GeV) on the hadron grid: with the default, a max. energy"
    " of 110 GeV or more; lower it or switch pp off for a smaller grid.\n",
)

# AM3's docstring gives no limit. With pp on, init_kernels and evolve_step
# raise for a transition energy above the hadron grid.
PP_TRANSITION = (
    "Default: 1e11\n",
    "Default: 1e11\nWith pp collisions on, it must lie on the hadron grid"
    " (update_energy_grid): init_kernels() and, if set after it, the next"
    " evolve_step() raise for an energy above the grid's top. A negative"
    " value stands for 1e11.\n",
)

# init_kernels builds the kernels of these four processes only with them on,
# as in AM3. Switched on after it, they make nothing, as the processes
# am3_mojo adds (photo-pion production raises: PHOTOPION). AM3's docstrings do
# not say so.
LATE = (
    " Set it before `init_kernels`: switched on after an `init_kernels`"
    " without it, it stays inactive.\n"
)
PP_LATE = (
    "Master switch for proton-proton interactions.\n",
    "Master switch for proton-proton interactions." + LATE,
)
BH_LATE = (
    "Master switch for Bethe-Heitler pair production (p gamma-> p e+ e-).\n",
    "Master switch for Bethe-Heitler pair production (p gamma-> p e+ e-)."
    + LATE,
)
GG_LATE = (
    "Master switch for photon-photon annihilation and resulting pair"
    " production (gamma gamma-> e+ e-). \n",
    "Master switch for photon-photon annihilation and resulting pair"
    " production (gamma gamma-> e+ e-)." + LATE,
)
IC_LATE = (
    "Master switch for electron/positron inverse Compton emission and"
    " cooling.\n",
    "Master switch for electron/positron inverse Compton emission and"
    " cooling." + LATE,
)

# AM3's docstrings give no range. More than one bin below a grid, AM3 returns
# a negative index, which numpy counts from the end of an array (up to one bin
# below, 0). am3_mojo raises below the grid's first point. Above the top point,
# the index is that of the point at or below the energy, as on the grid, so it
# reaches the grid's size only one bin above.
GRID_INDEX = (
    "grid array index\n",
    "grid array index\nRaises for an energy below the grid's first point. An"
    " energy one bin or more above the grid's top point gives the grid's size"
    " or more.\n",
)

CORRECTIONS = {
    "update_energy_grid": [GRID_RANGE],
    "set_process_pp": [PP_LATE],
    "set_process_bethe_heitler": [BH_LATE],
    "set_process_annihilation": [GG_LATE],
    "set_process_electron_compton": [IC_LATE],
    "eV_to_lep_grid_index": [GRID_INDEX],
    "eV_to_photon_grid_index": [GRID_INDEX],
    "eV_to_had_grid_index": [GRID_INDEX],
    "eV_to_neutrino_grid_index": [GRID_INDEX],
    "set_pp_transition_energy_to_delta_approx": [PP_TRANSITION],
    "set_lep_optimizer_on": [OPTIMIZER_LEP],
    "set_had_optimizer_on": [OPTIMIZER_HAD],
    "set_optimize_annihilation_pair_emission": [EIGHT_BIN],
    "set_powerlaw_injection_parameters_electrons": [VOLUME],
    "set_powerlaw_injection_parameters_protons": [VOLUME],
    "get_t_pair_emission_from_injection": [PAIR_EMISSION],
    "get_t_pair_emission_from_annihilation": [PAIR_EMISSION],
    "get_t_pair_emission_from_bethe_heitler": [PAIR_EMISSION],
    "get_t_pair_emission_from_photopion": [PAIR_EMISSION_DECAY],
    "get_t_pair_emission_from_proton_proton": [PAIR_EMISSION_DECAY],
    "get_t_pion_decay": [DECAY_OFF],
    "get_t_muon_decay": [DECAY_OFF],
    "clear_am3": [CLEAR_AM3],
    "set_process_parse_sed": [PARSE_SED],
    "get_profiled_times": [PROFILED],
    "init_kernels": [INIT_KERNELS],
    "set_escape_timescale": [ESCAPE_DEFAULT],
    "get_profiled_time_labels": [PROFILED],
    "set_process_pp_emission_pi0_into_cascade": [PI0_CASCADE],
    "set_solver_threshold_matrix": [THRESHOLD_MATRIX],
    "get_solver_threshold_matrix": [THRESHOLD_MATRIX],
    "set_solver_time_step": [TIME_STEP],
    "set_optimize_compton_emission_emin": [COMPTON_EMIN],
    "set_optimize_compton_target_emax": [COMPTON_EMAX],
    "set_optimize_bethe_heitler_incoming_protons_min": [BH_PROTONS_MIN],
    "set_optimize_bethe_heitler_target_photon_max": [
        BH_PHOTON_MAX,
        BH_PHOTON_MAX_2,
    ],
    "set_optimize_photopion_target_photon_max": [PHOTOPION_MAX],
    "set_optimize_compton_emission_grid": [COMPTON_EMISSION_GRID],
    "set_optimize_compton_target_grid": [COMPTON_TARGET_GRID],
    "set_optimize_photopion_target_photon_grid": [PHOTOPION_GRID],
    "set_optimize_bethe_heitler_outgoing_pairs_grid": [BH_PAIRS_GRID],
    "get_powerlaw_injection_parameters_electrons": [POWERLAW],
    "get_powerlaw_injection_parameters_protons": [POWERLAW],
    "set_pp_target_proton_density": [PP_DENSITY],
    "set_process_photopion": [PHOTOPION],
    "get_electrons": [MERGED],
    "get_positrons": [MERGED],
}

I = "typing.SupportsInt | typing.SupportsIndex"
funcs = []
regs = []
for name, doc in api:
    for old, new in CORRECTIONS.get(name, []):
        assert doc.count(old) == 1, name
        doc = doc.replace(old, new)
    if name in BOOL_RET_OVERRIDE:
        doc = doc.replace(
            f"{name}(self: {MODULE}.AM3) -> float",
            f"{name}(self: {MODULE}.AM3) -> bool",
            1,
        )
    first = doc.strip().split("\n")[0]
    if first.endswith("-> bool") or "arg0: bool" in first:
        doc = bool_doc(doc)
    if "Overloaded function" in doc or first.startswith(name + "(*args"):
        # pybind11 overload set of two identical definitions: take the first
        # signature
        m = re.search(
            r"1\. " + name + r"\(self: " + re.escape(MODULE) + r"\.AM3([^\n]*)",
            doc,
        )
        sig = "(" + m.group(1)
        doc_clean = doc
    else:
        sig = re.sub(r"^\w+\(self: " + re.escape(MODULE) + r"\.AM3", "(", first)
    args_part, ret = sig.rsplit(" -> ", 1)
    args = [
        a.strip()
        for a in args_part.strip("()").lstrip(",").split(", arg")
        if a.strip()
    ]
    types = [a.split(": ", 1)[1] for a in args]
    pnames = [f"a{k}" for k in range(len(types))]
    conv = []
    for p, t in zip(pnames, types):
        if t == "bool":
            conv.append(f'_to_bool({p}, "{name}")')
        elif t == F:
            conv.append(f'_to_f64({p}, "{name}")')
        elif t == I:
            conv.append(f'_to_int({p}, "{name}")')
        elif "ArrayLike" in t:
            conv.append(f"_to_vec({p})")
        else:
            raise SystemExit(f"unknown arg type {t} in {name}")
    call = f"self_[].{name}(" + ", ".join(conv) + ")"
    if ret == "None":
        body = f"    {call}\n    return PythonObject(None)"
    elif ret == "bool":
        body = f"    return PythonObject({call})"
    elif ret in ("float", "int"):
        body = f"    return PythonObject({call})"
    elif ret.startswith("numpy.typing.NDArray"):
        body = f"    return _to_numpy({call})"
    elif ret.startswith("tuple["):
        body = (
            f"    var t = {call}\n    return Python.tuple(t[0], t[1], "
            "t[2], t[3], t[4], t[5], t[6], t[7])"
        )
    elif ret == "list[float]":
        body = f"    return _to_pylist_f({call})"
    elif ret == "list[str]":
        body = f"    return _to_pylist_s({call})"
    else:
        raise SystemExit(f"unknown ret {ret} in {name}")
    params = "".join(f", {p}: PythonObject" for p in pnames)
    funcs.append(
        f"def _b_{name}(self_: Pointer[AM3, MutUnsafeAnyOrigin]{params}) "
        f"raises -> PythonObject:\n{body}\n"
    )
    regs.append(
        f'        _ = t.def_method[_b_{name}]("{name}", '
        f"docstring={mojo_str(doc)})"
    )

head = '''"""Python extension module `am3_mojo`.

Exposes class `AM3` with the methods, argument conventions and docstrings of
the AM3 v1.2.0 Python interface.
Generated by tools/gen_bindings.py. Do not edit it by hand.
"""

INPUTS_LINE

from std.os import getenv, setenv, unsetenv, abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python.numpy import copy_to_numpy_array
from std.runtime import initialize_runtime
from am3_mojo.api.am3class import AM3
from am3_mojo.utils.buffers import Vec

def _to_bool(o: PythonObject, name: StaticString) raises -> Bool:
    """On/off switch argument: only the Python booleans True and False."""
    var b = Python.import_module("builtins")
    if Bool(py=b.isinstance(o, b.bool)):
        return Bool(py=o)
    raise Error(String(name, "(): expected True or False, got ", \
String(b.repr(o))))

# Mojo's runtime sets these at the C level when its Python interop
# initializes during import. Child processes see them, but os.environ does
# not. If they stay in place, PYTHONEXECUTABLE=/usr/bin/python3 makes every
# Python subprocess started after the import (subprocess with sys.executable,
# multiprocessing workers) run outside the virtual environment.
# PyInit_am3_mojo restores the previous values.
comptime _N_ENV = 3
comptime _UNSET = "\\x01<unset>"

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

# The workers of Mojo's runtime spin for a while after each parallel section
# before they sleep. Spinning workers slow the serial work of the calling
# thread between the sections. This costs more than waking them: without
# spinning, a step on 4 threads takes 13-15 % less time. The runtime reads
# this variable when it starts. If the user did not set it, PyInit_am3_mojo
# sets it to 0 while the runtime starts.
comptime _BUSY_WAIT = "MODULAR_THREAD_BUSY_WAIT_US"

def _check_real(o: PythonObject, name: StaticString) raises:
    """A switch value or a complex number passed to a setter of a number
    raises, as booleans and complex numbers in an array do. Python and numpy
    convert True to 1, which, for example, picks pp pion model 1 or a field of
    1 G without a word. numpy keeps the real part of a complex number with
    only a warning. (The kind of numpy's array, not isinstance, because numpy
    complex scalars and 0-d arrays of booleans are no instances of bool or
    complex.)"""
    var b = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    var kind = String(np.asarray(o).dtype.kind)
    if kind == "b":
        raise Error(String(name, "(): expected a number, got ", \
String(b.repr(o))))
    if kind == "c":
        raise Error(String(name, "(): expected a real number, got ", \
String(b.repr(o))))

def _to_f64(o: PythonObject, name: StaticString) raises -> Float64:
    _check_real(o, name)
    return Float64(py=o)

def _to_int(o: PythonObject, name: StaticString) raises -> Int:
    # integers only (as AM3's pybind11 module): no floats, no strings
    _check_real(o, name)
    return Int(py=Python.import_module("operator").index(o))

def _booleans_among_numbers(o: PythonObject) raises -> Bool:
    """A list that mixes booleans with numbers converts to an array of
    numbers, True to 1, so the kind of the array does not show them."""
    var b = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    var types = b.set(b.map(b.type, o))
    if Bool(py=types.__contains__(b.bool)) or Bool(
        py=types.__contains__(np.bool_)
    ):
        return True
    if Bool(py=types.__contains__(np.ndarray)):
        for x in o:
            if Bool(py=b.isinstance(x, np.ndarray)) and String(
                x.dtype.kind
            ) == "b":
                return True
    return False

def _to_vec(o: PythonObject) raises -> Vec:
    var np = Python.import_module("numpy")
    if Int(py=np.ndim(o)) != 1:  # (ascontiguousarray makes scalars 1-d)
        raise Error("Number of dimensions must be one and/ or array size \
is wrong")
    # real numbers only, as for the setters of numbers: no strings, booleans
    # or complex numbers (the conversion drops their imaginary part)
    var a0 = np.asarray(o)
    var kind = String(a0.dtype.kind)
    if kind != "f" and kind != "i" and kind != "u":
        raise Error(String("expected an array of real numbers, got one of ", \
String(a0.dtype)))
    if not Bool(
        py=Python.import_module("builtins").isinstance(o, np.ndarray)
    ) and _booleans_among_numbers(o):
        raise Error("expected an array of real numbers, got booleans among \
numbers")
    var arr = np.ascontiguousarray(a0, dtype=np.float64)
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

# Without this guard, AM3.__init__ on an object that holds an AM3 already
# writes a new AM3 over it. It does not free the kernels and arrays of the old
# one (100 MB a call), and it resets the state without a message. A method of
# an object that __init__ has not run on yet raises. `probe` is such a method.
comptime _INIT_ONCE = """
def once(init, probe):
    def __init__(self):
        try:
            probe(self)
        except Exception:
            fresh = True
        else:
            fresh = False
        if not fresh:
            raise Exception(
                "AM3.__init__() on an object that is initialized already:"
                " create a new AM3 instead"
            )
        init(self)
    return __init__
"""

def _to_pylist_s(v: List[String]) raises -> PythonObject:
    var l = Python.list()
    for i in range(len(v)):
        l.append(PythonObject(v[i]))
    return l

'''
tail = """
@export
def PyInit_am3_mojo() abi("C") -> PythonObject:
    var env = _save_env()
    var spin = getenv(_BUSY_WAIT, _UNSET)
    if spin == _UNSET:
        _ = setenv(_BUSY_WAIT, "0", True)
    initialize_runtime()
    if spin == _UNSET:
        _ = unsetenv(_BUSY_WAIT)
    try:
        var m = PythonModuleBuilder("am3_mojo")
        ref t = m.add_type[AM3]("am3_mojo.AM3")
        _ = t.def_init_defaultable[AM3]()
%s
        var mod = m.finalize()
        # The type is registered as "am3_mojo.AM3" so that its __module__ is
        # "am3_mojo".
        var cls = mod.__getattr__("am3_mojo.AM3")
        mod.__setattr__("AM3", cls)
        cls.__setattr__(
            "__doc__",
            PythonObject(
                "One zone of the AM3 lepto-hadronic radiation code. Choose the"
                " processes and the energy grid, call init_kernels(), set the"
                " parameters, then advance it with evolve_step()."
            ),
        )
        var builtins = Python.import_module("builtins")
        var ns = Python.dict()
        _ = builtins.exec(PythonObject(_INIT_ONCE), ns)
        cls.__setattr__(
            "__init__",
            ns["once"](
                cls.__getattr__("__init__"),
                cls.__getattr__("get_verbosity_level"),
            ),
        )
        _ = builtins.delattr(mod, "am3_mojo.AM3")
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
""" % ("\n".join(regs), VERSION, API_VERSION)
head = head.replace("INPUTS_LINE", INPUTS_LINE + inputs_digest())
# A run that fails leaves the module as it was. (Written in place, an
# unformatted or cut module with the new digest passes --check from then on.)
with tempfile.TemporaryDirectory(dir=os.path.dirname(OUT)) as tmp:
    new = os.path.join(tmp, os.path.basename(OUT))
    with open(new, "w") as f:
        f.write(head + "\n".join(funcs) + tail)
    subprocess.run(
        [
            os.path.join(os.path.dirname(sys.executable), "mojo"),
            "format",
            "-q",
            new,
        ],
        check=True,
    )
    os.replace(new, OUT)
print(len(funcs), "bindings")
