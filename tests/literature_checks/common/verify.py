"""The registry of derivation checks and the equivalence tests they use.

A derivation check is a function decorated with @derivation("<inventory id>"). It returns a list of Result lines
and passes when all of them pass. The tests, strongest first:
  exact      the difference simplifies to zero (sympy), often after a substitution chosen per equation that makes
             it rational (see zero_numerator);
  numeric    the two sides agree at random points to `dps` digits (mpmath), in every branch;
  control    a deliberately broken README side must fail the same test, so that a test that cannot fail is noticed.

The derivation modules (MODULES) live in the process folders of tests/literature_checks/; load_modules() puts these
folders and common/ on sys.path (FOLDERS) and imports them, for run.py and inventory.py. Each module also lists the
inventory entries of its equations (INVENTORY, of Entry, and ALSO; see inventory.py).
"""
import functools
import importlib
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
import sympy as sp

import latex

# in the order of the checks (run.py) and of the inventory (inventory.py)
MODULES = ["annihilation", "triplet", "neutron_decay", "kaon_production", "kaon_decays", "charm_production",
           "charm_decays"]
# the folders of tests/literature_checks/ that hold them (tests/support/paths.py lists them too)
FOLDERS = ("common",) + latex.PROCESSES
CHECKS = {}


def load_modules(names=MODULES):
    """Imports the derivation modules `names` (of MODULES), with FOLDERS put on sys.path after the running script's
    own, and returns them; their checks are then registered in CHECKS."""
    unknown = [n for n in names if n not in MODULES]
    if unknown:
        sys.exit(f"no derivation module {', '.join(unknown)}; the modules: {' '.join(MODULES)}")
    base = Path(__file__).resolve().parents[1]   # tests/literature_checks/
    for folder in reversed(FOLDERS):
        if str(base / folder) not in sys.path:
            sys.path.insert(1, str(base / folder))
    return [importlib.import_module(n) for n in names]


def derivation(entry_id):
    def register(f):
        if entry_id in CHECKS:
            raise ValueError(f"two derivation checks for {entry_id}")
        CHECKS[entry_id] = f
        return f
    return register


@dataclass
class Result:
    what: str
    ok: bool
    detail: str


@dataclass
class Entry:
    """An equation in the inventory (inventory.py), listed in INVENTORY by the module that holds its derivation check:
      pin       a piece of its LaTeX, copied from its page, where it must occur exactly once
      relation  how it relates to its source (inventory.RELATIONS)
      source    the source it is cited from
      paper     the local copy of the source in knowledge/ (git-ignored), e.g. "TeX " + latex.HUEMMER or SVENSSON below
      checks    the other tests and scripts that check it
      note      a remark (not printed)
      file      the page holding its pins (a key of latex.FILES)
      also      the pins of the other relations of its display equation (from the module's ALSO, by inventory.py)"""
    id: str
    pin: str
    relation: str
    source: str
    paper: str = ""
    checks: str = ""
    note: str = ""
    file: str = "README"
    also: tuple = ()


# Local copies of the sources (git-ignored) for the entries, besides the arXiv TeX sources ("TeX " + latex.HUEMMER, ...)
SVENSSON = "scan ApJ-258-321_Svensson-1982.pdf"
PDG = "PDF PDG-2026_listing_K-plus-minus.pdf, _K-zero-L.pdf"
PDG_N = "PDF PDG-2026_listing_n.pdf"
PDG_C = "PDF PDG-2026_listing_D-plus-minus.pdf, _D-zero.pdf, _Ds-plus-minus.pdf, _lambdac-plus.pdf"
RB = "PostScript RMP-67-893_Richman-Burchat-1995_UCSB-HEP-95-08.ps"


def zero_numerator(expr):
    """The numerator of expr, expanded, after taking square roots of perfect squares exactly and splitting every
    logarithm into logarithms of its argument's irreducible factors, as independent symbols. If it is 0, expr is
    identically zero.

    For roots and logarithms of Lorentz factors, first substitute a rational parametrisation that makes the roots
    perfect squares, e.g. gamma = (v^2 + v^-2)/2 (then p = (v^2 - v^-2)/2 and ln(gamma + p) = 2 ln v) with
    v = 1 + w, w a positive symbol: every factor sympy takes out of a root is then known to be positive."""
    def root(p):
        return sp.powdenest(sp.sqrt(sp.factor(p.base)), force=True) ** (2 * p.exp)
    e = expr.replace(lambda a: isinstance(a, sp.acosh), lambda a: sp.log(a.args[0] + sp.sqrt(a.args[0]**2 - 1)))
    e = e.replace(lambda a: isinstance(a, sp.asinh), lambda a: sp.log(a.args[0] + sp.sqrt(a.args[0]**2 + 1)))
    e = e.replace(lambda a: isinstance(a, sp.atanh), lambda a: sp.log((1 + a.args[0]) / (1 - a.args[0])) / 2)
    for _ in range(3):
        e = e.replace(lambda p: p.is_Pow and p.exp.is_Rational and p.exp.q == 2, root)
    e = e.replace(lambda a: isinstance(a, sp.log),
                  lambda a: sp.expand_log(sp.log(sp.factor(a.args[0])), force=True))
    logs = sorted(e.atoms(sp.log), key=str)
    e = e.xreplace({lg: sp.Symbol(f"_log{i}") for i, lg in enumerate(logs)})
    return sp.expand(sp.numer(sp.together(e)))


def exact(what, difference):
    """The difference is identically zero (sympy, see zero_numerator)."""
    n = zero_numerator(difference)
    return Result(what, n == 0, "exact: 0" if n == 0 else f"exact: not zero ({str(n)[:100]})")


def exact_fails(what, difference):
    """The difference must NOT be zero: a control, or a printed form that is wrong."""
    n = zero_numerator(difference)
    return Result(what, n != 0, "exact: not zero, as it must be" if n != 0 else "exact: 0, but must differ")


def in_source(what, name, *snippets):
    """The snippets (the paper side of a check, copied from its TeX source) occur in that source in knowledge/papers/.
    Without the local copies (git-ignored), it passes as "not re-checked"."""
    if not re.fullmatch(r"[\w.+-]+", name):   # e.g. a pin passed for the source: it would never be found
        return Result(what, False, f"not a file name: {name}")
    found = latex.in_paper(name, *snippets)
    if found is None:
        return Result(what, True, f"not re-checked: {name} is not in knowledge/papers/")
    return Result(what, found, f"in the TeX source {name}" if found else f"not found in {name}")


def max_rel(f, g, points, dps=50):
    """Largest |f - g| / max(|f|, |g|, tiny) over the points (tuples of arguments), at dps digits."""
    worst = mp.mpf(0)
    with mp.workdps(dps):
        for args in points:
            a, b = f(*args), g(*args)
            scale = max(abs(a), abs(b), mp.mpf(10) ** (-dps))
            worst = max(worst, abs(a - b) / scale)
    return worst


def numeric(what, f, g, points, dps=50, tol=None):
    """f and g agree at the points to about dps digits (default tolerance 10^(10 - dps))."""
    tol = mp.mpf(10) ** (10 - dps) if tol is None else tol
    worst = max_rel(f, g, points, dps)
    return Result(what, worst <= tol, f"numeric: max rel. difference {mp.nstr(worst, 3)} at {len(points)} points, "
                                      f"{dps} digits")


def control(what, f, g_broken, points, dps=50):
    """The broken variant must NOT agree (a test that cannot fail checks nothing)."""
    tol = mp.mpf(10) ** (10 - dps)
    worst = max_rel(f, g_broken, points, dps)
    return Result(what, worst > tol, f"control: broken variant differs by {mp.nstr(worst, 3)} (must fail)")


def must_fail(what, results):
    """A control: the results of a check run on a broken README must include a failure."""
    bad = [r.what for r in results if not r.ok]
    return Result(what, bool(bad), f"control: fails ({bad[0]})" if bad else "control: passes, but must fail")


def mdot(a, b):
    """The Minkowski product (+ - - -) of two 4-vectors."""
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


@functools.cache
def dirac():
    """The gamma matrices (Dirac representation), gamma_5 = i g0 g1 g2 g3 and the unit matrix (sympy)."""
    I2, Z2 = sp.eye(2), sp.zeros(2)
    pauli = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
    gam = [sp.Matrix(sp.BlockMatrix([[I2, Z2], [Z2, -I2]]))] + [sp.Matrix(sp.BlockMatrix([[Z2, s_], [-s_, Z2]]))
                                                                for s_ in pauli]
    return gam, sp.I * gam[0] * gam[1] * gam[2] * gam[3], sp.eye(4)


def slash(a):
    """a-slash = gamma^mu a_mu of a 4-vector a."""
    gam = dirac()[0]
    return a[0] * gam[0] - a[1] * gam[1] - a[2] * gam[2] - a[3] * gam[3]


def log_uniform(rng, lo, hi):
    """A random mpf, uniform in ln between lo and hi."""
    return mp.mpf(lo) * (mp.mpf(hi) / lo) ** mp.mpf(rng.random())


def rng(seed=1):
    return random.Random(seed)
