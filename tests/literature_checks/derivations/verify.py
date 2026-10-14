"""The registry of derivation checks and the equivalence tests they use.

A derivation check is a function decorated with @derivation("<inventory id>"). It returns a list of
Result lines and passes when all of them pass. The tests, strongest first:
  exact      the difference simplifies to zero (sympy), often after a substitution, chosen by hand
             per equation, that makes it rational (see zero_numerator);
  numeric    the two sides agree at random points to `dps` digits (mpmath), in every branch;
  control    a deliberately broken variant of the README side must fail the same test, so that a
             test that cannot fail is noticed.
"""
import random
import re
from dataclasses import dataclass

import mpmath as mp
import sympy as sp

MODULES = ["annihilation", "triplet", "neutron_decay", "kaon_production", "kaon_evolution", "kaon_decays", "am3_cooling",
           "am3_kernels"]   # the modules with derivation checks, imported by inventory.py and run.py
CHECKS = {}


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


def zero_numerator(expr):
    """The numerator of expr as a polynomial, after taking square roots of perfect squares exactly and splitting
    every logarithm into the logarithms of the irreducible factors of its argument, treated as independent symbols.
    If it expands to 0, expr is identically zero.

    For square roots and logarithms of Lorentz factors, first substitute a rational parametrisation that makes the
    roots perfect squares, e.g. gamma = (v^2 + v^-2)/2 (then p = (v^2 - v^-2)/2 and ln(gamma + p) = 2 ln v), with
    v = 1 + w and w a positive symbol: every factor sympy takes out of a root is then known to be positive."""
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
    """The snippets (the paper side of a check, copied from the paper's TeX source) occur in that source among the
    local copies of the papers. Without the copies (git-ignored), the check says so."""
    import latex
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


def control(what, f, g_broken, points, dps=50, tol=None):
    """The broken variant must NOT agree (a test that cannot fail checks nothing)."""
    tol = mp.mpf(10) ** (10 - dps) if tol is None else tol
    worst = max_rel(f, g_broken, points, dps)
    return Result(what, worst > tol, f"control: broken variant differs by {mp.nstr(worst, 3)} (must fail)")


def must_fail(what, results):
    """A control: the results of a check run on a broken README must include a failure."""
    bad = [r.what for r in results if not r.ok]
    return Result(what, bool(bad), f"control: fails ({bad[0]})" if bad else "control: passes, but must fail")


def mdot(a, b):
    """The Minkowski product (+ - - -) of two 4-vectors."""
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def log_uniform(rng, lo, hi):
    """A random mpf, uniform in ln between lo and hi."""
    return mp.mpf(lo) * (mp.mpf(hi) / lo) ** mp.mpf(rng.random())


def rng(seed=1):
    return random.Random(seed)


def run():
    """Runs the registered checks, prints their results and exits non-zero if one fails."""
    import sys
    import time
    failed = 0
    for entry_id, f in CHECKS.items():
        t0 = time.time()
        results = f()
        print(f"{entry_id}: {f.__doc__.strip().splitlines()[0]} ({time.time() - t0:.1f} s)")
        for r in results:
            print(f"  {'ok  ' if r.ok else 'FAIL'} {r.what}: {r.detail}")
            failed += not r.ok
    print(f"\n{len(CHECKS)} checks, {failed} failed results")
    sys.exit(1 if failed else 0)
