"""Regression tests of the bugs am3_mojo fixes: AM3's and the port's own.

AM3's bugs are fixed in the C++ reference as well (tests/reference/am3_fixes.patch), apart from those of the interface
and the OpenMP race. The tests cover every fix of README "Differences from the C++ original"
(tests/reference/README.md, "Bugs of AM3 v1.2.0 that am3_mojo fixes") but the OpenMP race, which
tests/unit/test_threads.py covers. Each test fails without its fix. The tests of reads and writes past the end of an
array fail at least in the build with bounds checks (make test-asserts), as their effect depends on the memory there.
The exception is the hadrons' inverse-Compton cooling, read through an unchecked pointer (see that test).
"""
import os, subprocess, sys
from fractions import Fraction as Fr
import numpy as np
import pytest

from support import setups
from support.output import quiet
from support.scenarios import leptohadronic_params, sphere_volume
from support.setups import ADDED_LEPTONIC, CLOSED_BOX, LEPTONS_ALONE
from support.units import C_CGS, DLNE, MASS_EV, MASS_GEV, MEC2_AM3_EV, MP_C2_EV

UNIT = os.path.dirname(os.path.abspath(__file__))


def new(off=(), **switches):
    """A fresh, quiet object with the processes `off` switched off and the switches set; kernels not initialised."""
    return setups.new(off=off, init=False, **switches)


def rich(a, R=3e14, dtf=1e-2):
    """A dense hadronic source in a strong field (all hadronic processes
    matter): the leptohadronic scenario's parameters in 100 G; returns the time step."""
    return leptohadronic_params(a, R, 100.0, dtf)


def evolve(a, n):
    for _ in range(n):
        a.evolve_step()
    return a


# ---------------------------------------------------------------- the solver
def const_cool(t_cool_steps, below=None):
    """Protons with an energy-independent extra cooling time (in time steps) and escape in R/c = 100 steps: their
    energy content after 30 steps. With `below` (s), that cooling time applies only above 1e14 eV, `below` under it,
    and the result is the protons at 1.6e14 eV."""
    a = new(photopion=False, bethe_heitler=False, pp=False)
    quiet(a.init_kernels)
    R = 1e15
    dt = 1e-2 * R / C_CGS
    a.set_mag_field(1e-3)
    a.set_escape_timescale(R / C_CGS)
    a.set_solver_time_step(dt)
    a.set_powerlaw_injection_parameters_protons(sphere_volume(R), 1e44, 1e1 * MP_C2_EV,
                                                1e1 * MP_C2_EV, 1e8 * MP_C2_EV, 2.0, 2.0, 1.0)
    E = a.get_egrid_had()
    if t_cool_steps is not None:
        t = np.full(E.size, t_cool_steps * dt)
        a.set_t_proton_extra_cooling(t if below is None else np.where(E > 1e14, t, below))
    evolve(a, 30)
    if below is not None:
        return a.get_protons()[np.argmin(np.abs(E - 1.6e14))]
    return np.sum(a.get_protons() * E)


def test_fast_cooling_above_slow_cooling():
    """A bin's solution in a fast-cooling window involves only the bins above it, so it must not depend on how
    slowly the bins below the window cool. AM3 accumulated the cooling time from the lowest of the window's 10
    padding cells: bins of other regimes or, at the bottom of the grid, cells below it with A = 0 (bin 0:
    A = -1e-150). Where those cool about 1e15 times more slowly, this sum swamps the window's cooling times in
    double precision: the window's cooling is distorted (densities 28 % too low at 1e16), and from 1e17 on lost.
    This hits cooling faster than an e-fold per step that is not synchrotron-like. Here, 1e20 s in the lowest cell
    left the protons uncooled: 1.16 times the density without cooling, instead of 0.011."""
    n0 = const_cool(None, below=1e20)
    n15, n20 = const_cool(0.3, below=1e15), const_cool(0.3, below=1e20)
    assert n20 == pytest.approx(n15, rel=1e-9)
    assert n20 < 0.05 * n0


def electrons_box(dt, t_esc, B=0.0, t_cool=None, rk4=False):
    """Electrons injected with E^-1 per ln E from 1e8 to 1e13 eV that cool by
    synchrotron radiation in the field B, or with the energy-independent extra
    cooling time t_cool, and escape in t_esc; nothing else. Returns the object,
    the grid and the injection."""
    a = new(LEPTONS_ALONE + ADDED_LEPTONIC, electron_syn=t_cool is None)
    a.set_solver_time_step(dt)
    a.set_solver_simpleqintegration(rk4)
    quiet(a.init_kernels)
    a.set_escape_timescale(t_esc)
    a.set_mag_field(B)
    E = a.get_egrid_lep()
    if t_cool is not None:
        a.set_t_pair_extra_cooling(np.full(E.size, t_cool))
    q = np.where((E >= 1e8) & (E <= 1e13), (E / 1e8) ** -1.0, 0.0)
    a.set_injection_rate_electrons(q)
    return a, E, q


def source_integral(E, q, t_cool, t_esc, reach=np.inf, sub=50):
    """t_cool(x) int_x^min(x + reach, top) q(x') exp(-int_x^x' t_cool / t_esc dx'') dx' (x = ln E): the electrons
    of the source q that cool (e-folding time t_cool) and escape, from x' at most `reach` e-folds above x. As in the
    solver, q and ln t_cool are linear between grid points, and the integral is exact on sub-intervals."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        return _source_integral(E, q, t_cool, t_esc, reach, sub)


def _source_integral(E, q, t_cool, t_esc, reach, sub):
    x = np.log(E)
    xs = np.linspace(x[0], x[-1], (len(x) - 1) * sub + 1)
    lk = np.interp(xs, x, np.log(t_cool / t_esc))
    qs = np.interp(xs, x, q)
    h, d = np.diff(xs), np.diff(lk)
    D = h * np.exp(lk[:-1]) * np.where(np.abs(d) > 1e-12, np.expm1(d) / np.where(d == 0, 1, d), 1.0)
    Ds = np.maximum(D, 1e-300)
    m0 = np.where(D > 1e-12, -np.expm1(-D) / Ds, 1.0 - D / 2)
    m1 = np.where(D > 1e-3, (1.0 - (1.0 + D) * np.exp(-D)) / Ds ** 2, 0.5 - D / 3)
    f = h * (qs[:-1] * m0 + (qs[1:] - qs[:-1]) * m1)
    n = np.zeros(len(x))
    for i in range(len(x)):
        j0, j1 = i * sub, min((len(x) - 1) * sub, int(round((x[i] + min(reach, 1e9) - x[0]) / h[0])))
        K = np.concatenate([[0.0], np.cumsum(D[j0:j1])])
        n[i] = t_cool[i] * np.sum(f[j0:j1] * np.exp(-K[:-1]))
    return n


def test_source_integral_where_escape_dominates():
    """The semi-analytic regime 3 at a time step much longer than the escape time: its steady state (synchrotron
    cooling and escape) must be that of the continuous equation. AM3's source integral (simple_integral_q) summed
    q(x') exp(-int_x^(x' + dx) a/|A|) dx over the grid points, attenuating each by the cell above it. The densities
    were too low by z/(e^z - 1), z = dx a/|A| (0.08 at 1 GeV, 7e-4 just above regime 1, where escape is 100 times
    faster than cooling), and too high where cooling dominates. The test leaves out the top cell of the
    escape-dominated window below (regime 1), as the particles that cool into it escape near its upper edge: it
    holds the right number of particles, but not the density at its centre."""
    t_esc = 1e4
    a, E, q = electrons_box(1e4 * t_esc, t_esc, B=1.0)
    evolve(a, 6)
    t_syn = a.get_t_pair_syn()
    exact = source_integral(E, q, t_syn, t_esc)
    sel = (E >= 3e8) & (E <= 5e12)
    sel[np.nonzero(t_syn > 100 * t_esc)[0].max()] = False
    np.testing.assert_allclose(a.get_electrons()[sel], exact[sel], rtol=1e-2)


def synchrotron_electrons(dt, escape):
    """Electrons injected with 1e9 eV / E per ln E from 1e9 to 1e12 eV that
    cool by synchrotron radiation in 30 G, with or without escape in 1e4 s
    (tests/solver_checks/regime_boundary.py). Returns the object and the
    injection."""
    a = new(LEPTONS_ALONE + ADDED_LEPTONIC, escape=escape)
    a.set_solver_time_step(dt)
    quiet(a.init_kernels)
    a.set_mag_field(30.0)
    a.set_escape_timescale(1e4)
    E = a.get_egrid_lep()
    q = np.where((E >= 1e9) & (E <= 1e12), 1e9 / E, 0.0)
    a.set_injection_rate_electrons(q)
    return a, q


@pytest.mark.parametrize('dt', [1e2, 1e3, 1e4])
def test_particles_cooling_into_a_window_after_a_change(dt):
    """Synchrotron cooling only (no escape), electrons injected from t = 0: after k steps they number q k dt, as
    long as none has cooled to the bottom of the grid. AM3's 10-cell padding of the windows dropped 3 % (dt = 100 s)
    and 25-28 % (1e3 and 1e4 s) within 10-50 steps. The inflow into a Chang & Cooper window must be the electrons
    that cross into it within the step (taking the new densities of the window above instead adds too many in the
    first step)."""
    a, q = synchrotron_electrons(dt, False)
    for k in range(1, 11):
        a.evolve_step()
        assert np.sum(a.get_electrons()) / (np.sum(q) * k * dt) == pytest.approx(1.0, abs=1e-2), k


def electrons_on_a_cooling_step(dtf):
    """Electrons with escape (t_esc = 1e4 s) and an extra cooling time of 0.1 t_esc above 1e9 eV and 1e12 s below
    (an escape-dominated window, regime 1, there), injected per ln E as 1e9/E from 1e10 to 1e12 eV. Returns their
    number after 20 escape times, over Q t_esc (exact: 1)."""
    a = new(LEPTONS_ALONE + ADDED_LEPTONIC + ('electron_syn', 'quantum_syn'), escape=True)
    a.set_solver_time_step(dtf * 1e4)
    quiet(a.init_kernels)
    a.set_escape_timescale(1e4)
    E = a.get_egrid_lep()
    a.set_t_pair_extra_cooling(np.where(E > 1e9, 1e3, 1e12))
    q = np.where((E >= 1e10) & (E <= 1e12), 1e9 / E, 0.0)
    a.set_injection_rate_electrons(q)
    evolve(a, int(round(20 / dtf)))
    return np.sum(a.get_electrons()) / (np.sum(q) * 1e4)


@pytest.mark.parametrize('dtf', [1e-2, 0.1, 1.0])
def test_particles_cooling_into_an_escape_dominated_window(dtf):
    """Electrons that cool down to where their cooling stops pile up there and escape: in the steady state their
    number is Q t_esc. (AM3 gave the escape-dominated windows, regime 1, no inflow, so it dropped the particles that
    cooled into one from the window above: 72 % here. am3_mojo keeps them to 0.4 %.)"""
    assert electrons_on_a_cooling_step(dtf) == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize('dt', [30.0, 100.0, 300.0])
def test_particles_cooling_fast_into_a_slow_window(dt):
    """Electrons cooling in 10 s per e-fold (a window of regime 4) into a slow stretch (3000 s, a Chang & Cooper
    window) and then into none (regime 1), with escape in 1e4 s: in the steady state their number is Q t_esc. (The
    inflow into the Chang & Cooper window, counted along the characteristics, must not take the densities of the
    half cell above its edge from both cells: that counts the particles piled up in its top cell again, +12 to
    +53 % here and a growth to 1e89 at dt = 30 s.)"""
    a = new(LEPTONS_ALONE + ADDED_LEPTONIC + ('electron_syn', 'quantum_syn'), escape=True)
    a.set_solver_time_step(dt)
    quiet(a.init_kernels)
    a.set_escape_timescale(1e4)
    E = a.get_egrid_lep()
    a.set_t_pair_extra_cooling(np.where(E > 1e9, 10.0, np.where(E > 1e9 * np.exp(-1.2), 3000.0, 1e12)))
    q = np.where((E >= 1e10) & (E <= 1e12), 1e9 / E, 0.0)
    a.set_injection_rate_electrons(q)
    evolve(a, int(max(60, 3e5 / dt)))
    assert np.sum(a.get_electrons()) / (np.sum(q) * 1e4) == pytest.approx(1.0, abs=1.5e-2)


def test_steep_injection_tail_in_regime_4():
    """Synchrotron cooling only (regime 4) with an injection that falls by exp(-E / 1 GeV). Above the energy the
    cooling reaches within the step, one step from empty gives the steady state n |A| = int_x^top q dx. The test
    checks the ratio of the two sides against its value at 3 GeV, which removes the units of the cooling-time
    getter. AM3 accumulated int q dx from the bottom of the window and restarted it twice, where q dx / sum < 1e-15.
    The bins below each restart kept almost no significant digits of their injection, and those from 1.0e11 eV
    (q / q_max < 1e-47) on kept none (n = 0)."""
    a = new(LEPTONS_ALONE + ADDED_LEPTONIC, escape=False)
    a.set_solver_time_step(1e5)
    quiet(a.init_kernels)
    a.set_mag_field(30.0)
    E = a.get_egrid_lep()
    q = np.where(E >= 1e8, 1e8 / E * np.exp(-E / 1e9), 0.0)
    a.set_injection_rate_electrons(q)
    a.evolve_step()
    flux = a.get_electrons() / a.get_t_pair_syn()
    above = np.concatenate([np.cumsum((0.5 * (q[:-1] + q[1:]) * DLNE)[::-1])[::-1], [0.0]])
    sel = (E >= 3e9) & (E <= 1.5e11)
    ratio = flux[sel] / above[sel]
    np.testing.assert_allclose(ratio / ratio[0], 1.0, rtol=1e-8)


@pytest.mark.parametrize('t_cool, k, rk4', [(1e3, 5.0, False), (1e3, 50.0, False), (1e3, 0.05, False),
                                            (1e3, 0.05, True), (1e-5, None, False)],
                         ids=['escape-5x', 'escape-50x', 'escape-0.05x', 'escape-0.05x-rk4', 'escape-vanishing'])
def test_source_integral_of_one_step(t_cool, k, rk4):
    """One step of dt = 3 t_cool from empty, with an energy-independent cooling time t_cool: every bin holds
    the electrons injected at most 3 e-folds above it.
    * Escape in t_cool / k (regime 3): n/exact was 0.81 and 0.035 for k = 5 and 50. Where the sink is weak, the
      RK4 option (set_solver_simpleqintegration) took the electrons from up to the top of the solver's window,
      beyond those 3 e-folds (+4.4 % for k = 0.05).
    * t_cool = 10 us, escape in t_cool below 1e11 eV and none above: most of the electrons in the top 3 e-folds
      below 1e11 eV come from above it. The source integral of regime 3 (simple_integral_q) skipped the points
      where alpha/|A| < e^-300 with their injection: 46-90 % too few electrons in the 6 bins below 1e11 eV. Both
      codes add 1e-150 to alpha in the solver's units (time in units of 2.7e-23 s for the e+-), so without escape
      this happens where the e+- cool in less than about 1.4 ms. The case also needs the padding of the
      semi-analytic windows up to where particles reach them within the step (AM3's 10 cells: up to 34 % too few
      electrons below 1e11 eV)."""
    a, E, q = electrons_box(3 * t_cool, t_cool / k if k else t_cool, t_cool=t_cool, rk4=rk4)
    t_esc = t_cool / k if k else np.where(E < 1e11, t_cool, 1e250)
    if not k:
        a.set_t_pair_escape(t_esc)
    a.evolve_step()
    exact = source_integral(E, q, np.full(E.size, t_cool), t_esc, reach=3.0)
    sel = (E >= 2e8) & (E <= 5e12)
    np.testing.assert_allclose(a.get_electrons()[sel], exact[sel], rtol=5e-3)


def test_source_integral_of_a_steeply_rising_source():
    """One step of dt = 3 t_cool from empty (an energy-independent cooling time t_cool and escape in 20 t_cool:
    regime 3) with a source rising as E^2 per ln E from 1e7 to 1e16 eV. A source integral of regime 3 taken as
    the difference of two integrals up to the top of the window loses its digits where the electrons injected
    further up dominate: up to 45 % too many electrons at 1e7 eV."""
    t_cool = 1e3
    a, E, _ = electrons_box(3 * t_cool, 20 * t_cool, t_cool=t_cool)
    q = np.where((E >= 1e7) & (E <= 1e16), (E / 1e16) ** 2.0, 0.0)
    a.set_injection_rate_electrons(q)
    a.evolve_step()
    exact = source_integral(E, q, np.full(E.size, t_cool), 20 * t_cool, reach=3.0)
    sel = (E >= 1e7) & (E <= 3e14)
    np.testing.assert_allclose(a.get_electrons()[sel], exact[sel], rtol=5e-3)


@pytest.mark.parametrize('t_cool', [0.3, 0.5, 0.9])
def test_cooled_particles_in_regime_4(t_cool):
    """Protons with an energy-independent cooling time t_cool of 0.3 to 0.9 time steps and escape in 100 steps
    (regime 4). In the steady state their energy content is L t, 1/t = 1/t_cool + 1/t_esc. A bin x takes the
    particles n(y) A(y)/A(x) that cooling brings from y to x within the step. AM3 subtracted them instead: 6 %,
    23 % and 49 % too little energy for t_cool = 0.3, 0.5 and 0.9 steps."""
    U0 = const_cool(None)
    L_tesc = U0 / (1 - np.exp(-0.3))
    t = 1 / (1 / t_cool + 1 / 100.0) / 100.0  # in units of t_esc
    assert const_cool(t_cool) / (L_tesc * t) == pytest.approx(1.0, rel=0.01)


def rk4_one_step_across_an_escape_step():
    """The largest |n/exact - 1| of one step of dt = 3 t_cool from empty with the RK4 option, an
    energy-independent cooling time t_cool and escape in 10 t_cool below 1e10 eV and t_cool / 100 above."""
    t_cool, Es = 1e3, 1e10
    a, E, q = electrons_box(3 * t_cool, 10 * t_cool, t_cool=t_cool, rk4=True)
    t_esc = np.where(E < Es, 10 * t_cool, t_cool / 100)
    a.set_t_pair_escape(t_esc)
    a.evolve_step()
    exact = source_integral(E, q, np.full(E.size, t_cool), t_esc, reach=3.0)
    sel = (E >= 2e8) & (E <= 5e12)
    return float(np.max(np.abs(a.get_electrons()[sel] / exact[sel] - 1)))


def test_rk4_source_integral_across_an_escape_step():
    """The RK4 option (set_solver_simpleqintegration) with escape 1000 times faster above 1e10 eV than below. For
    the bins up to 0.3 e-folds below that energy, the first tenth of the integration range reaches across it, where
    the sink takes more than e^2 of the particles. The integration then narrows its range and widens it again. The
    widening must update the integration step: AM3's loop did not, and never ended. With the update, every bin is
    within 2 % of the exact integral. The case runs in a separate process, so that an endless loop fails this test
    instead of stopping the test run."""
    code = (f'import sys; sys.path[:0] = [{os.path.dirname(UNIT)!r}, {UNIT!r}]\n'
            'from test_fixes import rk4_one_step_across_an_escape_step as f\nprint(f())')
    try:
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail('the RK4 source integral did not end within 120 s')
    assert r.returncode == 0, r.stderr[-3000:]
    assert float(r.stdout.split()[-1]) < 0.02


# the species and their masses [GeV] in the code (AM3's)
SYN_SPECIES = [(s, MASS_GEV[m]) for s, m in (('pair', 'masse'), ('proton', 'massp'), ('pion', 'masspip'), ('muon', 'massmu'))]


@pytest.fixture(scope='module')
def synchrotron_cooling_times():
    """The field, the synchrotron cooling times of every species without and
    with the quantum suppression ({False: {species: t}, True: ...}) and the
    lepton and hadron grids."""
    B = 1e5
    t = {}
    for q in (False, True):
        a = new(quantum_syn_cooling=q)
        quiet(a.init_kernels)
        a.set_mag_field(B)
        a.evolve_step()  # the solvers compute their cooling rates in a step
        t[q] = {species: getattr(a, f'get_t_{species}_syn')() for species, _ in SYN_SPECIES}
    return B, t, a.get_egrid_lep(), a.get_egrid_had()


@pytest.mark.parametrize('species, mass', SYN_SPECIES)
def test_quantum_synchrotron_cooling_of_heavy_particles(synchrotron_cooling_times, species, mass):
    """AM3's quantum suppression of synchrotron cooling, 1 / (1 + sqrt(2) x^(2/3))^2 (an interpolation of Landau &
    Lifshitz), takes the quantum parameter x = chi = gamma B / B_cr(m), with B_cr ∝ m^2. So x is
    (m_e / m)^2 gamma B / B_cr(m_e) for a particle of mass m. AM3 used gamma B / m, m / m_e times that, which
    overstates the suppression of heavy particles."""
    B, times, egrid_lep, egrid_had = synchrotron_cooling_times
    t = {q: times[q][species] for q in (False, True)}
    E = egrid_lep if species == 'pair' else egrid_had
    gamma = E / (mass * 1e9)
    x = 1.2e-8 * B * gamma / (mass * 1e9) * (MASS_GEV['masse'] / mass)
    fin = t[False] < 1e199
    assert fin.sum() > 250
    np.testing.assert_allclose(t[True][fin] / t[False][fin], (1 + np.sqrt(2) * x[fin] ** (2 / 3)) ** 2,
                               rtol=2e-5)


def jones_series(n=18):
    """Taylor coefficients (to z^n) of the inverse-Compton cooling functions
    of Jones (1965, eqs. 14-15),
      f1 = (z + 6 + 3/z) L - (22z^3/3 + 24z^2 + 18z + 4) / (1 + 2z)^2 - 2 + 2 Li2(-2z),
      f2 = (z + 31/6 + 5/z + 3/(2z^2)) L
           - (22z^3/3 + 28z^2 + 103z/3 + 17 + 3/z) / (1 + 2z)^2 - 2 + Li2(-2z),
    from L = ln(1 + 2z) = sum (-1)^(k+1) (2z)^k / k, (1 + 2z)^-2 =
    sum (k + 1) (-2z)^k and Li2(-2z) = sum (-2z)^k / k^2, in exact fractions."""
    N = n + 3
    L = {k: Fr((-1) ** (k + 1) * 2 ** k, k) for k in range(1, N)}
    inv2 = {k: Fr((k + 1) * (-2) ** k) for k in range(N)}
    li2 = {k: Fr((-2) ** k, k * k) for k in range(1, N)}

    def mul(a, b):
        c = {}
        for i, x in a.items():
            for j, y in b.items():
                c[i + j] = c.get(i + j, 0) + x * y
        return c

    def lin(*terms):
        c = {}
        for s, t in terms:
            for k, v in t.items():
                c[k] = c.get(k, 0) + s * v
        return c

    f1 = lin((1, mul({1: 1, 0: 6, -1: 3}, L)), (-1, mul({3: Fr(22, 3), 2: 24, 1: 18, 0: 4}, inv2)),
             (1, {0: -2}), (2, li2))
    f2 = lin((1, mul({1: 1, 0: Fr(31, 6), -1: 5, -2: Fr(3, 2)}, L)),
             (-1, mul({3: Fr(22, 3), 2: 28, 1: Fr(103, 3), 0: 17, -1: 3}, inv2)), (1, {0: -2}), (1, li2))
    return [np.array([float(f.get(k, 0)) for k in range(n + 1)]) for f in (f1, f2)]


def test_inverse_compton_cooling_functions_below_z_of_one_tenth():
    """AM3 evaluates the cooling functions f1, f2 of the inverse-Compton loss,
    F(eps, gamma) = gamma [f1(eps g) - f1(eps / g)] - eps [f2(eps g) - f2(eps / g)], g = gamma + sqrt(gamma^2 - 1),
    by their Taylor series below z = 0.1. It kept only the first three terms of Jones' series, which are 2.4 % (f1)
    and 1.2 % (f2) off at z = 0.1, so the cooling time jumped by 2.4 % where eps g crosses 0.1. With the series to
    z^10 (Jones 1965, Table I), F follows the exact functions to 2e-5. Electrons in photons of one energy eps have
    the cooling time t ∝ gamma^3 / F(eps, gamma)."""
    a = new()
    quiet(a.init_kernels)
    Eg = a.get_egrid_photons()
    j = np.argmin(np.abs(Eg - 25.0))
    a.set_current_densities_photons(np.where(np.arange(len(Eg)) == j, 1e10, 0.0))
    a.evolve_step()  # the solvers compute their cooling rates in a step
    t = a.get_t_pair_compton()
    gamma = a.get_egrid_lep() / MEC2_AM3_EV  # (the code's m_e c^2)
    eps = Eg[j] / MEC2_AM3_EV
    g = gamma + np.sqrt(np.maximum(gamma * gamma - 1, 0.0))
    f1, f2 = (lambda z, c=c: np.polyval(c[::-1], z) for c in jones_series())
    z = eps * g
    sel = (z > 1e-3) & (z < 0.1)
    assert sel.sum() >= 30 and (z[sel] > 0.09).any()
    r = t * (gamma * (f1(z) - f1(eps / g)) - eps * (f2(z) - f2(eps / g))) / gamma ** 3
    r = r[sel] / np.median(r[sel & (z < 1e-2)])
    np.testing.assert_allclose(r, 1.0, rtol=5e-5)


# ---------------------------------------------------------------- switches
def test_kaons_switched_off_after_init():
    """Kaons switched off after init_kernels are off: no kaons, and AM3's pion
    multiplicities are no longer scaled for the kaon channels (they were, by
    up to 10 %, when the scale was set only at init)."""
    a = new()
    quiet(a.init_kernels)
    a.set_process_kaons(False)
    b = new(kaons=False)
    quiet(b.init_kernels)
    for x in (a, b):
        rich(x)
        evolve(x, 10)
    assert not a.get_kaons().any()
    for g in ('get_pion_minus_photopion', 'get_photons_pi0_decay_photopion', 'get_photons'):
        np.testing.assert_allclose(getattr(a, g)(), getattr(b, g)(), rtol=1e-12, atol=0, err_msg=g)


def test_parse_sed_does_not_change_the_photons():
    """parse_sed only splits the photons into components: the total must not
    change (with parse_sed off, AM3 dropped the photons of pi0 from pp)."""
    out = []
    for parse in (True, False):
        a = new(parse_sed=parse, pair_annihilation=False, kaons=False)
        quiet(a.init_kernels)
        rich(a)
        out.append(evolve(a, 20).get_photons())
    np.testing.assert_allclose(out[1], out[0], rtol=1e-9, atol=0)


MUONS = ['mu_minus_left', 'mu_plus_left', 'mu_minus_right', 'mu_plus_right']
NEUTRINOS = ['electron_neutrinos', 'electron_anti_neutrinos', 'muon_neutrinos', 'muon_anti_neutrinos']


@pytest.mark.parametrize('species', MUONS + NEUTRINOS)
def test_external_injection_of_muons_and_neutrinos(species):
    """An external injection of one muon or neutrino species makes that species only. (AM3 evolved the
    injection of mu+_L as mu-_R and that of mu-_R as mu+_L, added that of nu_e to all four neutrino
    species and ignored the other three.)"""
    group = MUONS if species in MUONS else NEUTRINOS
    a = new(photopion=False, bethe_heitler=False, pp=False, muon_decay=False)
    quiet(a.init_kernels)
    a.set_mag_field(1e-3)
    n = (a.get_egrid_had() if species in MUONS else a.get_egrid_neutrinos()).size
    q = np.zeros(n)
    q[n // 2] = 1.0
    getattr(a, 'set_injection_rate_' + species)(q)
    quiet(a.evolve_step)
    for other in group:
        got = getattr(a, 'get_' + other)()
        assert (got.max() > 0) if other == species else not got.any(), other


def test_parse_sed_with_injected_muons_and_pi0():
    """parse_sed only splits the outputs into components, also with external injections of muons and
    pi0: the e+- and photons must not change. (With parse_sed on, AM3 took the e+- of muon decays and
    the photons of pi0 decays as the sums of their photo-pion and pp components, which leave out the
    decay products of injected particles.)"""
    out = []
    for parse in (True, False):
        a = new(parse_sed=parse, photopion=False, bethe_heitler=False, pp=False)
        quiet(a.init_kernels)
        R = 3e14
        a.set_mag_field(1.0)
        a.set_escape_timescale(R / C_CGS)
        a.set_solver_time_step(1e-2 * R / C_CGS)
        E = a.get_egrid_had()
        q = np.where((E > 1e10) & (E < 1e15), 1e-3 * (E / 1e10) ** -1.0, 0.0)
        a.set_injection_rate_mu_minus_left(q)
        a.set_injection_rate_pi0(q)
        evolve(a, 20)
        out.append((a.get_pairs(), a.get_photons()))
    for on, off in zip(*out):
        assert off.max() > 0
        np.testing.assert_allclose(on, off, rtol=1e-9, atol=0)


def test_pion_decay_off_mid_run_stops_its_muons():
    """With pion decay switched off during a run, the muon sources from pion decays must stop (AM3 kept injecting
    the last step's muons). Cooling keeps the muon number and decay lowers it, so it then falls by at least the
    escape in every step (dt = t_esc/100), while it grew before."""
    a = new(kaons=False)
    quiet(a.init_kernels)
    rich(a)
    evolve(a, 9)
    N = []
    for step in range(7):
        if step == 2:
            assert N[1] > N[0] > 0
            a.set_process_pion_decay(False)
        a.evolve_step()
        N.append(np.sum(a.get_muons()))
        if step >= 2:
            assert N[-1] <= N[-2] * np.exp(-0.01) * (1 + 1e-9), step


def neutrinos_after_a_switch(switch, getters):
    """The number of the neutrinos of `getters` in each of 6 steps of the rich source without kaons (whose
    decays make neutrinos of every flavour), with set_process_<switch>(False) before the third."""
    a = new(kaons=False)
    quiet(a.init_kernels)
    rich(a)
    evolve(a, 9)
    N = []
    for step in range(6):
        if step == 2:
            getattr(a, 'set_process_' + switch)(False)
        a.evolve_step()
        N.append(sum(np.sum(getattr(a, g)()) for g in getters))
    return N


def test_muon_decay_off_mid_run_stops_its_neutrinos():
    """With muon decay switched off during a run, the electron neutrinos must stop coming (AM3 kept injecting the
    last step's nu_e and anti-nu_e). Their number then falls by the escape in every step,
    exp(-dt/t_esc) = exp(-0.01) within the solver's 2e-7 per step, while it grew before."""
    N = neutrinos_after_a_switch('muon_decay', ('get_electron_neutrinos', 'get_electron_anti_neutrinos'))
    assert N[1] > N[0] > 0
    for step in range(2, 6):
        assert N[step] <= N[step - 1] * np.exp(-0.01) * (1 + 1e-6), step


def test_pion_decay_off_mid_run_with_muon_decay():
    """With pion decay switched off during a run and muon decay on, the muon neutrinos come only from the
    decays of the muons left, which no longer get any sources: their number falls by the escape in every
    step, exp(-dt/t_esc) = exp(-0.01), within 1e-6. (AM3 then never reset the total nu_mu sources and
    added the muon-decay nu_mu of every step to them, so that the last step's sources kept coming.)"""
    N = neutrinos_after_a_switch('pion_decay', ('get_muon_neutrinos', 'get_muon_anti_neutrinos'))
    assert N[1] > N[0] > 0
    for step in range(2, 6):
        assert N[step] <= N[step - 1] * np.exp(-0.01) * (1 + 1e-6), step


def test_kaon_muons_without_pion_decay():
    """Kaon decays make muons (K -> mu nu, ...) whether or not pion decay is
    on."""
    on, off = new(pion_decay=False), new(pion_decay=False, kaons=False)
    for a in (on, off):
        quiet(a.init_kernels)
        rich(a)
        evolve(a, 10)
    assert on.get_muons().max() > 0 and on.get_muon_neutrinos().max() > 0
    assert not off.get_muons().any()


def test_kaon_pairs_without_muon_decay():
    """Kaon decays to e+- (K_e3) feed the pair components of their source
    whether or not muon decay is on."""
    a = new(muon_decay=False)
    quiet(a.init_kernels)
    rich(a)
    evolve(a, 10)
    for g in ('get_pairs_photopion', 'get_pairs_proton_proton'):
        assert getattr(a, g)().max() > 0, g


def test_estimated_electron_emax_with_merged_positrons():
    """estimate_max_energies with positrons merged into electrons must use
    the cooling of the merged population (it read the unevolved electron
    solver and gave the top of the grid)."""
    emax = {}
    for merged in (False, True):
        a = new(merge_positrons_into_electrons=merged)
        a.set_estimate_max_energies(True)
        quiet(a.init_kernels)
        rich(a)
        quiet(evolve, a, 20)
        emax[merged] = a.get_estimated_electron_emax()
    assert emax[False] < 1e15
    assert emax[True] == pytest.approx(emax[False], rel=0.3)


def test_estimated_max_energies_where_acceleration_wins_everywhere():
    """With estimate_max_energies and no cooling of the electrons and protons, acceleration wins on the whole grid.
    The estimated maximum energies are then the energy one bin above the top of the grid, and the injection that
    ends there is finite. (AM3 read the loss rates and the energy grid one element past their end. The injection
    became NaN when that memory read 0.)"""
    a = new(electron_syn=False, electron_compton=False, proton_syn=False, photopion=False,
            bethe_heitler=False, pp=False)
    a.set_estimate_max_energies(True)
    quiet(a.init_kernels)
    rich(a)
    evolve(a, 3)
    El, Eh = a.get_egrid_lep(), a.get_egrid_had()
    # (the estimate has AM3's m_e c^2 = 0.511 MeV; El[0] is m_e c^2 in the getter's units, gamma = 1)
    assert a.get_estimated_electron_emax() == pytest.approx(El[-1] ** 2 / El[-2] / El[0] * MEC2_AM3_EV, rel=1e-9)
    assert a.get_estimated_proton_emax() == pytest.approx(Eh[-1] ** 2 / Eh[-2], rel=1e-9)
    for n in (a.get_electrons(), a.get_protons()):
        assert np.all(np.isfinite(n)) and n.max() > 0


ONLY_PI0_PHOTONS = dict(photopion=False, bethe_heitler=False, pp=False, annihilation=False,
                        ssa=False, electron_syn=False, electron_compton=False, proton_syn=False,
                        pair_annihilation=False, triplet_pair_production=False, kaons=False)


def test_pi0_decay_photons_of_the_top_hadron_bin():
    """The pi0-decay kernel depends only on the ratio of the photon and pion energies. So on the common logarithmic
    grid, the photons of pi0 injected into the top hadron bin are those of pi0 injected ten bins lower, ten photon
    bins up (the solver leaves the two lowest bins and the top bin of a grid empty). (AM3 read the upper edge of the
    top bin, egrid_had[j+1], past the end of the grid: in 8 of 95 fresh processes with photo-pion production, the
    total photon spectrum was zero for the whole run.)"""
    out = []
    for below_top in (0, 10):
        a = new(**ONLY_PI0_PHOTONS)
        quiet(a.init_kernels)
        a.set_escape_timescale(1e4)
        a.set_solver_time_step(1e2)
        q = np.zeros(a.get_egrid_had().size)
        q[q.size - 1 - below_top] = 1.0
        a.set_injection_rate_pi0(q)
        a.evolve_step()
        out.append(a.get_photons())
    top, low = out[0][12:-1], out[1][2:-11]
    assert np.count_nonzero(top) > 500
    np.testing.assert_allclose(top, low, rtol=1e-12, atol=0)


def li2_negative(z):
    """The dilogarithm Li2(z) for z < 0: by Landen's identity from w = z/(z - 1) in (0, 1/2], below -1
    after the inversion Li2(z) = -pi^2/6 - ln^2(-z)/2 - Li2(1/z)."""
    z = np.asarray(z, dtype=float)
    zz = np.where(z < -1, 1 / z, z)
    w = zz / (zz - 1)
    k = np.arange(1, 80)
    li = -np.sum(w[..., None] ** k / k ** 2, axis=-1) - 0.5 * np.log1p(-zz) ** 2
    return np.where(z < -1, -np.pi ** 2 / 6 - 0.5 * np.log(-z) ** 2 - li, li)


def jones_functions(z):
    """The inverse-Compton cooling functions f1, f2 of Jones (1965, eqs. 14-15): their closed forms
    (jones_series) from z = 0.1, their Taylor series to z^18 below."""
    z = np.asarray(z, dtype=float)
    c1, c2 = jones_series()
    x = np.maximum(z, 0.1)
    L, d = np.log1p(2 * x), (1 + 2 * x) ** 2
    f1 = (x + 6 + 3 / x) * L - (22 * x ** 3 / 3 + 24 * x ** 2 + 18 * x + 4) / d - 2 + 2 * li2_negative(-2 * x)
    f2 = ((x + 31 / 6 + 5 / x + 3 / (2 * x * x)) * L - (22 * x ** 3 / 3 + 28 * x ** 2 + 103 * x / 3 + 17 + 3 / x) / d
          - 2 + li2_negative(-2 * x))
    return (np.where(z < 0.1, np.polyval(c1[::-1], z), f1), np.where(z < 0.1, np.polyval(c2[::-1], z), f2))


@pytest.mark.parametrize('species, mass', [('pion', MASS_EV['masspip']), ('muon', MASS_EV['massmu'])])
def test_inverse_compton_cooling_of_hadrons_in_every_bin(species, mass):
    """Pions and muons in photons of one energy eps: in every bin of the hadron grid, their inverse-Compton cooling
    time is t ∝ gamma^3 / F(eps, gamma) within 3e-3 (the photon line is interpolated onto the hadron grid). F is
    that of test_inverse_compton_cooling_functions_below_z_of_one_tenth, with eps in units of the particle's rest
    energy. The check includes the top 49 hadron bins, whose kernels reach past the photon grid. (AM3 read those
    photon bins past the end of the array. In 5 of 45 AM3 objects created after another one in the same process,
    the top 30 bins got cooling times of 1e-187 to 1e-18 s, and negative ones. am3_mojo reads the photons through
    an unchecked pointer, so a regression to that read shows here only if the memory past the array holds
    something other than zeros.)"""
    a = new(photopion=False, bethe_heitler=False, pp=False)
    quiet(a.init_kernels)
    Eg = a.get_egrid_photons()
    j = np.argmin(np.abs(Eg - 25.0))
    a.set_current_densities_photons(np.where(np.arange(Eg.size) == j, 1e10, 0.0))
    a.evolve_step()  # the solvers compute their cooling rates in a step
    t = getattr(a, f'get_t_{species}_compton')()
    gamma = a.get_egrid_had() / mass
    g = gamma + np.sqrt(gamma * gamma - 1)
    eps = Eg[j] / mass
    (a1, a2), (b1, b2) = jones_functions(eps * g), jones_functions(eps / g)
    r = t * (gamma * (a1 - b1) - eps * (a2 - b2)) / gamma ** 3
    assert np.all(np.isfinite(r)) and np.all(r > 0)
    np.testing.assert_allclose(r / np.median(r), 1.0, rtol=3e-3)


def test_no_neutrinos_without_pion_decay():
    """With pion decay off (and no kaons or neutron decay), no neutrinos are made in any bin of the neutrino grid:
    the pions do not decay, so no muons are made either. (AM3 read the nu_mu sources of pion decays, which then had
    the hadron grid's length, on the neutrino grid, past their end.)"""
    a = new(pion_decay=False, kaons=False, neutron_decay=False)
    quiet(a.init_kernels)
    rich(a)
    evolve(a, 5)
    assert a.get_pions().max() > 0
    for g in ('get_neutrinos', 'get_muon_neutrinos', 'get_muon_anti_neutrinos', 'get_electron_neutrinos'):
        assert not getattr(a, g)().any(), g


def test_pp_transition_energy_in_the_top_hadron_bin():
    """The transition energy of the pp delta approximation in the top bin of the hadron grid gives the same pp
    secondaries as one bin lower, as no protons get there. (The index of the transition in the multiplicity table
    was one past its end.)"""
    out = []
    for below_top in (0, 1):
        a = new(photopion=False, bethe_heitler=False)
        E = a.get_egrid_had()
        a.set_pp_transition_energy_to_delta_approx(E[E.size - 1 - below_top])
        quiet(a.init_kernels)
        rich(a)
        evolve(a, 3)
        out.append((a.get_pions(), a.get_photons_pi0_decay_proton_proton()))
    for top, lower in zip(*out):
        assert np.all(np.isfinite(top)) and top.max() > 0
        np.testing.assert_array_equal(top, lower)


# ---------------------------------------------------------------- optimisations
def test_bethe_heitler_minimum_raised_and_lowered():
    """Raising the minimum incoming-proton energy of the Bethe-Heitler pair
    kernel for a step and lowering it again must not lose kernel rows (the
    kernel was packed from the first step's minimum). The first step has no
    target photons, so both runs are the same physics."""
    out = []
    for raise_first in (False, True):
        a = new()
        quiet(a.init_kernels)
        rich(a)
        p0 = a.get_optimize_bethe_heitler_incoming_protons_min() * 1e9  # GeV -> eV
        if raise_first:
            a.set_optimize_bethe_heitler_incoming_protons_min(1e3 * p0)
        a.evolve_step()
        a.set_optimize_bethe_heitler_incoming_protons_min(p0)
        out.append(evolve(a, 5).get_pairs_bethe_heitler())
    assert out[0].max() > 0
    np.testing.assert_array_equal(out[1], out[0])


def test_compton_emission_minimum_after_init():
    """The minimum IC photon energy set after init_kernels applies: no IC
    photons below it (am3_mojo kept the value of init_kernels)."""
    a = new(hadronic=False)
    quiet(a.init_kernels)
    rich(a)
    a.set_optimize_compton_emission_emin(1e3)  # eV
    evolve(a, 5)
    eg = a.get_egrid_photons()
    ic = a.get_photons_injected_electrons_compton()
    assert ic.max() > 0
    assert not ic[eg < 0.9e3].any()


OPTIMISATION_LIMITS = {  # setter: grid whose end limits it
    'photopion_target_photon_max': 'photons', 'bethe_heitler_target_photon_max': 'photons',
    'compton_target_emax': 'photons', 'compton_emission_emin': 'photons',
    'bethe_heitler_incoming_protons_min': 'had'}


def test_optimisation_limits_beyond_the_grids():
    """Optimisation limits beyond the end of their grid (1e30 eV) act as limits at its end: they read back as that
    and give the same spectra. (AM3 turned them into an index past the end of the grid.)"""
    out = []
    for beyond in (True, False):
        a = new()
        for name, grid in OPTIMISATION_LIMITS.items():
            top = getattr(a, f'get_egrid_{grid}')()[-1]
            getattr(a, 'set_optimize_' + name)(1e30 if beyond else top)
        limits = [getattr(a, 'get_optimize_' + name)() for name in OPTIMISATION_LIMITS]
        quiet(a.init_kernels)
        rich(a)
        evolve(a, 3)
        out.append((limits, a.get_photons(), a.get_pairs(), a.get_protons()))
    assert out[0][0] == out[1][0]
    for beyond, top in zip(out[0][1:], out[1][1:]):
        assert np.all(np.isfinite(beyond)) and beyond.max() > 0
        np.testing.assert_array_equal(beyond, top)


TARGET_PHOTON_LIMITS = ['photopion_target_photon_max', 'bethe_heitler_target_photon_max', 'compton_target_emax']


def test_target_photon_limits_lowered_after_init():
    """Target-photon limits lowered after init_kernels act as the same limits set before it: the photons above them
    count as zero (100 eV, where all three change the spectra). The two agree to rounding, as kernels built over
    fewer target photons can group the same terms differently. (AM3 multiplies the kernels of init_kernels by the
    shorter photon vector, past its end. am3_mojo ignored a lowered photo-pion limit and read past the end of its
    Bethe-Heitler photon vector through an unchecked pointer, which the build with bounds checks catches.)"""
    out = []
    for when in ('before', 'after', None):
        a = new()
        for name in TARGET_PHOTON_LIMITS:
            if when == 'before':
                getattr(a, 'set_optimize_' + name)(1e2)  # eV
        quiet(a.init_kernels)
        for name in TARGET_PHOTON_LIMITS:
            if when == 'after':
                getattr(a, 'set_optimize_' + name)(1e2)
        rich(a)
        evolve(a, 5)
        out.append([a.get_photons(), a.get_pairs(), a.get_neutrinos()])
    before, after, unlimited = out
    for x, y in zip(before, after):
        np.testing.assert_allclose(y, x, rtol=1e-13, atol=0)
    assert not all(np.array_equal(x, z) for x, z in zip(before, unlimited))


# ---------------------------------------------------------------- the interface
def test_evolve_needs_init_kernels():
    """evolve_step() raises before init_kernels(), after update_energy_grid() without a new init_kernels()
    (the kernels are for the old grid), and after switching on the hadronic processes that were off at
    init_kernels()."""
    a = new()
    with pytest.raises(Exception, match='init_kernels'):
        a.evolve_step()
    quiet(a.init_kernels)
    a.update_energy_grid(1e-6, 1e2, 1e18)
    with pytest.raises(Exception, match='init_kernels'):
        a.evolve_step()
    quiet(a.init_kernels)
    a.evolve_step()
    assert a.get_protons().size == a.get_egrid_had().size
    b = new(hadronic=False)
    quiet(b.init_kernels)
    b.set_process_hadronic(True)
    with pytest.raises(Exception, match='init_kernels'):
        b.evolve_step()


def test_pp_transition_energy_above_the_hadron_grid():
    """pp collisions need the transition energy of their delta approximation on the hadron grid
    (above its top AM3 crashes, and unchecked the pp pions got no energy at all)."""
    a = new(pp=True)
    a.set_pp_transition_energy_to_delta_approx(1e24)
    with pytest.raises(Exception):
        quiet(a.init_kernels)


@pytest.mark.parametrize('grid', [(-1.0, 1e2, 1e21), (1e6, 1e2, 1e21), (1e-6, -1.0, 1e21), (1e-6, 1e10, 1e21),
                                  (1e-6, 1e2, 1e5), (1e-6, 1e2, float('inf'))])
def test_invalid_energy_grid(grid):
    a = new()
    n = a.get_egrid_had().size
    with pytest.raises(Exception):
        a.update_energy_grid(*grid)
    assert a.get_egrid_had().size == n


def test_larger_hadron_grid_with_pp():
    """A hadron grid larger than the default one, with pp collisions (the pp
    kernel's shifted grid kept the default size, written past its end)."""
    a = new(pp=True)
    a.update_energy_grid(1e-4, 1e2, 1e23)
    assert a.get_egrid_had().size > 314
    quiet(a.init_kernels)
    rich(a)
    evolve(a, 3)
    assert np.all(np.isfinite(a.get_pions())) and a.get_pions().max() > 0


def hadrons_on_photons(e_gamma_min, photopion, e_max=1e21):
    """One step of protons on the grid of update_energy_grid(e_gamma_min, 1e2, e_max) (the default grids
    for e_gamma_min = None): on a photon line at 5 eV with photo-pion production only, or in 1 kG with
    proton synchrotron radiation only."""
    off = ['bethe_heitler', 'pp', 'kaon_decay', 'kaon_syn', 'kaon_compton', 'pion_decay', 'pion_syn',
           'pion_compton', 'proton_compton']
    a = new(photopion=photopion, proton_syn=not photopion, **{name: False for name in off})
    if e_gamma_min is not None:
        a.update_energy_grid(e_gamma_min, 1e2, e_max)
    quiet(a.init_kernels)
    a.set_mag_field(1e3)
    a.set_escape_timescale(1e9)
    a.set_solver_time_step(1e2)
    E, eg = a.get_egrid_had(), a.get_egrid_photons()
    a.set_current_densities_protons(np.where((E > 1e15) & (E < 3e19), 1e-6 * (E / 1e15) ** -0.8, 0.0))
    if photopion:
        a.set_current_densities_photons(1e8 * np.exp(-np.log(eg / 5.0) ** 2 / (2 * 0.3 ** 2)))
    quiet(a.evolve_step)
    return a


def test_particles_in_the_top_bins_of_the_grids():
    """Electrons and photons in the top bins of their grids are evolved like those below (escape only), up to the
    second-highest bin (the solver leaves out the highest). Electrons injected into one of the highest evolved bins,
    cooling by synchrotron radiation in much less than the time step, are q (dx/2) t_syn there: those injected into
    the upper half of the bin's cell. (AM3's solver held the window of empty bins at the top of a grid to its
    minimum length of 6 bins, so it dropped the particles in the 6 highest evolved bins every step. It also set the
    top bin's cooling rate to zero, so the semi-analytic regime evolved the highest evolved bin as n + q dt. With
    the top bins kept, that bin's synchrotron photons fed a gamma-gamma pair cascade there that grew without
    bound.)"""
    a = new(LEPTONS_ALONE + ADDED_LEPTONIC + ('electron_syn',), escape=True)
    a.set_solver_time_step(1.0)
    quiet(a.init_kernels)
    a.set_escape_timescale(10.0)
    for density, get in [('set_current_densities_electrons', 'get_electrons'),
                         ('set_current_densities_photons', 'get_photons')]:
        n = getattr(a, get)().size
        n0 = np.zeros(n)
        n0[n - 30:] = 1.0
        getattr(a, density)(n0)
        quiet(a.evolve_step)
        n1 = getattr(a, get)()
        assert n1[n - 30] > 0.5
        np.testing.assert_allclose(n1[n - 8:n - 1], n1[n - 30], rtol=1e-12)
    a = new(LEPTONS_ALONE + ADDED_LEPTONIC, escape=False)
    a.set_solver_time_step(300.0)
    quiet(a.init_kernels)
    a.set_mag_field(100.0)
    n = a.get_egrid_lep().size
    for j in (n - 4, n - 3, n - 2):
        q = np.zeros(n)
        q[j] = 1.0
        a.set_injection_rate_electrons(q)
        a.set_current_densities_electrons(np.zeros(n))
        quiet(a.evolve_step)
        assert a.get_electrons()[j] == pytest.approx(0.05 * a.get_t_pair_syn()[j], rel=1e-2), j


@pytest.mark.parametrize('photopion', [True, False], ids=['photo-pion', 'proton synchrotron'])
def test_photons_on_the_hadron_grid_after_update_energy_grid(photopion):
    """update_energy_grid(1e-6, ...) and update_energy_grid(1.05e-6, ...) make the same grids, so the same protons
    and photons give the same photo-hadronic rates and proton-synchrotron photons. (For most E_gamma,min, every
    power of ten among them, AM3 put the photons of the hadron grid one bin below where the shift between the grids
    takes them. Photo-hadronic processes saw them at e^-0.1 of their energy (photo-pion production 8 % slower
    here), and the photons that hadrons emit landed e^0.1 too high.)"""
    a, b = hadrons_on_photons(1e-6, photopion), hadrons_on_photons(1.05e-6, photopion)
    np.testing.assert_array_equal(a.get_egrid_photons(), b.get_egrid_photons())
    np.testing.assert_array_equal(a.get_egrid_had(), b.get_egrid_had())
    if photopion:
        np.testing.assert_allclose(a.get_t_proton_photopion(), b.get_t_proton_photopion(), rtol=1e-12)
        for getter in ('get_pion_plus', 'get_pion_minus', 'get_kaon_plus'):
            assert getattr(a, getter)().max() > 0
            np.testing.assert_allclose(getattr(a, getter)(), getattr(b, getter)(), rtol=1e-12)
    else:
        assert a.get_photons_protons_syn().max() > 0
        np.testing.assert_allclose(a.get_photons_protons_syn(), b.get_photons_protons_syn(), rtol=1e-12)


def test_photopion_on_a_larger_hadron_grid():
    """update_energy_grid to a hadron grid larger than the default one (up to 1e24 eV) before init_kernels: in the
    bins that both grids have, the same protons and photons give the photo-pion rates and pions of the default
    grid. (The three arrays of AM3's photon-convolved photo-pion kernels kept the default grid's size and were
    written past their end: the C++ crashed in init_kernels.)"""
    a, b = hadrons_on_photons(None, True), hadrons_on_photons(1e-6, True, e_max=1e24)
    n = a.get_egrid_had().size
    assert b.get_egrid_had().size > n
    np.testing.assert_allclose(b.get_egrid_had()[:n], a.get_egrid_had(), rtol=1e-12)
    np.testing.assert_allclose(b.get_t_proton_photopion()[:n], a.get_t_proton_photopion(), rtol=1e-10)
    for getter in ('get_pion_plus', 'get_pion_minus', 'get_kaon_plus'):
        assert getattr(a, getter)().max() > 0
        np.testing.assert_allclose(getattr(b, getter)()[:n], getattr(a, getter)(), rtol=1e-10, atol=0)


@pytest.mark.parametrize('setter',['set_current_densities_protons', 'set_injection_rate_electrons',
                                    'set_t_proton_extra_cooling', 'set_current_densities_photons'])
def test_arrays_of_the_wrong_length_are_rejected(setter):
    a = new()
    with pytest.raises(Exception, match='elements'):
        getattr(a, setter)(np.ones(7))
    with pytest.raises(Exception):
        getattr(a, setter)(np.ones((2, 400)))


def test_integer_parameters():
    a = new()
    a.set_pp_model_charged_pions(np.int64(1))
    assert a.get_pp_model_charged_pions() == 1
    for bad in (1.5, '1', None):
        with pytest.raises(Exception):
            a.set_pp_model_charged_pions(bad)


EMISSION_TIMESCALES = ['get_t_photon_emission_from_ic', 'get_t_photon_emission_from_pi0',
                       'get_t_photon_emission_from_syn', 'get_t_pair_emission_from_annihilation',
                       'get_t_pair_emission_from_bethe_heitler', 'get_t_pair_emission_from_injection',
                       'get_t_pair_emission_from_photopion', 'get_t_pair_emission_from_proton_proton']


@pytest.mark.parametrize('getter', EMISSION_TIMESCALES)
def test_emission_timescales_before_init_kernels(getter):
    """AM3's 8 emission-timescale getters work before init_kernels and give an array for every bin of their grid.
    (They read arrays that AM3 allocates in init_kernels: called before it, they crashed.)"""
    a = new()
    t = getattr(a, getter)()
    grid = a.get_egrid_photons() if 'photon' in getter else a.get_egrid_lep()
    assert t.size == grid.size and np.all(np.isfinite(t))


def test_neutrino_timescales_cover_the_neutrino_grid():
    """The neutrino escape and extra-cooling timescales have a value for every bin of the neutrino grid
    (AM3 returned only the hadron grid's length of them)."""
    a = new()
    n = a.get_egrid_neutrinos().size
    assert n > a.get_egrid_had().size
    for g in ('get_t_neutrino_escape', 'get_t_neutrino_extra_cooling'):
        assert getattr(a, g)().size == n, g


def test_profile_timing_off_by_default():
    """The profiler is off unless switched on (profile_timing is uninitialised in AM3): no times are
    taken."""
    a = new()
    assert a.get_profile_timing() is False
    quiet(a.init_kernels)
    a.evolve_step()
    assert not any(a.get_profiled_times())


# ---------------------------------------------------------------- e+e- annihilation
def test_annihilation_of_the_pair_components():
    """With e+e- annihilation, the pair components are evolved per charge. The injected component (only electrons)
    annihilates at the electrons' rate, dn/dt = q - a_e- n, while the gamma-gamma pairs annihilate on the positrons
    too. The components add up to the total. (Evolving the component as a whole at a mean rate put 40 % too few
    pairs into it.)"""
    a = new(hadronic=False, escape=False, electron_syn=False, electron_compton=False, ssa=False,
            proton_syn=False, expansion=False, adiabatic_cooling=False, triplet_pair_production=False)
    dt = 1.0
    a.set_solver_time_step(dt)
    quiet(a.init_kernels)
    El, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    il = np.arange(El.size)
    q = np.where((il >= 2) & (il <= 60), 1e12, 0.0)  # electrons only
    a.set_injection_rate_electrons(q)
    a.set_current_densities_photons(np.where(np.abs(np.log(Eg / MEC2_AM3_EV / 30)) < 0.35, 3e14, 0.0))
    n = np.zeros(El.size)
    for _ in range(50):
        a.evolve_step()
        r = 1 / a.get_t_electron_annihilation()
        x = r * dt
        n = np.where(x < 0.1, (1 - x + x * x / 2) * n + (dt - dt * dt * r / 2) * q,
                     np.exp(-x) * n + q / np.maximum(r, 1e-300) * (1 - np.exp(-x)))
        n[:2] = 0
    ne, npos = a.get_electrons(), a.get_positrons()
    comp = a.get_pairs_injected()
    np.testing.assert_allclose(comp + a.get_pairs_annihilation(), ne + npos, rtol=1e-12)
    # where the gamma-gamma positrons are
    s = (npos > 1e-3 * npos.max()) & (il >= 2) & (il <= 58)
    assert s.sum() > 5
    np.testing.assert_allclose(comp[s], n[s], rtol=1e-2)


def test_annihilation_sink_of_a_charge_absent_at_the_start_of_a_step():
    """The positrons injected in a step annihilate on the electrons there at its start, also when there were
    no positrons at the start (the first positrons of a run): the sink of each charge comes from the density
    of the other one, while the photon source needs both. (Both sinks were zero when either charge was absent,
    and the timescale getters reported 3e177 s.)"""
    a = new(CLOSED_BOX + ('triplet_pair_production',))
    dt, q = 1e3, 1e9
    a.set_solver_time_step(dt)
    quiet(a.init_kernels)
    il = np.arange(a.get_egrid_lep().size)
    a.set_current_densities_electrons(np.where(il == 2, 1e12, 0.0))
    a.set_injection_rate_positrons(np.where(il == 2, q, 0.0))
    a.evolve_step()
    t = a.get_t_positron_annihilation()[2]
    assert t < dt * 10
    # no cooling or escape: the solver's escape-dominated regime, dn/dt = q - n/t exactly
    np.testing.assert_allclose(a.get_positrons()[2], q * t * (1 - np.exp(-dt / t)), rtol=1e-2)
    # and no sink for the electrons, as there were no positrons at the start of the step
    assert a.get_electrons()[2] == 1e12
    assert np.all(a.get_photons_pair_annihilation() == 0.0)
