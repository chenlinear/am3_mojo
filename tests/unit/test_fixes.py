"""Regression tests of the bugs am3_mojo fixes: AM3's and the port's own.

They cover every fix of tests/reference/README.md ("Bugs of AM3 v1.2.0 that am3_mojo fixes") but the OpenMP race,
which tests/unit/test_threads.py covers. Each test fails if any one of its fixes is reverted. The tests of reads and
writes past the end of an array fail at least in the build with bounds checks (make test-asserts), as their effect
depends on the memory there; the exception is the hadrons' inverse-Compton cooling (see its test).
"""
import os, subprocess, sys
import mpmath as mp
import numpy as np
import pytest

from support.output import quiet
from support.scenarios import leptohadronic_params, sphere_volume
from support.setups import ADDED_LEPTONIC, CLOSED_BOX, LEPTONS_ALONE, evolve, new, per_ln_e
from support.units import C_CGS, DLNE, MASS_EV, MASS_GEV, MEC2_EV, MP_C2_EV

UNIT = os.path.dirname(os.path.abspath(__file__))


def rich(a, steps=0):
    """a as the rich source (the leptohadronic scenario's parameters in 100 G, where all hadronic processes matter)
    after `steps` steps."""
    leptohadronic_params(a, bfield=100.0, time_step_frac=1e-2)
    return evolve(a, steps)


# ---------------------------------------------------------------- the solver
def const_cool(t_cool_steps, below=None):
    """Protons with an energy-independent extra cooling time of t_cool_steps time steps (none for None; with `below`
    [s], only above 1e14 eV and `below` under it) and escape in R/c = 100 steps: the hadron grid and the protons
    after 30 steps."""
    a = new(photopion=False, bethe_heitler=False, pp=False)
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
    return E, a.get_protons()


@pytest.fixture(scope='module')
def uncooled_protons():
    """const_cool without extra cooling."""
    return const_cool(None)


def test_fast_cooling_above_slow_cooling(uncooled_protons):
    """A bin's solution in a fast-cooling window involves only the bins above it, so it must not depend on how slowly
    the bins below cool. AM3 summed the cooling time from the lowest of the window's 10 padding cells (bins of other
    regimes or, below the grid, cells with A = 0; bin 0: A = -1e-150). Where those cool about 1e15 times more slowly,
    this sum swamps the window's cooling times in double precision: densities 28 % too low at 1e16, and from 1e17 on
    no cooling. This hits cooling faster than an e-fold per step that is not synchrotron-like. Here, 1e20 s in the
    lowest cell left the protons uncooled: 1.16 times the density without cooling instead of 0.011."""
    E, n0 = uncooled_protons
    i = np.argmin(np.abs(E - 1.6e14))
    n15, n20 = const_cool(0.3, below=1e15)[1][i], const_cool(0.3, below=1e20)[1][i]
    assert n20 == pytest.approx(n15, rel=1e-9)
    assert n20 < 0.05 * n0[i]


def electrons_box(dt, t_esc, B=0.0, t_cool=None, rk4=False):
    """Electrons injected as E^-1 per ln E from 1e8 to 1e13 eV, cooling by synchrotron radiation in the field B or in
    the energy-independent extra cooling time t_cool, and escaping in t_esc; nothing else: the object, the grid and
    the injection."""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC, electron_syn=t_cool is None, dt=dt, init=False)
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


def source_integral(E, q, t_cool, t_esc, reach=np.inf):
    """t_cool(x) int_x^min(x + reach, top) q(x') exp(-int_x^x' t_cool / t_esc dx'') dx' (x = ln E): the electrons
    of the source q that cool (e-folding time t_cool) and escape, from x' at most `reach` e-folds above x. As in the
    solver, q and ln t_cool are linear between grid points, and the integral is exact on sub-intervals."""
    with np.errstate(all='ignore'):
        sub = 50   # sub-intervals per grid cell
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
    """Regime 3 at a time step much longer than the escape time: the steady state with synchrotron cooling and escape
    must be that of the continuous equation. AM3's quadrature of the source integral (simple_integral_q) gave
    densities too low by z/(e^z - 1), z = dx a/|A| (0.08 at 1 GeV, 7e-4 just above regime 1, where escape is 100
    times faster than cooling), and too high where cooling dominates. The test leaves out the top cell of the
    escape-dominated window below: the particles that cool into it escape near its upper edge, so it holds the right
    number of particles, not the density at its centre."""
    t_esc = 1e4
    a, E, q = electrons_box(1e4 * t_esc, t_esc, B=1.0)
    evolve(a, 6)
    t_syn = a.get_t_pair_syn()
    exact = source_integral(E, q, t_syn, t_esc)
    sel = (E >= 3e8) & (E <= 5e12)
    sel[np.nonzero(t_syn > 100 * t_esc)[0].max()] = False
    np.testing.assert_allclose(a.get_electrons()[sel], exact[sel], rtol=1e-2)


@pytest.mark.parametrize('dt', [1e2, 1e3, 1e4])
def test_particles_cooling_into_a_window_after_a_change(dt):
    """Electrons injected per ln E as 1e9 eV / E from 1e9 to 1e12 eV from t = 0, cooling by synchrotron radiation in
    30 G, without escape (as in tests/reference/test_regime_boundary.py): after k steps they number q k dt, as long as
    none has cooled to the bottom of the grid. AM3's 10-cell padding of the windows dropped 3 % (dt = 100 s) and
    25-28 % (1e3 and 1e4 s) within 10-50 steps. The inflow into a Chang & Cooper window must be the electrons that
    cross into it within the step (the new densities of the window above add too many in the first step)."""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC, escape=False, dt=dt)
    a.set_mag_field(30.0)
    a.set_escape_timescale(1e4)
    q = per_ln_e(a.get_egrid_lep())
    a.set_injection_rate_electrons(q)
    for k in range(1, 11):
        a.evolve_step()
        assert np.sum(a.get_electrons()) / (np.sum(q) * k * dt) == pytest.approx(1.0, abs=1e-2), k


def cooled_into_a_window(dt, t_cool, steps):
    """Electrons escaping in t_esc = 1e4 s, with the extra cooling time t_cool(E), injected per ln E as 1e9/E from
    1e10 to 1e12 eV: their number after `steps` steps of dt over Q t_esc (steady state: 1)."""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC + ('electron_syn', 'quantum_syn'), escape=True, dt=dt)
    a.set_escape_timescale(1e4)
    E = a.get_egrid_lep()
    a.set_t_pair_extra_cooling(t_cool(E))
    q = np.where((E >= 1e10) & (E <= 1e12), 1e9 / E, 0.0)
    a.set_injection_rate_electrons(q)
    evolve(a, steps)
    return np.sum(a.get_electrons()) / (np.sum(q) * 1e4)


@pytest.mark.parametrize('dtf', [1e-2, 0.1, 1.0])
def test_particles_cooling_into_an_escape_dominated_window(dtf):
    """Electrons that cool down to where their cooling stops pile up there and escape: in the steady state their
    number is Q t_esc. Here (cooled_into_a_window) they cool in 0.1 t_esc above 1e9 eV and in 1e12 s below (an
    escape-dominated window, regime 1), for 20 escape times at dt = dtf t_esc. (AM3 gave the escape-dominated
    windows no inflow and dropped the particles that cooled into one from the window above: 72 % here. am3_mojo
    keeps them to 0.4 %.)"""
    n = cooled_into_a_window(dtf * 1e4, lambda E: np.where(E > 1e9, 1e3, 1e12), int(round(20 / dtf)))
    assert n == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize('dt', [30.0, 100.0, 300.0])
def test_particles_cooling_fast_into_a_slow_window(dt):
    """Electrons (cooled_into_a_window) cooling in 10 s per e-fold (regime 4) into a slow stretch (3000 s, a Chang &
    Cooper window) and then into none (regime 1): in the steady state they number Q t_esc. (The inflow into the Chang
    & Cooper window, counted along the characteristics, must not take the densities of the half cell above its edge
    from both cells: that counts the particles piled up in its top cell again, +12 to +53 % here and a growth to
    1e89 at dt = 30 s.)"""
    t_cool = lambda E: np.where(E > 1e9, 10.0, np.where(E > 1e9 * np.exp(-1.2), 3000.0, 1e12))
    assert cooled_into_a_window(dt, t_cool, int(max(60, 3e5 / dt))) == pytest.approx(1.0, abs=1.5e-2)


def test_steep_injection_tail_in_regime_4():
    """Synchrotron cooling only (regime 4), injection falling by exp(-E / 1 GeV): above the energy the cooling reaches
    within the step, one step from empty gives the steady state n |A| = int_x^top q dx (their ratio is compared with
    its value at 3 GeV, which removes the units of the cooling-time getter). AM3 accumulated int q dx from the bottom
    of the window, restarting it where q dx / sum < 1e-15 (twice here): the bins below each restart kept almost no
    significant digits of their injection, those from 1.0e11 eV (q / q_max < 1e-47) on none (n = 0)."""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC, escape=False, dt=1e5)
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


def one_step(t_cool, t_esc, escape=None, q=None, rk4=False):
    """One step of dt = 3 t_cool from empty of electrons_box with the cooling time t_cool, escape in t_esc or, if
    given, escape(E), and the injection q(E) if given: the grid, the electrons and the exact ones (source_integral
    over 3 e-folds)."""
    a, E, q0 = electrons_box(3 * t_cool, t_esc, t_cool=t_cool, rk4=rk4)
    if escape:
        t_esc = escape(E)
        a.set_t_pair_escape(t_esc)
    if q:
        q0 = q(E)
        a.set_injection_rate_electrons(q0)
    a.evolve_step()
    return E, a.get_electrons(), source_integral(E, q0, np.full(E.size, t_cool), t_esc, reach=3.0)


@pytest.mark.parametrize('t_cool, k, rk4', [(1e3, 5.0, False), (1e3, 50.0, False), (1e3, 0.05, True),
                                            (1e-5, None, False)],
                         ids=['escape-5x', 'escape-50x', 'escape-0.05x-rk4', 'escape-vanishing'])
def test_source_integral_of_one_step(t_cool, k, rk4):
    """One step of dt = 3 t_cool from empty, with an energy-independent cooling time t_cool: every bin holds the
    electrons injected at most 3 e-folds above it.
    * Escape in t_cool / k (regime 3): AM3's quadrature gave n/exact = 0.81 and 0.035 for k = 5 and 50. Where the
      sink is weak, its RK4 option (set_solver_simpleqintegration) integrated beyond those 3 e-folds, up to the top
      of the solver's window (+4.4 % for k = 0.05).
    * t_cool = 10 us, escape in t_cool below 1e11 eV and none above: most electrons in the top 3 e-folds below 1e11 eV
      come from above it. AM3's simple_integral_q skipped the points where alpha/|A| < e^-300 with their injection:
      46-90 % too few electrons in the 6 bins below 1e11 eV. As both codes add 1e-150 to alpha in the solver's units
      (time in units of 2.7e-23 s for the e+-), without escape this needs cooling in less than about 1.4 ms. The
      case also needs the semi-analytic windows padded up to where particles reach them within the step (AM3's 10
      cells: up to 34 % too few electrons below 1e11 eV)."""
    escape = None if k else lambda E: np.where(E < 1e11, t_cool, 1e250)
    E, n, exact = one_step(t_cool, t_cool / k if k else t_cool, escape, rk4=rk4)
    sel = (E >= 2e8) & (E <= 5e12)
    np.testing.assert_allclose(n[sel], exact[sel], rtol=5e-3)


def test_source_integral_of_a_steeply_rising_source():
    """one_step with escape in 20 t_cool (regime 3) and a source rising as E^2 per ln E from 1e7 to 1e16 eV. A
    source integral of regime 3 taken as the difference of two integrals up to the top of the window loses its digits
    where the electrons injected further up dominate: up to 45 % too many electrons at 1e7 eV."""
    t_cool = 1e3
    E, n, exact = one_step(t_cool, 20 * t_cool, q=lambda E: np.where((E >= 1e7) & (E <= 1e16), (E / 1e16) ** 2.0, 0.0))
    sel = (E >= 1e7) & (E <= 3e14)
    np.testing.assert_allclose(n[sel], exact[sel], rtol=5e-3)


@pytest.mark.parametrize('t_cool', [0.3, 0.5, 0.9])
def test_cooled_particles_in_regime_4(uncooled_protons, t_cool):
    """Protons with an energy-independent cooling time t_cool of 0.3 to 0.9 time steps and escape in 100 steps
    (regime 4). In the steady state their energy content is L t, 1/t = 1/t_cool + 1/t_esc. A bin x takes the
    particles n(y) A(y)/A(x) that cooling brings from y to x within the step. AM3 subtracted them instead: 6 %,
    23 % and 49 % too little energy for t_cool = 0.3, 0.5 and 0.9 steps."""
    E, n0 = uncooled_protons
    L_tesc = np.sum(n0 * E) / (1 - np.exp(-0.3))
    t = 1 / (1 / t_cool + 1 / 100.0) / 100.0  # in units of t_esc
    n = const_cool(t_cool)[1]
    assert np.sum(n * E) / (L_tesc * t) == pytest.approx(1.0, rel=0.01)


def rk4_one_step_across_an_escape_step():
    """The largest |n/exact - 1| of one_step with the RK4 option, t_cool = 1e3 s and escape in 10 t_cool below
    1e10 eV and t_cool / 100 above."""
    t_cool = 1e3
    E, n, exact = one_step(t_cool, 10 * t_cool, lambda E: np.where(E < 1e10, 10 * t_cool, t_cool / 100), rk4=True)
    sel = (E >= 2e8) & (E <= 5e12)
    return float(np.max(np.abs(n[sel] / exact[sel] - 1)))


def test_rk4_source_integral_across_an_escape_step():
    """The RK4 option (set_solver_simpleqintegration) with escape 1000 times faster above 1e10 eV than below. For the
    bins up to 0.3 e-folds below that energy, the first tenth of the integration range reaches across it, where the
    sink takes more than e^2 of the particles, so the integration narrows its range and widens it again. The widening
    must update the integration step: AM3's loop did not, and never ended. With the update, every bin is within 2 %
    of the exact integral. The case runs in a subprocess, so that an endless loop fails this test instead of stopping
    the run."""
    code = (f'import sys; sys.path[:0] = [{os.path.dirname(UNIT)!r}, {UNIT!r}]\n'
            'from test_fixes import rk4_one_step_across_an_escape_step as f\nprint(f())')
    try:
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail('the RK4 source integral did not end within 120 s')
    assert r.returncode == 0, r.stderr[-3000:]
    assert float(r.stdout.split()[-1]) < 0.02


# the species and their masses [GeV] in the code (AM3's)
SYN_SPECIES = [(s, MASS_GEV[m]) for s, m in (('pair', 'MASS_E'), ('proton', 'MASS_P'), ('pion', 'MASS_PI'),
                                             ('muon', 'MASS_MU'))]


@pytest.fixture(scope='module')
def synchrotron_cooling_times():
    """The field, the synchrotron cooling times of every species without and with the quantum suppression
    ({False: {species: t}, True: ...}), and the lepton and hadron grids."""
    B = 1e5
    t = {}
    for q in (False, True):
        a = new(quantum_syn_cooling=q)
        a.set_mag_field(B)
        a.evolve_step()  # the solvers compute their cooling rates in a step
        t[q] = {species: getattr(a, f'get_t_{species}_syn')() for species, _ in SYN_SPECIES}
    return B, t, a.get_egrid_lep(), a.get_egrid_had()


@pytest.mark.parametrize('species, mass', SYN_SPECIES)
def test_quantum_synchrotron_cooling_of_heavy_particles(synchrotron_cooling_times, species, mass):
    """AM3's quantum suppression of synchrotron cooling, 1 / (1 + sqrt(2) x^(2/3))^2 (an interpolation of Landau &
    Lifshitz), takes the quantum parameter x = chi = gamma B / B_cr(m) = (m_e / m)^2 gamma B / B_cr(m_e), as
    B_cr ∝ m^2. AM3 used gamma B / m, m / m_e times that, which overstates the suppression of heavy particles."""
    B, t, El, Eh = synchrotron_cooling_times
    t0, t1 = t[False][species], t[True][species]
    gamma = (El if species == 'pair' else Eh) / (mass * 1e9)
    x = 1.2e-8 * B * gamma / (mass * 1e9) * (MASS_GEV['MASS_E'] / mass)
    fin = t0 < 1e199
    assert fin.sum() > 250
    np.testing.assert_allclose(t1[fin] / t0[fin], (1 + np.sqrt(2) * x[fin] ** (2 / 3)) ** 2, rtol=2e-5)


def jones_F(eps, gamma):
    """The inverse-Compton cooling function F(eps, gamma) of Jones 1965 (eqs. 13-15), eps and gamma (mpf) in units of
    the particle's rest energy, at mpmath's working precision."""
    def f1(z):  # eq. 14
        return ((z + 6 + 3 / z) * mp.log(1 + 2 * z) - (mp.mpf(22) / 3 * z ** 3 + 24 * z ** 2 + 18 * z + 4)
                / (1 + 2 * z) ** 2 - 2 + 2 * mp.polylog(2, -2 * z))

    def f2(z):  # eq. 15
        return ((z + mp.mpf(31) / 6 + 5 / z + 3 / (2 * z ** 2)) * mp.log(1 + 2 * z)
                - (mp.mpf(22) / 3 * z ** 3 + 28 * z ** 2 + mp.mpf(103) / 3 * z + 17 + 3 / z) / (1 + 2 * z) ** 2
                - 2 + mp.polylog(2, -2 * z))

    g = gamma + mp.sqrt(gamma ** 2 - 1)  # gamma (1 + beta), after eq. 13
    return gamma * (f1(eps * g) - f1(eps / g)) - eps * (f2(eps * g) - f2(eps / g))  # eq. 13


@pytest.fixture(scope='module')
def photon_line():
    """Photons of one energy (1e10 at the grid point nearest 25 eV), with photo-pion production, Bethe-Heitler and pp
    off, after one step: the object, the photon energy and F(eps, gamma) of the inverse-Compton loss at each gamma
    (jones_F with 90 digits)."""
    a = new(photopion=False, bethe_heitler=False, pp=False)
    Eg = a.get_egrid_photons()
    j = np.argmin(np.abs(Eg - 25.0))
    a.set_current_densities_photons(np.where(np.arange(Eg.size) == j, 1e10, 0.0))
    a.evolve_step()

    def F(eps, gamma):
        with mp.workdps(90):
            return np.array([float(jones_F(mp.mpf(eps), mp.mpf(g))) for g in gamma])
    return a, Eg[j], F


def test_inverse_compton_cooling_functions_below_z_of_one_tenth(photon_line):
    """AM3 evaluates the cooling functions f1, f2 of the inverse-Compton loss,
    F(eps, gamma) = gamma [f1(eps g) - f1(eps / g)] - eps [f2(eps g) - f2(eps / g)], g = gamma + sqrt(gamma^2 - 1),
    by Taylor series below z = 0.1. It kept the first three terms of Jones' series, 2.4 % (f1) and 1.2 % (f2) off at
    z = 0.1: the cooling time jumped by 2.4 % where eps g crosses 0.1. With the series to z^10 (Jones 1965, Table I),
    F follows the exact functions to 2e-5. Electrons in photons of one energy eps (photon_line) have the cooling time
    t ∝ gamma^3 / F(eps, gamma)."""
    a, Eg, F = photon_line
    t = a.get_t_pair_compton()
    gamma = a.get_egrid_lep() / MEC2_EV
    eps = Eg / MEC2_EV
    z = eps * (gamma + np.sqrt(np.maximum(gamma * gamma - 1, 0.0)))
    sel = (z > 1e-3) & (z < 0.1)
    assert sel.sum() >= 30 and (z[sel] > 0.09).any()
    r = t[sel] * F(eps, gamma[sel]) / gamma[sel] ** 3
    r = r / np.median(r[z[sel] < 1e-2])
    np.testing.assert_allclose(r, 1.0, rtol=5e-5)


@pytest.mark.parametrize('species, mass', [('pion', MASS_EV['MASS_PI']), ('muon', MASS_EV['MASS_MU'])])
def test_inverse_compton_cooling_of_hadrons_in_every_bin(photon_line, species, mass):
    """Pions and muons in photons of one energy eps (photon_line): in every bin of the hadron grid, their
    inverse-Compton cooling time is t ∝ gamma^3 / F(eps, gamma) within 3e-3 (the photon line is interpolated onto the
    hadron grid), with eps in units of the particle's rest energy. This includes the top 49 hadron bins, whose kernels
    reach past the photon grid. (AM3 read those photon bins past the end of the array: in 5 of 45 AM3 objects created
    after another one in the same process, the top 30 bins got cooling times of 1e-187 to 1e-18 s, and negative ones.
    am3_mojo reads the photons through an unchecked pointer, so a regression to that read shows here only if the
    memory past the array is not all zeros.)"""
    a, Eg, F = photon_line
    t = getattr(a, f'get_t_{species}_compton')()
    gamma = a.get_egrid_had() / mass
    r = t * F(Eg / mass, gamma) / gamma ** 3
    assert np.all(np.isfinite(r)) and np.all(r > 0)
    np.testing.assert_allclose(r / np.median(r), 1.0, rtol=3e-3)


def test_muons_of_pion_decay():
    """Every pi+ decay makes one muon: a mu+_L with the energy fraction distribution (x - r) / ((1 - r)^2 x) on
    [r, 1], or a mu+_R with r (1 - x) / ((1 - r)^2 x), r = (m_mu/m_pi)^2 (AM3 paper; Lipari et al. 2007). AM3 took
    these at the grid points and dropped x = 1, where the first is largest: a decay made 0.876 muons (0.549 of them
    mu+_L for 0.592), with 0.668 of the pion's energy for (1 + r)/2 = 0.786. Pions injected at 1e12 eV into a closed
    box, with muon decay off, keep their decays' muons: one per decay, with the right helicities and their energy to
    the binning's 0.2 %."""
    a = new(off=('escape', 'expansion', 'adiabatic_cooling', 'pion_syn', 'pion_compton', 'muon_syn', 'muon_compton',
                 'muon_decay'), dt=1.0)  # far longer than the decay times of these pions (2e-4 s)
    E = a.get_egrid_had()
    q = np.where(np.abs(np.log(E / 1e12)) < DLNE / 2, 1.0, 0.0)
    a.set_injection_rate_pion_plus(q)
    evolve(a, 10)
    decays = (10 * np.sum(q) - np.sum(a.get_pion_plus())) * DLNE  # 10 steps of 1 s
    left, right = a.get_mu_plus_left(), a.get_mu_plus_right()
    n = np.sum(left + right) * DLNE
    r = (MASS_GEV['MASS_MU'] / MASS_GEV['MASS_PI']) ** 2
    assert abs(n / decays - 1) < 1e-4
    np.testing.assert_allclose(np.sum(left) * DLNE / n, 1 - r * (np.log(1 / r) - 1 + r) / (1 - r) ** 2, rtol=1e-9)
    np.testing.assert_allclose(np.sum((left + right) * E) * DLNE / (decays * E[q > 0][0]), (1 + r) / 2, rtol=5e-3)


def test_pairs_of_muon_decay_by_helicity():
    """In V-A muon decay the e+- have the spectrum of the decay's nu_mu-type neutrino, which depends on the muon's
    helicity h: 4/3 (1 - x^3) for h = +1 and 2 - 6x^2 + 4x^3 for h = -1 (mu+; the reverse for mu-), with mean energy
    fractions 0.4 and 0.3 (the nu_mu spectra of the AM3 paper and Lipari et al. 2007). AM3 gave the e+- of every
    muon 4/3 (1 - x^3) ("omitting chiralities"): those of the mu+_L and mu-_R took 0.40 of the muon energy instead
    of 0.30, and their decays made 10 % more energy than the muons had. Muons of each state injected at 1e12 eV into
    a closed box give one e+- per decay, with the mean energy of its helicity (to the kernel's discretisation, 0.7 %
    for h = +1)."""
    states = [('mu_plus_left', 'get_positrons', 0.3), ('mu_plus_right', 'get_positrons', 0.4),
              ('mu_minus_left', 'get_electrons', 0.4), ('mu_minus_right', 'get_electrons', 0.3)]
    for state, getter, mean in states:
        a = new(off=ADDED_LEPTONIC + ('photopion', 'pp', 'bethe_heitler', 'escape', 'expansion', 'adiabatic_cooling',
                                      'muon_syn', 'muon_compton', 'electron_syn', 'electron_compton', 'annihilation',
                                      'ssa'), dt=1.0)  # far longer than the decay time of these muons (2e-2 s)
        E, El = a.get_egrid_had(), a.get_egrid_lep()
        q = np.where(np.abs(np.log(E / 1e12)) < DLNE / 2, 1.0, 0.0)
        getattr(a, 'set_injection_rate_' + state)(q)
        evolve(a, 10)
        decays = (10 * np.sum(q) - np.sum(a.get_muons())) * DLNE  # 10 steps of 1 s
        n = getattr(a, getter)()
        assert abs(np.sum(n) * DLNE / decays - 1) < 5e-3, state
        np.testing.assert_allclose(np.sum(n * El) * DLNE / (decays * E[q > 0][0]), mean, rtol=1e-2, err_msg=state)


# ---------------------------------------------------------------- switches
def test_kaons_switched_off_after_init():
    """Kaons switched off after init_kernels are off: no kaons, and AM3's pion multiplicities are not scaled for the
    kaon channels (a scale set only at init_kernels keeps them scaled, by up to 10 %). Charm, whose decays make
    kaons, is off."""
    a = new(charm=False)
    a.set_process_kaons(False)
    b = new(kaons=False, charm=False)
    for x in (a, b):
        rich(x, 10)
    assert not a.get_kaons().any()
    for g in ('get_pion_minus_photopion', 'get_photons_pi0_decay_photopion', 'get_photons'):
        np.testing.assert_allclose(getattr(a, g)(), getattr(b, g)(), rtol=1e-12, atol=0, err_msg=g)


def test_parse_sed_does_not_change_the_outputs():
    """parse_sed only splits the outputs into components: in the rich source with injected mu-_L and pi0, parse_sed on
    and off give the same e+- and photons (rtol 1e-9). (With parse_sed off, AM3 dropped the photons of pi0 from pp.
    With parse_sed on, it took the e+- of muon decays and the photons of pi0 decays as the sums of their photo-pion
    and pp components, which leave out the decay products of injected particles.)"""
    out = []
    for parse in (True, False):
        a = rich(new(parse_sed=parse))
        E = a.get_egrid_had()
        q = np.where((E > 1e10) & (E < 1e15), 1e-3 * (E / 1e10) ** -1.0, 0.0)
        a.set_injection_rate_mu_minus_left(q)
        a.set_injection_rate_pi0(q)
        evolve(a, 20)
        out.append((a.get_pairs(), a.get_photons()))
    for on, off in zip(*out):
        assert off.max() > 0
        np.testing.assert_allclose(on, off, rtol=1e-9, atol=0)


MUONS = ['mu_minus_left', 'mu_plus_left', 'mu_minus_right', 'mu_plus_right']
NEUTRINOS = ['electron_neutrinos', 'electron_anti_neutrinos', 'muon_neutrinos', 'muon_anti_neutrinos']


@pytest.mark.parametrize('species', MUONS + NEUTRINOS)
def test_external_injection_of_muons_and_neutrinos(species):
    """An external injection of one muon or neutrino species makes that species only. (AM3 evolved the injection of
    mu+_L as mu-_R and that of mu-_R as mu+_L, added that of nu_e to all four neutrino species and ignored the other
    three.)"""
    group = MUONS if species in MUONS else NEUTRINOS
    a = new(photopion=False, bethe_heitler=False, pp=False, muon_decay=False)
    a.set_mag_field(1e-3)
    n = (a.get_egrid_had() if species in MUONS else a.get_egrid_neutrinos()).size
    q = np.zeros(n)
    q[n // 2] = 1.0
    getattr(a, 'set_injection_rate_' + species)(q)
    quiet(a.evolve_step)
    for other in group:
        got = getattr(a, 'get_' + other)()
        assert (got.max() > 0) if other == species else not got.any(), other


def test_pion_decay_off_mid_run():
    """Pion decay switched off during a run of the rich source without kaons and charm hadrons (whose decays make
    muons and neutrinos), before the third of 7 steps of dt = t_esc/100. The muons and the muon neutrinos grew before.
    * The muon sources from pion decays must stop (AM3 kept injecting the last step's muons). As cooling keeps the
      muon number and decay lowers it, it then falls by at least the escape, exp(-dt/t_esc) = exp(-0.01), in every
      step (within 1e-9).
    * With muon decay on, the muon neutrinos then come only from the decays of the muons left, which get no more
      sources: nu_mu + anti-nu_mu fall by the escape in every step, within 1e-6. (AM3 never reset the total nu_mu
      sources and added each step's muon-decay nu_mu to them, so the last step's sources kept coming.)"""
    a = rich(new(kaons=False, charm=False), 9)
    M, N = [], []
    for step in range(7):
        if step == 2:
            assert M[1] > M[0] > 0 and N[1] > N[0] > 0
            a.set_process_pion_decay(False)
        a.evolve_step()
        M.append(np.sum(a.get_muons()))
        N.append(np.sum(a.get_muon_neutrinos()) + np.sum(a.get_muon_anti_neutrinos()))
        if step >= 2:
            assert M[-1] <= M[-2] * np.exp(-0.01) * (1 + 1e-9), step
            assert N[-1] <= N[-2] * np.exp(-0.01) * (1 + 1e-6), step


def test_muon_decay_off_mid_run_stops_its_neutrinos():
    """With muon decay switched off during a run (the rich source without kaons and charm hadrons, whose decays make
    neutrinos of every flavour, before the third of 6 steps), the electron neutrinos must stop coming (AM3 kept
    injecting the last step's nu_e and anti-nu_e). Their number then falls by the escape in every step,
    exp(-dt/t_esc) = exp(-0.01) within the solver's 2e-7 per step, while it grew before."""
    a = rich(new(kaons=False, charm=False), 9)
    N = []
    for step in range(6):
        if step == 2:
            assert N[1] > N[0] > 0
            a.set_process_muon_decay(False)
        a.evolve_step()
        N.append(np.sum(a.get_electron_neutrinos()) + np.sum(a.get_electron_anti_neutrinos()))
        if step >= 2:
            assert N[-1] <= N[-2] * np.exp(-0.01) * (1 + 1e-6), step


def test_kaon_decays_without_pion_and_muon_decay():
    """Kaon decays make muons (K -> mu nu, ...) whether or not pion decay is on, and their e+- (K_e3) feed the pair
    components of their source whether or not muon decay is on. Charm, whose decays make muons and e+-, is off, so
    that all of them come from kaons. (test_no_neutrinos_without_pion_decay is the control without kaons: no
    muons.)"""
    a = rich(new(pion_decay=False, muon_decay=False, charm=False), 10)
    assert a.get_muons().max() > 0 and a.get_muon_neutrinos().max() > 0
    for g in ('get_pairs_photopion', 'get_pairs_proton_proton'):
        assert getattr(a, g)().max() > 0, g


def test_estimated_electron_emax_with_merged_positrons():
    """estimate_max_energies with positrons merged into electrons must use the cooling of the merged e+- (AM3 read
    the unevolved electron solver and gave the top of the grid)."""
    emax = {}
    for merged in (False, True):
        a = new(merge_positrons_into_electrons=merged, init=False)
        a.set_estimate_max_energies(True)
        quiet(a.init_kernels)
        rich(a)
        quiet(evolve, a, 20)
        emax[merged] = a.get_estimated_electron_emax()
    assert emax[False] < 1e15
    assert emax[True] == pytest.approx(emax[False], rel=0.3)


def test_estimated_max_energies_where_acceleration_wins_everywhere():
    """With estimate_max_energies and no cooling of the electrons and protons, acceleration wins on the whole grid:
    the estimated maximum energies are the energy one bin above the grid's top, and the injection ending there is
    finite. (AM3 read the loss rates and the energy grid one element past their end, and the injection became NaN
    when that memory read 0.)"""
    a = new(off=('electron_syn', 'electron_compton', 'proton_syn', 'photopion', 'bethe_heitler', 'pp'), init=False)
    a.set_estimate_max_energies(True)
    quiet(a.init_kernels)
    rich(a, 3)
    El, Eh = a.get_egrid_lep(), a.get_egrid_had()
    assert a.get_estimated_electron_emax() == pytest.approx(El[-1] ** 2 / El[-2], rel=1e-9)
    assert a.get_estimated_proton_emax() == pytest.approx(Eh[-1] ** 2 / Eh[-2], rel=1e-9)
    for n in (a.get_electrons(), a.get_protons()):
        assert np.all(np.isfinite(n)) and n.max() > 0


def test_pi0_decay_photons_of_the_top_hadron_bin():
    """The pi0-decay kernel depends only on the ratio of the photon and pion energies, so on the common logarithmic
    grid, pi0 injected into the top hadron bin make the photons of pi0 injected ten bins lower, ten photon bins up
    (the solver leaves the two lowest bins and the top bin of a grid empty). (AM3 read the top bin's upper edge,
    egrid_had[j+1], past the end of the grid: in 8 of 95 fresh processes with photo-pion production, the total photon
    spectrum was zero for the whole run.)"""
    out = []
    for below_top in (0, 10):
        a = new(off=('photopion', 'bethe_heitler', 'pp', 'annihilation', 'ssa', 'electron_syn', 'electron_compton',
                     'proton_syn', 'pair_annihilation', 'triplet_pair_production', 'kaons'))
        a.set_escape_timescale(1e4)
        a.set_solver_time_step(1e2)
        q = np.zeros(a.get_egrid_had().size)
        q[-1 - below_top] = 1.0
        a.set_injection_rate_pi0(q)
        a.evolve_step()
        out.append(a.get_photons())
    top, low = out[0][12:-1], out[1][2:-11]
    assert np.count_nonzero(top) > 500
    np.testing.assert_allclose(top, low, rtol=1e-12, atol=0)


def test_no_neutrinos_without_pion_decay():
    """With pion decay off (and no kaons, charm or neutron decay), no neutrinos are made in any bin of the neutrino
    grid, nor muons (the control of test_kaon_decays_without_pion_and_muon_decay). (AM3 read the nu_mu sources of
    pion decays, then of the hadron grid's length, on the neutrino grid, past their end.)"""
    a = rich(new(pion_decay=False, kaons=False, charm=False, neutron_decay=False), 5)
    assert a.get_pions().max() > 0 and not a.get_muons().any()
    for g in ('get_neutrinos', 'get_muon_neutrinos', 'get_muon_anti_neutrinos', 'get_electron_neutrinos'):
        assert not getattr(a, g)().any(), g


def test_pp_transition_energy_in_the_top_hadron_bin():
    """The transition energy of the pp delta approximation in the top bin of the hadron grid gives the same pp
    secondaries as one bin lower, as no protons get there. (AM3's index of the transition in the multiplicity table
    was one past its end.)"""
    out = []
    for below_top in (0, 1):
        a = new(photopion=False, bethe_heitler=False, init=False)
        a.set_pp_transition_energy_to_delta_approx(a.get_egrid_had()[-1 - below_top])
        quiet(a.init_kernels)
        rich(a, 3)
        out.append((a.get_pions(), a.get_photons_pi0_decay_proton_proton()))
    for top, lower in zip(*out):
        assert np.all(np.isfinite(top)) and top.max() > 0
        np.testing.assert_array_equal(top, lower)


# ---------------------------------------------------------------- optimisations
@pytest.fixture(scope='module')
def rich_run():
    """The rich source with every default switch and optimisation limit, after 5 steps."""
    return rich(new(), 5)


def test_bethe_heitler_minimum_raised_and_lowered(rich_run):
    """Raising the minimum incoming-proton energy of the Bethe-Heitler pair kernel for a step and lowering it again
    must not lose kernel rows, as a kernel packed from the first step's minimum would: after 5 steps, the pairs are
    those of rich_run, where the minimum stays. The first step has no target photons, so both runs are the same
    physics."""
    a = rich(new())
    p0 = a.get_optimize_bethe_heitler_incoming_protons_min() * 1e9  # GeV -> eV
    a.set_optimize_bethe_heitler_incoming_protons_min(1e3 * p0)
    a.evolve_step()
    a.set_optimize_bethe_heitler_incoming_protons_min(p0)
    ref = rich_run.get_pairs_bethe_heitler()
    assert ref.max() > 0
    np.testing.assert_array_equal(evolve(a, 4).get_pairs_bethe_heitler(), ref)


def test_compton_emission_minimum_after_init():
    """The minimum IC photon energy set after init_kernels applies, not the value at init_kernels: no IC photons
    below it."""
    a = rich(new(hadronic=False))
    a.set_optimize_compton_emission_emin(1e3)  # eV
    evolve(a, 5)
    ic = a.get_photons_injected_electrons_compton()
    assert ic.max() > 0
    assert not ic[a.get_egrid_photons() < 0.9e3].any()


OPTIMISATION_LIMITS = {  # setter: grid whose end limits it
    'photopion_target_photon_max': 'photons', 'bethe_heitler_target_photon_max': 'photons',
    'compton_target_emax': 'photons', 'compton_emission_emin': 'photons',
    'bethe_heitler_incoming_protons_min': 'had'}


def test_optimisation_limits_beyond_the_grids():
    """Optimisation limits beyond the end of their grid (1e30 eV) act as limits at its end: they read back as that
    and give the same spectra. (AM3 turned them into an index past the end of the grid.)"""
    out = []
    for beyond in (True, False):
        a = new(init=False)
        for name, grid in OPTIMISATION_LIMITS.items():
            top = getattr(a, f'get_egrid_{grid}')()[-1]
            getattr(a, 'set_optimize_' + name)(1e30 if beyond else top)
        limits = [getattr(a, 'get_optimize_' + name)() for name in OPTIMISATION_LIMITS]
        quiet(a.init_kernels)
        rich(a, 3)
        out.append((limits, a.get_photons(), a.get_pairs(), a.get_protons()))
    assert out[0][0] == out[1][0]
    for beyond, top in zip(out[0][1:], out[1][1:]):
        assert np.all(np.isfinite(beyond)) and beyond.max() > 0
        np.testing.assert_array_equal(beyond, top)


def test_target_photon_limits_lowered_after_init(rich_run):
    """Target-photon limits lowered after init_kernels act as the same limits set before it: the photons above them
    count as zero (100 eV, where all three change the spectra). The two agree to rounding, as kernels built over
    fewer target photons can group the same terms differently. (AM3 multiplies the kernels of init_kernels by the
    shorter photon vector, past its end.) The build with bounds checks catches a read past the end of the
    Bethe-Heitler photon vector through an unchecked pointer."""
    spectra = lambda a: [a.get_photons(), a.get_pairs(), a.get_neutrinos()]
    out = []
    for after_init in (False, True):
        a = new(init=after_init)
        for name in ('photopion_target_photon_max', 'bethe_heitler_target_photon_max', 'compton_target_emax'):
            getattr(a, 'set_optimize_' + name)(1e2)  # eV
        if not after_init:
            quiet(a.init_kernels)
        out.append(spectra(rich(a, 5)))
    before, after = out
    for x, y in zip(before, after):
        np.testing.assert_allclose(y, x, rtol=1e-13, atol=0)
    assert not all(np.array_equal(x, z) for x, z in zip(before, spectra(rich_run)))


# ---------------------------------------------------------------- the interface
def test_evolve_needs_init_kernels():
    """evolve_step() raises before init_kernels(), after update_energy_grid() without a new init_kernels() (the
    kernels are for the old grid), and after switching on the hadronic processes that were off at init_kernels()."""
    a = new(init=False)
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
    b.set_process_hadronic(True)
    with pytest.raises(Exception, match='init_kernels'):
        b.evolve_step()


def test_pp_transition_energy_above_the_hadron_grid():
    """pp collisions need the transition energy of their delta approximation on the hadron grid (above its top AM3
    crashes, and without the check the pp pions get no energy at all)."""
    a = new(pp=True, init=False)
    a.set_pp_transition_energy_to_delta_approx(1e24)
    with pytest.raises(Exception):
        quiet(a.init_kernels)


@pytest.mark.parametrize('grid', [(-1.0, 1e2, 1e21), (1e6, 1e2, 1e21), (1e-6, -1.0, 1e21), (1e-6, 1e10, 1e21),
                                  (1e-6, 1e2, 1e5), (1e-6, 1e2, float('inf'))])
def test_invalid_energy_grid(grid):
    a = new(init=False)
    n = a.get_egrid_had().size
    with pytest.raises(Exception):
        a.update_energy_grid(*grid)
    assert a.get_egrid_had().size == n


def test_larger_hadron_grid_with_pp():
    """A hadron grid larger than the default one, with pp collisions (AM3's shifted grid of the pp kernel kept the
    default size and was written past its end)."""
    a = rich(new(pp=True, grid=(1e-4, 1e2, 1e23)), 3)
    assert a.get_egrid_had().size > 314
    assert np.all(np.isfinite(a.get_pions())) and a.get_pions().max() > 0


def hadrons_on_photons(e_gamma_min, photopion, e_max=1e21):
    """One step of protons on the grids of update_energy_grid(e_gamma_min, 1e2, e_max) (the default grids for
    e_gamma_min = None): on a photon line at 5 eV with photo-pion production only, or in 1 kG with proton synchrotron
    radiation only."""
    a = new(off=('bethe_heitler', 'pp', 'pion_decay', 'pion_syn', 'pion_compton', 'kaon_decay', 'kaon_syn',
                 'kaon_compton', 'proton_compton'), photopion=photopion, proton_syn=not photopion,
            grid=None if e_gamma_min is None else (e_gamma_min, 1e2, e_max))
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
    cooling by synchrotron radiation much faster than the time step, are q (dx/2) t_syn there: those injected into
    the upper half of the bin's cell. (AM3 dropped the particles in the 6 highest evolved bins every step. Its
    semi-analytic regime also evolved the highest one as n + q dt, the top bin's cooling rate being zero: with only
    the first bug fixed, that bin's synchrotron photons fed a gamma-gamma pair cascade there that grew without
    bound.)"""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC + ('electron_syn',), escape=True, dt=1.0)
    a.set_escape_timescale(10.0)
    for species in ('electrons', 'photons'):
        n = getattr(a, 'get_' + species)().size
        n0 = np.zeros(n)
        n0[n - 30:] = 1.0
        getattr(a, 'set_current_densities_' + species)(n0)
        quiet(a.evolve_step)
        n1 = getattr(a, 'get_' + species)()
        assert n1[n - 30] > 0.5
        np.testing.assert_allclose(n1[n - 8:n - 1], n1[n - 30], rtol=1e-12)
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC, escape=False, dt=300.0)
    a.set_mag_field(100.0)
    n = a.get_egrid_lep().size
    for j in (n - 4, n - 3, n - 2):
        q = np.zeros(n)
        q[j] = 1.0
        a.set_injection_rate_electrons(q)
        a.set_current_densities_electrons(np.zeros(n))
        quiet(a.evolve_step)
        assert a.get_electrons()[j] == pytest.approx(0.05 * a.get_t_pair_syn()[j], rel=1e-2), j


def test_photons_on_the_hadron_grid_after_update_energy_grid():
    """update_energy_grid(1e-6, ...) and update_energy_grid(1.05e-6, ...) make the same grids, so the same protons
    give the same proton-synchrotron photons. (For most E_gamma,min, every power of ten among them, AM3 put the
    photons of the hadron grid one bin below where the shift between the grids takes them: the photons that hadrons
    emit landed e^0.1 too high. For the photo-hadronic side, see test_photopion_on_a_larger_hadron_grid.)"""
    a, b = hadrons_on_photons(1e-6, False), hadrons_on_photons(1.05e-6, False)
    np.testing.assert_array_equal(a.get_egrid_photons(), b.get_egrid_photons())
    np.testing.assert_array_equal(a.get_egrid_had(), b.get_egrid_had())
    assert a.get_photons_protons_syn().max() > 0
    np.testing.assert_allclose(a.get_photons_protons_syn(), b.get_photons_protons_syn(), rtol=1e-12)


def test_photopion_on_a_larger_hadron_grid():
    """update_energy_grid to a hadron grid larger than the default one (up to 1e24 eV) before init_kernels: in the
    bins both grids have, the same protons and photons give the photo-pion rates and pions of the default grid. (The
    three arrays of AM3's photon-convolved photo-pion kernels kept the default grid's size and were written past
    their end: the C++ crashed in init_kernels.) It also tests the fix of the photons on the hadron grid after
    update_energy_grid for photo-hadronic processes: for E_gamma,min = 1e-6 eV, AM3 put those photons one bin too
    low, so that these processes saw them at e^-0.1 of their energy."""
    a, b = hadrons_on_photons(None, True), hadrons_on_photons(1e-6, True, e_max=1e24)
    n = a.get_egrid_had().size
    assert b.get_egrid_had().size > n
    np.testing.assert_allclose(b.get_egrid_had()[:n], a.get_egrid_had(), rtol=1e-12)
    np.testing.assert_allclose(b.get_t_proton_photopion()[:n], a.get_t_proton_photopion(), rtol=1e-10)
    for getter in ('get_pion_plus', 'get_pion_minus', 'get_kaon_plus'):
        assert getattr(a, getter)().max() > 0
        np.testing.assert_allclose(getattr(b, getter)()[:n], getattr(a, getter)(), rtol=1e-10, atol=0)


@pytest.mark.parametrize('setter', ['set_current_densities_protons', 'set_injection_rate_electrons',
                                    'set_t_proton_extra_cooling', 'set_current_densities_photons'])
def test_arrays_of_the_wrong_length_are_rejected(setter):
    a = new(init=False)
    with pytest.raises(Exception, match='elements'):
        getattr(a, setter)(np.ones(7))
    with pytest.raises(Exception):
        getattr(a, setter)(np.ones((2, 400)))


def test_integer_parameters():
    a = new(init=False)
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
    (They read arrays that AM3 allocates in init_kernels, and crashed before it.)"""
    a = new(init=False)
    t = getattr(a, getter)()
    grid = a.get_egrid_photons() if 'photon' in getter else a.get_egrid_lep()
    assert t.size == grid.size and np.all(np.isfinite(t))


def test_neutrino_timescales_cover_the_neutrino_grid():
    """The neutrino escape and extra-cooling timescales have a value for every bin of the neutrino grid (AM3 returned
    only the hadron grid's length of them)."""
    a = new(init=False)
    n = a.get_egrid_neutrinos().size
    assert n > a.get_egrid_had().size
    for g in ('get_t_neutrino_escape', 'get_t_neutrino_extra_cooling'):
        assert getattr(a, g)().size == n, g


def test_profile_timing_off_by_default(rich_run):
    """The profiler is off unless switched on (profile_timing is uninitialised in AM3): no times are taken in
    rich_run."""
    assert rich_run.get_profile_timing() is False
    assert not any(rich_run.get_profiled_times())


# ---------------------------------------------------------------- e+e- annihilation
def test_annihilation_of_the_pair_components():
    """With e+e- annihilation, the pair components are evolved per charge. The injected component (only electrons)
    annihilates at the electrons' rate, dn/dt = q - Gamma_- n, while the gamma-gamma pairs annihilate on the positrons
    too. The components add up to the total. (Evolving the component as a whole at a mean rate puts 40 % too few
    pairs into it.)"""
    dt = 1.0
    a = new(off=('hadronic', 'escape', 'electron_syn', 'electron_compton', 'ssa', 'proton_syn', 'expansion',
                 'adiabatic_cooling', 'triplet_pair_production'), dt=dt)
    El, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    il = np.arange(El.size)
    q = np.where((il >= 2) & (il <= 60), 1e12, 0.0)  # electrons only
    a.set_injection_rate_electrons(q)
    a.set_current_densities_photons(np.where(np.abs(np.log(Eg / MEC2_EV / 30)) < 0.35, 3e14, 0.0))
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
    """The positrons injected in a step annihilate on the electrons there at its start, also when there were no
    positrons at the start (the first positrons of a run): each charge's sink comes from the other's density, while
    the photon source needs both. (With both sinks zero when either charge is absent, the timescale getters report
    3e177 s.)"""
    dt, q = 1e3, 1e9
    a = new(off=CLOSED_BOX + ('triplet_pair_production',), dt=dt)
    il = np.arange(a.get_egrid_lep().size)
    a.set_current_densities_electrons(np.where(il == 2, 1e12, 0.0))
    a.set_injection_rate_positrons(np.where(il == 2, q, 0.0))
    a.evolve_step()
    t = a.get_t_positron_annihilation()[2]
    assert t < dt * 10
    # no cooling or escape: the solver's escape-dominated regime, dn/dt = q - n/t exactly
    np.testing.assert_allclose(a.get_positrons()[2], q * t * (1 - np.exp(-dt / t)), rtol=1e-2)
    # no sink for the electrons: there were no positrons at the start of the step
    assert a.get_electrons()[2] == 1e12
    assert np.all(a.get_photons_pair_annihilation() == 0.0)
