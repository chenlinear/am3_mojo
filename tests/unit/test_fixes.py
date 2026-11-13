"""Regression tests of the bugs that am3_mojo fixes: AM3's and the port's own.

They cover every fix of tests/reference/README.md ("Bugs of AM3 v1.2.0 that
am3_mojo fixes") except two: the OpenMP race (tests/unit/test_threads.py) and
the energy of the pairs of the full gamma-gamma kernel
(tests/unit/test_kept_approximations.py). Each test fails if any one of its
fixes is reverted. The tests of reads and writes past the end of an array fail
at least in the build with bounds checks (make test-asserts). Their effect
depends on the memory past the end. The exception is the inverse-Compton
cooling of the hadrons (see its test).
"""

import os, subprocess, sys

import mpmath as mp
import numpy as np
import pytest

from support import harness
from support.compare import significant
from support.output import quiet
from support.scenarios import dump_state, leptohadronic_params, sphere_volume
from support.setups import (
    ADDED_LEPTONIC,
    CLOSED_BOX,
    LEPTONS_ALONE,
    PAIR_COMPONENTS,
    energy,
    evolve,
    line,
    new,
    number,
    per_ln_e,
    power_law,
    pulse_source,
)
from support.units import C_CGS, DLNE, MASS_EV, MASS_GEV, MEC2_EV, MP_C2_EV

UNIT = os.path.dirname(os.path.abspath(__file__))


def rich(a, steps=0):
    """In 100 G, all hadronic processes matter."""
    leptohadronic_params(a, bfield=100.0, time_step_frac=1e-2)
    return evolve(a, steps)


# ------------------------------------------------------------ the solver
def const_cool(t_cool_steps, below=None):
    a = new(photopion=False, bethe_heitler=False, pp=False)
    R = 1e15
    dt = 1e-2 * R / C_CGS
    a.set_mag_field(1e-3)
    a.set_escape_timescale(R / C_CGS)
    a.set_solver_time_step(dt)
    a.set_powerlaw_injection_parameters_protons(
        sphere_volume(R),
        1e44,
        1e1 * MP_C2_EV,
        1e1 * MP_C2_EV,
        1e8 * MP_C2_EV,
        2.0,
        2.0,
        1.0,
    )
    E = a.get_egrid_had()
    if t_cool_steps is not None:
        t = np.full(E.size, t_cool_steps * dt)
        a.set_t_proton_extra_cooling(
            t if below is None else np.where(E > 1e14, t, below)
        )
    evolve(a, 30)
    return E, a.get_protons()


@pytest.fixture(scope="module")
def uncooled_protons():
    return const_cool(None)


def test_fast_cooling_above_slow_cooling(uncooled_protons):
    """The solution of a bin in a fast-cooling window involves only the bins
    above it. So the solution must not depend on how slowly the bins below
    cool. AM3 summed the cooling time from the lowest of the 10 padding cells
    of the window. These cells are bins of other regimes or, below the grid,
    cells with A = 0 (bin 0: A = -1e-150). Where those cells cool about 1e15
    times more slowly, this sum swamps the cooling times of the window in
    double precision. The densities were then 28 % too low at 1e16, and from
    1e17 on there was no cooling. This hits cooling faster than an e-fold per
    step that is not synchrotron-like. Here, 1e20 s in the lowest cell left the
    protons uncooled: 1.16 times the density without cooling instead of
    0.011."""
    E, n0 = uncooled_protons
    i = np.argmin(np.abs(E - 1.6e14))
    n15, n20 = (
        const_cool(0.3, below=1e15)[1][i],
        const_cool(0.3, below=1e20)[1][i],
    )
    assert n20 == pytest.approx(n15, rel=1e-9)
    assert 0.005 * n0[i] < n20 < 0.05 * n0[i]


def electrons_box(dt, t_esc, B=0.0, t_cool=None, rk4=False):
    a = new(
        off=LEPTONS_ALONE + ADDED_LEPTONIC,
        electron_syn=t_cool is None,
        dt=dt,
        init=False,
    )
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
    """t_cool(x) int_x^min(x + reach, top) q(x') exp(-int_x^x' t_cool / t_esc
    dx'') dx' (x = ln E). As in the solver, q and ln t_cool are linear between
    grid points, and the integral is exact on sub-intervals."""
    with np.errstate(all="ignore"):
        sub = 50
        x = np.log(E)
        xs = np.linspace(x[0], x[-1], (len(x) - 1) * sub + 1)
        lk = np.interp(xs, x, np.log(t_cool / t_esc))
        qs = np.interp(xs, x, q)
        h, d = np.diff(xs), np.diff(lk)
        D = (
            h
            * np.exp(lk[:-1])
            * np.where(
                np.abs(d) > 1e-12, np.expm1(d) / np.where(d == 0, 1, d), 1.0
            )
        )
        Ds = np.maximum(D, 1e-300)
        m0 = np.where(D > 1e-12, -np.expm1(-D) / Ds, 1.0 - D / 2)
        m1 = np.where(
            D > 1e-3, (1.0 - (1.0 + D) * np.exp(-D)) / Ds**2, 0.5 - D / 3
        )
        f = h * (qs[:-1] * m0 + (qs[1:] - qs[:-1]) * m1)
        n = np.zeros(len(x))
        reach = np.broadcast_to(np.minimum(reach, 1e9), x.shape)
        for i in range(len(x)):
            j0, j1 = (
                i * sub,
                min(
                    (len(x) - 1) * sub,
                    int(round((x[i] + reach[i] - x[0]) / h[0])),
                ),
            )
            K = np.concatenate([[0.0], np.cumsum(D[j0:j1])])
            n[i] = t_cool[i] * np.sum(f[j0:j1] * np.exp(-K[:-1]))
        return n


def cooling_reach(E, t_cool, dt):
    """As in the solver, ln t_cool is linear between grid points."""
    x = np.log(E)
    xs = np.linspace(x[0], x[-1], (len(x) - 1) * 50 + 1)
    lt = np.interp(xs, x, np.log(t_cool))
    d = np.diff(lt)
    phi = np.where(np.abs(d) > 1e-12, np.expm1(d) / np.where(d == 0, 1, d), 1)
    T = np.concatenate([[0.0], np.cumsum(np.diff(xs) * np.exp(lt[:-1]) * phi)])
    return np.interp(np.interp(x, xs, T) + dt, T, xs) - x


def test_source_integral_where_escape_dominates():
    """In regime 3, the steady state with synchrotron cooling and escape must be
    that of the continuous equation. AM3's quadrature of the source integral
    (simple_integral_q) gave densities too low by z/(e^z - 1), z = dx a/|A|.
    This factor is 0.08 at 1 GeV and 7e-4 right above regime 1, where escape is
    100 times faster than cooling. Where cooling dominates, the densities were
    too high. The test leaves out the top cell of the escape-dominated window
    below. The particles that cool into this cell escape near its upper edge. So
    the cell holds the right number of particles, not the density at its
    center."""
    t_esc = 1e4
    a, E, q = electrons_box(1e4 * t_esc, t_esc, B=1.0)
    evolve(a, 6)
    t_syn = a.get_t_pair_syn()
    exact = source_integral(E, q, t_syn, t_esc)
    sel = (E >= 3e8) & (E <= 5e12)
    sel[np.nonzero(t_syn > 100 * t_esc)[0].max()] = False
    np.testing.assert_allclose(a.get_electrons()[sel], exact[sel], rtol=1e-2)


@pytest.mark.parametrize("dt", [1e2, 1e3, 1e4])
def test_particles_cooling_into_a_window_after_a_change(dt):
    """After k steps, the electrons number q k dt, as long as none of them
    reaches the bottom of the grid. AM3's 10-cell padding of the windows dropped
    3 % (dt = 100 s) and 25-28 % (1e3 and 1e4 s) within 10-50 steps. The inflow
    into a Chang & Cooper window must be the electrons that cross into it within
    the step. The new densities of the window above add too many in the first
    step."""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC, escape=False, dt=dt)
    a.set_mag_field(30.0)
    a.set_escape_timescale(1e4)
    q = per_ln_e(a.get_egrid_lep())
    a.set_injection_rate_electrons(q)
    for k in range(1, 11):
        a.evolve_step()
        assert np.sum(a.get_electrons()) / (
            np.sum(q) * k * dt
        ) == pytest.approx(1.0, abs=1e-2), k


def cooled_into_a_window(dt, t_cool, steps):
    a = new(
        off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn", "quantum_syn"),
        escape=True,
        dt=dt,
    )
    a.set_escape_timescale(1e4)
    E = a.get_egrid_lep()
    a.set_t_pair_extra_cooling(t_cool(E))
    q = np.where((E >= 1e10) & (E <= 1e12), 1e9 / E, 0.0)
    a.set_injection_rate_electrons(q)
    evolve(a, steps)
    return np.sum(a.get_electrons()) / (np.sum(q) * 1e4)


@pytest.mark.parametrize("dtf", [1e-2, 0.1, 1.0])
def test_particles_cooling_into_an_escape_dominated_window(dtf):
    """Electrons that cool down to where their cooling stops pile up there and
    escape. In the steady state, their number is Q t_esc. (AM3 gave the
    escape-dominated windows no inflow. It dropped the particles that cooled
    into one from the window above: 72 % here. am3_mojo keeps them to 0.4 %.)"""
    n = cooled_into_a_window(
        dtf * 1e4, lambda E: np.where(E > 1e9, 1e3, 1e12), int(round(20 / dtf))
    )
    assert n == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize("dt", [30.0, 100.0, 300.0])
def test_particles_cooling_fast_into_a_slow_window(dt):
    """In the steady state, the electrons number Q t_esc. (The inflow into the
    Chang & Cooper window must not take the densities of the half cell above its
    edge from both cells. That counts the particles piled up in the top cell of
    the window again. This gives +12 to +53 % here, and a growth to 1e89 at
    dt = 30 s.)"""
    t_cool = lambda E: np.where(
        E > 1e9, 10.0, np.where(E > 1e9 * np.exp(-1.2), 3000.0, 1e12)
    )
    assert cooled_into_a_window(
        dt, t_cool, int(max(60, 3e5 / dt))
    ) == pytest.approx(1.0, abs=1.5e-2)


def test_particles_cooling_in_from_bins_that_alternate():
    """The window below bin 150 is padded with the bins from which electrons
    reach it within the step. With ln t_cool linear between grid points, as in
    the solver, these are about 49 bins. Their cooling times add up to 1e3 s.
    The electrons there are then those of the source integral up to where they
    come from (source_integral with cooling_reach). (Summed as dx / |A| per bin,
    the cooling time reached 1e3 s within 22 bins. The padding stopped there,
    and the top bins of the window got 7-51 % too few electrons, 28 % in
    all.)"""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn",), dt=1e3)
    E = a.get_egrid_lep()
    k = np.arange(E.size)
    t_cool = np.where(k < 150, 500.0, np.where(k % 2 == 0, 952.4, 9.524))
    t_esc = np.where(k < 150, 5e3, 1e30)
    q = np.where((k >= 150) & (k <= 250), 1.0, 0.0)
    a.set_t_pair_extra_cooling(t_cool)
    a.set_t_pair_escape(t_esc)
    a.set_injection_rate_electrons(q)
    a.evolve_step()
    exact = source_integral(
        E, q, t_cool, t_esc, reach=cooling_reach(E, t_cool, 1e3)
    )
    sel = (k < 150) & (exact > 0.1 * exact[k < 150].max())
    assert sel.sum() >= 10
    np.testing.assert_allclose(a.get_electrons()[sel], exact[sel], rtol=1e-2)


def test_particles_cooling_in_across_a_slow_band():
    """The bins below the band are a cooling-only window. The bins of that
    window hold the source integral up to where their electrons come from,
    226 bins above them (source_integral with cooling_reach). The padding of the
    window must reach there. The padding counts its cooling time from the center
    of the top bin of the window, as the window does. (Counted from that bin's
    upper edge, the first half cell with the band's cooling time, the padding
    ended at bin 218 instead of 327. The window then got 0.53 to 0.86 times its
    electrons.)"""
    dt = 1e3
    a = new(off=CLOSED_BOX + ADDED_LEPTONIC, dt=dt)
    E = a.get_egrid_lep()
    k = np.arange(E.size)
    t_cool = np.where((k >= 100) & (k <= 105), 1667.0, 5.0)
    q = np.where((k >= 50) & (k <= 335), 1.0, 0.0)
    a.set_t_pair_extra_cooling(t_cool)
    a.set_injection_rate_electrons(q)
    a.evolve_step()
    exact = source_integral(
        E, q, t_cool, np.full(E.size, 1e300), cooling_reach(E, t_cool, dt)
    )
    sel = (k >= 20) & (k < 100)
    np.testing.assert_allclose(a.get_electrons()[sel], exact[sel], rtol=1e-3)


def test_particles_cooling_out_of_an_escape_dominated_window():
    """The cooling changes by 2 %, so the steady state is smooth across bin 150.
    (The Chang & Cooper window got no inflow from the escape-dominated window
    above, whose particles cool into it at |A| n / dx. The top cell of the
    Chang & Cooper window was 9 % low.)"""
    a = new(
        off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn", "quantum_syn"),
        dt=1e3,
    )
    a.set_escape_timescale(1e4)
    i = np.arange(a.get_egrid_lep().size)
    a.set_t_pair_extra_cooling(np.where(i < 150, 0.99e6, 1.01e6))
    a.set_injection_rate_electrons(np.where((i >= 60) & (i <= 220), 1.0, 0.0))
    e = evolve(a, 400).get_electrons()
    assert e[149] / e[150] == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize("dt", [1e3, 1e4])
def test_particles_escaping_fast_above_a_chang_cooper_window(dt):
    """The electrons cool across the lower edge of bin 150 at
    q t_esc / t_cool per s. After 2e5 s, the bins below hold that many to 1e-4.
    The inflow counts the electrons that cross within the step, attenuated by
    e^-S (S: the escape exponent from the edge). It counts them up to S = 60, in
    pieces over which S grows by at most 1 on average. (With at most 64 pieces
    over the cells within reach, the inflow counts 0.948 and 5.3e-4 of them at
    dt = 1e3 and 1e4 s. Across these cells, S grows by 1e3 and 1e4.)"""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn",), dt=dt)
    E = a.get_egrid_lep()
    k = np.arange(E.size)
    a.set_t_pair_extra_cooling(np.full(E.size, 1e6))
    a.set_t_pair_escape(np.where(k >= 150, 1.0, 1e30))
    a.set_injection_rate_electrons(np.where((k >= 150) & (k <= 170), 1.0, 0.0))
    n = evolve(a, int(round(2e5 / dt))).get_electrons()
    assert number(n[:150]) == pytest.approx(1.0 / 1e6 * 2e5, rel=1e-4)


def test_steep_injection_tail_in_regime_4():
    """Above the energy that the cooling reaches within the step, one step from
    empty gives the steady state n |A| = int_x^top q dx. The test compares the
    ratio of the two sides with its value at 3 GeV. This removes the units of
    the cooling-time getter. AM3 accumulated int q dx from the bottom of the
    window. It restarted the sum where q dx / sum < 1e-15 (twice here). The bins
    below each restart kept almost no significant digits of their injection. The
    bins from 1.0e11 eV (q / q_max < 1e-47) on kept none (n = 0)."""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC, escape=False, dt=1e5)
    a.set_mag_field(30.0)
    E = a.get_egrid_lep()
    q = np.where(E >= 1e8, 1e8 / E * np.exp(-E / 1e9), 0.0)
    a.set_injection_rate_electrons(q)
    a.evolve_step()
    flux = a.get_electrons() / a.get_t_pair_syn()
    above = np.concatenate(
        [np.cumsum((0.5 * (q[:-1] + q[1:]) * DLNE)[::-1])[::-1], [0.0]]
    )
    sel = (E >= 3e9) & (E <= 1.5e11)
    ratio = flux[sel] / above[sel]
    np.testing.assert_allclose(ratio / ratio[0], 1.0, rtol=1e-8)


@pytest.mark.parametrize("rk4, rtol", [(False, 2e-3), (True, 2e-2)])
def test_cooling_only_window_below_escape_dominated_bins(rk4, rtol):
    """Below 10^9.5 eV, the electrons cool 1e5 times faster than they escape
    (regime 4). Their steady state is the electrons that cool down from the bins
    above, attenuated by the escape there (source_integral). The padding of that
    cooling-only window holds those escape-dominated bins, and the window takes
    their escape. After 10 steps, the electrons from 1e8 to 1e10 eV are within
    2e-3 of the steady state. With the RK4 source integral
    (set_solver_simpleqintegration), they are within 2e-2. (AM3 solved the
    window and its padding without escape: 1.61 times the steady state below
    10^9.4 eV, 1.25 times from there to 1e10 eV. Its RK4 integral leaves out the
    sink where (y - x) a/|A| < 1e-3 at x, the bin itself. With a/|A| taken there
    alone, this gives 1.57 to 1.59 times the steady state.)"""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn",), dt=1e3)
    a.set_solver_simpleqintegration(rk4)
    E = a.get_egrid_lep()
    f = np.clip(np.log10(E) - 9.5, 0.0, 1.0)
    t_cool, t_esc = 10.0 * 1e3**f, 1e6 * 1e-5**f
    q = np.exp(-0.5 * (np.log(E / 10**10.2) / 0.3) ** 2)
    a.set_t_pair_extra_cooling(t_cool)
    a.set_t_pair_escape(t_esc)
    a.set_injection_rate_electrons(q)
    n = evolve(a, 10).get_electrons()
    exact = source_integral(E, q, t_cool, t_esc)
    sel = (E > 1e8) & (E < 1e10)
    np.testing.assert_allclose(n[sel], exact[sel], rtol=rtol)


def test_cooling_only_window_below_a_sharp_escape_step():
    """After 10 steps, the electrons below bin 150 are within 5e-3 of the steady
    state (source_integral). The cooling-only window takes the escape of its
    padding and its own. Between the bins, ln(a/|A|) is linear. The top bin of
    the window got a = 1e-150 instead of its escape. So the cell up to bin 150
    lost most of the escape of bin 150. This gave 1.31 times the steady state
    (1.05 to 1.59 times for an escape 10 to 1e4 times faster than cooling)."""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn",), dt=1e3)
    E = a.get_egrid_lep()
    k = np.arange(E.size)
    t_cool = np.full(E.size, 10.0)
    t_esc = np.where(k < 150, 1e6, 1e-2)
    q = np.where((k >= 150) & (k <= 170), 1.0, 0.0)
    a.set_t_pair_extra_cooling(t_cool)
    a.set_t_pair_escape(t_esc)
    a.set_injection_rate_electrons(q)
    n = evolve(a, 10).get_electrons()
    exact = source_integral(E, q, t_cool, t_esc)
    sel = (k >= 20) & (k < 150)
    np.testing.assert_allclose(n[sel], exact[sel], rtol=5e-3)


def test_escape_of_a_cooling_only_window_set_by_the_rates():
    """The electrons that reach the cooling-only window within the step come
    from up to 10 e-folds above it, across 1e11 eV. So the window takes the
    escape of its padding and its own. The test adds a second source of 1e-60
    per ln E above 1e11 eV, up to 1e13 eV. This source must not change the
    electrons below 1e9 eV in the first 3 steps. The rates of the padding decide
    whether the window takes the escape, whether or not the padding holds
    particles. (Decided by the occupied cells alone, the weak source switched
    the escape on: the electrons below 1e9 eV changed by up to 5.0 %.)"""

    def electrons(weak):
        a = new(off=CLOSED_BOX + ADDED_LEPTONIC, escape=True, dt=1e3)
        E = a.get_egrid_lep()
        a.set_t_pair_extra_cooling(np.full(E.size, 100.0))
        a.set_t_pair_escape(np.where(E < 1e11, 1.2e4, 10.0))
        q = np.where((E >= 1e8) & (E <= 1e9), 1.0, 0.0)
        a.set_injection_rate_electrons(
            q + np.where((E > 1e11) & (E <= 1e13), weak, 0.0)
        )
        return E, [evolve(a, 1).get_electrons() for _ in range(3)]

    E, n = electrons(0.0)
    sel = E <= 1e9
    for n1, n2 in zip(n, electrons(1e-60)[1]):
        assert n1[sel].max() > 0
        np.testing.assert_allclose(n2[sel], n1[sel], rtol=1e-12)


def test_escape_on_the_way_into_a_cooling_only_window():
    """The bins from E_b = 1e10 eV up are the padding of the cooling-only
    window, which the window takes with its escape. A bin x below E_b then holds
    the electrons from y = x + 2 e-folds, attenuated by exp(-int_x^y a/|A| dx).
    Here ln(a/|A|) is linear between the grid points, as in the solver. This
    holds to rounding in the 20 bins below E_b. (AM3 integrated the exponent
    with 5 RK4 steps from x to y. The points of these steps fall differently
    across the jump of the escape at E_b. This gave 0.88 to 1.04 times the exact
    value, in a pattern that repeats every 4 bins, the length of a step.)"""
    dt, t_cool, t_esc = 20.0, 10.0, 10.0
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn",), dt=dt)
    E = a.get_egrid_lep()
    k = np.arange(E.size)
    b = int(np.argmin(np.abs(E - 1e10)))
    a.set_t_pair_extra_cooling(np.full(E.size, t_cool))
    a.set_t_pair_escape(np.where(k < b, 1e6, t_esc))
    a.set_current_densities_electrons(
        np.where((k >= b) & (k <= b + 30), 1.0, 0.0)
    )
    a.evolve_step()
    r = t_cool / 1e6
    i = np.arange(b - 20, b)
    S = (b - 1 - i) * DLNE * r + DLNE * (1 - r) / np.log(1 / r)
    S += (i + 20 - b) * DLNE
    np.testing.assert_allclose(a.get_electrons()[i], np.exp(-S), rtol=1e-12)


@pytest.mark.parametrize("source", ["density", "injection"])
def test_escape_exponent_of_a_window_above_fast_escape(source):
    """The bins 5 to 35 above E_b = 1e10 eV hold the electrons from 2 e-folds
    above them, to rounding. That is e^-2 of them (density), or what the bins
    hold with an escape of 1e6 s below E_b (injection). The escape exponent of
    the window from E_b up starts at its first bin, because the cells below
    never feed the window. (The escape exponent was summed from the lowest cell
    below the window. The exponent of that cell is 1e15 times that of the
    window. So the sum kept none of the window's digits: 1 instead of e^-2 at
    1e-15 s, 1.0078 e^-2 at 1e-12 s. The injected electrons came out 0.005 to
    6.7 times their number at 1e-18 s.)"""

    def electrons(t_below):
        a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn",), dt=20.0)
        E = a.get_egrid_lep()
        k = np.arange(E.size)
        b = int(np.argmin(np.abs(E - 1e10)))
        a.set_t_pair_extra_cooling(np.full(E.size, 10.0))
        a.set_t_pair_escape(np.where(k < b, t_below, 10.0))
        n = np.where((k >= b + 25) & (k <= b + 55), 1.0, 0.0)
        if source == "density":
            a.set_current_densities_electrons(n)
        else:
            a.set_injection_rate_electrons(n)
        a.evolve_step()
        return a.get_electrons()[b + 5 : b + 36]

    n = electrons(1e-15)
    ref = np.full(n.size, np.exp(-2.0))
    if source == "injection":
        ref = electrons(1e6)
        assert np.all(ref > 0)
    np.testing.assert_allclose(n, ref, rtol=1e-12)


def one_step(t_cool, t_esc, escape=None, q=None, rk4=False):
    a, E, q0 = electrons_box(3 * t_cool, t_esc, t_cool=t_cool, rk4=rk4)
    if escape:
        t_esc = escape(E)
        a.set_t_pair_escape(t_esc)
    if q:
        q0 = q(E)
        a.set_injection_rate_electrons(q0)
    a.evolve_step()
    return (
        E,
        a.get_electrons(),
        source_integral(E, q0, np.full(E.size, t_cool), t_esc, reach=3.0),
    )


@pytest.mark.parametrize(
    "t_cool, k, rk4",
    [
        (1e3, 5.0, False),
        (1e3, 50.0, False),
        (1e3, 0.05, True),
        (1e-5, None, False),
    ],
    ids=["escape-5x", "escape-50x", "escape-0.05x-rk4", "escape-vanishing"],
)
def test_source_integral_of_one_step(t_cool, k, rk4):
    """One step of dt = 3 t_cool from empty, with an energy-independent cooling
    time t_cool. Every bin holds the electrons injected at most 3 e-folds above
    it.
    * Escape in t_cool / k (regime 3): AM3's quadrature gave n/exact = 0.81 and
      0.035 for k = 5 and 50. Where the sink is weak, its RK4 option
      (set_solver_simpleqintegration) integrated beyond those 3 e-folds, up to
      the top of the solver's window. This gave +4.4 % for k = 0.05.
    * t_cool = 10 us, escape in t_cool below 1e11 eV and none above. Most
      electrons in the top 3 e-folds below 1e11 eV come from above 1e11 eV.
      AM3's simple_integral_q skipped the points where alpha/|A| < e^-300 with
      their injection: 46-90 % too few electrons in the 6 bins below 1e11 eV.
      Both codes add 1e-150 to alpha in the solver's units (time in units of
      2.7e-23 s for the e+-). So without escape, this needs cooling in less
      than about 1.4 ms. The case also needs the semi-analytic windows padded
      up to where particles reach them within the step. AM3's 10 cells gave up
      to 34 % too few electrons below 1e11 eV."""
    escape = None if k else lambda E: np.where(E < 1e11, t_cool, 1e250)
    E, n, exact = one_step(t_cool, t_cool / k if k else t_cool, escape, rk4=rk4)
    sel = (E >= 2e8) & (E <= 5e12)
    np.testing.assert_allclose(n[sel], exact[sel], rtol=5e-3)


def test_source_integral_of_a_steeply_rising_source():
    """A source integral of regime 3 can be taken as the difference of two
    integrals up to the top of the window. This difference loses its digits
    where the electrons injected further up dominate: up to 45 % too many
    electrons at 1e7 eV."""
    t_cool = 1e3
    E, n, exact = one_step(
        t_cool,
        20 * t_cool,
        q=lambda E: np.where((E >= 1e7) & (E <= 1e16), (E / 1e16) ** 2.0, 0.0),
    )
    sel = (E >= 1e7) & (E <= 3e14)
    np.testing.assert_allclose(n[sel], exact[sel], rtol=5e-3)


@pytest.mark.parametrize("t_cool", [0.3, 0.5, 0.9])
def test_cooled_particles_in_regime_4(uncooled_protons, t_cool):
    """In the steady state, the protons' energy content is L t,
    1/t = 1/t_cool + 1/t_esc. A bin x takes the particles n(y) A(y)/A(x) that
    cooling brings from y to x within the step. AM3 subtracted them instead:
    6 %, 23 % and 49 % too little energy for t_cool = 0.3, 0.5 and 0.9 steps."""
    E, n0 = uncooled_protons
    L_tesc = np.sum(n0 * E) / (1 - np.exp(-0.3))
    t = 1 / (1 / t_cool + 1 / 100.0) / 100.0  # in units of t_esc
    n = const_cool(t_cool)[1]
    assert np.sum(n * E) / (L_tesc * t) == pytest.approx(1.0, rel=0.01)


def rk4_one_step_across_an_escape_step():
    t_cool = 1e3
    E, n, exact = one_step(
        t_cool,
        10 * t_cool,
        lambda E: np.where(E < 1e10, 10 * t_cool, t_cool / 100),
        rk4=True,
    )
    sel = (E >= 2e8) & (E <= 5e12)
    return float(np.max(np.abs(n[sel] / exact[sel] - 1)))


def test_rk4_source_integral_across_an_escape_step():
    """The RK4 option (set_solver_simpleqintegration) with escape 1000 times
    faster above 1e10 eV than below. For the bins up to 0.3 e-folds below that
    energy, the first tenth of the integration range reaches across it. Over
    this first tenth, the sink takes more than e^2 of the particles. So the
    integration narrows its range and widens it again. The widening must update
    the integration step: AM3's loop did not, and never ended. With the update,
    every bin is within 2 % of the exact integral. The case runs in a
    subprocess, so that an endless loop fails this test instead of stopping
    the run."""
    code = (
        f"import sys; sys.path[:0] = [{os.path.dirname(UNIT)!r}, "
        f"{UNIT!r}]\nfrom test_fixes import "
        "rk4_one_step_across_an_escape_step as f\nprint(f())"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the RK4 source integral did not end within 120 s")
    assert r.returncode == 0, r.stderr[-3000:]
    assert float(r.stdout.split()[-1]) < 0.02


def test_rk4_components_add_up():
    """With the RK4 source integral, the injected pairs add up to all pairs to
    rounding, in the bins of the comparison criterion. The injected pairs
    (get_pairs_injected) are one solve of both injections. All pairs are the
    electrons and positrons, each solved apart. The rates set the cells from
    which a window takes the particles that cool into it within the step. So the
    windows of the electrons take the same points of the integral as those of
    the pairs. (Up to the highest bin with particles only, the electrons' window
    took fewer cells: up to 1.1e-2 apart.)"""
    a = new(
        off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn",),
        dt=30.0,
        init=False,
    )
    a.set_solver_simpleqintegration(True)
    quiet(a.init_kernels)
    E = a.get_egrid_lep()
    k = np.arange(E.size)
    b = int(np.argmin(np.abs(E - 1e10)))
    a.set_t_pair_extra_cooling(np.full(E.size, 10.0))
    a.set_t_pair_escape(np.full(E.size, 100.0))
    a.set_injection_rate_electrons(np.where((k >= b) & (k <= b + 20), 1.0, 0.0))
    a.set_injection_rate_positrons(
        np.where((k >= b + 60) & (k <= b + 80), 1.0, 0.0)
    )
    for _ in range(3):
        pairs = evolve(a, 1).get_pairs()
        sig = significant(pairs, 1e-12, E) & (pairs > 0)
        assert sig.sum() >= 50
        np.testing.assert_allclose(
            a.get_pairs_injected()[sig], pairs[sig], rtol=1e-12
        )


def fast_band(n_fast, above, steps):
    a = new(off=CLOSED_BOX + ADDED_LEPTONIC, escape=True, dt=1e3)
    a.set_escape_timescale(1e4)
    E = a.get_egrid_lep()
    q = np.where((E >= 1e9) & (E <= 1e12), 1.0, 0.0)
    top = np.nonzero(q)[0].max()
    t = np.full(E.size, 1e12)
    t[top - n_fast + 1 : None if above else top + 1] = 10.0
    a.set_t_pair_extra_cooling(t)
    a.set_injection_rate_electrons(q)
    return q, top, evolve(a, steps).get_electrons()


def test_fast_cooling_below_a_slow_cell():
    """After 10 escape times, the electrons number between 0.9 Q t_esc and
    Q t_esc, the steady state without cooling. The inverse of the cumulative
    cooling time is exact in each cell for the solver's 1/|A| = e^L, L linear
    across the cell. (AM3 interpolated this inverse linearly in the cooling
    time. Across the cell above the top fast bin, 1/|A| rises 1e11 times. So the
    top fast bin took its particles from next to itself and kept them and its
    injection without cooling. This bin fed the bins below: 12 Q t_esc after 10
    escape times, 50 after 40.)"""
    q, top, n = fast_band(8, False, 100)
    assert 0.9 * q.sum() * 1e4 < n.sum() < q.sum() * 1e4


def test_cooling_time_across_a_steep_cell():
    """A bin x below 1e10 eV holds the electrons injected between x and y,
    n = t(x) q (y - x). Here y is the energy from which they cool to x within
    the step. They cool across the cell below 1e10 eV, in which ln t rises by
    25.3. The solver sums the cells' cooling times exactly for its ln t, linear
    across a cell, so n is right to rounding. (AM3's 10 RK4 steps per cell take
    1.2 % too long across that cell: 2.3e-5 too few electrons in these bins.)"""
    dt, t = 1e3, 500.0
    a = new(off=CLOSED_BOX + ADDED_LEPTONIC, dt=dt)
    E = a.get_egrid_lep()
    k = np.arange(E.size)
    b = int(np.argmin(np.abs(E - 1e10)))
    t_fast, s = t / 1e11, np.log(1e11)
    a.set_t_pair_extra_cooling(np.where(k < b, t_fast, t))
    q = np.where((E >= 1e8) & (E <= 1e13), 1.0, 0.0)
    a.set_injection_rate_electrons(q)
    a.evolve_step()
    sel = (E >= 1e9) & (k < b)
    y_x = (b - k[sel]) * DLNE + (
        dt - (b - 1 - k[sel]) * DLNE * t_fast - DLNE * (t - t_fast) / s
    ) / t
    np.testing.assert_allclose(
        a.get_electrons()[sel], t_fast * q[sel] * y_x, rtol=1e-12
    )


def test_short_cooling_window_above_escape_dominated_bins():
    """The 3 escape-dominated bins right below the cooling-only window hold
    about q t_esc after 20 escape times (between 0.5 and 5 q t_esc). (AM3 held
    the cooling-only window to at least 6 bins. It solved the 3 bins below
    without their escape, so they gained q dt every step: 20 q t_esc after 20
    escape times.)"""
    q, top, n = fast_band(3, True, 200)
    below = n[top - 5 : top - 2] / 1e4
    assert np.all((below > 0.5) & (below < 5.0)), below


def test_chang_cooper_window_above_fast_cooling_bins():
    """After one step from empty, the bins from 1e6 eV to E_b = 1e10 eV hold the
    electrons that cool to them within the step. This holds to 5e-3
    (source_integral with cooling_reach), because the Chang & Cooper window ends
    where the regime changes. The test adds a second source of 1e-60 per ln E
    above 1e12 eV, up to 1e13 eV. This source must not change the electrons
    below E_b in the first 3 steps. (The window was held to at least 6 bins from
    its top, the highest bin with particles. So it took in the 3 bins below E_b
    and solved them with its matrix at their Courant number of 200. This gave
    0.38 of their electrons after one step. With the weak source, the window
    started higher up and did not take them in: those bins changed by
    +160 %.)"""

    def electrons(weak):
        a = new(off=CLOSED_BOX + ADDED_LEPTONIC, escape=True, dt=1e3)
        E = a.get_egrid_lep()
        a.set_t_pair_extra_cooling(t_cool(E))
        a.set_t_pair_escape(np.full(E.size, 1e5))
        a.set_injection_rate_electrons(
            q(E) + np.where((E > 1e12) & (E <= 1e13), weak, 0.0)
        )
        return E, [evolve(a, 1).get_electrons() for _ in range(3)]

    t_cool = lambda E: np.where(E >= 1e10, 5000.0, 50.0)
    q = lambda E: np.where((E >= 1e10) & (E <= 1e10 * np.exp(0.25)), 1.0, 0.0)
    E, n = electrons(0.0)
    tc = t_cool(E)
    exact = source_integral(
        E, q(E), tc, np.full(E.size, 1e5), cooling_reach(E, tc, 1e3)
    )
    sel = (E >= 1e6) & (E < 1e10)
    np.testing.assert_allclose(n[0][sel], exact[sel], rtol=5e-3)
    for n1, n2 in zip(n, electrons(1e-60)[1]):
        np.testing.assert_allclose(n2[E < 1e10], n1[E < 1e10], rtol=1e-12)


def below_chang_cooper(q, t_below):
    """The electrons cool in 5000 s per e-fold from E_b = 1e10 eV up (a
    Chang & Cooper window, Courant number 2)."""
    a = new(off=CLOSED_BOX + ADDED_LEPTONIC, escape=True, dt=1e3)
    E = a.get_egrid_lep()
    a.set_t_pair_extra_cooling(np.where(E >= 1e10, 5000.0, t_below))
    a.set_t_pair_escape(np.full(E.size, 1e15))
    a.set_injection_rate_electrons(q(E))
    return E, q(E), evolve(a, 100).get_electrons()


def injection_band(E):
    return np.where((E >= 1e10) & (E <= 1e10 * np.exp(0.25)), 1.0, 0.0)


@pytest.mark.parametrize(
    "q",
    [
        injection_band,
        lambda E: np.where((E >= 1e8) & (E <= 1e13), 1e10 / E, 0.0),
    ],
    ids=["band", "falling"],
)
def test_cooling_only_window_below_a_chang_cooper_window(q):
    """Below E_b, in the steady state, the electrons times their cooling rate
    are the flux of the source, to 1e-6. This flux is the whole injection above
    a bin and half of its own. (The windows below read the Chang & Cooper
    densities as values at the bin centers. But in the steady state, each is the
    flux through the cell's lower face, over the cooling rate. So the windows
    counted the injection between that face and the center twice. This gave
    1.167 times the flux below a band of injection and 1.04 times under the
    falling source. Below the band, a window with escape took in 1.168 times
    the flux. An escape-dominated window let 1.146 times the particles of the
    source escape. The windows of all three regimes read the same densities.)"""
    E, s, n = below_chang_cooper(q, 50.0)
    flux = np.cumsum((s * DLNE)[::-1])[::-1] - 0.5 * s * DLNE
    sel = (E > 1e8) & (E < 1e10)
    np.testing.assert_allclose(n[sel] / 50.0, flux[sel], rtol=1e-6)


def test_top_of_a_falling_spectrum_in_a_chang_cooper_window():
    """In the Chang & Cooper window, each step takes the least implicit theta
    that keeps the term of n in the cell's own row non-negative. So this term is
    zero. Once the source is off, the highest bin with particles loses them all
    in a step. So the top of the spectrum falls by a bin per step. (As the
    difference n - 2 (1 - theta) PD n, the term left 1e-16 n in the bin, at
    4e-18 per ln E next to 4e-4. Such a bin holds nothing else, and its value
    differs between a species and its components. With hadronic processes, when
    the production stopped, the pions differed from the sum of their components
    by up to 7 %. This was in a bin at 1e-14 of the peak.)"""
    a = new(off=CLOSED_BOX + ADDED_LEPTONIC, escape=True, dt=1e3)
    E = a.get_egrid_lep()
    a.set_t_pair_extra_cooling(1429.0 * (E / 1e10) ** -0.3)
    a.set_t_pair_escape(20.0 * (E / 1e10) ** 0.2)
    q = np.where(
        (E > 1e9) & (E < 1e11), np.exp(-0.5 * np.log(E / 1e10) ** 2), 0
    )
    a.set_injection_rate_electrons(q)
    a.set_injection_rate_positrons(0.37 * q)
    evolve(a, 3)
    a.set_injection_rate_electrons(np.zeros(E.size))
    a.set_injection_rate_positrons(np.zeros(E.size))
    tops = []
    for _ in range(6):
        n = evolve(a, 1).get_pairs()
        tops.append(np.flatnonzero(n > 1e-70 * n.max())[-1])
        assert np.all(n[tops[-1] + 1 :] < 1e-50 * n.max())
    assert np.array_equal(np.diff(tops), [-1] * 5), tops


def compton_line(B, photons=True):
    """The electrons cool by inverse Compton only above about 1.1e7 eV, where
    they are more energetic than the photons."""
    a = new(
        off=(
            "hadronic",
            "annihilation",
            "ssa",
            "proton_syn",
            "expansion",
            "adiabatic_cooling",
            "electron_syn_emission",
            "electron_compton_emission",
        )
        + ADDED_LEPTONIC,
        dt=2e8,
    )
    a.set_mag_field(B)
    a.set_escape_timescale(1e10)
    E, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    a.set_injection_rate_electrons(
        np.where((E >= 1e6) & (E <= 1e11), 1e6 / E, 0.0)
    )
    for _ in range(10):
        a.set_current_densities_photons(line(1e10 if photons else 0.0, 1e7)(Eg))
        a.evolve_step()
    return a.get_electrons()


def test_cooling_window_above_bins_that_do_not_cool():
    """The electrons' cooling stops below about 1.1e7 eV, where the electrons
    that cool down to there pile up and escape. Escape takes every energy alike,
    so the electrons number as many as without the photons. They are also as in
    1e-8 G, whose synchrotron cooling is negligible. The lowest bin that cools
    is in regime 3 (escape 1.2 % of its cooling), and the bins below it are in
    regime 1. (The solver held a window of regime 3 to at least 6 bins, so that
    bin's window took in 5 bins without cooling. Their 1/|A|, 1e150 in the
    solver's units for A = 0, swamped the 1/|A| of that bin in the summed
    cooling time. So that bin did not cool. This gave 16 times too many
    electrons there after 10 steps (85 times after 200), and 40 % too few in
    1e-8 G. Nor did those bins keep the electrons that cooled out of the window
    into them: 8 % of all electrons were lost. With extra cooling in a band of
    3 bins, the band did not cool either.)"""
    n = compton_line(0.0)
    assert n.max() > 0
    assert np.sum(n) == pytest.approx(
        np.sum(compton_line(0.0, photons=False)), rel=1e-3
    )
    np.testing.assert_allclose(compton_line(1e-8), n, rtol=1e-6)


def test_zero_time_step():
    """A step of zero length leaves the electrons as they were. (The inflow into
    a window from the one above, divided by the time step, was 0/0 = NaN. This
    NaN emptied the window: 90 % of the electrons.)"""
    a = new(off=CLOSED_BOX + ADDED_LEPTONIC, escape=True, dt=1e3)
    a.set_escape_timescale(1e4)
    E = a.get_egrid_lep()
    a.set_t_pair_extra_cooling(np.where((E > 1e8) & (E <= 1e10), 3e3, 1e12))
    a.set_injection_rate_electrons(per_ln_e(E))
    n0 = evolve(a, 20).get_electrons()
    assert n0.max() > 0
    a.set_solver_time_step(0.0)
    a.evolve_step()
    np.testing.assert_allclose(a.get_electrons(), n0, rtol=1e-12)


def test_chang_cooper_window_at_large_courant_numbers():
    """The Courant number of the Chang & Cooper window is C = |A| dt / dx = 7,
    and a dt = 50. The densities reach the steady state (source_integral) within
    the first step. From 1e9 to 1e11 eV, they stay within 1 % of it in each of
    12 steps. The step of a cell is Crank-Nicolson's only where its right-hand
    side keeps its diagonal 1 - (C + a dt) / 2 >= 0. Elsewhere, the step is the
    least implicit one that keeps the diagonal non-negative. (AM3's
    Crank-Nicolson steps rang: 1.92, 0.15, 1.79, 0.27, 1.67, ... times the
    steady state at 1e10 eV, still 1.41 in step 11.)"""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn",), dt=1e3)
    E = a.get_egrid_lep()
    t_cool, t_esc = np.full(E.size, 1429.0), np.full(E.size, 20.0)
    q = np.exp(-0.5 * np.log(E / 1e10) ** 2)
    a.set_t_pair_extra_cooling(t_cool)
    a.set_t_pair_escape(t_esc)
    a.set_injection_rate_electrons(q)
    exact = source_integral(E, q, t_cool, t_esc)
    sel = (E >= 1e9) & (E <= 1e11)
    for step in range(1, 13):
        n = evolve(a, 1).get_electrons()
        np.testing.assert_allclose(n[sel], exact[sel], rtol=1e-2, err_msg=step)


def test_chang_cooper_steps_linear_in_the_sources():
    """The test adds a second source of 1e-60 per ln E above 3e9 eV, up to
    1e12 eV. This source must not change the electrons below 3e9 eV in the first
    3 steps. This holds because the step is linear in the sources. The rates in
    each cell set the weight theta of the step there. (If the whole window has
    one theta, that of its largest Courant number, the weak source moves its top
    from 3e9 eV to 1e10 eV. It also makes the step of the window more implicit.
    The electrons below 3e9 eV then change by up to 20 % in the first step, 8 %
    in the third.)"""

    def electrons(weak):
        a = new(off=CLOSED_BOX + ADDED_LEPTONIC, escape=True, dt=1e3)
        a.set_escape_timescale(1e6)
        E = a.get_egrid_lep()
        a.set_t_pair_extra_cooling(1e3 * (1e10 / E))
        q = np.where((E >= 1e7) & (E <= 3e9), 1.0, 0.0)
        a.set_injection_rate_electrons(
            q + np.where((E > 3e9) & (E <= 1e12), weak, 0.0)
        )
        return E, [evolve(a, 1).get_electrons() for _ in range(3)]

    E, n = electrons(0.0)
    sel = (E >= 1e7) & (E <= 3e9)
    for n1, n2 in zip(n, electrons(1e-60)[1]):
        assert np.all(n1[sel] > 0)
        np.testing.assert_allclose(n2[sel], n1[sel], rtol=1e-12)


def test_chang_cooper_window_above_fast_escape():
    """Below 1e9 eV, which the Gaussian pulse does not reach, the electrons
    escape in 1 s or in 1e5 s. The electrons above 1e9 eV must not depend on the
    escape below 1e9 eV. (The matrix of the Chang & Cooper window holds the 10
    cells below it. Suppose that the window and these cells have one weight
    theta of the implicit half of the step. Then the escape-dominated cells make
    the window's steps nearly implicit: the peak is 10 % lower, 0.620 instead of
    0.692 of the exact one.)"""

    def electrons(t_below):
        a = new(off=CLOSED_BOX + ADDED_LEPTONIC, escape=True, dt=1e3)
        E = a.get_egrid_lep()
        x = np.log(E / 1e11)
        a.set_t_pair_extra_cooling(np.full(E.size, 2e4))
        a.set_t_pair_escape(np.where(E < 1e9, t_below, 1e5))
        a.set_current_densities_electrons(np.exp(-0.5 * (x / 0.3) ** 2))
        return E, evolve(a, 20).get_electrons()

    E, n = electrons(1e5)
    sel = E > 1e9
    assert n[sel].max() > 0.3
    np.testing.assert_allclose(electrons(1.0)[1][sel], n[sel], rtol=1e-12)


SYN_SPECIES = [
    (s, MASS_GEV[m])
    for s, m in (
        ("pair", "MASS_E"),
        ("proton", "MASS_P"),
        ("pion", "MASS_PI"),
        ("muon", "MASS_MU"),
    )
]


@pytest.fixture(scope="module")
def synchrotron_cooling_times():
    B = 1e5
    t = {}
    for q in (False, True):
        a = new(quantum_syn_cooling=q)
        a.set_mag_field(B)
        a.evolve_step()  # the solvers compute their cooling rates in a step
        t[q] = {
            species: getattr(a, f"get_t_{species}_syn")()
            for species, _ in SYN_SPECIES
        }
    return B, t, a.get_egrid_lep(), a.get_egrid_had()


@pytest.mark.parametrize("species, mass", SYN_SPECIES)
def test_quantum_synchrotron_cooling_of_heavy_particles(
    synchrotron_cooling_times, species, mass
):
    """AM3's quantum suppression of synchrotron cooling is
    1 / (1 + sqrt(2) x^(2/3))^2, an interpolation of Landau & Lifshitz. It
    takes the quantum parameter
    x = chi = gamma B / B_cr(m) = (m_e / m)^2 gamma B / B_cr(m_e), because
    B_cr ∝ m^2. AM3 used gamma B / m, m / m_e times that, which overstates the
    suppression of heavy particles."""
    B, t, El, Eh = synchrotron_cooling_times
    t0, t1 = t[False][species], t[True][species]
    gamma = (El if species == "pair" else Eh) / (mass * 1e9)
    x = 1.2e-8 * B * gamma / (mass * 1e9) * (MASS_GEV["MASS_E"] / mass)
    fin = t0 < 1e199
    assert fin.sum() > 250
    np.testing.assert_allclose(
        t1[fin] / t0[fin], (1 + np.sqrt(2) * x[fin] ** (2 / 3)) ** 2, rtol=2e-5
    )


def k53_integral(x):
    """The factor e^-x is taken out, because the tolerance of the quadrature is
    absolute."""
    with mp.workdps(20):
        nu, x = mp.mpf(5) / 3, mp.mpf(x)
        f = lambda s: mp.besselk(nu, x + s) * mp.exp(x)
        return float(mp.quad(f, [0, 1, 4, 16, 64]) * mp.exp(-x))


def test_integral_of_k53(tmp_path):
    """The integral of K_5/3 from x to infinity, which the quantum-synchrotron
    kernel holds (specfun.mojo int_bessel_k53), to 3e-14 at x = 1e-4 to 400.
    AM3's integral (math.cc) summed 100 rectangles in ln t from x to 10. Above
    x = 10, it took (1.253314 + 0.139338 / x) e^-x / sqrt(x). It was 3.1 % too
    large right above x = 1e-3 and 1.9 % at 0.1-1. It was 5.4 % too small
    right above x = 10, where it jumped, and 2.9 % at 20. The 1/x term of the
    asymptotic series is sqrt(pi/2) 55/72 = 0.957 / x."""
    xs = [1e-4, 1e-3, 1.0001e-3, 0.1, 1.0, 9.99, 10.0001, 20.0, 50.0, 400.0]
    got = harness.run("bessel_k53_mojo", tmp_path, *map(repr, xs))
    np.testing.assert_allclose(
        got, [k53_integral(x) for x in xs], rtol=1e-13, atol=0.0
    )


def test_quantum_synchrotron_emission_takes_the_energy_of_the_cooling():
    """In the classical limit (chi < 1e-8), the quantum-synchrotron emission
    carries the energy that the cooling takes, dt sum n E dlnE / t_syn. The
    classical kernel carries 0.99996 of it. The quantum-synchrotron kernel
    carries 0.9996, because its average over the pitch angle leaves out
    |mu| > 0.99 and takes Simpson's rule in mu. With AM3's integral of K_5/3
    (test_integral_of_k53), the emission carried 1.019 times that energy."""
    a = new(
        off=LEPTONS_ALONE + ADDED_LEPTONIC + ("escape",),
        quantum_syn=True,
        quantum_syn_cooling=True,
        dt=1.0,
    )
    a.set_mag_field(1.0)
    E, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    n = power_law(1e3, 1e10, 1e11)(E)
    a.set_current_densities_electrons(n)
    a.evolve_step()
    t = a.get_t_pair_syn()
    cooled = np.sum(np.where(n > 0, n * E / t, 0.0)) * DLNE  # in dt = 1 s
    assert energy(a.get_photons(), Eg) / cooled == pytest.approx(1, abs=1e-3)


def jones_F(eps, gamma):
    """The inverse-Compton cooling function F(eps, gamma) of Jones 1965
    (eqs. 13-15), eps and gamma (mpf) in units of the particle's rest energy, at
    mpmath's working precision."""

    def f1(z):  # eq. 14
        return (
            (z + 6 + 3 / z) * mp.log(1 + 2 * z)
            - (mp.mpf(22) / 3 * z**3 + 24 * z**2 + 18 * z + 4)
            / (1 + 2 * z) ** 2
            - 2
            + 2 * mp.polylog(2, -2 * z)
        )

    def f2(z):  # eq. 15
        return (
            (z + mp.mpf(31) / 6 + 5 / z + 3 / (2 * z**2)) * mp.log(1 + 2 * z)
            - (
                mp.mpf(22) / 3 * z**3
                + 28 * z**2
                + mp.mpf(103) / 3 * z
                + 17
                + 3 / z
            )
            / (1 + 2 * z) ** 2
            - 2
            + mp.polylog(2, -2 * z)
        )

    g = gamma + mp.sqrt(gamma**2 - 1)  # gamma (1 + beta), after eq. 13
    return gamma * (f1(eps * g) - f1(eps / g)) - eps * (
        f2(eps * g) - f2(eps / g)
    )  # eq. 13


@pytest.fixture(scope="module")
def photon_line():
    a = new(photopion=False, bethe_heitler=False, pp=False)
    Eg = a.get_egrid_photons()
    j = np.argmin(np.abs(Eg - 25.0))
    a.set_current_densities_photons(
        np.where(np.arange(Eg.size) == j, 1e10, 0.0)
    )
    a.evolve_step()

    def F(eps, gamma):
        with mp.workdps(90):
            return np.array(
                [float(jones_F(mp.mpf(eps), mp.mpf(g))) for g in gamma]
            )

    return a, Eg[j], F


def test_inverse_compton_cooling_functions_below_z_of_one_tenth(photon_line):
    """AM3 evaluates the cooling functions f1, f2 of the inverse-Compton loss by
    Taylor series below z = 0.1. The loss is
    F(eps, gamma) = gamma [f1(eps g) - f1(eps / g)] - eps
    [f2(eps g) - f2(eps / g)], with g = gamma + sqrt(gamma^2 - 1). It kept the
    first three terms of Jones' series, 2.4 % (f1) and 1.2 % (f2) off at
    z = 0.1. So the cooling time jumped by 2.4 % where eps g crosses 0.1. With
    the series to z^10 (Jones 1965, Table I), F follows the exact functions to
    2e-5. Electrons in photons of one energy eps (photon_line) have the cooling
    time t ∝ gamma^3 / F(eps, gamma)."""
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


def test_inverse_compton_cooling_functions_with_the_dilogarithm(photon_line):
    """From z = 0.1 on, the cooling functions f1, f2 of the inverse-Compton loss
    (test_inverse_compton_cooling_functions_below_z_of_one_tenth) take their
    closed forms, with the dilogarithm Li_2(-2z). AM3 integrates the dilogarithm
    with 20 RK4 steps from Li_2(-0.1) = -0.097605235 for -20 < -2z < -0.1. Below
    -2z = -20, it takes an asymptotic series. Its Li_2 is 3.0e-4 off at
    -2z = -20. The functions f1 and f2 are up to 4.6e-5 and 1.9e-5 off below
    z = 10, where they jump. With Li_2 to rounding, the electrons in photons of
    one energy (photon_line) have the cooling time t ∝ gamma^3 / F(eps, gamma)
    of Jones' functions. This holds to 1e-9 from z = 0.1 to 1e3 (AM3's
    dilogarithm: 4.4e-5)."""
    a, Eg, F = photon_line
    t = a.get_t_pair_compton()
    gamma = a.get_egrid_lep() / MEC2_EV
    eps = Eg / MEC2_EV
    z = eps * (gamma + np.sqrt(np.maximum(gamma * gamma - 1, 0.0)))
    sel = (z > 0.1) & (z < 1e3)
    assert sel.sum() >= 90
    r = t[sel] * F(eps, gamma[sel]) / gamma[sel] ** 3
    np.testing.assert_allclose(r / np.median(r), 1.0, rtol=1e-9)


@pytest.mark.parametrize(
    "species, mass",
    [("pion", MASS_EV["MASS_PI"]), ("muon", MASS_EV["MASS_MU"])],
)
def test_inverse_compton_cooling_of_hadrons_in_every_bin(
    photon_line, species, mass
):
    """Pions and muons are in photons of one energy eps (photon_line), with eps
    in units of the rest energy of the particle. In every bin of the hadron
    grid, their inverse-Compton cooling time is t ∝ gamma^3 / F(eps, gamma)
    within 3e-3. This includes the top 49 hadron bins, whose kernels reach past
    the photon grid. (AM3 read those photon bins past the end of the array. The
    top 30 bins got cooling times of 1e-187 to 1e-18 s, and negative ones. This
    happened in 5 of 45 AM3 objects created after another one in the same
    process. am3_mojo reads the photons through an unchecked pointer. So a
    regression to that read shows here only if the memory past the array is not
    all zeros.)"""
    a, Eg, F = photon_line
    t = getattr(a, f"get_t_{species}_compton")()
    gamma = a.get_egrid_had() / mass
    r = t * F(Eg / mass, gamma) / gamma**3
    assert np.all(np.isfinite(r)) and np.all(r > 0)
    np.testing.assert_allclose(r / np.median(r), 1.0, rtol=3e-3)


def kn_average(L):
    """The Klein-Nishina cross section averaged over the angles between a photon
    and e+- of Lorentz factor gamma >> 1, in closed form. It is (2/L^2)
    int_0^L x sigma_KN(x) dx / sigma_T with L = 2 gamma eps."""
    with mp.workdps(50):
        L = mp.mpf(L)
        s = (
            -2 * L * (4 + 9 * L + L * L)
            + (4 + L) * (1 + 2 * L) ** 2 * mp.log(1 + 2 * L)
            + 4 * L * (1 + 2 * L) * mp.polylog(2, -2 * L)
        )
        return float(3 * s / (8 * L**3 * (1 + 2 * L)))


def test_inverse_compton_photon_loss_with_the_dilogarithm():
    """Photons scatter off e+- of one Lorentz factor gamma >> 1 at the rate
    sigma_T c n kn_average(L), L = 2 gamma eps. The code takes this closed form,
    with the dilogarithm Li_2(-2L), for 1e-3 <= L <= 1e3. There, the photons'
    inverse-Compton timescale is proportional to 1 / kn_average(L) to 1e-6 (1e-7
    from the form's own rounding near L = 1e-3). (AM3's dilogarithm, math.cc's
    polylog, sums 49 terms of the power series above -1. Between -10 and -1, it
    sums 99 rectangles of the integral. Below -10, it takes -ln^2(-z)/2 - 1.6.
    The rate was up to 1.8 % too large at L = 1.1 and 0.9 % too small at
    L = 5.4.)"""
    a = new(
        off=CLOSED_BOX + ADDED_LEPTONIC,
        electron_compton=True,
        electron_compton_cooling=False,
    )
    E, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    n = line(1e10, 1e9)(E)
    a.set_current_densities_electrons(n)
    a.evolve_step()
    t = a.get_t_photon_compton()
    L = 2 * (E[n > 0][0] / MEC2_EV) * (Eg / MEC2_EV)
    sel = (L >= 1e-3) & (L <= 1e3)
    r = t[sel] * np.array([kn_average(x) for x in L[sel]])
    r /= np.mean(r[(L[sel] >= 1e-2) & (L[sel] <= 0.1)])
    np.testing.assert_allclose(r, 1.0, rtol=1e-6)


def test_dilogarithm_outside_its_domain(tmp_path):
    """polylog gives Li_2(z) to rounding for real z <= 1, and NaN for z > 1,
    where Li_2 is complex, and for a NaN. (It returned pi^2/6 for both: a
    finite value, which hid a NaN argument from the solver's checks of its
    rates.)"""
    zs = [-20.0, -1.0, 0.3, 0.9, 1.0]
    got = harness.run(
        "dilogarithm_mojo", tmp_path, *map(repr, zs + [1.5, 10.0, np.nan])
    )
    np.testing.assert_allclose(
        got[: len(zs)],
        [float(mp.polylog(2, z)) for z in zs],
        rtol=1e-14,
        atol=0.0,
    )
    assert np.isnan(got[len(zs) :]).all()


def test_muons_of_pion_decay():
    """Every pi+ decay makes one muon: a mu+_L with the energy fraction
    distribution (x - r) / ((1 - r)^2 x) on [r, 1], or a mu+_R with
    r (1 - x) / ((1 - r)^2 x), r = (m_mu/m_pi)^2 (AM3 paper and
    Lipari et al. 2007). AM3 took these at the grid points and dropped x = 1,
    where the first is largest. A decay made 0.876 muons (0.549 of them mu+_L
    for 0.592), with 0.668 of the pion's energy for (1 + r)/2 = 0.786. The muons
    are one per decay, with the right helicities, and their energy is right to
    the binning's 0.2 %."""
    a = new(
        off=(
            "escape",
            "expansion",
            "adiabatic_cooling",
            "pion_syn",
            "pion_compton",
            "muon_syn",
            "muon_compton",
            "muon_decay",
        ),
        dt=1.0,
    )  # far longer than the decay times of these pions (2e-4 s)
    E = a.get_egrid_had()
    q = np.where(np.abs(np.log(E / 1e12)) < DLNE / 2, 1.0, 0.0)
    a.set_injection_rate_pion_plus(q)
    evolve(a, 10)
    decays = (10 * np.sum(q) - np.sum(a.get_pion_plus())) * DLNE
    left, right = a.get_mu_plus_left(), a.get_mu_plus_right()
    n = np.sum(left + right) * DLNE
    r = (MASS_GEV["MASS_MU"] / MASS_GEV["MASS_PI"]) ** 2
    assert abs(n / decays - 1) < 1e-4
    np.testing.assert_allclose(
        np.sum(left) * DLNE / n,
        1 - r * (np.log(1 / r) - 1 + r) / (1 - r) ** 2,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.sum((left + right) * E) * DLNE / (decays * E[q > 0][0]),
        (1 + r) / 2,
        rtol=5e-3,
    )


def test_pairs_of_muon_decay_by_helicity():
    """In V-A muon decay, the e+- have the spectrum of the decay's nu_mu-type
    neutrino. This spectrum depends on the muon's helicity h: 4/3 (1 - x^3) for
    h = +1 and 2 - 6x^2 + 4x^3 for h = -1 (for mu+, the reverse for mu-). The
    mean energy fractions are 0.4 and 0.3 (the nu_mu spectra of the AM3 paper
    and Lipari et al. 2007). AM3 gave the e+- of every muon 4/3 (1 - x^3)
    ("omitting chiralities"). So the e+- of the mu+_L and mu-_R took 0.40 of the
    muon energy instead of 0.30. Their decays made 10 % more energy than the
    muons had. Muons of each state give one e+- per decay, with the mean energy
    of the muon's helicity. This holds to the kernel's discretization, 0.7 % for
    h = +1."""
    states = [
        ("mu_plus_left", "get_positrons", 0.3),
        ("mu_plus_right", "get_positrons", 0.4),
        ("mu_minus_left", "get_electrons", 0.4),
        ("mu_minus_right", "get_electrons", 0.3),
    ]
    for state, getter, mean in states:
        # far longer than the decay time of these muons (2e-2 s)
        a = new(
            off=ADDED_LEPTONIC
            + (
                "photopion",
                "pp",
                "bethe_heitler",
                "escape",
                "expansion",
                "adiabatic_cooling",
                "muon_syn",
                "muon_compton",
                "electron_syn",
                "electron_compton",
                "annihilation",
                "ssa",
            ),
            dt=1.0,
        )
        E, El = a.get_egrid_had(), a.get_egrid_lep()
        q = np.where(np.abs(np.log(E / 1e12)) < DLNE / 2, 1.0, 0.0)
        getattr(a, "set_injection_rate_" + state)(q)
        evolve(a, 10)
        decays = (10 * np.sum(q) - np.sum(a.get_muons())) * DLNE
        n = getattr(a, getter)()
        assert abs(np.sum(n) * DLNE / decays - 1) < 5e-3, state
        np.testing.assert_allclose(
            np.sum(n * El) * DLNE / (decays * E[q > 0][0]),
            mean,
            rtol=1e-2,
            err_msg=state,
        )


# ------------------------------------------------------------ switches
def test_kaons_switched_off_after_init():
    """Kaons switched off after init_kernels are off: no kaons, and AM3's pion
    multiplicities are not scaled for the kaon channels. (A scale set only at
    init_kernels keeps them scaled, by up to 10 %.) Charm, whose decays make
    kaons, is off."""
    a = new(charm=False)
    a.set_process_kaons(False)
    b = new(kaons=False, charm=False)
    for x in (a, b):
        rich(x, 10)
    assert not a.get_kaons().any()
    for g in (
        "get_pion_minus_photopion",
        "get_photons_pi0_decay_photopion",
        "get_photons",
    ):
        np.testing.assert_allclose(
            getattr(a, g)(), getattr(b, g)(), rtol=1e-12, atol=0, err_msg=g
        )


def test_parse_sed_does_not_change_the_outputs():
    """parse_sed only splits the outputs into components. In the rich source
    with injected mu-_L and pi0, parse_sed on and off give the same e+- and
    photons (rtol 1e-9). (With parse_sed off, AM3 dropped the photons of pi0
    from pp.
    With parse_sed on, it took the e+- of muon decays and the photons of pi0
    decays as the sums of their photo-pion and pp components. These sums leave
    out the decay products of injected particles.)"""
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


MUONS = ["mu_minus_left", "mu_plus_left", "mu_minus_right", "mu_plus_right"]
NEUTRINOS = [
    "electron_neutrinos",
    "electron_anti_neutrinos",
    "muon_neutrinos",
    "muon_anti_neutrinos",
]


@pytest.mark.parametrize("species", MUONS + NEUTRINOS)
def test_external_injection_of_muons_and_neutrinos(species):
    """An external injection of one muon or neutrino species makes that species
    only. (AM3 evolved the injection of mu+_L as mu-_R and that of mu-_R as
    mu+_L. It added that of nu_e to all four neutrino species and ignored the
    other three.)"""
    group = MUONS if species in MUONS else NEUTRINOS
    a = new(photopion=False, bethe_heitler=False, pp=False, muon_decay=False)
    a.set_mag_field(1e-3)
    n = (
        a.get_egrid_had() if species in MUONS else a.get_egrid_neutrinos()
    ).size
    q = np.zeros(n)
    q[n // 2] = 1.0
    getattr(a, "set_injection_rate_" + species)(q)
    quiet(a.evolve_step)
    for other in group:
        got = getattr(a, "get_" + other)()
        assert (got.max() > 0) if other == species else not got.any(), other


def test_pion_decay_off_mid_run():
    """Pion decay is switched off during a run of the rich source. The run is
    without kaons and charm hadrons, whose decays make muons and neutrinos.
    * The muon sources from pion decays must stop (AM3 kept injecting the last
      step's muons). Cooling keeps the muon number, and decay lowers it. So the
      muon number then falls by at least the escape,
      exp(-dt/t_esc) = exp(-0.01), in every step (within 1e-9).
    * With muon decay on, the muon neutrinos then come only from the decays of
      the muons left, which get no more sources. So nu_mu + anti-nu_mu fall by
      the escape in every step, within 1e-6. (AM3 never reset the total nu_mu
      sources and added each step's muon-decay nu_mu to them, so the last step's
      sources kept coming.)"""
    a = rich(new(kaons=False, charm=False), 9)
    M, N = [], []
    for step in range(7):
        if step == 2:
            assert M[1] > M[0] > 0 and N[1] > N[0] > 0
            a.set_process_pion_decay(False)
        a.evolve_step()
        M.append(np.sum(a.get_muons()))
        N.append(
            np.sum(a.get_muon_neutrinos()) + np.sum(a.get_muon_anti_neutrinos())
        )
        if step >= 2:
            assert M[-1] <= M[-2] * np.exp(-0.01) * (1 + 1e-9), step
            assert N[-1] <= N[-2] * np.exp(-0.01) * (1 + 1e-6), step


def test_muon_decay_off_mid_run_stops_its_neutrinos():
    """Muon decay is switched off during a run of the rich source. The run is
    without kaons and charm hadrons, whose decays make neutrinos of every
    flavor. The electron neutrinos must then stop coming (AM3 kept injecting the
    last step's nu_e and anti-nu_e). Their number then falls by the escape in
    every step, exp(-dt/t_esc) = exp(-0.01), within the solver's 2e-7 per
    step."""
    a = rich(new(kaons=False, charm=False), 9)
    N = []
    for step in range(6):
        if step == 2:
            assert N[1] > N[0] > 0
            a.set_process_muon_decay(False)
        a.evolve_step()
        N.append(
            np.sum(a.get_electron_neutrinos())
            + np.sum(a.get_electron_anti_neutrinos())
        )
        if step >= 2:
            assert N[-1] <= N[-2] * np.exp(-0.01) * (1 + 1e-6), step


def test_kaon_decays_without_pion_and_muon_decay():
    """Kaon decays make muons (K -> mu nu, ...) whether or not pion decay is on.
    Their e+- (K_e3) feed the pair components of their source whether or not
    muon decay is on. Charm, whose decays make muons and e+-, is off, so that
    all of them come from kaons. (test_no_neutrinos_without_pion_decay is the
    control without kaons: no muons.)"""
    a = rich(new(pion_decay=False, muon_decay=False, charm=False), 10)
    assert a.get_muons().max() > 0 and a.get_muon_neutrinos().max() > 0
    for g in ("get_pairs_photopion", "get_pairs_proton_proton"):
        assert getattr(a, g)().max() > 0, g


DECAYS = ("pion_decay", "muon_decay", "kaon_decay", "charm_decay")
DECAY_OUTPUTS = (
    "get_protons",
    "get_neutrons",
    "get_pions",
    "get_muons",
    "get_kaons",
    "get_charm_hadrons",
    "get_neutrinos",
    "get_pairs",
    "get_photons",
)


def assert_same_outputs(a, b):
    for g in DECAY_OUTPUTS:
        np.testing.assert_array_equal(getattr(a, g)(), getattr(b, g)(), g)


def test_decays_switched_off_after_init():
    """Decays switched off between init_kernels and the first step act as if
    switched off before it. The same holds for neutron decay switched on after
    an init_kernels without it and a clear_am3. Neutron decay then has no
    kernels, as its docstring warns. (The solvers took the decay switches at
    init_kernels, and the products followed the switches. So pions and muons
    switched off kept decaying without products. Kaons and charm hadrons
    switched off decayed as if on. The neutrons decayed without products.)"""
    off = dict.fromkeys(DECAYS + ("neutron_decay",), False)
    ref = rich(new(bethe_heitler=False, **off), 5)
    a = rich(new(bethe_heitler=False))
    for d in off:
        getattr(a, "set_process_" + d)(False)
    b = new(bethe_heitler=False, **off)
    b.set_process_neutron_decay(True)
    b.clear_am3()
    rich(b)
    for x in (a, b):
        assert_same_outputs(evolve(x, 5), ref)


def test_decays_switched_on_after_init():
    """Pion, muon, kaon and charm decays switched on between init_kernels and
    the first step act as if switched on before it. (Their solvers had no decay
    sink: pions and muons made products without decaying, and kaons and charm
    hadrons did not decay.)"""
    ref = rich(new(bethe_heitler=False), 5)
    a = rich(new(bethe_heitler=False, **dict.fromkeys(DECAYS, False)))
    for d in DECAYS:
        getattr(a, "set_process_" + d)(True)
    assert_same_outputs(evolve(a, 5), ref)


HADRONIC_PHOTONS = (
    "get_photons_pi0_decay_photopion",
    "get_photons_pi0_decay_proton_proton",
    "get_photons_protons_syn",
    "get_photons_protons_compton",
    "get_photons_pions_syn",
    "get_photons_pions_compton",
    "get_photons_muons_syn",
    "get_photons_muons_compton",
)


def test_hadronic_processes_off_mid_run():
    """The hadronic processes are switched off after 10 steps of the rich
    source. Their e+- and photons stop coming. The run goes on for 20 steps of
    an escape time. After them, the photo-pion and pp e+-, the photons of
    hadronic origin and the electrons of neutron decay are below 1e-6 of what
    they were. The e+- components then add up to all pairs. (These e+- and
    photons kept the sources of the last step with the hadronic processes. The
    photon components grew up to x8, above all photons. The photo-pion and pp
    e+- got the charm-decay e+- of that step again in every step: 1e-4 of them
    left here. The totals of the e+- of muon decays kept those of the last step.
    All pairs came out x1.2 after 100 steps of a hundredth of an escape time,
    x237 after 1000.)"""
    a = rich(new(bethe_heitler=False), 10)
    a.set_process_hadronic(False)
    gone = HADRONIC_PHOTONS + (
        "get_pairs_photopion",
        "get_pairs_proton_proton",
        "get_electrons_neutron_decay",
    )
    before = [np.sum(getattr(a, g)()) for g in gone]
    assert min(before) > 0
    a.set_solver_time_step(a.get_escape_timescale())
    evolve(a, 20)
    for g, n in zip(gone, before):
        assert np.sum(getattr(a, g)()) < 1e-6 * n, g
    comp = sum(getattr(a, g)() for g in PAIR_COMPONENTS)
    np.testing.assert_allclose(
        comp + a.get_electrons_neutron_decay(),
        a.get_pairs(),
        rtol=1e-12,
        atol=1e-12 * a.get_pairs().max(),
    )


@pytest.mark.parametrize("emission", ["syn", "compton"])
def test_electron_emission_off_mid_run(emission):
    """The electrons' synchrotron or inverse-Compton emission is switched off
    after 20 steps. The photons that the emission made escape: in 15 more steps,
    all but 1e-6 of them (the component of the injected electrons). (AM3 kept
    adding the photons of the last step with the emission: they stayed.)"""
    a = new(
        off=(
            "hadronic",
            "annihilation",
            "ssa",
            "proton_syn",
            "expansion",
            "adiabatic_cooling",
        )
        + ADDED_LEPTONIC,
        dt=1e4,
    )
    a.set_mag_field(1.0)
    a.set_escape_timescale(1e4)
    a.set_injection_rate_electrons(per_ln_e(a.get_egrid_lep()))
    photons = "get_photons_injected_electrons_" + emission
    n0 = np.sum(getattr(evolve(a, 20), photons)())
    getattr(a, f"set_process_electron_{emission}_emission")(False)
    assert n0 > 0
    assert np.sum(getattr(evolve(a, 15), photons)()) < 1e-6 * n0


def test_pp_switched_on_after_init():
    """pp collisions switched on after an init_kernels without them make no
    secondaries, because their kernels were not built (as in AM3). The rich
    source evolves as with pp off. (The reset of the kernels zeroed the shifted
    proton energies of the delta approximation. So its pion sources were 0/0,
    and the solvers left the pions and the photons unchanged.)"""
    a, b = (rich(new(bethe_heitler=False, pp=False), 3) for _ in range(2))
    a.set_process_pp(True)
    assert_same_outputs(evolve(a, 3), evolve(b, 3))


def test_photopion_switched_on_after_init():
    """Photo-pion production switched on after an init_kernels without it has
    no kernels. So evolve_step raises, as after the hadronic processes are
    switched on (test_evolve_needs_init_kernels). (It ran on empty kernels,
    with no pions, neutrinos or photo-pion losses, while
    get_process_photopion() said True. AM3 crashes.)"""
    a = new(bethe_heitler=False, photopion=False)
    a.set_process_photopion(True)
    with pytest.raises(Exception, match="init_kernels"):
        rich(a, 1)


@pytest.mark.parametrize(
    "switch",
    [
        "photopion_emission",
        "photopion_cooling",
        "bethe_heitler_emission",
        "bethe_heitler_cooling",
        "proton_compton_emission",
        "pion_compton_emission",
    ],
)
def test_hadronic_kernels_whatever_the_switches_at_init(switch):
    """The photo-pion, Bethe-Heitler and hadron inverse-Compton kernels are
    built whatever the emission and cooling switches at init_kernels. With one
    switched on after init_kernels, the rich source is as with it on from the
    start. (Each kernel was built only with its switch on. An emission switched
    on later made nothing, or, in the hadrons' inverse Compton, photons at the
    energies of the targets. Photo-pion cooling switched on later found no loss
    kernel, while its re-injection created nucleons: x1.6 in 5 steps.
    Bethe-Heitler cooling switched on later made pairs without proton
    losses.)"""
    a = rich(new(**{switch: False}))
    getattr(a, "set_process_" + switch)(True)
    assert_same_outputs(evolve(a, 3), rich(new(), 3))


@pytest.mark.parametrize("process", ["pp", "photopion"])
def test_hadronic_kernels_kept_by_an_init_with_their_process_off(process):
    """The process (pp collisions or photo-pion production) is on at an
    init_kernels, off at the next one (on the same grid) and switched on again.
    The rich source is as with the process on throughout. The process runs on
    the kernels of the first init_kernels (as in AM3). (Every init_kernels
    emptied these kernels: the process switched on again made no secondaries and
    took no energy.)"""
    a = new(bethe_heitler=False)
    getattr(a, "set_process_" + process)(False)
    quiet(a.init_kernels)
    getattr(a, "set_process_" + process)(True)
    assert_same_outputs(rich(a, 3), rich(new(bethe_heitler=False), 3))


@pytest.mark.parametrize("process", ["pp", "photopion"])
def test_kaons_and_charm_switched_on_after_an_init_without_them(process):
    """As test_hadronic_kernels_kept_by_an_init_with_their_process_off, with the
    kaons and charm hadrons off at the second init_kernels too. This frees their
    decay tables, but the kernels kept from the first init_kernels can make
    them. Switched on again with the process, the kaons and charm hadrons stay
    off, as after any init_kernels without them. The rich source is as with them
    off throughout. (They were made and removed by decay without decay products.
    Through pp, this gave kaons but no neutrinos of their decays. With kaons on
    throughout, these neutrinos make 11 % of all.)"""
    a = new(bethe_heitler=False)
    switches = (process, "kaons", "charm")
    for s in switches:
        getattr(a, "set_process_" + s)(False)
    quiet(a.init_kernels)
    for s in switches:
        getattr(a, "set_process_" + s)(True)
    ref = rich(new(bethe_heitler=False, kaons=False, charm=False), 3)
    assert_same_outputs(rich(a, 3), ref)


@pytest.mark.parametrize(
    "switch", ["electron_compton_emission", "compton_photon_energy_loss"]
)
def test_compton_kernels_whatever_the_switches_at_init(switch):
    """The e+- inverse-Compton emission and photon-loss kernels are built
    whatever their switches at init_kernels. With one switched on after
    init_kernels, the rich source is as with it on from the start. (Each kernel
    was built only with its switch on, while the e+- cooling follows the
    switches every step. So the emission switched on later made no photons, and
    the photon loss removed none.)"""
    a = rich(new(hadronic=False, **{switch: False}))
    getattr(a, "set_process_" + switch)(True)
    assert_same_outputs(evolve(a, 3), rich(new(hadronic=False), 3))


def test_compton_switched_on_after_init():
    """Inverse Compton of the e+- switched on after an init_kernels without it
    neither cools them nor makes photons. Its kernels were not built (as in
    AM3). So the rich source evolves as with it off. (The e+- cooling followed
    the switch: they lost energy by inverse Compton without photons.)"""
    a, b = (
        rich(new(hadronic=False, electron_compton=False), 3) for _ in range(2)
    )
    a.set_process_electron_compton(True)
    assert_same_outputs(evolve(a, 3), evolve(b, 3))


@pytest.mark.parametrize("switch", ["bethe_heitler", "annihilation"])
def test_pair_production_switched_on_after_init(switch):
    """Bethe-Heitler and gamma-gamma pair production switched on after an
    init_kernels without them make nothing, as their docstrings say: their
    kernels were not built (as in AM3). The rich source evolves as with them
    off."""
    a, b = (
        rich(new(**{"bethe_heitler": False, switch: False}), 3)
        for _ in range(2)
    )
    getattr(a, "set_process_" + switch)(True)
    assert_same_outputs(evolve(a, 3), evolve(b, 3))


def test_late_switches_say_that_they_stay_inactive():
    """The docstrings of the master switches whose processes stay inactive when
    switched on after an init_kernels without them say so: AM3's pp collisions,
    Bethe-Heitler and gamma-gamma pair production and e+- inverse Compton, as
    the processes am3_mojo adds. (AM3's four said nothing, nor did README.md.
    These processes then made nothing without a word, while photo-pion
    production raised.)"""
    a = new(init=False)
    for name in (
        "pp",
        "bethe_heitler",
        "annihilation",
        "electron_compton",
        "pair_annihilation",
        "triplet_pair_production",
        "kaons",
        "charm",
    ):
        doc = " ".join(getattr(type(a), "set_process_" + name).__doc__.split())
        assert (
            "switched on after an `init_kernels` without it, it stays inactive"
            in doc
        ), name


def test_photopion_cooling_with_its_emission_off_at_init():
    """Photo-pion cooling with the emission off from init_kernels on re-injects
    the nucleons that interact. The rich source is as with the emission switched
    off after init_kernels. (The re-injection takes the emission kernel, which
    was built only with the emission on. So the interacting nucleons vanished,
    10 % of their energy per step instead of 1.8 %. No neutrons were made.)"""
    a = rich(new(photopion_emission=False))
    b = rich(new())
    b.set_process_photopion_emission(False)
    assert_same_outputs(evolve(a, 3), evolve(b, 3))


def test_gamma_gamma_pairs_without_their_photon_loss():
    """gamma-gamma pair production with its photon loss (cooling) off. Photons
    make the pairs they make with the photon loss on. The pairs take their
    number from the photon loss rates, also computed for the emission. (AM3
    computed them only with the photon loss on. With the photon loss off from
    the start, no pairs were made. With it switched off mid-run, the pairs came
    from the rates of its last step.)"""
    pairs = []
    for cooling in (False, True):
        a = new(
            off=CLOSED_BOX + ADDED_LEPTONIC,
            annihilation=True,
            annihilation_cooling=cooling,
        )
        Eg = a.get_egrid_photons()
        for _ in range(3):
            a.set_current_densities_photons(line(1e10, 30 * MEC2_EV)(Eg))
            a.evolve_step()
        pairs.append(a.get_pairs_annihilation())
    assert pairs[1].max() > 0
    np.testing.assert_allclose(pairs[0], pairs[1], rtol=1e-12)


def test_gamma_gamma_full_pair_kernel_after_init():
    """The full gamma-gamma pair kernel, with the eight-bin approximation
    switched off after an init_kernels with it on. Photons make the pairs they
    make with the approximation off from the start. (The kernel was built only
    with the approximation off at init_kernels: no pairs were made while the
    photons were absorbed.)"""
    pairs = []
    for after in (False, True):
        a = new(off=CLOSED_BOX + ADDED_LEPTONIC, annihilation=True, init=False)
        if not after:
            a.set_optimize_annihilation_pair_emission(False)
        quiet(a.init_kernels)
        a.set_optimize_annihilation_pair_emission(False)
        Eg = a.get_egrid_photons()
        for _ in range(3):
            a.set_current_densities_photons(line(1e10, 30 * MEC2_EV)(Eg))
            a.evolve_step()
        pairs.append(a.get_pairs_annihilation())
    assert pairs[0].max() > 0
    np.testing.assert_allclose(pairs[1], pairs[0], rtol=1e-12)


def test_gamma_gamma_pairs_after_a_switch_off():
    """gamma-gamma pair production is switched off for a step, while the
    photons fall from 1e10 to 1e8, and then on again. The first step after the
    switch-on makes no pairs, like the first step after init_kernels. This is
    because the pairs take the photon loss rates of the step before, which are
    zero while the process is off. (These rates were those of its last step
    before the switch-off: 100 times the pairs of the photons absorbed.)"""
    a = new(off=CLOSED_BOX + ADDED_LEPTONIC, annihilation=True)
    Eg = a.get_egrid_photons()
    pairs = []
    for n, on in ((1e10, True), (1e10, True), (1e8, False), (1e8, True)):
        a.set_process_annihilation(on)
        a.set_current_densities_photons(line(n, 30 * MEC2_EV)(Eg))
        a.evolve_step()
        pairs.append(a.get_pairs_annihilation())
    assert pairs[0].max() == 0.0 < pairs[1].max()
    # (in the closed box the pairs of earlier steps stay as they are)
    np.testing.assert_allclose(pairs[3], pairs[1], rtol=1e-12)


@pytest.mark.parametrize(
    "switch, getter",
    [
        ("bethe_heitler_emission", "get_pairs_bethe_heitler"),
        ("annihilation_pair_emission", "get_pairs_annihilation"),
        ("photopion_emission", "get_muons_photopion"),
    ],
)
def test_components_after_their_emission_stops(switch, getter):
    """The components of AM3's processes go on evolving after their emission is
    switched off. So they escape and decay like the totals they are part of.
    After 20 escape times, the rich source holds less than 1e-6 of the
    component. The pair components add up to all pairs. (They were evolved only
    with the emission on, so they stayed. The pair components were above all
    pairs, by up to x1e270 in emptied bins.)"""
    a = rich(new(), 10)
    n0 = np.sum(getattr(a, getter)())
    assert n0 > 0
    getattr(a, "set_process_" + switch)(False)
    a.set_solver_time_step(a.get_escape_timescale())
    evolve(a, 20)
    assert np.sum(getattr(a, getter)()) < 1e-6 * n0
    comp = sum(getattr(a, g)() for g in PAIR_COMPONENTS)
    np.testing.assert_allclose(
        comp + a.get_electrons_neutron_decay(),
        a.get_pairs(),
        rtol=1e-12,
        atol=1e-12 * a.get_pairs().max(),
    )


# the photon components, which add up to all photons but the injected ones
PHOTON_COMPONENTS = (
    "get_photons_injected_electrons_syn_compton",
    "get_photons_annihilation_pairs_syn_compton",
    "get_photons_bethe_heitler_pairs_syn_compton",
    "get_photons_photo_pion_pairs_syn_compton",
    "get_photons_proton_proton_pairs_syn_compton",
    "get_photons_protons_syn_compton",
    "get_photons_pions_syn_compton",
    "get_photons_muons_syn_compton",
    "get_photons_pi0_decay",
    "get_photons_pair_annihilation",
    "get_photons_triplet_pairs_syn_compton",
    "get_photons_neutron_decay_electrons_syn_compton",
    "get_photons_kaons_syn",
    "get_photons_kaons_compton",
    "get_photons_charm_syn",
    "get_photons_charm_compton",
)


@pytest.mark.parametrize(
    "process, held, photons",
    [
        (
            "triplet_pair_production",
            "get_pairs_triplet",
            "get_photons_triplet_pairs_syn",
        ),
        (
            "neutron_decay",
            "get_electrons_neutron_decay",
            "get_photons_neutron_decay_electrons_syn",
        ),
    ],
    ids=["triplet_pair_production", "neutron_decay"],
)
def test_photons_of_added_pairs_after_their_process_stops(
    process, held, photons
):
    """The e+- of triplet pair production or neutron decay left when the process
    is switched off radiate like all e+-. The synchrotron photons of these e+-
    are then in their photon component. The photon components add up to all
    photons but the injected ones. (The photon components were evolved only
    while the process acted or they held photons. So they stayed empty. They
    missed 0.54 % of the photons of the triplet set-up and all but 2e-7 of the
    other.)"""
    off = (
        "pair_annihilation",
        "electron_syn_emission",
        "electron_compton_emission",
    )
    if process == "neutron_decay":
        a = new(off=off + ("bethe_heitler", "photopion", "pp"), dt=1e4)
        a.set_escape_timescale(1e6)
        E = a.get_egrid_had()
        a.set_injection_rate_neutrons(
            np.where((E >= 1e10) & (E <= 1e14), 1e10 / E, 0.0)
        )
    else:
        a = new(off=off + ("hadronic",), dt=1e3)
        a.set_escape_timescale(1e4)
        a.set_injection_rate_electrons(per_ln_e(a.get_egrid_lep()))
        Eg = a.get_egrid_photons()
        a.set_injection_rate_photons(line(1e9, 0.03 * MEC2_EV)(Eg))
    a.set_mag_field(1.0)
    evolve(a, 10)
    getattr(a, "set_process_" + process)(False)
    a.set_process_electron_syn_emission(True)
    evolve(a, 5)
    assert getattr(a, held)().any() and getattr(a, photons)().any()
    comp = sum(getattr(a, g)() for g in PHOTON_COMPONENTS)
    tot = a.get_photons_except_injected()
    np.testing.assert_allclose(comp, tot, rtol=1e-12, atol=1e-12 * tot.max())


def test_photopion_pairs_after_pion_decay_off():
    """Pion decay is switched off. The muons left still decay, and their e+-
    feed the photo-pion pair component as they feed all pairs. So the emission
    timescale of the component stays finite in every bin. (The component took
    the e+- of muon decay only with pion decay on. The totals kept them, and the
    components missed 2 % of all pairs after 8 steps of a muon-dominated
    source.)"""
    a = rich(new("mojo-am3", bethe_heitler=False, annihilation=False))
    a.set_injection_rate_electrons(np.zeros(a.get_grid_size_lep()))
    evolve(a, 40)
    finite = np.count_nonzero(a.get_t_pair_emission_from_photopion() < 1e100)
    a.set_process_pion_decay(False)
    a.evolve_step()
    t = a.get_t_pair_emission_from_photopion()
    assert finite > 300 and np.count_nonzero(t < 1e100) == finite


def electrons_in_10_gauss(estimate):
    a = new(off=("hadronic",) + ADDED_LEPTONIC, init=False)
    a.set_estimate_max_energies(estimate)
    quiet(a.init_kernels)
    a.set_mag_field(10.0)
    a.set_escape_timescale(3.3e4)
    a.set_powerlaw_injection_parameters_electrons(
        4.2e45, 1e44, 1e7, 1e7, 1e12, 2.0, 2.0, 1.0
    )
    return a


def electrons_in_1_gauss(off, dt):
    a = new(off=off, dt=dt, init=False)
    a.set_estimate_max_energies(True)
    quiet(a.init_kernels)
    a.set_mag_field(1.0)
    a.set_escape_timescale(1e6)
    a.set_powerlaw_injection_parameters_electrons(
        1e48, 1e46, 1e9, 1e9, 1e16, 2.0, 2.0, 1.0
    )
    return a


def test_cooling_rates_of_a_step_without_cooling():
    """A step without cooling has zero cooling rates, as the timescale getters
    and the maximum-energy estimate read them. With both e+- cooling switches
    switched off, the pairs' synchrotron and inverse-Compton cooling times
    exceed 1e100 s in every bin. Without inverse Compton, and without triplet
    pair production on the photons made in the field, the field is set to 0
    after 3 steps. The estimated maximum electron energy 3 steps later is one
    bin above the top of the grid, as with no field from the start. (The solvers
    kept the rates of their last step with cooling: 1.9e6 s and 1.6e4 s at
    2e8 eV. The estimate stayed at the bottom of the grid, 511 keV, where it
    ended the electron injection. The solvers also took the cooling switches at
    init_kernels, so the cooling went on after they were switched off.)"""
    a = quiet(evolve, electrons_in_1_gauss(("hadronic",), 3e3), 5)
    for t in (a.get_t_pair_syn(), a.get_t_pair_compton()):
        assert t.min() < 1e100
    a.set_process_electron_syn_cooling(False)
    a.set_process_electron_compton_cooling(False)
    quiet(a.evolve_step)
    assert a.get_t_pair_syn().min() > 1e100
    assert a.get_t_pair_compton().min() > 1e100
    a = electrons_in_1_gauss(
        ("hadronic", "electron_compton", "triplet_pair_production"), 1e4
    )
    quiet(evolve, a, 3)
    a.set_mag_field(0.0)
    quiet(evolve, a, 3)
    E = a.get_egrid_lep()
    assert a.get_estimated_electron_emax() == pytest.approx(
        E[-1] ** 2 / E[-2], rel=1e-9
    )


# synchrotron self-absorption: through the e+- synchrotron, without which it
# does not act
LOSS_RATE_GETTERS = {
    "bethe_heitler_cooling": "get_t_proton_bethe_heitler",
    "photopion_cooling": "get_t_proton_photopion",
    "pp_cooling": "get_t_proton_proton_proton",
    "photopion_photon_loss": "get_t_photon_photopion",
    "compton_photon_energy_loss": "get_t_photon_compton",
    "electron_syn": "get_t_photon_ssa",
}


def test_loss_rates_of_processes_switched_off():
    """The loss rates of a process that is off are zero, as the timescale
    getters and the proton maximum-energy estimate read them. With the switches
    of LOSS_RATE_GETTERS off, each getter exceeds 1e100 s in every bin. (They
    kept the rates of the last step with the process on, for example a
    photo-pion loss time of the protons of 2.4e4 s. The rates of synchrotron
    self-absorption, which acts only with the e+- synchrotron, were computed
    without it: 0.09 s at the minimum.)"""
    a = rich(new(), 5)
    for switch in LOSS_RATE_GETTERS:
        getattr(a, "set_process_" + switch)(False)
    a.evolve_step()
    for getter in LOSS_RATE_GETTERS.values():
        assert getattr(a, getter)().min() > 1e100, getter


@pytest.mark.parametrize("switch", ["electron_syn", "hadronic", "quantum_syn"])
def test_synchrotron_kernels_of_a_field_set_with_the_switch_off(switch):
    """set_mag_field with e+- synchrotron, the hadronic processes or quantum
    synchrotron off, then the switch on. The rich source is as with the switch
    on when the field was set. (set_mag_field built only the kernels of the
    switches on. The e+- kernel stayed empty, with no synchrotron photons or
    self-absorption. The kernel of the hadrons kept the earlier field: 28 times
    too few proton synchrotron photons. Quantum synchrotron switched on later
    kept the classical kernel.)"""
    out = []
    for on in (False, True):
        a = rich(new(bethe_heitler=False))
        getattr(a, "set_process_" + switch)(on)
        a.set_mag_field(30.0)
        getattr(a, "set_process_" + switch)(True)
        out.append(evolve(a, 3))
    assert_same_outputs(*out)


def test_kaons_left_after_their_production_stops():
    """Kaons left when their production stops decay and escape like AM3's pions.
    After a step of 0.01 s, 90 % of the kaons are left, and less than 1e-6 after
    one of 100 escape times. (Kaons and charm hadrons were evolved only while
    made. They dropped out of the totals at once, made no decay products, and
    came back unchanged when their production resumed.)"""
    a = rich(new(bethe_heitler=False, charm=False, pp=False), 20)
    a.set_solver_time_step(1e-2)
    k0 = np.sum(evolve(a, 1).get_kaons())
    a.set_process_photopion_emission(False)
    assert np.sum(evolve(a, 1).get_kaons()) > 0.9 * k0
    a.set_solver_time_step(100 * a.get_escape_timescale())
    assert np.sum(evolve(a, 1).get_kaons()) < 1e-6 * k0


def parse_sed_toggled(source, first):
    if source == "leptons":
        a = new(
            off=LEPTONS_ALONE + ("triplet_pair_production",),
            dt=1e3,
            parse_sed=first,
        )
        a.set_mag_field(1.0)
        a.set_escape_timescale(1e4)
        a.set_injection_rate_electrons(per_ln_e(a.get_egrid_lep()))
        a.set_process_pair_annihilation(False)
        evolve(a, 20)
        a.set_process_parse_sed(False)
        a.evolve_step()
        a.set_process_pair_annihilation(True)
        a.evolve_step()
    else:
        a = rich(new(bethe_heitler=False, parse_sed=first), 4)
        a.set_process_parse_sed(False)
        evolve(a, 3)
    a.set_process_parse_sed(True)
    return dump_state(evolve(a, 3))


@pytest.mark.parametrize("source", ["leptons", "rich"])
def test_components_after_parse_sed_is_switched_on_again(source):
    """parse_sed off for some steps, then on again: the components (and the
    e-/e+ halves of the pair components) count the particles from that step on.
    So every getter is that of the run with parse_sed off until then (bitwise),
    totals included. (The components went on from the values that the steps
    without parse_sed left, which no longer added up to the totals. The injected
    pairs of the electrons alone were 3.5 times those of the run with parse_sed
    off until then. In the rich source, the pair components held 71 % of the
    pairs instead of 31 %. The photo-hadronic and pp neutrinos were 3-12 % and
    37-51 % higher.)"""
    toggled = parse_sed_toggled(source, True)
    for k, v in parse_sed_toggled(source, False).items():
        np.testing.assert_array_equal(toggled[k], v, err_msg=k)


@pytest.mark.parametrize("annihilation", [False, True])
def test_positrons_merged_for_a_few_steps(annihilation):
    """Positrons merged into the electrons for 5 steps, then split again for one
    step. While merged, the e- and e+ follow the pairs and add up to them. The
    pairs after the last step are those of a run that stays merged, to 1e-12
    above 1e-12 of the peak. With e+e- annihilation, which acts only on split
    pairs, they agree to 1e-6. The pair components add up to the pairs. (AM3
    kept the e- and e+ of the step before the merge. The step after the merge
    summed them into the pairs: 61 % of the pairs were lost.)"""
    out = []
    for split in (True, False):
        a = pulse_source(3e3, pair_annihilation=annihilation)
        evolve(a, 5)
        a.set_process_merge_positrons_into_electrons(True)
        evolve(a, 5)
        np.testing.assert_allclose(
            a.get_electrons() + a.get_positrons(), a.get_pairs(), rtol=1e-12
        )
        a.set_process_merge_positrons_into_electrons(not split)
        out.append(evolve(a, 1))
    split, merged = out
    pairs = split.get_pairs()
    np.testing.assert_allclose(
        pairs,
        merged.get_pairs(),
        rtol=1e-6 if annihilation else 1e-12,
        atol=1e-12 * pairs.max(),
    )
    comp = sum(getattr(split, g)() for g in PAIR_COMPONENTS)
    np.testing.assert_allclose(comp, pairs, rtol=1e-9, atol=1e-12 * pairs.max())


@pytest.mark.parametrize("merge_first", [True, False])
def test_density_setters_with_merged_positrons(merge_first):
    """With the positrons merged into the electrons, the pairs that the next
    step evolves are the sum of the e- and e+ as set. This holds whether the
    merge is switched on before the e- and e+ density setters (at init_kernels)
    or after them. get_pairs() is the sum of the e- and e+ at once. After a
    step, the pairs are those of a run without the merge, to 1e-12 above 1e-12
    of the peak. So are the e- and e+ in the bins set. Below these bins, the
    split of the merged pairs counts the pairs there as electrons. (The setters
    changed only the e- and e+, and switching the merge on kept the pairs of the
    last step. The split after the step overwrote the e- and e+ from the pairs
    that the step evolved, here none. So the particles set were lost.)"""
    out = []
    for merged in (True, False):
        a = new(
            off=LEPTONS_ALONE + ADDED_LEPTONIC,
            dt=1e3,
            merge_positrons_into_electrons=merged and merge_first,
        )
        a.set_mag_field(1.0)
        a.set_escape_timescale(1e6)
        n = 1e3 * per_ln_e(a.get_egrid_lep())
        a.set_current_densities_electrons(n)
        a.set_current_densities_positrons(0.5 * n)
        if merged and not merge_first:
            a.set_process_merge_positrons_into_electrons(True)
        if merged:
            np.testing.assert_allclose(a.get_pairs(), 1.5 * n, rtol=1e-15)
        out.append(evolve(a, 1))
    merged, split = out
    for g, sel in (
        ("get_pairs", n >= 0),
        ("get_electrons", n > 0),
        ("get_positrons", n > 0),
    ):
        ref = getattr(split, g)()
        assert ref.max() > 0
        np.testing.assert_allclose(
            getattr(merged, g)()[sel],
            ref[sel],
            rtol=1e-12,
            atol=1e-12 * ref.max(),
            err_msg=g,
        )


def test_estimated_electron_emax_with_merged_positrons():
    """estimate_max_energies with positrons merged into electrons must use the
    cooling of the merged e+-. (AM3 read the unevolved electron solver and
    gave the top of the grid.)"""
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


def test_estimated_electron_emax_when_the_merge_is_switched():
    """The e+- maximum-energy estimate takes the cooling of the solver that
    evolved the e+- in the last step. The estimate of the first merged step is
    that of a run without the merge. The estimate of the first step after the
    merge is that of a run that stays merged. (The estimate took the solver of
    the merge switch, which did not evolve the e+- in the last step. In the
    first merged step, without cooling, it was the grid's top, 3.0e20 eV, for
    1.7e13 eV. Three steps later, the pairs at 1.5e13 eV were 1.7 times those
    without the jump to 100 G. In the first step after the merge, the electrons'
    cooling of the last step before it, in 10 G, gave 6.1e13 eV for
    6.1e12 eV.)"""

    def estimates(merged):
        a = electrons_in_10_gauss(estimate=True)
        emax = []
        for step in range(24):
            if step == 21:
                a.set_mag_field(100.0)
            a.set_process_merge_positrons_into_electrons(step in merged)
            a.evolve_step()
            emax.append(a.get_estimated_electron_emax())
        return emax

    never, merged, toggled = (
        estimates(m) for m in ((), range(20, 24), range(20, 23))
    )
    assert toggled[20] == never[20]
    assert toggled[23] == merged[23]


def test_pair_cooling_timescales_when_the_merge_is_switched():
    """get_t_pair_syn and get_t_pair_compton report the cooling of the solver
    that evolved the e+- in the last step. (The getters took the solver of the
    merge switch, which did not evolve the e+-. The synchrotron timescale at bin
    100 went from 3.5e2 s to 2.7e177 s.)"""
    a = evolve(electrons_in_10_gauss(estimate=False), 5)
    before = a.get_t_pair_syn(), a.get_t_pair_compton()
    a.set_process_merge_positrons_into_electrons(True)
    after = a.get_t_pair_syn(), a.get_t_pair_compton()
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])


def test_estimated_electron_emax_with_triplet_cooling():
    """The maximum-energy estimate of the e+- takes the energy loss of triplet
    pair production, which their cooling has too. The estimated maximum is
    2.7e13 eV with the triplet cooling on. This is below the 2.0e15 eV without
    it. (The estimate left this loss out.)"""
    emax = []
    for tpp in (False, True):
        a = new(
            off=("hadronic", "electron_compton_cooling"),
            triplet_pair_production_cooling=tpp,
            init=False,
        )
        a.set_estimate_max_energies(True)
        quiet(a.init_kernels)
        a.set_mag_field(1e-3)
        Eg = a.get_egrid_photons()
        for _ in range(2):
            a.set_current_densities_photons(line(1e17, 0.03 * MEC2_EV)(Eg))
            a.evolve_step()
        emax.append(a.get_estimated_electron_emax())
    assert emax[1] < 1e-1 * emax[0]


@pytest.mark.parametrize("species", ["pair", "proton"])
def test_estimated_max_energies_with_extra_cooling(species):
    """The maximum-energy estimates take the extra cooling of the e+- and the
    protons, which their cooling has too. With 1 s above 1e11 eV, the estimate
    is the first energy where acceleration takes longer. This energy is
    9.1e12 eV for the e+- and 9.9e12 eV for the protons. (Both estimates left
    the extra cooling out: 5.0e13 and 2.2e20 eV.)"""
    pair = species == "pair"
    if pair:
        a = electrons_in_1_gauss(("hadronic",), 1e4)
    else:
        a = new(off=("bethe_heitler", "photopion", "pp"), dt=1e4, init=False)
        a.set_estimate_max_energies(True)
        quiet(a.init_kernels)
        a.set_mag_field(1.0)
        a.set_escape_timescale(1e6)
        a.set_powerlaw_injection_parameters_protons(
            1e48, 1e46, 1e10, 1e10, 1e20, 2.0, 2.0, 1.0
        )
    E = a.get_egrid_lep() if pair else a.get_egrid_had()
    getattr(a, f"set_t_{species}_extra_cooling")(np.where(E > 1e11, 1.0, 1e30))
    evolve(a, 5)
    slower = getattr(a, f"get_t_{species}_acceleration")() > 1.0
    emax = (
        a.get_estimated_electron_emax()
        if pair
        else a.get_estimated_proton_emax()
    )
    assert emax == pytest.approx(E[np.argmax(slower & (E > 1e11))], rel=1e-12)


def test_estimated_max_energies_where_acceleration_wins_everywhere():
    """With estimate_max_energies and no cooling of the electrons and protons,
    acceleration wins on the whole grid. The estimated maximum energies are
    then the energy one bin above the top of the grid, and the injection that
    ends there is finite. (AM3 read the loss rates and the energy grid one
    element past their end. When that memory read 0, the injection became
    NaN.)"""
    a = new(
        off=(
            "electron_syn",
            "electron_compton",
            "proton_syn",
            "photopion",
            "bethe_heitler",
            "pp",
        ),
        init=False,
    )
    a.set_estimate_max_energies(True)
    quiet(a.init_kernels)
    rich(a, 3)
    El, Eh = a.get_egrid_lep(), a.get_egrid_had()
    assert a.get_estimated_electron_emax() == pytest.approx(
        El[-1] ** 2 / El[-2], rel=1e-9
    )
    assert a.get_estimated_proton_emax() == pytest.approx(
        Eh[-1] ** 2 / Eh[-2], rel=1e-9
    )
    for n in (a.get_electrons(), a.get_protons()):
        assert np.all(np.isfinite(n)) and n.max() > 0


def test_estimated_proton_emax_after_pp_cooling_is_switched_on_again():
    """The proton maximum-energy estimate takes the loss rates of the step
    before. So the estimate of the first step after pp cooling is switched on
    again has no pp loss, as with pp cooling still off. (The estimate read the
    pp loss of the last step with pp cooling, at 1e13 cm^-3: 8.9e16 eV instead
    of 1.2e18 eV.)"""

    def estimate(on):
        a = new(init=False)
        a.set_estimate_max_energies(True)
        quiet(a.init_kernels)
        rich(a)
        a.set_pp_target_proton_density(1e13)
        evolve(a, 3)
        a.set_pp_target_proton_density(1e8)
        a.set_process_pp_cooling(False)
        a.evolve_step()
        a.set_process_pp_cooling(on)
        a.evolve_step()
        return a.get_estimated_proton_emax()

    assert estimate(True) == estimate(False)


def test_estimated_max_energies_after_a_clear():
    """clear_particle_densities(), which clear_am3() calls, resets the estimated
    maximum energies to those of a fresh object, which the first step of a run,
    without an estimate, reports. (The clears kept the estimates of the run
    before, 9.2e10 eV for the e+- and 4.3e22 eV for the protons. A fresh object
    has 1e10 and 1e15 eV.)"""

    def run(a, steps):
        a.set_mag_field(1.0)
        a.set_escape_timescale(1e5)
        a.set_solver_time_step(1e3)
        a.set_electron_accel_factor(1e4)
        a.set_powerlaw_injection_parameters_electrons(
            1e47, 1e46, 1e7, 1e7, 1e13, 2.0, 2.0, 1.0
        )
        return evolve(a, steps)

    reused, fresh = (new(off=("hadronic",), init=False) for _ in range(2))
    for a in (reused, fresh):
        a.set_estimate_max_energies(True)
        quiet(a.init_kernels)
    estimates = ("get_estimated_electron_emax", "get_estimated_proton_emax")
    run(reused, 10)
    assert reused.get_estimated_electron_emax() > 1e10
    reused.clear_particle_densities()
    run(reused, 1)
    run(fresh, 1)
    for g in estimates:
        assert getattr(reused, g)() == getattr(fresh, g)(), g


def test_pi0_decay_photons_of_the_top_hadron_bin():
    """The pi0-decay kernel depends only on the ratio of the photon and pion
    energies. So on the common logarithmic grid, pi0 injected into the top
    hadron bin make the photons of pi0 ten bins lower, ten photon bins up.
    The solver leaves the two lowest bins and the top bin of a grid empty.
    (AM3 read the top bin's upper edge, egrid_had[j+1], past the end of the
    grid. In 8 of 95 fresh processes with photo-pion production, the total
    photon spectrum was zero for the whole run.)"""
    out = []
    for below_top in (0, 10):
        a = new(
            off=(
                "photopion",
                "bethe_heitler",
                "pp",
                "annihilation",
                "ssa",
                "electron_syn",
                "electron_compton",
                "proton_syn",
                "pair_annihilation",
                "triplet_pair_production",
                "kaons",
            )
        )
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
    """With pion decay off (and no kaons, charm or neutron decay), no neutrinos
    are made in any bin of the neutrino grid, nor muons. This is the control
    of test_kaon_decays_without_pion_and_muon_decay. (AM3 read the nu_mu
    sources of pion decays, then of the hadron grid's length, on the neutrino
    grid, past their end.)"""
    a = rich(
        new(pion_decay=False, kaons=False, charm=False, neutron_decay=False), 5
    )
    assert a.get_pions().max() > 0 and not a.get_muons().any()
    for g in (
        "get_neutrinos",
        "get_muon_neutrinos",
        "get_muon_anti_neutrinos",
        "get_electron_neutrinos",
    ):
        assert not getattr(a, g)().any(), g


def test_pp_transition_energy_in_the_top_hadron_bin():
    """The transition energy of the pp delta approximation in the top bin of the
    hadron grid gives the same pp secondaries as one bin lower. This is
    because no protons get there. (AM3's index of the transition in the
    multiplicity table was one past its end.)"""
    out = []
    for below_top in (0, 1):
        a = new(photopion=False, bethe_heitler=False, init=False)
        a.set_pp_transition_energy_to_delta_approx(
            a.get_egrid_had()[-1 - below_top]
        )
        quiet(a.init_kernels)
        rich(a, 3)
        out.append((a.get_pions(), a.get_photons_pi0_decay_proton_proton()))
    for top, lower in zip(*out):
        assert np.all(np.isfinite(top)) and top.max() > 0
        np.testing.assert_array_equal(top, lower)


def test_pp_after_an_init_on_a_larger_grid():
    """pp collisions are switched on after an init_kernels without them, on a
    smaller grid than that of an earlier init_kernels with them. The delta
    approximation stays on the grid, and the outputs are finite. (Its loop ran
    to the transition energy of the earlier init_kernels, 119 bins on a grid
    of 70. So it read and wrote past the arrays. This fails in the build with
    bounds checks.)"""
    a = new(off=("photopion", "bethe_heitler"), init=False)
    a.set_pp_transition_energy_to_delta_approx(1e15)
    quiet(a.init_kernels)
    a.set_process_pp(False)
    a.update_energy_grid(1e-4, 1e-2, 1e12)
    quiet(a.init_kernels)
    a.set_process_pp(True)
    a.set_pp_target_proton_density(1e10)
    E = a.get_egrid_had()
    a.set_current_densities_protons(np.where(E > 1e9, 1e-6 * 1e9 / E, 0.0))
    evolve(a, 2)
    for g in ("get_protons", "get_pions", "get_photons"):
        assert np.all(np.isfinite(getattr(a, g)())), g


def test_pp_delta_approximation_between_grid_points():
    """The delta approximation of pp collisions reads the protons between grid
    points, linearly in E between the two points around each energy. Protons
    have n = E per ln E (dN/dE constant) below 100 GeV, from the grid's bottom
    or from 30 GeV up. The transition energy is at 10 TeV, so that only the
    approximation makes their pions. After a step, the second run has the pions
    of the first where it reads protons above the edge. It has none where it
    reads below the grid point under the edge. In the one bin that reads between
    those two points, it has a fraction of them. (AM3 read every energy from the
    two grid points above it and extrapolated downward. That bin had the pions
    of the first run, and below it the protons were negative, clipped.)"""
    out = []
    for emin in (1e9, 3e10):
        a = new(
            photopion=False,
            bethe_heitler=False,
            kaons=False,
            charm=False,
            pp_cooling=False,
            expansion=False,
            adiabatic_cooling=False,
            init=False,
        )
        a.set_pp_transition_energy_to_delta_approx(1e13)
        quiet(a.init_kernels)
        a.set_mag_field(0.0)
        a.set_pp_target_proton_density(1e8)
        E = a.get_egrid_had()
        a.set_current_densities_protons(
            np.where((E >= emin) & (E < 1e11), E / 1e9, 0.0)
        )
        a.evolve_step()
        out.append(a.get_pion_plus())
    read = out[0] > 0
    r = out[1][read] / out[0][read]
    assert np.count_nonzero(read) > 10
    assert np.count_nonzero((r > 1e-3) & (r < 1 - 1e-3)) == 1


# ------------------------------------------------------------ optimizations
@pytest.fixture(scope="module")
def rich_run():
    return rich(new(), 5)


def test_bethe_heitler_minimum_raised_and_lowered(rich_run):
    """Raising the minimum incoming-proton energy of the Bethe-Heitler pair
    kernel for a step and lowering it again must not lose kernel rows. If the
    kernel is packed from the minimum of the first step, it loses these rows.
    The pairs are those of rich_run, where the minimum stays. The first step has
    no target photons, so both runs are the same physics."""
    a = rich(new())
    p0 = a.get_optimize_bethe_heitler_incoming_protons_min()
    a.set_optimize_bethe_heitler_incoming_protons_min(1e3 * p0)
    a.evolve_step()
    a.set_optimize_bethe_heitler_incoming_protons_min(p0)
    ref = rich_run.get_pairs_bethe_heitler()
    assert ref.max() > 0
    np.testing.assert_array_equal(evolve(a, 4).get_pairs_bethe_heitler(), ref)


OPTIMISATION_LIMITS = {
    "photopion_target_photon_max": "photons",
    "bethe_heitler_target_photon_max": "photons",
    "compton_target_emax": "photons",
    "compton_emission_emin": "photons",
    "bethe_heitler_incoming_protons_min": "had",
}


def optimisation_limits(a):
    return [
        getattr(a, "get_optimize_" + name)() for name in OPTIMISATION_LIMITS
    ]


def test_optimisation_limits_beyond_the_grids():
    """Optimization limits beyond the end of their grid (1e30 eV) act as the
    limits they read back as, at the end of the grid. The Bethe-Heitler proton
    minimum reads back as one point above the end, which leaves no protons.
    Set to these values, the limits read back the same and give the same
    spectra. (AM3 turned them into an index past the end of the grid.)"""
    out = []
    for beyond in (True, False):
        a = new(init=False)
        for k, name in enumerate(OPTIMISATION_LIMITS):
            getattr(a, "set_optimize_" + name)(1e30 if beyond else out[0][0][k])
        limits = optimisation_limits(a)
        quiet(a.init_kernels)
        rich(a, 3)
        out.append((limits, a.get_photons(), a.get_pairs(), a.get_protons()))
    tops = [
        getattr(a, f"get_egrid_{grid}")()[-1]
        for grid in OPTIMISATION_LIMITS.values()
    ]
    assert out[0][0][:4] == tops[:4]
    assert out[0][0][4] == pytest.approx(tops[4] * np.exp(DLNE), rel=1e-12)
    assert out[0][0] == out[1][0]
    for beyond, top in zip(out[0][1:], out[1][1:]):
        assert np.all(np.isfinite(beyond)) and beyond.max() > 0
        np.testing.assert_array_equal(beyond, top)


def test_optimisation_limits_read_back_in_ev():
    """The optimization getters return the limits in eV, as their setters take
    them. By default, the limits are 511 keV for the target photons and
    1.1e12 eV for the Bethe-Heitler protons. Setting a limit to what its
    getter returns keeps it. (The getters returned the grid's units, m_e c^2
    and GeV. That round trip lowered the photo-pion target-photon limit from
    511 keV to 1 eV.)"""
    a = new(init=False)
    assert a.get_optimize_photopion_target_photon_max() == pytest.approx(
        MEC2_EV, rel=1e-12
    )
    p_min = a.get_optimize_bethe_heitler_incoming_protons_min()
    assert p_min == pytest.approx(1.1e12, rel=0.01)
    for name in OPTIMISATION_LIMITS:
        v = getattr(a, "get_optimize_" + name)()
        getattr(a, "set_optimize_" + name)(v)
        assert getattr(a, "get_optimize_" + name)() == v, name


def test_bethe_heitler_minimum_above_the_hadron_grid():
    """A minimum proton energy of the Bethe-Heitler pairs above the hadron grid
    leaves them no protons. The grid of update_energy_grid(3.3e-3, 2e5, 3e10)
    ends at 2.7e10 eV. On this grid, the default minimum of 1.1e12 eV reads back
    as 3.0e10 eV, one bin past the top, also when set to that. Protons then make
    no pairs in a step, unlike with a minimum of 1e9 eV. (The minimum was moved
    to two bins below the top, 2.5e10 eV: 7.4e-9 cm^-3 of pairs.)"""

    def one_step(p_min=None):
        a = new(grid=(3.3e-3, 2e5, 3e10), photopion=False, pp=False, init=False)
        if p_min is not None:
            a.set_optimize_bethe_heitler_incoming_protons_min(p_min)
        quiet(a.init_kernels)
        E = a.get_egrid_had()
        a.set_current_densities_protons(np.where(E > 2e9, 1e6 / E, 0.0))
        a.set_current_densities_photons(line(1e12, 1e5)(a.get_egrid_photons()))
        a.evolve_step()
        return a

    a = one_step()
    p_min = a.get_optimize_bethe_heitler_incoming_protons_min()
    assert p_min == pytest.approx(a.get_egrid_had()[-1] * np.exp(DLNE), 1e-12)
    a.set_optimize_bethe_heitler_incoming_protons_min(p_min)
    assert a.get_optimize_bethe_heitler_incoming_protons_min() == p_min
    assert not a.get_pairs_bethe_heitler().any()
    assert one_step(1e9).get_pairs_bethe_heitler().max() > 0


def test_grid_index_of_a_grid_point():
    """For the energy of every point of its grid, eV_to_*_grid_index gives
    the point's own index. For the energies 1e-6 of a bin below and above that
    point, it gives the point below and the point itself. A power-law
    injection whose minimum energy is that of a grid point starts there. (The
    integer part of ln(E / E_0) / dlnE gave the point below to 14 of the 340
    lepton points of the default grids. It did the same to 13 photon, 11
    hadron and 7 neutrino points. An injection from lepton point 43 started
    at 42.)"""
    a = new(init=False)
    for grid, getter in (
        ("lep", "lep"),
        ("photon", "photons"),
        ("had", "had"),
        ("neutrino", "neutrinos"),
    ):
        E = getattr(a, f"get_egrid_{getter}")()
        index = getattr(a, f"eV_to_{grid}_grid_index")
        own = list(range(E.size))
        assert [index(e) for e in E] == own, grid
        assert [index(e * (1 + 1e-7)) for e in E] == own, grid
        assert [index(e * (1 - 1e-7)) for e in E[1:]] == own[:-1], grid
    E = a.get_egrid_lep()
    for k in (43, 44, 81):
        a.set_powerlaw_injection_parameters_electrons(
            1e44, 1e42, E[k], E[k], 1e12, 2.0, 2.0, 1.0
        )
        assert np.flatnonzero(a.get_injection_rate_electrons())[0] == k


def test_grids_from_the_energies_of_grid_points():
    """update_energy_grid with energies of grid points makes the grids that
    hold them. With E_max that of lepton or hadron point k, that grid has k
    bins (k + 1 for an odd k, as sizes are even). So the grid ends at
    E_max or one bin below it. With a minimum energy k bins below the
    electron rest energy (photons) or 1 GeV (neutrinos), the grid starts
    there. (The integer part of ln(E / E_0) / dlnE came out one less for some
    points. For k = 329 and 339 of the 100 points from 240 on, the grid then
    had k - 1 bins. It ended two bins below E_max. For 6 of the 200 points
    from 100 on, the first at 162, the photon or neutrino grid started one
    bin above its minimum.)"""
    a = new(init=False)
    for k in range(240, 340):
        a.update_energy_grid(1e-6, 1e2, 1e9 * np.exp(k * DLNE))
        assert a.get_grid_size_had() == k + k % 2, k
        a.update_energy_grid(1e-6, 1e2, MEC2_EV * np.exp(k * DLNE))
        assert a.get_grid_size_lep() == k + k % 2, k
    for k in range(100, 300):
        e_min = MEC2_EV * np.exp(-k * DLNE)
        a.update_energy_grid(e_min, 1e2, 1e21)
        assert a.get_egrid_photons()[0] == pytest.approx(
            e_min, rel=1e-12, abs=0.0
        ), k
        e_min = 1e9 * np.exp(-k * DLNE)
        a.update_energy_grid(1e-6, e_min, 1e21)
        assert a.get_egrid_neutrinos()[0] == pytest.approx(
            e_min, rel=1e-12, abs=0.0
        ), k


def test_optimisation_limits_on_another_grid():
    """update_energy_grid keeps the optimization limits where they are. It
    keeps the defaults on the photon grids from 1e-4, 1.1e-4, 1e-5 and
    1e-6 eV, also after 12 changes between the first two. It also keeps the
    limits of set_lep_optimizer_on and set_had_optimizer_on. These limits are
    not the energies of grid points and read back as the grid points below
    them. (The indices of the limits were rounded up to even ones. The grids
    from 1e-4 and 1e-6 eV have an odd first photon index. On them, the
    default 7.8e-3 eV and 511 keV became 8.6e-3 eV and 565 keV. The 12
    changes took them to 2.6e-2 eV and 1.7 MeV.)"""
    for optimizers in (False, True):
        a = new(init=False)
        if optimizers:
            a.set_lep_optimizer_on()
            a.set_had_optimizer_on()
        limits = optimisation_limits(a)
        if optimizers:
            assert limits != optimisation_limits(new(init=False))
            for limit, asked in zip(limits, (1e6, 1e6, 1e6, 1e-3, 1e12)):
                assert asked * np.exp(-DLNE) < limit <= asked
        for e_min in (1e-4, 1.1e-4, 1e-5, 1e-6):
            b = new(init=False)
            if optimizers:
                b.set_lep_optimizer_on()
                b.set_had_optimizer_on()
            b.update_energy_grid(e_min, 1e3, 1e21)
            assert optimisation_limits(b) == pytest.approx(limits, rel=1e-12)
        for _ in range(6):
            a.update_energy_grid(1e-4, 1e3, 1e21)
            a.update_energy_grid(1.1e-4, 1e3, 1e21)
        assert optimisation_limits(a) == pytest.approx(limits, rel=1e-12)


def test_optimisation_limits_that_a_grid_cut():
    """update_energy_grid takes the limits of the new grid from the energies
    that the setters received. A Bethe-Heitler proton minimum of 1e30 eV
    means no protons on the default grid and on one up to 1e25 eV. On these
    grids, the minimum reads back as the next grid point above the grid. A
    photon grid from 1 eV raises the default minimum of the inverse-Compton
    photons to its start. On a grid from 1e-6 eV, this minimum is 7.8e-3 eV
    again. (The limits came from those of the old grid. The Bethe-Heitler
    minimum became 4.3e22 eV on the grid up to 8.7e24 eV, and the photon
    minimum stayed at 1.1 eV.)"""
    a = new(init=False)
    emin = a.get_optimize_compton_emission_emin()
    a.set_optimize_bethe_heitler_incoming_protons_min(1e30)
    for e_max in (None, 1e25):
        if e_max is not None:
            a.update_energy_grid(1.0, 1e3, e_max)
            assert a.get_optimize_compton_emission_emin() == pytest.approx(
                a.get_egrid_photons()[0], rel=1e-12
            )
        assert a.get_optimize_bethe_heitler_incoming_protons_min() == (
            pytest.approx(a.get_egrid_had()[-1] * np.exp(DLNE), rel=1e-12)
        )
    a.update_energy_grid(1e-6, 1e3, 1e21)
    assert a.get_optimize_compton_emission_emin() == pytest.approx(
        emin, rel=1e-12
    )


def test_pp_pion_models_that_do_not_exist():
    """set_pp_model_charged_pions takes the models 0 and 1 and
    set_pp_model_neutral_pions 0 to 3: another number raises and leaves the
    model as it is. (Any number was taken and read back, while init_kernels
    printed that it used model 0 for that number. Model 2 for the charged
    pions, a model of the neutral ones only, ran with SIBYLL.)"""
    a = new(init=False)
    for pions, known in (("charged", (0, 1)), ("neutral", (0, 1, 2, 3))):
        setter = getattr(a, f"set_pp_model_{pions}_pions")
        getter = getattr(a, f"get_pp_model_{pions}_pions")
        for model in known:
            setter(model)
            assert getter() == model
        for unknown in (-1, known[-1] + 1, 99):
            with pytest.raises(Exception, match="expected 0"):
                setter(unknown)
            assert getter() == known[-1]


def test_timescale_setters_take_back_what_the_getters_return():
    """Each of the 16 set_t_* array setters, given what its getter returns,
    leaves that as it is, twice over. This holds on a new object and with an
    escape timescale set. A new object has no escape and no extra cooling.
    The getters then return 1e200 s, in the units of their rates: 2.7e177 s
    for the e+- and photons. (A rate of zero came back as 1e-200 per second.
    The getter then returned a half, a third, a quarter of its 1e200 s.)"""
    a = new(init=False)
    names = [n[len("set_") :] for n in dir(a) if n.startswith("set_t_")]
    assert len(names) == 16
    for escape in (False, True):
        if escape:
            a.set_escape_timescale(1e4)
        for name in names:
            t = np.array(getattr(a, "get_" + name)())
            assert np.all(t > 0)
            for _ in range(2):
                getattr(a, "set_" + name)(t)
                again = np.array(getattr(a, "get_" + name)())
                np.testing.assert_allclose(again, t, rtol=1e-12, err_msg=name)
                t = again


def test_solver_settings_after_init():
    """The solver's matrix threshold and source integral set after
    init_kernels act as the same settings set before it. The rich source is the
    same, and unlike the defaults'. (The solvers took them when they were
    built, at init_kernels, and ignored later changes, while the getters
    reported them.)"""

    def settings(a):
        a.set_solver_simpleqintegration(True)
        a.set_solver_threshold_matrix(100.0)

    a = rich(new(hadronic=False))
    settings(a)
    b = new(hadronic=False, init=False)
    settings(b)
    quiet(b.init_kernels)
    rich(b)
    assert_same_outputs(evolve(a, 3), evolve(b, 3))
    ref = rich(new(hadronic=False), 3)
    assert not np.array_equal(a.get_photons(), ref.get_photons())


def test_target_photon_limits_lowered_after_init(rich_run):
    """Target-photon limits lowered after init_kernels act as the same limits
    set before it (100 eV, where all three change the spectra). The next step
    rebuilds the kernels for them (as test_kernel_settings_after_init), and
    the spectra are the same. (AM3 multiplies the kernels of init_kernels by
    the shorter photon vector, past its end.)"""
    spectra = lambda a: [a.get_photons(), a.get_pairs(), a.get_neutrinos()]
    out = []
    for after_init in (False, True):
        a = new(init=after_init)
        for name in (
            "photopion_target_photon_max",
            "bethe_heitler_target_photon_max",
            "compton_target_emax",
        ):
            getattr(a, "set_optimize_" + name)(1e2)  # eV
        if not after_init:
            quiet(a.init_kernels)
        out.append(spectra(rich(a, 5)))
    before, after = out
    for x, y in zip(before, after):
        np.testing.assert_array_equal(y, x)
    assert not all(
        np.array_equal(x, z) for x, z in zip(before, spectra(rich_run))
    )


# values that change the rich source (raised target-photon limits, lowered
# minimum energies), and the processes off to save time
PP_ONLY = ("bethe_heitler", "photopion")
KERNEL_SETTINGS = {
    "pp_transition_energy": (
        {"pp_transition_energy_to_delta_approx": 1e12},
        PP_ONLY,
    ),
    "pp_model_charged_pions": ({"pp_model_charged_pions": 1}, PP_ONLY),
    "pp_model_neutral_pions": ({"pp_model_neutral_pions": 2}, PP_ONLY),
    "photopion_limit": (
        {"optimize_photopion_target_photon_max": 1e8},
        ("bethe_heitler", "pp"),
    ),
    "bethe_heitler_limits": (
        {
            "optimize_bethe_heitler_target_photon_max": 1e8,
            "optimize_bethe_heitler_incoming_protons_min": 1e10,
        },
        ("photopion", "pp"),
    ),
    "compton_target_limit": (
        {"optimize_compton_target_emax": 1e8},
        ("hadronic",),
    ),
    "compton_emission_minimum": (
        {"optimize_compton_emission_emin": 1e-4},
        ("hadronic",),
    ),
}


@pytest.mark.parametrize("case", list(KERNEL_SETTINGS))
def test_kernel_settings_after_init(case):
    """Settings the kernels are built with, changed after init_kernels, act as
    set before it. The next step rebuilds the kernels for them, and the rich
    source is the same, and unlike the default's. This holds for the pp
    transition energy and pion models. It also holds for raised target-photon
    limits and lowered minimum energies of the photo-pion, Bethe-Heitler and
    inverse-Compton kernels. (AM3's kernels keep the settings of init_kernels,
    while the getters report the new ones. The rich source then stays the
    default's. The settings change the pions of the pp transition energy at
    1e12 eV by up to 8 %. They change the pions of pp model 1 by up to 48 % and
    the photons of model 2 by up to 47 %. They change the neutrinos of the
    photo-pion limit at 1e8 eV by half. They change the Bethe-Heitler pairs of
    protons from 1e10 eV by up to 63 %. The inverse-Compton limits change its
    photons by less than 1e-8.)"""
    settings, off = KERNEL_SETTINGS[case]
    out = []
    for after_init in (True, False):
        a = new(off=off, init=False)
        if after_init:
            quiet(a.init_kernels)
        for name, value in settings.items():
            getattr(a, "set_" + name)(value)
        if not after_init:
            quiet(a.init_kernels)
        out.append(rich(a, 3))
    assert_same_outputs(*out)
    default = rich(new(off=off), 3)
    assert not all(
        np.array_equal(getattr(out[0], g)(), getattr(default, g)())
        for g in DECAY_OUTPUTS
    )


@pytest.fixture(scope="module")
def pair_kernels(tmp_path_factory):
    exe = harness.build("pair_kernels_mojo", tmp_path_factory.mktemp("pk"))
    return {
        k: harness.output(exe, k)
        for k in (
            "bethe_heitler",
            "bethe_heitler_emission",
            "gamma_gamma",
            "gamma_gamma_limits",
        )
    }


def test_bethe_heitler_pair_kernel_from_its_diagonals(pair_kernels):
    """The Bethe-Heitler pair kernel depends on the e+- bin i, proton bin j and
    photon bin k only through i - j and j + k. On grids of one dlnE, it
    depends on the e+- and proton energies through their ratio, and on the
    photon energy through gamma_p eps. So init_kernels computes the kernel
    once per diagonal (once per i - j where 2 gamma_p eps >= 600). This gives
    0.13 million values for the 23 million entries of the default grids and
    limits. From these values, init_kernels packs the non-zero row windows
    of the kernel, 1.9 million values. So every entry equals that of
    (i + 1, j + 1, k - 1). (Computed entry by entry, as AM3 does, the kernel
    took 4.6 s on 4 threads instead of 0.01 s. This was nine tenths of
    blazar_simple's init_kernels. 12 entries differed by a float32 ulp.
    Stored densely before the packing, it added 116 MB to the peak memory of
    blazar_simple.)"""
    n = pair_kernels["bethe_heitler"]
    assert round(n[1] / 1e6, 1) == 1.9 and n[2] == 0


def test_bethe_heitler_pair_kernel_not_kept_without_the_process(pair_kernels):
    """With Bethe-Heitler pair production off at init_kernels, no pair kernel
    is kept. (AM3 keeps a zero kernel, 114 MiB on the default grids.)"""
    assert pair_kernels["bethe_heitler"][0] == 0


def test_bethe_heitler_pair_emission_with_the_saturated_part(pair_kernels):
    """Where 2 gamma_p eps >= 600, the integral of the Bethe-Heitler pair kernel
    over omega ends at 600 itself. So the kernel entry is
    B(i - j) / (gamma_p eps)^2. These entries are four fifths of the non-zero
    entries of the kernel. The emission takes them as a Toeplitz product of the
    e+- and proton bins and a sum over the photons. The emission equals the sum
    over the whole kernel in float64 to 1e-6. The saturated diagonals give 1.9%
    of it. (Stored as packed entries, the saturated ones took the pair
    coefficients of a blazar_simple step on 4 threads to 0.21 ms. With AM3's
    kernel, zero above 2 gamma_p eps ~ 8000, the pair coefficients took 0.09 ms.
    Factorized, the pair coefficients take 0.08 ms. Dropping one of the
    saturated diagonals changes the emission by up to 26%.)"""
    worst, share, entries = pair_kernels["bethe_heitler_emission"]
    assert worst < 1e-6 and share > 1e-2 and round(entries, 1) == 0.8


def test_gamma_gamma_full_kernel_not_built_without_its_photon_loss(
    pair_kernels,
):
    """gamma-gamma pair production off at init_kernels and switched on later,
    with the full pair kernel (no eight-bin approximation), makes no pairs.
    Its photon loss kernel was not built, so no full kernel is built for
    them. With the photon loss kernel (the eight-bin approximation switched
    off after init_kernels), the first step builds the full kernel and makes
    pairs. (The full kernel was built in both cases: 1.2 s, 319 MiB and 4 ms
    per step on one thread for no pairs.)"""
    (built, size, q), (built1, size1, q1) = pair_kernels["gamma_gamma"].reshape(
        2, 3
    )
    assert built == 0 and size == 0 and q == 0.0
    assert built1 == 1 and size1 > 0 and q1 > 0.0


def test_gamma_gamma_full_kernel_limits_at_large_lorentz_factors(
    pair_kernels,
):
    """The integration limits of the full gamma-gamma pair kernel leave out none
    of its non-zero entries in the lepton bins above 11 GeV. The pair sums
    within the limits equal those from the lowest photon bins on, to 4e-16. The
    harness docstring has the bins below. (Both limits canceled at large Lorentz
    factors gamma of the lepton. The outer one, gamma (1 - beta) / 2, was 74%
    too high at gamma = 8.8e7 and 0 from 9.5e7 on. So its log was -inf. The
    inner one of the photons x > 1, gamma - x + (f + 1/f) / 2, came out of the
    rounding, 0 or below. The pairs were up to 7.2% too low at 45 TeV, and the
    gamma-gamma pairs of the exotic scenario changed by 0.3%.)"""
    off, worst, compared = pair_kernels["gamma_gamma_limits"]
    assert compared > 0 and off == 0 and worst < 1e-12


# ------------------------------------------------------------ the interface
def test_evolve_needs_init_kernels():
    """evolve_step() raises before init_kernels(), and after
    update_energy_grid() without a new init_kernels() (the kernels are for the
    old grid). It also raises after switching on the hadronic processes that
    were off at init_kernels()."""
    a = new(init=False)
    with pytest.raises(Exception, match="init_kernels"):
        a.evolve_step()
    quiet(a.init_kernels)
    a.update_energy_grid(1e-6, 1e2, 1e18)
    with pytest.raises(Exception, match="init_kernels"):
        a.evolve_step()
    quiet(a.init_kernels)
    a.evolve_step()
    assert a.get_protons().size == a.get_egrid_had().size
    b = new(hadronic=False)
    b.set_process_hadronic(True)
    with pytest.raises(Exception, match="init_kernels"):
        b.evolve_step()


def test_pp_transition_energy_above_the_grid_after_init():
    """A transition energy of the pp delta approximation above the hadron
    grid, set after init_kernels, is rejected by the next evolve_step. That
    step rebuilds the pp kernels for it. (Set after init_kernels, it was
    ignored, as in AM3.)"""
    a = new(pp=True, bethe_heitler=False)
    a.set_pp_transition_energy_to_delta_approx(1e24)
    with pytest.raises(Exception):
        a.evolve_step()


def test_pp_negative_transition_energy_above_a_small_grid():
    """A negative transition energy of the pp delta approximation stands for
    100 GeV, which must lie on the hadron grid too. On a grid to 5e10 eV,
    init_kernels raises. (The check took the value as set: the pions then came
    from the protons above 20 GeV without their energy loss.)"""
    a = new(pp=True, init=False, grid=(1e-4, 1e-2, 5e10))
    a.set_pp_transition_energy_to_delta_approx(-1.0)
    with pytest.raises(Exception):
        quiet(a.init_kernels)


def test_grid_for_the_default_pp_transition_energy():
    """pp collisions need the transition energy of their delta approximation on
    the hadron grid, as the docstrings of its setter and of update_energy_grid
    say. With the default, 100 GeV, init_kernels needs a grid to 110 GeV or
    more. A grid to 1.1e11 eV ends at 110 GeV, and one to 1.09e11 eV at
    90 GeV. (Above the top of the grid, AM3 crashes. Without the check, the pp
    pions get no energy at all. The setter's docstring gave no limit.
    update_energy_grid takes every maximum from 10 GeV on, and its docstring
    did not say that a grid to less than 110 GeV needs a lower transition
    energy or pp switched off.)"""
    a = new(pp=True, init=False, grid=(1e-4, 1e-2, 1.1e11))
    doc = a.set_pp_transition_energy_to_delta_approx.__doc__
    assert "it must lie on the hadron grid" in doc
    doc = a.update_energy_grid.__doc__
    assert "10 GeV <= max. energy" in doc
    assert "a max. energy of 110 GeV or more" in doc
    assert a.get_egrid_had()[-1] == pytest.approx(1.1e11, rel=1e-3)
    quiet(a.init_kernels)
    b = new(pp=True, init=False, grid=(1e-4, 1e-2, 1.09e11))
    assert b.get_egrid_had()[-1] == pytest.approx(9.0e10, rel=1e-3)
    with pytest.raises(Exception, match="transition energy"):
        quiet(b.init_kernels)


@pytest.mark.parametrize("e_trans", [1e9, 1e10])
def test_pp_transition_energy_below_the_first_pion_bin(e_trans):
    """A transition energy of the pp delta approximation below about 12 GeV
    leaves the approximation no pion bins. The rich source has finite pions from
    the kernel alone. (AM3 started the pions of the kernel at a negative bin and
    wrote before the arrays. At 1 GeV, where the multiplicity is 0, it started
    them at bin (int)(-inf).)"""
    a = new(bethe_heitler=False, init=False)
    a.set_pp_transition_energy_to_delta_approx(e_trans)
    quiet(a.init_kernels)
    pions = rich(a, 3).get_pions()
    assert np.all(np.isfinite(pions)) and np.all(pions >= 0) and pions.max() > 0


@pytest.mark.parametrize(
    "grid",
    [
        (-1.0, 1e2, 1e21),
        (1e6, 1e2, 1e21),
        (1e-6, -1.0, 1e21),
        (1e-6, 1e10, 1e21),
        (1e-6, 1e2, 1e5),
        (1e-6, 1e2, float("inf")),
    ],
)
def test_invalid_energy_grid(grid):
    a = new(init=False)
    n = a.get_egrid_had().size
    with pytest.raises(Exception):
        a.update_energy_grid(*grid)
    assert a.get_egrid_had().size == n


def test_larger_hadron_grid_with_pp():
    """A hadron grid larger than the default one, with pp collisions. (AM3's
    shifted grid of the pp kernel kept the default size and was written past
    its end.)"""
    a = rich(new(pp=True, grid=(1e-4, 1e2, 1e23)), 3)
    assert a.get_egrid_had().size > 314
    assert np.all(np.isfinite(a.get_pions())) and a.get_pions().max() > 0


def hadrons_on_photons(grid=None):
    a = new(
        off=(
            "bethe_heitler",
            "pp",
            "pion_decay",
            "pion_syn",
            "pion_compton",
            "kaon_decay",
            "kaon_syn",
            "kaon_compton",
            "proton_compton",
            "proton_syn",
        ),
        grid=grid,
    )
    a.set_mag_field(1e3)
    a.set_escape_timescale(1e9)
    a.set_solver_time_step(1e2)
    E, eg = a.get_egrid_had(), a.get_egrid_photons()
    a.set_current_densities_protons(
        np.where((E > 1e15) & (E < 3e19), 1e-6 * (E / 1e15) ** -0.8, 0.0)
    )
    a.set_current_densities_photons(
        1e8 * np.exp(-(np.log(eg / 5.0) ** 2) / (2 * 0.3**2))
    )
    quiet(a.evolve_step)
    return a


def test_particles_in_the_top_bins_of_the_grids():
    """Electrons and photons in the top bins of their grids are evolved like
    those below (escape only), up to the second-highest bin. The solver leaves
    out the highest bin. Electrons are injected into one of the highest evolved
    bins and cool by synchrotron radiation much faster than the time step.
    There, they are q (dx/2) t_syn: those injected into the upper half of the
    bin's cell. (AM3 dropped the particles in the 6 highest evolved bins every
    step. Its semi-analytic regime also evolved the highest one as n + q dt,
    because the cooling rate of the top bin was zero. With only the first bug
    fixed, the synchrotron photons of that bin fed a gamma-gamma pair cascade
    there that grew without bound.)"""
    a = new(
        off=LEPTONS_ALONE + ADDED_LEPTONIC + ("electron_syn",),
        escape=True,
        dt=1.0,
    )
    a.set_escape_timescale(10.0)
    for species in ("electrons", "photons"):
        n = getattr(a, "get_" + species)().size
        n0 = np.zeros(n)
        n0[n - 30 :] = 1.0
        getattr(a, "set_current_densities_" + species)(n0)
        quiet(a.evolve_step)
        n1 = getattr(a, "get_" + species)()
        assert n1[n - 30] > 0.5
        np.testing.assert_allclose(n1[n - 8 : n - 1], n1[n - 30], rtol=1e-12)
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC, escape=False, dt=300.0)
    a.set_mag_field(100.0)
    n = a.get_egrid_lep().size
    for j in (n - 4, n - 3, n - 2):
        q = np.zeros(n)
        q[j] = 1.0
        a.set_injection_rate_electrons(q)
        a.set_current_densities_electrons(np.zeros(n))
        quiet(a.evolve_step)
        assert a.get_electrons()[j] == pytest.approx(
            0.05 * a.get_t_pair_syn()[j], rel=1e-2, abs=0.0
        ), j


def test_photopion_on_a_larger_hadron_grid():
    """update_energy_grid goes to a hadron grid larger than the default one
    before init_kernels. In the bins both grids have, the same protons and
    photons give the photo-pion rates and pions of the default grid. (The three
    arrays of AM3's photon-convolved photo-pion kernels kept the default grid's
    size and were written past their end. The C++ crashed in init_kernels.) The
    test also covers the photons on the hadron grid after update_energy_grid.
    AM3 put them one bin below where the shift between the grids takes them,
    for most E_gamma,min, every power of ten among them. So the photo-hadronic
    processes saw the photons at e^-0.1 of their energy, and the photons that
    hadrons emit landed e^0.1 too high. Both read the same shift."""
    a, b = hadrons_on_photons(), hadrons_on_photons((1e-6, 1e2, 1e24))
    n = a.get_egrid_had().size
    assert b.get_egrid_had().size > n
    np.testing.assert_allclose(
        b.get_egrid_had()[:n], a.get_egrid_had(), rtol=1e-12
    )
    np.testing.assert_allclose(
        b.get_t_proton_photopion()[:n], a.get_t_proton_photopion(), rtol=1e-10
    )
    for getter in ("get_pion_plus", "get_pion_minus", "get_kaon_plus"):
        assert getattr(a, getter)().max() > 0
        np.testing.assert_allclose(
            getattr(b, getter)()[:n], getattr(a, getter)(), rtol=1e-10, atol=0
        )


@pytest.mark.parametrize(
    "setter",
    [
        "set_current_densities_protons",
        "set_injection_rate_electrons",
        "set_t_proton_extra_cooling",
        "set_current_densities_photons",
    ],
)
def test_arrays_of_the_wrong_length_are_rejected(setter):
    a = new(init=False)
    with pytest.raises(Exception, match="elements"):
        getattr(a, setter)(np.ones(7))
    with pytest.raises(Exception):
        getattr(a, setter)(np.ones((2, 400)))


def assert_same_state(a, state):
    for k, v in dump_state(a).items():
        np.testing.assert_array_equal(v, state[k], err_msg=k)


def grid_of(setter):
    for key, grid in (
        ("neutrino", "neutrinos"),
        ("photon", "photons"),
        ("electron", "lep"),
        ("positron", "lep"),
        ("pair", "lep"),
    ):
        if key in setter:
            return grid
    return "had"


def leptonic_source():
    a = new(off=("hadronic",))
    a.set_mag_field(1.0)
    a.set_escape_timescale(1e5)
    a.set_solver_time_step(1e3)
    a.set_powerlaw_injection_parameters_electrons(
        1e47, 1e46, 1e7, 1e7, 1e12, 2.0, 2.0, 1.0
    )
    return evolve(a, 3)


def test_arrays_that_are_not_finite_are_rejected():
    """Every array setter raises for an array with a NaN or an infinity, and
    leaves the object as it was. Every getter returns what it did, and one more
    step gives the run without the calls. (The values reached the next step.
    With a NaN in the photon injection, the step raised in the photon solver
    after the e+- advanced. The step retried gave 24 % more electrons.)"""
    a, ref = leptonic_source(), leptonic_source()
    state = dump_state(a)
    setters = [
        name
        for name in dir(a)
        if name.startswith(("set_current_densities_", "set_injection_rate_"))
        or name.startswith("set_t_")
    ]
    assert len(setters) == 47
    for setter in setters:
        for bad in (np.nan, np.inf):
            v = np.ones(getattr(a, "get_grid_size_" + grid_of(setter))())
            v[-1] = bad
            with pytest.raises(Exception, match="not finite"):
                getattr(a, setter)(v)
    assert_same_state(a, state)
    assert_same_outputs(evolve(a, 1), evolve(ref, 1))


def test_arrays_that_are_not_real_numbers_are_rejected():
    """Every array setter raises for an array of complex numbers, strings,
    booleans or Python objects, and for a list that mixes booleans with
    numbers. It leaves the object as it was. Every getter returns what it did,
    and one more step gives the run without the calls. Arrays of integers and
    lists of floats are taken. (The setters converted any array numpy can. A
    complex array lost its imaginary part with only a numpy warning. Strings
    such as "1.5" gave their number, and booleans gave 1.0. But the setters of
    numbers reject strings and complex numbers. A list that mixed True with
    floats still gave 1.0 after arrays of booleans raised.)"""
    a, ref = leptonic_source(), leptonic_source()
    state = dump_state(a)
    setters = [
        name
        for name in dir(a)
        if name.startswith(("set_current_densities_", "set_injection_rate_"))
        or name.startswith("set_t_")
    ]
    assert len(setters) == 47
    for setter in setters:
        n = getattr(a, "get_grid_size_" + grid_of(setter))()
        for bad in (
            np.full(n, 1.0 + 1.0j),
            ["1.5"] * n,
            np.ones(n, bool),
            np.array([1.0] * n, dtype=object),
            [True] + [1.0] * (n - 1),
            [1] * (n - 1) + [np.True_],
            [np.array(True)] + [1.0] * (n - 1),
        ):
            with pytest.raises(Exception, match="real numbers"):
                getattr(a, setter)(bad)
    assert_same_state(a, state)
    assert_same_outputs(evolve(a, 1), evolve(ref, 1))
    n = a.get_grid_size_photons()
    a.set_injection_rate_photons(np.arange(n))
    a.set_injection_rate_photons([2.0] * n)
    assert (a.get_injection_rate_photons() == 2.0).all()


def test_integer_parameters():
    a = new(init=False)
    a.set_pp_model_charged_pions(np.int64(1))
    assert a.get_pp_model_charged_pions() == 1
    for bad in (1.5, "1", None):
        with pytest.raises(Exception):
            a.set_pp_model_charged_pions(bad)


NUMBER_SETTERS = {
    "set_solver_time_step": 1,
    "set_escape_timescale": 1,
    "set_expansion_timescale": 1,
    "set_mag_field": 1,
    "set_pp_target_proton_density": 1,
    "set_pp_transition_energy_to_delta_approx": 1,
    "set_electron_accel_factor": 1,
    "set_proton_accel_factor": 1,
    "set_solver_threshold_cool_dom": 1,
    "set_solver_threshold_esc_dom": 1,
    "set_solver_threshold_matrix": 1,
    "set_powerlaw_injection_parameters_electrons": 8,
    "set_powerlaw_injection_parameters_protons": 8,
    **{
        "set_escape_fraction_" + species: 1
        for species in (
            "pairs",
            "photons",
            "protons",
            "neutrons",
            "pions",
            "muons",
            "neutrinos",
            "charged_particles",
            "neutral_particles",
        )
    },
    **{"set_optimize_" + name: 1 for name in OPTIMISATION_LIMITS},
}


def test_numbers_that_are_not_finite_are_rejected():
    """Every setter of numbers raises for a NaN or an infinity (in any of its
    arguments), and the time step also for a negative value. The setters
    leave the object as it was: every getter returns what it did. (The
    setters took these values. A NaN time step emptied every species, and a
    NaN escape timescale the e+-. A NaN pp transition energy left no pp
    pions, and the clamp of an optimization limit took a NaN to the top of
    its grid.)"""
    a = new(init=False)
    state = dump_state(a)
    for setter, n in NUMBER_SETTERS.items():
        good = [1.0] * n
        for k in range(n):
            for bad in (np.nan, np.inf, -np.inf):
                with pytest.raises(Exception, match="finite"):
                    getattr(a, setter)(*good[:k], bad, *good[k + 1 :])
    with pytest.raises(Exception, match=">= 0"):
        a.set_solver_time_step(-1.0)
    assert_same_state(a, state)


GRIDS = ("lep", "photon", "had", "neutrino")
NUMBER_ARGUMENTS = {
    **NUMBER_SETTERS,
    "set_pp_model_charged_pions": 1,
    "set_pp_model_neutral_pions": 1,
    "update_energy_grid": 3,
    **{f"eV_to_{grid}_grid_index": 1 for grid in GRIDS},
}


def rejects_in_every_argument(a, values, match):
    for method, n in NUMBER_ARGUMENTS.items():
        for k in range(n):
            for bad in values:
                args = [1] * n
                args[k] = bad
                with pytest.raises(Exception, match=match):
                    getattr(a, method)(*args)


def test_numbers_that_are_booleans_are_rejected():
    """Every setter of numbers, the pp model setters, update_energy_grid and
    the eV_to_*_grid_index methods raise for True, False, a numpy boolean or a
    0-d array of one in any of their arguments, as the array setters do for
    arrays of booleans, and leave the object as it was. (The setters took True
    as 1: set_mag_field(True) gave a field of 1 G, and
    set_pp_model_charged_pions(True) pion model 1, QGSJET. A 0-d array,
    np.array(True), still gave 1 G after the others raised.)"""
    a = new(init=False)
    state = dump_state(a)
    rejects_in_every_argument(
        a, (True, False, np.True_, np.array(True)), "expected a number"
    )
    assert_same_state(a, state)


def test_numbers_that_are_complex_are_rejected():
    """The same methods raise for a complex number in any of their arguments,
    also numpy's and a 0-d array of one, as the array setters do for complex
    arrays, and leave the object as it was. (They took a numpy complex number
    as its real part, with only a numpy warning: set_mag_field(np.complex128(3
    + 4j)) gave a field of 3 G.)"""
    a = new(init=False)
    state = dump_state(a)
    rejects_in_every_argument(
        a,
        (1 + 1j, np.complex128(1 + 1j), np.array(1 + 0j)),
        "expected a real number",
    )
    assert_same_state(a, state)


def test_negative_field_is_rejected():
    """set_mag_field raises for a negative field, -0.0 too, as
    set_solver_time_step for a negative time step, and leaves the object as
    it was. (The setter took negative fields. The classical synchrotron
    kernels of the e+- and the hadrons then raised negative numbers to the
    power 0.33333, NaN. So every step raised after it advanced the e+-. With
    quantum synchrotron on and the hadronic processes off, a negative field
    acted as none.)"""
    a = new(init=False)
    state = dump_state(a)
    for b in (-10.0, -0.0):
        with pytest.raises(Exception, match=">= 0"):
            a.set_mag_field(b)
    assert_same_state(a, state)


def test_timescales_that_are_not_positive_are_rejected():
    """Every timescale setter raises for a timescale of 0, -0.0 or below, or one
    whose rate 1/t overflows (1e-320 s). Every escape-fraction setter raises for
    a negative fraction. The setters leave the object as it was: every getter
    returns what it did, and one more step gives the run without the calls. (The
    setters took these values. With a photon escape time of 0 in bin 300, the
    next step raised in the photon solver after it advanced the e+-. With an
    escape timescale of -1e4 s, the electrons grew to 76 times those of 1e4 s in
    40 steps.)"""
    a, ref = leptonic_source(), leptonic_source()
    state = dump_state(a)
    bad = (0.0, -0.0, -1e4, 1e-320)
    for setter in ("set_escape_timescale", "set_expansion_timescale"):
        for t in bad:
            with pytest.raises(Exception, match="timescale >= 1e-200 s"):
                getattr(a, setter)(t)
    setters = [name for name in dir(a) if name.startswith("set_t_")]
    assert len(setters) == 16
    for setter in setters:
        for t in bad:
            v = np.ones(getattr(a, "get_grid_size_" + grid_of(setter))())
            v[300] = t
            with pytest.raises(Exception, match="timescale >= 1e-200 s"):
                getattr(a, setter)(v)
    fractions = [n for n in dir(a) if n.startswith("set_escape_fraction_")]
    assert len(fractions) == 9
    for setter in fractions:
        with pytest.raises(Exception, match=">= 0"):
            getattr(a, setter)(-0.5)
    assert_same_state(a, state)
    assert_same_outputs(evolve(a, 1), evolve(ref, 1))


def test_timescales_too_short_for_the_solver_are_rejected():
    """Every timescale setter raises for a timescale below 1e-200 s. Every
    escape-fraction setter raises for a fraction above 1e100. The setters leave
    the object as it was: every getter returns what it did, and one more step
    gives the run without the calls. A step runs with a fraction of 1e100 and
    timescales of 1e-200 s. (The setters took these values. The rate 1/t is
    finite down to 5.6e-309 s, but its sum over a grid is not. With a photon
    escape time of 1e-306 s, the next step raised in the photon solver after it
    advanced the e+-. The step retried gave 0.64 to 3.5 times the electrons.
    With an escape fraction of 1e110 and an escape timescale of 1e-200 s, the
    e+- solver raised.)"""
    a, ref = leptonic_source(), leptonic_source()
    state = dump_state(a)
    for setter in ("set_escape_timescale", "set_expansion_timescale"):
        with pytest.raises(Exception, match="timescale >= 1e-200 s"):
            getattr(a, setter)(1e-306)
    for setter in [name for name in dir(a) if name.startswith("set_t_")]:
        n = getattr(a, "get_grid_size_" + grid_of(setter))()
        with pytest.raises(Exception, match="timescale >= 1e-200 s"):
            getattr(a, setter)(np.full(n, 1e-306))
    for setter in [n for n in dir(a) if n.startswith("set_escape_fraction_")]:
        with pytest.raises(Exception, match="<= 1e100"):
            getattr(a, setter)(1e101)
    assert_same_state(a, state)
    assert_same_outputs(evolve(a, 1), evolve(ref, 1))
    a.set_escape_fraction_pairs(1e100)
    a.set_escape_timescale(1e-200)
    a.set_t_photon_escape(np.full(a.get_grid_size_photons(), 1e-200))
    a.evolve_step()


def test_numbers_and_arrays_out_of_range_are_rejected():
    """set_pp_target_proton_density raises for a density < 0, and the
    acceleration factors for a factor <= 0 (-0.0 too). The setters of densities
    and injection rates raise for an array with a negative value. The setters
    leave the object as it was: every getter returns what it did, and one more
    step gives the run without the calls. A density of 0 and an array of -0.0
    are taken. (The setters took these values. A target density of -1e8 cm^-3
    made the protons' pp loss timescale negative in 312 of 314 bins of a pp
    source. This source then made no pions. An acceleration factor of -1 put the
    estimated maximum electron energy at the bottom of the grid, 5.1e5 eV. So
    the power-law injection peaked 130 times higher there. A factor of 0 gave
    NaN acceleration timescales without a field. An electron density of -1e10
    stayed negative in steps without injection or cooling.)"""
    a, ref = leptonic_source(), leptonic_source()
    state = dump_state(a)
    with pytest.raises(Exception, match=">= 0"):
        a.set_pp_target_proton_density(-1e8)
    for setter in ("set_electron_accel_factor", "set_proton_accel_factor"):
        for eta in (0.0, -0.0, -1.0):
            with pytest.raises(Exception, match="positive"):
                getattr(a, setter)(eta)
    setters = [
        name
        for name in dir(a)
        if name.startswith(("set_current_densities_", "set_injection_rate_"))
    ]
    assert len(setters) == 31
    for setter in setters:
        v = np.ones(getattr(a, "get_grid_size_" + grid_of(setter))())
        v[-1] = -1.0
        with pytest.raises(Exception, match="not >= 0"):
            getattr(a, setter)(v)
    assert_same_state(a, state)
    assert_same_outputs(evolve(a, 1), evolve(ref, 1))
    a.set_pp_target_proton_density(0.0)
    a.set_injection_rate_photons(-np.zeros(a.get_grid_size_photons()))


@pytest.mark.parametrize("species", ["electrons", "protons"])
def test_power_law_injection_parameters_out_of_range_are_rejected(species):
    """The power-law injection setters raise for a volume <= 0 or a
    luminosity < 0. They also raise for a minimum, break or maximum energy <= 0,
    or a cutoff steepness < 0. The setters leave the object as it was. A
    luminosity of 0 (no injection) is taken. (The setters took these values. A
    volume of 0 gave an infinite injection, and the next step raised with it. A
    negative volume or luminosity gave a negative injection. A minimum
    energy <= 0 gave an injection from the bottom of the grid. A break or
    maximum energy of 0 gave none. A negative break gave a negative injection
    (down to -7.3e49 for the electrons). A negative maximum gave an injection
    that is not finite. A steepness of -1 gave an injection cut below the
    maximum energy, with 99 % of its luminosity above it.)"""
    a = new(init=False)
    state = dump_state(a)
    setter = getattr(a, "set_powerlaw_injection_parameters_" + species)
    good = [1e48, 1e46, 1e9, 1e9, 1e12, 2.0, 2.0, 1.0]
    for k, bad in [(1, -1.0), (7, -1.0)] + [
        (k, v) for k in (0, 2, 3, 4) for v in (0.0, -1.0)
    ]:
        with pytest.raises(Exception, match="positive|>= 0"):
            setter(*good[:k], bad, *good[k + 1 :])
    assert_same_state(a, state)
    setter(*good[:1], 0.0, *good[2:])
    assert not getattr(a, "get_injection_rate_" + species)().any()


@pytest.mark.parametrize(
    "species, index, steepness",
    [
        ("electrons", -20.0, 0.0),
        ("electrons", -60.0, 1.0),
        ("electrons", 400.0, 1.0),
        ("protons", -60.0, 1.0),
        ("protons", 400.0, 1.0),
    ],
)
def test_power_law_injection_of_a_hard_or_steep_spectrum(
    species, index, steepness
):
    """A power-law injection carries the energy of the same injection with
    index 2, to 1e-12. It does so also with a hard or steep spectrum, in finite
    bins. (AM3 scaled the shape by L_density / integral. With index -20 and no
    cutoff, the factor underflowed to 0, which left the electrons no injection.
    With index -60 or 400, the shape overflowed, which made bins infinite or
    NaN, and the next step raised.)"""
    a = new(init=False)
    setter = getattr(a, "set_powerlaw_injection_parameters_" + species)
    getter = getattr(a, "get_injection_rate_" + species)
    E = a.get_egrid_lep() if species == "electrons" else a.get_egrid_had()
    energies = (1e6, 1e8, 1e12) if species == "electrons" else (1e9, 1e12, 1e18)
    setter(1e40, 1e40, *energies, 2.0, 2.0, steepness)
    expected = energy(getter(), E)
    setter(1e40, 1e40, *energies, index, index, steepness)
    q = getter()
    assert np.isfinite(q).all()
    assert energy(q, E) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_grid_index_of_an_energy_that_is_not_positive():
    """The eV_to_*_grid_index methods raise for an energy that is not positive
    or not finite. (They took the integer part of ln(E) / dlnE without a
    check. This gave -2^63 for E = 0, bin 0 for a negative energy or NaN, and
    2^63 - 1 for infinity.)"""
    a = new(init=False)
    for grid in ("lep", "photon", "had", "neutrino"):
        index = getattr(a, f"eV_to_{grid}_grid_index")
        assert index(1e12) > 0
        for bad in (0.0, -0.0, -1.0, np.nan, np.inf, -np.inf):
            with pytest.raises(Exception, match="positive"):
                index(bad)


@pytest.mark.parametrize("grid", [None, (1e-2, 1e3, 3e14)])
def test_grid_index_below_the_grid(grid):
    """The eV_to_*_grid_index methods raise for an energy below the first point
    of their grid, on the default grids and on others. Within 1e-9 of a bin
    below that point, an energy is the point's. (They gave the index 0 up to
    one bin below the grid and a negative index further below, which numpy
    counts from the end of an array: eV_to_had_grid_index(1e8) gave -23, which
    read the proton bin at 4.3e21 eV.)"""
    a = new(grid=grid, init=False)
    for name, getter in zip(GRIDS, ("lep", "photons", "had", "neutrinos")):
        E0 = getattr(a, f"get_egrid_{getter}")()[0]
        index = getattr(a, f"eV_to_{name}_grid_index")
        assert index(E0) == index(E0 * (1 - 1e-12)) == 0, name
        for e in (E0 * (1 - 1e-7), E0 / 2, E0 * 1e-10):
            with pytest.raises(Exception, match="below the grid"):
                index(e)


def test_grid_index_above_the_grid():
    """Up to one bin above the top point of a grid, eV_to_*_grid_index gives
    that point, the point at or below the energy. From one bin above it on, it
    gives the grid's size or more, as the docstrings say. (They said that every
    energy above the top gave the size or more.)"""
    a = new(init=False)
    for name, getter in zip(GRIDS, ("lep", "photons", "had", "neutrinos")):
        E = getattr(a, f"get_egrid_{getter}")()
        index = getattr(a, f"eV_to_{name}_grid_index")
        assert index(E[-1] * 1.05) == E.size - 1, name
        assert index(E[-1] * np.exp(DLNE)) == E.size, name
        assert (
            "An energy one bin or more above the grid's top point gives the"
            " grid's size or more" in " ".join(index.__doc__.split())
        ), name


EMISSION_TIMESCALES = [
    "get_t_photon_emission_from_ic",
    "get_t_photon_emission_from_pi0",
    "get_t_photon_emission_from_syn",
    "get_t_pair_emission_from_annihilation",
    "get_t_pair_emission_from_bethe_heitler",
    "get_t_pair_emission_from_injection",
    "get_t_pair_emission_from_photopion",
    "get_t_pair_emission_from_proton_proton",
]


@pytest.mark.parametrize("getter", EMISSION_TIMESCALES)
def test_emission_timescales_before_init_kernels(getter):
    """AM3's 8 emission-timescale getters work before init_kernels and give an
    array for every bin of their grid.
    (They read arrays that AM3 allocates in init_kernels, and crashed before
    it.)"""
    a = new(init=False)
    t = getattr(a, getter)()
    grid = a.get_egrid_photons() if "photon" in getter else a.get_egrid_lep()
    assert t.size == grid.size and np.all(np.isfinite(t))


def test_neutrino_timescales_cover_the_neutrino_grid():
    """The neutrino escape and extra-cooling timescales have a value for every
    bin of the neutrino grid (AM3 returned only the hadron grid's length of
    them)."""
    a = new(init=False)
    n = a.get_egrid_neutrinos().size
    assert n > a.get_egrid_had().size
    for g in ("get_t_neutrino_escape", "get_t_neutrino_extra_cooling"):
        assert getattr(a, g)().size == n, g


def test_pi0_emission_timescale_without_parse_sed():
    """The emission timescale of the photons of pi0 decays divides the photons
    by the source they evolve with. The timescale is the same with parse_sed on
    and off. It is finite in more than 500 bins. (AM3 divided by the photo-pion
    and pp components. These are zero with parse_sed off, which gave >= 1e100 s
    in every bin with photons. They also leave out injected pi0.)"""
    t = []
    for parse in (True, False):
        a = rich(new(bethe_heitler=False, parse_sed=parse), 5)
        t.append(a.get_t_photon_emission_from_pi0())
    np.testing.assert_allclose(t[1], t[0], rtol=1e-9)
    assert np.count_nonzero(t[1] < 1e100) > 500


def test_pi0_emission_timescale_with_pion_decay_off():
    """With pion decay off, the source of the emission timescale of the photons
    of pi0 decays holds the photons of the charm decays. These photons give a
    finite timescale (< 1e100 s) in more than 100 bins. (The timescale took the
    source as zero with pion decay off: >= 1e100 s in every bin.)"""
    a = rich(new(bethe_heitler=False, pion_decay=False), 3)
    t = a.get_t_photon_emission_from_pi0()
    assert np.count_nonzero(t < 1e100) > 100


@pytest.mark.parametrize("radiation", ["syn", "compton"])
def test_photon_emission_timescales_of_kaons(radiation):
    """The emission timescales of the photons of all synchrotron and all
    inverse-Compton radiation take the kaons and charm hadrons too. The emission
    of the e+-, protons, pions and muons is off. The timescales are then finite
    (< 1e100 s) wherever the kaons' photons are above 1e-6 of their peak. (They
    left out the kaons and charm hadrons: >= 1e100 s in every bin.)"""
    names = ("electron", "proton", "pion", "muon")
    off = {f"{n}_{radiation}_emission": False for n in names}
    a = rich(new(bethe_heitler=False, **off), 3)
    t = getattr(
        a, "get_t_photon_emission_from_" + radiation[:3].replace("com", "ic")
    )()
    k = getattr(a, "get_photons_kaons_" + radiation)()
    assert k.max() > 0
    assert np.all(t[k > 1e-6 * k.max()] < 1e100)


def test_pair_emission_timescales_with_merged_positrons():
    """With the positrons merged into the electrons, the e+- emission timescales
    divide all pairs by the sources of both. The timescale of the injection is
    above 1 s wherever the pairs are above 1e-6 of their peak. (The timescales
    divided the electrons, not evolved then: 1e-62 s where electrons are
    injected.)"""
    a = rich(new(hadronic=False, merge_positrons_into_electrons=True), 5)
    n = a.get_pairs()
    t = a.get_t_pair_emission_from_injection()
    m = n > 1e-6 * n.max()
    assert np.isfinite(n).all() and m.sum() >= 10
    assert np.all(t[m] > 1.0)


def test_charm_chain_components_after_parse_sed_is_switched_on():
    """The pion, muon and neutrino components of the charm decay chain are
    evolved while the kaons of the charm decays decay. (Only the charm decays
    switched these components on: the kaons' decay products went into the totals
    alone.)"""
    a = rich(new(bethe_heitler=False, parse_sed=False), 5)
    a.set_process_charm_decay(False)
    a.set_process_parse_sed(True)
    assert evolve(a, 3).get_muon_neutrinos_charm_decay_photopion().max() > 0


def hadrons_alone(off):
    a = new(off=off + ("proton_syn", "proton_compton"), dt=1e3)
    a.set_mag_field(1e-3)
    a.set_escape_timescale(1e30)
    return a, a.get_egrid_had(), a.get_egrid_photons()


def test_photopion_nucleons_above_the_target_photon_limit():
    """The test puts protons on photons above the default target-photon limit of
    photo-pion production (511 keV). This limit leaves their interactions out as
    a whole: the nucleons are all there after a step (to 1e-9). (The nucleon
    loss took every photon, and the re-injection and the products only those
    below the limit: 4 % of the nucleons vanished.)"""
    a, E, Eg = hadrons_alone(("bethe_heitler", "pp"))
    a.set_current_densities_protons(line(1.0, 1e12)(E))
    a.set_current_densities_photons(line(1e14, 2e6)(Eg))
    n0 = number(a.get_protons())
    a.evolve_step()
    n1 = number(a.get_protons() + a.get_neutrons())
    assert n1 == pytest.approx(n0, rel=1e-9, abs=0.0)


def test_photopion_photon_loss_timescale_on_the_photon_grid():
    """get_t_photon_photopion gives the photons' photo-pion loss rate on the
    photon grid, as they evolve with it. The photo-pion photon loss is the only
    process that acts on the photons. The photons then decay in a step at the
    rate 1/t (to 1e-6) wherever the rate is above 1e-3 of its peak. (The getter
    returned the rate on the photon grid of the hadrons, 0.21 bin below. This
    rate was 4.8 % off in the median of these bins, and 22 % at 460 keV.)"""
    a, E, Eg = hadrons_alone(
        ("bethe_heitler", "pp", "photopion_emission", "annihilation")
        + ("neutron_decay",)
    )
    a.set_solver_time_step(1.0)
    a.set_current_densities_protons(
        np.where((E > 1e10) & (E < 1e17), 1e37 / E**2, 0.0)
    )
    n0 = np.where((Eg > 0.1) & (Eg < 1e8), 1e12 / Eg**1.5, 0.0)
    a.set_current_densities_photons(n0)
    a.evolve_step()
    rate = 1 / a.get_t_photon_photopion()
    m = rate > 1e-3 * rate.max()
    assert np.isfinite(rate).all() and m.sum() >= 30
    np.testing.assert_allclose(
        -np.log(a.get_photons()[m] / n0[m]), rate[m], rtol=1e-6
    )


def test_bethe_heitler_pairs_of_ultra_high_energy_protons():
    """The test puts protons at 1e18 eV on photons at 5 eV. There,
    2 gamma_p eps ~ 2e4: the largest photon energy in the proton's frame, in
    units of m_e c^2. Bethe-Heitler pair production makes pairs from the
    collisions below 600, where AM3's Born limit ends its cross section. (AM3
    spread its 5 integration steps over the whole range. Above ~8000, none
    fell below 600, and the protons lost energy without making pairs.)"""
    a, E, Eg = hadrons_alone(("photopion", "pp"))
    a.set_current_densities_protons(line(1e-6, 1e18)(E))
    a.set_current_densities_photons(line(1e10, 5.0)(Eg))
    a.evolve_step()
    assert a.get_pairs_bethe_heitler().max() > 0


def test_bethe_heitler_pair_kernel_at_the_born_limit():
    """The Bethe-Heitler pair spectrum depends on the e+- and proton energies
    only through their ratio and on gamma_p eps. So protons one bin lower on
    photons one bin higher make the same spectrum one bin lower. This holds to
    1e-9 in the bins above 1e-3 of the peak. The protons are at 4e17 eV on
    photons at 1 eV, with 2 gamma_p eps ~ 1600. This is beyond the
    omega = 600 at which the integration ends. (The integration summed its
    steps to its last point, which rounding put slightly above 600 for a
    tenth of the e+- and proton bins. So 5-11 % of their pairs were left
    out, for example in e+- bin 222 of protons in bin 198.)"""
    out = []
    for d in (0, 1):
        a, E, Eg = hadrons_alone(
            ("photopion", "pp", "electron_syn", "electron_compton")
            + ("annihilation",)
            + ADDED_LEPTONIC
        )
        p = np.zeros(E.size)
        p[198 - d] = 1e-6
        g = np.zeros(Eg.size)
        g[np.argmin(np.abs(np.log(Eg))) + d] = 1e10
        a.set_current_densities_protons(p)
        a.set_current_densities_photons(g)
        a.evolve_step()
        out.append(a.get_pairs_bethe_heitler())
    hi, lo = out[0][1:], out[1][:-1]
    sig = hi > 1e-3 * hi.max()
    assert sig[221]  # (e+- bin 222)
    ratio = lo[sig] / hi[sig]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


def test_power_law_injection_that_misses_the_grid():
    """A power-law injection whose spectrum has no bin on the grid
    (from 1e30 eV) injects nothing. (AM3 divided by its zero integral: NaN in
    every bin, with which the solvers left the species unchanged.)"""
    a = new(init=False)
    p = (1e40, 1e40, 1e30, 1e30, 1e32, 2.0, 2.0, 1.0)
    a.set_powerlaw_injection_parameters_electrons(*p)
    a.set_powerlaw_injection_parameters_protons(*p)
    for g in ("get_injection_rate_electrons", "get_injection_rate_protons"):
        assert np.all(getattr(a, g)() == 0.0), g


def test_quantum_synchrotron_without_a_field():
    """Quantum synchrotron without a field (B = 0): the e+- emit no synchrotron
    photons, and injected photons build up. (AM3's kernel was inf * 0 = NaN,
    with which the photons stayed as they were.)"""
    a = new(off=LEPTONS_ALONE + ("pair_annihilation",), quantum_syn=True)
    a.set_mag_field(0.0)
    a.set_escape_timescale(1e30)
    a.set_injection_rate_electrons(per_ln_e(a.get_egrid_lep()))
    a.set_injection_rate_photons(line(1.0, 1e3)(a.get_egrid_photons()))
    n = evolve(a, 3).get_photons()
    assert np.all(np.isfinite(n)) and n.max() > 0


def test_source_that_is_not_finite():
    """A source that is not finite stops evolve_step with an exception. Here,
    the power-law injection overflows to NaN and infinity. The setter takes its
    parameters, which are finite. (The solver's test of the source failed for
    NaN, so it left the species unchanged without a word.)"""
    a = new(off=CLOSED_BOX)
    a.set_powerlaw_injection_parameters_electrons(
        1e-300, 1e300, 1e9, 1e9, 1e12, 2.0, 2.0, 1.0
    )
    # (with the solver's message: the parallel evolution kept only that it
    #  failed)
    with pytest.raises(Exception, match="is not finite"):
        a.evolve_step()


def test_rates_that_are_not_finite():
    """Escape or cooling rates that are not finite stop evolve_step with an
    exception, as a source that is not finite does. Here, a field of 1e160 G,
    which the setter takes, overflows B^2 and the synchrotron cooling rates.
    (The solver set the NaN densities they make to zero without a word.
    Electrons injected as per_ln_e for 3 steps in 10 G all vanished in the
    step with that field.)"""
    a = new(dt=1e3)
    a.set_mag_field(10.0)
    a.set_injection_rate_electrons(per_ln_e(a.get_egrid_lep()))
    evolve(a, 3)
    a.set_mag_field(1e160)
    with pytest.raises(Exception, match="rates of .* are not finite"):
        a.evolve_step()


def test_solution_that_is_not_finite():
    """The source integral of the cooling-only window below 1e12 eV overflows
    where its padding escapes 1e172 times faster than it cools. The window's
    solution is then NaN. evolve_step stops with an exception. (The solver set
    the NaN densities to zero without a word, as AM3 does: all 9 bins of the
    window.)"""
    a = new(off=LEPTONS_ALONE + ADDED_LEPTONIC, dt=1e3)
    a.set_mag_field(1.0)
    E = a.get_egrid_lep()
    a.set_t_pair_escape(np.where(E > 1e12, 1e-170, 1e6))
    a.set_injection_rate_electrons(
        np.where((E >= 1e9) & (E <= 1e13), 1e9 / E, 0.0)
    )
    with pytest.raises(Exception, match="solution for electron is NaN"):
        evolve(a, 5)


def test_spline_and_rk4_guards(tmp_path):
    """Three guards of the solver. A linear spline evaluated at NaN reports an
    error, as one outside its range does. (A range check that NaN passed let it
    on to the search, and the spline returned NaN without an error.) Where the
    summed steps round past the top of its range, rk4_gslspl_a ends its last
    step at that top. (The last step evaluated the spline past its top, an
    error, and the integral was NaN.) So does rk4_gslspl_q, with the sink and
    without it, where it is negligible. (Its last step went past the top of the
    source's spline, an error that empties the species for the step. It also
    went past the range of rk4_gslspl_a, an exception.)"""
    exe = harness.build("solver_mojo", tmp_path)
    error, value = harness.output(exe, "spline")
    assert error == 1 and np.isnan(value)
    error, value = harness.output(exe, "rk4")
    assert error == 0
    assert value == pytest.approx(-0.3 * np.exp(-1.0), rel=1e-12)
    k = np.exp(-1.0)
    error, value = harness.output(exe, "rk4q")
    assert error == 0
    assert value == pytest.approx(2 * -np.expm1(-0.3 * k) / k, rel=1e-6)
    error, value = harness.output(exe, "rk4q_no_sink")
    assert error == 0
    assert value == pytest.approx(2 * 0.3, rel=1e-12)


def test_hadron_timescales_after_the_hadronic_processes_are_switched_off():
    """After a step with the hadronic processes off, no cooling or loss of the
    hadrons acts. Their timescale getters read 1e200 s, as in a run with these
    processes off from the start. The rates are back in the next step with
    the processes. The arrays themselves keep the rates, which the proton
    maximum-energy estimate reads in the first step after the switch. (The
    getters reported the rates of the last step with the processes, for
    example 7.7e6 s for get_t_proton_bethe_heitler in the rich source.)"""
    getters = (
        "get_t_proton_syn",
        "get_t_proton_compton",
        "get_t_proton_bethe_heitler",
        "get_t_proton_photopion",
        "get_t_proton_proton_proton",
        "get_t_pion_syn",
        "get_t_pion_compton",
        "get_t_muon_syn",
        "get_t_muon_compton",
        "get_t_photon_photopion",
    )
    a = rich(new(pp=True), 3)
    a.set_pp_target_proton_density(1e8)
    evolve(a, 2)
    on = {g: getattr(a, g)() for g in getters}
    for g, t in on.items():
        assert t.min() < 1e100, g
    # (reported as those of the last step until a step runs without them)
    a.set_process_hadronic(False)
    for g, t in on.items():
        np.testing.assert_array_equal(getattr(a, g)(), t, err_msg=g)
    evolve(a, 1)
    for g in getters:
        np.testing.assert_array_equal(getattr(a, g)(), 1e200, err_msg=g)
    a.set_process_hadronic(True)
    evolve(a, 1)
    for g in getters:
        assert getattr(a, g)().min() < 1e100, g


def test_kaon_cooling_timescales_of_charm_decay_kaons():
    """With kaons off and charm hadrons on, all kaons come from charm decays. So
    the kaon cooling timescales are theirs. They are finite in every bin with
    kaons. (They were read from the solver of the photo-hadronic or pp kaons,
    which did not run: 1e200 s in every bin.)"""
    a = rich(new(bethe_heitler=False, kaons=False), 5)
    k = a.get_kaons() > 0
    assert np.count_nonzero(k) > 100
    for g in ("get_t_kaon_syn", "get_t_kaon_compton"):
        assert np.all(getattr(a, g)()[k] < 1e100), g


def kaons_in_100_gauss():
    a = new(grid=(1e-2, 1e6, 1e16))
    a.set_mag_field(100.0)
    eg = a.get_egrid_photons()
    photons = np.where((eg > 10) & (eg < 1e5), 1e10, 0.0)
    a.set_current_densities_photons(photons)
    a.set_injection_rate_photons(photons / 1e6)
    a.set_powerlaw_injection_parameters_protons(
        1.13e50, 1e44, 1e10, 1e10, 1e16, 2.0, 2.0, 1.0
    )
    return evolve(a, 5)


def test_kaon_and_charm_cooling_timescales_after_they_are_gone():
    """With no kaons or charm hadrons left to evolve, their cooling timescales
    are those of the current field. The K+- and D+- synchrotron timescales in
    1e-3 G are 1e10 times those in 100 G. (They were read from the solvers of
    the last step that evolved them, in 100 G: 3.5e6 s instead of 3.5e16 s for
    the kaons.)"""
    a = kaons_in_100_gauss()
    getters = ("get_t_kaon_syn", "get_t_charm_syn")
    t100 = [getattr(a, g)() for g in getters]
    a.set_process_kaons(False)
    a.set_process_charm(False)
    a.set_solver_time_step(1e9)
    evolve(a, 3)
    assert not a.get_kaons().any() and not a.get_charm_hadrons().any()
    a.set_mag_field(1e-3)
    a.evolve_step()
    for g, t in zip(getters, t100):
        np.testing.assert_allclose(
            getattr(a, g)(), 1e10 * t, rtol=1e-9, err_msg=g
        )


def test_kaon_cooling_timescales_with_the_hadronic_processes_off():
    """With the hadronic processes off, no kaons are evolved. The kaon cooling
    timescales are those of the current field, also with kaons of charm decays
    left. The K+- synchrotron timescales in 1e-3 G are 1e10 times those in
    100 G. (The kaons of the charm decays counted as evolved. So the timescales
    were read from the solver of these kaons, which did not run: 1.6e9 s instead
    of 1.6e19 s at 2.2e13 eV.)"""
    a = kaons_in_100_gauss()
    t100 = a.get_t_kaon_syn()
    a.set_process_hadronic(False)
    a.set_mag_field(1e-3)
    a.evolve_step()
    np.testing.assert_allclose(a.get_t_kaon_syn(), 1e10 * t100, rtol=1e-9)


def test_kaon_and_charm_timescales_after_a_clear():
    """After clear_am3(), with which every init_kernels() ends, the 17 kaon and
    charm timescales before the first step are those of a fresh object. (The
    clears kept the flags of the kaons and charm hadrons that the last step
    evolved, and the cooling getters follow these flags. So the getters read the
    zero rates of the rebuilt solvers: 1e200 s in every bin. A fresh object has
    3.4e13 s (K+- synchrotron) and 7.1e15 s (D+-) at the bottom of the grid.)"""
    fresh = rich(new())
    a = rich(new(), 5)
    quiet(a.clear_am3)
    rich(a)
    getters = [
        g
        for g in dir(a)
        if g.startswith(("get_t_kaon", "get_t_charm", "get_t_D"))
        or g == "get_t_Lambda_c_decay"
    ]
    assert len(getters) == 17
    for g in getters:
        np.testing.assert_array_equal(
            getattr(a, g)(), getattr(fresh, g)(), err_msg=g
        )


def test_rerun_after_clear_particle_densities(rich_run):
    """A run repeated after clear_particle_densities is that of a fresh object.
    Charm hadrons are on, and their decays feed the nucleons of the next step.
    (The nucleon sources of the last step's charm decays survived the clear. The
    neutrons went from none to 8e-7 at the first step, x29 at the second and
    +3 % after 30 steps.)"""
    a = rich(new(), 5)
    a.clear_particle_densities()
    assert_same_outputs(evolve(a, 5), rich_run)


def test_profile_timing_off_by_default(rich_run):
    """The profiler is off unless switched on (profile_timing is uninitialized
    in AM3): no times are taken in rich_run."""
    assert rich_run.get_profile_timing() is False
    assert not any(rich_run.get_profiled_times())


# ------------------------------------------------------------ e+e- annihilation
def test_annihilation_of_the_pair_components():
    """With e+e- annihilation, the pair components are evolved per charge. The
    injected component (only electrons) annihilates at the electrons' rate,
    dn/dt = q - Gamma_- n, while the gamma-gamma pairs annihilate on the
    positrons too. The components add up to the total. (Evolving the component
    as a whole at a mean rate puts 40 % too few pairs into it.)"""
    dt = 1.0
    a = new(
        off=(
            "hadronic",
            "escape",
            "electron_syn",
            "electron_compton",
            "ssa",
            "proton_syn",
            "expansion",
            "adiabatic_cooling",
            "triplet_pair_production",
        ),
        dt=dt,
    )
    El, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    il = np.arange(El.size)
    q = np.where((il >= 2) & (il <= 60), 1e12, 0.0)
    a.set_injection_rate_electrons(q)
    a.set_current_densities_photons(
        np.where(np.abs(np.log(Eg / MEC2_EV / 30)) < 0.35, 3e14, 0.0)
    )
    n = np.zeros(El.size)
    for _ in range(50):
        a.evolve_step()
        r = 1 / a.get_t_electron_annihilation()
        x = r * dt
        n = np.where(
            x < 0.1,
            (1 - x + x * x / 2) * n + (dt - dt * dt * r / 2) * q,
            np.exp(-x) * n + q / np.maximum(r, 1e-300) * (1 - np.exp(-x)),
        )
        n[:2] = 0
    ne, npos = a.get_electrons(), a.get_positrons()
    comp = a.get_pairs_injected()
    np.testing.assert_allclose(
        comp + a.get_pairs_annihilation(), ne + npos, rtol=1e-12
    )
    s = (npos > 1e-3 * npos.max()) & (il >= 2) & (il <= 58)
    assert s.sum() > 5
    np.testing.assert_allclose(comp[s], n[s], rtol=1e-2)


def test_annihilation_sink_of_a_charge_absent_at_the_start_of_a_step():
    """The positrons injected in a step annihilate on the electrons there at its
    start. They do so also when there were no positrons at the start (the
    first positrons of a run). Each charge's sink comes from the other's
    density, while the photon source needs both. (With both sinks zero when
    either charge is absent, the timescale getters report 3e177 s.)"""
    dt, q = 1e3, 1e9
    a = new(off=CLOSED_BOX + ("triplet_pair_production",), dt=dt)
    il = np.arange(a.get_egrid_lep().size)
    a.set_current_densities_electrons(np.where(il == 2, 1e12, 0.0))
    a.set_injection_rate_positrons(np.where(il == 2, q, 0.0))
    a.evolve_step()
    t = a.get_t_positron_annihilation()[2]
    assert t < dt * 10
    # no cooling or escape: the solver's escape-dominated regime,
    # dn/dt = q - n/t exactly
    np.testing.assert_allclose(
        a.get_positrons()[2], q * t * (1 - np.exp(-dt / t)), rtol=1e-2
    )
    # no sink for the electrons: there were no positrons at the start of the
    # step
    assert a.get_electrons()[2] == 1e12
    assert np.all(a.get_photons_pair_annihilation() == 0.0)


# -------------------------------------------------- energy of the products
DECAY_ALONE = (
    "photopion",
    "pp",
    "bethe_heitler",
    "escape",
    "expansion",
    "adiabatic_cooling",
    "electron_syn",
    "electron_compton",
    "annihilation",
    "proton_syn",
    "proton_compton",
) + ADDED_LEPTONIC


def test_pi0_decay_photons_carry_the_energy_of_the_pions():
    """With pi0 decay the only process, there are 2 photons per pi0, with
    cosh(dlnE / 2) = 1.00125 times the pions' energy. This is the energy of a
    spectrum flat in energy booked at the centers of its bins. (AM3 took the
    point values of the spectrum of a pi0 at (E_j + E_j+1) / 2. This gave 1.048
    times the energy and 1.9967 photons per pi0: AM3 bug.)"""
    a = new(off=DECAY_ALONE, dt=1.0, muon_decay=False)
    Eh, Eg = a.get_egrid_had(), a.get_egrid_photons()
    pions = power_law(1.0, 1e12, 1e16)(Eh)
    a.set_injection_rate_pi0(pions)
    a.evolve_step()
    photons = a.get_photons()
    assert number(photons) / number(pions) == pytest.approx(2.0, rel=1e-12)
    assert energy(photons, Eg) / energy(pions, Eh) == pytest.approx(
        np.cosh(DLNE / 2), rel=1e-4
    )


def test_number_split_keeps_number_and_energy(tmp_path):
    """number_split shares a product at the position x between the grid points
    around it, so that they hold its energy. This energy is e^(x dlnE) in
    units of the grid's first energy. The points also hold its number, as the
    shares add up to 1. (AM3's share of the upper point,
    f + f (1 - f) dlnE / 2 for the fraction f of x, is the weight of an
    interpolation. This share booked the photo-pion pions, kaons and charm
    hadrons up to 0.25 % too high in energy: AM3 bug.)"""
    x = np.array([-3.7, -0.5, 0.25, 0.5, 0.75, 2.9, 17.123])
    i, r = (
        harness.run("number_split_mojo", tmp_path, DLNE, *x.tolist())
        .reshape(-1, 2)
        .T
    )
    assert np.array_equal(i, np.floor(x))
    np.testing.assert_allclose(
        (1 - r) * np.exp(i * DLNE) + r * np.exp((i + 1) * DLNE),
        np.exp(x * DLNE),
        rtol=1e-14,
    )


def pp_photons(model, e_proton):
    a = new(off=DECAY_ALONE, pp=True, dt=1.0, init=False)
    a.set_pp_model_neutral_pions(model)
    quiet(a.init_kernels)
    a.set_pp_target_proton_density(1e6)
    a.set_current_densities_protons(line(1.0, e_proton)(a.get_egrid_had()))
    a.evolve_step()
    return a.get_photons()


def test_pp_photons_of_sibyll_and_qgsjet_below_100_gev():
    """Kafexhiu et al. (2014) give the pp photon spectrum of SIBYLL and QGSJET
    (models 0 and 1) from a proton kinetic energy of 100 GeV on. Below this
    energy, they give that of Geant4 (model 2). So protons at 67 GeV make the
    same photons with all three, unlike protons at 148 GeV. With Pythia 8
    (model 3, from 50 GeV on), these protons make other photons. (AM3 took
    the spectral shape of SIBYLL and QGSJET from 50 GeV on, with the peak of
    Geant4 up to 100 GeV. At 67 GeV, the photons had 1.44 and 1.03 times the
    energy of Geant4's: AM3 bug.)"""
    geant4 = pp_photons(2, 67e9)
    assert geant4.max() > 0
    for model in (0, 1):
        assert np.array_equal(pp_photons(model, 67e9), geant4), model
        assert number(pp_photons(model, 148e9)) > 1.1 * number(
            pp_photons(2, 148e9)
        )
    assert number(pp_photons(3, 67e9)) > 1.01 * number(geant4)


def test_pp_transition_energy_at_a_grid_point():
    """The transition energy of the pp delta approximation is set to the energy
    of a hadron grid point. The pp loss times of the protons are then those of
    an energy 1e-12 above that point. The point belongs to the range above the
    transition. (The comparison of the point's energy with the transition energy
    was exact. Rounding put the point below it for 52 of 87 points, and the loss
    rate of that bin changed by up to 0.9 %.)"""
    a = new(off=("photopion", "bethe_heitler"), pp=True, init=False)
    E = a.get_egrid_had()

    def loss_times(e_trans):
        a.set_pp_transition_energy_to_delta_approx(e_trans)
        quiet(a.init_kernels)
        a.set_pp_target_proton_density(1.0)
        a.set_current_densities_protons(np.where(E > 2e9, 1.0, 0.0))
        a.evolve_step()
        return a.get_t_proton_proton_proton()

    for k in (40, 52, 61, 70, 73, 76):
        assert np.array_equal(
            loss_times(E[k]), loss_times(E[k] * (1 + 1e-12))
        ), k


# ------------------------------------------------------------ the interface
def test_bethe_heitler_minimum_one_bin_above_the_grid():
    """A minimum proton energy of the Bethe-Heitler pairs one bin above the
    top of the default hadron grid reads back as that energy. Here, the test
    computes this minimum from the top's energy. It leaves no protons, as
    eV_to_had_grid_index places it. (It was compared exactly with the energy
    one bin above the top as the code computes it, an ulp higher. So the
    minimum became the top bin.)"""
    a = new(init=False)
    above = a.get_egrid_had()[-1] * np.exp(DLNE)
    assert a.eV_to_had_grid_index(above) == a.get_grid_size_had()
    a.set_optimize_bethe_heitler_incoming_protons_min(above)
    assert a.get_optimize_bethe_heitler_incoming_protons_min() == pytest.approx(
        above, rel=1e-12
    )


@pytest.mark.parametrize(
    "grid",
    [(1e-3, 1e-300, 1e15), (1e-304, 1e2, 1e15), (1e-101, 1e2, 1e15)],
)
def test_update_energy_grid_minima_far_below_any_particle(grid):
    """update_energy_grid raises for a lowest photon or neutrino energy below
    1e-100 eV and leaves the grids as they are. At 1e-100 eV, it makes the
    grids, which start at their grid point at or above it (1.0026e-100 eV).
    (Any positive minimum passed. For 1e-300 eV, the number of bins
    overflowed and the process crashed. From 1e-155 eV down, the first step
    stopped with rates that are not finite.)"""
    a = new(init=False)
    sizes = a.get_egrid_photons().size, a.get_egrid_neutrinos().size
    with pytest.raises(Exception, match="1e-100 eV"):
        a.update_energy_grid(*grid)
    assert (a.get_egrid_photons().size, a.get_egrid_neutrinos().size) == sizes
    a.update_energy_grid(1e-100, 1e-100, 1e15)
    for E in (a.get_egrid_photons(), a.get_egrid_neutrinos()):
        assert 1e-100 <= E[0] < 1e-100 * np.exp(DLNE)


DECAY_GETTERS = {
    "pion_decay": ("get_t_pion_decay",),
    "muon_decay": ("get_t_muon_decay",),
    "neutron_decay": ("get_t_neutron_decay",),
    "kaon_decay": (
        "get_t_kaon_decay",
        "get_t_kaon_0S_decay",
        "get_t_kaon_0L_decay",
    ),
    "charm_decay": (
        "get_t_D_plus_decay",
        "get_t_D_zero_decay",
        "get_t_Ds_decay",
        "get_t_Lambda_c_decay",
    ),
}


@pytest.mark.parametrize("switch", DECAY_GETTERS)
def test_decay_timescales_follow_the_decay_switches(switch):
    """With a decay switched off, the getters of its decay times report none
    (1e200 s), as the evolution applies none. The getters of the other
    decays report theirs, before and after a step. (They reported the decay
    time whatever the switch, as AM3 does for the pions and muons: 1.9e-7 s
    for pions that did not decay.)"""
    a = new(**{switch: False})
    for _ in range(2):
        for name, getters in DECAY_GETTERS.items():
            for getter in getters:
                t = getattr(a, getter)()
                if name == switch:
                    assert np.all(t == 1e200), getter
                else:
                    assert 0 < t.min() < 1e4, getter
        a.evolve_step()


def test_decay_timescales_without_the_decay_tables():
    """A decay switched on whose tables init_kernels did not build does not act.
    Its getters report no decay (1e200 s), before and after a step. Kaon and
    charm decay are on, with kaons and charm off at init_kernels, which builds
    their decay tables only with them. (The getters followed the switches alone:
    935 s for neutrons that did not decay.)"""
    a = new(neutron_decay=False, kaons=False, charm=False)
    a.set_process_neutron_decay(True)
    assert a.get_process_kaon_decay() and a.get_process_charm_decay()
    for _ in range(2):
        for name in ("neutron_decay", "kaon_decay", "charm_decay"):
            for getter in DECAY_GETTERS[name]:
                assert np.all(getattr(a, getter)() == 1e200), getter
        assert a.get_t_pion_decay().min() < 1e4
        a.evolve_step()


def test_second_init_of_an_object():
    """AM3.__init__ called on an object that holds a simulation raises and
    leaves the object as it is. (It wrote a new simulation over the old one
    without freeing it, 100 MB of kernels and arrays per call. It also reset
    every setting without a word.)"""
    a = new()
    a.set_mag_field(3.0)
    with pytest.raises(Exception, match="initialized already"):
        a.__init__()
    assert a.get_mag_field() == 3.0
    a.evolve_step()
