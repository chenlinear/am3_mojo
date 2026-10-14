"""The kaon production channels of physics/photopion.mojo (`kaon_production_channels`):

* the two-body channels gamma N -> K Y against the measured angular distributions and cross sections they are taken
  from (tests/support/measured/ky_angular.py): the energy fraction chi and the cross section of every step and
  channel column (p -> K+, p -> K0, n -> K+, n -> K0);
* the kaons of the multi-pion types: in steps of the photon energy, made only where a kaon comes with a third
  particle, so that they do not repeat the two-body channels; the two steps of type M4 (5-50 GeV) against the
  strange particles measured at 20 GeV (tests/support/measured/strange_20gev.py), in number and in energy.
"""
import numpy as np
import pytest

from support import harness
from support.measured import ky_angular as ky
from support.measured import strange_20gev as g20
from support.paths import load_derivation

derivation = load_derivation('kaon_production')   # the README's thresholds and multi-pion types

KP, KM, KS, KL = range(4)
# the multi-pion types M2 ... M7 of Hümmer et al. (2010, Table 6) as the README gives them (photon energies [GeV]),
# and the steps of the README's table, (t, eps_min, eps_max) of type M(t + 1)
TYPE_RANGE = {t: r for t, r in derivation.readme_type_ranges().items() if t >= 1}
MPI_STEPS = derivation.readme_mpi_steps()
K0S = derivation.numeric_neutral()   # the K0_S among K0 and anti-K0


def steps_of(t):
    """The README's steps of type M(t + 1), (eps_min, eps_max) in order."""
    return [(lo, hi) for tt, lo, hi in MPI_STEPS if tt == t]


@pytest.fixture(scope='module')
def tables(tmp_path_factory):
    """The lines of the harness: the kaon channels, and AM3's multi-pion entries ("pi ...")."""
    return harness.run('kaon_channels_mojo', tmp_path_factory.mktemp('harness'), text=True)


@pytest.fixture(scope='module')
def all_channels(tables):
    """Rows emin, emax, chi_p[4], chi_n[4], mp[4], mn[4], sigma, mtype of every kaon channel."""
    return np.array([[float(x) for x in line.split()] for line in tables if not line.startswith('pi')])


@pytest.fixture(scope='module')
def multi_pion(tables):
    """Rows emin, emax, sigma, chi, M(pi0, pi+, pi-) from protons, from neutrons of AM3's 14 multi-pion
    entries (two per type: the low-energy and the leading pions)."""
    rows = np.array([[float(x) for x in line.split()[2:]] for line in tables if line.startswith('pi')])
    assert rows.shape == (14, 10)
    return rows


@pytest.fixture(scope='module')
def channels(all_channels):
    """The two-body steps."""
    return all_channels[all_channels[:, -1] == -1]


def columns(row):
    """chi and cross section [mubarn] of the four channel columns of a step."""
    chi_p, chi_n, mp, mn = row[2:6], row[6:10], row[10:14], row[14:18]
    return {'p->K+': (chi_p[KP], mp[KP]), 'p->K0': (chi_p[KS], mp[KS] + mp[KL]),
            'n->K+': (chi_n[KP], mn[KP]), 'n->K0': (chi_n[KS], mn[KS] + mn[KL])}


@pytest.mark.parametrize('column', list(ky.COLUMNS))
def test_chi_from_the_measured_angular_distributions(channels, column):
    for step, row in enumerate(channels):
        chi, sigma = columns(row)[column]
        if sigma == 0.0:
            continue
        # rounded to two decimals
        assert chi == pytest.approx(ky.step_chi(column, step), abs=0.005 * 1.001), (column, ky.STEPS[step])
        # neutral kaons: the same chi for K0_S and K0_L
        chi_p, chi_n = row[2:6], row[6:10]
        assert chi_p[KS] == chi_p[KL] and chi_n[KS] == chi_n[KL]


@pytest.mark.parametrize('column', list(ky.COLUMNS))
def test_cross_sections_from_the_measurements(channels, column):
    """The steps of the measured cross sections, and their means, integrated
    over angle, rounded to two decimals (two significant digits below 0.1
    mubarn)."""
    assert [tuple(r[:2]) for r in channels] == ky.STEPS
    for step, row in enumerate(channels):
        sigma = columns(row)[column][1]
        want = ky.column_sigma(column, step)
        tol = 0.005 if want >= 0.1 else 0.0005
        assert sigma == pytest.approx(want, abs=tol * 1.001), (column, ky.STEPS[step])


def test_multi_pion_kaon_steps(all_channels, multi_pion):
    """The kaons of the multi-pion types are those made with further particles (K Y pi, K anti-K N, ...). The
    two-body reactions gamma N -> K Lambda, K Sigma have their own channels, with the measured cross sections.
    Below the threshold of gamma N -> K Lambda pi (1.15 GeV) only the two-body reactions make a kaon, so the kaons
    of type M2 (0.9-1.5 GeV) start there. Strange-particle production with further particles rises most across
    types M3 (1.5-5 GeV) and M4 (5-50 GeV), so the kaons of each are in two steps (1.5-2.5 and 2.5-5 GeV; 5-15 and
    15-50 GeV). From step to step, protons make more of every kaon species per interaction."""
    multi = all_channels[all_channels[:, -1] >= 0]
    assert sorted(set(multi[:, -1])) == sorted(TYPE_RANGE)
    # gamma N -> K Lambda pi in its four charge states (the README's s, PDG 2026 masses)
    three_body = derivation.threshold_span(derivation.NUCLEON, derivation.KAON, ('Lambda',), derivation.PION)
    start = multi[:, 0].min()
    assert three_body[2] == 4 and start <= three_body[0] < 1.01 * start  # 1.15 GeV; 1.152-1.168 GeV by charge state
    assert start > max(ky.threshold(r) for r in ky.REACTIONS)  # above every two-body threshold (0.91-1.05 GeV)
    for t, (lo, hi) in TYPE_RANGE.items():
        # AM3's table has the type's range and cross section (Hümmer et al., Table 6)
        assert tuple(multi_pion[2 * t, :2]) == tuple(multi_pion[2 * t + 1, :2]) == (lo, hi)
        mine = multi[multi[:, -1] == t]
        assert np.all(mine[:, 18] == multi_pion[2 * t, 2])
        # the type's kaons are in consecutive steps from the type's start (or the three-body threshold) to its
        # end, each with a low-energy and a leading part
        steps = sorted(set(map(tuple, mine[:, :2])))
        assert steps[0][0] == max(lo, start) and steps[-1][1] == hi
        assert all(a[1] == b[0] for a, b in zip(steps[:-1], steps[1:]))
        assert all(np.sum(np.all(mine[:, :2] == st, axis=1)) == 2 for st in steps)
    for row in multi:
        # antikaons need a kaon as well: not below the threshold of gamma p -> p K+ K- (1.507 GeV)
        if row[10 + KM] > 0 or row[14 + KM] > 0:
            assert row[0] >= 0.99 * derivation.threshold(ky.MP, ky.MP, ky.MKP, ky.MKP)
    # the two-body steps start at their thresholds
    assert all_channels[all_channels[:, -1] == -1][:, 0].min() == pytest.approx(ky.threshold('K+L'), abs=2e-3)
    # the steps rise in every species
    steps = sorted(set(map(tuple, multi[:, :2])))
    assert steps == sorted((lo, hi) for _, lo, hi in derivation.readme_mpi_steps())   # the README's table
    per_step = [multi[np.all(multi[:, :2] == st, axis=1)][:, 10:14].sum(axis=0) for st in steps]
    for a, b in zip(per_step[:-1], per_step[1:]):
        assert np.all(b > a), (a, b)
    # one kaon for every antikaon at least (strangeness), and K0 or anti-K0 as K0_S and K0_L in equal parts
    for m in per_step:
        assert m[KP] + m[KS] + m[KL] >= 2 * m[KM] and m[KS] == m[KL]


def proton_kaons(row):
    """Cross sections [mubarn] of (K+, K0, K-, anti-K0) of a channel for protons. A multi-pion channel makes
    the isospin mirror from neutrons (K+ <-> K0, K- <-> anti-K0), which tells its K0 from its anti-K0; a
    two-body step makes no antikaons."""
    sigma, mp, mn = row[18], row[10:14], row[14:18]
    if row[-1] < 0:
        return sigma * np.array([mp[KP], mp[KS] + mp[KL], 0.0, 0.0])
    assert mp[KS] + mp[KL] == pytest.approx(mn[KP] + mn[KM], rel=1e-12)
    return sigma * np.array([mp[KP], mn[KP], mp[KM], mn[KM]])


def test_m4_steps_from_the_strange_particles_at_20_gev(all_channels, multi_pion):
    """The kaons of type M4 (5-50 GeV), in two steps (5-15 and 15-50 GeV), against the strange particles measured
    at 20 GeV (Abe et al. 1985 and 1984; tests/support/measured/strange_20gev.py).

    Numbers: every species of a step lies, to the rounding of the multiplicities, on a line in the logarithm of the
    photon energy, taken at the step's middle. The line runs through the code's kaons at 2.5-5 GeV (the upper step
    of M3, with its two-body kaons) and the species that balance the strange particles measured at 20 GeV.

    Energy: the K0_S of a step have the mean energy fraction chi of the x and pT^2 distribution measured at 20 GeV,
    taken to the step's middle by Feynman scaling. Their leading part has about the measured share and energy of the
    K0_S that go towards the nucleon (to 0.03 and 15 %)."""
    rows = [(r[0], r[1], proton_kaons(r)) for r in all_channels]
    m3_upper = steps_of(2)[1]
    low, e_low = g20.ln_mean(rows, *m3_upper), np.sqrt(m3_upper[0] * m3_upper[1])
    assert np.all(multi_pion[6:8, :2] == TYPE_RANGE[3])
    sigma, chi_pions = multi_pion[6, 2], multi_pion[6:8, 3]  # of M4; chi of its low-energy and leading pions
    centres = {}
    for lo, hi in steps_of(3):
        mine = all_channels[(all_channels[:, 0] == lo) & (all_channels[:, 1] == hi)]
        assert len(mine) == 2 and np.all(mine[:, -1] == 3) and np.all(mine[:, 18] == sigma)
        centre = centres[lo] = np.sqrt(lo * hi)
        code = sum(proton_kaons(r) for r in mine)
        want = g20.trend(centre, e_low, low)
        # the multiplicities per interaction are rounded to four decimals
        assert np.all(np.abs(code - want) <= 0.5e-4 * 1.001 * sigma), (lo, hi, code, want)
        # the mean chi of the K0_S (from protons), and their leading part (the channel with the leading pions' chi)
        ks, chi = mine[:, 10 + KS], mine[:, 2 + KS]
        assert np.dot(ks, chi) / ks.sum() == pytest.approx(g20.mean_chi(centre)[0], rel=0.02)
        lead = chi == chi_pions[1]
        assert lead.sum() == 1 and chi[~lead][0] < chi[lead][0]
        lead_chi, lead_share = g20.mean_chi(centre, x_below=-0.2)
        assert ks[lead][0] / ks.sum() == pytest.approx(lead_share, abs=0.03)
        assert chi[lead][0] == pytest.approx(lead_chi, rel=0.15)
    # at 20 GeV itself (in the upper step, which stands for 15-50 GeV, where the yields go on rising)
    at20 = sum(proton_kaons(r) for r in all_channels if r[0] <= g20.E_GAMMA < r[1])
    assert K0S(at20[1], at20[3]) == pytest.approx(g20.K0S[0], rel=0.15)  # 10.7 against 9.66 mubarn
    assert at20[0] + at20[1] - at20[2] - at20[3] == pytest.approx(g20.hyperon_strangeness(), rel=0.06)  # 8.6, 8.5
    # the kaons of the next type (M5, 50-500 GeV, set independently) follow the same line: the kaons, the antikaons
    # and the K0_S within 12 % (6, 6 and 11 %), each species within 20 % (the K0 18 %: the line keeps the excess of
    # K+ over K0 rising, M5 has less)
    m5 = sum(proton_kaons(r) for r in all_channels if r[-1] == 4)
    line = g20.trend(np.sqrt(TYPE_RANGE[4][0] * TYPE_RANGE[4][1]), e_low, low)
    for part in ([0, 1], [2, 3], [1, 3]):
        assert m5[part].sum() == pytest.approx(line[part].sum(), rel=0.12)
    assert np.all(np.abs(m5 / line - 1.0) < 0.20)
