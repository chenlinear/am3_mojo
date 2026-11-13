"""Prints what am3_mojo's Bethe-Heitler and gamma-gamma pair kernels hold on the
default grids, for tests/unit/test_fixes.py. Argument: bethe_heitler,
bethe_heitler_emission, gamma_gamma or gamma_gamma_limits.

bethe_heitler: the number of values of the packed pair kernel after
reinitialize_empty_kernels (as init_kernels with the process off) and after
initialize_kernels with the default limits (proton bins from 70, photon bins
below 280). Then it prints the number of its entries (e+- bin i, proton bin j,
photon bin k) that differ from (i + 1, j + 1, k - 1).
bethe_heitler_emission: for photons and protons of density 1/E on the hadron
grid and the default limits, the largest relative difference between the pair
emission (Calc_BH_Rad) and the sum over the whole kernel in float64. The
entries of this kernel come from bh_kernal_elec_gen once per diagonal
(i - j, j + k). Then it prints the share of that sum from the saturated
diagonals j + k >= sat_s, and their share of the kernel's non-zero entries
(in float32).
gamma_gamma: for the photon loss kernel not built (reinitialize_empty_kernels
alone) and built (initialize_kernels with the eight-bin approximation), the
full pair emission (Calc_Pair_Lep) of photons at 30 m_e c^2. It prints whether
the emission built the full kernel, the number of values of this kernel and the
largest pair source.
gamma_gamma_limits: for photons of density eps^-0.6
(1e-10 < eps < 1e10 m_e c^2), the sums of the full pair kernel within its
integration limits (outerPP, innerPP) and from the lowest photon bins on. Both
sums are taken before their normalization, over the photon pairs that
pair_lep_work takes (x x1 < e^13.8, energies in m_e c^2). It prints the number
of lepton bins from 100 on (above 11 GeV) where the sums differ by 1e-12 or
more (or are not finite). Then it prints their largest relative difference
there, and the number of bins compared. These are the bins where either sum is
not zero: up to bin 229, as the photons end at 1e10 m_e c^2. Below bin 100, the
sums agree to rounding, except in bin 23 (gamma = 10). There the sum from the
lowest bins is 2.8e-3 larger. For gp > 10 >= gm, the lower bound of x and x1
in RggD1 takes 1 / (4 gp) for (gp - sqrt(gp^2 - 1)) / 2. This value is a little
less, so the bound admits two pairs below the kinematic limit. The integration
limits leave these pairs out."""

from std.math import isnan
from std.sys import argv

from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import (
    P_protons,
    P_photons,
    P_photons_on_had_grid,
)
from am3_mojo.model.runparams import RunParams
from am3_mojo.physics.bethe_heitler import bh_kernal_elec_gen, BetheHeitler
from am3_mojo.physics.pair_production import (
    PairInitJob,
    pair_init_work,
    PairLepJob,
    pair_lep_work,
    Pair,
)
from am3_mojo.utils.buffers import zeros, zerosi
from am3_mojo.utils.constants import (
    PI,
    SIGMA_T,
    C_LIGHT,
    MASS_E,
    ALPHA,
    MASS_P,
)
from am3_mojo.utils.libm import c_pow
from am3_mojo.utils.linalg import MatI
from am3_mojo.utils.parallel import default_num_threads, parallel_run


def gamma_gamma_limits(mut dat: AM3Arrays):
    var rp = RunParams()
    var pair = Pair(rp, dat, False)
    pair.nthreads = default_num_threads()
    pair.reinitialize_empty_kernels(dat, False)
    pair.initialize_kernels(rp, dat, False)
    for j in range(dat.nbins_pho):
        var e = dat.egrid_pho[j]
        dat.n[P_photons][j] = c_pow(e, -0.6) if (
            e > 1e-10 and e < 1e10
        ) else 0.0
    pair.Calc_Pair_Phot(dat)
    pair.Calc_Pair_Lep(dat, 138)
    var job = PairLepJob(
        Pointer(to=pair).unsafe_origin_cast[MutUntrackedOrigin](),
        Pointer(to=dat).unsafe_origin_cast[MutUntrackedOrigin](),
        138,
    )
    parallel_run[pair_lep_work](job, pair.nthreads)
    var within = pair.epspairL.copy()
    pair.KER_outerPP = zerosi(dat.nbins_lep)
    pair.KER_innerPP = MatI(dat.nbins_lep, dat.nbins_pho)
    var init = PairInitJob(
        Pointer(to=pair).unsafe_origin_cast[MutUntrackedOrigin](),
        Pointer(to=dat).unsafe_origin_cast[MutUntrackedOrigin](),
    )
    parallel_run[pair_init_work](init, pair.nthreads)
    parallel_run[pair_lep_work](job, pair.nthreads)
    var off = 0
    var worst = 0.0
    var compared = 0
    for i in range(100, dat.nbins_lep - 1):
        var every = pair.epspairL[i]
        if within[i] != 0.0 or every != 0.0:
            compared += 1
            var d = abs(within[i] / every - 1.0)
            if not (d < 1e-12):
                off += 1
            worst = max(worst, d)
    print(off, worst, compared)


def bethe_heitler_emission(mut dat: AM3Arrays) raises:
    var nh = dat.nbins_had
    var nl = dat.nbins_lep
    var pmin = 70
    var kmax = min(280, dat.nbins_pho_hadgrid)
    for k in range(dat.nbins_pho_hadgrid):
        dat.n[P_photons_on_had_grid][k] = 1.0 / dat.egrid_had_photons[k]
    for j in range(nh):
        dat.n[P_protons][j] = 1.0 / dat.egrid_had[j]
    var bh = BetheHeitler(dat, False)
    bh.nthreads = default_num_threads()
    bh.reinitialize_empty_kernels(dat)
    bh.initialize_kernels(dat, pmin, 280)
    bh.Calc_BH_Rad(dat)
    var d0 = 1 - nh
    var ns = nh - pmin + kmax - 1
    var nd = nl - pmin - d0
    var T = zeros(nd * ns)
    for r in range(nd):
        var d = d0 + r
        var jlo = max(pmin, -d)
        var jhi = min(nh, nl - d)
        for s in range(jlo, jhi + kmax - 1):
            var j = max(jlo, s - kmax + 1)
            T[r * ns + s - pmin] = bh_kernal_elec_gen(
                dat.egrid_had_leptons[d + j] / MASS_E,
                dat.egrid_had[j] / MASS_P,
                dat.egrid_had_photons[s - j] / MASS_E,
            )
    var k_inj = C_LIGHT * 3.0 / (16.0 * PI) * SIGMA_T * ALPHA
    var q = zeros(nl)
    var tot = 0.0
    var sat = 0.0
    var nz = 0
    var nz_sat = 0
    for i in range(nl):
        for j in range(pmin, nh):
            for k in range(kmax):
                var t = T[(i - j - d0) * ns + j + k - pmin]
                var c = (
                    t * dat.n[P_protons][j] * dat.n[P_photons_on_had_grid][k]
                )
                q[i] += c
                tot += c
                var nonzero = Int(Float32(t) != 0.0)
                nz += nonzero
                if j + k >= bh.sat_s:
                    sat += c
                    nz_sat += nonzero
        q[i] *= dat.dlnE * dat.dlnE * k_inj
    var qmax = 0.0
    for i in range(nl):
        qmax = max(qmax, q[i])
    var worst = 0.0
    for i in range(nl):
        if q[i] > 1e-10 * qmax:
            var d = abs(bh.QBH_LGen[i] / q[i] - 1.0)
            # (max() drops a NaN deviation)
            worst = d if d > worst or isnan(d) else worst
    print(worst, sat / tot, Float64(nz_sat) / Float64(nz))


def main() raises:
    var args = argv()
    var dat = AM3Arrays()
    if args[1] == "bethe_heitler_emission":
        bethe_heitler_emission(dat)
        return
    if args[1] == "gamma_gamma_limits":
        gamma_gamma_limits(dat)
        return
    if args[1] == "bethe_heitler":
        var bh = BetheHeitler(dat, False)
        bh.nthreads = default_num_threads()
        bh.reinitialize_empty_kernels(dat)
        print(len(bh.packed.data))
        bh.initialize_kernels(dat, 70, 280)
        print(len(bh.packed.data))
        var n = 0
        ref K = bh.packed
        for i in range(K.nmat - 1):
            for k in range(1, K.cols):
                for j in range(70, K.rows - 1):
                    if K.get(i, j, k) != K.get(i + 1, j + 1, k - 1):
                        n += 1
        print(n)
        return
    var rp = RunParams()
    # photons at 30 m_e c^2
    var k30 = dat.ind_min_energy_pho + 34
    for built in range(2):
        var pair = Pair(rp, dat, False)
        pair.nthreads = default_num_threads()
        pair.reinitialize_empty_kernels(dat, False)
        if built == 1:
            pair.initialize_kernels(rp, dat, True)
        dat.n[P_photons][k30] = 1e10
        pair.Calc_Pair_Phot(dat)
        pair.Calc_Pair_Lep(dat, 138)
        var size = 0
        for v in pair.KER_RggD1:
            size += len(v)
        var qmax = 0.0
        for q in pair.epspairL:
            qmax = max(qmax, q)
        print(Int(pair.full_built), size, qmax)
