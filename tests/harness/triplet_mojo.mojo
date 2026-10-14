"""Prints what am3_mojo's triplet pair production (physics/triplet.mojo) computes, for the unit tests and the triplet
checks (tests/support/harness.py, Triplet). The first argument selects the output:
  tables        TPP_LY0, TPP_DLY and the number of table nodes; per node: k, sigma(k) / (alpha r_e^2), the number
                of ln y bins nb, then the tabulated dN/dln y of the positron, the lower-energy electron and the
                recoil (y = xi k), nb values each
  sigma k ...   sigma(k) / (alpha r_e^2) at these k (the threshold expansion, the spline between the nodes, the
                high-energy expansion)
  dist k ...    per k: nb, then distributions(k), the three products' dN/dln y as the code takes them (interpolated
                between the nodes, extrapolated above the last, normalised to one per interaction)
  rows          the kernels of the default grid: m0, dlnE and the number of rows; per row q (P = gamma eps =
                exp((m0 + q) dlnE)): P, the rate R [sigma_T], the energy-loss rate Lxi and nd, then the deposits
                of the positron and of the lower-energy electron at the offsets 0 .. nd - 1 below the target bin
  extrap k_lo k_hi k ...   the tables cut to the nodes k <= k_hi and the asymptotic laws fitted to the nodes from
                k_lo on (as the code fits them from K_FIT on), then per k as `dist`."""

from std.sys import argv

from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.physics.triplet import TripletPairProduction, N_KR, N_ROLES, tpp_nbins
from am3_mojo.physics.triplet_tables import TPP_LY0, TPP_DLY
from am3_mojo.utils.buffers import Vec, VecI
from am3_mojo.utils.libm import c_exp


def print_distributions(T: TripletPairProduction, k: Float64):
    var out = Vec()
    T.distributions(k, out)
    print(tpp_nbins(k))
    for v in out:
        print(v)


def main() raises:
    var args = argv()
    var mode = String(args[1])
    var dat = AM3Arrays()
    var T = TripletPairProduction(dat, False)
    T._parse_tables()
    if mode == "tables":
        var N = len(T.tk)
        print(TPP_LY0, TPP_DLY, N)
        for n in range(N):
            var nb = tpp_nbins(T.tk[n])
            print(T.tk[n], T.tsig[n], nb)
            for r in range(N_ROLES):
                for b in range(nb):
                    print(T._node(n, r, b))
    elif mode == "sigma":
        for i in range(2, len(args)):
            print(T.sigma(Float64(String(args[i]))))
    elif mode == "dist":
        for i in range(2, len(args)):
            print_distributions(T, Float64(String(args[i])))
    elif mode == "rows":
        T.initialize_kernels(dat)
        var dl = dat.dlnE
        print(T.m0, dl, len(T.R))
        for q in range(len(T.R)):
            print(c_exp(Float64(T.m0 + q) * dl), T.R[q], T.Lxi[q], T.nd[q])
            for r in range(N_KR):
                for o in range(T.nd[q]):
                    print(T.W[r][T.woff[q] + o])
    elif mode == "extrap":
        var k_lo = Float64(String(args[2]))
        var k_hi = Float64(String(args[3]))
        var tk = Vec()
        var tsig = Vec()
        var toff = VecI()
        for n in range(len(T.tk)):
            if T.tk[n] <= k_hi:
                tk.append(T.tk[n])
                tsig.append(T.tsig[n])
                toff.append(T.toff[n])
        T.tk = tk^
        T.tsig = tsig^
        T.toff = toff^
        T._fit_asymptotics(k_lo)
        for i in range(4, len(args)):
            print_distributions(T, Float64(String(args[i])))
    else:
        raise Error("unknown mode: " + mode)
