"""Photo-hadronic kaons, kaons/README.md "Kaons in multi-pion production": the
kaons of the multi-pion types M5 (50-500 GeV), M6 (500-5000 GeV) and M7
(5000 GeV to 5e6 GeV). They follow from the kaons per pion of hadron collisions.
They are compared with the strange particles measured at 20 GeV by the SLAC
Hybrid Facility Photon Collaboration (tests/support/measured/strange_20gev.py:
Abe et al. 1985, Phys. Rev. D 32, 2869, and Abe et al. 1984,
Phys. Rev. D 29, 1877).

  * Kaons per pion. NA49 measured pp collisions at the center-of-momentum energy
    sqrt(s) = 17.3 GeV (Anticic et al. 2010, arXiv:1004.1889). In full phase
    space, that is, over all momenta, it found 0.066 charged kaons per charged
    pion (Table 6). NA27 measured 0.075 at 27.4 GeV (Aguilar-Benitez et al.
    1991, doi:10.1007/BF01551452, compiled by Becattini & Heinz 1997,
    arXiv:hep-ph/9702274, Table 2). In the central region of anti-p p
    collisions, around rapidity zero, UA5 derives the ratio from the K0_S
    (Alner et al. 1985, doi:10.1016/0550-3213(85)90624-8, and
    Ansorge et al. 1987, doi:10.1016/0370-2693(87)91380-3). It finds
    0.092 +- 0.015, 0.095 +- 0.010 and 0.105 +- 0.010 at 200, 546 and 900 GeV.
    UA5 also quotes about 0.08 at the energies of CERN's Intersecting Storage
    Rings (ISR). Photons can make more. At 20 GeV, Abe et al. 1984 find about
    20 % more strange particles per event in gamma p collisions than in pi+- p
    collisions. The pion beam has the momentum 18.5 GeV/c. The middles of M5, M6
    and M7 are at sqrt(s) = 17.3, 54.5 and 545 GeV. At these energies, the code
    takes 0.070, 0.085 and 0.100 kaons per pion of the type, before the energy
    bookkeeping (kaons/README.md) scales the type's pions.
  * Kaons and antikaons. The kaons exceed the antikaons by the hyperons less the
    antihyperons, whose strangeness they balance. In M5, M6 and M7, the excess
    is 0.080, 0.090 and 0.100 per interaction. At 20 GeV, these hyperons are
    0.074 per interaction. There, a K0 comes with 0.40 of the Lambda (from the
    K0_S Lambda pairs). It comes with 0.38 of all hyperons (the charged Sigma as
    in ABBHHM's events). So the kaons that balance the hyperons give 0.015 or
    0.017 more K+ than K0 per interaction. M5-M7 have 0.015 more K+ than K0, as
    if their kaon pairs had as many K0 as K+. They also have 0.005 more K- than
    anti-K0. With these differences, the kaons of a type split into the four
    species of its row.
  * The leading shares. The leading part holds the share h+ of the K+ and K0 and
    h- = 0.6 h+ of the antikaons (kaons/README.md). NA49 finds a similar ratio
    in pp collisions at 17.3 GeV. Its Table 5, with the tails extrapolated
    exponentially (na49_leading), gives the kaons at x_F > 0.2. Like
    Feynman's x of m4_steps.py, x_F is the longitudinal momentum along one
    proton over its maximum. The share of the K- there among all K- is 0.51
    times the share of the K+ there among all K+.
  * Energy. Feynman scaling keeps the K0_S distribution of 20 GeV in x and pT^2
    unchanged. With this scaling, <chi> is 0.060, 0.038 and 0.022 at the middles
    of M5, M6 and M7. The code has 0.059, 0.031 and 0.020 there. The K0 and
    anti-K0 of the low part are at f = 1.6 x the chi of the type's low-energy
    pions. The shares h+ and h- of them in the leading part are at the chi of
    the type's leading pions. Also printed: the h+ (with h- = 0.6 h+) that gives
    the code's K0_S that mean.
  * Number. The code's leading kaons, 0.036, 0.052 and 0.060 per interaction,
    are about as many as the data have going toward the nucleon. The data give
    0.034 from the K0_S at x < -0.2 at 20 GeV. NA49's K+- at x_F > 0.2 give
    0.055 per hemisphere of a pp collision. Both figures count as many neutral
    kaons as charged ones (isospin). The leading part has the chi of the leading
    pions, 0.5 in M5 and M6 and 0.6 in M7. This is above the 0.36 measured, so
    the number and the energy cannot both match. Also printed: what M5, M6 and
    M7 have with the h+ of the mean chi.

The code's tables are read from src/ by code_tables.py."""

import math, os, sys

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
from code_tables import code_it3, code_mpi
from support.measured import strange_20gev as g20
from support.paths import load_derivation

# the cells of a TeX table row (fields)
derivation = load_derivation("kaon_production")

KAI = code_it3("IT3kai")
# [t, eps_min, eps_max, K+, K0, K-, anti-K0, h+, h-, f]
MPI = [r for r in code_mpi() if int(r[0]) >= 4]
H_RATIO = 0.6  # h- / h+ (kaons/README.md)

# NA49 (Anticic et al. 2010), pp at 158 GeV/c (sqrt(s) = 17.3 GeV), full phase
# space. Table 6: the mean multiplicities of the charged kaons (positives,
# negatives, total). Table 5, from x_F = 0.2: per x_F, F, dn/dx_F, <pT> and
# <pT^2> of K+, then of K- (each followed by its statistical error in %), then
# y and dn/dy of K+ and K-.
NA49_T6 = r"$\langle n_{\textrm{K}} \rangle$ & 0.227 & 0.130 & 0.357 \\"
NA49_T5 = [
    r"0.2 & 0.3614 & 1.07 & 0.1635 & 1.06 & 0.5037 & 0.69 & 0.3449 &"
    r" 1.58 & 0.1568 & 1.31 & 0.07101 & 1.22 & 0.5006 & 0.71 & 0.3369 &"
    r" 1.44 & 1.6 & 0.03680 & 0.01677 \\",
    r"0.25 & 0.2965 & 1.79 & 0.1104 & 1.70 & 0.5127 & 0.94 & 0.3551 &"
    r" 1.77 & 0.1041 & 1.90 & 0.03880 & 2.10 & 0.5080 & 1.08 & 0.3460 &"
    r" 2.19 & 1.8 & 0.03030 & 0.01182 \\",
    r"0.3 & 0.2373 & 1.90 & 0.07491 & 1.90 & 0.5184 & 0.97 & 0.3620 &"
    r" 1.98 & 0.07112 & 2.53 & 0.02248 & 2.53 & 0.5059 & 1.44 & 0.3445 &"
    r" 2.83 & 2.0 & 0.02389 & 0.00774 \\",
    r"0.4 & 0.1481 & 1.44 & 0.03569 & 1.43 & 0.5259 & 0.96 & 0.3705 &"
    r" 1.92 & 0.03296 & 2.51 & 0.007959 & 2.51 & 0.4933 & 1.49 & 0.3305"
    r" & 2.88 & 2.2 & 0.01788 & 0.00485 \\",
    r"0.5 & && && && && 0.01370 & 5.36 & 0.002667 & 5.37 & 0.5117 & 2.87"
    r" & 0.3440 & 4.37 & 2.4 & 0.01253 & 0.00281 \\",
]


def na49_leading():
    """((n, share) of K+, (n, share) of K-). n: the kaons at x_F >= 0.2 in a
    hemisphere of a pp collision at 17.3 GeV (NA49, Table 5). share: their share
    of all K+ or K- of the hemisphere (half of Table 6's)."""
    rows, means = (
        [derivation.fields(r) for r in NA49_T5],
        derivation.fields(NA49_T6),
    )
    out = []
    for col, mean in ((3, means[1]), (11, means[2])):
        pts = [(float(f[0]), float(f[col])) for f in rows if f[col]]
        n = sum(
            (ya - yb) * (xb - xa) / math.log(ya / yb)
            for (xa, ya), (xb, yb) in zip(pts, pts[1:])
        )
        (xa, ya), (xb, yb) = pts[-2:]
        n += yb * (xb - xa) / math.log(ya / yb)
        out.append((n, n / (float(mean) / 2)))
    return out


def leading(r, hp, hm):
    return hp * (r[3] + r[4]) + hm * (r[5] + r[6])


per_int = 1.0 / g20.SIGMA_TOT[0]
hyperons, a_lam, a_hyp = (
    g20.hyperon_strangeness(),
    g20.k0_share_with_lambda(),
    g20.k0_share_with_hyperons(),
)
species = g20.species()
print(
    "Kaons less antikaons per interaction: the hyperons less the antihyperons"
    " that they balance"
)
print(
    f"  at 20 GeV: {hyperons:.2f} of {g20.SIGMA_TOT[0]:g} mub,"
    f" {hyperons * per_int:.4f} per interaction; a K0 comes with {a_lam:.2f}"
    f" of the Lambda and {a_hyp:.2f} of all hyperons, so their kaons give"
    f" {(1 - 2 * a_lam) * hyperons * per_int:.4f} or"
    f" {(1 - 2 * a_hyp) * hyperons * per_int:.4f} more K+ than K0"
)
print(
    f"    all kaons there: K+ - K0 {(species[0] - species[1]) * per_int:.4f},"
    f" K- - anti-K0 {(species[2] - species[3]) * per_int:.4f} per interaction"
    " (the K anti-K pairs with ABBHHM's shares)"
)
for r in MPI:
    print(
        f"  M{int(r[0]) + 1}: {r[3] + r[4] - r[5] - r[6]:.3f}; K+ - K0"
        f" {r[3] - r[4]:.3f}, K- - anti-K0 {r[5] - r[6]:.3f}"
    )

print(
    "\nEnergy fraction chi of the K0_S at the middle of each type"
    " (in ln eps), with the distribution of 20 GeV in x and pT^2 unchanged"
    " (Feynman scaling), and the code's"
)
needed = {}
for r in MPI:
    t = int(r[0])
    k0, k0b, hp, hm, f = r[4], r[6], r[7], r[8], r[9]
    centre = math.sqrt(r[1] * r[2])
    data = g20.mean_chi(centre)[0]
    chi_l, chi_h = f * KAI[2 * t], KAI[2 * t + 1]
    h_mix = (hp * k0 + hm * k0b) / (k0 + k0b)
    code = (1 - h_mix) * chi_l + h_mix * chi_h
    needed[t] = (
        (data - chi_l) * (k0 + k0b) / ((k0 + H_RATIO * k0b) * (chi_h - chi_l))
    )
    print(
        f"  M{t + 1}, {r[1]:g}-{r[2]:g} GeV ({centre:.3g} GeV):"
        f" mean {data:.4f}; the code {code:.4f} ({code / data - 1:+.0%}):"
        f" {h_mix:.1%} at {chi_h:g}, the others at {f:g} x {KAI[2 * t]:g} ="
        f" {chi_l:.4f}; the mean needs h+ {needed[t]:.4f} and h-"
        f" {H_RATIO * needed[t]:.4f} (the code: {hp:g}, {hm:g})"
    )

# K0 + anti-K0 = 2 K0_S, as many charged
lead20 = 2 * 2 * g20.K0S[0] * g20.mean_chi(x_below=-0.2)[1] * per_int
(n_kp, _), (n_km, _) = na49_leading()
print("\nLeading kaons per interaction (all four species)")
print(
    f"  in the data, towards the nucleon: {lead20:.4f} at 20 GeV"
    f" (the K0_S at x < -0.2: {g20.mean_chi(x_below=-0.2)[1]:.1%} of"
    f" {g20.K0S[0]:g} mub); {2 * (n_kp + n_km):.4f} in pp at 17.3 GeV (NA49:"
    f" K+ {n_kp:.4f}, K- {n_km:.4f} at x_F > 0.2 per hemisphere)"
)
for r in MPI:
    t = int(r[0])
    print(
        f"  M{t + 1}: the code {leading(r, r[7], r[8]):.4f}; with the h+ of"
        f" the mean chi {leading(r, needed[t], H_RATIO * needed[t]):.4f}"
    )
