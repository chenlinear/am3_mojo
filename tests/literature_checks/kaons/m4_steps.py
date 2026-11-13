"""Photo-hadronic kaons, kaons/README.md "Kaons in multi-pion production": the
kaons of type M4 (5-50 GeV) in its two steps (5-15, 15-50 GeV), in number and
energy. They follow from the strange particles measured at 20 GeV by the SLAC
Hybrid Facility Photon Collaboration (Abe et al. 1985, Phys. Rev. D 32, 2869:
cross sections, and Abe et al. 1984, Phys. Rev. D 29, 1877: K0_S
distributions). The data module tests/support/measured/strange_20gev.py holds
these data and the balance below. Its values are checked against the HEPData
records ins213727 and ins190724 if local copies exist. eps_r is the photon
energy in the rest frame of the proton.

  * Numbers at 20 GeV. The SLAC hybrid facility measured the strange particles
    in 284,000 hadronic events at 20 GeV (Abe et al. 1985, Table 2, and
    HEPData ins213727). Their inclusive cross sections, out of the total
    115 mubarn, are 9.663 +- 0.272 mubarn of K0_S and 5.603 +- 0.180 mubarn of
    Lambda. That of Sigma0 is 1.65 +- 0.44 mubarn. All three are corrected for
    the decays not seen. Anti-Lambda and Xi are measured too. The kaons less the
    antikaons balance the strangeness of the hyperons less the antihyperons. The
    charged Sigma are taken as Sigma+ + Sigma- = 2 Sigma0. This holds if the
    three are made alike. It also holds for production linear in I3, and for a
    strange quark that replaces one of the quarks of the proton. The last case
    needs the Sigma0 to be a quarter of Lambda and Sigma0 (as in SU(6)). The
    measurement gives 1.65 of 5.60 mubarn. The split of the kaons into species
    uses the measured K0_S Lambda pairs. They put a K0 in 0.40 +- 0.02 of the
    events with a Lambda (preprint SLAC-PUB-3634, Table 2, against 0.48 in
    ABBHHM). The other K0 and anti-K0 shares are ABBHHM's
    (multibody_strangeness.py). The balance then gives
    K+, K0, K-, anti-K0 = 14.5, 10.9, 8.5, 8.5 mubarn, 1.50 kaons per antikaon.
    These kaons give 0.95 mubarn of K0_S K0_S pairs, a quarter of the K0 anti-K0
    pairs if these are made without coherence. The measured value is
    0.973 +- 0.040 mubarn. Varied within their errors and alternatives (the
    Sigma0, the charged Sigma, the Xi0, extra K anti-K pairs with the Lambda),
    the assumptions change a species by up to 6 %. ABBHHM's shares in the
    K anti-K pairs change a species by up to 20 % when varied one at a time.
    Varied together, they change one by up to 31 %.
  * The line. Each species is then taken linear in ln eps_r, on a line from the
    code's kaons at 2.5-5 GeV to the species at 20 GeV. The line starts from
    M3's upper step with its two-body kaons, 7.7, 5.4, 2.6 and 2.4 mubarn,
    placed at the step's middle in ln eps_r. A step of M4 takes the line's value
    at its own middle in ln eps_r. This value is also the line's mean over the
    step with the weight 1/eps_r, that for a photon field n(eps) ~ 1/eps.
    Divided by the type's 120 mubarn, it gives the step's row
    (tests/unit/test_added_processes.py checks the code's steps against the
    line). The line keeps the measurement at 20 GeV and adds the rise that the
    other data show. The bubble chambers find fewer strange particles below
    10 GeV, K0_S production in pi- p collisions rises, and M5 has more kaons.
    The line has a cost at 20 GeV, because the upper step stands for all of
    15-50 GeV. The two steps have 6.9 and 10.7 mubarn of K0_S. So the code is
    11 % above the measurement at 20 GeV, although the line passes through it.
  * Three checks of the line. First, the line is extended to M5 (50-500 GeV),
    whose kaons are set independently (higher_types.py). There it meets M5's
    kaons, antikaons and K0_S within 6-11 %, and its single species within 18 %.
    Second, the code's Lambda at 2.5-5 GeV, 6.3 mubarn, are 12 % more than the
    5.6 +- 0.2 mubarn at 20 GeV. The fragmentation of the target is expected to
    make about as many at both energies. Third, K0_S production in pi- p
    collisions rises the same way with the available energy E_A of
    kaons/README.md (Abe et al. 1984, Fig. 3). Abe et al. (1985), Table 3, give
    0.043 +- 0.003 K0_S per inelastic interaction at E_A = 2.41 GeV. They give
    0.091 +- 0.004 and 0.075 +- 0.005 at 4.31 and 4.89 GeV. The first over the
    mean of the other two is 0.52 +- 0.04. With the measured total cross
    sections, the line gives the ratio 0.56 between photons of 5.5 GeV and
    16 GeV. These photons have the same available energies.
  * Energy. A kaon's share chi of the nucleon energy, and Feynman's x, are those
    of kaons/README.md. At 20 GeV, Abe et al. 1984 give two K0_S distributions
    (Tables III and VI). One is in x, and the other is in the square pT^2 of the
    transverse momentum. These give the mean <chi> = 0.112. The distribution in
    the kaon's share z = E_K/E_gamma of the photon energy, in the proton rest
    frame (Table V), gives 0.116. A tenth of the kaons go toward the nucleon, at
    x < -0.2, with <chi> = 0.36. The code's leading part stands for them. The
    others have <chi> = 0.085. Under Feynman scaling, the invariant cross
    section stays the same function of x and pT^2 at other energies. With this
    scaling, <chi> is 0.156 at the middle of the lower step and 0.100 at that of
    the upper one. The leading part has the chi of the type's leading pions
    (0.35). To reach these means, the low part needs 1.82 and 0.93 times the chi
    of the type's low-energy pions (0.07). The code takes 1.8 and 0.93, which
    give 0.154 and 0.100. higher_types.py compares the higher types with the
    same scaling.

The code's tables are read from src/ by code_tables.py. The available energy
E_A is read from kaons/README.md."""

import os, sys

import numpy as np

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
from code_tables import code_it3, code_mpi, two_body_steps
from support.measured import strange_20gev as g20
from support.paths import load_derivation

# the page's s, neutral kaons, the code's masses
derivation = load_derivation("kaon_production")
L = load_derivation("latex")
hepdata = L.hepdata
PAGE = "kaons"

SIG3, KAI = code_it3("IT3sig"), code_it3("IT3kai")
MPI = code_mpi()  # [t, eps_min, eps_max, K+, K0, K-, anti-K0, h+, h-, f]
# the K0_S among K0 and anti-K0 (kaons/README.md: half of each)
K0S = derivation.numeric_neutral()
_, M3_UPPER = sorted((r for r in MPI if int(r[0]) == 2), key=lambda r: r[1])
(F_HIGHER,) = {r[9] for r in MPI if int(r[0]) >= 4}
fmt = lambda v, d=2: ", ".join(f"{x:.{d}f}" for x in v)
pm = lambda v, d=3: f"{v[0]:.{d}f} +- {v[1]:.{d}f}"

# the code's kaons for protons [mub]: (eps_min, eps_max, (K+, K0, K-, anti-K0))
KY = two_body_steps()
channels = [(r[0], r[1], np.array([r[6], r[7], 0.0, 0.0])) for r in KY]
channels += [
    (row[1], row[2], SIG3[2 * int(row[0])] * np.array(row[3:7])) for row in MPI
]

print(
    "The values of tests/support/measured/strange_20gev.py against the"
    " HEPData records:"
)
rec84, rec85 = hepdata("ins190724"), hepdata("ins213727")
# (with knowledge/HEPData present, a missing record is a failure)
assert not L.HEPDATA.is_dir() or (rec84 is not None and rec85 is not None), (
    "knowledge/HEPData lacks the records ins190724 and ins213727"
)
if rec84 is None or rec85 is None:
    print(
        "  (not checked: knowledge/HEPData does not have the records"
        " ins190724 and ins213727)"
    )
else:

    def same(name, table, values, edges):
        rows = np.array(rec84[table][0][2], float)
        ok = np.allclose(
            rows[:, 3], values, rtol=1e-9, atol=1e-12
        ) and np.allclose(list(rows[:, 1]) + [rows[-1, 2]], edges)
        print(
            f"  Abe et al. 1984, {name}: {'the same' if ok else 'DIFFERENT'}"
            f" ({len(rows)} values)"
        )
        return ok

    ok = same(
        "K0_S in x (Table III; HEPData Table 1)",
        "Table1",
        g20.F_K0S,
        g20.X_EDGES,
    )
    ok &= same(
        "K0_S in z (Table V; HEPData Table 4)",
        "Table4",
        g20.DNDZ_K0S,
        g20.Z_EDGES,
    )
    ok &= same(
        "K0_S in pT^2 (Table VI; HEPData Table 7)",
        "Table7",
        g20.DSDPT2_K0S,
        g20.PT2_EDGES,
    )
    want = {
        "GAMMA P --> KS X": g20.K0S,
        "GAMMA P --> LAMBDA X": g20.LAMBDA,
        "GAMMA P --> LAMBDABAR X": g20.ANTI_LAMBDA,
        "GAMMA P --> XI- X": g20.XI_MINUS,
        "GAMMA P --> XIBAR+ X": g20.ANTI_XI_PLUS,
        "GAMMA P --> SIGMA0 X": g20.SIGMA0,
        "GAMMA P --> SIGMA(1385P13)+ X": g20.SIGMA1385_PLUS,
        "GAMMA P --> SIGMA(1385P13)- X": g20.SIGMA1385_MINUS,
    }
    got = {r: rows[0] for r, _, rows in rec85["Table1"]}
    ok85 = all(
        abs(float(got[r][3]) - v) < 1e-9 and abs(float(got[r][4]) - e) < 1e-9
        for r, (v, e) in want.items()
    )
    ok85 &= got["GAMMA P --> OMEGA- X"][3] == "<%g" % g20.OMEGA_MINUS_LIMIT
    print(
        f"  Abe et al. 1985, the {len(got)} cross sections (HEPData Table 1):"
        f" {'the same' if ok85 else 'DIFFERENT'}"
    )
    assert ok and ok85

print("\nAt 20 GeV (Abe et al. 1985 and 1984):")
_, _, w84 = g20.chi_samples()
print(
    "  the distributions of 1984 against the inclusive cross section of their"
    f" sample: K0_S {w84.sum():.2f} mub in pT^2"
    f" ({w84.sum() / g20.K0S_1984[0] - 1:+.1%}),"
    f" {g20.mean_chi_from_z()[1] * g20.SIGMA_TOT[0]:.2f} mub in z"
    f" ({
        g20.mean_chi_from_z()[1] * g20.SIGMA_TOT[0] / g20.K0S_1984[0]
        - 1:+.1%})"
)
sp = g20.species()
hyperons = g20.hyperon_strangeness()
print(
    "  the balance of strangeness: hyperons less antihyperons"
    f" {hyperons:.2f} mub (charged Sigma {g20.SIGMA_PM:g} x Sigma0,"
    f" Xi0 = Xi-; Lambda or Sigma0 then {g20.lambda_share():.2f} of the"
    f" hyperons, against {g20.LAMBDA_SHARE_ABBHHM} in ABBHHM's events);"
)
print(
    "  pairs of neutral strange particles in one event [mub]"
    " (Abe et al. 1985, Table 2 of the preprint): K0_S K0_S"
    f" {pm(g20.K0S_K0S)}, K0_S Lambda {pm(g20.K0S_LAMBDA)}, K0_S anti-Lambda"
    f" {pm(g20.K0S_ANTI_LAMBDA)}, Lambda anti-Lambda"
    f" {pm(g20.LAMBDA_ANTI_LAMBDA)}, Lambda Lambda {pm(g20.LAMBDA_LAMBDA)}"
)
a_lam = g20.k0_share_with_lambda()
a_lam_err = a_lam * g20.K0S_LAMBDA[1] / g20.K0S_LAMBDA[0]
print(
    f"    a K0_S comes with {g20.K0S_LAMBDA[0] / g20.LAMBDA[0]:.3f} of the"
    f" Lambda: the K0 among the kaons that come with a Lambda {a_lam:.2f}"
    f" +- {a_lam_err:.2f} (ABBHHM: {g20.K0_SHARE_LAMBDA} in its events with"
    f" further particles), {g20.k0_share_with_lambda(0.1):.2f} with 0.1 extra"
    " K anti-K pairs per Lambda; with the charged Sigma as in ABBHHM's"
    f" events ({g20.K0_SHARE_SIGMA_PLUS}, {g20.K0_SHARE_SIGMA_MINUS}),"
    f" {g20.k0_share_with_hyperons():.2f} among the kaons that come"
    " with a hyperon"
)
print(
    f"    with K0 {g20.K0_SHARE} of the kaons of K anti-K pairs and anti-K0"
    f" {g20.ANTI_K0_SHARE} of the antikaons (ABBHHM): K+, K0, K-, anti-K0 ="
    f" {fmt(sp)}; kaons / antikaons = {(sp[0] + sp[1]) / (sp[2] + sp[3]):.2f}"
    f" (measured: K*+(890) / K*-(890) = {pm(g20.kstar_ratio(), 2)})"
)
anti = sp[2] + sp[3]
print(
    "    check: K0_S K0_S pairs, if the K0 anti-K0 pairs are made without"
    " coherence (a C-even pair gives K0_S K0_S twice as often, a C-odd one"
    f" never): {g20.K0_SHARE} x {g20.ANTI_K0_SHARE} x {anti:.2f} / 4 ="
    f" {g20.K0_SHARE * g20.ANTI_K0_SHARE * anti / 4:.3f} mub, measured"
    f" {pm(g20.K0S_K0S)}"
)
for label, kwargs in (
    (
        "0.1 extra K anti-K pairs per Lambda",
        dict(k0_with_hyperons=g20.k0_share_with_hyperons(0.1)),
    ),
    (
        f"ABBHHM's K0 share for the kaons with a hyperon too ({g20.K0_SHARE})",
        dict(k0_with_hyperons=g20.K0_SHARE),
    ),
    (
        f"charged Sigma {g20.SIGMA_PM_ALT} x Sigma0 (as the Sigma*(1385) with"
        " Sigma+ = Sigma0)",
        dict(sigma_pm=g20.SIGMA_PM_ALT),
    ),
    ("Xi0 = 2 Xi-", dict(xi0_over_xim=2.0)),
    ("K0 = K+ in the K anti-K pairs (share 0.5)", dict(k0_in_pairs=0.5)),
    *(
        (f"anti-K0 share {a}", dict(anti_k0_share=a))
        for a in g20.ANTI_K0_SHARE_RANGE
    ),
    (
        f"K0 share {g20.K0_SHARE_ALL} in the K anti-K pairs",
        dict(k0_in_pairs=g20.K0_SHARE_ALL),
    ),
    (
        "both of ABBHHM's shares with the reactions seen only when a"
        f" K+- decays (K0 {g20.K0_SHARE_ALL} in the pairs, anti-K0"
        f" {g20.ANTI_K0_SHARE_RANGE[0]})",
        dict(
            k0_in_pairs=g20.K0_SHARE_ALL,
            anti_k0_share=g20.ANTI_K0_SHARE_RANGE[0],
        ),
    ),
):
    print(f"      with {label}: {fmt(g20.species(**kwargs))}")
# the Sigma0 cross section enters only through the charged Sigma,
# sigma_pm x Sigma0
for sign in (-1, 1):
    s0 = g20.SIGMA0[0] + sign * g20.SIGMA0[1]
    print(
        f"      with Sigma0 = {s0:.2f}:"
        f" {fmt(g20.species(sigma_pm=g20.SIGMA_PM * s0 / g20.SIGMA0[0]))}"
    )

M3_UP = tuple(M3_UPPER[1:3])
low, e_low = g20.ln_mean(channels, *M3_UP), float(np.sqrt(M3_UP[0] * M3_UP[1]))
print(
    f"\nThe code at {M3_UP[0]:g}-{M3_UP[1]:g} GeV (upper step of M3 with the"
    f" two-body kaons; middle {e_low:.2f} GeV) [mub]: {fmt(low)};"
    f" K0_S {K0S(low[1], low[3]):.2f}"
)
# K+ Lambda and K+ Sigma0
two_body_lambda = g20.ln_mean([(r[0], r[1], r[6]) for r in KY], *M3_UP)
m3_hyperons = (
    low[0]
    + low[1]
    - low[2]
    - low[3]
    - g20.ln_mean([(r[0], r[1], r[6] + r[7]) for r in KY], *M3_UP)
)
fl = g20.LAMBDA_SHARE_ABBHHM
print(
    f"  Lambda (with those from Sigma0) there, from ABBHHM's events: {fl} x"
    f" {m3_hyperons:.2f} (with further particles) + {two_body_lambda:.2f}"
    f" (two-body) = {fl * m3_hyperons + two_body_lambda:.2f} mub; measured at"
    f" 20 GeV: {pm(g20.LAMBDA, 2)}"
)
print(
    "The code's steps of M4 [mub], from the line in ln E through both at each"
    " step's middle; M5 against the line"
    " (per interaction):"
)
for row in MPI:
    t, lo, hi = int(row[0]), row[1], row[2]
    if t not in (3, 4):
        continue
    centre = float(np.sqrt(lo * hi))
    code = np.array(row[3:7])
    if t == 3:
        print(f"  {lo:g}-{hi:g} GeV (middle {centre:.2f} GeV):")
        print(
            f"      [mub] K0_S {SIG3[6] * K0S(code[1], code[3]):.2f}, events"
            " with a hyperon"
            f" {SIG3[6] * (code[0] + code[1] - code[2] - code[3]):.2f},"
            f" with K anti-K {SIG3[6] * (code[2] + code[3]):.2f}"
        )
        continue
    line = g20.trend(centre, e_low, low) / SIG3[2 * t]
    print(
        f"  {lo:g}-{hi:g} GeV (middle {centre:.2f} GeV): {fmt(line, 4)};"
        f" the code {fmt(code, 4)}  (M5, not changed: "
        + ", ".join(f"{c / l - 1:+.0%}" for c, l in zip(code, line))
        + f"; kaons {(code[0] + code[1]) / (line[0] + line[1]) - 1:+.0%},"
        f" antikaons {(code[2] + code[3]) / (line[2] + line[3]) - 1:+.0%},"
        f" K0_S {K0S(code[1], code[3]) / K0S(line[1], line[3]) - 1:+.0%})"
    )
at20 = sum(v for a, b, v in channels if a <= g20.E_GAMMA < b)
print(
    f"  at 20 GeV: K0_S of the code {K0S(at20[1], at20[3]):.2f} mub"
    f" ({K0S(at20[1], at20[3]) / g20.K0S[0] - 1:+.0%}; the line itself goes"
    f" through the measured {g20.K0S[0]:.2f})"
)


def available_energy(m_a, m_b, E_a):
    tex = L.sides(L.inline(r"E_A = \sqrt{s} - m_a - m_b", PAGE))[1]
    return float(
        L.to_sympy(
            tex, {"s": m_a**2 + m_b**2 + 2 * m_b * E_a, "m_a": m_a, "m_b": m_b}
        )
    )


# K0_S per inelastic pi- p interaction against the available energy
# E_A = sqrt(s) - m_a - m_b (Abe et al. 1985, Table 3, and the points of
# Abe et al. 1984, Fig. 3): (beam momentum [GeV/c], K0_S per inelastic
# interaction, error)
PIMP = [(6.0, 0.043, 0.003), (15.0, 0.091, 0.004), (18.5, 0.075, 0.005)]
M_PI = derivation.MASS["pi+"]
ea_pi = [
    available_energy(M_PI, g20.M_P, float(np.hypot(p, M_PI)))
    for p, _, _ in PIMP
]
# total gamma p cross sections [mub] of the same experiments: (photon
# energy [GeV], sigma)
SIGMA_TOT = [t for t in g20.SIGMA_TOT_SLAC if t[0] > 4] + [
    (g20.E_GAMMA, g20.SIGMA_TOT[0])
]


def sigma_tot(e):
    return float(
        np.interp(
            np.log(e),
            np.log([x for x, _ in SIGMA_TOT]),
            [y for _, y in SIGMA_TOT],
        )
    )


def photon_energy(ea):
    return derivation.threshold(g20.M_P, ea + g20.M_P)


k0s_line = lambda e: float(K0S(*g20.trend(e, e_low, low)[[1, 3]]))
e1, e2 = photon_energy(ea_pi[0]), photon_energy(0.5 * (ea_pi[1] + ea_pi[2]))
r_pi = PIMP[0][1] / (0.5 * (PIMP[1][1] + PIMP[2][1]))
r_pi_err = r_pi * np.hypot(
    PIMP[0][2] / PIMP[0][1],
    0.5 * np.hypot(PIMP[1][2], PIMP[2][2]) / (0.5 * (PIMP[1][1] + PIMP[2][1])),
)
r_line = (k0s_line(e1) / sigma_tot(e1)) / (k0s_line(e2) / sigma_tot(e2))
print(
    "  energy dependence, pi- p (Abe et al. 1985, Table 3): K0_S per"
    f" inelastic interaction {PIMP[0][1]} +- {PIMP[0][2]} at E_A ="
    f" {ea_pi[0]:.2f} GeV, {PIMP[1][1]} +- {PIMP[1][2]} and {PIMP[2][1]}"
    f" +- {PIMP[2][2]} at {ea_pi[1]:.2f} and {ea_pi[2]:.2f} GeV:"
    f" ratio {r_pi:.2f} +- {r_pi_err:.2f}"
)
print(
    f"    the line at the same E_A (photons of {e1:.1f} and {e2:.1f} GeV):"
    f" K0_S {k0s_line(e1):.2f} of {sigma_tot(e1):.0f} mub and"
    f" {k0s_line(e2):.2f} of {sigma_tot(e2):.0f} mub: ratio {r_line:.2f}"
)

print(
    "\nEnergy fraction chi of the K0_S (fraction of the nucleon energy in the"
    " frame in which the nucleon is fast)"
)
chi20, _ = g20.mean_chi()
chi_z = g20.mean_chi_from_z()[0]
lead20, others20 = g20.mean_chi(x_below=-0.2), g20.mean_chi(x_above=-0.2)
print(
    f"  at 20 GeV: mean {chi20:.4f} from the distributions in x and pT^2"
    f" (Tables III, VI); {chi_z:.4f} from those in z and pT^2 (Tables V, VI)"
)
print(
    f"    towards the nucleon (x < -0.2): {lead20[1]:.1%} of them, mean chi"
    f" {lead20[0]:.3f}; the others {others20[0]:.4f}"
)
print(
    "  at the middle of the steps, with the same invariant distribution,"
    " and the code:"
)
for row in MPI:
    t, lo, hi, (kp, k0, km, k0b, hp, hm, f) = (
        int(row[0]),
        row[1],
        row[2],
        row[3:],
    )
    if t != 3:
        continue
    centre = float(np.sqrt(lo * hi))
    data, lead = g20.mean_chi(centre)[0], g20.mean_chi(centre, x_below=-0.2)
    chi_l, chi_h = f * KAI[2 * t], KAI[2 * t + 1]
    h_mix = (hp * k0 + hm * k0b) / (k0 + k0b)
    code = (1 - h_mix) * chi_l + h_mix * chi_h
    f_data = (data - h_mix * chi_h) / (1 - h_mix) / KAI[2 * t]
    with_16 = (1 - h_mix) * F_HIGHER * KAI[2 * t] + h_mix * chi_h
    e_kaons = sum(
        m * ((1 - h) * chi_l + h * chi_h)
        for m, h in ((kp, hp), (k0, hp), (km, hm), (k0b, hm))
    )
    print(
        f"    {lo:g}-{hi:g} GeV ({centre:.2f} GeV): mean {data:.4f}"
        f" (x < -0.2: {lead[1]:.1%} at {lead[0]:.3f});"
    )
    print(
        f"        the code {code:.4f} ({code / data - 1:+.1%}): {h_mix:.1%}"
        f" at {chi_h:g} and the others at {f:g} x {KAI[2 * t]:g} ="
        f" {chi_l:.4f} (the mean needs {f_data:.2f} x; with {F_HIGHER:g} x,"
        f" as in the other types: {with_16:.4f}, {with_16 / data - 1:+.0%})"
    )
    print(
        f"        energy of the kaons per interaction: {e_kaons:.4f} of the"
        f" nucleon energy ({kp + k0 + km + k0b:.3f} kaons)"
    )
all20 = float(np.sum(sp)) / SIG3[6]
print(
    f"  from the measurement at 20 GeV: {np.sum(sp):.1f} mub of kaons,"
    f" {all20:.3f} per interaction of the type ({SIG3[6]:g} mub),"
    f" x {chi20:.4f} = {all20 * chi20:.4f}"
)
