"""PDG 2026 data of the charm decays, for the derivation checks of the Decays
section of charm/README.md in charm_decays.py.

Edition: Review of Particle Physics 2026, F. Takahashi et al. (Particle Data
Group), Int. J. Mod. Phys. A 41, 2630011 (2026). Source: the 2026 Particle
Listings, https://pdg.lbl.gov/2026/listings/rpp2026-list-<name>.pdf (all
"Created: 6/1/2026"), in knowledge/papers/ as PDG-2026_listing_<name>.pdf
(git-ignored). Every value was read on the rendered PDF page. 'page' is the
printed page number of the listing of the decaying particle ("Page N" in the
footer, equal to the PDF page index). For branching fractions, it is the page of
the summary decay table. Cross-check:
every PDG input was compared with the PDG Python API (package pdg 2026.0,
edition 2026, data release 2026-05-30 13:41:24 PDT). All agree.

Conventions
 * Branching fractions as fractions (not %). Floats: a value transcribed as
   printed has that decimal as its repr, which charm_decays.py reads exactly
   (sp.Rational(repr(value))). The isospin shares of K*(892) in RESONANCE_DECAYS
   are computed.
 * 'err' is the 1-sigma error. It is a tuple (plus, minus) where PDG quotes
   asymmetric errors. For derived values, the error is propagated in quadrature
   (the larger side of asymmetric inputs, resonance BRs included).
 * Particle names: pi+ pi- pi0 K+ K- K0 K0bar eta etap omega phi rho+ rho- rho0
   Kstar+ Kstar- Kstar0 Kstar0bar f0_980 p pbar n nbar Lambda Sigma+ Sigma0
   Sigma- Xi0 Xi- e+ e- mu+ mu- tau+ tau- nu_e nu_mu nu_tau gamma, antineutrinos
   nu_e_bar nu_mu_bar nu_tau_bar. Lambda1520 (one Lambda_c+ semileptonic mode)
   is outside this list.
 * Only D+, D0, Ds+, Lambda_c+ are given. Their antiparticles decay into the
   CP-conjugate final states.
   The tau is given as PDG lists it (tau-) with the charge-conjugate final state
   for tau+ in 'fs_tau_plus'.
 * DECAYS[species]: the hadronic, semileptonic and leptonic modes, the leptonic
   limits and the modes left out (excluded). A mode has fs (final state), br,
   err, 'pdg' (the PDG entry, Gamma<n> of the species' summary table, and its
   label), 'pdg_value' (the value as printed), page, kind ('total', 'k0s',
   'remainder', 'resonant' or 'derived'), 'calc' (the branching fraction from
   the printed values, G<n>: Gamma_n of the species, B_*: SUBTRACTION_BRS),
   which charm_decays.py recomputes, 'inputs' (the other entries 'calc' uses, as
   printed), and the free texts k0 (the K0S conversion) and note.

Selection and bookkeeping rules (judgment calls are spelled out in each mode's
'note' / 'k0')
 1. HADRONIC: every mode with BR >= 0.5 %. The threshold applies to the PDG
    value after the K0S -> K0bar conversion of rule 2. No mode is at 0.5 %, so
    README.md's "above 0.5 %" selects the same. For a final state of stable
    particles (pi, K, p, n, gamma) plus eta, eta', omega, phi and hyperons,
    PDG's TOTAL is used, not its resonant submodes (K*, rho, a1, Sigma(1385),
    ...). A resonant mode (kind='resonant') is used only where PDG gives no
    total for (some of) its final states.
 2. K0S: B(K0bar X) = 2 B(K0S X) (one neutral kaon, K0bar = (K0S + K0L)/2),
    unless PDG lists the K0bar mode itself. Two K0S: factor 4 (K0 K0bar pair
    taken as incoherent / identical K0bar K0bar: SS:SL:LL = 1:2:1).
    The D+ and D0 tables warn (p2, p15) that interference between
    Cabibbo-allowed and doubly Cabibbo-suppressed amplitudes can invalidate
    2 Gamma(K0S) = Gamma(K0bar). Where PDG also measures the K0L mode, its value
    is quoted in 'k0'. The flavor (K0 or K0bar) follows from the quark content
    of the dominant transition.
 3. No final state is counted twice. A multi-hadron total can contain the decay
    products of an eta/eta'/omega/phi mode that is listed separately. If the
    experiment did not veto that resonance, the contribution
    B(listed mode) x B(resonance -> products) is subtracted (kind='remainder').
    For Ds+, the K*(892)+ K*(892)0bar parts of the K+ K- pi+ pi0 and K0 K- 2pi+
    totals are also subtracted. The subtraction uses the listed mode's BR, so
    that listed mode + remainder reproduces the PDG total of the final state.
    Overlaps below 1e-4 absolute are ignored. A PDG total >= 0.5 % is kept even
    if its remainder is < 0.5 %.
 4. SEMILEPTONIC: every exclusive mode with BR >= 0.1 % (e and mu separately).
    Resonant ones are marked resonant=True. Non-resonant / S-wave complements
    are listed (swave=True) even below 0.1 % when they complete a resonant mode.
    in_sum=True marks the non-overlapping set (resonant totals + complements).
    Overlapping Kpi / pipi totals have in_sum=False. README.md's "above 0.1 %"
    leaves out Lambda_c+ -> Lambda(1520) e+ nu_e, at 0.1 % exactly.
 5. charm_decays.py computes the sums of the hadronic modes.
 6. RESONANCE_DECAYS: every mode >= 0.5 % of the short-lived particles of the D+
    and D0 semileptonic modes. The final states of these modes show which listed
    mode a left-out total contains. SHORT_LIVED: the same for every short-lived
    product. It also holds the largest mode below 0.5 % and the entries of
    0.5 % and above that are no final state.
 7. INCLUSIVE_E: the inclusive e+ rate the semileptonic modes are scaled to.
    SL_FORM_FACTORS: r_V and r_2 of the P -> V l+ nu modes, PDG's averages or,
    where there is none, its single measurement.
"""

# Resonance branching fractions used in the overlap subtractions (PDG 2026).
SUBTRACTION_BRS = {
    "B_ETA_3PIC": (
        0.2302,
        0.0025,
        "eta -> pi+ pi- pi0 (23.02 +- 0.25) %, eta p2",
    ),
    "B_ETA_3PI0": (0.3256, 0.0021, "eta -> 3pi0 (32.56 +- 0.21) %, eta p1"),
    "B_ETAP_PIPIETA": (
        0.425,
        0.005,
        "eta' -> pi+ pi- eta (42.5 +- 0.5) %, eta' p1",
    ),
    "B_ETAP_3PIC": (
        0.00361,
        0.00017,
        "eta' -> pi+ pi- pi0 (3.61 +- 0.17)e-3, eta' p2",
    ),
    "B_OMEGA_3PI": (
        0.892,
        0.007,
        "omega -> pi+ pi- pi0 (89.2 +- 0.7) %, omega p4",
    ),
    "B_OMEGA_2PI": (
        0.0153,
        0.0012,
        "omega -> pi+ pi- (1.53 +- 0.12) %, omega p4",
    ),
    "B_PHI_KK": (0.501, 0.006, "phi -> K+ K- (50.1 +- 0.6) %, phi p4"),
    "B_PHI_KSKL": (0.336, 0.004, "phi -> K0L K0S (33.6 +- 0.4) %, phi p4"),
    "B_PHI_3PI": (
        0.147,
        0.004,
        "phi -> rho pi + pi+ pi- pi0 (14.7 +- 0.4) %, phi p4",
    ),
}


def entry(fs, br, err, pdg, pdg_value, page, **kw):
    return dict(
        fs=fs, br=br, err=err, pdg=pdg, pdg_value=pdg_value, page=page, **kw
    )


def mode(fs, br, err, pdg, pdg_value, page, kind="total", calc=None, **kw):
    return entry(
        fs,
        br,
        err,
        pdg,
        pdg_value,
        page,
        kind=kind,
        calc=calc or pdg.split()[0].replace("Gamma", "G"),
        **kw,
    )


def sl(fs, *args, resonant=False, in_sum=True, swave=False, **kw):
    return mode(
        fs,
        *args,
        lepton="e" if "e+" in fs else "mu",
        resonant=resonant,
        in_sum=in_sum,
        swave=swave,
        **kw,
    )


DECAYS = {}

DECAYS["D+"] = dict(
    hadronic=[
        mode(
            ["K0bar", "pi+"],
            3.1080e-02,
            6.6e-04,
            "Gamma66 K0S pi+",
            "(1.554 +- 0.033) % S=1.8",
            4,
            "k0s",
            "2*G66",
            k0="K0L pi+ (G67) = (1.46 +- 0.05) %: K0S + K0L = 3.01 %.",
        ),
        mode(
            ["K-", "pi+", "pi+"],
            9.3100e-02,
            1.4e-03,
            "Gamma68 K- 2pi+",
            "(9.31 +- 0.14) % S=1.4 [a]",
            4,
            note="incl. (K-pi+)S-wave, K*0bar pi+, ... (G69-G77).",
        ),
        mode(
            ["K0bar", "pi+", "pi0"],
            1.4620e-01,
            4.0e-03,
            "Gamma78 K0S pi+ pi0",
            "(7.31 +- 0.20) % [a]",
            4,
            "k0s",
            "2*G78",
            note="incl. K0S rho+ (G79) etc.",
        ),
        mode(
            ["K0bar", "pi+", "omega"],
            1.4200e-02,
            1.0e-03,
            "Gamma88 K0S pi+ omega",
            "(7.1 +- 0.5) x 10^-3",
            5,
            "k0s",
            "2*G88",
        ),
        mode(
            ["K0bar", "pi+", "eta"],
            2.5400e-02,
            1.0e-03,
            "Gamma89 K0S pi+ eta",
            "(1.27 +- 0.05) %",
            5,
            "k0s",
            "2*G89",
            note="ABLIKIM 20V (arXiv:2004.13910): eta via gamma gamma; "
            "eta' vetoed in pi pi eta combinations.",
        ),
        mode(
            ["K-", "pi+", "pi+", "pi0"],
            6.3500e-02,
            8.0e-04,
            "Gamma93 K- 2pi+ pi0",
            "(6.35 +- 0.08) % [c]",
            5,
            note="incl. K*0bar rho+ etc. (G94-G106).",
        ),
        mode(
            ["K0bar", "pi+", "pi+", "pi-"],
            6.1383e-02,
            1.6e-03,
            "Gamma107 K0S 2pi+ pi-",
            "(3.08 +- 0.08) % [c]",
            5,
            "remainder",
            "2*(G107 - G88*B_OMEGA_2PI)",
            inputs=["Gamma88 K0S pi+ omega = (7.1 +- 0.5) x 10^-3 (p5)"],
        ),
        mode(
            ["K0bar", "pi+", "pi0", "pi0"],
            5.7800e-02,
            1.8e-03,
            "Gamma108 K0S pi+ 2pi0",
            "(2.89 +- 0.09) %",
            5,
            "k0s",
            "2*G108",
            note="ABLIKIM 22Y: K0S->pi0pi0 peaking background estimated "
            "and removed.",
        ),
        mode(
            ["K-", "pi+", "pi+", "pi+", "pi-"],
            5.7000e-03,
            5.0e-04,
            "Gamma117 K- 3pi+ pi-",
            "(5.7 +- 0.5) x 10^-3 S=1.1 [a]",
            6,
        ),
        mode(
            ["K0bar", "pi+", "pi+", "pi-", "pi0"],
            1.2087e-02,
            1.9e-03,
            "Gamma124 K0S 2pi+ pi- pi0",
            "(1.53 +- 0.08) %",
            6,
            "remainder",
            "2*(G124 - G88*B_OMEGA_3PI - G89*B_ETA_3PIC)",
            note="ABLIKIM 22Y (arXiv:2205.14031, Table 3) total includes K pi "
            "omega (omega->3pi) 6.29e-3 and K pi eta (eta->3pi) 3.00e-3 "
            "(non-eta,omega 5.99e-3).",
            inputs=[
                "Gamma88 K0S pi+ omega = (7.1 +- 0.5) x 10^-3 (p5)",
                "Gamma89 K0S pi+ eta = (1.27 +- 0.05) % (p5)",
            ],
        ),
        mode(
            ["K0bar", "pi+", "pi0", "pi0", "pi0"],
            2.7298e-03,
            1.1e-03,
            "Gamma125 K0S pi+ 3pi0",
            "(5.5 +- 0.5) x 10^-3",
            6,
            "remainder",
            "2*(G125 - G89*B_ETA_3PI0)",
            note="ABLIKIM 22Y total includes K pi eta (eta->3pi0) 4.28e-3 "
            "(non-eta 1.26e-3). PDG total x2 = 1.10 %, so the remainder "
            "below 0.5 % is kept.",
            inputs=["Gamma89 K0S pi+ eta = (1.27 +- 0.05) % (p5)"],
        ),
        mode(
            ["pi+", "pi+", "pi-", "pi0"],
            1.0812e-02,
            3.0e-04,
            "Gamma147 2pi+ pi- pi0",
            "(1.165 +- 0.030) %",
            6,
            "remainder",
            "G147 - G185*B_PHI_3PI",
            note="ABLIKIM 22BG (arXiv:2206.13864): only K0S vetoed, "
            "eta/omega/phi intermediate states included. eta pi+ (0.375 %) and "
            "omega pi+ (0.028 %) are below threshold, not listed, so their 3pi "
            "parts stay here.",
            inputs=["Gamma185 phi pi+ = (5.70 +- 0.14) x 10^-3 (p7)"],
        ),
        mode(
            ["pi+", "pi+", "pi-", "pi0", "pi0"],
            1.0700e-02,
            5.0e-04,
            "Gamma150 2pi+ pi- 2pi0",
            "(1.07 +- 0.05) %",
            6,
            note="ABLIKIM 22BG: includes omega pi+ pi0 (0.39 %), eta pi+ pi0 "
            "(0.205 %) and phi pi+ pi0; none of these is listed, "
            "so no subtraction.",
        ),
        mode(
            ["K0bar", "K+"],
            6.0400e-03,
            1.8e-04,
            "Gamma170 K0S K+",
            "(3.02 +- 0.09) x 10^-3 S=2.2",
            7,
            "k0s",
            "2*G170",
            k0="K0L K+ (G171) = (3.21 +- 0.16) x 10^-3: K0S + K0L = 0.623 %.",
        ),
        mode(
            ["K0bar", "K+", "pi0"],
            1.0140e-02,
            6.0e-04,
            "Gamma172 K0S K+ pi0",
            "(5.07 +- 0.30) x 10^-3",
            7,
            "k0s",
            "2*G172",
            k0="K0L K+ pi0 (G176) = (5.24 +- 0.31) x 10^-3: "
            "K0S + K0L = 1.031 %.",
        ),
        mode(
            ["K+", "K-", "pi+"],
            6.7543e-03,
            1.9e-04,
            "Gamma177 K+ K- pi+",
            "(9.61 +- 0.17) x 10^-3 [a]",
            7,
            "remainder",
            "G177 - G185*B_PHI_KK",
            note="minus phi pi+ (phi->K+K-) as generated from the listed phi "
            "pi+ (0.286 %); the PDG Dalitz fit fraction G184 = "
            "(2.67 +0.06 -0.08) x 10^-3 would give 0.694 %.",
            inputs=["Gamma185 phi pi+ = (5.70 +- 0.14) x 10^-3 (p7)"],
        ),
        mode(
            ["phi", "pi+"],
            5.7000e-03,
            1.4e-04,
            "Gamma185 phi pi+",
            "(5.70 +- 0.14) x 10^-3",
            7,
            note="all phi decays (Unseen decay modes included, listing p45).",
        ),
        mode(
            ["K+", "K-", "pi+", "pi0"],
            6.6200e-03,
            3.2e-04,
            "Gamma186 K+ K- pi+ pi0",
            "(6.62 +- 0.32) x 10^-3",
            7,
            note="includes phi pi+ pi0 / phi rho+ (phi->K+K-). G199-G201 "
            "(excluded) contradict this total: phi pi+ pi0 <= "
            "0.662/0.501 = 1.32 %.",
        ),
        mode(
            ["K0", "K0bar", "pi+"],
            1.1880e-02,
            4.0e-04,
            "Gamma187 K0S K0S pi+",
            "(2.97 +- 0.10) x 10^-3",
            8,
            "k0s",
            "4*G187",
            k0="the K0 K0bar pair taken as incoherent, as for K*+ K0bar; "
            "PDG lists no K0S K0L pi+ for D+.",
            note="K0S K*(892)+ (K*+ -> K0S pi+) is 2.90e-3 of 2.97e-3. phi pi+ "
            "(phi -> K0S K0L) is not contained (SS only).",
        ),
        mode(
            ["K0", "K0bar", "pi+", "pi0"],
            5.3600e-03,
            8.4e-04,
            "Gamma190 K0S K0S pi+ pi0",
            "(1.34 +- 0.21) x 10^-3",
            8,
            "k0s",
            "4*G190",
            k0="Borderline: 0.536 %.",
        ),
        mode(
            ["K+", "K0bar", "K0bar"],
            1.0160e-02,
            5.2e-04,
            "Gamma128 K+ 2K0S",
            "(2.54 +- 0.13) x 10^-3",
            6,
            "k0s",
            "4*G128",
            k0="two identical "
            "K0bar; "
            "Cabibbo-favoured "
            "final state K+ "
            "K0bar K0bar.",
        ),
    ],
    semileptonic=[
        sl(
            ["K0bar", "e+", "nu_e"],
            8.8100e-02,
            7.0e-04,
            "Gamma17 K0bar e+ nu_e",
            "(8.81 +- 0.07) %",
            2,
        ),
        sl(
            ["K0bar", "pi0", "e+", "nu_e"],
            1.8860e-02,
            3.2e-04,
            "Gamma19 K0S pi0 e+ nu_e",
            "(9.43 +- 0.16) x 10^-3",
            2,
            "k0s",
            "2*G19",
            in_sum=False,
            note="overlaps K*0bar e nu (G21: K*->K0S pi0 = 8.82e-3) and "
            "(K0S pi0)S-wave (G20).",
        ),
        sl(
            ["K-", "pi+", "e+", "nu_e"],
            4.0000e-02,
            1.7e-03,
            "Gamma22 K- pi+ e+ nu_e",
            "(4.00 +- 0.17) % S=3.1",
            2,
            in_sum=False,
            note="overlaps K*0bar e nu (G23 = 3.75 %), (K-pi+)S-wave (G25); "
            "G24 (K-pi+)[0.8-1.0 GeV] = 3.39 % is a mass-window subset; "
            "G28 nonresonant < 7e-3.",
        ),
        sl(
            ["Kstar0bar", "e+", "nu_e"],
            5.4000e-02,
            1.0e-03,
            "Gamma29 K*(892)0bar e+ nu_e",
            "(5.40 +- 0.10) % S=1.1",
            3,
            resonant=True,
            note="all K*0bar decays.",
        ),
        sl(
            ["K-", "pi+", "e+", "nu_e"],
            2.2800e-03,
            1.1e-04,
            "Gamma25 (K- pi+)S-wave e+ nu_e",
            "(2.28 +- 0.11) x 10^-3",
            2,
            swave=True,
            note="complement of K*0bar e nu in K- pi+ e nu.",
        ),
        sl(
            ["K0bar", "pi0", "e+", "nu_e"],
            1.2060e-03,
            4.6e-05,
            "Gamma20 (K0S pi0)S-wave e+ nu_e",
            "(6.03 +- 0.23) x 10^-4",
            2,
            "k0s",
            "2*G20",
            swave=True,
            note="complement of K*0bar e nu in K0bar pi0 e nu.",
        ),
        sl(
            ["pi0", "e+", "nu_e"],
            3.7200e-03,
            1.7e-04,
            "Gamma44 pi0 e+ nu_e",
            "(3.72 +- 0.17) x 10^-3 S=2.0",
            3,
        ),
        sl(
            ["eta", "e+", "nu_e"],
            1.1100e-03,
            7.0e-05,
            "Gamma46 eta e+ nu_e",
            "(1.11 +- 0.07) x 10^-3",
            3,
        ),
        sl(
            ["pi+", "pi-", "e+", "nu_e"],
            2.4500e-03,
            8.0e-05,
            "Gamma48 pi- pi+ e+ nu_e",
            "(2.45 +- 0.08) x 10^-3",
            3,
            in_sum=False,
            note="= rho0 e nu (G50) + f0(500) e nu (G49).",
        ),
        sl(
            ["rho0", "e+", "nu_e"],
            1.8700e-03,
            6.0e-05,
            "Gamma50 rho0 e+ nu_e",
            "(1.87 +- 0.06) x 10^-3",
            3,
            resonant=True,
        ),
        sl(
            ["pi+", "pi-", "e+", "nu_e"],
            6.2000e-04,
            4.0e-05,
            "Gamma49 f0(500)0 e+ nu_e, f0(500)0 -> pi+ pi-",
            "(6.2 +- 0.4) x 10^-4",
            3,
            swave=True,
            note="S-wave complement of rho0 e nu, below 0.1 %; f0(500) is not "
            "in the particle list.",
        ),
        sl(
            ["omega", "e+", "nu_e"],
            1.6900e-03,
            1.1e-04,
            "Gamma54 omega e+ nu_e",
            "(1.69 +- 0.11) x 10^-3",
            3,
            resonant=True,
        ),
        sl(
            ["K-", "pi+", "pi0", "e+", "nu_e"],
            1.2300e-03,
            8.0e-05,
            "Gamma37 K1(1270)0bar e+ nu_e, K1(1270)0bar -> K- pi+ pi0",
            "(1.23 +- 0.08) x 10^-3 S=1.3",
            3,
            resonant=True,
            note="K1 -> K- pi+ pi0 only (K1 is not in the particle list); G38 "
            "K1 -> K0S pi+ pi- (1.5 +0.5 -0.4) x 10^-4.",
        ),
        sl(
            ["K0bar", "mu+", "nu_mu"],
            8.6800e-02,
            1.0e-03,
            "Gamma18 K0bar mu+ nu_mu",
            "(8.68 +- 0.10) %",
            2,
        ),
        sl(
            ["K0bar", "pi0", "mu+", "nu_mu"],
            1.7920e-02,
            3.8e-04,
            "Gamma30 K0S pi0 mu+ nu_mu",
            "(8.96 +- 0.19) x 10^-3",
            3,
            "k0s",
            "2*G30",
            in_sum=False,
            note="overlaps K*0bar mu nu (G32 8.32e-3 with K*->K0S pi0) and "
            "(K0S pi0)S-wave (G31).",
        ),
        sl(
            ["K-", "pi+", "mu+", "nu_mu"],
            3.6200e-02,
            3.3e-03,
            "Gamma33 K- pi+ mu+ nu_mu",
            "(3.62 +- 0.33) %",
            3,
            in_sum=False,
            note="overlaps K*0bar mu nu (G34 = 3.52 %) and K-pi+ mu nu "
            "nonresonant (G35).",
        ),
        sl(
            ["Kstar0bar", "mu+", "nu_mu"],
            5.2600e-02,
            1.5e-03,
            "Gamma36 K*(892)0bar mu+ nu_mu",
            "(5.26 +- 0.15) % S=1.1",
            3,
            resonant=True,
            note="all K*0bar decays.",
        ),
        sl(
            ["K-", "pi+", "mu+", "nu_mu"],
            1.9000e-03,
            5.0e-04,
            "Gamma35 K- pi+ mu+ nu_mu nonresonant",
            "(1.9 +- 0.5) x 10^-3",
            3,
            swave=True,
            note="complement of K*0bar mu nu in K- pi+ mu nu.",
        ),
        sl(
            ["K0bar", "pi0", "mu+", "nu_mu"],
            1.2800e-03,
            1.4e-04,
            "Gamma31 (K0S pi0)S-wave mu+ nu_mu",
            "(6.4 +- 0.7) x 10^-4",
            3,
            "k0s",
            "2*G31",
            swave=True,
            note="complement of K*0bar mu nu in K0bar pi0 mu nu.",
        ),
        sl(
            ["pi0", "mu+", "nu_mu"],
            3.5000e-03,
            1.5e-04,
            "Gamma45 pi0 mu+ nu_mu",
            "(3.50 +- 0.15) x 10^-3",
            3,
        ),
        sl(
            ["eta", "mu+", "nu_mu"],
            1.0400e-03,
            1.1e-04,
            "Gamma47 eta mu+ nu_mu",
            "(1.04 +- 0.11) x 10^-3",
            3,
        ),
        sl(
            ["rho0", "mu+", "nu_mu"],
            1.6400e-03,
            1.6e-04,
            "Gamma53 rho0 mu+ nu_mu",
            "(1.64 +- 0.16) x 10^-3",
            3,
            resonant=True,
            note="pi- pi+ mu+ nu_mu total (G51) is listed without a value.",
        ),
        sl(
            ["pi+", "pi-", "mu+", "nu_mu"],
            7.2000e-04,
            1.5e-04,
            "Gamma52 f0(500)0 mu+ nu_mu, f0 -> pi+ pi-",
            "(7.2 +- 1.5) x 10^-4",
            3,
            swave=True,
            note="complement of rho0 mu nu, below 0.1 %.",
        ),
        sl(
            ["omega", "mu+", "nu_mu"],
            1.7700e-03,
            2.1e-04,
            "Gamma55 omega mu+ nu_mu",
            "(1.77 +- 0.21) x 10^-3",
            3,
            resonant=True,
        ),
        sl(
            ["K-", "pi+", "pi0", "mu+", "nu_mu"],
            1.0900e-03,
            (1.2e-04, 1.6e-04),
            "Gamma40 K1(1270)0bar mu+ nu_mu, K1(1270)0bar -> K- pi+ pi0",
            "(1.09 +0.12 -0.16) x 10^-3",
            3,
            resonant=True,
            note="K1 -> K- pi+ pi0 only. G39 K- pi+ pi0 mu+ nu_mu total < "
            "1.5e-3 CL90.",
        ),
    ],
    leptonic=[
        mode(
            ["mu+", "nu_mu"],
            4.0200e-04,
            9.0e-06,
            "Gamma15 mu+ nu_mu",
            "(4.02 +- 0.09) x 10^-4",
            2,
        ),
        mode(
            ["tau+", "nu_tau"],
            9.9000e-04,
            1.2e-04,
            "Gamma16 tau+ nu_tau",
            "(9.9 +- 1.2) x 10^-4",
            2,
        ),
    ],
    leptonic_limits=[
        dict(
            mode="e+ nu_e",
            limit=9.7e-07,
            pdg="Gamma13 < 9.7 x 10^-7 CL=90%",
            page=2,
        ),
    ],
    excluded=[
        "Gamma126 K- 2pi+ 2pi0 (4.95 +- 0.32) x 10^-3, p6: below 0.5 % "
        "(borderline)",
        "Gamma168 eta'(958) pi+ (4.96 +- 0.18) x 10^-3, p7: below 0.5 % "
        "(borderline)",
        "Gamma146 pi+ 2pi0 (4.61 +- 0.15) x 10^-3, p6; Gamma148 pi+ 3pi0 "
        "(4.17 +- 0.26) x 10^-3, p6; Gamma164 eta 2pi+ pi- pi0 "
        "(3.88 +- 0.34) x 10^-3, p7",
        "Gamma154 eta pi+ (3.75 +- 0.09) x 10^-3, p7 (below 0.5 %; its "
        "eta -> 3pi parts stay inside the 2pi+ pi- pi0 total)",
        "Gamma196 K0S K- 2pi+ (2.27 +- 0.13) x 10^-3, p8: x2 = 0.454 %; "
        "Gamma92 K0S pi+ eta' (1.90 +- 0.21) x 10^-3, p5: x2 = 0.38 %; "
        "Gamma194 K+ K0S pi+ pi- (1.89 +- 0.13) x 10^-3, p8: x2 = 0.378 %",
        "Gamma199 phi pi+ pi0 (2.3 +- 1.0) %, Gamma200 phi rho+ < 1.5 % CL90, "
        'Gamma201 K+ K- pi+ pi0 non-phi (1.5 +0.7 -0.6) %, p8 ("A few poorly '
        'measured branching fractions"): not used, inconsistent with the '
        "K+ K- pi+ pi0 total Gamma186 = (6.62 +- 0.32) x 10^-3",
        "resonant submodes (K0S rho+, K*0bar pi+, K*0bar rho+, ...) and K0L "
        "modes (Gamma67, 171, 176; quoted in k0) not used",
    ],
)

DECAYS["D0"] = dict(
    hadronic=[
        mode(
            ["K-", "pi+"],
            3.9360e-02,
            3.0e-04,
            "Gamma47 K- pi+",
            "(3.936 +- 0.030) % S=1.3",
            17,
        ),
        mode(
            ["K0bar", "pi0"],
            2.4780e-02,
            4.4e-04,
            "Gamma48 K0S pi0",
            "(1.239 +- 0.022) %",
            17,
            "k0s",
            "2*G48",
            k0="K0L pi0 (G49) = (9.76 +- 0.32) x 10^-3: K0S + K0L = 2.215 %.",
        ),
        mode(
            ["K0bar", "pi+", "pi-"],
            5.6460e-02,
            3.2e-03,
            "Gamma53 K0S pi+ pi-",
            "(2.84 +- 0.16) % S=1.1 [e]",
            17,
            "remainder",
            "2*(G53 - G129*B_OMEGA_2PI)",
            note="Dalitz total, incl. K0S rho0, K*- pi+, ... and K0S "
            "omega(->pi+pi-) G55.",
            inputs=["Gamma129 K0S omega = (1.11 +- 0.06) % (p19)"],
        ),
        mode(
            ["K-", "pi+", "pi0"],
            1.4400e-01,
            5.0e-03,
            "Gamma68 K- pi+ pi0",
            "(14.4 +- 0.5) % S=2.0 [e]",
            17,
            note="incl. K- rho+ (11.2 %), K*- pi+, K*0bar pi0 ...",
        ),
        mode(
            ["K0bar", "pi0", "pi0"],
            1.8200e-02,
            2.2e-03,
            "Gamma77 K0S 2pi0",
            "(9.1 +- 1.1) x 10^-3 S=2.2",
            18,
            "k0s",
            "2*G77",
            k0="K0L pi0 pi0 (G78) = (1.26 +- 0.06) %: K0S + K0L = 2.17 %.",
        ),
        mode(
            ["K0bar", "pi0", "pi0", "pi0"],
            1.1892e-02,
            8.0e-04,
            "Gamma86 K0S 3pi0",
            "(7.6 +- 0.4) x 10^-3",
            18,
            "remainder",
            "2*(G86 - G128*B_ETA_3PI0)",
            note="ABLIKIM 22Y total (7.64e-3) includes K0S eta "
            "(eta->3pi0) 1.66e-3.",
            inputs=["Gamma128 K0S eta = (5.08 +- 0.13) x 10^-3 (p19)"],
        ),
        mode(
            ["K-", "pi+", "pi+", "pi-"],
            8.1481e-02,
            1.4e-03,
            "Gamma87 K- 2pi+ pi-",
            "(8.20 +- 0.14) % S=1.2 [e]",
            18,
            "remainder",
            "G87 - G107*B_OMEGA_2PI",
            note="K0S K- pi+ (K0S->pi+pi-) is removed as peaking background in "
            "the dominant CLEO measurement (BONVICINI 14, arXiv:1312.6775).",
            inputs=["Gamma107 K- pi+ omega = (3.39 +- 0.10) % (p19)"],
        ),
        mode(
            ["K0bar", "pi+", "pi-", "pi0"],
            8.0390e-02,
            1.2e-02,
            "Gamma98 K0S pi+ pi- pi0",
            "(5.2 +- 0.6) % [g]",
            18,
            "remainder",
            "2*(G98 - G128*B_ETA_3PIC - G129*B_OMEGA_3PI - "
            "G141/B_PHI_KK*B_PHI_3PI - G109*B_OMEGA_2PI)",
            note="total (COFFMAN 92B + ratios; listing p54: eta, omega "
            'submodes "fairly well determined") includes K0S eta '
            "(eta->3pi, G99) and K0S omega (omega->3pi, G100).",
            inputs=[
                "Gamma109 K0S pi0 omega = (8.5 +- 0.6) x 10^-3 (p19)",
                "Gamma128 K0S eta = (5.08 +- 0.13) x 10^-3 (p19)",
                "Gamma129 K0S omega = (1.11 +- 0.06) % (p19)",
                "Gamma141 K0S phi, phi -> K+ K- = (2.06 +- 0.14) x 10^-3 (p20)",
            ],
        ),
        mode(
            ["K-", "pi+", "pi0", "pi0"],
            8.8600e-02,
            2.3e-03,
            "Gamma101 K- pi+ 2pi0",
            "(8.86 +- 0.23) %",
            18,
            note="ABLIKIM 19AK (arXiv:1903.06316): K0S->pi0pi0 vetoed "
            "(removes K- K0S pi+).",
        ),
        mode(
            ["K-", "pi+", "pi0", "pi0", "pi0"],
            3.3787e-03,
            4.3e-04,
            "Gamma102 K- pi+ 3pi0",
            "(9.5 +- 0.4) x 10^-3",
            18,
            "remainder",
            "G102 - G113*B_ETA_3PI0",
            note="ABLIKIM 22Y total (9.54e-3) includes K- pi+ eta (eta->3pi0) "
            "6.06e-3. PDG total >= 0.5 %, remainder below but kept.",
            inputs=["Gamma113 K- pi+ eta = (1.88 +- 0.05) % S=1.4 (p19)"],
        ),
        mode(
            ["K0bar", "pi+", "pi-", "pi0", "pi0"],
            5.5860e-03,
            1.6e-03,
            "Gamma103 K0S pi+ pi- 2pi0",
            "(1.27 +- 0.06) %",
            19,
            "remainder",
            "2*(G103 - G109*B_OMEGA_3PI - G110*B_ETA_3PIC)",
            note="ABLIKIM 22Y total (12.66e-3) includes K pi omega "
            "(omega->3pi) 7.14e-3 and K pi eta (eta->3pi) 2.31e-3.",
            inputs=[
                "Gamma109 K0S pi0 omega = (8.5 +- 0.6) x 10^-3 (p19)",
                "Gamma110 K0S eta pi0 = (1.01 +- 0.05) % (p19)",
            ],
        ),
        mode(
            ["K-", "pi+", "pi+", "pi-", "pi0"],
            8.4334e-03,
            4.1e-03,
            "Gamma104 K- 2pi+ pi- pi0",
            "(4.3 +- 0.4) %",
            19,
            "remainder",
            "G104 - G107*B_OMEGA_3PI - G113*B_ETA_3PIC",
            note="total of old ratio measurements, no vetoes (submode G106 "
            "K*0bar omega with omega->3pi).",
            inputs=[
                "Gamma107 K- pi+ omega = (3.39 +- 0.10) % (p19)",
                "Gamma113 K- pi+ eta = (1.88 +- 0.05) % S=1.4 (p19)",
            ],
        ),
        mode(
            ["K-", "pi+", "omega"],
            3.3900e-02,
            1.0e-03,
            "Gamma107 K- pi+ omega",
            "(3.39 +- 0.10) %",
            19,
            note="ABLIKIM 22U 3.392 +- 0.044 +- 0.085 %; all omega decays.",
        ),
        mode(
            ["K0bar", "pi0", "omega"],
            1.7000e-02,
            1.2e-03,
            "Gamma109 K0S pi0 omega",
            "(8.5 +- 0.6) x 10^-3",
            19,
            "k0s",
            "2*G109",
        ),
        mode(
            ["K0bar", "eta", "pi0"],
            2.0200e-02,
            1.0e-03,
            "Gamma110 K0S eta pi0",
            "(1.01 +- 0.05) %",
            19,
            "k0s",
            "2*G110",
        ),
        mode(
            ["K-", "pi+", "eta"],
            1.8800e-02,
            5.0e-04,
            "Gamma113 K- pi+ eta",
            "(1.88 +- 0.05) % S=1.4",
            19,
            note="incl. K*0bar eta, a0+ K- submodes.",
        ),
        mode(
            ["K0bar", "pi+", "pi-", "eta"],
            5.6000e-03,
            4.2e-04,
            "Gamma118 K0S pi+ pi- eta",
            "(2.80 +- 0.21) x 10^-3",
            19,
            "k0s",
            "2*G118",
            note="ABLIKIM 20V: eta' vetoed in pi+pi-eta, so no overlap with "
            "K0bar eta'.",
        ),
        mode(
            ["K0bar", "pi+", "pi+", "pi-", "pi-"],
            5.4000e-03,
            5.8e-04,
            "Gamma120 K0S 2pi+ 2pi-",
            "(2.70 +- 0.29) x 10^-3",
            19,
            "k0s",
            "2*G120",
        ),
        mode(
            ["K0bar", "eta"],
            1.0160e-02,
            2.6e-04,
            "Gamma128 K0S eta",
            "(5.08 +- 0.13) x 10^-3",
            19,
            "k0s",
            "2*G128",
            k0="K0L eta (G50) = (4.34 +- 0.16) x 10^-3: K0S + K0L = 0.942 %.",
            note='"corrected for unseen decays of the resonances" (p19).',
        ),
        mode(
            ["K0bar", "omega"],
            2.2200e-02,
            1.2e-03,
            "Gamma129 K0S omega",
            "(1.11 +- 0.06) %",
            19,
            "k0s",
            "2*G129",
            k0="K0L omega (G52) = (1.16 +- 0.04) %: K0S + K0L = 2.27 %.",
        ),
        mode(
            ["K0bar", "etap"],
            1.8980e-02,
            6.4e-04,
            "Gamma130 K0S eta'(958)",
            "(9.49 +- 0.32) x 10^-3",
            19,
            "k0s",
            "2*G130",
            k0="K0L eta' (G51) = (8.12 +- 0.35) x 10^-3: K0S + K0L = 1.761 %.",
        ),
        mode(
            ["K-", "pi+", "etap"],
            6.4300e-03,
            3.4e-04,
            "Gamma133 K- pi+ eta'(958)",
            "(6.43 +- 0.34) x 10^-3",
            19,
        ),
        mode(
            ["K0bar", "etap", "pi0"],
            5.0400e-03,
            5.4e-04,
            "Gamma134 K0S eta'(958) pi0",
            "(2.52 +- 0.27) x 10^-3",
            19,
            "k0s",
            "2*G134",
            k0="Borderline: 0.504 %.",
        ),
        mode(
            ["K0bar", "K+", "K-"],
            4.8600e-03,
            6.4e-04,
            "Gamma136 K0S K+ K-",
            "(4.49 +- 0.29) x 10^-3",
            20,
            "remainder",
            "2*(G136 - G141)",
            note="G141 (phi->K+K-) is generated by the K0bar phi mode below. "
            "PDG total x2 = 0.898 %.",
            inputs=[
                "Gamma141 K0S phi, phi -> K+ K- = (2.06 +- 0.14) x 10^-3 (p20)"
            ],
        ),
        mode(
            ["K0bar", "phi"],
            8.2236e-03,
            5.7e-04,
            "Gamma141 K0S phi, phi -> K+ K-",
            "(2.06 +- 0.14) x 10^-3",
            20,
            "derived",
            "2*G141/B_PHI_KK",
            k0="cross-check: K0L phi (G142) = (4.06 +- 0.23) x 10^-3 "
            "(x2 = 0.812 %).",
            note="PDG gives no K0S phi total in the summary table: G141 / "
            "B(phi->K+K-).",
        ),
        mode(
            ["pi+", "pi-", "pi0"],
            1.4900e-02,
            6.0e-04,
            "Gamma153 pi+ pi- pi0",
            "(1.49 +- 0.06) % S=2.1",
            20,
            note="K0S vetoed in the measurements.",
        ),
        mode(
            ["pi+", "pi+", "pi-", "pi-"],
            7.1500e-03,
            2.5e-04,
            "Gamma171 2pi+ 2pi-",
            "(7.15 +- 0.25) x 10^-3 S=2.2",
            20,
        ),
        mode(
            ["pi+", "pi-", "pi0", "pi0"],
            1.0020e-02,
            3.1e-04,
            "Gamma212 pi+ pi- 2pi0",
            "(1.002 +- 0.031) %",
            22,
            note="incl. eta pi0 (not listed) etc.; non-eta part G213 = "
            "(9.51 +- 0.33) x 10^-3.",
        ),
        mode(
            ["K0", "K-", "pi+"],
            6.8000e-03,
            1.0e-03,
            "Gamma268 K0S K- pi+",
            "(3.4 +- 0.5) x 10^-3 S=1.1",
            24,
            "k0s",
            "2*G268",
            k0="neutral kaon is K0 (K*+ K-, K*0bar K0 submodes).",
        ),
    ],
    semileptonic=[
        sl(
            ["K-", "e+", "nu_e"],
            3.5370e-02,
            1.7e-04,
            "Gamma20 K- e+ nu_e",
            "(3.537 +- 0.017) % S=1.1",
            16,
        ),
        sl(
            ["Kstar-", "e+", "nu_e"],
            2.1600e-02,
            1.6e-03,
            "Gamma22 K*(892)- e+ nu_e",
            "(2.16 +- 0.16) %",
            16,
            resonant=True,
            note="all K*- decays.",
        ),
        sl(
            ["K-", "pi0", "e+", "nu_e"],
            1.6000e-02,
            (1.3e-02, 5.0e-03),
            "Gamma24 K- pi0 e+ nu_e",
            "(1.6 +1.3 -0.5) %",
            16,
            in_sum=False,
            note="old, poorly measured total; overlaps K*- e nu "
            "(K*- -> K- pi0 = 1/3).",
        ),
        sl(
            ["K0bar", "pi-", "e+", "nu_e"],
            1.4470e-02,
            3.3e-04,
            "Gamma25 K0bar pi- e+ nu_e",
            "(1.447 +- 0.033) %",
            16,
            in_sum=False,
            note="overlaps K*- e nu (K*- -> K0bar pi- = 2/3) and the S-wave "
            "piece G26.",
        ),
        sl(
            ["K0bar", "pi-", "e+", "nu_e"],
            8.5000e-04,
            6.0e-05,
            "Gamma26 (K0bar pi-)S-wave e+ nu_e",
            "(8.5 +- 0.6) x 10^-4",
            16,
            swave=True,
            note="complement of K*- e nu, below 0.1 %. (K- pi0)S-wave e nu "
            "not measured.",
        ),
        sl(
            ["pi-", "e+", "nu_e"],
            2.9100e-03,
            4.0e-05,
            "Gamma39 pi- e+ nu_e",
            "(2.91 +- 0.04) x 10^-3",
            16,
        ),
        sl(
            ["pi-", "pi0", "e+", "nu_e"],
            1.4500e-03,
            7.0e-05,
            "Gamma41 pi- pi0 e+ nu_e",
            "(1.45 +- 0.07) x 10^-3",
            16,
            in_sum=False,
            note="overlaps rho- e nu.",
        ),
        sl(
            ["rho-", "e+", "nu_e"],
            1.4600e-03,
            8.0e-05,
            "Gamma42 rho- e+ nu_e",
            "(1.46 +- 0.08) x 10^-3 S=2.0",
            16,
            resonant=True,
        ),
        sl(
            ["K-", "mu+", "nu_mu"],
            3.4180e-02,
            1.9e-04,
            "Gamma21 K- mu+ nu_mu",
            "(3.418 +- 0.019) %",
            16,
        ),
        sl(
            ["Kstar-", "mu+", "nu_mu"],
            2.0100e-02,
            6.0e-04,
            "Gamma23 K*(892)- mu+ nu_mu",
            "(2.01 +- 0.06) % S=1.7",
            16,
            resonant=True,
            note="all K*- decays.",
        ),
        sl(
            ["K-", "pi0", "mu+", "nu_mu"],
            7.1000e-03,
            2.1e-04,
            "Gamma27 K- pi0 mu+ nu_mu",
            "(7.10 +- 0.21) x 10^-3 S=1.7",
            16,
            in_sum=False,
            note="overlaps K*- mu nu and G28.",
        ),
        sl(
            ["K-", "pi0", "mu+", "nu_mu"],
            4.0900e-04,
            3.5e-05,
            "Gamma28 (K- pi0)S-wave mu+ nu_mu",
            "(4.09 +- 0.35) x 10^-4",
            16,
            swave=True,
            note="complement of K*- mu nu, below 0.1 %.",
        ),
        sl(
            ["K0bar", "pi-", "mu+", "nu_mu"],
            1.4000e-02,
            4.0e-04,
            "Gamma29 K0bar pi- mu+ nu_mu",
            "(1.40 +- 0.04) % S=1.7",
            16,
            in_sum=False,
            note="overlaps K*- mu nu and G30.",
        ),
        sl(
            ["K0bar", "pi-", "mu+", "nu_mu"],
            7.5000e-04,
            1.6e-04,
            "Gamma30 (K0bar pi-)S-wave mu+ nu_mu",
            "(7.5 +- 1.6) x 10^-4",
            16,
            swave=True,
            note="complement of K*- mu nu, below 0.1 %.",
        ),
        sl(
            ["pi-", "mu+", "nu_mu"],
            2.6700e-03,
            1.2e-04,
            "Gamma40 pi- mu+ nu_mu",
            "(2.67 +- 0.12) x 10^-3 S=1.3",
            16,
        ),
        sl(
            ["rho-", "mu+", "nu_mu"],
            1.3500e-03,
            1.3e-04,
            "Gamma43 rho- mu+ nu_mu",
            "(1.35 +- 0.13) x 10^-3",
            16,
            resonant=True,
            note="pi- pi0 mu+ nu_mu total not measured.",
        ),
    ],
    leptonic=[],
    leptonic_limits=[],
    excluded=[
        "Gamma244 2pi+ 2pi- 2pi0 (4.8 +- 0.4) x 10^-3, p23; Gamma117 K- pi+ "
        "pi0 eta (4.49 +- 0.27) x 10^-3, p19; Gamma266 K+ K- (4.07 +- 0.06) x "
        "10^-3, p24",
        "Gamma279 K0S K+ pi- (2.19 +- 0.34) x 10^-3, p24: x2 = 0.438 % "
        "(K0bar K+ pi-); Gamma119 K0S 2pi0 eta (1.76 +- 0.26) x 10^-3, p19: "
        "x2 = 0.352 %",
        "Gamma144 3K0S (7.6 +- 0.7) x 10^-4, p20: K0bar K0 K0bar = x4 if two "
        "kaons come from a C-even K0K0bar (a0/f0) pair = 0.30 %, x8 if all "
        "three are incoherent = 0.61 %; excluded with the (physical) x4",
        "Gamma242 2pi+ 2pi- pi0 (3.46 +- 0.21) x 10^-3; Gamma289 K+ K- pi0 "
        "(3.41 +- 0.14) x 10^-3; Gamma255 pi+ pi- pi0 eta "
        "(3.23 +- 0.22) x 10^-3; Gamma303 K+ K- pi+ pi- (2.78 +- 0.17) x 10^-3",
        'resonant / "corrected for unseen decays" modes whose final states are '
        "covered by totals: Gamma131 K*0bar pi+ pi- pi0 1.9 %, Gamma132 K*0bar "
        "eta 1.41 %, Gamma108 K*0bar omega 1.1 %, Gamma69 K- rho+ 11.2 %, "
        "Gamma92 K- a1(1260)+ 4.31 %, ...",
        "K0L modes Gamma49-52, 78, 142 (quoted in k0)",
    ],
)

DECAYS["Ds+"] = dict(
    hadronic=[
        mode(
            ["K+", "K0bar"],
            2.9500e-02,
            1.4e-03,
            "Gamma51 K+ K0bar",
            "(2.95 +- 0.14) %",
            5,
            k0="PDG lists K+ K0bar itself; cross-check K+K0S 1.500 % + K+K0L "
            "1.49 % = 2.99 %.",
        ),
        mode(
            ["K+", "K-", "pi+"],
            3.1955e-02,
            2.2e-03,
            "Gamma52 K+ K- pi+",
            "(5.45 +- 0.08) % S=1.3 [f]",
            5,
            "remainder",
            "G52 - G53*B_PHI_KK",
            note="minus phi pi+ (phi->K+K-) as generated from the listed phi "
            "pi+ (2.25 %); the PDG Dalitz fit fraction G54 = 2.25 % "
            "(footnote [g]: decoupled from G53).",
            inputs=["Gamma53 phi pi+ = (4.5 +- 0.4) % [d,g] (p5)"],
        ),
        mode(
            ["phi", "pi+"],
            4.5000e-02,
            4.0e-03,
            "Gamma53 phi pi+",
            "(4.5 +- 0.4) % [d,g]",
            5,
            note="all phi decays [d].",
        ),
        mode(
            ["K+", "K0bar", "pi0"],
            2.9420e-02,
            5.6e-04,
            "Gamma65 K+ K0S pi0",
            "(1.471 +- 0.028) %",
            5,
            "k0s",
            "2*G65",
            k0="no K0L measurement; no 2Gamma(K0S) warning in the Ds header.",
        ),
        mode(
            ["K0", "K0bar", "pi+"],
            1.8120e-02,
            1.6e-03,
            "Gamma46 K0S K0L pi+",
            "(1.86 +- 0.07) %",
            5,
            "derived",
            "2*G67 + G46 - G53*B_PHI_KSKL",
            k0="K0 K0bar pi+ = K0SK0S + K0SK0L + K0LK0L with K0LK0L = K0SK0S "
            "(true for C-even and for incoherent pairs): 2*G67 + G46.",
            note="minus phi pi+ (phi -> K0L K0S, only SL) as generated from "
            "phi pi+; measured G47 phi pi+, phi->K0LK0S = 1.31 %. G71 K0 K0bar "
            "pi+ has no value.",
            inputs=[
                "Gamma53 phi pi+ = (4.5 +- 0.4) % [d,g] (p5)",
                "Gamma67 2K0S pi+ = (7.32 +- 0.14) x 10^-3 (p5)",
            ],
        ),
        mode(
            ["phi", "rho+"],
            4.9000e-02,
            8.0e-03,
            "Gamma73 phi rho+",
            "(4.9 +- 0.8) % S=3.0",
            5,
            "resonant",
            "G73",
            note="PDG gives no phi pi+ pi0 total. All phi decays.",
        ),
        mode(
            ["K+", "K-", "pi+", "pi0"],
            1.8218e-02,
            4.4e-03,
            "Gamma74 K+ K- pi+ pi0",
            "(5.53 +- 0.15) % S=1.3",
            5,
            "remainder",
            "G74 - G73*B_PHI_KK - G84*(1/3)*(2/3)",
            note="ABLIKIM 24BD total; K*+ K*0bar part: K*+->K+pi0, "
            "K*0bar->K-pi+ (1/3 x 2/3).",
            inputs=[
                "Gamma73 phi rho+ = (4.9 +- 0.8) % S=3.0 (p5)",
                "Gamma84 K*(892)+ K*(892)0bar = (5.64 +- 0.35) % [d] (p6)",
            ],
        ),
        mode(
            ["K0", "K-", "pi+", "pi+"],
            6.3133e-03,
            1.7e-03,
            "Gamma82 K0S K- 2pi+",
            "(1.569 +- 0.028) %",
            6,
            "remainder",
            "2*G82 - G84*(2/3)*(2/3)",
            k0="neutral kaon is K0, from K*+ -> K0 pi+.",
            note="K*+ K*0bar part: K*+->K0 pi+, K*0bar->K- pi+ (2/3 x 2/3).",
            inputs=["Gamma84 K*(892)+ K*(892)0bar = (5.64 +- 0.35) % [d] (p6)"],
        ),
        mode(
            ["K+", "K0bar", "pi+", "pi-"],
            1.8680e-02,
            4.4e-04,
            "Gamma89 K+ K0S pi+ pi-",
            "(9.34 +- 0.22) x 10^-3",
            6,
            "k0s",
            "2*G89",
        ),
        mode(
            ["Kstar+", "Kstar0bar"],
            5.6400e-02,
            3.5e-03,
            "Gamma84 K*(892)+ K*(892)0bar",
            "(5.64 +- 0.35) % [d]",
            6,
            "resonant",
            "G84",
            note="PDG gives no totals for two of its final states "
            "(K0 K0bar pi+ pi0: 2/9, K+ K0bar pi0 pi0: 1/9); its K0 K- 2pi+ "
            "(4/9) and K+ K- pi+ pi0 (2/9) parts are subtracted from those "
            "totals. All K* decays [d].",
        ),
        mode(
            ["K+", "K-", "pi+", "pi+", "pi-"],
            5.3790e-04,
            1.0e-03,
            "Gamma90 K+ K- 2pi+ pi-",
            "(6.6 +- 0.6) x 10^-3",
            6,
            "remainder",
            "G90 - G91*B_PHI_KK",
            note="PDG total >= 0.5 %, remainder (0.05 %) kept.",
            inputs=["Gamma91 phi 2pi+ pi- = (1.21 +- 0.16) % [d] (p6)"],
        ),
        mode(
            ["phi", "pi+", "pi+", "pi-"],
            1.2100e-02,
            1.6e-03,
            "Gamma91 phi 2pi+ pi-",
            "(1.21 +- 0.16) % [d]",
            6,
            note="all phi decays [d].",
        ),
        mode(
            ["pi+", "pi+", "pi-"],
            1.0900e-02,
            1.4e-04,
            "Gamma99 2pi+ pi-",
            "(1.090 +- 0.014) %",
            6,
            note="omega pi+ (omega->pi+pi-) and f0 pi+ etc. are submodes; "
            "omega pi+ not listed.",
        ),
        mode(
            ["pi+", "pi0", "pi0"],
            5.2000e-03,
            5.0e-04,
            "Gamma110 pi+ 2pi0",
            "(5.2 +- 0.5) x 10^-3 S=1.1",
            7,
        ),
        mode(
            ["pi+", "pi+", "pi-", "pi0"],
            1.3214e-02,
            1.1e-03,
            "Gamma117 (2pi+ pi- pi0)non-eta",
            "(2.04 +- 0.09) %",
            7,
            "remainder",
            "G117 - G53*B_PHI_3PI - G143*B_OMEGA_2PI - G154*B_ETAP_3PIC",
            note="PDG gives (2pi+pi-pi0)non-eta (G114 2pi+pi-pi0 has no value; "
            "eta pi+ is listed separately). Minus phi pi+ (phi->3pi; "
            "measured submode G124 phi->rho pi = 0.508 %). omega pi+ "
            "(0.19 %) not listed, stays in.",
            inputs=[
                "Gamma143 omega pi+ pi0 = (2.8 +- 0.7) % [d] (p8)",
                "Gamma154 eta'(958) pi+ = (3.95 +- 0.08) % [c,d] (p8)",
                "Gamma53 phi pi+ = (4.5 +- 0.4) % [d,g] (p5)",
            ],
        ),
        mode(
            ["eta", "pi+"],
            1.6860e-02,
            2.7e-04,
            "Gamma116 eta pi+",
            "(1.686 +- 0.027) % [d]",
            7,
            note="all eta decays [d].",
        ),
        mode(
            ["pi+", "pi+", "pi+", "pi-", "pi-"],
            7.7552e-03,
            8.0e-04,
            "Gamma129 3pi+ 2pi-",
            "(8.0 +- 0.8) x 10^-3",
            7,
            "remainder",
            "G129 - G153*B_OMEGA_2PI",
            inputs=["Gamma153 omega 2pi+ pi- = (1.6 +- 0.5) % [d] (p8)"],
        ),
        mode(
            ["pi+", "pi+", "pi-", "pi0", "pi0"],
            1.1699e-02,
            6.7e-03,
            "Gamma130 2pi+ pi- 2pi0",
            "(4.41 +- 0.20) %",
            7,
            "remainder",
            "G130 - G143*B_OMEGA_3PI - G73*B_PHI_3PI - G158*B_ETAP_3PIC",
            note="ABLIKIM 25F (arXiv:2501.04451): eta (from pi+ pi0 eta) and "
            "K0S vetoed; omega rho+, phi rho+, b1 pi, rho(1450) pi0 "
            "(omega->3pi) are inside.",
            inputs=[
                "Gamma143 omega pi+ pi0 = (2.8 +- 0.7) % [d] (p8)",
                "Gamma158 eta'(958) pi+ pi0 = (6.14 +- 0.18) % (p8)",
                "Gamma73 phi rho+ = (4.9 +- 0.8) % S=3.0 (p5)",
            ],
        ),
        mode(
            ["eta", "pi+", "pi0"],
            9.1000e-02,
            1.7e-03,
            "Gamma140 eta pi+ pi0",
            "(9.10 +- 0.17) %",
            8,
            note="incl. eta rho+ G139 = 8.9 %; all eta decays.",
        ),
        mode(
            ["omega", "pi+", "pi0"],
            2.8000e-02,
            7.0e-03,
            "Gamma143 omega pi+ pi0",
            "(2.8 +- 0.7) % [d]",
            8,
            note="all omega decays [d].",
        ),
        mode(
            ["eta", "pi+", "pi+", "pi-"],
            3.0800e-02,
            8.0e-04,
            "Gamma144 2pi+ pi- eta",
            "(3.08 +- 0.08) %",
            8,
            note="ABLIKIM 24BD (arXiv:2403.19256): eta'->pi+pi-eta vetoed, "
            "so no overlap with eta' pi+.",
        ),
        mode(
            ["pi+", "pi+", "pi+", "pi-", "pi-", "pi0"],
            2.1995e-02,
            3.2e-02,
            "Gamma152 3pi+ 2pi- pi0",
            "(4.9 +- 3.2) %",
            8,
            "remainder",
            "G152 - G153*B_OMEGA_3PI - G154*B_ETAP_PIPIETA*B_ETA_3PIC - "
            "G144*B_ETA_3PIC - G91*B_PHI_3PI",
            note="BARLAG 92C 0.049 +0.033 -0.030 (old, no vetoes, poorly "
            "measured). Remainder uncertain at the +-3 % level.",
            inputs=[
                "Gamma144 2pi+ pi- eta = (3.08 +- 0.08) % (p8)",
                "Gamma153 omega 2pi+ pi- = (1.6 +- 0.5) % [d] (p8)",
                "Gamma154 eta'(958) pi+ = (3.95 +- 0.08) % [c,d] (p8)",
                "Gamma91 phi 2pi+ pi- = (1.21 +- 0.16) % [d] (p6)",
            ],
        ),
        mode(
            ["omega", "pi+", "pi+", "pi-"],
            1.6000e-02,
            5.0e-03,
            "Gamma153 omega 2pi+ pi-",
            "(1.6 +- 0.5) % [d]",
            8,
            note="all omega decays [d] (GE 09A).",
        ),
        mode(
            ["etap", "pi+"],
            3.9500e-02,
            8.0e-04,
            "Gamma154 eta'(958) pi+",
            "(3.95 +- 0.08) % [c,d]",
            8,
            note="all eta' decays [c,d].",
        ),
        mode(
            ["omega", "eta", "pi+"],
            5.4000e-03,
            1.3e-03,
            "Gamma156 omega eta pi+",
            "(5.4 +- 1.3) x 10^-3 [d]",
            8,
            note="all omega and eta decays [d].",
        ),
        mode(
            ["etap", "pi+", "pi0"],
            6.1400e-02,
            1.8e-03,
            "Gamma158 eta'(958) pi+ pi0",
            "(6.14 +- 0.18) %",
            8,
            note="incl. eta' rho+ G157 = 5.8 %.",
        ),
        mode(
            ["K+", "pi+", "pi-"],
            6.2300e-03,
            1.0e-04,
            "Gamma165 K+ pi+ pi-",
            "(6.23 +- 0.10) x 10^-3",
            9,
            note="ABLIKIM 24BD: K0S veto in pi+pi-.",
        ),
        mode(
            ["K0", "pi+", "pi0"],
            1.0180e-02,
            4.4e-04,
            "Gamma176 K0S pi+ pi0",
            "(5.09 +- 0.22) x 10^-3",
            9,
            "k0s",
            "2*G176",
            k0="neutral kaon is K0: Cabibbo-suppressed c -> d.",
        ),
        mode(
            ["K0", "pi+", "pi+", "pi-"],
            5.6000e-03,
            2.0e-03,
            "Gamma182 K0S 2pi+ pi-",
            "(2.8 +- 1.0) x 10^-3",
            9,
            "k0s",
            "2*G182",
            k0="neutral kaon is K0.",
        ),
        mode(
            ["K+", "pi+", "pi-", "pi0"],
            9.7000e-03,
            6.0e-04,
            "Gamma183 K+ pi+ pi- pi0",
            "(9.7 +- 0.6) x 10^-3",
            9,
            note="K+ omega (0.099 %) and K+ eta not listed, so no subtraction.",
        ),
    ],
    semileptonic=[
        sl(
            ["phi", "e+", "nu_e"],
            2.3400e-02,
            1.2e-03,
            "Gamma26 phi e+ nu_e",
            "(2.34 +- 0.12) % S=1.2 [d]",
            4,
            resonant=True,
            note="all phi decays [d].",
        ),
        sl(
            ["eta", "e+", "nu_e"],
            2.2700e-02,
            6.0e-04,
            "Gamma30 eta e+ nu_e",
            "(2.27 +- 0.06) % [d]",
            4,
        ),
        sl(
            ["etap", "e+", "nu_e"],
            8.1000e-03,
            4.0e-04,
            "Gamma31 eta'(958) e+ nu_e",
            "(8.1 +- 0.4) x 10^-3 [d]",
            4,
        ),
        sl(
            ["K0", "e+", "nu_e"],
            2.8800e-03,
            2.6e-04,
            "Gamma35 K0 e+ nu_e",
            "(2.88 +- 0.26) x 10^-3 S=1.2",
            4,
        ),
        sl(
            ["Kstar0", "e+", "nu_e"],
            2.0500e-03,
            2.0e-04,
            "Gamma36 K*(892)0 e+ nu_e",
            "(2.05 +- 0.20) x 10^-3 [d]",
            4,
            resonant=True,
            note="all K*0 decays [d].",
        ),
        sl(
            ["f0_980", "e+", "nu_e"],
            2.4300e-03,
            2.0e-04,
            "Gamma40 f0(980) e+ nu_e, f0 -> pi+ pi-",
            "(1.64 +- 0.13) x 10^-3",
            4,
            "derived",
            "G40 + G39",
            resonant=True,
            note="the f0 -> pi pi part (G40 + G39); f0 -> K Kbar not measured.",
            inputs=[
                "Gamma39 f0(980) e+ nu_e, f0 -> pi0 pi0 = (7.9 +- 1.5) x "
                "10^-4 (p4)"
            ],
        ),
        sl(
            ["phi", "mu+", "nu_mu"],
            2.2400e-02,
            1.1e-03,
            "Gamma29 phi mu+ nu_mu",
            "(2.24 +- 0.11) %",
            4,
            resonant=True,
        ),
        sl(
            ["eta", "mu+", "nu_mu"],
            2.2400e-02,
            7.0e-04,
            "Gamma32 eta mu+ nu_mu",
            "(2.24 +- 0.07) %",
            4,
        ),
        sl(
            ["etap", "mu+", "nu_mu"],
            8.0000e-03,
            6.0e-04,
            "Gamma33 eta'(958) mu+ nu_mu",
            "(8.0 +- 0.6) x 10^-3",
            4,
            note="K0 mu nu, K*0 mu nu and f0(980) mu nu (f0->pi pi) are not "
            "measured (only f0 mu nu, f0->K+K- < 5.45e-4).",
        ),
    ],
    leptonic=[
        mode(
            ["mu+", "nu_mu"],
            5.3700e-03,
            1.1e-04,
            "Gamma21 mu+ nu_mu",
            "(5.37 +- 0.11) x 10^-3",
            4,
        ),
        mode(
            ["tau+", "nu_tau"],
            5.3900e-02,
            9.0e-04,
            "Gamma22 tau+ nu_tau",
            "(5.39 +- 0.09) %",
            4,
        ),
    ],
    leptonic_limits=[
        dict(
            mode="e+ nu_e",
            limit=8.3e-05,
            pdg="Gamma20 < 8.3 x 10^-5 CL=90%",
            page=4,
        ),
    ],
    excluded=[
        "Gamma164 K+ eta' (2.68 +- 0.24) x 10^-3; Gamma115 omega pi+ "
        "(1.93 +- 0.30) x 10^-3; Gamma162 K+ eta (1.76 +- 0.08) x 10^-3; "
        "Gamma202 p nbar (1.22 +- 0.11) x 10^-3; Gamma97 2K0S 2pi+ pi- "
        "(8.0 +- 3.3) x 10^-4 (x4 = 0.32 %)",
        "resonant modes not used (totals exist): Gamma55 K+ K*0bar "
        "(12.7 +4.0 -3.1) % (= Gamma55/Gamma72 x Gamma72, p23; inconsistent "
        "with Gamma56 = 2.61 %), Gamma72 K*+ K0bar 5.4 %, Gamma139 eta rho+ "
        "8.9 %, Gamma157 eta' rho+ 5.8 %",
        "Gamma44 K+ K0S, Gamma45 K+ K0L: "
        "superseded by the K+ K0bar entry Gamma51",
    ],
)

DECAYS["Lambda_c+"] = dict(
    hadronic=[
        mode(
            ["p", "K0bar"],
            3.2600e-02,
            1.4e-03,
            "Gamma1 p K0S",
            "(1.63 +- 0.07) % S=1.1",
            2,
            "k0s",
            "2*G1",
            k0="p K0L (G2) = (1.67 +- 0.07) % (consistent).",
        ),
        mode(
            ["p", "K-", "pi+"],
            6.3700e-02,
            2.1e-03,
            "Gamma3 p K- pi+",
            "(6.37 +- 0.21) % S=1.2",
            2,
            note="incl. p K*0bar, Delta++ K-, Lambda(1520) pi+, ... (G4-G16).",
        ),
        mode(
            ["p", "K0bar", "pi0"],
            4.2400e-02,
            1.8e-03,
            "Gamma17 p K0S pi0",
            "(2.12 +- 0.09) % S=1.3",
            3,
            "k0s",
            "2*G17",
            k0="p K0L pi0 (G18) = (2.02 +- 0.14) %.",
        ),
        mode(
            ["n", "K0bar", "pi+"],
            3.7200e-02,
            1.8e-03,
            "Gamma19 n K0S pi+",
            "(1.86 +- 0.09) %",
            3,
            "k0s",
            "2*G19",
        ),
        mode(
            ["n", "K0bar", "pi+", "pi0"],
            1.7000e-02,
            2.6e-03,
            "Gamma21 n K0S pi+ pi0",
            "(8.5 +- 1.3) x 10^-3",
            3,
            "k0s",
            "2*G21",
        ),
        mode(
            ["n", "K-", "pi+", "pi+"],
            1.9000e-02,
            1.2e-03,
            "Gamma22 n K- pi+ pi+",
            "(1.90 +- 0.12) %",
            3,
        ),
        mode(
            ["p", "K0bar", "eta"],
            9.0000e-03,
            6.0e-04,
            "Gamma23 p K0bar eta",
            "(9.0 +- 0.6) x 10^-3 S=1.1",
            3,
            k0="PDG lists p K0bar eta itself.",
        ),
        mode(
            ["p", "K0bar", "pi+", "pi-"],
            3.2600e-02,
            2.0e-03,
            "Gamma24 p K0S pi+ pi-",
            "(1.63 +- 0.10) % S=1.1",
            3,
            "k0s",
            "2*G24",
            k0="p K0L pi+ pi- (G25) = (1.69 +- 0.11) %.",
        ),
        mode(
            ["p", "K-", "pi+", "pi0"],
            4.5400e-02,
            2.7e-03,
            "Gamma26 p K- pi+ pi0",
            "(4.54 +- 0.27) % S=1.6",
            3,
            note="incl. p K*- pi+, p (K-pi+)nonres pi0.",
        ),
        mode(
            ["p", "K-", "pi+", "pi0", "pi0"],
            1.0000e-02,
            5.0e-03,
            "Gamma31 p K- pi+ 2pi0",
            "(1.0 +- 0.5) %",
            3,
        ),
        mode(
            ["n", "pi+", "pi0"],
            6.4000e-03,
            9.0e-04,
            "Gamma40 n pi+ pi0",
            "(6.4 +- 0.9) x 10^-3",
            3,
            note="ABLIKIM 23A (arXiv:2210.03375): Lambda pi+ (Lambda->n pi0), "
            "Sigma+ pi0 (Sigma+->n pi+), Sigma0 pi+ rejected.",
        ),
        mode(
            ["Lambda", "pi+"],
            1.3200e-02,
            5.0e-04,
            "Gamma50 Lambda pi+",
            "(1.32 +- 0.05) % S=1.1",
            3,
        ),
        mode(
            ["Lambda", "pi+", "pi0"],
            7.1600e-02,
            3.3e-03,
            "Gamma52 Lambda pi+ pi0",
            "(7.16 +- 0.33) %",
            4,
            note="incl. Lambda rho+ (4.1 %), Sigma(1385) pi.",
        ),
        mode(
            ["Lambda", "pi+", "pi+", "pi-"],
            3.6671e-02,
            2.6e-03,
            "Gamma56 Lambda pi- 2pi+",
            "(3.69 +- 0.26) % S=1.5",
            4,
            "remainder",
            "G56 - G63*B_OMEGA_2PI",
            inputs=["Gamma63 Lambda pi+ omega = (1.5 +- 0.5) % [a] (p4)"],
        ),
        mode(
            ["Lambda", "pi+", "pi+", "pi-", "pi0"],
            5.2002e-03,
            9.2e-03,
            "Gamma62 Lambda pi- pi0 2pi+ total",
            "(2.3 +- 0.8) %",
            4,
            "remainder",
            "G62 - G63*B_OMEGA_3PI - G65*B_ETA_3PIC",
            note='PDG "Lambda pi- pi0 2pi+ total"; CRONIN-HENNESSY 03 finds it '
            "dominantly Lambda eta pi+ and Lambda omega pi+ (listing p19). "
            "PDG no-eta-or-omega limit G64 < 8e-3.",
            inputs=[
                "Gamma63 Lambda pi+ omega = (1.5 +- 0.5) % [a] (p4)",
                "Gamma65 Lambda pi+ eta = (1.92 +- 0.06) % [a] (p4)",
            ],
        ),
        mode(
            ["Lambda", "pi+", "omega"],
            1.5000e-02,
            5.0e-03,
            "Gamma63 Lambda pi+ omega",
            "(1.5 +- 0.5) % [a]",
            4,
            note="all omega decays [a].",
        ),
        mode(
            ["Lambda", "pi+", "eta"],
            1.9200e-02,
            6.0e-04,
            "Gamma65 Lambda pi+ eta",
            "(1.92 +- 0.06) % [a]",
            4,
            note="all eta decays [a].",
        ),
        mode(
            ["Lambda", "K+", "K0bar"],
            5.9000e-03,
            5.0e-04,
            "Gamma69 Lambda K+ K0bar",
            "(5.9 +- 0.5) x 10^-3 S=1.2",
            4,
            k0="PDG lists Lambda K+ K0bar itself.",
        ),
        mode(
            ["Sigma0", "pi+"],
            1.2900e-02,
            5.0e-04,
            "Gamma71 Sigma0 pi+",
            "(1.29 +- 0.05) % S=1.1",
            4,
        ),
        mode(
            ["Sigma0", "pi+", "eta"],
            7.6000e-03,
            8.0e-04,
            "Gamma72 Sigma0 pi+ eta",
            "(7.6 +- 0.8) x 10^-3",
            4,
        ),
        mode(
            ["Sigma+", "pi0"],
            1.2700e-02,
            1.0e-03,
            "Gamma73 Sigma+ pi0",
            "(1.27 +- 0.10) % S=1.1",
            4,
        ),
        mode(
            ["Sigma+", "pi+", "pi-"],
            4.5437e-02,
            1.8e-03,
            "Gamma76 Sigma+ pi+ pi-",
            "(4.57 +- 0.18) % S=1.1",
            4,
            "remainder",
            "G76 - G82*B_OMEGA_2PI",
            inputs=["Gamma82 Sigma+ omega = (1.72 +- 0.20) % (p4)"],
        ),
        mode(
            ["Sigma-", "pi+", "pi+"],
            1.8700e-02,
            1.8e-03,
            "Gamma78 Sigma- 2pi+",
            "(1.87 +- 0.18) %",
            4,
        ),
        mode(
            ["Sigma0", "pi+", "pi0"],
            3.6000e-02,
            4.0e-03,
            "Gamma79 Sigma0 pi+ pi0",
            "(3.6 +- 0.4) %",
            4,
        ),
        mode(
            ["Sigma+", "pi0", "pi0"],
            1.5700e-02,
            1.4e-03,
            "Gamma80 Sigma+ pi0 pi0",
            "(1.57 +- 0.14) %",
            4,
        ),
        mode(
            ["Sigma0", "pi+", "pi+", "pi-"],
            1.1300e-02,
            3.1e-03,
            "Gamma81 Sigma0 pi- 2pi+",
            "(1.13 +- 0.31) %",
            4,
        ),
        mode(
            ["Sigma+", "omega"],
            1.7200e-02,
            2.0e-03,
            "Gamma82 Sigma+ omega",
            "(1.72 +- 0.20) %",
            4,
        ),
        mode(
            ["Sigma-", "pi+", "pi+", "pi0"],
            2.1000e-02,
            4.0e-03,
            "Gamma83 Sigma- pi0 2pi+",
            "(2.1 +- 0.4) %",
            4,
        ),
        mode(
            ["Xi0", "K+"],
            5.5000e-03,
            7.0e-04,
            "Gamma90 Xi0 K+",
            "(5.5 +- 0.7) x 10^-3",
            4,
        ),
        mode(
            ["Xi-", "K+", "pi+"],
            6.3000e-03,
            5.0e-04,
            "Gamma91 Xi- K+ pi+",
            "(6.3 +- 0.5) x 10^-3",
            4,
            note="incl. Xi(1530)0 K+ (G94 0.49 %, "
            "Xi(1530) not in particle list).",
        ),
        mode(
            ["Xi0", "K+", "pi0"],
            7.8000e-03,
            1.7e-03,
            "Gamma92 Xi0 K+ pi0",
            "(7.8 +- 1.7) x 10^-3",
            4,
        ),
        mode(
            ["Xi0", "K0", "pi+"],
            7.4000e-03,
            1.2e-03,
            "Gamma93 Xi0 K0S pi+",
            "(3.7 +- 0.6) x 10^-3",
            5,
            "k0s",
            "2*G93",
            k0="neutral kaon is K0: Xi0 K0 pi+ has S = -1.",
        ),
    ],
    semileptonic=[
        sl(
            ["Lambda", "e+", "nu_e"],
            3.5600e-02,
            1.3e-03,
            "Gamma109 Lambda e+ nu_e",
            "(3.56 +- 0.13) %",
            5,
        ),
        sl(
            ["Lambda1520", "e+", "nu_e"],
            1.0000e-03,
            5.0e-04,
            "Gamma113 Lambda(1520) e+ nu_e",
            "(1.0 +- 0.5) x 10^-3",
            5,
            resonant=True,
            note="Lambda(1520) is not in the particle list (it would need its "
            "own decays). Below 0.1 %: "
            "p K- e nu (G111) = (8.8 +- 1.8) x 10^-4, "
            "Lambda(1405) e nu, ->p K- (G114) = (4.2 +- 1.9) x 10^-4.",
        ),
        sl(
            ["Lambda", "mu+", "nu_mu"],
            3.4800e-02,
            1.7e-03,
            "Gamma115 Lambda mu+ nu_mu",
            "(3.48 +- 0.17) %",
            5,
            note="no other mu modes measured.",
        ),
    ],
    leptonic=[],
    leptonic_limits=[],
    excluded=[
        "Gamma94 Xi(1530)0 K+ (4.9 +- 0.6) x 10^-3 (resonant; inside the Xi- "
        "K+ pi+ and Xi0 K+ pi0 totals)",
        "Gamma37 p pi+ pi- (4.69 +- 0.22) x 10^-3; Gamma42 n pi+ pi- pi+ "
        "(4.5 +- 0.8) x 10^-3; Gamma75 Sigma+ eta' (4.2 +- 0.9) x 10^-3; "
        "Gamma87 Sigma+ phi (4.0 +- 0.5) x 10^-3; Gamma85 Sigma+ K+ K- "
        "(3.66 +- 0.35) x 10^-3; Gamma74 Sigma+ eta (3.4 +- 0.4) x 10^-3; "
        "Gamma97 Lambda K0S pi+ (1.73 +- 0.28) x 10^-3 (x2 = 0.346 %)",
        "resonant submodes not used: Lambda rho+ 4.1 %, p K*0bar 1.42 %, "
        "Delta++ K- 1.80 %, Lambda pi+ rho0 1.5 %, p K*- pi+ 1.4 %, "
        "Sigma(1385) modes, Sigma(1385)+ eta 0.67 %, ...",
    ],
)

# tau- basis modes [a] ("Basis mode for the tau", constrained fit summing to
# one, p13) with BR >= 0.5 %. The (ex.K0) and (ex.K0,omega) basis modes are
# used, so K0bar and omega modes are not double counted.
TAU_DECAYS = [
    mode(
        ["mu-", "nu_mu_bar", "nu_tau"],
        1.7370e-01,
        4.0e-04,
        "Gamma3 mu- nubar_mu nu_tau",
        "(17.37 +- 0.04) % [a]",
        6,
        fs_tau_plus=["mu+", "nu_mu", "nu_tau_bar"],
        note="includes mu nu nu gamma (G4).",
    ),
    mode(
        ["e-", "nu_e_bar", "nu_tau"],
        1.7850e-01,
        4.0e-04,
        "Gamma5 e- nubar_e nu_tau",
        "(17.85 +- 0.04) % [a]",
        6,
        fs_tau_plus=["e+", "nu_e", "nu_tau_bar"],
        note="includes e nu nu gamma (G6).",
    ),
    mode(
        ["pi-", "nu_tau"],
        1.0820e-01,
        5.0e-04,
        "Gamma9 pi- nu_tau",
        "(10.82 +- 0.05) % [a]",
        6,
        fs_tau_plus=["pi+", "nu_tau_bar"],
    ),
    mode(
        ["K-", "nu_tau"],
        6.9700e-03,
        1.0e-04,
        "Gamma10 K- nu_tau",
        "(6.97 +- 0.10) x 10^-3 [a]",
        6,
        fs_tau_plus=["K+", "nu_tau_bar"],
    ),
    mode(
        ["pi-", "pi0", "nu_tau"],
        2.5490e-01,
        9.0e-04,
        "Gamma14 pi- pi0 nu_tau",
        "(25.49 +- 0.09) % [a]",
        7,
        fs_tau_plus=["pi+", "pi0", "nu_tau_bar"],
        note="dominantly rho- nu (non-rho part G15 = (3.0 +- 3.2) x 10^-3).",
    ),
    mode(
        ["pi-", "pi0", "pi0", "nu_tau"],
        9.2600e-02,
        1.0e-03,
        "Gamma20 pi- 2pi0 nu_tau (ex.K0)",
        "(9.26 +- 0.10) % [a]",
        7,
        fs_tau_plus=["pi+", "pi0", "pi0", "nu_tau_bar"],
        note="K0 -> pi0 pi0 contributions are in the K0 modes.",
    ),
    mode(
        ["pi-", "pi0", "pi0", "pi0", "nu_tau"],
        1.0400e-02,
        7.0e-04,
        "Gamma27 pi- 3pi0 nu_tau (ex.K0)",
        "(1.04 +- 0.07) % [a]",
        7,
        fs_tau_plus=["pi+", "pi0", "pi0", "pi0", "nu_tau_bar"],
    ),
    mode(
        ["pi-", "K0bar", "nu_tau"],
        8.3800e-03,
        1.4e-04,
        "Gamma36 pi- K0bar nu_tau",
        "(8.38 +- 0.14) x 10^-3 [a]",
        7,
        fs_tau_plus=["pi+", "K0", "nu_tau_bar"],
        note="PDG lists K0bar itself (dominantly K*(892)- nu).",
    ),
    mode(
        ["pi-", "pi+", "pi-", "nu_tau"],
        8.9900e-02,
        5.0e-04,
        "Gamma70 pi- pi+ pi- nu_tau (ex.K0,omega)",
        "(8.99 +- 0.05) % [a]",
        8,
        fs_tau_plus=["pi+", "pi-", "pi+", "nu_tau_bar"],
        note="omega pi- is listed separately.",
    ),
    mode(
        ["pi-", "pi+", "pi-", "pi0", "nu_tau"],
        2.7400e-02,
        7.0e-04,
        "Gamma78 pi- pi+ pi- pi0 nu_tau (ex.K0,omega)",
        "(2.74 +- 0.07) % [a]",
        8,
        fs_tau_plus=["pi+", "pi-", "pi+", "pi0", "nu_tau_bar"],
    ),
    mode(
        ["pi-", "omega", "nu_tau"],
        1.9500e-02,
        6.0e-04,
        "Gamma178 pi- omega nu_tau",
        "(1.95 +- 0.06) % [a]",
        11,
        fs_tau_plus=["pi+", "omega", "nu_tau_bar"],
        note="all omega decays.",
    ),
]
# Left out: Gamma16 K- pi0 nu (4.33 +- 0.15) x 10^-3, Gamma180 h- omega pi0 nu
# (4.1 +- 0.4) x 10^-3, Gamma41 pi- K0bar pi0 nu (3.82 +- 0.13) x 10^-3,
# Gamma97 K- pi+ pi- nu (ex.K0,omega) (2.93 +- 0.07) x 10^-3 and the other basis
# modes, all < 0.5 %. Aggregate (non-basis) entries such as 1-prong,
# h- >= 1 neutrals, (5pi)- nu (7.8 +- 0.5) x 10^-3 (Gamma130) and K*(892)- nu
# are not final states. The largest left out bounds the selection from below:
TAU_LARGEST_EXCLUDED = dict(
    pdg="Gamma16 K- pi0 nu_tau",
    br=4.33e-3,
    pdg_value="(4.33 +- 0.15) x 10^-3 [a]",
    page=7,
)

# Tau decay parameters, PDG 2026 OUR FIT values (tau listing, "tau-DECAY
# PARAMETERS"): the Michel parameters of its leptonic decays and the chirality
# parameters xi of pi, rho and a1(1260). Each block prints "(V-A) theory
# predicts rho = 0.75." (xi = 1, eta = 0, (delta xi) = 0.75, xi(pi) = 1,
# xi(rho) = 1, xi(a1) = 1). TAU_VA gives these values.
TAU_DECAY_PARAMETERS = {
    "rho(e or mu)": (0.745, 0.008, 83),  # (value, error, page)
    "xi(e or mu)": (0.985, 0.030, 85),
    "eta(e or mu)": (0.013, 0.020, 86),
    "(delta xi)(e or mu)": (0.746, 0.021, 87),
    "xi(pi)": (0.993, 0.022, 88),
    "xi(rho)": (0.994, 0.008, 88),
    "xi(a1)": (1.001, 0.027, 89),
}
TAU_VA = {
    "rho(e or mu)": 0.75,
    "xi(e or mu)": 1,
    "eta(e or mu)": 0,
    "(delta xi)(e or mu)": 0.75,
    "xi(pi)": 1,
    "xi(rho)": 1,
    "xi(a1)": 1,
}

# K*(892): PDG gives (K pi)+- and (K pi)0 only (K-star-892 p9). The charge split
# uses the isospin Clebsch-Gordan factors 2/3 : 1/3.
_K_PM = ("Gamma2 (K pi)+-", 0.99896, 0.00010, "(99.896 +- 0.010) %")
_K_00 = ("Gamma3 (K pi)0", 0.99753, 0.00021, "(99.753 +- 0.021) %")


def kstar(fs_2, fs_1, gamma, value, err, pdg_value):
    return [
        entry(fs, f * value, f * err, f"{gamma} x {s} (isospin)", pdg_value, 9)
        for fs, f, s in ((fs_2, 2 / 3, "2/3"), (fs_1, 1 / 3, "1/3"))
    ]


# The short-lived particles of the D+ and D0 semileptonic modes (rule 6). 'page'
# is a page of the particle's own listing.
RESONANCE_DECAYS = {
    "eta": [
        entry(
            ["gamma", "gamma"],
            0.3936,
            0.0018,
            "Gamma2 2gamma",
            "(39.36 +- 0.18) % S=1.1",
            1,
        ),
        entry(
            ["pi0", "pi0", "pi0"],
            0.3256,
            0.0021,
            "Gamma3 3pi0",
            "(32.56 +- 0.21) % S=1.2",
            1,
        ),
        entry(
            ["pi+", "pi-", "pi0"],
            0.2302,
            0.0025,
            "Gamma9 pi+ pi- pi0",
            "(23.02 +- 0.25) % S=1.2",
            2,
        ),
        entry(
            ["pi+", "pi-", "gamma"],
            0.0428,
            0.0007,
            "Gamma10 pi+ pi- gamma",
            "(4.28 +- 0.07) % S=1.1",
            2,
        ),
        entry(
            ["e+", "e-", "gamma"],
            6.96e-3,
            0.18e-3,
            "Gamma11 e+ e- gamma",
            "(6.96 +- 0.18) x 10^-3",
            2,
        ),
    ],
    "omega": [
        entry(
            ["pi+", "pi-", "pi0"],
            0.892,
            0.007,
            "Gamma1 pi+ pi- pi0",
            "(89.2 +- 0.7) %",
            4,
        ),
        entry(
            ["pi0", "gamma"],
            0.0833,
            0.0025,
            "Gamma2 pi0 gamma",
            "(8.33 +- 0.25) % S=2.1",
            4,
        ),
        entry(
            ["pi+", "pi-"],
            0.0153,
            0.0012,
            "Gamma3 pi+ pi-",
            "(1.53 +- 0.12) % S=1.2",
            4,
        ),
        # not used: Gamma4 "neutrals (excluding pi0 gamma)" (7 +7 -5) x 10^-3
        # S=1.1, p4, not a definite final state (its identified pieces eta gamma
        # (4.5e-4) and pi0 pi0 gamma (6.7e-5) are tiny).
    ],
    "rho0": [
        entry(
            ["pi+", "pi-"],
            1.0,
            None,
            "Gamma6 pi+ pi-",
            "~ 100 %",
            12,
            note="Gamma7 pi+ pi- gamma (9.9 +- 1.6) x 10^-3 "
            "(radiative pi pi, photon-energy cut) not listed "
            "separately; all others < 0.5 %",
        ),
    ],
    "rho-": [
        entry(["pi-", "pi0"], 1.0, None, "Gamma2 pi+- pi0", "~ 100 %", 12)
    ],
    "Kstar-": kstar(["K0bar", "pi-"], ["K-", "pi0"], *_K_PM),
    "Kstar0bar": kstar(["K-", "pi+"], ["K0bar", "pi0"], *_K_00),
    # K*(892) radiative modes (K0 gamma (2.47 +- 0.21) x 10^-3, K+- gamma
    # (1.04 +- 0.10) x 10^-3, p9) are < 0.5 %.
}

# The rule of the K0S modes (rule 2), as the D+- listing states it in its decay
# table (p. 2, and the D0 listing alike on p. 15): interference "can invalidate
# the assumption that 2 Gamma(K0S) = Gamma(K0)".
K0S_ASSUMPTION = dict(text="2 Gamma(K0S) = Gamma(K0)", listing="D+-", page=2)

# PDG's inclusive e+ rates, to which the semileptonic modes are scaled (rule 7).
# "e+ semileptonic" (D+, Ds+) leaves out the e+ of tau decays (Ds+ footnote [a]:
# "the e+ fraction from tau+ decays has been subtracted off", p. 10). D0 and
# Lambda_c+ have no tau modes.
INCLUSIVE_E = {
    "D+": entry(
        ["e+", "X"],
        0.1607,
        0.0030,
        "Gamma1 e+ semileptonic",
        "(16.07 +- 0.30) %",
        2,
    ),
    "D0": entry(
        ["e+", "X"],
        0.0649,
        0.0011,
        "Gamma5 e+ anything",
        "(6.49 +- 0.11) % [d]",
        15,
    ),
    "Ds+": entry(
        ["e+", "X"],
        0.0633,
        0.0015,
        "Gamma1 e+ semileptonic",
        "(6.33 +- 0.15) % [a]",
        3,
    ),
    "Lambda_c+": entry(
        ["e+", "X"],
        0.0406,
        0.0013,
        "Gamma116 e+ anything",
        "(4.06 +- 0.13) %",
        5,
    ),
}

# The form-factor ratios r_V = V(0)/A1(0) and r_2 = A2(0)/A1(0) of the P -> V l+
# nu modes, from the SEMILEPTONIC FORM FACTORS blocks of the parent's listing:
# (value, error, label, page). A single measurement's error is (stat, syst).
# PDG writes r_v in the D+- and Ds+- listings, r_V in the D0 listing.
SL_FORM_FACTORS = {
    ("D+", "Kstar0bar"): dict(
        listing="D+-",
        block="r_v and r_2 in D+ -> K*(892)0bar l+ nu_l",
        rV=(1.47, 0.04, "OUR AVERAGE", 66),
        r2=(0.786, 0.018, "OUR AVERAGE", 67),
    ),
    ("D+", "rho0"): dict(
        listing="D+-",
        block="r_v and r_2 in D+,D0 -> rho e+ nu_e",
        rV=(1.53, 0.08, "OUR AVERAGE", 66),
        r2=(0.82, 0.05, "OUR AVERAGE", 66),
    ),
    ("D+", "omega"): dict(
        listing="D+-",
        block="r_v and r_2 in D+ -> omega e+ nu_e",
        rV=(1.24, (0.09, 0.06), "ABLIKIM 15W", 65),
        r2=(1.06, (0.15, 0.05), "ABLIKIM 15W", 65),
    ),
    ("D0", "Kstar-"): dict(
        listing="D0",
        block="r_V and r_2 in D0 -> K*(892)- l+ nu_l",
        rV=(1.45, 0.04, "OUR AVERAGE", 129),
        r2=(0.720, 0.033, "OUR AVERAGE", 129),
    ),
    # the D0 listing (p. 129): "See the D+ listings for combinations of
    # D0 -> rho- e+ nu_e and D+ -> rho0 e+ nu_e"
    ("D0", "rho-"): dict(
        listing="D+-",
        block="r_v and r_2 in D+,D0 -> rho e+ nu_e",
        rV=(1.53, 0.08, "OUR AVERAGE", 66),
        r2=(0.82, 0.05, "OUR AVERAGE", 66),
    ),
    ("Ds+", "phi"): dict(
        listing="Ds+-",
        block="r_v and r_2 in Ds+ -> phi l+ nu_l",
        rV=(1.76, 0.07, "OUR AVERAGE", 60),
        r2=(0.83, 0.08, "OUR AVERAGE", 59),
    ),
    ("Ds+", "Kstar0"): dict(
        listing="Ds+-",
        block="r_v and r_2 in Ds+ -> K*(892)0 e+ nu_e",
        rV=(1.67, (0.34, 0.16), "ABLIKIM 19D", 62),
        r2=(0.77, (0.28, 0.07), "ABLIKIM 19D", 62),
    ),
}

# I(J^P) of the charmed mesons that give the pole masses of the semileptonic
# form factors (p. 1 of each listing, PDG-2026_listing_<file>.pdf). Isospin 1/2:
# c qbar with q = u, d. Isospin 0: c sbar.
POLE_STATES = {
    "D*(2010)": ("1/2(1-)", "D-star-2010-plus-minus"),
    "D0*(2300)": ("1/2(0+)", "D0-star-2300"),
    "D1(2420)": ("1/2(1+)", "D1-2420"),
    "D": ("1/2(0-)", "D-plus-minus"),
    "Ds*": ("0(1-)", "Ds-star-plus-minus"),
    # "J, P need confirmation."
    "Ds0*(2317)": ("0(0+)", "Ds0-star-2317-plus-minus"),
    "Ds1(2460)": ("0(1+)", "Ds1-2460-plus-minus"),
    "Ds": ("0(0-)", "Ds-plus-minus"),
}


def below(pdg, br, pdg_value, page):
    return dict(pdg=pdg, br=br, pdg_value=pdg_value, page=page)


# The decays of the short-lived products of the charm decays (README:
# Short-lived products). Each particle has every mode of 0.5 % and above of its
# own listing ('modes'), the largest mode below 0.5 % ('below'), and the entries
# of 0.5 % and above that are no final state of their own ('not_final').
# K*(892)+ and K*(892)0 decay as the conjugates of K*(892)- and K*(892)0bar.
# f0(980) (listing p. 7: pi pi "seen", K Kbar "seen") and a1(1260)
# (p. 5: 3pi "seen") have no branching fractions. The limits are no modes:
# rho+- -> pi+- eta < 6 x 10^-3 and Sigma0 -> Lambda gamma gamma < 3 %.
SHORT_LIVED = {
    "eta": dict(
        modes=RESONANCE_DECAYS["eta"],
        below=below(
            "Gamma12 mu+ mu- gamma", 2.98e-4, "(2.98 +- 0.13) x 10^-4", 2
        ),
        not_final=[
            "Gamma1 neutral modes (71.95 +- 0.29) %",
            "Gamma8 charged modes (28.05 +- 0.29) %",
        ],
    ),
    "etap": dict(
        modes=[
            entry(
                ["pi+", "pi-", "eta"],
                0.425,
                0.005,
                "Gamma1 pi+ pi- eta",
                "(42.5 +- 0.5) %",
                1,
            ),
            entry(
                ["rho0", "gamma"],
                0.2948,
                0.0035,
                "Gamma2 rho0 gamma (including non-resonant pi+ pi- gamma)",
                "(29.48 +- 0.35) %",
                1,
            ),
            entry(
                ["pi0", "pi0", "eta"],
                0.224,
                0.005,
                "Gamma4 pi0 pi0 eta",
                "(22.4 +- 0.5) %",
                1,
            ),
            entry(
                ["omega", "gamma"],
                0.0252,
                0.0007,
                "Gamma5 omega gamma",
                "(2.52 +- 0.07) %",
                2,
            ),
            entry(
                ["gamma", "gamma"],
                0.02307,
                0.00033,
                "Gamma7 gamma gamma",
                "(2.307 +- 0.033) %",
                2,
            ),
        ],
        below=below(
            "Gamma12 (pi+ pi- pi0) S-wave", 3.8e-3, "(3.8 +- 0.5) x 10^-3", 2
        ),
        not_final=[],
    ),
    "omega": dict(
        modes=RESONANCE_DECAYS["omega"],
        below=below("Gamma6 pi0 e+ e-", 7.7e-4, "(7.7 +- 0.6) x 10^-4", 4),
        not_final=["Gamma4 neutrals (excluding pi0 gamma) (7 +7 -5) x 10^-3"],
    ),
    "phi": dict(
        modes=[
            entry(
                ["K+", "K-"],
                0.501,
                0.006,
                "Gamma1 K+ K-",
                "(50.1 +- 0.6) % S=1.6",
                4,
            ),
            entry(
                ["K0", "K0bar"],
                0.336,
                0.004,
                "Gamma2 K0L K0S",
                "(33.6 +- 0.4) % S=1.3",
                4,
            ),
            entry(
                ["pi+", "pi-", "pi0"],
                0.147,
                0.004,
                "Gamma3 rho pi + pi+ pi- pi0",
                "(14.7 +- 0.4) % S=1.5",
                4,
            ),
            entry(
                ["eta", "gamma"],
                0.01306,
                0.00024,
                "Gamma5 eta gamma",
                "(1.306 +- 0.024) % S=1.2",
                4,
            ),
        ],
        below=below("Gamma6 pi0 gamma", 1.33e-3, "(1.33 +- 0.05) x 10^-3", 4),
        not_final=[],
    ),
    "rho0": dict(
        modes=RESONANCE_DECAYS["rho0"],
        below=below("Gamma8 pi0 gamma", 4.7e-4, "(4.7 +- 0.8) x 10^-4", 12),
        not_final=[
            "Gamma7 pi+ pi- gamma (9.9 +- 1.6) x 10^-3: "
            "radiative pi+ pi-, inside Gamma6 (~ 100 %)"
        ],
    ),
    "rho-": dict(
        modes=RESONANCE_DECAYS["rho-"],
        below=below("Gamma3 pi+- gamma", 4.5e-4, "(4.5 +- 0.5) x 10^-4", 12),
        not_final=[],
    ),
    "Kstar-": dict(
        modes=RESONANCE_DECAYS["Kstar-"],
        below=below("Gamma5 K+- gamma", 1.04e-3, "(1.04 +- 0.10) x 10^-3", 9),
        not_final=[],
    ),
    "Kstar0bar": dict(
        modes=RESONANCE_DECAYS["Kstar0bar"],
        below=below("Gamma4 K0 gamma", 2.47e-3, "(2.47 +- 0.21) x 10^-3", 9),
        not_final=[],
    ),
    "Lambda": dict(
        modes=[
            entry(
                ["p", "pi-"], 0.641, 0.005, "Gamma1 p pi-", "(64.1 +- 0.5) %", 3
            ),
            entry(
                ["n", "pi0"], 0.359, 0.005, "Gamma2 n pi0", "(35.9 +- 0.5) %", 3
            ),
        ],
        below=below("Gamma4 p pi- gamma", 8.5e-4, "(8.5 +- 1.4) x 10^-4", 3),
        not_final=[],
    ),
    "Sigma+": dict(
        modes=[
            entry(
                ["p", "pi0"],
                0.5147,
                0.0030,
                "Gamma1 p pi0",
                "(51.47 +- 0.30) %",
                3,
            ),
            entry(
                ["n", "pi+"],
                0.4843,
                0.0030,
                "Gamma2 n pi+",
                "(48.43 +- 0.30) %",
                3,
            ),
        ],
        below=below(
            "Gamma3 p gamma", 1.04e-3, "(1.04 +- 0.06) x 10^-3 S=2.4", 3
        ),
        not_final=[],
    ),
    "Sigma0": dict(
        modes=[
            entry(
                ["Lambda", "gamma"],
                1.0,
                None,
                "Gamma1 Lambda gamma",
                "100 %",
                2,
            ),
            entry(
                ["Lambda", "e+", "e-"],
                5e-3,
                None,
                "Gamma3 Lambda e+ e-",
                "[a] 5 x 10^-3",
                2,
                note="[a] A theoretical value using QED.",
            ),
        ],
        below=None,
        not_final=[],
    ),
    "Sigma-": dict(
        modes=[
            entry(
                ["n", "pi-"],
                0.99848,
                0.00005,
                "Gamma1 n pi-",
                "(99.848 +- 0.005) %",
                3,
            )
        ],
        below=below(
            "Gamma3 n e- nu_e_bar", 1.017e-3, "(1.017 +- 0.034) x 10^-3", 3
        ),
        not_final=[],
    ),
    "Xi0": dict(
        modes=[
            entry(
                ["Lambda", "pi0"],
                0.99517,
                0.00012,
                "Gamma1 Lambda pi0",
                "(99.517 +- 0.012) % S=1.1",
                2,
            )
        ],
        below=below(
            "Gamma4 Sigma0 gamma", 3.33e-3, "(3.33 +- 0.10) x 10^-3", 2
        ),
        not_final=[],
    ),
    "Xi-": dict(
        modes=[
            entry(
                ["Lambda", "pi-"],
                0.99887,
                0.00035,
                "Gamma1 Lambda pi-",
                "(99.887 +- 0.035) %",
                3,
            )
        ],
        below=below(
            "Gamma3 Lambda e- nu_e_bar", 5.63e-4, "(5.63 +- 0.31) x 10^-4", 3
        ),
        not_final=[],
    ),
}
