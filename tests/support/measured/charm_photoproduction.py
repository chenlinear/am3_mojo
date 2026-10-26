"""Charm photoproduction: the measured values (and the calculated or digitised curves) behind the charm hadrons of the
multi-pion types M4-M7 (tests/literature_checks/charm/README.md, "Production in photo-hadronic interactions").
tests/literature_checks/charm/charm_production.py (cp.*) derives that page's table from them and checks the
values taken from HEPData records against the local copies in knowledge/HEPData/ (git-ignored).

Units, unless a docstring says otherwise: photon energies E [GeV] in the nucleon rest frame (the page's eps_r);
gamma p centre-of-momentum energies W [GeV]; cross sections [mubarn] per nucleon. Errors are (stat, syst) or (+, -),
and the values not in a HEPData record were read from the papers, as each docstring says. The block between the
"generated" markers is DIGITISED by tools/charm_photoproduction_extract.py.
"""

# --- total charm cross sections
SIGMA_CC = (
    # SLAC Hybrid Facility Photon Collaboration, 20 GeV photons on hydrogen (bubble chamber)
    dict(key='SLAC86', exp='SLAC HFP', ref='Abe et al. 1986, PRD 33, 1 (SLAC-PUB-3722): abstract, Sec. III.D.1',
         hepdata=('ins214735', 'Table 1'), target='p', E=20.0, sigma=0.062, stat=(0.008, 0.008),
         syst=(0.015, 0.010)),
    # Photon Emulsion Collaboration (WA58), 20-70 GeV photons, emulsion, 36 c cbar events
    dict(key='PEC', exp='PEC (WA58)', ref='Adamovich et al. 1987, PLB 187, 437 (CERN-EP/86-69): abstract',
         hepdata=('ins230940', 'Table 1'), target='emulsion', E=45.0, E_range=(20.0, 70.0), sigma=0.230,
         stat=(0.057, 0.057), syst=(0.0, 0.0),
         note='abstract: (230 +- 57) nb/nucleon; the HEPData record has 250 +- 57; Goncalves et al. 2009 (Fig. 1) '
              'plot 230 +- 57 (digitised: 229.9 nb, bar 173.4-286.9 nb)'),
    # NA14/2, CERN SPS, 40-160 GeV photons, silicon active target
    dict(key='NA14/2', exp='NA14/2', ref='Alvarez et al. 1993, Z. Phys. C 60, 53 (CERN-PPE/92-28)',
         hepdata=('ins333271', 'Table 3'), target='Si', E=100.0, E_range=(40.0, 160.0), sigma=0.540,
         stat=(0.060, 0.060), syst=(0.100, 0.100), note='total non-diffractive open charm'),
    # E691 (Tagged Photon Spectrometer), Fermilab, 80-230 GeV photons on beryllium, x_F > 0
    dict(key='E691', exp='E691', ref='Anjos et al. 1989, PRL 62, 513', hepdata=('ins289318', 'Table 9'),
         target='Be', E=145.0, E_range=(80.0, 230.0), sigma=0.58, stat=(0.01, 0.01), syst=(0.06, 0.06),
         note='per nucleon with sigma_A ~ A^0.93 (the E691 choice, FN-555 p. 15); the record\'s second row, 0.50 '
              '+- 0.01 +- 0.05 (labelled alpha = 0.9), is sigma_Be / 9, i.e. alpha = 1'),
    # BFP (Berkeley-Fermilab-Princeton multimuon spectrometer), 209 GeV muons on iron, extrapolated to Q^2 = 0
    dict(key='BFP100', exp='BFP', ref='Clark et al. 1980, PRL 45, 682 (FERMILAB-Pub-80/35-EXP): abstract, p. 7',
         hepdata=None, target='Fe', E=100.0, sigma=0.560, stat=(0.0, 0.0), syst=(0.200, 0.120),
         note='virtual-photon cross section extrapolated to Q^2 = 0; errors "largely systematic"'),
    dict(key='BFP178', exp='BFP', ref='Clark et al. 1980, PRL 45, 682 (FERMILAB-Pub-80/35-EXP): abstract, p. 7',
         hepdata=None, target='Fe', E=178.0, sigma=0.750, stat=(0.0, 0.0), syst=(0.180, 0.130)),
    # ZEUS, HERA (ep, Q^2 < 4 GeV^2), D*+- extrapolated to all of phase space with m_c = 1.5 GeV
    dict(key='ZEUS163', exp='ZEUS', ref='Derrick et al. 1995, PLB 349, 225 (hep-ex/9502002), Table 2',
         hepdata=('ins392566', 'Table 3'), target='p', W=163.0, W_range=(115.0, 205.0), sigma=6.3,
         stat=(2.2, 2.2), syst=(6.3, 3.0)),
    dict(key='ZEUS243', exp='ZEUS', ref='Derrick et al. 1995, PLB 349, 225 (hep-ex/9502002), Table 2',
         hepdata=('ins392566', 'Table 3'), target='p', W=243.0, W_range=(205.0, 275.0), sigma=16.9,
         stat=(5.2, 5.2), syst=(13.9, 8.5)),
    # H1, HERA: tagged (Q^2 < 0.01 GeV^2) and untagged (Q^2 < 4 GeV^2) samples; syst = experimental, pdf = the
    # extrapolation with other parton densities (the paper's third error)
    dict(key='H1tag200', exp='H1', ref='Aid et al. 1996, NPB 472, 32 (hep-ex/9604005), Table 3 (tagged)',
         hepdata=('ins417960', 'Table 3'), target='p', W=200.0, W_range=(159.0, 242.0), sigma=13.2,
         stat=(2.2, 2.2), syst=(2.1, 1.7), pdf=(9.9, 4.8)),
    dict(key='H1untag142', exp='H1', ref='Aid et al. 1996, NPB 472, 32 (hep-ex/9604005), Table 3 (untagged)',
         hepdata=('ins417960', 'Table 4'), target='p', W=142.0, W_range=(95.0, 190.0), sigma=9.3,
         stat=(2.1, 2.1), syst=(1.9, 1.8), pdf=(6.9, 3.2)),
    dict(key='H1untag230', exp='H1', ref='Aid et al. 1996, NPB 472, 32 (hep-ex/9604005), Table 3 (untagged)',
         hepdata=('ins417960', 'Table 4'), target='p', W=230.0, W_range=(190.0, 268.0), sigma=20.6,
         stat=(5.5, 5.5), syst=(4.3, 3.9), pdf=(15.4, 7.2)),
)
"""Total charm photoproduction cross sections sigma(gamma N -> c cbar X) per nucleon [mubarn], one dict per point:
the photon energy E [GeV] (fixed target; E_range: the beam's range) or the gamma p energy W [GeV] (HERA; W_range: the
bin), the value, its statistical and systematic errors as (up, down), for H1 also the extrapolation error of the
parton densities (pdf). The fit of the charm page adds the errors of a point in quadrature, on each side. Left out: the
combined W bins of ZEUS and H1 (the same events as the two bins each), and SLAC HFP 1984 (superseded by the 1986
value, which uses the sensitivity of both experiments: SLAC-PUB-3722, Sec. II.C.3). E687 (preliminary) is in
E687_FN555."""

E687_FN555 = dict(
    ref='Buchholz for the E687 Collaboration, "Photoproduction of D mesons", Fermilab FN-555 (October 1990), '
        'ICHEP 1990 (Singapore); PRELIMINARY',
    dstar=((140.0, (100.0, 150.0), 5.88, 1.16, 1.57), (175.0, (150.0, 200.0), 4.31, 0.78, 1.15),
           (209.0, (200.0, 250.0), 4.15, 0.95, 1.10), (261.0, (250.0, 300.0), 4.29, 1.38, 1.14),
           (292.0, (300.0, 350.0), 6.72, 1.94, 2.64)),
    dstar_per_cc=0.604, a_be=9.01, alpha=0.93, extra_syst=0.05,
)
"""E687 preliminary (FN-555, Table 1): the D*+- photoproduction cross section on beryllium, all x_F [mubarn per Be
nucleus], (central photon energy [GeV], (range), sigma, stat, syst). FN-555 (pp. 15-16) converts it to sigma(c cbar)
per nucleon: sigma(D*+-)/Be / (0.604 x 9.01^0.93), with 0.604 D*+- per c cbar event (LUND string fragmentation, its
Table 4) and a further 5 % systematic error. (Its Tables 2 and 3, other selections of the same events, are left
out.) No refereed E687 total cross section was found (INSPIRE); Frixione et al. 1997 (hep-ph/9702287) show E687's
later preliminary values (Bellini, La Thuile 1994) and call some measurements 'totally incompatible with one
another', E687 against E691 their example."""

# --- associated production
SLAC_ASSOCIATED = (0.71, 0.11, 0.06)
"""sigma(gamma p -> anti-D Lambda_c+ X) / sigma(gamma p -> charm) at 20 GeV, (value, stat, syst): Abe et al. 1986
(PRD 33, 1; SLAC-PUB-3722), abstract and Sec. III.C.1: the excess of anti-D over D decays (75 against 23 seen),
weighted by the inverse detection efficiency, taken as anti-D Lambda_c+ X."""
SLAC_E_GAMMA = 20.0
"""Photon energy of the SLAC HFP experiment [GeV] (a backscattered laser beam)."""

FOCUS_LC_ASYMMETRY = (0.111, 0.018, 0.012)
"""(sigma(Lambda_c+) - sigma(anti-Lambda_c-)) / (sum), corrected, (value, stat, syst): FOCUS (E831), Link et al.
2004, PLB 581, 39 (hep-ex/0311022v1), abstract and Table 1. Photons on BeO (nucleons of both kinds); Lambda_c in
40 < p_L < 200 GeV/c and p_T^2 < 6 GeV^2 (laboratory), i.e. forward of the centre of momentum."""
FOCUS_E_GAMMA = (175.0, 45.0)
"""Photon energy [GeV] of the FOCUS sample: mean and width of the Gaussian that describes it for reconstructed charm
events (hep-ex/0311022, data selection; '<E> ~ 175 GeV' in the description of the experiment)."""

# --- resolved (hadronic) share
FMNR95_W = (30.0, 100.0, 280.0)
FMNR95_POINTLIKE = (1.090, 2.826, 5.858)
FMNR95_HADRONIC = (0.099, 0.673, 2.435)
"""NLO charm cross sections sigma(gamma p -> c cbar X) [mubarn] at W = 30, 100, 280 GeV, m_c = 1.5 GeV, mu = m_c:
Frixione, Mangano, Nason & Ridolfi 1995 (PLB 348, 633; hep-ph/9412348v1), Table 1 (pointlike component, proton set
MRSA) and Table 2 (hadronic = resolved component, photon set GRV-HO, proton set MRSA)."""

# --- E691 distributions
E691_XF_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
E691_DSDXF = ((10.31, 3.38), (10.38, 1.06), (10.69, 0.68), (9.66, 0.56), (7.37, 0.90), (4.36, 0.37), (2.54, 0.41),
              (2.01, 0.74), (0.52, 4.05))
"""E691 (Anjos et al. 1989; HEPData ins289318 Table 11): dsigma/dx_F of D+ + D- + D0 + anti-D0 on Be at <E> = 145
GeV, (value [mubarn per Be nucleus and unit x_F], total error), in the bins E691_XF_EDGES. x_F = p_L/p_L,max in the
centre-of-momentum frame, positive along the photon (record: 'DEF=PL/PLMAX')."""
E691_PT2_EDGES = tuple(0.5 * i for i in range(21))
E691_DSDPT2 = ((5.79, 0.37), (2.75, 0.16), (1.59, 0.12), (0.932, 0.088), (0.733, 0.067), (0.537, 0.066),
               (0.299, 0.051), (0.271, 0.048), (0.174, 0.028), (0.12, 0.024), (0.0774, 0.0152), (0.0794, 0.0248),
               (0.0652, 0.0288), (0.0426, 0.0145), (0.0396, 0.0241), (0.018, 0.00855), (0.00976, 0.00415),
               (0.00484, 0.004), (0.00908, 0.00653), (0.00146, 0.00103))
"""E691 (HEPData ins289318 Table 10): dsigma/dp_T^2 of the same D mesons, (value [mubarn per Be nucleus and GeV^2],
total error), in the bins E691_PT2_EDGES [GeV^2] (the record's unit 'MUB/0.5 GEV**2': these values times 0.5 add up
to 6.77 mubarn, Table 7's x_F > 0 total being 7.06; mean p_T^2 1.16 GeV^2 as in the paper)."""
E691_E_GAMMA = 145.0
"""Mean photon energy of the E691 distributions [GeV]."""

# --- total gamma p cross section
PDG_FIT = dict(M=2.1206, H=0.2720, eta1=0.4473, delta=3.063e-3, P_pp=34.41, R1_gp=0.0139, m_rho=0.77526)
"""Total gamma p cross section [mb]: the COMPAS group's HPR1R2 fit, sigma = delta [H ln^2(s/s_M) + P^pp]
+ R1^gp (s/s_M)^-eta1, s_M = (m_rho + m_p + M)^2 [GeV^2] (m_rho(770) for the photon; fitted to data at
sqrt(s) >= 5 GeV). The parameters were last printed in the PDG 2016 review "Plots of cross sections and related
quantities", Sec. 51 (C. Patrignani et al., Chin. Phys. C 40, 100001), table on p. 9 (M, H, eta1, delta, P of
pbar(p)/p, R1 of gamma/p); m_rho = 775.26 MeV of that edition. The 2022 review (Sec. 53.5) refers to it for the
parametrisation; the 2024 and 2026 reviews have no total-cross-section section, and the PDG 2026 web page
(pdg.lbl.gov/2026/hadronic-xsections) serves the COMPAS plots and data files of 2022 (rpp2022-*), whose fit curve
is PDG_CURVE_2026 below."""

# --- generated by tools/charm_photoproduction_extract.py ---
GMM09_IIM = (  # (W [GeV], sigma(gamma p -> c cbar X) [nb]), IIM
    (4.991, 12.0), (6.284, 68.7), (7.911, 198.3), (10.0, 401.3), (12.589, 655.1), (15.784, 941.4),
    (19.871, 1248.0), (25.119, 1568.0), (31.623, 1892.2), (39.647, 2238.0), (49.913, 2594.1), (62.837, 2966.9),
    (79.107, 3348.0), (100.0, 3778.0), (125.893, 4206.4), (157.839, 4683.3), (198.708, 5179.5), (251.189, 5728.2),
    (316.228, 6335.0), (396.474, 6959.3), (499.131, 7593.9), (628.368, 8342.2), (791.069, 9103.0), (1000.0, 9933.1),
)
GMM09_BCGC = (  # (W [GeV], sigma(gamma p -> c cbar X) [nb]), bCGC
    (6.284, 49.1), (7.911, 153.7), (10.0, 332.6), (12.589, 572.8), (15.784, 862.7), (19.871, 1182.7),
    (25.119, 1536.7), (31.623, 1892.2), (39.647, 2268.2), (49.913, 2664.7), (62.837, 3047.6), (79.107, 3462.3),
    (100.0, 3880.8), (125.893, 4320.8), (157.839, 4778.6), (198.708, 5249.5), (251.189, 5728.2), (316.228, 6208.7),
    (396.474, 6684.6), (499.131, 7196.9), (628.368, 7748.4), (791.069, 8286.4), (1000.0, 8861.8),
)
GMM09_IPSAT = (  # (W [GeV], sigma(gamma p -> c cbar X) [nb]), IP-SAT
    (6.284, 84.6), (7.911, 296.7), (10.0, 664.0), (12.589, 1143.7), (15.784, 1676.8), (19.871, 2253.0),
    (25.119, 2830.7), (31.623, 3439.1), (39.647, 4067.5), (49.913, 4714.9), (62.837, 5428.7), (79.107, 6208.7),
    (100.0, 7006.2), (125.893, 7906.0), (157.839, 8861.8), (198.708, 9933.1), (251.189, 11133.9), (316.228, 12396.4),
    (396.474, 13709.6), (499.131, 15264.2), (628.368, 16881.3), (791.069, 18669.7), (1000.0, 20509.4),
)
PDG_CURVE_2026 = (  # (sqrt(s) [GeV], sigma_tot(gamma p) [mubarn])
    (25.059, 119.60), (50, 128.96), (99.763, 141.44), (250.59, 164.14), (500, 184.39),
    (997.63, 208.57), (2811.7, 250.47), (9976.3, 311.27), (28117, 369.35),
)
# --- end of generated block ---
"""GMM09_IIM, GMM09_BCGC, GMM09_IPSAT: Goncalves, Machado & Meneses 2009 (PRD 80, 034021; arXiv:0905.2067v1), Fig. 1
left, sigma(gamma p -> c cbar X) [nb] against W [GeV] in three dipole models (IIM, bCGC, IP-SAT; m_c = 1.4 GeV,
direct photon only, times (1 - x)^7 towards threshold), from the vector file charm_nucleon.eps of the arXiv source.
The curves are drawn at every 0.1 in log10 W up to W = 1000 GeV, the end of the figure; the digitisation is exact
to the file's resolution (0.67 % in sigma).
PDG_CURVE_2026: the gamma p fit curve of the summary figure the PDG 2026 web page serves
(https://pdg.lbl.gov/2026/hadronic-xsections/rpp2022-totalxsec_ReImAmp.pdf, raster 5834 x 5671 px, 0.33 % per
pixel in sigma), at nine points where no data point covers the line."""
