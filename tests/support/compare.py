"""The criterion of README.md, "Comparison with the original C++ AM3".

The arrays are E dN/dE per ln E (particle number), and flux plots show E^2 dN/dE
(energy). Arrays must have the same shape, except the two neutrino timescales
that AM3 returns with the hadron grid's length (tests/reference/README.md, "Bugs
of AM3 v1.2.0 that am3_mojo fixes"). These two are compared on the common bins.
The optimization getters, which AM3 returns in the grid's units, are compared in
eV (UNITS_IN_AM3). The e+- emission timescales with merged positrons differ by
design and are left out (MERGED_IN_AM3). These three allowances are for a C++
reference. compare(..., cpp=False) compares two outputs of am3_mojo as they are.
"""

import numpy as np

from support.units import DLNE  # AM3's logarithmic grids: E_i = E_0 exp(i DLNE)
from support.units import MEC2_EV

# the criterion: fluxes to RTOL on the bins above FLOOR of the peak
RTOL, FLOOR = 1e-6, 1e-12


def kind(name):
    getter = name.split("/")[-1]
    return (
        "timescale"
        if getter.startswith("get_t_")
        else "grid"
        if getter.startswith(
            ("get_egrid_", "get_powerlaw_injection_parameters_")
        )
        else "flux"
    )


# getters whose array AM3 returns shorter than am3_mojo (module docstring)
SHORTER_IN_AM3 = ("get_t_neutrino_escape", "get_t_neutrino_extra_cooling")
# getters that AM3 returns in the grid's units, m_e c^2 or GeV, and am3_mojo in
# eV (tests/reference/README.md): the factor from AM3's value to am3_mojo's
UNITS_IN_AM3 = {
    "get_optimize_compton_emission_emin": MEC2_EV,
    "get_optimize_compton_target_emax": MEC2_EV,
    "get_optimize_bethe_heitler_incoming_protons_min": 1e9,
    "get_optimize_bethe_heitler_target_photon_max": MEC2_EV,
    "get_optimize_photopion_target_photon_max": MEC2_EV,
}


# Getters that differ by design where the positrons are merged into the
# electrons. AM3 divides the electrons, which are not evolved then. am3_mojo
# divides all pairs (tests/reference/README.md).
MERGED_IN_AM3 = "get_t_pair_emission_from_"


def merged(container, key):
    pre = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
    k = pre + "get_process_merge_positrons_into_electrons"
    return k in container.files and bool(container[k])


def same_in_ev(name, r, t):
    """AM3's value r and am3_mojo's t are the same grid point (to the rounding
    of the unit conversion)."""
    f = UNITS_IN_AM3[name.split("/")[-1]]
    return np.allclose(t, np.asarray(r, float) * f, rtol=1e-15, atol=0.0)


def grid_of(container, key, n):
    """AM3's grids share the bin width, so any n-bin grid gives E up to a
    constant factor."""
    pre = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
    for g in container.files:
        if (
            g.startswith(pre)
            and g[len(pre) :].startswith("get_egrid_")
            and "/" not in g[len(pre) :]
        ):
            E = np.asarray(container[g], float)
            if E.shape == (n,) and np.all(E > 0):
                return E
    return np.exp(DLNE * np.arange(n))


def significant(r, floor, E=None):
    a = np.abs(np.asarray(r, float))
    fin = np.isfinite(a)
    sig = np.zeros(a.shape, bool)
    if not fin.any():
        return sig
    if np.max(a[fin]) > 0:
        sig |= fin & (a > floor * np.max(a[fin]))
    if E is not None:
        e = np.where(fin, a * E, 0.0)
        if np.max(e) > 0:
            sig |= fin & (e > floor * np.max(e))
    return sig


def checked(r, t, floor, E=None):
    return (
        significant(r, floor, E)
        | significant(t, floor, E)
        | ~(np.isfinite(r) & np.isfinite(t))
    )


def relative(r, t):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            np.isfinite(r) & np.isfinite(t), np.abs(t / r - 1.0), np.inf
        )


def two_sided(r, t):
    """max(t/r, r/t) - 1: relative() both ways, because |t/r - 1| stays below 1
    however far t lies below r."""
    return np.maximum(relative(r, t), relative(t, r))


def deviations(r, t, floor, E=None, rate=False):
    """With rate=True (timescales), the checked bins are those where the
    reference's rate 1/r is above `floor` of its largest (a timescale matters
    where it is short)."""
    if r.size <= 1:
        return None
    if rate:
        pos = np.isfinite(r) & (r > 0)
        sig = ~(np.isfinite(r) & np.isfinite(t)) | (
            pos
            & significant(
                np.where(pos, 1.0 / np.where(pos, r, 1.0), 0.0), floor
            )
        )
    else:
        sig = checked(r, t, floor, E)
    return two_sided(r[sig], t[sig]).max(initial=0.0), int(sig.sum())


def compare(R, T, floor, cpp=True):
    out = dict(flux=[], timescale=[], grid=[], shape=[], missing=[])
    # Every reference array must exist in the test. Arrays that only the test
    # has (added processes) are not compared.
    for k in sorted(R.files):
        if k not in T:
            out["missing"].append(k)
            continue
        r, t = R[k], T[k]
        name = k.split("/")[-1]
        if r.shape != t.shape:
            if (
                cpp
                and name in SHORTER_IN_AM3
                and r.ndim == t.ndim == 1
                and r.size < t.size
            ):
                t = t[: r.size]
            else:
                out["shape"].append(f"{k} {r.shape} vs {t.shape}")
                continue
        if cpp and name.startswith(MERGED_IN_AM3) and merged(R, k):
            continue
        if cpp and name in UNITS_IN_AM3:
            if not same_in_ev(k, r, t):
                out["grid"].append((float(np.max(np.abs(t - r))), k))
            continue
        kd = kind(k)
        if kd == "grid" or r.size <= 1:
            if not np.array_equal(r, t, equal_nan=True):
                out["grid"].append((float(np.max(np.abs(t - r))), k))
            continue
        E = grid_of(R, k, r.size) if (kd == "flux" and r.ndim == 1) else None
        out[kd].append(
            (deviations(r, t, floor, E, rate=(kd == "timescale"))[0], k)
        )
    for key in ("flux", "timescale", "grid"):
        out[key].sort(reverse=True)
    return out


def worst(o, key):
    return o[key][0] if o[key] else (0.0, "-")


def mismatches(o):
    return o["missing"] + o["shape"] + [k for _, k in o["grid"]]
