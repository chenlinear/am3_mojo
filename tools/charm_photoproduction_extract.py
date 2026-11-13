"""Regenerate the digitized curves of
tests/support/measured/charm_photoproduction.py (the block between its
"generated" markers). The curves are the dipole-model charm cross sections of
Goncalves, Machado & Meneses (2009) and the gamma p total cross section of the
PDG/COMPAS fit.

usage: uv run --no-project --with pymupdf --with numpy python -I
       tools/charm_photoproduction_extract.py [pdf]
       > results/tools/charm_photoproduction_extract.txt
pdf: the PDG figure below (default:
knowledge/papers/PDG-2026_rpp2022-totalxsec_ReImAmp.pdf if present, else
downloaded from the PDG into a temporary directory, which needs network
access). The arXiv source of Goncalves et al. is taken from
knowledge/papers/arXiv-0905.2067v1.tar.gz (git-ignored) and nothing else. Prints
the block, laid out by the `ruff format` of .venv. On stderr, it prints the data
points of the dipole-model figure and the pixel runs of the PDG curve.

* GMM09_IIM, GMM09_BCGC, GMM09_IPSAT: Fig. 1 (left, charm) of Goncalves et al.,
  the vector file charm_nucleon.eps of the arXiv source (xmgr). The tick marks
  give x = 435 + 560 log10(W / GeV) and y = 369 + 343 (log10(sigma / nb) - 1)
  in device units (1/0.24 pt). The legend (lines 966-1014 of the file) names
  the curves: solid black IIM, dashed blue ([20 20]) bCGC, dash-dotted red
  ([40 20 12 20]) IP-SAT. One device unit is 0.67 % in sigma, the resolution of
  the digitization.
* PDG_CURVE_2026: the gamma p fit curve of the summary figure that the PDG 2026
  web page serves
  (https://pdg.lbl.gov/2026/hadronic-xsections/rpp2022-totalxsec_ReImAmp.pdf,
  the COMPAS plots of 2022: image xref 27, RGB, 5834 x 5671 px, with its soft
  mask xref 26). A decade spans 673 px in sigma (0.34 % per pixel). The blue
  line between 0.07 and 0.4 mb is the gamma p fit (the gamma d fit above it is
  red). The block keeps nine points from 25 GeV to 28 TeV where no data point
  covers the line. There, the line's run of blue pixels is at most 10 px (the
  line itself is 8-10 px wide).
"""

import os, re, subprocess, sys, tarfile, tempfile, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GMM_TARBALL = os.path.join(
    ROOT, "knowledge", "papers", "arXiv-0905.2067v1.tar.gz"
)
PDG_LOCAL = os.path.join(
    ROOT, "knowledge", "papers", "PDG-2026_rpp2022-totalxsec_ReImAmp.pdf"
)
PDG_URL = (
    "https://pdg.lbl.gov/2026/hadronic-xsections/rpp2022-totalxsec_ReImAmp.pdf"
)
# the data module of the block, whose path gives `ruff format` its settings
MODULE = os.path.join(
    ROOT, "tests", "support", "measured", "charm_photoproduction.py"
)

# Goncalves et al., Fig. 1 left: device coordinates of the axes
X0, XDEC, Y0, YDEC = 435.0, 560.0, 369.0, 343.0
MODELS = {
    ("0 0 0", "[]"): "IIM",
    ("0 0 1", "[20 20]"): "bCGC",
    ("1 0 0", "[40 20 12 20]"): "IP-SAT",
}

# the PDG figure: log10 sqrt(s) -> column, log10 sigma [mb] -> row (centers of
# the grid lines)
COLS = {
    0: 374.5,
    1: 1271.5,
    2: 2169.0,
    3: 3066.0,
    4: 3963.5,
    5: 4860.5,
    6: 5757.5,
}
ROWS = {
    4: 18.5,
    3: 691.5,
    2: 1364.0,
    1: 2037.0,
    0: 2710.0,
    -1: 3383.0,
    -2: 4055.5,
    -3: 4728.5,
    -4: 5401.5,
}
# sqrt(s) = 5 GeV x 10^(0.05 k): 25 GeV ... 28 TeV
SAMPLE_K = (14, 20, 26, 34, 40, 46, 55, 66, 75)
MAX_RUN = 10  # px: a longer run means a data point covers the line


def log(*a):
    print(*a, file=sys.stderr)


# -------------------------------------------------------- Goncalves et al. 2009
def w_of(x):
    return 10 ** ((x - X0) / XDEC)


def sig_of(y):
    return 10 ** (1 + (y - Y0) / YDEC)


def gmm_curves():
    if not os.path.exists(GMM_TARBALL):
        sys.exit(
            "knowledge/papers/arXiv-0905.2067v1.tar.gz is missing "
            "(the source of https://arxiv.org/abs/0905.2067)"
        )
    with tarfile.open(GMM_TARBALL) as t:
        lines = (
            t.extractfile("charm_nucleon.eps")
            .read()
            .decode("latin-1")
            .split("\n")
        )
    colour, dash, width = "0 0 0", "[]", 1
    paths, cur, symbols = [], None, []
    for s in (ln.strip() for ln in lines):
        m = re.match(r"^([\d.]+) ([\d.]+) ([\d.]+) setrgbcolor$", s)
        if m:
            colour = " ".join(str(round(float(v))) for v in m.groups())
            continue
        if s.endswith("setdash"):
            dash = s.split("]")[0] + "]"
            continue
        m = re.match(r"^(\d+) setlinewidth$", s)
        if m:
            width = int(m.group(1))
            continue
        m = re.match(r"^(-?\d+) (-?\d+) (m|l)$", s)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            if m.group(3) == "m":
                if cur and len(cur["pts"]) > 1:
                    paths.append(cur)
                cur = {
                    "colour": colour,
                    "dash": dash,
                    "width": width,
                    "pts": [(x, y)],
                }
            else:
                cur["pts"].append((x, y))
            continue
        if s == "s" and cur:
            if len(cur["pts"]) > 1:
                paths.append(cur)
            cur = None
        m = re.match(r"^(\d+) (\d+) (\d+) (osq|oa)$", s)
        if m:
            symbols.append((m.group(4), int(m.group(2)), int(m.group(3))))
    curves = {}
    # the model curves: width 5, more than 10 points, above the bottom of the
    # frame
    for p in paths:
        name = MODELS.get((p["colour"], p["dash"]))
        if p["width"] == 5 and name and len(p["pts"]) > 10:
            if name in curves:
                raise ValueError(f"two curves {name}")
            curves[name] = [
                (round(w_of(x), 3), round(sig_of(y), 1))
                for x, y in p["pts"]
                if y > Y0 + 1
            ]
    bars = [
        (
            p["pts"][0][0],
            min(p["pts"][0][1], p["pts"][1][1]),
            max(p["pts"][0][1], p["pts"][1][1]),
        )
        for p in paths
        if p["width"] == 3
        and len(p["pts"]) == 2
        and p["pts"][0][0] == p["pts"][1][0]
    ]
    for kind, x, y in symbols:
        ebar = [
            (round(sig_of(lo), 1), round(sig_of(hi), 1))
            for bx, lo, hi in bars
            if bx == x and lo < y < hi
        ]
        log(
            f"Goncalves et al. Fig. 1, data point ({kind}): W = {w_of(x):.2f} "
            f"GeV, sigma = {sig_of(y):.1f} nb, error bars {ebar}"
        )
    return {name: curves[name] for name in ("IIM", "bCGC", "IP-SAT")}


# -------------------------------------------------------- the PDG figure
def pdg_pdf():
    if len(sys.argv) > 1:
        return sys.argv[1]
    if os.path.exists(PDG_LOCAL):
        return PDG_LOCAL
    path = os.path.join(tempfile.mkdtemp(), os.path.basename(PDG_URL))
    with (
        urllib.request.urlopen(PDG_URL, timeout=300) as r,
        open(path, "wb") as f,
    ):
        f.write(r.read())
    return path


def pdg_curve():
    import numpy as np
    import pymupdf

    doc = pymupdf.open(pdg_pdf())
    rgb, mask = pymupdf.Pixmap(doc, 27), pymupdf.Pixmap(doc, 26)
    a = (
        np.frombuffer(mask.samples, np.uint8)
        .reshape(mask.height, mask.width)
        .astype(float)
        / 255.0
    )
    c = (
        np.frombuffer(rgb.samples, np.uint8)
        .reshape(rgb.height, rgb.width, rgb.n)
        .astype(float)
    )
    im = c * a[:, :, None] + 255.0 * (1.0 - a[:, :, None])
    blue = (im[:, :, 2] > 180) & (im[:, :, 0] < 90) & (im[:, :, 1] < 90)
    pcol = np.polyfit(sorted(COLS), [COLS[k] for k in sorted(COLS)], 1)
    prow = np.polyfit(sorted(ROWS), [ROWS[k] for k in sorted(ROWS)], 1)
    r_top, r_bot = (
        int(np.polyval(prow, np.log10(0.4))),
        int(np.polyval(prow, np.log10(0.07))),
    )
    out = []
    for k in range(0, 107):
        lg = np.log10(5.0) + 0.05 * k
        col = int(round(np.polyval(pcol, lg)))
        rows = np.nonzero(blue[r_top:r_bot, col])[0] + r_top
        if rows.size == 0:
            continue
        runs = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
        run = max(runs, key=len)  # the line's own run of pixels
        rs, sig = 10**lg, 10 ** ((run.mean() - prow[1]) / prow[0]) * 1e3
        log(
            f"PDG gamma p curve: sqrt(s) = {rs:9.5g} GeV, sigma = {sig:7.2f} "
            f"mubarn ({len(run)} px)" + ("  kept" if k in SAMPLE_K else "")
        )
        if k in SAMPLE_K:
            if len(run) > MAX_RUN:
                raise ValueError(
                    f"a data point covers the line at sqrt(s) = {rs:.5g} GeV"
                )
            out.append((rs, sig))
    if len(out) != len(SAMPLE_K):
        raise ValueError("the line is missing at a sample point")
    return out


def rows_of(cells):
    """The trailing comma keeps `ruff format` from joining the lines."""
    return ["    " + c + "," for c in cells]


def ruff_format(text):
    """The ruff is that of .venv, because this script runs in a uv environment
    of its own."""
    ruff = os.path.join(ROOT, ".venv", "bin", "ruff")
    return subprocess.run(
        [ruff, "format", "--stdin-filename", MODULE, "-"],
        input=text,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout


def main():
    curves = gmm_curves()
    pdg = pdg_curve()
    out = ["# --- generated by tools/charm_photoproduction_extract.py ---"]
    for name, label in (
        ("IIM", "GMM09_IIM"),
        ("bCGC", "GMM09_BCGC"),
        ("IP-SAT", "GMM09_IPSAT"),
    ):
        out.append(
            f"{label} = (  # (W [GeV], sigma(gamma p -> c cbar X) [nb]), {name}"
        )
        out += rows_of([f"({w!r}, {s!r})" for w, s in curves[name]]) + [")"]
    out.append(
        "PDG_CURVE_2026 = (  # (sqrt(s) [GeV], sigma_tot(gamma p) [mubarn])"
    )
    out += rows_of([f"({rs:.5g}, {sig:.2f})" for rs, sig in pdg]) + [")"]
    out.append("# --- end of generated block ---")
    print(ruff_format("\n".join(out) + "\n"), end="")


if __name__ == "__main__":
    main()
