"""Regenerate DATA of tests/support/measured/ky_angular.py: per energy bin, the cross section and the mean CM angle
<cos theta_K> of the kaon of every two-body reaction gamma N -> K Y, from the published differential cross sections.

usage: .venv/bin/python tools/ky_angular_extract.py [workdir]
       > results/tools/ky_angular_extract.txt
Takes the HEPData records (CSV) and the arXiv source of Compton et al. (2017) from local copies if present, else
downloads them into workdir (default: a temporary directory; needs network access). Prints the DATA block, which
`make literature_checks` saves to results/tools/ky_angular_extract.txt.

sigma and <cos theta> integrate dsigma/dcos theta over the measured bins, the gaps and the edges (moments()). Only bins
with data reaching cos theta <= -0.6 and >= 0.6 are kept (A2: -0.7, 0.7; CLAS K0 Lambda: -0.5, 0.6; CLAS K+ Sigma-:
-0.1, 0.6, which keeps its bins above 2.85 GeV, measured forward of cos theta = -0.4 ... -0.1), so that the edges hold a
small part of the cross section. They hold more at high energies: the CLAS K0 Lambda tables reach forward only to
cos theta = 0.7-0.8, and above 3 GeV the CLAS K+ Lambda data only to 0.87-0.89.
"""
import csv, glob, io, os, re, shutil, subprocess, sys, tarfile, tempfile, urllib.request
import numpy as np
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))   # tests/
from support.paths import load_derivation
derivation = load_derivation('kaon_production')  # the s of tests/literature_checks/kaons/README.md

MP, MN = derivation.MASS["p"], derivation.MASS["n"]   # the code's (PDG 2026; kaons/kaon_production.py)
HEPDATA = 'https://www.hepdata.net/download/submission/ins{}/1/csv'
ARXIV_COMPTON = 'https://arxiv.org/e-print/1706.04748'
# reaction: [(label, source, selector, target mass, coverage cuts)]
SOURCES = {
    'K+L': [('SAPHIR (Glander et al. 2004)', 626695, 'K+ LAMBDA', MP, (-0.6, 0.6)),
            ('CLAS (McCracken et al. 2010)', 840934, 'K+ LAMBDA', MP, (-0.6, 0.6))],
    'K+S0': [('SAPHIR (Glander et al. 2004)', 626695, 'K+ SIGMA0', MP, (-0.6, 0.6)),
             ('CLAS (Dey et al. 2010)', 857728, 'K+ SIGMA0', MP, (-0.6, 0.6))],
    'K0S+': [('SAPHIR (Lawall et al. 2005)', 680746, 'K0 SIGMA+', MP, (-0.6, 0.6))],
    'K+S-': [('CLAS (Anefalos Pereira et al. 2010)', 841145, 'K+ SIGMA-', MN, (-0.1, 0.6))],
    'K0L': [('CLAS (Compton et al. 2017)', 'compton', None, MN, (-0.5, 0.6)),
            ('A2 (Akondi et al. 2019)', 1703675, 'K0 LAMBDA', MN, (-0.7, 0.7))],
    'K0S0': [('A2 (Akondi et al. 2019)', 1703675, 'K0 SIGMA0', MN, (-0.7, 0.7))],
}


KNOWLEDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'knowledge')


def local_copy(pattern):
    """A local copy (git-ignored, may be absent), or None."""
    hits = sorted(glob.glob(os.path.join(KNOWLEDGE, pattern)))
    return hits[-1] if hits else None


def fetch(url, path):
    """Download url to path, with curl if available (HEPData refuses the user agents of Python and of browsers without
    JavaScript)."""
    if shutil.which('curl'):
        subprocess.run(['curl', '-sSfL', '-m', '300', '-o', path, url], check=True)
        return
    with urllib.request.urlopen(url, timeout=300) as r, open(path, 'wb') as f:
        f.write(r.read())


def e_from_w(w, m):
    """The photon energy on a nucleon m at rest that gives the invariant mass w (the s of kaons/README.md)."""
    return derivation.threshold(m, w)


def moments(c, y, lo, hi):
    """(sigma, <cos>) of dsigma/dcos theta given in bins [lo, hi] (bin centres c): the bins, the gaps between them
    (linear between the two bin values) and the edges to cos theta = -1 and +1 (the mean of the two nearest bins)."""
    o = np.argsort(c)
    y, lo, hi = (np.asarray(v, float)[o] for v in (y, lo, hi))
    y = np.maximum(y, 0.0)
    lo, hi = np.maximum(lo, -1.0), np.minimum(hi, 1.0)
    sig = np.sum(y * (hi - lo))
    mom = np.sum(y * (hi ** 2 - lo ** 2) / 2)
    segs = [(hi[k], lo[k + 1], y[k], y[k + 1]) for k in range(len(y) - 1)]
    back, front = np.mean(y[:2]), np.mean(y[-2:])
    segs += [(-1.0, lo[0], back, back), (hi[-1], 1.0, front, front)]
    for a, b, ya, yb in segs:
        if b - a > 1e-9:  # y linear from ya at a to yb at b
            sig += (ya + yb) / 2 * (b - a)
            mom += ya * (b * b - a * a) / 2 + (yb - ya) * ((b ** 3 - a ** 3) / 3 - a * (b * b - a * a) / 2) / (b - a)
    return sig, mom / sig


def hepdata_blocks(workdir, ins):
    """[(qualifiers, header, rows)] of every data block of a HEPData record."""
    d = os.path.join(workdir, f'ins{ins}')
    if not os.path.isdir(d):
        tgz = local_copy(os.path.join('HEPData', f'HEPData-ins{ins}-v*_csv.tgz')) or d + '.tgz'
        if not os.path.exists(tgz):
            fetch(HEPDATA.format(ins), tgz)
        with tarfile.open(tgz) as t:
            t.extractall(d, filter="data")
    out = []
    for f in sorted(glob.glob(os.path.join(d, '**', '*.csv'), recursive=True)):
        meta, qual, block = {}, {}, None
        for raw in open(f, encoding='utf-8', errors='replace'):
            line = raw.rstrip('\n')
            if line.startswith('#:'):
                body = line[2:].strip()
                if ',,,' in body:
                    if block is not None:
                        out.append(block)
                        block = None
                    k, v = body.split(',,,', 1)
                    qual[k.strip()] = v.strip()
                else:
                    m = re.match(r'([^:]+):\s*(.*)', body)
                    if m:
                        meta[m.group(1).strip()] = m.group(2).strip()
            elif line.startswith('"') or (line and line[0].isalpha()):
                if block is not None:
                    out.append(block)
                block = (dict(meta, **qual), next(csv.reader(io.StringIO(line))), [])
                qual = {}
            elif line.strip() and block is not None:
                try:
                    block[2].append([float(x) for x in next(csv.reader(io.StringIO(line)))])
                except ValueError:
                    pass
        if block is not None:
            out.append(block)
    return out


def block_energy(meta, m):
    for k, v in meta.items():
        if k.startswith('E(P=1)'):
            nums = [float(x) for x in re.findall(r'\d+\.?\d*(?:[eE][-+]?\d+)?', v)]
            return nums[0] if '(' in v else float(np.mean(nums[:2]))
    w = re.search(r'W\s*=\s*([\d.]+)\s*GeV', meta.get('description', ''))
    if w:
        return e_from_w(float(w.group(1)), m)
    pl = meta.get('keyword plabs', '')
    if pl:
        return float(np.mean([float(x) for x in pl.split(',')]))
    raise ValueError(f'no energy in {meta}')


def hepdata_points(workdir, ins, reaction, m):
    out = []
    for meta, header, rows in hepdata_blocks(workdir, ins):
        if reaction not in meta.get('keyword reactions', '') + meta.get('RE', ''):
            continue
        ycol = 3 if len(header) > 3 and 'LOW' in header[1].upper() else 1
        name = header[ycol].upper()
        if 'SIG' not in name or 'POL' in name or 'COS' not in header[0].upper():
            continue
        B = np.array([r for r in rows if len(r) > ycol])
        if len(B) < 3:
            continue
        y = B[:, ycol] * (2 * np.pi if 'OMEGA' in name else 1.0)
        if ycol == 3:
            lo, hi = B[:, 1], B[:, 2]
        else:
            d = np.min(np.diff(np.sort(B[:, 0])))
            lo, hi = B[:, 0] - d / 2, B[:, 0] + d / 2
        sig, cos = moments(B[:, 0], y, lo, hi)
        out.append((block_energy(meta, m), sig, cos, lo.min(), hi.max()))
    return out


def compton_bins(workdir):
    """[(E, cos theta, dsigma/dcos theta)] per energy bin of the CLAS g10 and g13 tables of gamma n -> K0 Lambda,
    as published (CrosssectionTable.tex of the arXiv source; the unused drafts g10table.tex and g13table.tex drop
    one bin at each column break). The angular bins are 0.1 wide."""
    d = os.path.join(workdir, 'compton17')
    if not os.path.isdir(d):
        tgz = local_copy(os.path.join('papers', 'arXiv-1706.04748v*.tar.gz')) or d + '.tgz'
        if not os.path.exists(tgz):
            fetch(ARXIV_COMPTON, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(d, filter="data")
    pts = {}
    run = None
    for line in open(os.path.join(d, 'CrosssectionTable.tex')):
        m = re.search(r'\\caption\{(G1[03]) differential', line)
        if m:
            run = m.group(1)
        if '&' not in line or 'boldmath' in line:
            continue
        cells = [x.replace('\\hline', '').replace('\\\\', '').strip() for x in line.split('&')]
        try:
            E, c, ds, st, sy = (float(x) for x in cells)
        except ValueError:
            continue
        pts.setdefault((run, E), {}).setdefault(round(c, 3), []).append((ds, np.hypot(st, sy)))
    out = []
    for (run, E), dd in sorted(pts.items(), key=lambda t: t[0][1]):
        c = np.array(sorted(dd))
        y = np.array([np.average([v for v, e in dd[k]], weights=[1 / e ** 2 for v, e in dd[k]]) for k in c])
        out.append((E, c, y))
    return out


def compton_points(workdir):
    """(E, sigma, <cos theta>, lowest and highest measured cos theta) per energy bin of the CLAS tables of
    gamma n -> K0 Lambda (compton_bins)."""
    out = []
    for E, c, y in compton_bins(workdir):
        sig, cos = moments(c, y, c - 0.05, c + 0.05)
        out.append((E, sig, cos, c.min() - 0.05, c.max() + 0.05))
    return out


def data(workdir):
    """{reaction: [(label, [(E, sigma, <cos>), ...])]}, the DATA of tests/support/measured/ky_angular.py."""
    out = {}
    for r, sets in SOURCES.items():
        out[r] = []
        for label, src, sel, m, (cmin, cmax) in sets:
            P = compton_points(workdir) if src == 'compton' else hepdata_points(workdir, src, sel, m)
            out[r].append((label, sorted((E, s, c) for E, s, c, lo, hi in P
                                         if s > 0 and -1 < c < 1 and lo <= cmin and hi >= cmax)))
    return out


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    print('DATA = {')
    for r, sets in data(workdir).items():
        print(f'    {r!r}: [')
        for label, P in sets:
            print(f'        ({label!r}, [')
            row = [f'({E:.4f}, {s:.4f}, {c:+.4f})' for E, s, c in P]
            for i in range(0, len(row), 4):
                print('            ' + ', '.join(row[i:i + 4]) + ',')
            print('        ]),')
        print('    ],')
    print('}')


if __name__ == '__main__':
    main()
