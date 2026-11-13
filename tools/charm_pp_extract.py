"""Regenerate the data block of tests/support/measured/charm_pp.py. The block
holds the published NLO charm spectra of pp collisions, digitized from the
vector paths of the arXiv source figures. It also holds the tables that go with
them, read from the LaTeX sources.

usage: .venv/bin/python tools/charm_pp_extract.py
       > results/tools/charm_pp_extract.txt
Takes the arXiv source tarballs from knowledge/papers/arXiv-<id>v<N>.tar.gz
(git-ignored, the latest version present) and nothing else. If a tarball is
missing, the script stops and names it. Prints the data block (Python literals,
between the "generated" markers of the data module, laid out by `ruff format`)
and, on stderr, the calibration and identification checks.

Figures (vector PDFs inside the tarballs, read without rendering):
  * Bhattacharya et al. 2015 (arXiv:1502.01076) Fig. 2, dsigdx1.pdf:
    dsigma/dx_c [pb] of the charm quark against x_c = E_c/E_p at E_p = 1e3, 1e6
    and 1e9 GeV (legends "10^3 GeV", "10^6 GeV" and "10^9 GeV").
  * Bhattacharya et al. 2016 (arXiv:1607.00193) Fig. 12, dsdxf-pert-xx3.pdf:
    charm-quark dsigma/dx_F [mub] at E_p = 1e6 and 1e9 GeV, legends
    "nCTEQ15-Proton" and "BERSS". The figure uses Mathematica subset fonts
    without character maps. So the glyph codes of every label are matched
    against the printed strings (one consistent glyph -> character map per
    font, checked).
  * Carpio et al. 2020 (arXiv:2007.07945) Fig. 7, dsdx_D0_ErrorBars.pdf: x_E
    dsigma/dx_E [mub] of pp -> D0 X at E_p = 1e11 GeV, legend "NLO QCD".
Tables (LaTeX sources): Bhattacharya et al. 2016 Table 1, Carpio et al. Table I,
Lisovyi et al. 2016 (arXiv:1509.01061) Table 12 (file Tables/average_tab2.txt,
"Constrained S" column) and Kniehl & Kramer 2006 (arXiv:hep-ph/0607306) Table I
(LO, c -> hadron). Of the last three, the block keeps the rows of D+, D0, Ds+
and Lambda_c+. Of Bhattacharya et al.'s Table 1, it keeps every row (one per
proton energy).

The residuals of the axis calibration are printed (all below 0.02 pt). Vertices
on the upper or lower frame edge are the plotting program's clipping and are
dropped. The coordinates in the files have three decimals (1e-3 pt). The curves
are polylines through the plotted points, so the digitization adds no error
beyond that of the plotted curves themselves.
"""

import glob, math, os, re, subprocess, sys, tarfile, zlib

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KNOWLEDGE = os.path.join(ROOT, "knowledge", "papers")
# the data module of the block, whose path gives `ruff format` its settings
MODULE = os.path.join(ROOT, "tests", "support", "measured", "charm_pp.py")
HADRONS = ("D+", "D0", "Ds+", "Lc+")


def log(*a):
    print(*a, file=sys.stderr)


def tarball(arxiv_id):
    hits = sorted(
        glob.glob(
            os.path.join(
                KNOWLEDGE, f"arXiv-{arxiv_id.replace('/', '')}v*.tar.gz"
            )
        ),
        key=lambda p: int(re.search(r"v(\d+)\.tar\.gz$", p).group(1)),
    )
    if not hits:
        sys.exit(
            f"knowledge/papers/arXiv-{arxiv_id.replace('/', '')}v<N>.tar.gz is "
            f"missing (the source of https://arxiv.org/abs/{arxiv_id})"
        )
    return hits[-1]


def member(arxiv_id, name):
    path = tarball(arxiv_id)
    with tarfile.open(path) as t:
        return t.extractfile(name).read(), os.path.basename(path)


# ============================================================ PDF reading
class Ref(tuple):
    pass


class Name(str):
    pass


class Op(str):
    pass


_WS = b" \t\r\n\x0c\x00"
_DELIM = b"()<>[]{}/%"
_ESC = {
    b"n": 10,
    b"r": 13,
    b"t": 9,
    b"b": 8,
    b"f": 12,
    b"(": 40,
    b")": 41,
    b"\\": 92,
}


def _lex(buf, i=0):
    n = len(buf)
    while i < n:
        c = buf[i : i + 1]
        if c in _WS:
            i += 1
        elif c == b"%":
            while i < n and buf[i : i + 1] not in (b"\r", b"\n"):
                i += 1
        elif c == b"/":
            j = i + 1
            while (
                j < n
                and buf[j : j + 1] not in _WS
                and buf[j : j + 1] not in _DELIM
            ):
                j += 1
            raw = re.sub(
                rb"#([0-9A-Fa-f]{2})",
                lambda m: bytes([int(m.group(1), 16)]),
                buf[i + 1 : j],
            )
            yield Name(raw.decode("latin1")), j
            i = j
        elif c == b"(":
            depth, j, out = 1, i + 1, bytearray()
            while j < n:
                ch = buf[j]
                if ch == 0x5C:
                    e = buf[j + 1 : j + 2]
                    j += 1
                    if e in _ESC:
                        out.append(_ESC[e])
                        j += 1
                    elif e in (b"\r", b"\n"):
                        j += 2 if buf[j : j + 2] == b"\r\n" else 1
                    elif e.isdigit():
                        k = j
                        while (
                            k < j + 3
                            and buf[k : k + 1] in b"01234567"
                            and buf[k : k + 1]
                        ):
                            k += 1
                        out.append(int(buf[j:k], 8) & 255)
                        j = k
                    else:
                        out += e
                        j += 1
                    continue
                if ch == 0x28:
                    depth += 1
                elif ch == 0x29:
                    depth -= 1
                    if depth == 0:
                        break
                out.append(ch)
                j += 1
            yield bytes(out), j + 1
            i = j + 1
        elif c == b"<":
            if buf[i + 1 : i + 2] == b"<":
                yield "<<", i + 2
                i += 2
            else:
                j = buf.index(b">", i)
                h = re.sub(rb"\s", b"", buf[i + 1 : j])
                yield bytes.fromhex((h + b"0" * (len(h) % 2)).decode()), j + 1
                i = j + 1
        elif c == b">" and buf[i + 1 : i + 2] == b">":
            yield ">>", i + 2
            i += 2
        elif c in (b"[", b"]", b"{", b"}"):
            yield c.decode(), i + 1
            i += 1
        else:
            j = i
            while (
                j < n
                and buf[j : j + 1] not in _WS
                and buf[j : j + 1] not in _DELIM
            ):
                j += 1
            j = max(j, i + 1)
            tok = buf[i:j]
            try:
                v = int(tok)
            except ValueError:
                try:
                    v = float(tok)
                except ValueError:
                    v = Op(tok.decode("latin1"))
            yield v, j
            i = j


def _parse(tokens, t):
    if t in ("[", "<<"):
        end = "]" if t == "[" else ">>"
        items = []
        for u, _ in tokens:
            if u == end:
                break
            items.append(_parse(tokens, u))
            if (
                isinstance(items[-1], Op)
                and items[-1] == "R"
                and len(items) >= 3
            ):
                items[-3:] = [Ref((items[-3], items[-2]))]
        return (
            items
            if t == "["
            else {items[k]: items[k + 1] for k in range(0, len(items) - 1, 2)}
        )
    return t


class PDF:
    """The indirect objects of a PDF without object streams (the output of
    cairo, Mathematica and matplotlib used here)."""

    def __init__(self, data):
        self.data = data
        self.offset = {
            int(m.group(1)): m.end()
            for m in re.finditer(rb"(?<![0-9])(\d+)\s+\d+\s+obj\b", data)
        }

    def get(self, num):
        it = _lex(self.data, self.offset[num])
        tok, end = next(it)
        obj = _parse(it, tok)
        stream = None
        if isinstance(obj, dict):
            depth, pos = 0, self.offset[num]
            for tok, pos in _lex(self.data, pos):
                depth += (tok == "<<") - (tok == ">>")
                if depth == 0:
                    break
            m = re.compile(rb"\s*stream\r?\n").match(self.data, pos)
            if m:
                length = self.resolve(obj.get("Length"))
                raw = (
                    self.data[m.end() : m.end() + length]
                    if isinstance(length, int)
                    else self.data[
                        m.end() : self.data.index(b"endstream", m.end())
                    ]
                )
                filt = self.resolve(obj.get("Filter"))
                if filt in ("FlateDecode", ["FlateDecode"]):
                    raw = zlib.decompress(raw)
                elif filt:
                    raise ValueError(f"unsupported filter {filt}")
                stream = raw
        return obj, stream

    def resolve(self, v):
        while isinstance(v, Ref):
            v = self.get(v[0])[0]
        return v

    def stream(self, ref):
        return self.get(ref[0])[1]

    def first_page(self):
        for num in sorted(self.offset):
            o = self.get(num)[0]
            if isinstance(o, dict) and o.get("Type") == "Page":
                return o
        raise ValueError("no page")


_GLYPHS = {
    "space": " ",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "period": ".",
    "comma": ",",
    "hyphen": "-",
    "minus": "-",
    "plus": "+",
    "equal": "=",
    "slash": "/",
    "bracketleft": "[",
    "bracketright": "]",
    "parenleft": "(",
    "parenright": ")",
    "mu": "μ",
    "sigma": "σ",
    "arrowright": "→",
}


def _tounicode(cmap):
    out = {}
    for blk in re.findall(rb"beginbfchar(.*?)endbfchar", cmap, re.S):
        for a, b in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            out[bytes.fromhex(a.decode())] = bytes.fromhex(b.decode()).decode(
                "utf-16-be", "replace"
            )
    for blk in re.findall(rb"beginbfrange(.*?)endbfrange", cmap, re.S):
        for a, b, d in re.findall(
            rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<[0-9A-Fa-f]+>|"
            rb"\[[^\]]*\])",
            blk,
        ):
            lo, hi, w = int(a, 16), int(b, 16), len(a) // 2
            if d.startswith(b"["):
                dst = [
                    bytes.fromhex(x.decode()).decode("utf-16-be", "replace")
                    for x in re.findall(rb"<([0-9A-Fa-f]+)>", d)
                ]
            else:
                d0 = bytes.fromhex(d[1:-1].decode())
                dst = [
                    (int.from_bytes(d0, "big") + k)
                    .to_bytes(len(d0), "big")
                    .decode("utf-16-be", "replace")
                    for k in range(hi - lo + 1)
                ]
            for k in range(hi - lo + 1):
                out[(lo + k).to_bytes(w, "big")] = dst[k]
    return out


class Font:
    def __init__(self, pdf, ref, name):
        f = pdf.resolve(ref)
        self.name = name
        self.two_byte = f.get("Subtype") == "Type0"
        self.uni = (
            _tounicode(pdf.stream(f["ToUnicode"])) if "ToUnicode" in f else {}
        )
        self.diff = {}
        enc = pdf.resolve(f.get("Encoding"))
        if isinstance(enc, dict):
            code = 0
            for v in pdf.resolve(enc.get("Differences", [])):
                if isinstance(v, int):
                    code = v
                else:
                    self.diff[code] = str(v)
                    code += 1
        self.scale = (
            pdf.resolve(f["FontMatrix"])[0]
            if f.get("Subtype") == "Type3"
            else 0.001
        )
        self.widths, self.dw = {}, 0
        if self.two_byte:
            d = pdf.resolve(pdf.resolve(f["DescendantFonts"])[0])
            self.dw = d.get("DW", 1000)
            w, k = pdf.resolve(d.get("W", [])), 0
            while k < len(w):
                if isinstance(w[k + 1], list):
                    self.widths.update(
                        {w[k] + j: v for j, v in enumerate(w[k + 1])}
                    )
                    k += 2
                else:
                    self.widths.update(
                        {c: w[k + 2] for c in range(w[k], w[k + 1] + 1)}
                    )
                    k += 3
        else:
            fc = f.get("FirstChar", 0)
            self.widths = {
                fc + j: v
                for j, v in enumerate(pdf.resolve(f.get("Widths", [])))
            }

    def codes(self, s):
        return (
            [s[k : k + 2] for k in range(0, len(s) - 1, 2)]
            if self.two_byte
            else [s[k : k + 1] for k in range(len(s))]
        )

    def char(self, code):
        u = self.uni.get(code)
        if u is not None and u != "\ufffd":
            return u
        c = int.from_bytes(code, "big")
        if c in self.diff:
            g = self.diff[c]
            return _GLYPHS.get(
                g,
                chr(int(g[3:], 16))
                if re.fullmatch(r"uni[0-9A-F]{4}", g)
                else g
                if len(g) == 1
                else None,
            )
        if not self.two_byte and not self.uni and 32 <= c < 127:
            return chr(c)
        return None

    def width(self, code):
        return (
            self.widths.get(int.from_bytes(code, "big"), self.dw) * self.scale
        )


def _mul(a, b):
    return [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]


def _apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def _rgb(args):
    v = [float(a) for a in args if isinstance(a, (int, float))]
    if len(v) == 1:
        return (v[0],) * 3
    if len(v) == 4:
        return tuple((1 - c) * (1 - v[3]) for c in v[:3])
    return tuple(v[:3]) if len(v) == 3 else (0.0, 0.0, 0.0)


def page_items(pdf):
    """(paths, texts) of the first page in PDF points (y up)."""
    pg = pdf.first_page()
    c = pg["Contents"]
    data = b"\n".join(
        pdf.stream(r) for r in (c if isinstance(c, list) else [c])
    )
    paths, texts = [], []
    gs = dict(
        ctm=[1, 0, 0, 1, 0, 0],
        stroke=(0.0, 0.0, 0.0),
        fill=(0.0, 0.0, 0.0),
        dash=(),
    )
    _run(pdf, pdf.resolve(pg.get("Resources", {})), data, gs, paths, texts, 0)
    return paths, texts


def _run(pdf, res, data, gs, paths, texts, depth):
    fonts = {
        k: Font(pdf, v, k) for k, v in pdf.resolve(res.get("Font", {})).items()
    }
    xobj = pdf.resolve(res.get("XObject", {}))
    stack, args, cur, sub = [], [], [], None
    tm = tlm = [1, 0, 0, 1, 0, 0]
    font, fs, tc, tw, th, tl, rise = None, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    it = _lex(data)
    for tok, _ in it:
        if not isinstance(tok, Op):
            args.append(_parse(it, tok) if tok in ("[", "<<") else tok)
            continue
        op, a, args = str(tok), args, []
        if op == "q":
            stack.append(dict(gs))
        elif op == "Q":
            gs = stack.pop() if stack else gs
        elif op == "cm":
            gs["ctm"] = _mul([float(v) for v in a[-6:]], gs["ctm"])
        elif op == "d":
            gs["dash"] = (
                tuple(float(v) for v in a[0])
                if a and isinstance(a[0], list)
                else ()
            )
        elif op in ("RG", "G", "K", "SC", "SCN"):
            gs["stroke"] = _rgb(a)
        elif op in ("rg", "g", "k", "sc", "scn"):
            gs["fill"] = _rgb(a)
        elif op == "m":
            sub = [_apply(gs["ctm"], float(a[-2]), float(a[-1]))]
            cur.append(sub)
        elif op == "l":
            if sub is None:
                sub = []
                cur.append(sub)
            sub.append(_apply(gs["ctm"], float(a[-2]), float(a[-1])))
        elif op in ("c", "v", "y"):
            v = (
                [float(x) for x in a[-6:]]
                if op == "c"
                else [float(x) for x in a[-4:]]
            )
            p0 = sub[-1]
            pts = [
                _apply(gs["ctm"], v[k], v[k + 1]) for k in range(0, len(v), 2)
            ]
            c1, c2, p3 = (
                pts
                if op == "c"
                else [p0] + pts
                if op == "v"
                else [pts[0], pts[1], pts[1]]
            )
            sub.extend(
                tuple(
                    (1 - t) ** 3 * p0[i]
                    + 3 * (1 - t) ** 2 * t * c1[i]
                    + 3 * (1 - t) * t * t * c2[i]
                    + t**3 * p3[i]
                    for i in range(2)
                )
                for t in (k / 8 for k in range(1, 9))
            )
        elif op == "h":
            if sub:
                sub.append(sub[0])
        elif op == "re":
            x, y, w, h = (float(v) for v in a[-4:])
            sub = [
                _apply(gs["ctm"], *p)
                for p in (
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h),
                    (x, y),
                )
            ]
            cur.append(sub)
        elif op in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"):
            if op != "n" and cur:
                kind = (
                    "S"
                    if op in ("S", "s")
                    else "f"
                    if op in ("f", "F", "f*")
                    else "B"
                )
                paths.append(
                    dict(
                        op=kind,
                        stroke=gs["stroke"],
                        fill=gs["fill"],
                        dash=gs["dash"],
                        subpaths=[s for s in cur if s],
                    )
                )
            cur, sub = [], None
        elif op == "BT":
            tm = tlm = [1, 0, 0, 1, 0, 0]
        elif op == "Tf":
            font, fs = fonts.get(a[-2]), float(a[-1])
        elif op in ("Tc", "Tw", "Tz", "TL", "Ts"):
            v = float(a[-1])
            tc, tw, th, tl, rise = (
                v if op == "Tc" else tc,
                v if op == "Tw" else tw,
                v / 100 if op == "Tz" else th,
                v if op == "TL" else tl,
                v if op == "Ts" else rise,
            )
        elif op == "Tm":
            tm = tlm = [float(v) for v in a[-6:]]
        elif op in ("Td", "TD"):
            if op == "TD":
                tl = -float(a[-1])
            tm = tlm = _mul([1, 0, 0, 1, float(a[-2]), float(a[-1])], tlm)
        elif op == "T*":
            tm = tlm = _mul([1, 0, 0, 1, 0, -tl], tlm)
        elif op in ("Tj", "TJ", "'", '"'):
            if op in ("'", '"'):
                tm = tlm = _mul([1, 0, 0, 1, 0, -tl], tlm)
            if font is None:
                continue
            runs, run = [], []
            for s in a[-1] if op == "TJ" else [a[-1]]:
                # a gap: new run
                if isinstance(s, (int, float)) and abs(s) > 300 and run:
                    runs.append(run)
                    run = []
                run.append(s)
            runs.append(run)
            for run in runs:
                m0, chars, keys = None, [], []
                for s in run:
                    if isinstance(s, (int, float)):
                        tm = _mul([1, 0, 0, 1, -s / 1000 * fs * th, 0], tm)
                        continue
                    for code in font.codes(s):
                        m0 = m0 or _mul(tm, gs["ctm"])
                        chars.append(font.char(code))
                        keys.append((font.name, code))
                        tm = _mul(
                            [
                                1,
                                0,
                                0,
                                1,
                                (
                                    font.width(code) * fs
                                    + tc
                                    + (tw if code == b" " else 0)
                                )
                                * th,
                                0,
                            ],
                            tm,
                        )
                if keys:
                    x0, y0 = _apply(m0, 0, rise)
                    x1, _ = _apply(_mul(tm, gs["ctm"]), 0, rise)
                    texts.append(
                        dict(
                            text=None if None in chars else "".join(chars),
                            keys=keys,
                            x=x0,
                            y=y0,
                            x1=x1,
                            size=fs
                            * math.sqrt(abs(m0[0] * m0[3] - m0[1] * m0[2])),
                            upright=abs(m0[1]) < 1e-9 * abs(m0[0]) + 1e-12,
                        )
                    )
        elif op == "Do":
            o, s = pdf.get(xobj[a[-1]][0])
            if o.get("Subtype") == "Form" and depth < 8:
                g2 = dict(gs)
                if "Matrix" in o:
                    g2["ctm"] = _mul([float(v) for v in o["Matrix"]], g2["ctm"])
                _run(
                    pdf,
                    pdf.resolve(o.get("Resources", res)),
                    s,
                    g2,
                    paths,
                    texts,
                    depth + 1,
                )


# ============================================================ figure geometry
def frame(paths):
    h, v = [], []
    for p in paths:
        if p["op"] == "f" or max(p["stroke"]) > 0.05:
            continue
        for s in p["subpaths"]:
            if len(s) == 2:
                (xa, ya), (xb, yb) = s
                if abs(ya - yb) < 0.01 and abs(xb - xa) > 100:
                    h.append((min(xa, xb), max(xa, xb), ya))
                if abs(xa - xb) < 0.01 and abs(yb - ya) > 100:
                    v.append((min(ya, yb), max(ya, yb), xa))
    return (
        min(t[2] for t in v),
        min(t[2] for t in h),
        max(t[2] for t in v),
        max(t[2] for t in h),
    )


def ticks(paths, fr, axis):
    x0, y0, x1, y1 = fr
    out = []
    for p in paths:
        if p["op"] == "f" or max(p["stroke"]) > 0.05:
            continue
        for s in p["subpaths"]:
            if len(s) != 2:
                continue
            (xa, ya), (xb, yb) = s
            if (
                axis == "x"
                and abs(xa - xb) < 0.01
                and x0 - 0.01 <= xa <= x1 + 0.01
                and abs(min(ya, yb) - y0) < 0.05
                and 0.5 < abs(yb - ya) < 0.1 * (y1 - y0)
            ):
                out.append((xa, abs(yb - ya)))
            if (
                axis == "y"
                and abs(ya - yb) < 0.01
                and y0 - 0.01 <= ya <= y1 + 0.01
                and abs(min(xa, xb) - x0) < 0.05
                and 0.5 < abs(xb - xa) < 0.1 * (x1 - x0)
            ):
                out.append((ya, abs(xb - xa)))
    big = max(t[1] for t in out)
    return sorted({round(t[0], 4) for t in out if t[1] > 0.8 * big})


def lines(texts, sel):
    groups = []
    for t in sorted(
        (t for t in texts if t["upright"] and sel(t)), key=lambda t: t["x"]
    ):
        best = None
        for g in groups:
            main = max(g, key=lambda u: u["size"])
            gap, dy = t["x"] - g[-1]["x1"], abs(t["y"] - main["y"])
            same = (
                dy < 0.1 * main["size"]
                if t["size"] > 0.9 * main["size"]
                else dy < 0.8 * main["size"]
            )
            if (
                same
                and -0.15 * main["size"] < gap < 0.5 * main["size"]
                and (best is None or abs(gap) + dy < best[0])
            ):
                best = (abs(gap) + dy, g)
        if best:
            best[1].append(t)
        else:
            groups.append([t])
    out = []
    for g in groups:
        main = max(g, key=lambda u: u["size"])
        text, keys, sup = "", [], False
        for t in g:
            raised = (
                t["size"] < 0.9 * main["size"]
                and t["y"] > main["y"] + 0.15 * main["size"]
            )
            if raised and not sup:
                text = None if text is None else text + "^"
                keys.append("^")
            sup = raised
            text = (
                None if text is None or t["text"] is None else text + t["text"]
            )
            keys += t["keys"]
        out.append(
            dict(
                text=text,
                keys=keys,
                xc=(g[0]["x"] + g[-1]["x1"]) / 2,
                y=main["y"],
                size=main["size"],
                x0=g[0]["x"],
                x1=g[-1]["x1"],
            )
        )
    return out


def decode(labels, printed, key):
    """labels and printed are in the same order, and key collects the
    glyph -> character map."""
    for lab, s in zip(labels, printed):
        if lab["text"] is not None:
            if lab["text"].replace(" ", "") != s.replace(" ", ""):
                raise ValueError(f"label {lab['text']!r} printed as {s!r}")
            continue
        glyphs = [k for k in lab["keys"] if k != "^"]
        chars = [c for c in s if c != "^"]
        if len(glyphs) != len(chars) or [k == "^" for k in lab["keys"]].count(
            True
        ) != s.count("^"):
            raise ValueError(f"{len(glyphs)} glyphs for {s!r}")
        for g, c in zip(glyphs, chars):
            if key.setdefault(g, c) != c:
                raise ValueError(f"glyph {g} is both {key[g]!r} and {c!r}")
        lab["text"] = s
    by_font = {}
    for (f, code), c in key.items():
        by_font.setdefault((f, c), set()).add(code)
    clash = {k: v for k, v in by_font.items() if len(v) > 1}
    if clash:
        raise ValueError(f"one character, several glyphs: {clash}")


def value(label):
    s = label.replace("−", "-").replace(" ", "")
    if "^" in s:
        m, e = s.split("^")
        return (
            float(m) ** float(e)
            if m == "10"
            else float(m.rstrip("×x*")) * 10 ** float(e)
        )
    return float(s)


def calibrate(paths, texts, fr, axis, scale, printed=None, key=None):
    """(a, b): position = a + b v, v the value (lin) or its log10 (log)."""
    x0, y0, x1, y1 = fr
    tk = ticks(paths, fr, axis)
    if axis == "x":
        labs = [
            l
            for l in lines(texts, lambda t: t["y"] < y0 - 2)
            if l["x1"] > x0 + 1 and l["x0"] < x1 - 1
        ]
        labs = sorted(
            (l for l in labs if l["y"] > max(m["y"] for m in labs) - 3),
            key=lambda l: l["xc"],
        )
    else:
        labs = sorted(
            lines(
                texts,
                lambda t: (
                    t["x1"] < x0 - 1
                    and y0 - 0.6 * t["size"] < t["y"] < y1 + 0.8 * t["size"]
                ),
            ),
            key=lambda l: l["y"],
        )
        labs = [
            l for l in labs if abs(l["x1"] - max(m["x1"] for m in labs)) < 3
        ]
    if printed is not None:
        decode(labs, printed, key)
    pts = []
    for l in labs:
        pos = l["xc"] if axis == "x" else l["y"] + 0.33 * l["size"]
        t = min(tk, key=lambda u: abs(u - pos))
        if abs(t - pos) > 0.4 * min(b - a for a, b in zip(tk, tk[1:])):
            raise ValueError(f"label {l['text']} at {pos:.2f} has no tick")
        v = value(l["text"])
        pts.append(
            (math.log10(v) if scale == "log" else v, t, pos - t, l["text"])
        )
    n = len(pts)
    sv, sp = sum(p[0] for p in pts), sum(p[1] for p in pts)
    svv, svp = sum(p[0] ** 2 for p in pts), sum(p[0] * p[1] for p in pts)
    b = (n * svp - sv * sp) / (n * svv - sv * sv)
    a = (sp - b * sv) / n
    res = max(abs(a + b * p[0] - p[1]) for p in pts)
    log(
        f"    {axis} axis ({scale}): {n} labels {[p[3] for p in pts]} on major "
        f"ticks; {b:.4f} pt per unit; max residual {res:.4f} pt; label offsets "
        f"{min(p[2] for p in pts):+.2f}..{max(p[2] for p in pts):+.2f} pt"
    )
    if res > 0.05:
        raise ValueError("axis calibration residual above 0.05 pt")
    return a, b


def style(p):
    return (
        tuple(round(c, 3) for c in p["stroke"]),
        tuple(round(d, 3) for d in p["dash"]),
    )


def legend_style(paths, texts, label, undecodable=False, key=None):
    labs = lines(texts, lambda t: True)
    if undecodable:
        cand = [
            l
            for l in labs
            if l["text"] is None
            and len([k for k in l["keys"] if k != "^"])
            == len(label.replace("^", ""))
        ]
        found = []
        for l in cand:
            trial = dict(key)
            try:
                decode([l], [label], trial)
                found.append((l, trial))
            except ValueError:
                pass
        if len(found) != 1:
            raise ValueError(f"{len(found)} candidates for legend {label!r}")
        lab, k2 = found[0]
        key.update(k2)
    else:
        hits = [
            l
            for l in labs
            if l["text"] is not None
            and l["text"].replace(" ", "") == label.replace(" ", "")
        ]
        if len(hits) != 1:
            raise ValueError(f"{len(hits)} texts {label!r}")
        lab = hits[0]
    best = None
    for p in paths:
        if p["op"] != "S":
            continue
        for s in p["subpaths"]:
            if len(s) == 2 and abs(s[0][1] - s[1][1]) < 0.01:
                xr, yy = max(s[0][0], s[1][0]), s[0][1]
                dx, dy = lab["x0"] - xr, yy - lab["y"]
                if (
                    0 <= dx < 3 * lab["size"]
                    and -0.2 * lab["size"] < dy < 0.9 * lab["size"]
                ):
                    if best is None or dx < best[0]:
                        best = (dx, p, yy)
    if best is None:
        raise ValueError(f"no legend sample for {label!r}")
    _, p, yy = best
    st = style(p)
    sample = [
        q
        for q in paths
        if q["op"] == "S"
        and style(q) == st
        and all(
            abs(pt[1] - yy) < 0.3
            and lab["x0"] - 6 * lab["size"] < pt[0] <= lab["x0"]
            for s in q["subpaths"]
            for pt in s
        )
    ]
    return st, sample, lab


def curve(paths, fr, st, exclude):
    x0, y0, x1, y1 = fr
    groups = []
    for p in paths:
        if p["op"] != "S" or style(p) != st or any(p is q for q in exclude):
            continue
        pts = [
            pt
            for s in p["subpaths"]
            for pt in s
            if x0 - 0.01 <= pt[0] <= x1 + 0.01 and y0 + 0.01 < pt[1] < y1 - 0.01
        ]
        if pts:
            groups.append(pts)
    return groups


def to_data(pts, cx, cy, xscale, yscale):
    out = {}
    for px, py in pts:
        u, v = (px - cx[0]) / cx[1], (py - cy[0]) / cy[1]
        x = 10**u if xscale == "log" else u
        y = 10**v if yscale == "log" else v
        out[round(x, 7)] = y
    return sorted(out.items())


# ============================================================ the figures
def berss15_fig2():
    data, src = member("1502.01076", "dsigdx1.pdf")
    log(f"Bhattacharya et al. 2015 Fig. 2 (dsigdx1.pdf in {src})")
    paths, texts = page_items(PDF(data))
    fr = frame(paths)
    cx = calibrate(paths, texts, fr, "x", "lin")
    cy = calibrate(paths, texts, fr, "y", "log")
    out = {}
    for E, lab in ((1e3, "10^3 GeV"), (1e6, "10^6 GeV"), (1e9, "10^9 GeV")):
        st, sample, _ = legend_style(paths, texts, lab)
        g = curve(paths, fr, st, sample)
        out[E] = to_data([p for grp in g for p in grp], cx, cy, "lin", "log")
        log(
            f"    {lab}: style {st}, legend sample {len(sample)} path(s), "
            f"{len(g)} curve paths, {len(out[E])} points, "
            f"x {out[E][0][0]:.4f}..{out[E][-1][0]:.4f}"
        )
    return out


def berss16_fig12():
    data, src = member("1607.00193", "dsdxf-pert-xx3.pdf")
    log(f"Bhattacharya et al. 2016 Fig. 12 (dsdxf-pert-xx3.pdf in {src})")
    paths, texts = page_items(PDF(data))
    fr = frame(paths)
    key = {}
    cx = calibrate(
        paths,
        texts,
        fr,
        "x",
        "lin",
        ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"],
        key,
    )
    cy = calibrate(
        paths,
        texts,
        fr,
        "y",
        "log",
        ["1", "10", "100", "1000", "10^4", "10^5", "10^6"],
        key,
    )
    # The curve labels inside the frame, bottom to top as printed:
    # E_p = 10^6 GeV, E_p = 10^9 GeV (subscript p). The decoding checks that the
    # lower exponent is the glyph '6' of the axis label 10^6.
    inner = sorted(
        (
            l
            for l in lines(
                texts,
                lambda t: fr[0] < t["x"] < fr[2] and fr[1] < t["y"] < fr[3],
            )
            if len(l["keys"]) > 6 and "^" in l["keys"]
        ),
        key=lambda l: l["y"],
    )
    printed = ["Ep=10^6GeV", "Ep=10^9GeV"]
    decode(inner, printed, key)
    energies = [1e6, 1e9]
    log(
        f"    curve labels (bottom to top): {printed}, glyphs consistent with "
        "the axis labels"
    )
    out = {}
    for name in ("nCTEQ15-Proton", "BERSS"):
        st, sample, _ = legend_style(
            paths, texts, name, undecodable=True, key=key
        )
        g = curve(paths, fr, st, sample)
        if len(g) != 2:
            raise ValueError(f"{len(g)} paths for {name}")
        # one path per energy, and the 1e9 GeV curve is above the 1e6 GeV one
        g.sort(key=lambda pts: max(p[1] for p in pts))
        for E, pts in zip(energies, g):
            out[(name, E)] = to_data(pts, cx, cy, "lin", "log")
            log(
                f"    {name} E_p = {E:.0e}: style {st}, "
                f"{len(out[(name, E)])} points, "
                f"x {out[(name, E)][0][0]:.4f}..{out[(name, E)][-1][0]:.4f}"
            )
    log(
        f"    glyph map (font, code -> char): {len(key)} glyphs, consistent "
        "over all labels"
    )
    return out


def carpio20_fig7():
    data, src = member("2007.07945", "dsdx_D0_ErrorBars.pdf")
    log(f"Carpio et al. 2020 Fig. 7 (dsdx_D0_ErrorBars.pdf in {src})")
    paths, texts = page_items(PDF(data))
    fr = frame(paths)
    cx = calibrate(paths, texts, fr, "x", "log")
    cy = calibrate(paths, texts, fr, "y", "log")
    log(
        "    title: "
        f"{[l['text'] for l in lines(texts, lambda t: t['y'] > fr[3])]}"
    )
    st, sample, _ = legend_style(paths, texts, "NLO QCD")
    g = curve(paths, fr, st, sample)
    pts = to_data([p for grp in g for p in grp], cx, cy, "log", "log")
    log(
        f"    NLO QCD: style {st}, {len(g)} curve paths, {len(pts)} points, "
        f"x {pts[0][0]:.3g}..{pts[-1][0]:.3g}"
    )
    return pts


# ============================================================ tables (LaTeX)
def _tex(arxiv_id, name):
    data, src = member(arxiv_id, name)
    return re.sub(r"(?<!\\)%.*", "", data.decode("utf-8", "replace")), src


def _num(s):
    s = s.replace("$", "").replace("{", "").replace("}", "").strip()
    m = re.fullmatch(r"([\d.]+)\s*\\times\s*10\^(-?\d+)", s)
    return float(f"{m.group(1)}e{m.group(2)}") if m else float(s)


def berss16_table1():
    t, src = _tex("1607.00193", "PromptNeutrinoFlux_Revision.tex")
    tab = re.search(
        r"\\begin\{table\}.*?\\label\{table:sigccb2\}", t, re.S
    ).group(0)
    out = {}
    for row in re.findall(r"\n\s*\$10\^\{?(\d+)\}?\$\s*&(.*?)\\\\", tab):
        out[10.0 ** int(row[0])] = tuple(_num(c) for c in row[1].split("&"))
    log(f"Bhattacharya et al. 2016 Table 1 ({src}): {len(out)} rows")
    return out


def carpio20_table1():
    t, src = _tex("2007.07945", "main.tex")
    tab = re.search(
        r"Particle & \$N\$ & \$\\epsilon\$(.*?)\\label\{Table1\}", t, re.S
    ).group(1)
    names = {"D^0": "D0", "D^+": "D+", "D_s^+": "Ds+", "\\Lambda_c^+": "Lc+"}
    out = {
        names[n.strip()]: (float(N), float(e))
        for n, N, e in re.findall(
            r"\$([^$]+)\$\s*&\s*([\d.]+)\s*&\s*([\d.]+)", tab
        )
    }
    log(f"Carpio et al. Table I ({src}): {out}")
    return {h: out[h] for h in HADRONS}


def kk06_table1():
    t, src = _tex("hep-ph/0607306", "add.tex")
    tab = re.search(r"\\label\{tab:par\}(.*?)\\end\{tabular\}", t, re.S).group(
        1
    )
    names = {
        "$D^0$": "D0",
        "$D^+$": "D+",
        "$D^{*+}$": "D*+",
        "$D_s^+$": "Ds+",
        "$\\Lambda_c^+$": "Lc+",
    }
    out, cur = {}, None
    for line in tab.split("\\\\"):
        cells = [c.strip() for c in line.replace("\\hline", "").split("&")]
        if len(cells) != 7:
            continue
        cur = names.get(cells[0], cur) if cells[0] else cur
        if cells[1] == "LO" and cells[2] == "$c$":
            out[cur] = (float(cells[3]), float(cells[6]))
    log(f"Kniehl & Kramer 2006 Table I, LO, c ({src}): {out}")
    return {h: out[h] for h in HADRONS}


def lisovyi_table12():
    data, src = member("1509.01061", "Tables/average_tab2.txt")
    names = {
        "D^0": "D0",
        "D^+": "D+",
        "D^{+}_{s}": "Ds+",
        "\\Lambda_c^{+}": "Lc+",
        "D^{*+}": "D*+",
        "D^{*0}": "D*0",
        "D^{*+}_{s}": "Ds*+",
    }
    out = {}
    for name, f, df in re.findall(
        r"\$f\(c\\rightarrow\{\}([^)]*)\)\$&\$([\d.]+)\\pm([\d.]+)\$",
        data.decode(),
    ):
        out[names[name]] = (float(f), float(df))
    log(f"Lisovyi et al. Tables/average_tab2.txt ({src}): {out}")
    return {h: out[h] for h in HADRONS}


# ============================================================ output
def fmt_pts(pts, xf, yf, indent="    "):
    return fmt_items([f"({x:{xf}}, {y:{yf}})" for x, y in pts], indent)


def fmt_items(items, indent):
    """The trailing comma keeps `ruff format` from joining the lines."""
    return [indent + item + "," for item in items]


def head(assign, note):
    line = f"{assign}  # {note}"
    return [line] if len(line) <= 80 else [f"# {note}", assign]


def fmt_dict(name, d, note):
    return (
        head(f"{name} = {{", note)
        + fmt_items([f"{k!r}: {v!r}" for k, v in d.items()], "    ")
        + ["}"]
    )


def ruff_format(text):
    ruff = os.path.join(os.path.dirname(sys.executable), "ruff")
    return subprocess.run(
        [ruff, "format", "--stdin-filename", MODULE, "-"],
        input=text,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout


def main():
    f2, f12, f7 = berss15_fig2(), berss16_fig12(), carpio20_fig7()
    t1, ct, lis, kk = (
        berss16_table1(),
        carpio20_table1(),
        lisovyi_table12(),
        kk06_table1(),
    )
    L = ["# --- generated by tools/charm_pp_extract.py ---"]
    L += head("BERSS15_FIG2 = {", "E_p [GeV]: [(x_c, dsigma/dx_c [pb])]")
    for E, pts in f2.items():
        L.append(f"    {E:.0e}: [")
        L += fmt_pts(pts, ".5f", ".5g", " " * 8) + ["    ],"]
    L.append("}")
    L += head(
        "BERSS16_FIG12 = {", "(curve, E_p [GeV]): [(x_F, dsigma/dx_F [mub])]"
    )
    for (name, E), pts in f12.items():
        L.append(f"    ({name!r}, {E:.0e}): [")
        L += fmt_pts(pts, ".5f", ".5g", " " * 8) + ["    ],"]
    L.append("}")
    L += head(
        "CARPIO20_FIG7_NLO = [",
        "(x_E, x_E dsigma/dx_E [mub]), pp -> D0 X at E_p = 1e11 GeV, NLO QCD",
    )
    L += fmt_pts(f7, ".5g", ".5g") + ["]"]
    L += head(
        "BERSS16_TABLE1 = {",
        "E_p [GeV]: (pp m_T, pp m_c, pA/A m_T, "
        "pA/A m_c [mub], ratio m_T, ratio m_c)",
    )
    L += [f"    {E:.0e}: {row!r}," for E, row in t1.items()] + ["}"]
    L += fmt_dict("CARPIO20_TABLE1", ct, "hadron: (N, epsilon)")
    L += fmt_dict(
        "LISOVYI16_TABLE12", lis, "hadron: (f, error), constrained-S column"
    )
    L += fmt_dict(
        "KK06_TABLE1_LO_C",
        kk,
        "hadron: (N, epsilon), LO, c -> hadron, mu_0 = m_c = 1.5 GeV",
    )
    L.append("# --- end of generated block ---")
    print(ruff_format("\n".join(L) + "\n"), end="")


if __name__ == "__main__":
    main()
