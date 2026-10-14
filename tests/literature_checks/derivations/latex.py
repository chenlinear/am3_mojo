"""Reading equations out of README.md and parsing their LaTeX with sympy. It also reads the local citation index
citations.md, the Results of tests/literature_checks/README.md (file="checks") and tests/reference/README.md
(file="reference").

The checks take their README side from the file itself, so that an edited equation is checked again.
`display(pin)` returns the $$ line holding the pin; `inline(math)` asserts that the inline span $math$ occurs
exactly once and returns it. `parts`, `sides` and `between` cut these into the pieces that `to_sympy` parses. The
checks assert verbatim the pieces the parser cannot read (evaluation brackets, limits, min).

`to_sympy(tex, tokens)` replaces every symbol of the formula, given as its LaTeX (`r"\\gamma_+"`, also bare
`r"\\gamma"`), by a placeholder before parsing and by its sympy object after. Parsing fails loudly on anything left
over: a symbol not in `tokens` (`\\pi` and `e` become the constants, which sympy reads as plain symbols), or a
function neither elementary nor in `funcs` (`\\gamma(x)` is the gamma function). `\\ln^2 2k`, which sympy reads as
ln(k)^22, is rewritten first. A symbol before parentheses (`k(S - k)`, a call for sympy) is a product unless the
name is in `funcs`, and a/(b) c is (a/(b)) c. `\\log` is the natural logarithm, as parse_latex reads it (the check
of Koers et al.'s fit substitutes the README's log10).

The paper side: `in_paper(name, *snippets)` finds snippets copied from a paper's TeX source in its arXiv tarball
among the local copies of the papers (git-ignored; None without them), and `to_sympy` parses them too.
`patched(old, new)` makes the file read with one equation broken, for the checks' controls. Sources without TeX
(scans, books) are transcribed in the paper's notation, with their page and equation numbers, in the module that
uses them.
"""
import gzip
import re
import tarfile
import warnings
from pathlib import Path

import sympy as sp
from sympy.core.function import AppliedUndef

warnings.filterwarnings("ignore", message=".*antlr4.*")
from sympy.parsing.latex import parse_latex  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FILES = {"README": ROOT / "README.md", "citations": ROOT / "knowledge" / "citations.md",
         "checks": ROOT / "tests" / "literature_checks" / "README.md",
         "reference": ROOT / "tests" / "reference" / "README.md"}
PAPERS = ROOT / "knowledge" / "papers"
# the arXiv sources in PAPERS of the papers whose TeX the checks read
LEE = "arXiv-2010.15430v1.tar.gz"             # Lee, Lyubyakin & Stotsky
LEE98 = "arXiv-astro-ph9604098v1.tar.gz"      # Lee (1998)
ANGUELOV = "arXiv-astro-ph0001252v1.tar.gz"   # Anguelov et al.
LIPARI = "arXiv-0704.0718v1.tar.gz"           # Lipari, Lusignoli & Meloni
HUEMMER = "arXiv-1002.1310v3.tar.gz"          # Hümmer et al.
HAYEN = "arXiv-1709.07530v2.tar.gz"           # Hayen et al.
DIMITRAKOUDIS = "arXiv-1209.0413v1.tar.gz"    # Dimitrakoudis et al.
KOERS = "arXiv-hep-ph0611219v1.tar.gz"        # Koers et al.
KAFEXHIU = "arXiv-1406.7369v2.tar.gz"         # Kafexhiu et al.
KELNER = "arXiv-astro-ph0606058v1.tar.gz"     # Kelner et al.
NA49 = "arXiv-1004.1889v1.tar.gz"             # Anticic et al. (NA49)
KLINGER = "arXiv-2312.13371v2.tar.gz"         # Klinger et al., the AM3 paper
GAO = "arXiv-1610.05306v2.tar.gz"             # Gao et al.
BS96 = "arXiv-astro-ph9506112v1.tar.gz"       # Böttcher & Schlickeiser
ELEMENTARY = (sp.log, sp.exp, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh,
              sp.asinh, sp.acosh, sp.atanh)
PLACEHOLDER = "\\Omega_{%d}"   # a symbol of the formula while it is parsed
FUNCTION = "\\Phi_{%d}"        # a function applied in the formula


_OVERRIDE = {}
LINK = re.compile(r"\[([^\]]*)\]\([^)\s]*\)")   # a Markdown link [text](target)


def text(file="README"):
    """The file as the checks read it: every Markdown link reduced to its text, so that the citation links do not
    break the pins (no link stands inside the mathematics)."""
    return _OVERRIDE[file] if file in _OVERRIDE else LINK.sub(r"\1", FILES[file].read_text())


class patched:
    """Context in which the file reads with one occurrence of old replaced by new (for the controls:
    a broken equation must fail the check that the real one passes)."""
    def __init__(self, old, new, file="README"):
        self.old, self.new, self.file = old, new, file

    def __enter__(self):
        t = text(self.file)
        if t.count(self.old) != 1:
            raise LookupError(f"{self.file}: found {t.count(self.old)} times: {self.old}")
        self.outer = _OVERRIDE.get(self.file)
        _OVERRIDE[self.file] = t.replace(self.old, self.new)

    def __exit__(self, *exc):
        if self.outer is None:
            del _OVERRIDE[self.file]
        else:
            _OVERRIDE[self.file] = self.outer


def line(pin, file="README"):
    """The line holding the pin, which must occur exactly once in the file."""
    t = text(file)
    n = t.count(pin)
    if n != 1:
        raise LookupError(f"{file}: found {n} times: {pin}")
    start = t.rfind("\n", 0, t.index(pin)) + 1
    end = t.find("\n", start)
    return t[start:end if end >= 0 else len(t)]


def display(pin, file="README"):
    """The LaTeX of the display equation (a line holding only $$...$$) that contains the pin."""
    s = line(pin, file).strip()
    if not (s.startswith("$$") and s.endswith("$$")):
        raise ValueError(f"not a display equation: {s[:80]}")
    return s[2:-2]


def inline(math, file="README"):
    """Asserts that the inline span $math$ occurs exactly once in the file, and returns math."""
    line("$" + math + "$", file)
    return math


def tex_source(name):
    """The TeX (all .tex files) of an arXiv source among the local copies of the papers, or None if it is not there."""
    path = PAPERS / name
    if not path.exists():
        return None
    try:
        with tarfile.open(path) as tar:
            return "\n".join(tar.extractfile(m).read().decode("latin-1") for m in tar.getmembers()
                             if m.isfile() and m.name.endswith(".tex"))
    except tarfile.ReadError:   # a single gzipped .tex file
        return gzip.open(path).read().decode("latin-1")


def in_paper(name, *snippets):
    """Whether each snippet, copied from the TeX source `name`, occurs in it (ignoring whitespace):
    True/False, or None when the source is not among the local copies of the papers (git-ignored)."""
    tex = tex_source(name)
    if tex is None:
        return None
    flat = re.sub(r"\s+", "", tex)
    return all(re.sub(r"\s+", "", s) in flat for s in snippets)


def _depths(tex):
    """Bracket depth before each character ({, (, [ open; }, ), ] close)."""
    depth, out = 0, []
    for ch in tex:
        if ch in "})]":
            depth -= 1
        out.append(depth)
        if ch in "{([":
            depth += 1
    return out


def split_top(tex, sep):
    """Splits at the occurrences of sep that are outside all brackets."""
    d, pieces, last, i = _depths(tex), [], 0, 0
    while True:
        i = tex.find(sep, i)
        if i < 0:
            break
        if d[i] == 0:
            pieces.append(tex[last:i])
            last = i + len(sep)
        i += len(sep)
    pieces.append(tex[last:])
    return pieces


def parts(tex):
    """The comma/quad-separated parts of a display equation, without trailing punctuation."""
    out = [tex]
    for sep in (r",\qquad", r",\quad", r"\qquad"):
        out = [q for p in out for q in split_top(p, sep)]
    return [p.strip().rstrip(".,").strip() for p in out if p.strip()]


def part(tex, start):
    """The one part of a display equation that starts with `start`."""
    found = [p for p in parts(tex) if p.startswith(start)]
    if len(found) != 1:
        raise LookupError(f"{len(found)} parts start with {start}")
    return found[0]


def sides(p):
    """The sides of an equation a = b = ... (split at top-level =)."""
    return [s.strip() for s in split_top(p, "=")]


def between(tex, left, right):
    """The text between the first `left` and the next `right` after it."""
    i = tex.index(left) + len(left)
    return tex[i:tex.index(right, i)]


def instantiate(tex, sign):
    """The + (sign = +1) or - (sign = -1) instance of a formula written with \\pm and \\mp."""
    a, b = ("+", "-") if sign > 0 else ("-", "+")
    return tex.replace(r"\pm", a).replace(r"\mp", b)


def normalise(tex):
    """Removes spacing and sizing commands and rewrites what parse_latex reads wrongly."""
    tex = re.sub(r"\\(thinspace|thickspace|negthinspace|textstyle|displaystyle|[,;!])", " ", tex)
    tex = re.sub(r"\\qty\s*(?=[(\[])", "", tex)                       # physics package: \qty(...) is (...)
    tex = re.sub(r"\\mathrm\{\\?log\}", r"\\ln", tex)
    tex = re.sub(r"\{([^{}]*)\\over([^{}]*)\}", r"\\frac{\1}{\2}", tex)    # {a \over b}
    tex = re.sub(r"\\(left|right|[Bb]igg?[lr]?)(?![A-Za-z])", "", tex)
    tex = tex.replace("\\ ", " ").replace(r"\qquad", " ").replace(r"\quad", " ")
    tex = re.sub(r"\\[td]frac", r"\\frac", tex)
    tex = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", tex)
    tex = re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", tex)
    tex = re.sub(r"\\ln\s*\^\s*\{?(\d)\}?\s*(\([^()]*\)|\{[^{}]*\}|[0-9A-Za-z]+)", r"(\\ln{\2})^{\1}", tex)
    tex = re.sub(r"\\ln\{(\([^()]*\)|\{[^{}]*\})\}", r"\\ln\1", tex)
    return _slash_groups(tex.replace("[", "(").replace("]", ")"))


def _slash_groups(tex):
    """a/(b) c is (a/(b)) c, as physics texts mean it; parse_latex would read a/((b) c)."""
    out, i = [], 0
    while i < len(tex):
        j = i + 1
        while tex[i] == "/" and j < len(tex) and tex[j] == " ":
            j += 1
        if tex[i] == "/" and j < len(tex) and tex[j] == "(":
            depth, end = 0, j
            for end in range(j, len(tex)):
                depth += {"(": 1, ")": -1}.get(tex[end], 0)
                if depth == 0:
                    break
            out.append(r"\cdot\frac{1}{" + tex[j:end + 1] + "}")
            i = end + 1
        else:
            out.append(tex[i])
            i += 1
    return "".join(out)


def _replace_token(tex, token, repl, call=False):
    """Replaces the token where it stands as a symbol (as a function name before "(" if call): a
    command token (\\gamma_+) not followed by letters; a letter token (m, p_+) not inside a command
    name (the c of \\cdot), but also next to other letters (mc^2 is m c^2)."""
    out, i, count, n = [], 0, 0, len(token)

    def fits(j):
        if not tex.startswith(token, j):
            return False
        after = tex[j + n:].lstrip()
        if call and not after.startswith("("):
            return False
        return not (token.startswith("\\") and token[-1].isalpha() and tex[j + n:j + n + 1].isalpha())
    while i < len(tex):
        if fits(i):
            out.append(repl)
            i += n
            count += 1
        elif tex[i] == "\\":
            cmd = re.match(r"\\([A-Za-z]+|.)", tex[i:]).group(0)
            out.append(cmd)
            i += len(cmd)
        else:
            out.append(tex[i])
            i += 1
    return "".join(out), count


def to_sympy(tex, tokens, funcs=None, sign=None):
    """Parses a formula. tokens maps the LaTeX of each symbol to its sympy object; funcs maps the
    LaTeX name of each function applied in the formula ("G", r"H_+") to a callable that returns a
    sympy expression. sign instantiates \\pm and \\mp first."""
    funcs = funcs or {}
    if sign is not None:
        tex = instantiate(tex, sign)
    tex = normalise(tex)
    calls = {}
    for n, name in enumerate(sorted(funcs, key=len, reverse=True)):
        tex, count = _replace_token(tex, name, FUNCTION % n, call=True)
        calls["Phi_{%d}" % n] = funcs[name]
    targets = {}
    for n, token in enumerate(sorted(tokens, key=len, reverse=True)):
        tex, count = _replace_token(tex, token, " " + PLACEHOLDER % n + " ")
        if count:
            targets[sp.Symbol(PLACEHOLDER.replace("\\", "") % n)] = sp.sympify(tokens[token])
    # a symbol before parentheses multiplies them: n (a/b)^{1/2} is not the call n(a/b), square-rooted
    tex = re.sub(r"(\\Omega_\{\d+\})\s*\(", r"\1\\cdot(", tex)
    expr = parse_latex(tex)
    for f in sorted(expr.atoms(AppliedUndef), key=lambda f: -sp.count_ops(f)):
        name = f.func.__name__
        if name in calls:
            continue
        if len(f.args) != 1:
            raise ValueError(f"unexpected function {f} in {tex}")
        expr = expr.xreplace({f: sp.Symbol(name) * f.args[0]})   # a symbol before (...): a product
    expr = expr.xreplace(targets).xreplace({sp.Symbol("pi"): sp.pi, sp.Symbol("e"): sp.E})
    expr = expr.replace(lambda x: isinstance(x, sp.log) and len(x.args) == 2 and x.args[1] == sp.E,
                        lambda x: sp.log(x.args[0]))
    allowed = set().union(set(), *[v.free_symbols for v in targets.values()])
    unknown = expr.free_symbols - allowed
    if unknown:
        raise ValueError(f"symbols without a token: {sorted(map(str, unknown))} in {tex}")
    given_undef = {f.func.__name__ for v in targets.values() for f in v.atoms(AppliedUndef)}   # e.g. N(E_N) as a token
    for f in expr.atoms(AppliedUndef):
        if f.func.__name__ not in calls and f.func.__name__ not in given_undef:
            raise ValueError(f"unexpected function {f} in {tex}")
    given = tuple({type(f) for v in targets.values() for f in v.atoms(sp.Function)})   # e.g. zeta(3) as a token
    for f in expr.atoms(sp.Function) - expr.atoms(AppliedUndef):
        if not isinstance(f, ELEMENTARY + given):
            raise ValueError(f"unexpected function {f.func} in {tex} (a symbol read as a function?)")
    for name, call in calls.items():
        expr = expr.replace(sp.Function(name), call)
    return _evaluated(expr)


def _evaluated(e):
    """parse_latex builds unevaluated expressions (an exponent -1/2 stays Mul(-1, 1/2)); rebuilt node by
    node with evaluation on, they take sympy's canonical form."""
    return e if e.is_Atom else e.func(*[_evaluated(a) for a in e.args])
