"""The code's decay modes, read as text from src/am3_mojo/physics/decays.mojo (charm_decay_modes, tau_decay_modes,
short_lived_modes) for tests/unit/test_added_processes.py and literature_checks/kaons/multibody_strangeness.py.
A final state is a key of charm/charm_decays.py: the sorted names of pdg_charm.py, K0 and anti-K0 alike (the
code's X_K0)."""
import re
from pathlib import Path

import sympy as sp

from support.paths import ROOT, load_derivation

L = load_derivation('latex')
key = load_derivation('charm_decays').key   # a final state as a sorted tuple, K0 and anti-K0 alike
DECAYS_MOJO = Path(ROOT, 'src', 'am3_mojo', 'physics', 'decays.mojo')


def mojo_def(src, name):
    """The text of the Mojo function or struct `name`, up to the next top-level statement."""
    m = re.search(r'^(?:def|struct) ' + re.escape(name) + r'\b', src, re.M)
    rest = src[m.end():]
    n = re.search(r'^(?:def |struct |comptime |@|# ---|# ===)', rest, re.M)
    return src[m.start():m.end() + (n.start() if n else len(rest))]


def decays_def(name):
    return mojo_def(L.text(DECAYS_MOJO), name)


def flat(text):
    return re.sub(r'\s+', ' ', text).strip()


def code_number(text):
    """A literal or expression of the code (0.0926 + 0.0899, 2.0 / 3.0), exact."""
    return sp.sympify(text.strip(), rational=True)


# the code's particles X_* (decays.mojo) by the names of pdg_charm; X_K0 is a K0 or an anti-K0
X_NAMES = {'X_PIP': 'pi+', 'X_PIM': 'pi-', 'X_PI0': 'pi0', 'X_MUP': 'mu+', 'X_MUM': 'mu-', 'X_KP': 'K+', 'X_KM': 'K-',
           'X_K0': 'K0', 'X_P': 'p', 'X_N': 'n', 'X_EP': 'e+', 'X_EM': 'e-', 'X_NUE': 'nu_e', 'X_NUEB': 'nu_e_bar',
           'X_NUM': 'nu_mu', 'X_NUMB': 'nu_mu_bar', 'X_NUT': 'nu_tau', 'X_NUTB': 'nu_tau_bar', 'X_GAM': 'gamma',
           'X_TAUP': 'tau+', 'X_ETA': 'eta', 'X_ETAP': 'etap', 'X_OMEGA': 'omega', 'X_PHI': 'phi', 'X_RHOP': 'rho+',
           'X_RHO0': 'rho0', 'X_RHOM': 'rho-', 'X_KSTP': 'Kstar+', 'X_KSTM': 'Kstar-', 'X_KST0': 'Kstar0',
           'X_KST0B': 'Kstar0bar', 'X_F0': 'f0_980', 'X_A1P': 'a1(1260)+', 'X_LAM': 'Lambda', 'X_SIGP': 'Sigma+',
           'X_SIG0': 'Sigma0', 'X_SIGM': 'Sigma-', 'X_XI0': 'Xi0', 'X_XIM': 'Xi-'}
BRANCH = {'CDP': 'D+', 'CD0': 'D0', 'CDSP': 'Ds+', None: 'Lambda_c+'}


def xnames(text):
    return [X_NAMES[x.strip()] for x in text.split(',') if x.strip()]


def x_mode(kind, args):
    """(br, final state, {keyword: value}) of the arguments of an x_ps, x_two or x_michel call."""
    if kind == 'ps':
        br, fs = re.fullmatch(r'([^,]+), \[([^\]]*)\]', args).groups()
        return code_number(br), key(xnames(fs)), {}
    parts = [a.strip() for a in args.split(',')]
    return (code_number(parts[0]), key([X_NAMES[p] for p in parts[1:] if p.startswith('X_')]),
            dict(p.split('=') for p in parts[1:] if '=' in p))


def branches(body, var, name):
    """{name(value): text} of the if/elif/else chains on `var` at the function's first indentation (several chains
    add up), comments removed and whitespace collapsed."""
    out, cur = {}, None
    for ln in body.splitlines()[1:]:
        m = re.match(r'    (?:if|elif) ' + var + r' == (\w+):\s*$', ln)
        if m or re.match(r'    else:\s*$', ln):
            cur = name(m.group(1) if m else None)
        elif re.match(r'    \S', ln):
            cur = None
        elif cur is not None:
            out[cur] = out.get(cur, '') + ' ' + ln.split('#')[0]
    return {k: flat(v) for k, v in out.items()}


SL_CALLS = {'pp': r'_sl_pp\(sl, ([^,]+), ([^,]+), (X_\w+), (True|False), ([^)]+)\)',
            'pv': r'_sl_pv\(sl, ([^,]+), ([^,]+), (X_\w+), (True|False), ([^,]+), ([^)]+)\)',
            'ps': r'_sl_ps\(sl, ([^,]+), ([^,]+), \[([^\]]*)\]\)',
            'bl3': r'sl\.append\(x_bl3\(([^,]+), (X_\w+), (X_EP|X_MUP), (X_NUE|X_NUM), (\d)\)\)'}
LEP_CALL = r'lep\.append\(x_two\(([^,]+), (X_MUP|X_TAUP), (X_NUM|X_NUT), P0=([^)]+)\)\)'
HAD_CALL = r'had\.append\( ?x_ps\(([^,]+), \[([^\]]*)\]\) ?\)'
LEPTONS = {'e': ('e+', 'nu_e'), 'mu': ('mu+', 'nu_mu')}


def code_charm_modes():
    """{species: dict(sl, lep, had, incl)} of charm_decay_modes (decays.mojo); a mode is a dict with its final state
    fs, branching ratio br (as written, before the scaling), the lepton of a semileptonic mode and its parameters."""
    out = {}
    for s, text in branches(decays_def('charm_decay_modes'), 's', BRANCH.get).items():
        sl, lep, had, parsed = [], [], [], 0
        for kind, pattern in SL_CALLS.items():
            for g in re.findall(pattern, text):
                parsed += 1
                if kind == 'bl3':
                    lepton = 'e' if g[2] == 'X_EP' else 'mu'
                    sl.append(dict(kind=kind, lepton=lepton, br=code_number(g[0]), hadrons=[X_NAMES[g[1]]],
                                   fs=key([X_NAMES[g[1]], *LEPTONS[lepton]]), bl=int(g[4])))
                    continue
                hadrons = xnames(g[2]) if kind == 'ps' else [X_NAMES[g[2]]]
                params = {'pp': ('alpha',), 'pv': ('rV', 'r2'), 'ps': ()}[kind]
                extra = dict(cs=g[3] == 'True', **{p: code_number(v) for p, v in zip(params, g[4:])}) if params else {}
                for lepton, br in (('e', code_number(g[0])), ('mu', code_number(g[1]))):
                    if kind != 'ps' or br > 0:
                        sl.append(dict(kind=kind, lepton=lepton, br=br, hadrons=hadrons,
                                       fs=key(hadrons + list(LEPTONS[lepton])), **extra))
        for g in re.findall(LEP_CALL, text):
            parsed += 1
            lep.append(dict(br=code_number(g[0]), fs=key([X_NAMES[g[1]], X_NAMES[g[2]]]), P0=code_number(g[3])))
        for g in re.findall(HAD_CALL, text):
            parsed += 1
            had.append(dict(br=code_number(g[0]), fs=key(xnames(g[1]))))
        statements = len(re.findall(r'\b_sl_p[pvs]\(|\.append\(', text))
        if statements != parsed:
            raise ValueError(f'{s}: {statements} modes in charm_decay_modes, {parsed} parsed')
        incl = re.findall(r'incl = ([\d.eE+-]+)', text)
        out[s] = dict(sl=sl, lep=lep, had=had, incl=code_number(incl[0]) if len(incl) == 1 else None)
    return out


def code_tau_modes():
    """The modes of tau_decay_modes: dict(kind, br, fs, a0)."""
    out = []
    for kind, args in re.findall(r't\.append\(x_(michel|two|ps)\((.*?)\)\)', flat(decays_def('tau_decay_modes'))):
        br, fs, kw = x_mode(kind, args)
        out.append(dict(kind=kind, br=br, fs=fs, a0=kw.get('a0')))
    return out


def code_short_lived():
    """{X name: [dict(br, fs)]} of short_lived_modes."""
    out = {}
    for x, text in branches(decays_def('short_lived_modes'), 'x', lambda v: X_NAMES[v]).items():
        modes = [x_mode(*call) for call in re.findall(r't\.append\(x_(two|ps)\((.*?)\)\)', text)]
        out[x] = [dict(br=br, fs=fs) for br, fs, _ in modes]
    return out
