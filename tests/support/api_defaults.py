"""Default state of a fresh AM3 object after init_kernels, and the error
behaviour of a few calls, as JSON (run in its own process per implementation).

usage: python tests/support/api_defaults.py --impl IMPL --out FILE.json
"""
import argparse, contextlib, io, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.paths import load  # noqa: E402
from support.scenarios import dump_state  # noqa: E402

CALLS = [('set_injection_rate_protons', (np.zeros((2, 2)),)), ('set_mag_field', ()),
         ('set_mag_field', ('abc',)), ('eV_to_had_grid_index', (1e12,)), ('get_grid_size_had', (1,))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    mod = load(a.impl)
    obj = mod.AM3()
    with contextlib.redirect_stdout(io.StringIO()):
        obj.init_kernels()
    res = {k: v.tolist() for k, v in dump_state(obj).items()}
    errs = {}
    for name, args in CALLS:
        try:
            errs[f'{name}{len(args)}'] = ['ok', repr(getattr(obj, name)(*args))]
        except Exception:
            errs[f'{name}{len(args)}'] = ['error', '']
    res['__calls__'] = errs
    json.dump(res, open(a.out, 'w'))


if __name__ == '__main__':
    main()
