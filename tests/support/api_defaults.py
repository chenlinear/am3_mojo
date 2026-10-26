"""State of a fresh AM3 object after init_kernels and the error behaviour of a few calls, as JSON (one process per
implementation).

usage: python tests/support/api_defaults.py --impl IMPL --out FILE.json
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tests/
from support.output import silenced
from support.paths import new_am3
from support.scenarios import dump_state

CALLS = [('set_injection_rate_protons', (np.zeros((2, 2)),)), ('set_mag_field', ()),
         ('set_mag_field', ('abc',)), ('eV_to_had_grid_index', (1e12,)), ('get_grid_size_had', (1,))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    obj = new_am3(a.impl)
    with silenced():
        obj.init_kernels()
    res = {k: v.tolist() for k, v in dump_state(obj).items()}
    res['__calls__'] = errs = {}
    for name, args in CALLS:
        try:
            errs[f'{name}{len(args)}'] = ['ok', repr(getattr(obj, name)(*args))]
        except Exception:
            errs[f'{name}{len(args)}'] = ['error', '']
    json.dump(res, open(a.out, 'w'))


if __name__ == '__main__':
    main()
