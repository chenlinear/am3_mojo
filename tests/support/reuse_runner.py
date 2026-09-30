"""Run a sequence of parameter configurations with one am3 implementation.

usage: python tests/support/reuse_runner.py --impl IMPL --setup SETUP --mode MODE --out FILE.npz

  mode "reused": one AM3 object, init_kernels() once, then for every
                 configuration: clear_am3() (except before the first), set the
                 parameters, evolve, dump all getters.
  mode "fresh":  a new AM3 object and a new init_kernels() for every
                 configuration.
  Deliberately wrong sequences, used by tests/plots/plot_reuse.py for contrast:
  mode "no_clear":        like "reused" but clear_am3() is never called.
  mode "no_escape_reset": like "reused" but set_escape_timescale() is not
                          called again after clear_am3().

The output holds f'{config_index}/{getter}' for every public getter, plus the
wall-clock seconds spent in init_kernels (`__init_s`) and in evolving
(`__evolve_s`).
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.paths import load  # noqa: E402
from support.scenarios import dump_state, ME_C2_EV, MP_C2_EV, C_CGS  # noqa: E402

N_STEPS = 100


def switches_leptohadronic(a):
    """Every process on, including quantum synchrotron, whose e+/- kernel
    depends on B and is rebuilt by set_mag_field."""
    a.set_verbosity_level(False)
    a.set_process_parse_sed(True)
    a.set_process_escape(True)
    a.set_process_expansion(True)
    a.set_process_adiabatic_cooling(True)
    a.set_process_quantum_syn(True)
    a.set_process_quantum_syn_cooling(True)
    a.set_process_pp(True)
    a.set_process_annihilation(True)


def switches_blazar(a):
    """The docs/examples blazar model (proton synchrotron, photo-hadronic)."""
    a.set_verbosity_level(False)
    a.set_process_parse_sed(True)
    a.set_process_escape(True)
    a.set_process_adiabatic_cooling(False)
    a.set_process_expansion(False)
    a.set_process_pion_syn(False)
    a.set_process_muon_syn(False)
    a.set_process_pion_compton(False)
    a.set_process_muon_compton(False)
    a.set_process_annihilation(False)
    a.set_process_pp(False)


R = 3e14
V = 4 / 3.0 * np.pi * R ** 3
T_LC = R / C_CGS
# Configurations differ in B, escape and expansion time, pp target density,
# time step and injection. The last one repeats the first.
CONFIGS = [
    dict(B=300.0, t_esc=1.0, t_exp=3.0, n_pp=1e8, dt=3e-3,
         e=(3e41, 1e2, 1e3, 1e5, 1.8, 2.8), p=(1e46, 1e1, 1e1, 1e9, 2.0, 2.0)),
    dict(B=30.0, t_esc=3.0, t_exp=10.0, n_pp=1e9, dt=1e-2,
         e=(1e42, 3e1, 3e1, 1e6, 2.2, 2.2), p=(3e45, 3e0, 3e0, 1e8, 2.3, 2.3)),
    dict(B=1000.0, t_esc=0.5, t_exp=2.0, n_pp=0.0, dt=5e-3,
         e=(1e41, 1e3, 1e4, 1e7, 1.5, 3.0), p=(1e44, 1e2, 1e2, 1e10, 1.8, 1.8)),
]
CONFIGS.append(dict(CONFIGS[0]))

SETUPS = {'leptohadronic': switches_leptohadronic, 'blazar': switches_blazar}


def apply_config(a, c, set_escape=True):
    a.set_mag_field(c['B'])
    if set_escape:
        a.set_escape_timescale(c['t_esc'] * T_LC)
    a.set_expansion_timescale(c['t_exp'] * T_LC)
    a.set_pp_target_proton_density(c['n_pp'])
    a.set_solver_time_step(c['dt'] * T_LC)
    L, emin, eb, emax, i1, i2 = c['e']
    a.set_powerlaw_injection_parameters_electrons(
        V, L, emin * ME_C2_EV, eb * ME_C2_EV, emax * ME_C2_EV, i1, i2, 1.0)
    L, emin, eb, emax, i1, i2 = c['p']
    a.set_powerlaw_injection_parameters_protons(
        V, L, emin * MP_C2_EV, eb * MP_C2_EV, emax * MP_C2_EV, i1, i2, 1.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', required=True)
    ap.add_argument('--setup', required=True, choices=sorted(SETUPS))
    ap.add_argument('--mode', required=True, choices=['reused', 'fresh', 'no_clear', 'no_escape_reset'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    am3mod = load(args.impl)
    setup, mode, out = args.setup, args.mode, args.out
    res = {}
    t_init = t_run = 0.0
    a = None
    for k, c in enumerate(CONFIGS):
        if mode == 'fresh' or a is None:
            a = am3mod.AM3()
            SETUPS[setup](a)
            t0 = time.perf_counter()
            a.init_kernels()
            t_init += time.perf_counter() - t0
        elif mode != 'no_clear':
            a.clear_am3()
        apply_config(a, c, set_escape=(mode != 'no_escape_reset' or k == 0))
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            a.evolve_step()
        t_run += time.perf_counter() - t0
        for name, v in dump_state(a).items():
            res[f'{k}/{name}'] = v
    res['__init_s'] = np.array(t_init)
    res['__evolve_s'] = np.array(t_run)
    np.savez_compressed(out, **res)


if __name__ == '__main__':
    main()
