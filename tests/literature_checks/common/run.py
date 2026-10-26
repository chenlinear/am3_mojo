"""Runs the derivation checks of the modules named (default: every module of verify.MODULES), prints the results and
exits non-zero on a failure. The 245 checks take about 80 s on an Apple M5. From the repository root:
  .venv/bin/python tests/literature_checks/common/run.py [module ...]
e.g. `run.py kaon_decays charm_decays` runs the checks of kaons/kaon_decays.py and charm/charm_decays.py only.
"""
import re
import sys
import time

import verify

# words whose full stop does not end a sentence ("eq. 6", "Hümmer et al.'s", "p. 324", "prop. to")
ABBREVIATIONS = {"eq", "eqs", "p", "pp", "fig", "figs", "al", "prop", "e.g", "i.e", "cf", "vs"}


def first_sentence(doc):
    """The first sentence of a docstring, on one line: up to the first full stop that ends a word other than an
    abbreviation or an ellipsis and is followed by a space or the end (a spaced dot product, p_a . p_b, is no stop)."""
    text = " ".join(doc.split())
    for m in re.finditer(r"(\S+)\.(?=\s|$)", text):
        word = m.group(1).lstrip("(").lower()
        if word not in ABBREVIATIONS and not word.endswith(".."):
            return text[:m.end()]
    return text


def run(names):
    """Runs the checks of the modules `names` (of verify.MODULES), prints each check with the first sentence of its
    docstring and its results, and exits non-zero if one fails. The checks are those the modules register in
    verify.CHECKS when load_modules imports them."""
    verify.load_modules(names)
    checks = [(entry_id, f) for entry_id, f in verify.CHECKS.items() if f.__module__ in names]
    failed = 0
    for entry_id, f in checks:
        t0 = time.time()
        results = f()
        print(f"{entry_id}: {first_sentence(f.__doc__)} ({time.time() - t0:.1f} s)")
        for r in results:
            print(f"  {'ok  ' if r.ok else 'FAIL'} {r.what}: {r.detail}")
            failed += not r.ok
    print(f"\n{len(checks)} checks, {failed} failed results")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    run(sys.argv[1:] or verify.MODULES)
