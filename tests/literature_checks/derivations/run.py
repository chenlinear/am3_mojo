"""Runs the derivation checks of every module (verify.MODULES), prints their results and exits non-zero if one fails.
The 144 checks take about 50 s on an Apple M5. Run from the repository root:
  .venv/bin/python tests/literature_checks/derivations/run.py
"""
import verify

for module in verify.MODULES:
    __import__(module)

if __name__ == "__main__":
    verify.run()
