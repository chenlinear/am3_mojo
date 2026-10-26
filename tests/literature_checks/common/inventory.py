"""Inventory of the equations of the process pages (the README.md of each process folder of tests/literature_checks/):
for each, the source it is cited from, how it relates to that source, the checks that cover it and its derivation
check.

Each derivation module lists the entries of its equations (INVENTORY, of verify.Entry, and ALSO); this script gathers
them in the order of verify.MODULES. Each entry pins a piece of the equation's LaTeX, copied from its page (the entry's
file, a key of latex.FILES), where it must occur exactly once. A display equation (a formula line between two lines
holding only $$) with several relations has one entry; ALSO pins its other relations to that entry, whose derivation
check covers them too. Run from the repository root:
  .venv/bin/python tests/literature_checks/common/inventory.py
It prints the table with each equation's page and line. It fails if a relation of a display equation (a part of its
formula as latex.parts splits it) on a page of latex.FILES holds no pin, if a pin no longer occurs exactly once in its
page (the equation was edited: recheck the entry and its derivation check), or if an equation has no derivation check.
A line holding only $$...$$ counts as a display equation too. The pages include the main README.md, which holds no
display equation. For a model or standard entry, the derivation check derives what follows from it: a count, a
conservation law, a limit, a continuity, the kinematics.

Relations:
  transcribed  the source's equation in the README's notation (units, symbols)
  rearranged   rewritten or combined from several equations of the source
  derived      derived here: an integral, primitive, limit or expansion of a cited equation, or a
               consequence of kinematics stated without a source
  corrected    the source has a misprint; the README states the correct form and why
  model        a modelling or numerical choice of AM3 or am3_mojo (nothing to derive; the tests
               and the reference comparison check the code against it)
  standard     textbook kinematics or definitions
"""
import sys

import latex as L
import verify

RELATIONS = ("transcribed", "rearranged", "derived", "corrected", "model", "standard")


def gather():
    """The entries of the modules of verify.MODULES, in that order, each with the ALSO pins of its module, and the
    problems of the modules' ALSO (an id without an entry of the module). Importing the modules registers their
    derivation checks (verify.CHECKS)."""
    entries, problems = [], []
    for module in verify.load_modules():
        ids = [e.id for e in module.INVENTORY]
        problems += [f"ALSO: unknown id {i}" for i in module.ALSO if i not in ids]
        for e in module.INVENTORY:
            e.also = module.ALSO.get(e.id, ())
        entries += module.INVENTORY
    return entries, problems


def display_equations(text):
    """(line number, $$formula$$) of every display equation: a formula line between two lines holding only $$, or a
    line holding only $$formula$$."""
    lines = [line.strip() for line in text.splitlines()]
    blocks = [(i + 1, "$$" + lines[i] + "$$") for i in range(1, len(lines) - 1)
              if lines[i - 1] == "$$" and lines[i + 1] == "$$" and lines[i] not in ("", "$$")]
    one_line = [(i + 1, s) for i, s in enumerate(lines) if s.startswith("$$") and s.endswith("$$") and len(s) > 4]
    return sorted(blocks + one_line)


def main():
    entries, also_problems = gather()
    ids = [e.id for e in entries]
    problems = [f"duplicate id {i}" for i in sorted(set(ids)) if ids.count(i) > 1]
    problems += [f"{e.id}: unknown relation {e.relation}" for e in entries if e.relation not in RELATIONS]
    lines = {}
    problems += also_problems
    for e in entries:
        text = L.text(e.file)
        if text.count(e.pin) != 1:
            problems.append(f"{e.id}: pin found {text.count(e.pin)} times in {e.file}: {e.pin}")
        else:
            lines[e.id] = text[:text.index(e.pin)].count("\n") + 1
        problems += [f"{e.id}: ALSO pin found {text.count(p)} times in {e.file}: {p}" for p in e.also
                     if text.count(p) != 1]
    pins = [p.strip("$") for e in entries for p in (e.pin,) + e.also]
    for page in L.FILES:
        for i, eq in display_equations(L.text(page)):
            for rel in L.parts(eq[2:-2]):
                if not any(p in rel or rel in p for p in pins):
                    problems.append(f"{page} line {i}: a relation without an inventory pin: {rel[:80]}")
    checks = verify.CHECKS
    problems += [f"derivation check for unknown id {i}" for i in checks if i not in ids]
    problems += [f"{e.id}: an equation without a derivation check" for e in entries if e.id not in checks]

    print("| id | line | relation | source | local copy | existing checks | derivation check |")
    print("|---|---|---|---|---|---|---|")
    for e in entries:
        derived = checks.get(e.id)
        where = lines.get(e.id, "(absent)")
        if e.file != "README":
            where = f"{e.file}:{where}"
        print(f"| {e.id} | {where} | {e.relation} | {e.source} | {e.paper or '—'} | "
              f"{e.checks or '—'} | {derived.__module__ + '.py' if derived else '—'} |")
    n = {r: sum(e.relation == r for e in entries) for r in RELATIONS}
    print(f"\n{len(entries)} equations of the process pages: " + ", ".join(f"{n[r]} {r}" for r in RELATIONS))
    print(f"{sum(e.id in checks for e in entries)} of them have a derivation check")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
