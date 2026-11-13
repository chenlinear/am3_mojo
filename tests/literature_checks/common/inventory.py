"""Inventory of the equations of the process pages (the README.md of each
process folder of tests/literature_checks/). It gives the source of each
equation, how the equation relates to that source, the checks that cover it and
its derivation check.

Each derivation module lists the entries of its equations (INVENTORY, of
verify.Entry, and ALSO). Each entry pins a piece of the equation's LaTeX, copied
from its page (the entry's file, a key of latex.FILES). The pin must occur
exactly once in that page.
A display equation (a formula line between two lines holding only $$) with
several relations has one entry. ALSO pins its other relations to that entry,
whose derivation check covers them too. Run from the repository root:
  .venv/bin/python tests/literature_checks/common/inventory.py
It prints the table with each equation's page and line. It fails in these
cases. First, a relation of a display equation on a page of latex.FILES holds
no pin of that page. (A relation is a part of the formula, as latex.parts
splits it.) Second, a pin or an ALSO pin no longer occurs exactly once in its
page (the equation was edited: recheck the entry and its derivation check).
Third, an equation has no derivation check. It also fails on a duplicate id,
an unknown relation, and an ALSO entry or a derivation check whose id is not
in the inventory.
A line holding only $$...$$ counts as a display equation too. For a model or
standard entry, the derivation check derives what follows from it: a count, a
conservation law, a limit, a continuity, the kinematics.

Relations:
  transcribed  the source's equation in the README's notation (units, symbols)
  rearranged   rewritten or combined from several equations of the source
  derived      derived here: an integral, primitive, limit or expansion of a
               cited equation, or a consequence of kinematics stated without a
               source
  corrected    the source has a misprint, and the README states the correct
               form and why
  model        a modeling or numerical choice of AM3 or am3_mojo (nothing to
               derive: the tests and the reference comparison check the code
               against it)
  standard     textbook kinematics or definitions
"""

import sys

import latex as L
import verify

RELATIONS = (
    "transcribed",
    "rearranged",
    "derived",
    "corrected",
    "model",
    "standard",
)


def gather():
    """Importing the modules registers their derivation checks
    (verify.CHECKS)."""
    entries, problems = [], []
    for module in verify.load_modules():
        ids = [e.id for e in module.INVENTORY]
        problems += [
            f"ALSO: unknown id {i}" for i in module.ALSO if i not in ids
        ]
        for e in module.INVENTORY:
            e.also = module.ALSO.get(e.id, ())
        entries += module.INVENTORY
    return entries, problems


def display_equations(text):
    lines = [line.strip() for line in text.splitlines()]
    blocks = [
        (i + 1, "$$" + lines[i] + "$$")
        for i in range(1, len(lines) - 1)
        if lines[i - 1] == "$$"
        and lines[i + 1] == "$$"
        and lines[i] not in ("", "$$")
    ]
    one_line = [
        (i + 1, s)
        for i, s in enumerate(lines)
        if s.startswith("$$") and s.endswith("$$") and len(s) > 4
    ]
    return sorted(blocks + one_line)


def main():
    entries, also_problems = gather()
    ids = [e.id for e in entries]
    problems = [
        f"duplicate id {i}" for i in sorted(set(ids)) if ids.count(i) > 1
    ]
    problems += [
        f"{e.id}: unknown relation {e.relation}"
        for e in entries
        if e.relation not in RELATIONS
    ]
    lines = {}
    problems += also_problems
    for e in entries:
        text = L.text(e.file)
        if text.count(e.pin) != 1:
            problems.append(
                f"{e.id}: pin found {text.count(e.pin)} times in "
                f"{e.file}: {e.pin}"
            )
        else:
            lines[e.id] = text[: text.index(e.pin)].count("\n") + 1
        problems += [
            f"{e.id}: ALSO pin found {text.count(p)} times in {e.file}: {p}"
            for p in e.also
            if text.count(p) != 1
        ]
    for page in L.FILES:
        # (only a pin of the page itself covers its relation)
        pins = [
            p.strip("$")
            for e in entries
            if e.file == page
            for p in (e.pin,) + e.also
        ]
        for i, eq in display_equations(L.text(page)):
            for rel in L.parts(eq[2:-2]):
                if not any(p in rel or rel in p for p in pins):
                    problems.append(
                        f"{page} line {i}: a relation without an "
                        f"inventory pin: {rel[:80]}"
                    )
    checks = verify.CHECKS
    problems += [
        f"derivation check for unknown id {i}" for i in checks if i not in ids
    ]
    problems += [
        f"{e.id}: an equation without a derivation check"
        for e in entries
        if e.id not in checks
    ]

    print(
        "| id | line | relation | source | local copy | existing checks | "
        "derivation check |"
    )
    print("|---|---|---|---|---|---|---|")
    for e in entries:
        derived = checks.get(e.id)
        where = lines.get(e.id, "(absent)")
        print(
            f"| {e.id} | {e.file}:{where} | {e.relation} | {e.source} | "
            f"{e.paper or '—'} | {e.checks or '—'} | "
            f"{derived.__module__ + '.py' if derived else '—'} |"
        )
    n = {r: sum(e.relation == r for e in entries) for r in RELATIONS}
    print(
        f"\n{len(entries)} equations of the process pages: "
        + ", ".join(f"{n[r]} {r}" for r in RELATIONS)
    )
    print(
        f"{sum(e.id in checks for e in entries)} of them have a "
        "derivation check"
    )
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
