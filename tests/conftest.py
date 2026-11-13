import os

import pytest

from support.paths import (
    available,
    built_from_another_patch,
    LABELS,
    MOJO_BUILD,
)


TESTS = os.path.dirname(os.path.abspath(__file__))


def pytest_configure(config):
    # without this error, if a build asked for by name (a variant,
    # build-asserts/) is not there, the run skips every test and passes
    if "AM3_MOJO_BUILD" in os.environ and not available("mojo"):
        raise pytest.UsageError(
            f"AM3_MOJO_BUILD={os.environ['AM3_MOJO_BUILD']}: no am3_mojo "
            f"module in {MOJO_BUILD}"
        )


def pytest_collection_modifyitems(config, items):
    # (the reference tests load both C++ builds, and most load the patched one)
    mojo = available("mojo")
    cpp = available("cpp") and available("cpp-fixed")
    stale = cpp and built_from_another_patch()
    for item in items:
        # (by the folder within tests/, not a folder of the checkout's path)
        rel = os.path.relpath(str(item.path), TESTS)
        if rel.split(os.sep)[0] == "reference":
            item.add_marker(pytest.mark.reference)
            if not cpp:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{LABELS['cpp']} not built: "
                        "run `make reference` first"
                    )
                )
            elif stale:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{LABELS['cpp-fixed']} built from another "
                        "patch: run `make reference`"
                    )
                )
        if not mojo:
            item.add_marker(
                pytest.mark.skip(
                    reason="am3_mojo not built: run `make build` first"
                )
            )
