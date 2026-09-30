"""pytest configuration of the am3_mojo test suite.

tests/unit       am3_mojo on its own (needs `make build`)
tests/reference  comparison with the original C++ AM3 (needs `make reference`;
                 skipped with a message if the reference is not built)
"""
import os, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from support.paths import available, LABELS  # noqa: E402


def pytest_collection_modifyitems(config, items):
    for item in items:
        if os.sep + 'reference' + os.sep in str(item.fspath):
            item.add_marker(pytest.mark.reference)
            if not available('cpp'):
                item.add_marker(pytest.mark.skip(
                    reason=f'{LABELS["cpp"]} not built: run `make reference` first'))
    if not available('mojo'):
        for item in items:
            item.add_marker(pytest.mark.skip(reason='am3_mojo not built: run `make build` first'))
