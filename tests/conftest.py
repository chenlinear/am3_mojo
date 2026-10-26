"""pytest configuration of the am3_mojo test suite.

tests/unit       am3_mojo on its own (needs `make build`)
tests/reference  comparison with the original C++ AM3 (needs `make reference`;
                 skipped with a message if the reference is not built)
"""
import os
import pytest

from support.paths import available, LABELS


def pytest_collection_modifyitems(config, items):
    mojo, cpp = available('mojo'), available('cpp')
    for item in items:
        if os.sep + 'reference' + os.sep in str(item.path):
            item.add_marker(pytest.mark.reference)
            if not cpp:
                item.add_marker(pytest.mark.skip(reason=f'{LABELS["cpp"]} not built: run `make reference` first'))
        if not mojo:
            item.add_marker(pytest.mark.skip(reason='am3_mojo not built: run `make build` first'))
