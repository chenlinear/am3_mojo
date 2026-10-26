"""Silencing the standard output, also what the module writes to file descriptor 1 directly (init_kernels prints its
parameters there), which contextlib.redirect_stdout misses."""
import contextlib
import os
import sys


@contextlib.contextmanager
def silenced():
    """Sends everything written to the standard output in the block to /dev/null."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, 'w') as null:
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


def quiet(f, *args):
    """f(*args), silenced; returns its result."""
    with silenced():
        return f(*args)
