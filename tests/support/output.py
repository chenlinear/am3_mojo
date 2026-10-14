"""Silencing the standard output: what Python writes and what the module writes to file descriptor 1 directly (as
init_kernels does with its parameters), which contextlib.redirect_stdout does not catch."""
import contextlib
import os
import sys


@contextlib.contextmanager
def silenced():
    """Sends everything written to the standard output inside the block to /dev/null."""
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
