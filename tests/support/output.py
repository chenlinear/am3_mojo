"""init_kernels prints its parameters to file descriptor 1 directly, which
contextlib.redirect_stdout misses."""

import contextlib, os, sys


@contextlib.contextmanager
def silenced():
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


def quiet(f, *args):
    with silenced():
        return f(*args)
