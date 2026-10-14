"""Thread-count selection for the parallel loops.

Parallel sections run on the Mojo runtime's worker pool through
`std.runtime._asyncrt.TaskGroup`, the mechanism behind `parallelize()` in MAX.
The number of tasks defaults to the runtime's parallelism level; the
AM3_MOJO_THREADS environment variable overrides it (1 = fully serial).
"""

from std.os import getenv
from std.runtime import parallelism_level
from std.runtime._asyncrt import TaskGroup


def default_num_threads() -> Int:
    var s = getenv("AM3_MOJO_THREADS")
    if s.byte_length() > 0:
        try:
            return max(1, Int(s))
        except:
            pass
    return max(1, parallelism_level())


async def _task[
    T: AnyType, work: def(Pointer[T, MutUntrackedOrigin], Int, Int) thin -> None
](ctx: Pointer[T, MutUntrackedOrigin], t: Int, nt: Int):
    work(ctx, t, nt)


def parallel_run[
    T: AnyType,
    //,
    work: def(Pointer[T, MutUntrackedOrigin], Int, Int) thin -> None,
](mut ctx: T, nthreads: Int):
    """Run `work(ctx, t, nthreads)` for t = 0 .. nthreads-1 on the worker pool
    (inline when nthreads <= 1); `work` partitions the job by t.

    `ctx` is borrowed for the duration of the call, so the borrow checker
    keeps it alive while the workers use the (untracked) pointer to it."""
    var p = Pointer(to=ctx).unsafe_origin_cast[MutUntrackedOrigin]()
    if nthreads <= 1:
        work(p, 0, 1)
        return
    var tg = TaskGroup()
    for t in range(nthreads):
        tg.create_task(_task[T, work](p, t, nthreads))
    tg.wait()
