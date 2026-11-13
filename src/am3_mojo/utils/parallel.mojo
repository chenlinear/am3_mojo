"""Parallel sections run through `std.runtime._asyncrt.TaskGroup`, the
mechanism behind `parallelize()` in MAX. The number of tasks defaults to the
runtime's parallelism level. The AM3_MOJO_THREADS environment variable
overrides it (1 = fully serial), up to the number of logical CPUs.
"""

from std.atomic import Atomic
from std.os import getenv
from std.runtime import parallelism_level
from std.runtime._asyncrt import TaskGroup
from std.sys import num_logical_cores


def default_num_threads() -> Int:
    var s = getenv("AM3_MOJO_THREADS")
    var n = max(1, parallelism_level())
    if s.byte_length() > 0:
        try:
            var k = Int(s)
            # (more threads than logical CPUs only take memory and time)
            var cpus = max(1, num_logical_cores())
            if k > cpus:
                print(
                    "am3_mojo: AM3_MOJO_THREADS=",
                    s,
                    " is more than the ",
                    cpus,
                    " logical CPUs; using ",
                    cpus,
                    " threads",
                    sep="",
                )
                return cpus
            if k >= 1:
                return k
            print(
                "am3_mojo: AM3_MOJO_THREADS=",
                s,
                " is below 1; using ",
                n,
                " threads",
                sep="",
            )
        except:
            print(
                "am3_mojo: AM3_MOJO_THREADS=",
                s,
                " is not a whole number; using ",
                n,
                " threads",
                sep="",
            )
    return n


__async def _task[
    T: AnyType, work: def(Pointer[T, MutUntrackedOrigin], Int, Int) thin -> None
](ctx: Pointer[T, MutUntrackedOrigin], t: Int, nt: Int):
    work(ctx, t, nt)


def parallel_run[
    T: AnyType,
    //,
    work: def(Pointer[T, MutUntrackedOrigin], Int, Int) thin -> None,
](mut ctx: T, nthreads: Int):
    """`work` partitions the job by t. The calling thread takes its share
    instead of waiting idle. This also spares the wake-up of a pool thread,
    most of the cost of a short section. `ctx` is borrowed for the duration of
    the call. So the borrow checker keeps it alive while the workers use the
    (untracked) pointer to it."""
    var p = Pointer(to=ctx).unsafe_origin_cast[MutUntrackedOrigin]()
    if nthreads <= 1:
        work(p, 0, 1)
        return
    var tg = TaskGroup()
    for t in range(1, nthreads):
        tg.create_task(_task[T, work](p, t, nthreads))
    work(p, 0, nthreads)
    tg.wait()


struct _TaskCounter[T: AnyType]:
    var ctx: Pointer[Self.T, MutUntrackedOrigin]
    var next: Atomic[Int]
    var ntasks: Int

    def __init__(
        out self, ctx: Pointer[Self.T, MutUntrackedOrigin], ntasks: Int
    ):
        self.ctx = ctx
        self.next = Atomic[Int](0)
        self.ntasks = ntasks


def _take_tasks[
    T: AnyType, work: def(Pointer[T, MutUntrackedOrigin], Int) thin -> None
](c: Pointer[_TaskCounter[T], MutUntrackedOrigin], t: Int, nt: Int):
    var k = c[].next.fetch_add(1)
    while k < c[].ntasks:
        work(c[].ctx, k)
        k = c[].next.fetch_add(1)


def parallel_tasks[
    T: AnyType,
    //,
    work: def(Pointer[T, MutUntrackedOrigin], Int) thin -> None,
](mut ctx: T, ntasks: Int, nthreads: Int):
    """Each thread takes the next task when it is done with one, so that a
    slower thread takes fewer. Each task must write only its own outputs: the
    results do not depend on which thread runs it."""
    var c = _TaskCounter[T](
        Pointer(to=ctx).unsafe_origin_cast[MutUntrackedOrigin](), ntasks
    )
    parallel_run[_take_tasks[T, work]](c, min(nthreads, ntasks))
