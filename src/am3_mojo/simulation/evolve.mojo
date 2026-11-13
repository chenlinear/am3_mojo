"""The `Evolution` rows of a group are independent (distinct solvers and
densities), so they run in parallel on the Mojo worker pool.
"""

from am3_mojo.solver.solver import SolverAlgorithm
from am3_mojo.utils.buffers import Vec, WD
from am3_mojo.utils.parallel import parallel_tasks

comptime VPtr = Pointer[Vec, MutUntrackedOrigin]
comptime SPtr = Pointer[SolverAlgorithm, MutUntrackedOrigin]


@inline(.always)
def vp(ref v: Vec) -> VPtr:
    return (
        Pointer(to=v)
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


@fieldwise_init
struct Evolution(Copyable, ImplicitlyCopyable, Movable):
    var solver: Int
    var target: Int
    var a: Int
    var A: Int
    var q: Int
    var nphot: Int


struct EvolveJob(Copyable, Movable):
    """The inputs are read-only, and `n` is updated."""

    var sa: SPtr
    var a: VPtr
    var A: VPtr
    var q: VPtr
    var n: VPtr
    var failed: Bool
    var msg: String

    def __init__(out self, sa: SPtr, a: VPtr, A: VPtr, q: VPtr, n: VPtr):
        self.sa = sa
        self.a = a
        self.A = A
        self.q = q
        self.n = n
        self.failed = False
        self.msg = String()


def _work(jobs: Pointer[List[EvolveJob], MutUntrackedOrigin], k: Int):
    ref J = jobs[][k]
    try:
        J.sa[].evolve(J.a[], J.A[], J.q[], J.n[])
    except e:
        J.failed = True
        J.msg = String(e)


def run_jobs(mut jobs: List[EvolveJob], nthreads: Int) raises:
    parallel_tasks[_work](jobs, len(jobs), nthreads)
    for i in range(len(jobs)):
        if jobs[i].failed:
            raise Error(
                String("error in SolverAlgorithm.evolve: ", jobs[i].msg)
            )


struct CoolJob(Copyable, Movable):
    var sa: SPtr
    var n: VPtr
    var b: Float64

    def __init__(out self, sa: SPtr, n: VPtr, b: Float64):
        self.sa = sa
        self.n = n
        self.b = b


struct CoolTasks(Movable):
    var jobs: Pointer[List[CoolJob], MutUntrackedOrigin]
    var on: List[Int]
    var nch: Int

    def __init__(
        out self,
        jobs: Pointer[List[CoolJob], MutUntrackedOrigin],
        var on: List[Int],
        nch: Int,
    ):
        self.jobs = jobs
        self.on = on^
        self.nch = nch


def _cool_work(tasks: Pointer[CoolTasks, MutUntrackedOrigin], k: Int):
    ref T = tasks[]
    ref J = T.jobs[][T.on[k // T.nch]]
    var nb8 = J.sa[].N_BINS // WD
    var c = k % T.nch
    var i0 = c * nb8 // T.nch * WD
    var i1 = (c + 1) * nb8 // T.nch * WD if c + 1 < T.nch else J.sa[].N_BINS
    J.sa[].compute_syn_ic(J.b, J.n[], i0, i1)


def run_cooling(mut jobs: List[CoolJob], nthreads: Int):
    """Compute the cooling rates of distinct solvers in parallel. Each task
    writes only the rates of its own bins."""
    var on = List[Int]()
    for i in range(len(jobs)):
        if jobs[i].sa[].start_cooling(jobs[i].b, jobs[i].n[]):
            on.append(i)
    if len(on) == 0:
        return
    var nch = max(1, nthreads // len(on))
    var ntasks = len(on) * nch
    var tasks = CoolTasks(
        Pointer(to=jobs).unsafe_origin_cast[MutUntrackedOrigin](), on^, nch
    )
    parallel_tasks[_cool_work](tasks, ntasks, nthreads)
