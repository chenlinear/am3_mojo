"""Table-driven evolution of particle species.

Each `Evolution` row names a solver, the density it advances in place and the
coefficient arrays it uses. The rows of a group are independent (distinct
solvers and densities), so they run in parallel on the Mojo worker pool.
"""

from am3_mojo.utils.parallel import parallel_run
from am3_mojo.utils.buffers import Vec
from am3_mojo.solver.solver import SolverAlgorithm

comptime VPtr = Pointer[Vec, MutUntrackedOrigin]
comptime SPtr = Pointer[SolverAlgorithm, MutUntrackedOrigin]


@always_inline
def vp(ref v: Vec) -> VPtr:
    return (
        Pointer(to=v)
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


@fieldwise_init
struct Evolution(Copyable, ImplicitlyCopyable, Movable):
    """One species update, with cooling rates from the photon density
    `n[nphot]`: `n[target] <- solver.evolve(c[a], c[A], c[q], n[target])`."""

    var solver: Int
    var target: Int
    var a: Int
    var A: Int
    var q: Int
    var nphot: Int


struct EvolveJob(Copyable, Movable):
    """A row resolved to pointers (inputs are read-only, `n` is updated)."""

    var sa: SPtr
    var a: VPtr
    var A: VPtr
    var q: VPtr
    var n: VPtr
    var failed: Bool

    def __init__(out self, sa: SPtr, a: VPtr, A: VPtr, q: VPtr, n: VPtr):
        self.sa = sa
        self.a = a
        self.A = A
        self.q = q
        self.n = n
        self.failed = False


def _work(jobs: Pointer[List[EvolveJob], MutUntrackedOrigin], t: Int, nt: Int):
    var k = t
    while k < len(jobs[]):
        ref J = jobs[][k]
        try:
            J.sa[].evolve(J.a[], J.A[], J.q[], J.n[])
        except:
            J.failed = True
        k += nt


def run_jobs(mut jobs: List[EvolveJob], nthreads: Int) raises:
    """Run independent jobs, in parallel when more than one thread is allowed.
    """
    var nt = min(nthreads, len(jobs))
    parallel_run[_work](jobs, nt)
    for i in range(len(jobs)):
        if jobs[i].failed:
            raise Error("error in SolverAlgorithm.evolve")
