"""The tooling: the build that tests/support/paths.py takes for the default
one, the options of the scripts, the files they write, the benchmark's
timings, the source archive and the phase space of the triplet Monte Carlo."""

import hashlib, importlib.util, os, subprocess, sys, tarfile, time, warnings

import numpy as np
import pytest

from support import scenarios
from support.paths import ROOT, SUPPORT, TESTS, use_tool
from support.runs import save_npz
from support.time_scenario import Timed


@pytest.mark.parametrize(
    "build, suffix",
    [(None, ""), ("build", ""), ("build-other", "-build-other")],
)
def test_build_that_am3_mojo_build_names(build, suffix):
    """AM3_MOJO_BUILD=build names the default build, as no AM3_MOJO_BUILD
    does. The raw outputs of this build are those that the figures read (no
    suffix of the cache folders). The build can write the published results.
    Another folder is a variant. (The variable alone made a variant:
    `AM3_MOJO_BUILD=build make plots` stopped, because results/plots held "the
    default build's results".)"""
    env = {k: v for k, v in os.environ.items() if k != "AM3_MOJO_BUILD"}
    if build is not None:
        env["AM3_MOJO_BUILD"] = build
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(1, sys.argv[1]);"
            " from support.paths import CACHE_SUFFIX;"
            " print(repr(CACHE_SUFFIX))",
            TESTS,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == repr(suffix)


def test_run_scenario_every_without_steps(tmp_path):
    """run_scenario.py stops with a usage error for --every without --steps.
    (It dropped the option and wrote the scenario's own checkpoints.)"""
    out = tmp_path / "run.npz"
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(SUPPORT, "run_scenario.py"),
            "--impl",
            "mojo",
            "--scenario",
            "ssc_external",
            "--out",
            str(out),
            "--every",
            "10",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2
    assert "--every needs --steps" in r.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    "options",
    [
        ("--steps", "0"),
        ("--steps", "-3"),
        ("--steps", "4", "--every", "0"),
        ("--steps", "4", "--every", "-1"),
    ],
)
def test_run_scenario_step_counts_below_one(options, tmp_path):
    """run_scenario.py stops with a usage error for a --steps or --every
    below 1. (--steps 0 counted as no option and ran the scenario's own
    steps. --steps -3 wrote an empty file. --every 0 counted as no option,
    and --every -1 never ended.)"""
    out = tmp_path / "run.npz"
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(SUPPORT, "run_scenario.py"),
            "--impl",
            "mojo",
            "--scenario",
            "ssc_external",
            "--out",
            str(out),
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 2
    assert "is not 1 or more" in r.stderr
    assert not out.exists()


def test_evolve_with_checkpoints_every_step_count_below_one():
    """scenarios.evolve raises for checkpoints every 0 or -1 steps, before
    the first step. (It never ended: the steps done went down.)"""
    for every in (0, -1):
        with pytest.raises(ValueError, match="1 or more"):
            next(scenarios.evolve(None, 4, every))


def test_save_npz_replaces_the_file_whole(tmp_path):
    """A reader that opened the old file still reads the old arrays from it. A
    new reader gets the new arrays, and no other file is left. (The files were
    rewritten in place. A run that was stopped left a part of a file, which
    counted as current. A reader of the old file then met a bad CRC.)"""
    out = str(tmp_path / "run.npz")
    old = np.arange(200000.0)
    save_npz(out, a=old, b=old[::-1])
    reader = np.load(out)  # (reads an array when asked for it)
    save_npz(out, a=np.zeros(3))
    assert np.array_equal(reader["a"], old)
    assert np.array_equal(reader["b"], old[::-1])
    new = np.load(out)
    assert new.files == ["a"] and np.array_equal(new["a"], np.zeros(3))
    assert os.listdir(tmp_path) == ["run.npz"]


def test_literature_checks_without_a_worker():
    """run_literature_checks.py stops with a usage error for -j 0. (It
    stopped with a traceback of the thread pool.)"""
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(
                TESTS, "literature_checks", "run_literature_checks.py"
            ),
            "-j",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 2
    assert "is not 1 or more" in r.stderr


def test_triplet_phase_space_divides_by_no_zero():
    """The phase space of the triplet Monte Carlo (tools/triplet/kinematics.py)
    gives events with a forward recoil at the largest pair mass without a
    warning, from k = 1e10 to 1e14. (Its backward form of the recoil's
    light-cone momentum was computed for every event and divided by zero for
    these. `make literature_checks` then printed a RuntimeWarning into the
    output of machinery.py.)"""
    use_tool("triplet")
    from kinematics import generate

    u = np.full((50, 4), 0.5)
    u[:, 0] = np.linspace(0.0, 1.0, 50)
    u[:, 1] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for k in (1e10, 1e12, 1e14):
            assert generate(u, k)[0].sum() > 40, k


class Slow:
    def init_kernels(self):
        time.sleep(0.02)

    set_mag_field = evolve_step = lambda self, *args: time.sleep(0.02)


def test_benchmark_set_up_counts_the_field_set_before_the_steps():
    """The benchmark's set-up holds init_kernels, the set_mag_field calls
    before the first of the steps and the first evolve_step. A field set
    during the run counts with the steps. (set_mag_field was not timed. The
    C++ builds its kernels of the field there, 1.4 % of its set-up of
    leptohadronic_all, and am3_mojo builds them in the first step.)"""
    t = Timed(Slow())
    t.init_kernels()
    t.set_mag_field(1.0)
    t.evolve_step()
    assert 0.06 <= t.t_init < 0.2 and t.t_evolve == 0 and t.steps == 0
    t.evolve_step()
    t.set_mag_field(2.0)
    assert 0.06 <= t.t_init < 0.2 and 0.04 <= t.t_evolve < 0.2
    assert t.steps == 1


def load(path):
    spec = importlib.util.spec_from_file_location(
        os.path.basename(path)[: -len(".py")], path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_without_the_reference_writes_no_results(
    tmp_path, monkeypatch, capsys
):
    """Without the C++ reference, benchmark.py times am3_mojo and leaves
    benchmark.json and benchmark.md, which hold the comparison, as they are.
    (It stopped with a KeyError after the timings, and the json that it
    wrote before the error lacked the C++ runs.)"""
    benchmark = load(os.path.join(TESTS, "benchmark", "benchmark.py"))
    timed = []

    def time_scenario(impl, scenario, repeats, env):
        timed.append(impl)
        return dict(init=[1.0], evolve=[1.0], steps=10, max_rss=1e9)

    monkeypatch.setattr(
        benchmark, "available", lambda impl: impl != "cpp-fixed"
    )
    monkeypatch.setattr(benchmark, "published_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(benchmark, "time_scenario", time_scenario)
    benchmark.main()
    assert set(timed) == {"mojo", "mojo-am3"}
    assert "not written" in capsys.readouterr().out
    assert os.listdir(tmp_path) == []


def test_sdist_is_the_same_from_the_same_sources(tmp_path):
    """Two source archives built from the same files are the same bytes.
    All entries have the same time and no owner. Of tools/, the archive holds
    the build backend and the inputs of the bindings. (The entries had the
    times of the files and the login of the builder. The metadata had the
    time 0, and the gzip header had the time of the build. The archive also
    held the scripts that extract tables from papers.)"""
    backend = load(os.path.join(ROOT, "tools", "mojo_build_backend.py"))
    digests = set()
    for k in range(2):
        folder = tmp_path / str(k)
        folder.mkdir()
        path = folder / backend.build_sdist(str(folder))
        digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(digests) == 1
    with tarfile.open(path) as t:
        members = t.getmembers()
    assert {(m.uid, m.gid, m.uname, m.gname) for m in members} == {
        (0, 0, "", "")
    }
    assert len({m.mtime for m in members}) == 1 and members[0].mtime > 0
    tools = sorted(
        m.name.split("/tools/")[1] for m in members if "/tools/" in m.name
    )
    assert tools == [
        "am3_api.json",
        "am3_mojo_api.json",
        "gen_bindings.py",
        "mojo_build_backend.py",
    ]


def test_reference_set_up_in_a_folder_with_a_space(tmp_path):
    """setup_reference.sh stops at once, with the reason, for a work
    directory whose path holds a space, and makes nothing there. (It failed
    in the build of GSL, after the downloads, with libtool's message.)"""
    work = tmp_path / "a b"
    r = subprocess.run(
        ["bash", os.path.join(TESTS, "reference", "setup_reference.sh")],
        capture_output=True,
        text=True,
        env={**os.environ, "AM3_REFERENCE_DIR": str(work)},
        timeout=120,
    )
    assert r.returncode == 1
    assert "holds a space" in r.stdout
    assert not work.exists()
