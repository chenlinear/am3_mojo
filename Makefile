# am3_mojo: build, test, compare with the original C++ AM3, benchmark.
# Run `uv sync` once first (creates .venv with Mojo, numpy, pytest, ...).

PY     := .venv/bin/python
MOJO   := .venv/bin/mojo
SRC    := $(shell find src -name '*.mojo')
MODULE := build/am3_mojo.so
# the same module with Mojo's assertions and bounds checks on (make test-asserts)
ASSERT_MODULE := build-asserts/am3_mojo.so
# the Python bindings, generated from the API descriptions in tools/
BINDINGS := src/am3_mojo_module.mojo

.PHONY: help build bindings install reference test test-unit test-reference \
        test-asserts benchmark plots checks all clean distclean

help:
	@echo "make build           build the am3_mojo Python module (build/am3_mojo.so)"
	@echo "make install         make 'import am3_mojo' work anywhere in the .venv"
	@echo "make test-unit       am3_mojo tests (no C++ needed)"
	@echo "make reference       download and build the original C++ AM3 (reference)"
	@echo "make test-reference  compare am3_mojo with the C++ reference"
	@echo "make test            all tests (reference tests are skipped if not built)"
	@echo "make test-asserts    the unit tests on a build with bounds checks (slower)"
	@echo "make benchmark       am3_mojo vs C++ AM3 timings and per-stage profiles"
	@echo "make plots           comparison, kernel-reuse and added-process figures"
	@echo "make checks          the literature and solver checks of tests/ (outputs in build/checks/)"
	@echo "make all             build + reference + test + benchmark + plots + checks"
	@echo "results go to results/"

build: $(MODULE)

$(MODULE): $(SRC) $(BINDINGS)
	@mkdir -p build
	$(MOJO) build -D ASSERT=none --emit shared-lib -I src \
		src/am3_mojo_module.mojo -o $(MODULE)
	@$(PY) -c "import sys; sys.path.insert(0, 'build'); import am3_mojo; \
		print('built', '$(MODULE)', 'am3_mojo', am3_mojo.__version__)"

# the bindings are regenerated when the generator or an API description changes;
# `make bindings` regenerates them unconditionally
$(BINDINGS): tools/gen_bindings.py tools/am3_api.json tools/am3_mojo_api.json
	$(PY) tools/gen_bindings.py

bindings:
	$(PY) tools/gen_bindings.py

install: build
	@$(PY) -c "import site, os; p = os.path.join(site.getsitepackages()[0], 'am3_mojo.pth'); \
		open(p, 'w').write(os.path.abspath('build') + '\n'); print('wrote', p)"

reference:
	tests/reference/setup_reference.sh

test-unit: build
	$(PY) -m pytest tests/unit

test-reference: build
	$(PY) -m pytest tests/reference

test: build
	$(PY) -m pytest tests

$(ASSERT_MODULE): $(SRC) $(BINDINGS)
	@mkdir -p build-asserts
	$(MOJO) build -D ASSERT=all --emit shared-lib -I src \
		src/am3_mojo_module.mojo -o $(ASSERT_MODULE)

test-asserts: $(ASSERT_MODULE)
	AM3_MOJO_BUILD=build-asserts AM3_MOJO_ASSERT=all $(PY) -m pytest tests/unit

benchmark: build
	$(PY) tests/benchmark/benchmark.py
	$(PY) tests/benchmark/profile_stages.py --scenario blazar_simple
	$(PY) tests/benchmark/profile_stages.py --scenario leptohadronic_all

plots: build
	$(PY) tests/plots/plot_spectra.py
	$(PY) tests/plots/plot_reuse.py
	$(PY) tests/plots/plot_added_processes.py

# every script of tests/literature_checks and tests/solver_checks (tests/run_checks.py)
checks: build
	$(PY) tests/run_checks.py

all: build reference test benchmark plots checks

clean:
	rm -rf build build-asserts results/runs* results/reuse* .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf tests/reference/work
