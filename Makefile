# am3_mojo: build, test, compare with the original C++ AM3, benchmark.
# If .venv is missing, for example after `make clean`, make runs `uv sync`.

PY     := .venv/bin/python
MOJO   := .venv/bin/mojo
RUFF   := .venv/bin/ruff
SRC    := $(shell find src -name '*.mojo')
MODULE := build/am3_mojo.so
ASSERT_MODULE := build-asserts/am3_mojo.so
BINDINGS := src/am3_mojo_module.mojo

.PHONY: help build bindings install format reference test-unit test-reference \
        test-asserts test benchmark plots literature_checks all clean FORCE
# a module that fails its import check is deleted, not kept as up to date
.DELETE_ON_ERROR:
# The targets build on each other in the order that `all` lists them and share
# the module and results/. So they run one at a time, also under make -j.
.NOTPARALLEL:

help:
	@echo "make build              build the am3_mojo Python module (build/am3_mojo.so)"
	@echo "make install            make 'import am3_mojo' work anywhere in the .venv"
	@echo "make format             format the Mojo (mojo format) and the Python (ruff format)"
	@echo "make reference          download and build the original C++ AM3 (reference)"
	@echo "make test-unit          am3_mojo tests (no C++ needed)"
	@echo "make test-reference     compare am3_mojo with the C++ reference"
	@echo "make test-asserts       the unit tests on a build with bounds checks (slower)"
	@echo "make test               all tests (reference tests are skipped if not built)"
	@echo "make benchmark          am3_mojo vs C++ AM3 timings and per-stage profiles"
	@echo "make plots              comparison, kernel-reuse and added-process figures"
	@echo "make literature_checks  the checks of tests/literature_checks/ (outputs in results/literature_checks/)"
	@echo "make all                format + build + reference + test + benchmark + plots + literature_checks"
	@echo "make clean              delete the builds, the C++ reference, the caches, results/literature_checks/ and .venv"
	@echo "results go to results/"

$(PY):
	uv sync
$(MOJO) $(RUFF): | $(PY)

build: $(MODULE)
	@$(PY) -W ignore::SyntaxWarning -m compileall -q -j 0 .venv/lib

# also rebuilt when `uv sync` installs another Mojo (it rewrites $(MOJO))
$(MODULE): $(SRC) $(BINDINGS) $(MOJO)
	@mkdir -p build
	$(MOJO) build -D ASSERT=none --emit shared-lib -I src \
		src/am3_mojo_module.mojo -o $(MODULE)
	@$(PY) -c "import sys; sys.path.insert(0, 'build'); import am3_mojo; \
		print('built', '$(MODULE)', 'am3_mojo', am3_mojo.__version__)"

# The bindings are regenerated when they were not generated from the current
# generator and API descriptions (`gen_bindings.py --check`, run by every
# build). `make bindings` regenerates them unconditionally.
$(BINDINGS): FORCE | $(PY)
	@$(PY) tools/gen_bindings.py --check || $(PY) tools/gen_bindings.py
FORCE:

bindings: | $(PY)
	$(PY) tools/gen_bindings.py

install: build
	@$(PY) -c "import site, os; p = os.path.join(site.getsitepackages()[0], 'am3_mojo.pth'); \
		open(p, 'w').write(os.path.abspath('build') + '\n'); print('wrote', p)"

format: | $(MOJO) $(RUFF)
	$(MOJO) format -q $(shell find src tests -name '*.mojo')
	$(RUFF) format

reference: | $(PY)
	tests/reference/setup_reference.sh

test-unit: build
	$(PY) -m pytest tests/unit

test-reference: build
	$(PY) -m pytest tests/reference

$(ASSERT_MODULE): $(SRC) $(BINDINGS) $(MOJO)
	@mkdir -p build-asserts
	$(MOJO) build -D ASSERT=all --emit shared-lib -I src \
		src/am3_mojo_module.mojo -o $(ASSERT_MODULE)

test-asserts: $(ASSERT_MODULE)
	AM3_MOJO_BUILD=build-asserts AM3_MOJO_ASSERT=all $(PY) -m pytest tests/unit

test: build
	$(PY) -m pytest tests

benchmark: build
	$(PY) tests/benchmark/benchmark.py
	$(PY) tests/benchmark/profile_stages.py --scenario blazar_simple
	$(PY) tests/benchmark/profile_stages.py --scenario leptohadronic_all

plots: build
	$(PY) tests/plots/plot_spectra.py
	$(PY) tests/plots/plot_reuse.py
	$(PY) tests/plots/plot_added_processes.py

literature_checks: build
	$(PY) tests/literature_checks/run_literature_checks.py

all: format build reference test benchmark plots literature_checks

# A Finder window on the folder can write .DS_Store into .venv while rm deletes
# .venv ("Directory not empty"). The second rm removes what Finder wrote.
clean:
	rm -rf build build-asserts results/runs* results/reuse* results/literature_checks tests/reference/work .pytest_cache .ruff_cache
	rm -rf .venv 2>/dev/null || rm -rf .venv
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
