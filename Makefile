# am3_mojo: build, test, compare with the original C++ AM3, benchmark.
# Run `uv sync` once first (creates .venv with Mojo, numpy, pytest, ...).

PY     := .venv/bin/python
MOJO   := .venv/bin/mojo
SRC    := $(shell find src -name '*.mojo')
MODULE := build/am3_mojo.so

.PHONY: help build bindings install reference test test-unit test-reference \
        benchmark plots all clean distclean

help:
	@echo "make build           build the am3_mojo Python module (build/am3_mojo.so)"
	@echo "make install         make 'import am3_mojo' work anywhere in the .venv"
	@echo "make test-unit       am3_mojo tests (no C++ needed)"
	@echo "make reference       download and build the original C++ AM3 (reference)"
	@echo "make test-reference  compare am3_mojo with the C++ reference"
	@echo "make test            all tests (reference tests are skipped if not built)"
	@echo "make benchmark       am3_mojo vs C++ AM3 timings and per-stage profiles"
	@echo "make plots           comparison and kernel-reuse figures"
	@echo "make all             build + reference + test + benchmark + plots"
	@echo "results go to results/"

build: $(MODULE)

$(MODULE): $(SRC)
	@mkdir -p build
	$(MOJO) build -D ASSERT=none --fp-mode contract=off --emit shared-lib -I src \
		src/am3_mojo_module.mojo -o $(MODULE)
	@$(PY) -c "import sys; sys.path.insert(0, 'build'); import am3_mojo; \
		print('built', '$(MODULE)', 'am3_mojo', am3_mojo.__version__)"

# regenerate the Python bindings (only needed after changing tools/gen_bindings.py)
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

benchmark: build
	$(PY) tests/benchmark/benchmark.py
	$(PY) tests/benchmark/profile_stages.py --scenario blazar_simple
	$(PY) tests/benchmark/profile_stages.py --scenario leptohadronic_all

plots: build
	$(PY) tests/plots/plot_spectra.py
	$(PY) tests/plots/plot_reuse.py

all: build reference test benchmark plots

clean:
	rm -rf build results/runs results/reuse .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf tests/reference/work
