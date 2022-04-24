include ./Makefile.vars

.DEFAULT_GOAL := default

default: clean install

.PHONY: clean
clean:
	rm -rf .mypy_cache/ .pytest_cache/ .tox/ build/ coverage_html/ dist/ GPErks.egg-info/ .coverage coverage.xml mypy_result.xml

.PHONY: uninstall
uninstall:
	pip uninstall GPerks

.PHONY: install
install:
	pip install .

.PHONY: install-dev
install-dev:
	pip install .[${PY_EXTRA_DEV}]

.ONESHELL:
.PHONY: install-macos
install-macos:
	conda env create -f ./requirements_osx_arm64/environment.yml
	$(CONDA_ACTIVATE) gperks
	pip install --no-deps .

.PHONY: uninstall-macos
uninstall-macos:
	conda remove --name gperks --all
