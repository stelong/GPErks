include ./Makefile.vars

.DEFAULT_GOAL := default

default: clean install

clean:
	rm -rf .mypy_cache/ .pytest_cache/ .tox/ build/ coverage_html/ dist/ GPErks.egg-info/ .coverage coverage.xml mypy_result.xml
.PHONY: clean

uninstall:
	pip uninstall GPerks
.PHONY: install

install:
	pip install .
.PHONY: install

install-dev:
	pip install .[${PY_EXTRA_DEV}]
.PHONY: install-dev

