PY := .venv/bin/python

all: results test

.venv:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

results: .venv
	$(PY) run.py

test: .venv
	$(PY) -m pytest -q

refresh: .venv
	$(PY) run.py --refresh

.PHONY: all results test refresh
