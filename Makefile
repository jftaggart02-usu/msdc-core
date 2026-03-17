venv:
	python3 -m venv .venv

install:
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .

lint:
	.venv/bin/pylint msdc_core || PYLINT_FAIL=1; \
	.venv/bin/mypy msdc_core || MYPY_FAIL=1; \
	test -z "$$PYLINT_FAIL$$MYPY_FAIL"

test:
	.venv/bin/pytest tests -v; \
	STATUS=$$?; \
	if [ $$STATUS -ne 0 ] && [ $$STATUS -ne 5 ]; then \
		exit $$STATUS; \
	fi
