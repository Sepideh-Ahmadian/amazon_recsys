# DOMAIN is optional; the CLI falls back to default_domain in configs/base.yaml
DOMAIN ?=
DOMAIN_FLAG = $(if $(DOMAIN),--domain $(DOMAIN),)
EXP ?= mf
SEED ?= 0
LIMIT ?=
LIMIT_FLAG = $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: data enrich judge calibrate split train report test test-all test-integration lint

data:
	recsys ingest $(DOMAIN_FLAG)
	recsys filter $(DOMAIN_FLAG)

enrich:
	recsys enrich $(DOMAIN_FLAG) $(LIMIT_FLAG)

judge:
	recsys judge $(DOMAIN_FLAG) $(LIMIT_FLAG)

calibrate:
	recsys calibrate $(DOMAIN_FLAG) export

split:
	recsys interactions $(DOMAIN_FLAG)
	recsys split $(DOMAIN_FLAG)

train:
	recsys train --exp $(EXP) --seed $(SEED)

report:
	python scripts/make_report.py

test:
	pytest -m "not slow and not integration" --cov=amazon_recsys

test-all:
	pytest -m "not integration" --cov=amazon_recsys

test-integration:
	RUN_INTEGRATION=1 pytest -m integration

lint:
	ruff check .
	ruff format --check .
