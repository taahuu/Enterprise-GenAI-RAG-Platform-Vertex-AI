.PHONY: install test ingest eval run

install:
	pip install -r requirements-dev.txt

test:            ## offline, no model downloads
	python -m pytest -q

ingest:          ## index data/sample_docs with the configured (Hugging Face) models
	python -m scripts.ingest

eval:
	python -m scripts.evaluate

run:
	uvicorn app.main:app --reload --port 8000
