.PHONY: install test schema collect validate build-v3 build-v4

install:
	python -m pip install -r requirements.txt

test:
	pytest -q

schema:
	python collector.py --schema-only

collect:
	python collector.py --players-per-division 25 --matches-per-player 10

validate:
	python validate_collection.py

build-v3:
	python build_behavior_dataset.py

build-v4:
	python run_v4_pipeline.py --skip-v3-build
