install:
	python -m pip install -r requirements.txt

smoke:
	python -m scripts.run_smoke --samples 9 --epochs 1

build-demo:
	python pipelines/build_dataset.py --source synthetic --limit 24

build-objaverse:
	python pipelines/build_dataset.py --source objaverse --category chair --limit 50

train-demo:
	python -m src.training.train_cgan --manifest data/processed/training_manifest.parquet --epochs 5 --encoder-backend hash --no-mlflow

train:
	python -m src.training.train_cgan --manifest data/processed/training_manifest.parquet --epochs 20 --encoder-backend t5

test:
	pytest -q

lint:
	ruff check src pipelines scripts tests

serve:
	uvicorn src.inference.api:app --reload
