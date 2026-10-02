# Validation record

This repository was validated after the end-to-end refactor with the following checks.

## Unit tests

```text
6 passed
```

The tests cover text cleaning/model shapes plus deterministic hash encoding, synthetic voxel construction and checkpoint-based inference.

## Offline end-to-end smoke test

Command:

```bash
python -m scripts.run_smoke --samples 6 --epochs 1
```

Observed result:

```text
epoch=1/1 g_loss=8.1954 d_loss=0.5032
SMOKE TEST PASSED
manifest=data/smoke/training_manifest.csv
checkpoint=artifacts/smoke/generator_checkpoint.pt
mesh=artifacts/smoke/generated_cube.obj
```

The smoke run proves that dataset construction, feature persistence, training, checkpoint serialisation, checkpoint loading, inference and mesh export are connected.

## Validation boundary

The real Objaverse/T5 execution path was not exercised in this packaging environment because the optional Objaverse/Open3D dependencies and external model/data downloads were not available here. The code path is implemented, but it should be validated on the target development machine before claiming a completed real-data experiment.
