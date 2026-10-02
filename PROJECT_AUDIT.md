# Original Notebook Audit

## Strong concepts already present
- Objaverse/LVIS ingestion
- 3D GLB mesh processing with Open3D
- voxelisation and conversion to PyTorch tensors
- T5-based text representations
- 3D GAN and conditional GAN architectures
- marching-cubes voxel-to-mesh conversion
- exploratory NeRF implementation

## Problems corrected in the refactor
- Colab/Google Drive paths coupled to notebook runtime
- dataframe description creation is commented out before later use
- category-specific annotations and UID slices can become misaligned
- `o3d_voxelize()` is invoked without its required filepath in one cell
- expensive T5 inference is performed from Dataset `__getitem__`
- conditional training creates `generated_samples` from random latent values instead of the matching text embedding
- duplicated class names and weight-init functions
- training/evaluation split and quantitative 3D metrics are absent
- model/data lineage is not recorded
- NeRF section references undefined/incomplete components and does not use a real multi-view image/camera dataset

## Refactor decision
Use the text-conditioned 3D voxel GAN as the main ML system and keep NeRF as an optional future research module. This makes the project coherent, demonstrable, and technically defensible in interviews.
