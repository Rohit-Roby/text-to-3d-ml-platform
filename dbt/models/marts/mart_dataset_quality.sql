select
  category,
  count(*) as asset_count,
  countif(description is null or trim(description) = '') as missing_descriptions,
  countif(voxel_path is null) as missing_voxels,
  countif(embedding_path is null) as missing_embeddings
from {{ ref('stg_assets') }}
group by 1
