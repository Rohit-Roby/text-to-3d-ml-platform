select
  cast(uid as string) as asset_id,
  lower(category) as category,
  description,
  voxel_path,
  embedding_path
from {{ source('raw', 'training_manifest') }}
where uid is not null
