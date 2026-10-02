from src.preprocessing.synthetic import make_voxel_shape


def test_synthetic_shape_has_expected_dimensions():
    voxel = make_voxel_shape("sphere", size=32)
    assert voxel.shape == (32, 32, 32)
    assert voxel.max() == 1.0
    assert voxel.min() == 0.0
