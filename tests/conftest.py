import pytest


@pytest.fixture(scope="session")
def textured_glb(tmp_path_factory):
    """A textured, UV-mapped sphere at (0, 2, 0) exported as GLB."""
    pytest.importorskip("trimesh")
    from make_fixtures import quadrant_sphere_glb

    path = tmp_path_factory.mktemp("glb") / "model.glb"
    path.write_bytes(quadrant_sphere_glb())
    return path
