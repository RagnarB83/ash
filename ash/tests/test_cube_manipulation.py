import numpy as np
import pytest

from ash.functions.functions_elstructure import cube_planar_profile, read_cube


def write_test_cube(path):
    path.write_text(
        "Test cube\n"
        "Electron density\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.5 0.0\n"
        "2 0.5 1.0 0.0\n"
        "2 0.0 0.0 2.0\n"
        "1 1.0 0.0 0.0 0.0\n"
        "1.0 2.0 3.0 4.0 5.0 6.0\n"
        "7.0 8.0\n",
        encoding="utf-8",
    )


def test_read_cube_preserves_full_voxel_vectors_and_uses_full_voxel_volume(tmp_path, capsys):
    cubefile = tmp_path / "skewed.cube"
    write_test_cube(cubefile)

    cube = read_cube(str(cubefile))

    np.testing.assert_allclose(
        cube["grid_vectors"],
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 2.0]],
    )
    assert "Integrated volume: 54.0" in capsys.readouterr().out


@pytest.mark.parametrize(
    "axis, expected",
    [
        ("x", [[0.0, 10.0 * np.sqrt(5.0)],
               [1.5 / np.sqrt(5.0), 26.0 * np.sqrt(5.0)]]),
        ("y", [[0.0, 14.0 * np.sqrt(5.0)],
               [1.5 / np.sqrt(5.0), 22.0 * np.sqrt(5.0)]]),
        ("z", [[0.0, 12.0], [2.0, 15.0]]),
    ],
)
def test_cube_planar_profile_integrates_skewed_voxels(tmp_path, axis, expected):
    cubefile = tmp_path / "skewed.cube"
    write_test_cube(cubefile)

    profile = cube_planar_profile(read_cube(str(cubefile)), axis=axis)

    np.testing.assert_allclose(profile, expected)


def test_cube_planar_profile_rejects_unknown_axis():
    with pytest.raises(ValueError, match="axis must"):
        cube_planar_profile({}, axis="q")


def test_cube_planar_profile_rejects_incomplete_values():
    cube = {
        "nx": 2,
        "ny": 2,
        "nz": 2,
        "vals": [[1.0, 2.0]],
        "grid_vectors": np.eye(3),
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "orgx": 0.0,
        "orgy": 0.0,
        "orgz": 0.0,
    }

    with pytest.raises(ValueError, match="expected 8"):
        cube_planar_profile(cube)
