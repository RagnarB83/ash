# Cube planar profiles

ASH provides `cube_planar_profile` for extracting the plane-integrated profile
along a cube grid axis. The returned columns are the plane coordinate and the
integrated cube data. For electron-density cubes, values are electrons per
coordinate-length unit from the input file. The calculation uses all three
voxel vectors, including skewed grids.

Run `planar_average.py density.cube` to print the profile along the z grid
index. In Python, call `ash.cube_planar_profile(ash.read_cube("density.cube"),
axis="z")` to obtain the values as an array.
