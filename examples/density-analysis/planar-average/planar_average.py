"""Print the z-axis planar profile of an electron-density cube file."""

import sys

import ash


if len(sys.argv) != 2:
    raise SystemExit("Usage: python planar_average.py density.cube")

cube = ash.read_cube(sys.argv[1])
profile = ash.cube_planar_profile(cube, axis="z")
for coordinate, plane_integral in profile:
    print(f"{coordinate:16.8f} {plane_integral:20.10f}")
