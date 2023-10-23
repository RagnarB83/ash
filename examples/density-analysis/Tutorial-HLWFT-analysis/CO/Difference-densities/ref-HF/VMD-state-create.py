from ash import *

diff_cubefiles=glob.glob("*diff_density.cube")

#Optional VMD state-file
write_VMD_script_cube(cubefiles=diff_cubefiles,VMDfilename="Diffdens.vmd",
                          isovalue=0.001, isosurfacecolor_pos="blue", isosurfacecolor_neg="red")
