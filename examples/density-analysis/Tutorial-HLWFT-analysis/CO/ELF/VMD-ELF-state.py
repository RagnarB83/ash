from ash import *

cubefiles=glob.glob("*cube")

#Optional VMD state-file
write_VMD_script_cube(cubefiles=cubefiles,VMDfilename="ELF.vmd",
                          isovalue=0.8, isosurfacecolor_pos="blue", isosurfacecolor_neg="red")
