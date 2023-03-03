from ash import *
from ash.functions.functions_elstructure import  read_cube, write_cube_diff

#########################################
# Difference density from CUBE files
##########################################
#Use this script if you have Cube-files already calculatd
#And you want to get the difference density
#Warning: Cube files must have prepared equivalently


file1="calc1.eldens.cube"
file2="calc2.eldens.cube"

#Read Cubefiles from disk.
cube_data1 = read_cube(file1)
cube_data2 = read_cube(file2)

#Write out difference density as a Cubefile
write_cube_diff(cube_data1, cube_data2, "diffence_density")

