from ash import *
from ash.functions.functions_elstructure import  read_cube, write_cube_diff

#########################################
# Difference density from ORCA GBW FILES
##########################################
#Use this script if you have ORCA GBW and density files for 2 states already calculated
#And you want to get the difference density

#Warning: the filename.densities files typically has to be available as well
#even though orca_plot can create the density without the density file this does not work when electron count differs


file1="calc1.gbw"
file2="calc2.gbw"

#Run orca_plot to request electron density creation from ORCA gbw file
run_orca_plot(file1, "density", gridvalue=80)
#Run orca_plot to request electron density creation from ORCA gbw file
run_orca_plot(file2, "density", gridvalue=80)

#Read Cubefiles from disk
cube_data1 = read_cube(f"{file1.split('.')[0]}.eldens.cube")
cube_data2 = read_cube(f"{file2.split('.')[0]}.eldens.cube")

#Write out difference density as a Cubefile
write_cube_diff(cube_data1, cube_data2, "diffence_density")

