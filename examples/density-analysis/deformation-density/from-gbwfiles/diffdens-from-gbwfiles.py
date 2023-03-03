from ash import *
from ash.functions.functions_elstructure import  read_cube, write_cube_diff

#########################################
# Deformation density from ORCA GBW FILES
##########################################
#Use this script if you have ORCA GBW for a promolecular density (created via orca_mergefrag) 
# and the final density 
#And you want to get the deformation density
#Warning: geometry needs to be the same

#Define names of GBW files (should be in directory)
file1="promolecular.gbw"
file2="final.gbw"

#Run orca_plot to request electron density creation from ORCA gbw file
run_orca_plot(file1, "density", gridvalue=80)
#Run orca_plot to request electron density creation from ORCA gbw file
run_orca_plot(file2, "density", gridvalue=80)

#Read Cubefiles from disk
cube_data1 = read_cube(f"{file1.split('.')[0]}.eldens.cube")
cube_data2 = read_cube(f"{file2.split('.')[0]}.eldens.cube")

#Write out difference density as a Cubefile
write_cube_diff(cube_data1, cube_data2, "deformation_density")

