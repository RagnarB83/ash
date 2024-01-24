from ash import *
from ash.functions.functions_elstructure import  read_cube, write_cube_diff

#########################################
# Difference density from scratch
##########################################
#Use this script if you want to calculate difference density
#from 2 new calculations

fragment1=Fragment(xyzfile="hf.xyz", charge=0, mult=1)

#------------------
#Calculation 1
#------------------
label1="calc1"
calc1=ORCATheory(orcasimpleinput="! BP86 def2-SVP tightscf notrah",filename=label1)

result_calc1=Singlepoint(theory=calc1, fragment=fragment1)
#Run orca_plot to request electron density creation from ORCA gbw file
run_orca_plot(label1+".gbw", "density", gridvalue=80)


#------------------
#Calculation 2
#Vertical ionization (same geometry
#------------------

label2="calc2"
calc2=ORCATheory(orcasimpleinput="! BP86 def2-SVP tightscf notrah",filename=label2)

result_calc2=Singlepoint(theory=calc2, fragment=fragment1, charge=1, mult=2)
#Run orca_plot to request electron density creation from ORCA gbw file
run_orca_plot(label2+".gbw", "density", gridvalue=80)

file1="file1.gbw"
file2="file2.gbw"


#Read Cubefiles from disk
cube_data1 = read_cube(f"{label1}.eldens.cube")
cube_data2 = read_cube(f"{label2}.eldens.cube")

#Write out difference density as a Cubefile
write_cube_diff(cube_data1, cube_data2, "diffence_density")

