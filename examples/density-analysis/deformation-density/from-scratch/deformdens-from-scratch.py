from ash import *
from ash.functions.functions_elstructure import  read_cube, write_cube_diff

#########################################
# Deformation density from scratch
##########################################
#Use this script if you want to calculate deformation density from scratch


#Fragments
AB_molecule=Fragment(xyzfile="hf.xyz", charge=0, mult=1)
A_molecule=Fragment(xyzfile="h.xyz", charge=0, mult=2)
B_molecule=Fragment(xyzfile="f.xyz", charge=0, mult=2)

#General theory level information
orca_simple="! UKS BP86 def2-SVP tightscf notrah"
orca_blocks=""

#-------------------------
#Calculation on AB
#------------------------
labelAB="calc_AB"
calc_AB=ORCATheory(orcasimpleinput=orca_simple, orcablocks=orca_blocks,filename=labelAB)

result_calcAB=Singlepoint(theory=calc_AB, fragment=AB_molecule)
#Run orca_plot to request electron density creation from ORCA gbw file
run_orca_plot(labelAB+".gbw", "density", gridvalue=80)


#-------------------------
#Calculation on A
#------------------------
labelA="calc_A"
calc_A=ORCATheory(orcasimpleinput=orca_simple, orcablocks=orca_blocks,filename=labelA)

result_calcA=Singlepoint(theory=calc_A, fragment=AB_molecule)
#Run orca_plot to request electron density creation from ORCA gbw file
run_orca_plot(labelA+".gbw", "density", gridvalue=80)

#-------------------------
#Calculation on B
#------------------------
labelB="calc_B"
calc_B=ORCATheory(orcasimpleinput=orca_simple, orcablocks=orca_blocks,filename=labelB)

result_calcB=Singlepoint(theory=calc_B, fragment=B_molecule)
#Run orca_plot to request electron density creation from ORCA gbw file
run_orca_plot(labelB+".gbw", "density", gridvalue=80)

#-----------------------------------------
# merge A + B to get promolecular density
#-----------------------------------------

p = sp.run(['orca_mergefrag', labelA+".gbw", labelB+".gbw", "promolecule_AB.gbw"], encoding='ascii')

#Get density
run_orca_plot("promolecule_AB.gbw", "density", gridvalue=80)

#-----------------------------------------
# Make deformation density as difference
#-----------------------------------------

#Read Cubefiles from disk
cube_data1 = read_cube("promolecule_AB.eldens.cube")
cube_data2 = read_cube(f"{labelAB}.eldens.cube")

#Write out difference density as a Cubefile
write_cube_diff(cube_data1, cube_data2, "deformation_density")

