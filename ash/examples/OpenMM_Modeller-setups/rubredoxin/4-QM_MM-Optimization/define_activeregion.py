from ash import *

#Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment, PDB-file)
lastpdbfile="final_MDfrag_laststep_imaged.pdb"
fragment=Fragment(pdbfile=lastpdbfile)

#Creating new OpenMM object from OpenMM XML files (built-in CHARMM36 and a user-defined one)
omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", "./specialresidue.xml"], pdbfile=lastpdbfile, periodic=True,
            platform='CPU',  autoconstraints=None, rigidwater=False)


#Defining active region as within X Å from originatom 755 (Fe)
actregiondefine(mmtheory=omm, fragment=fragment, radius=12, originatom=755)
