from ash import *
from math import isclose


#Read solvated PDB-file, create OpenMMTheory job and run MM singlepoint
def test_openmm_basic():

    numcores=2
    #Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment or PDB-file)
    pdbfile=f"{ashpath}/tests/pdbfiles/1aki_solvated.pdb"
    fragment=Fragment(pdbfile=pdbfile)

    #Creating new OpenMM object from OpenMM full system file
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"], pdbfile=pdbfile, periodic=True,
            numcores=numcores, autoconstraints=None, rigidwater=False)
    #Singlepoint MM energy
    Singlepoint(theory=omm, fragment=fragment, Grad=True)

#Read raw PDB-file, fix using pdbfixer, setup using Modeller and optimize
def test_openmm_modeller():

    pdbfile=f"{ashpath}/tests/pdbfiles/1aki.pdb"

    #Setting up new system, adding hydrogens, solvent, ions and defining forcefield, topology
    openmmobject, ashfragment = OpenMM_Modeller(pdbfile=pdbfile, forcefield='CHARMM36', watermodel="tip3p", pH=7.0,
        solvent_padding=10.0, ionicstrength=0.1, platform='OpenCL')

    #MM minimization to get rid the worst contacts
    OpenMM_Opt(fragment=ashfragment, theory=openmmobject, maxiter=100, tolerance=1000)
    
#test_openmm_basic()
#test_openmm_modeller()
