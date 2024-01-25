from ash import *
from math import isclose


#Read solvated PDB-file, create OpenMMTheory job, define pyscftheory, create QM/MM and run SP
def test_qm_mm_pyscf_openmm():

    numcores=2
    #Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment or PDB-file)
    pdbfile=f"{ashpath}/tests/pdbfiles/1aki_solvated.pdb"
    fragment=Fragment(pdbfile=pdbfile)

    #Creating new OpenMM object from OpenMM full system file
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"], pdbfile=pdbfile, periodic=True,
            numcores=numcores, autoconstraints=None, rigidwater=False)
    #QM
    qmatomlist = [1013,1014,1015,1016,1017,1018]
    qm = PySCFTheory(scf_type="RKS",  functional="BP86", basis="def2-SVP", densityfit=True)
    #qm = xTBTheory()
    # Create QM/MM OBJECT
    qmmmobject = QMMMTheory(qm_theory=qm, mm_theory=omm, qm_charge=-1, qm_mult=1,
        fragment=fragment, embedding="Elstat", qmatoms=qmatomlist, printlevel=2)

    Singlepoint(theory=qmmmobject, fragment=fragment, Grad=True)

#test_qm_mm_pyscf_openmm()
