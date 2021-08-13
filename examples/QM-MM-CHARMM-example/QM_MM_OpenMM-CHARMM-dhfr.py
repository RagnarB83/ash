from ash import *

numcores=1

forcefielddir="./"
psffile=forcefielddir+"step3_pbcsetup.psf"
topfile=forcefielddir+"top_all36_prot.rtf"
prmfile=forcefielddir+"par_all36_prot.prm"
xyzfile=forcefielddir+"coordinates.xyz"

#Read coordinates from XYZ-file
frag = Fragment(xyzfile=xyzfile)

#Creating OpenMM object
openmmobject = OpenMMTheory(psffile=psffile, CHARMMfiles=True, charmmtopfile=topfile,
    charmmprmfile=prmfile, periodic=True, charmm_periodic_cell_dimensions=[80.0, 80.0, 80.0, 90.0, 90.0, 90.0], do_energy_decomposition=True)


#Creating ORCATheory object
orcadir="/Applications/orca_4_2_1_macosx_openmpi314"
ORCAinpline="! HF-3c tightscf"
ORCAblocklines="""
%maxcore 2000
"""
#Create ORCA QM object. Attaching numcores so that ORCA runs in parallel
orcaobject = ORCATheory(orcadir=orcadir, charge=0,mult=1, orcasimpleinput=ORCAinpline,
                        orcablocks=ORCAblocklines, numcores=numcores)

#act and qmatoms lists. Defines QM-region (atoms described by QM) and Active-region (atoms allowed to move)
#IMPORTANT: atom indices begin at 0.
#Here selecting the side-chain of threonine
qmatoms = [569,570,571,572,573,574,575,576]
actatoms = qmatoms


# Create QM/MM OBJECT by combining QM and MM objects above
qmmmobject = QMMMTheory(qm_theory=orcaobject, mm_theory=openmmobject, printlevel=2,
                        fragment=frag, embedding="Elstat", qmatoms=qmatoms)

#Run geometry optimization using geomeTRIC optimizer and HDLC coordinates. Using active region.
geomeTRICOptimizer(theory=qmmmobject, fragment=frag, ActiveRegion=True, actatoms=actatoms,
                    maxiter=500, coordsystem='hdlc')
