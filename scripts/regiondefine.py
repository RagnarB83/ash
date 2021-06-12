#Simple script to define QM-region and Active region for QM/MM system
#Note: It's only recommended to define the active-region once to create the actatoms file.
#Do not redefine the active region for every new calculation

from ash import *

#Forcefield files
forcefielddir="/home/bjornsson/ASH-vs-chemshell-protein/QM-MM/FeMoco-FULL-BOX/"
topfile=forcefielddir+"top_all36_prot.rtf"
parfile=forcefielddir+"par_all36_prot.prm"
psffile="new-box-delH2o.psf"

#Define fragment (can be XYZ-file, PDBfile, GROfile, ygg-file)
frag = Fragment(pdbfile="new-box-delH2o.pdb", conncalc=False)

#Creating OpenMMobject (in order to get residue information)
openmmobject = OpenMMTheory(psffile=psffile, CHARMMfiles=True, charmmtopfile=topfile, charmmprmfile=parfile)

###########################################
# QM-region definition and book-keeping
###########################################
#Note: ASH-indexing starts at 0
femoco=[31648,31649,31650,31651,31652,31653,31654,31655,31656,31657,31658,31659,31660,31661,31662,31663,31664,31665]
hca= [31714,31715, 31716, 31717, 31718, 31719, 31720, 31721, 31722, 31723, 31724, 31725, 31726, 31727, 31728, 31729, 31730, 31731, 31732, 31733, 31734]
cys275 = [4175,4176,4177,4178]
his442 = [6912,6913,6914,6915,6916,6917,6918,6919,6920,6921,6922]
qmatoms = sorted(femoco + hca + cys275 + his442)
writelisttofile(qmatoms, "qmatoms")

print("qmatoms:", qmatoms)
print("QM-region size:", len(qmatoms))

#Print XYZ file for QM-region to check if correct
module_coords.write_XYZ_for_atoms(frag.coords,frag.elems, qmatoms, "QMRegion")
print("Wrote QM region XYZfile: QMRegion.xyz  (inspect with visualization program)")
############################################

############################################
# Active region definition
# (approx. spherical region around an atom)
############################################

actatoms = actregiondefine(mmtheory=openmmobject, fragment=frag, radius=11, originatom=31665)
print("actatoms:", actatoms)