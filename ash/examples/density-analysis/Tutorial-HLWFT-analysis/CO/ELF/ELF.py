from ash import *
#This requires https://github.com/keelan542/AutoELF from Keelan
#Auto-ElF does core and valence assignment of  the nuclear attractors and adds their positions to the cubefile for visualization
from ash.external.additional_python_modules import AutoELF

moldenfiles=glob.glob("*molden*")
#Looping over Moldenfiles to do ELF analysis via Multiwfn
for moldenfile in moldenfiles:
    cubefile = multiwfn_run(moldenfile, option='elf', grid=3, numcores=1)
    #Auto-ELF assignment (requires library)
    AutoELF.auto_elf_assign(cubefile, "attractors.pdb", interest_atoms=[0,1])
