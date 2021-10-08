from ash import *
numcores = 8

forcefielddir="/Users/bjornsson/ownCloud/ASH-tests/testsuite/OpenMM-files-for-tests/dhfr/charmm/"
psffile=forcefielddir+"step3_pbcsetup.psf"
topfile=forcefielddir+"top_all36_prot.rtf"
prmfile=forcefielddir+"par_all36_prot.prm"
xyzfile=forcefielddir+"file.xyz"

frag = Fragment(xyzfile=xyzfile, conncalc=False)

#Periodic OpenMM on DHFR system using CHARMM files. Reference data from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549999/
#Periodic cell according to CHARMM inputfile, PBE parameters according to CHARMM settings also
#No long-range dispersion used
#Nonbonded cutoff: 12 Å, switching function: 10 Å
openmmobject = OpenMMTheory(CHARMMfiles=True, psffile=psffile, charmmtopfile=topfile,
    charmmprmfile=prmfile, periodic=True, charmm_periodic_cell_dimensions=[80, 80, 80, 90, 90, 90], 
    dispersion_correction=False, periodic_nonbonded_cutoff=12, switching_function_distance=10,
    PMEparameters=[1.0/0.34, 90, 90, 90])


OpenMM_MD(theory=openmmobject, fragment=frag, integrator="LangevinMiddleIntegrator", traj_frequency=10, coupling_frequency=1,  simulation_time=10)
