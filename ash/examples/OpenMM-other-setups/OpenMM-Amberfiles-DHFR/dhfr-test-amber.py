from ash import *
numcores = 8

forcefielddir="."
inpcrdfile=forcefielddir+"dhfr.pbc.rst7"
prmtopfile=forcefielddir+"dhfr.pbc.parm7"

frag=Fragment(amber_prmtopfile=prmtopfile, amber_inpcrdfile=inpcrdfile)

#Periodic OpenMM on DHFR system using Amber files. Reference data from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549999/

openmmobject = OpenMMTheory(Amberfiles=True, amberprmtopfile=prmtopfile,
    periodic=True, autoconstraints='HBonds', dispersion_correction=True, periodic_nonbonded_cutoff=10)


OpenMM_MD(theory=openmmobject, fragment=frag, integrator="LangevinMiddleIntegrator", traj_frequency=10, coupling_frequency=1,  simulation_time=10)
