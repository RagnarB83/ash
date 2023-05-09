from ash import *

biasdir="./biasdirectory"

frag = Fragment(databasefile="butane.xyz", charge=0, mult=1)

theory = xTBTheory(runmode='library')

OpenMM_metadynamics(fragment=frag, theory=theory, timestep=0.001, 
              simulation_time=10,
              traj_frequency=1, temperature=300, integrator='LangevinMiddleIntegrator',
              CV1_atoms=[0,1,2,3], CV1_type='dihedral', biasfactor=6, height=1, 
              CV1_biaswidth=0.25,
              frequency=1, savefrequency=1,
              biasdir=biasdir)

# Free-energy plot
metadynamics_plot_data(biasdir=biasdir, dpi=200, imageformat='png')
