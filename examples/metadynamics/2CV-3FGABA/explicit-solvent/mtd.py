from ash import *

numcores=12
qmcores=1
biasdir="/home/321.5-COMX/321.5.1-Commun/ragnar/ASH-metadynamics/3fgaba/tutorial/QM-MM/100ps-MTD-timesteps/1fs/biasdirectory"

#System
xyzfile="system_aftersolvent.xyz"
pdbfile="system_aftersolvent.pdb"
frag = Fragment(xyzfile=xyzfile, charge=0, mult=1)
qmatoms = list(range(0,16))


#Define QM, MM and QM/MM Theory
qm_theory = xTBTheory(runmode='inputfile', printlevel=0, numcores=qmcores)
mm_theory = OpenMMTheory(xmlfiles=[f"{ashpath}/databases/forcefields/tip3p_water_ions.xml", "LIG.xml"], 
    pdbfile=pdbfile, platform='CPU', rigidwater=True, printlevel=0, numcores=qmcores, periodic=True)
qm_mm_theory = QMMMTheory(qm_theory=qm_theory, mm_theory = mm_theory, qmatoms=qmatoms, fragment=frag, printlevel=0)


#Call metadynamics
OpenMM_metadynamics(fragment=frag, theory=qm_mm_theory, timestep=0.001,
              simulation_time=500, printlevel=0, enforcePeriodicBox=False,
              traj_frequency=100, temperature=300, integrator='LangevinMiddleIntegrator',
              CV1_type="torsion", CV1_atoms=[1,3,4,5],
              CV2_type="torsion", CV2_atoms=[3,4,5,6],
              biasfactor=6, height=1,
              CV1_biaswidth=0.5, CV2_biaswidth=0.5,
              frequency=10, savefrequency=10,
              biasdir=biasdir)

