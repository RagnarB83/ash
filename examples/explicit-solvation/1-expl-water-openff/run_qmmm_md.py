from ash import *

#Read in coordinates of the full system
#Note that for QM/MM you must use a box where the molecule is centered. Re-image the file using MDtraj_imagetraj if necessary
pdbfile="equilibration_NPT_imaged.pdb"
fragment = Fragment(pdbfile=pdbfile)
#Create an OpenMMTheory object based on PDB-file and XML-files for water and small-molecule
omm =OpenMMTheory(xmlfiles=["openff_LIG.xml", "amber/tip3p_standard.xml"],
            pdbfile=pdbfile, periodic=True, rigidwater=True, autoconstraints='HBonds')
#Create a QM/MM object
qm = xTBTheory(xtbmethod='GFN2')
#Defining QM-atoms to be the solute.  Note that the atom indices are 0-based
qmatomlist = list(range(0,13))
#QM/MM from QM and MM objects. Setting QM-region charge and multiplicity
qm_mm = QMMMTheory(qm_theory=qm, mm_theory=omm, fragment=fragment, qmatoms=qmatomlist,
        qm_charge=0, qm_mult=1)

#Run a NVT MD simulation (NPT could also be performed if you add a barostat)
#Note: timesteps for QM/MM must be much smaller than in MM
OpenMM_MD(fragment=fragment, theory=qm_mm, timestep=0.001, simulation_time=10, traj_frequency=10,
    temperature=300, integrator='LangevinMiddleIntegrator', coupling_frequency=1,
    trajfilename='QM_MM_NVT-MD',trajectory_file_option='DCD')
