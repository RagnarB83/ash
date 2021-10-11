from ash import *
#######################
# MOLCRYS INPUT          #
#######################
cif_file="/Users/bjornsson/ownCloud/ASH-tests/testsuite/bf3hcn-fromxtl.cif"
sphereradius=15

#Number of cores available for either ORCA/xTB parallelization or multiprocessing
numcores=1

#Charge-iteration QMinput. Using xTB theory. Uses xTB default charges
#Defining QM theory without fragment, charge or mult
xtbdir='/opt/xtb-6.4.0/bin'
xtbmethod='GFN2'
xtbcalc = xTBTheory(xtbdir=xtbdir, xtbmethod=xtbmethod, runmode='inputfile', numcores=numcores)


#Chargemodel options: CHELPG, Hirshfeld, CM5, NPA, Mulliken
chargemodel='Hirshfeld'
#Shortrange model. Usually Lennard-Jones. Options: UFF, 
shortrangemodel='UFF'

#Define fragment types in crystal: Descriptive name, formula, charge and mult
#TODO: Have alternative for formula being nuccharge, as a backup. Maybe mass too (rounded to 1 amu)
mainfrag = Fragmenttype("BF3HCN","BF3HCN", charge=0,mult=1)
#counterfrag1 = Fragmenttype("Sodium","Na", charge=1,mult=1)
#Define list of fragmentobjects. Passed on to molcrys 
fragmentobjects=[mainfrag]

#Define global system settings (currently scale and tol keywords for connectivity)
settings_ash.scale=1.0
settings_ash.tol=0.1
# Modified radii to assist with connectivity.
#Setting radius of Na to 0. Na will not bond
#eldict_covrad['Na']=0.0001
#eldict_covrad['H']=0.15
#print(eldict_covrad)


#Calling molcrys function and define Cluster object
Cluster = molcrys(cif_file=cif_file, fragmentobjects=fragmentobjects, theory=xtbcalc, 
        numcores=numcores, clusterradius=sphereradius, chargemodel=chargemodel, shortrangemodel=shortrangemodel, auto_connectivity=True)


print("Cluster.atomtypes", Cluster.atomtypes)

try:
    print("Cluster:", Cluster)
except:
    print("No Cluster object found. Reading from file")
    Cluster=Fragment(fragfile='Cluster.ygg')        

#Once molcrys is done we have a Cluster object (named Cluster) in memory and also printed to disk
# We can then do optimization right here using Cluster object. 
#Alternatively or for restart purposes we can read Cluster object into a separate QM/MM Opt job.
#READ fragmentobjects also from file. Must have charge and mult attributes
#Something similar to mainfrag-info.txt / counterfrag-info.txt but in one file ???

#Optimization
print("Now Doing Optimization")
# Defining Centralmainfrag (list of atoms) for optimization
#Can also be done manually I guess
Centralmainfrag=fragmentobjects[0].clusterfraglist[0]
#Centralmainfrag=[0, 4, 10, 14, 15, 16, 78, 333, 7100, 7101, 7102, 7105, 7107, 7108, 7110, 7111, 7112, 7116, 7117, 7118, 7178, 7179, 7180, 7762]
print("Centralmainfrag:", Centralmainfrag)

charge=fragmentobjects[0].Charge
mult=fragmentobjects[0].Mult
#
Cluster_FF=MMforcefield_read('Cluster_forcefield.ff')

#Theory level for Optimization
xtbmethod='GFN2'
xtbcalc = xTBTheory(xtbdir=xtbdir, runmode='inputfile', numcores=numcores, charge=charge, mult=mult, xtbmethod=xtbmethod)
MMpart = NonBondedTheory(charges = Cluster.atomcharges, atomtypes=Cluster.atomtypes, forcefield=Cluster_FF, LJcombrule='geometric')
print("MMpart:", MMpart.__dict__)
QMMMobject = QMMMTheory(fragment=Cluster, qm_theory=xtbcalc, mm_theory=MMpart, 
    qmatoms=Centralmainfrag, charges=Cluster.atomcharges, embedding='Elstat', numcores=numcores)


geomeTRICOptimizer(theory=QMMMobject, fragment=Cluster, maxiter=70, ActiveRegion=True, actatoms=Centralmainfrag )

#Update charges??


sys.exit(0)
