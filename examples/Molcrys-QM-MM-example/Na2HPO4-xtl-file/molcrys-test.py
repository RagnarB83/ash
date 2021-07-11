from ash import *

#######################
# MOLCRYS INPUT          #
#######################
xtl_file="na2hpo4-baldus1995-1.xtl"
sphereradius=12

#Number of cores available to ASH. Used by QM-code or ASH.
numcores=1

#Theory level for charge iterations
orcadir='/Applications/orca_4_2_1_macosx_openmpi314'
orcasimpleinput="! BP86 def2-SVP def2/J Grid5 Finalgrid6 tightscf"
orcablocks="%scf maxiter 200 end"
ORCAcalc = ORCATheory(
        orcadir=orcadir,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        nprocs=numcores,
        )

#Chargemodel. Options: CHELPG, Hirshfeld, CM5, NPA, Mulliken
chargemodel='Hirshfeld'
#Shortrange model. Usually Lennard-Jones. Options: UFF_all, UFF_modH
shortrangemodel='UFF_modH'

#Define fragment types in crystal: Descriptive name, formula, charge and multiplicity
mainfrag = Fragmenttype("Phosphate","PO4H", charge=-2,mult=1)
counterfrag1 = Fragmenttype("Sodium1","Na", charge=1,mult=1)
#Define list of fragmentobjects. Passed on to molcrys
fragmentobjects = [
        mainfrag,
        counterfrag1,
        ]

#Modify global connectivity settings (scale and tol keywords)
settings_ash.settings_dict["scale"]=1.0
settings_ash.settings_dict["tol"]=0.3
# Modified radii to assist with connectivity.
#Setting radius of Na to almost 0. Na will then not bond
eldict_covrad['Na']=0.0001
print(eldict_covrad)


#Calling molcrys function and define Cluster object
Cluster = molcrys(
        xtl_file=xtl_file,
        fragmentobjects=fragmentobjects,
        theory=ORCAcalc,
        auto_connectivity=True,
        numcores=numcores,
        clusterradius=sphereradius,
        chargemodel=chargemodel,
        shortrangemodel=shortrangemodel,
        )


######### Second Section: QMMM Geometry Opt ###################################
#Once molcrys is done we have a Cluster object (named Cluster) in memory and also printed to disk as Cluster.ygg
# We can then do optimization right here using that Cluster object.
#Alternatively or for restart purposes we can read a Cluster object into a separate QM/MM Opt job like this:
#Cluster=Fragment(fragfile='Cluster.ygg')
print("Now Doing Optimization")

# Defining Centralmainfrag (a list of atoms) for optimization. Can be done in multiple ways:
#Centralmainfrag=fragmentobjects[0].clusterfraglist[0]
#Read list of atom indices from file (created by molcrys): Centralmainfrag = read_intlist_from_file("Centralmainfrag")
#Can also be done manually: Centralmainfrag=[0, 1, 5, 8, 9, 12, 14]
#Easiest way:
Centralmainfrag = Cluster.Centralmainfrag
print("Centralmainfrag:", Centralmainfrag)

charge=fragmentobjects[0].Charge
mult=fragmentobjects[0].Mult
#
Cluster_FF=MMforcefield_read('Cluster_forcefield.ff')

#Defining, QM, MM and QM/MM theory levels for Optimization
#If same theory as used in molcrys, then orcadir, orcasimpleinput and orcablocks can be commented out/deleted.
orcasimpleinput="! BP86 def2-SVP def2/J Grid5 Finalgrid6 tightscf"
orcablocks="%scf maxiter 200 end"
ORCAQMpart = ORCATheory(
        orcadir=orcadir,
        charge=charge,
        mult=mult,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        )
MMpart = NonBondedTheory(
        charges = Cluster.atomcharges,
        atomtypes=Cluster.atomtypes,
        forcefield=Cluster_FF,
        LJcombrule='geometric',
        )
QMMM_object = QMMMTheory(
        fragment=Cluster,
        qm_theory=ORCAQMpart,
        mm_theory=MMpart,
        actatoms=Centralmainfrag,
        qmatoms=Centralmainfrag,
        charges=Cluster.atomcharges,
        embedding='Elstat',
        nprocs=numcores,
        )

geomeTRICOptimizer(
        theory=QMMM_object,
        fragment=Cluster,
        coordsystem='tric',
        maxiter=170,
        ActiveRegion=True,
        actatoms=Centralmainfrag,
        )
