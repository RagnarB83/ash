from yggdrasill import *
import settings_yggdrasill
from functions_MM import *
########################################################
# YGGDRASILL - A GENERAL COMPCHEM AND QM/MM ENVIRONMENT#
########################################################
print_yggdrasill_header()
#TODO: Write Psi4 interface
#Todo: Write Dalton interface

#Define global system settings ( scale, tol and conndepth keywords for connectivity)
settings_yggdrasill.init() #initialize

print(settings_yggdrasill.scale)
print(settings_yggdrasill.tol)
#General program
path="/Users/bjornssonsu/ownCloud/PyQMMM-project/Yggdrasill-testdir"
os.chdir(path)


#GITHUB test2

#Define optionalXYZ coordinates as multi-line string
#fragcoords="""
#H 0.0 0.0 0.0
#F 0.0 0.0 1.0
#"""

#FRAGMENT CREATION.

#Option 1.
#Step 1. Creation of empty fragment
#HF_frag=Fragment()
#print("HF_frag dict", HF_frag.__dict__)
#Add coordinates to fragment
#HF_frag.coords_from_string(fragcoords)
#print("")
#print("HF_frag dict", HF_frag.__dict__)
#print("-------------")

#Option 2.
#Direct creation of new fragment with coords
#HF_frag2=Fragment(coordsstring=fragcoords)
#print("HF_frag2 dict", HF_frag2.__dict__)
#print("-------------")

#Option 3
#Create fragment from xyzfile via initialization and then run read_xyz class-function
#HF_frag3=Fragment()
#HF_frag3.read_xyzfile("hf.xyz")
#print("HF_frag3 dict", HF_frag3.__dict__)
#print("-------------")

#Option 4
#Create fragment from xyzfile directly
#HF_frag4 = Fragment(xyzfile="hf.xyz")
#print("HF_frag4 dict", HF_frag4.__dict__)
#print("-------------")

#
#Option 5
#Replace coords and elems of fragment object with new lists
#elems=['H', 'Cl']
#coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
#HF_frag5=Fragment()
#HF_frag5.replace_coords(elems,coords)
#print("HF_frag5 dict", HF_frag5.__dict__)
#print("-------------")

#Option 6:
#Add  atoms to fragment multiple times
#Empty fragment
#print("Fragment defined")
#HCl_H2O=Fragment()
#elems=['H', 'Cl']
#coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
#HCl_H2O.add_coords(elems,coords)
#elems=['O', 'H', 'H']
#coords=[[2.172092667, -1.008295059, 2.571252403],[1.454071996, -0.674986676, 2.018792653],
#        [2.237951295, -1.940936718, 2.331039759]]
#HCl_H2O.add_coords(elems,coords)
#print("HCl_H2O dict", HCl_H2O.__dict__)
#print("-------------")
#HCl_H2O.print__coords()

#Option 7. Add molecule from XYZ and calculate connectivity:
#FeFeH2ase = Fragment(xyzfile="fefhe2ase.xyz")
#print("FeFeH2ase dict", FeFeH2ase.__dict__)
#FeFeH2ase.calc_connectivity()
#print("FeFeH2ase dict", FeFeH2ase.__dict__)
#conn = FeFeH2ase.connectivity
#print("conn:", conn)
#print("Number of subfragments in FeFeH2ase", len(conn))
#print("Number of atoms in FeFeH2ase", FeFeH2ase.numatoms)
#print("Number atoms in connectivity in FeFeH2ase", FeFeH2ase.connected_atoms_number)

################################
# SINGLE-POINT QM CALCULATIONS #
################################
#ORCA example.
orcadir="/Applications/orca_4.2.1"
xtbdir="/Applications/xtb"
#Define input
orcasimpleinput="! BP86 def2-SVP Grid5 Finalgrid6 tightscf"
orcablocks="%scf maxiter 200 end"
xtbmethod='GFN2-xTB'
#ORCASPcalculation = ORCATheory(orcadir=orcadir, fragment=HF_frag5, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
#xTBSPcalculation = xTBTheory(xtbdir=xtbdir, fragment=HF_frag5, charge=0, mult=1, xtbmethod=xtbmethod)

#Simple Energy SP calc
#ORCASPcalculation.run()
#Energy+Gradient calculation
#ORCASPcalculation.run(Grad=True)

#energy=ORCASPcalculation.energy
#blankline()
#xTBSPcalculation.run()
#Todo: Write BS functionality into ORCATheory
# Todo. Add extra basis functionality

###########################################
# POINT-CHARGE EMBEDDED QM-SP CALCULATION #
###########################################
#TODO: Add ORCA extrabasis-feature on specific atoms
#Todo: Add embedding ECP atoms

#HCl_vdw = Fragment(xyzfile="HCl-H2O-vdw.xyz")

#QM atoms: H and Cl. Rest (H2O) is MM atoms
#qmatoms=[0,1]
# Charge definitions for whole fragment.
#Manual definition
#atomcharges=[0.4, -0.3, -0.8, 0.4, 0.4]

#MM forcefield defined as dict:
#MM_forcefield={}

#For atomtype OT, defining both charge and LJ parameters
#MM_forcefield = {'OT':AtomMMobject(atomcharge=3.0, LJparameters=[0.15, 2.7])}


# Defining only LJ parameters. charge may come from list instead
#These are atom-specific parameters. Pair parameters calculated automatically from these
#First sigma_i, then epsilon.
#Note: sigma = 2**(1/6) * Rmin

#Read from file
#MM_forcefield=MMforcefield_read('forcefield.ff')


#Will probably never define things like this but good to keep as example
#MM_forcefield['OT'] = AtomMMobject(LJparameters=[2.77, 2.7])
#MM_forcefield['HT'] = AtomMMobject(LJparameters=[0.0, 0.0])
#MM_forcefield['CLX'] = AtomMMobject(LJparameters=[3.16, 1.08])
#MM_forcefield['HX'] = AtomMMobject(LJparameters=[0.00, 0.00])
#atomtypeslist=['HX','CLX','OT','HT','HT']

#PointchargeMMtheory = Theory( atom_types=atomtypes force_field=MMdefinition)
#MyMMtheory = NonBondedTheory(charges = atomcharges, atomtypes=atomtypeslist, LJ=False, forcefield=MM_forcefield)

#e,g=MyMMtheory.run(coords=HCl_vdw.coords, charges=atomcharges)
#Creating ORCA theory object without fragment information
#ORCAQMtheory = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

#Create QM/MM theory object
#without MM theory but with atomcharges
#QMMM_SP_ORCAcalculation = QMMMTheory(fragment=HCl_H2O, qm_theory=ORCAQMtheory, qmatoms=qmatoms, atomcharges=atomcharges)
#QMMM_SP_ORCAcalculation.run(Grad=True)

#with MM theory
#print(HCl_H2O.__dict__)

#New_HCl_H2O = Fragment(xyzfile="HCl-H2O.xyz")
#New_HCl_H2O.print__coords()
#QMMM_SP_ORCAcalculation_2 = QMMMTheory(fragment=New_HCl_H2O, qm_theory=ORCAQMtheory,
 #                                      mm_theory=MyMMtheory, qmatoms=qmatoms, atomcharges=atomcharges,
 #                                      embedding='Elstat')

#QMMM_SP_ORCAcalculation_2.run(Grad=True)

#############################
# QM/MM GEOMETRY OPTIMIZATION
############################
orcasimpleinput="! BP86 MINIX Grid4 Finalgrid5 D3BJ tightscf"
orcablocks="%scf maxiter 200 end"
H2O_MeOH = Fragment(xyzfile="h2o_MeOH.xyz")
#H2O_MeOH = Fragment(xyzfile="h2o_MeOH-chemsh-optimized.xyz")
#Calculate connectivity.
# TODO: Make use default??

H2O_MeOH.calc_connectivity(scale=settings_yggdrasill.scale,tol=settings_yggdrasill.tol)

print(H2O_MeOH.__dict__)
print(H2O_MeOH.connectivity)
#Defining optimization object

#QM atoms: H and Cl. Rest (H2O) is MM atoms
qmatoms=[3,4,5,6,7,8]
# Charge definitions for whole fragment.
atomcharges=[-0.8, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#MM info
atomtypes=['OT','HT','HT','CX','HX', 'HX', 'HX', 'OT', 'HT']
#Read from file
MM_forcefield=MMforcefield_read('forcefield.ff')
print(MM_forcefield.items())
for atomtype,property in MM_forcefield.items():
    print("Atomtype:", atomtype)
    print("Charge:", property.atomcharge)
    print("LJParameters:", property.LJparameters)
    blankline()



#Creating ORCA theory object without fragment information
ORCAQMtheory = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

#PointchargeMMtheory = Theory( atom_types=atomtypes force_field=MMdefinition)
MyMMtheory = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, LJ=False, forcefield=MM_forcefield, LJcombrule='geometric')


#Create QM/MM theory object. fragment always defined with it
QMMM_SP_ORCAcalculation_2 = QMMMTheory(fragment=H2O_MeOH, qm_theory=ORCAQMtheory,
                                       mm_theory=MyMMtheory, qmatoms=qmatoms, atomcharges=atomcharges,
                                       embedding='Elstat')

#QMMM_SP_ORCAcalculation_2.run()
#exit()
#Frozen atoms
#geomeTRICOptimizer(QMMM_SP_ORCAcalculation_2,New_HCl_H2O, frozenatoms=[2,3,4])

#coordsystem options: 'tric', 'hdlc', 'dlc', 'prim', 'cart'
#Bond constraints
#bondconstraints=[[2,3],[2,4],[3,4]]

#geomeTRICOptimizer(theory=QMMM_SP_ORCAcalculation_2, fragment=H2O_MeOH, bondconstraints=[[0,1],[0,2],[1,2]], coordsystem='tric', maxiter=70)


geomeTRICOptimizer(theory=QMMM_SP_ORCAcalculation_2, fragment=H2O_MeOH, frozenatoms=[0,1,2,3,4,5,6], coordsystem='tric', maxiter=70)

#QMMM_Opt_frag = Optimizer(fragment=New_HCl_H2O, theory=QMMM_SP_ORCAcalculation_2, optimizer='KNARR-LBFGS')
#QMMM_Opt_frag.run()


exit()
############################
# QM GEOMETRY OPTIMIZATION  BERNY#
############################
blankline()
#QM only optimization
#print("NewHCl_H2O dict", NewHCl_H2O.__dict__)

H2O=Fragment(xyzfile='h2o_strained.xyz')
print("Printing unoptimized coords:")
H2O.print_coords()
#Creating ORCA theory object without fragment information
ORCAQMtheory = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

#Basic Cartesian optimization with KNARR-LBFGS
#Opt_frag = Optimizer(fragment=H2O, theory=ORCAQMtheory, optimizer='KNARR-LBFGS')
#Opt_frag.run()

#Using PyBernyOpt optimizer from Github
#BernyOpt(ORCAQMtheory,H2O)
# Internal coords but no constraints

#Using geomeTRIC optimization from Github
#geomeTRICOptimizer(ORCAQMtheory,H2O)
#geomeTRICOptimizer(ORCAQMtheory,H2O,frozenatoms=[0])
#print("H2O fragment energy:", H2O.energy)
#blankline()
# Get optimized geometry from fragment. Has been replaced
#H2O.print_coords()


#print("ORCAQMtheory dict:", ORCAQMtheory.__dict__)
#Defining optimization object
#Opt_frag = Optimizer(fragment=H2O, theory=ORCAQMtheory, optimizer='KNARR-LBFGS')
#print("Opt_frag dict:", Opt_frag.__dict__)
#Run optimization
#Opt_frag.run()

#exit()
##############################
# QM NUMERICAL FREQUENCIES   #
##############################
#TODO: Allow for QM/MM Hessian. Requires some thinking.
#TODO: Project out translational and rotational modes.
#Creating ORCA theory object without fragment information
#numcores=8
#BORCAQMtheory = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks )

#Create one-point NumFreq object. Will print out frequencies in output, Hessian as separate hessian-file called
#Numfreq_frag_onepoint.hessian. Print normalmodes in output or as separate file??

#Trying partial Hessian
#Numfreq_frag_onepoint = NumericalFrequencies(fragment=New_HCl_H2O, theory=QMMM_SP_ORCAcalculation_2,
#                                             npoint=1, displacement=0.0052917721079999999276, hessatoms=[2,3,4], numcores=numcores)
#Numfreq_frag_onepoint.run()


#blankline()
#Optional independent diagonalization of Hessian
#freqs_onepoint=diagonalizeHessian(hessian_onepoint,NewHCl_H2O.list_of_masses,NewHCl_H2O.elems)[0]
#freqs_twopoint=diagonalizeHessian(hessian_twopoint,NewHCl_H2O.list_of_masses,NewHCl_H2O.elems)[0]
#print("freqs_onepoint:", freqs_onepoint)
#print("freqs_twopoint:", freqs_twopoint)
exit()

###############################
# QM/MM GEOMETRY OPTIMIZATION #
###############################
#TODO: Write code for creating Lennard-Jones parameters for any molecule
#TODO: Interface MEDFF??
# TODO: Interface something for bonding MM and QM/MM??? orca_mm ???


###########################
# MOLECULAR DYNAMICS      #
###########################
#TODO: Create initial Python-based MD code for basic functionality and understanding.
#TODO: Replace with C++/Fortran written code?? Or interface something ??
#TODO: Add thermostat and Shake functionality?
#TODO: Maybe just do xtb MD with solute and a few xtB MD waters. Rest is frozen TIP3P.
#TODO: Alternatively look into GFN-FF
#Creating ORCA theory object without fragment information
#ORCAQMtheory = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

#Defining MD object

#MD_frag = MolecularDynamics(fragment=NewHCl_H2O, theory=ORCAQMtheory, ensemble='NVT', temperature=300)

#xtb MD: https://xtb-docs.readthedocs.io/en/latest/md.html


#MD program options:
#https://github.com/openmm/openmm
#http://openmd.org/category/examples/

#MD analysis: https://github.com/MDAnalysis/mdanalysis

#Metadynamics: https://github.com/openmm/openmm/issues/2126
#http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.metadynamics.Metadynamics.html