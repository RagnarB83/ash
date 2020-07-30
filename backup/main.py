from yggdrasill import *
import settings_yggdrasill
########################################################
# YGGDRASILL - A GENERAL COMPCHEM AND QM/MM ENVIRONMENT#
########################################################
print_yggdrasill_header()
#TODO: Write Psi4 interface
#Todo: Write Dalton interface

#Define global system settings ( scale, tol and conndepth keywords for connectivity)
settings_yggdrasill.init() #initialize
#General program
path= "/"
os.chdir(path)

#Define optionalXYZ coordinates as multi-line string
fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""

#FRAGMENT CREATION.

#Option 1.
#Step 1. Creation of empty fragment
HF_frag=Fragment()
print("HF_frag dict", HF_frag.__dict__)
#Add coordinates to fragment
HF_frag.coords_from_string(fragcoords)
print("")
print("HF_frag dict", HF_frag.__dict__)
print("-------------")

#Option 2.
#Direct creation of new fragment with coords
HF_frag2=Fragment(coordsstring=fragcoords)
print("HF_frag2 dict", HF_frag2.__dict__)
print("-------------")

#Option 3
#Create fragment from xyzfile via initialization and then run read_xyz class-function
HF_frag3=Fragment()
HF_frag3.read_xyzfile("hf.xyz")
print("HF_frag3 dict", HF_frag3.__dict__)
print("-------------")

#Option 4
#Create fragment from xyzfile directly
HF_frag4 = Fragment(xyzfile="hf.xyz")
print("HF_frag4 dict", HF_frag4.__dict__)
print("-------------")

#Option 5
#Replace coords and elems of fragment object with new lists
elems=['H', 'Cl']
coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
HF_frag5=Fragment()
HF_frag5.replace_coords(elems,coords)
print("HF_frag5 dict", HF_frag5.__dict__)
print("-------------")

#Option 6:
#Add  atoms to fragment multiple times
#Empty fragment
HCl_H2O=Fragment()
elems=['H', 'Cl']
coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
HCl_H2O.add_coords(elems,coords)
elems=['O', 'H', 'H']
coords=[[2.172092667, -1.008295059, 2.571252403],[1.454071996, -0.674986676, 2.018792653],
        [2.237951295, -1.940936718, 2.331039759]]
HCl_H2O.add_coords(elems,coords)
print("HCl_H2O dict", HCl_H2O.__dict__)
print("-------------")

#Option 7. Add molecule from XYZ and calculate connectivity:
FeFeH2ase = Fragment(xyzfile="fefhe2ase.xyz")
print("FeFeH2ase dict", FeFeH2ase.__dict__)
FeFeH2ase.calc_connectivity()
print("FeFeH2ase dict", FeFeH2ase.__dict__)
conn = FeFeH2ase.connectivity
print("conn:", conn)
print("Number of subfragments in FeFeH2ase", len(conn))
print("Number of atoms in FeFeH2ase", FeFeH2ase.numatoms)
print("Number atoms in connectivity in FeFeH2ase", FeFeH2ase.connected_atoms_number)

################################
# SINGLE-POINT QM CALCULATIONS #
################################
blankline()
#ORCA example.
orcadir="/Applications/orca_4.2.1"
xtbdir="/Applications/xtb"
#Define input
orcasimpleinput="! BP86 def2-TZVPP tightscf"
orcablocks="%scf maxiter 200 end"
xtbmethod='GFN2-xTB'
ORCASPcalculation = ORCATheory(orcadir=orcadir, fragment=HF_frag5, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
xTBSPcalculation = xTBTheory(xtbdir=xtbdir, fragment=HF_frag5, charge=0, mult=1, xtbmethod=xtbmethod)

#Simple Energy SP calc
#ORCASPcalculation.run()
#Energy+Gradient calculation
#ORCASPcalculation.run(Grad=True)

#energy=ORCASPcalculation.energy
#blankline()
#xTBSPcalculation.run()
#Todo: Write BS functionality into ORCATheory
# Todo. Add extra basis functionality
#Todo: Add ECP atoms

###########################
# POINT-CHARGE EMBEDDED QM-SP CALCULATION#
###########################
print("Fragment HCl_H2O:", HCl_H2O.__dict__)
#QM atoms: H and Cl. Rest (H2O) is MM atoms
qmatoms=[0,1]

# Charge definitions for whole fragment.
#Manual definition
atomcharges=[3.0, 1.0, -0.8, 0.4, 0.4]

#MM forcefield defined as dict:
MM_forcefield={}
#For atomtype OT, defining both charge and LJ parameters
MM_forcefield = {'OT':AtomMMobject(atomcharge=3.0, LJparameters=[0.15, 2.7])}
# Defining only LJ parameters. charge may come from list instead
#MM_forcefield = {'OT':AtomMMobject(LJparameters=[0.15, 2.7])}
#Todo: Define forcefield as simple textfile that can be read in. Possibly with Chemshell syntax
print(MM_forcefield['OT'])
print(MM_forcefield['OT'].atomcharge)
print(MM_forcefield['OT'].LJparameters)

blankline()
blankline()
#Creating ORCA theory object without fragment information
ORCAQMtheory = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
#PointchargeMMtheory = Theory( atom_types=atomtypes force_field=MMdefinition)
PointchargeMMtheory = NonBondedTheory( charges = atomcharges, LJ=False)

#Create QM/MM theory object
#without MM theory but with atomcharges
QMMM_SP_ORCAcalculation = QMMMTheory(fragment=HCl_H2O, qm_theory=ORCAQMtheory, qmatoms=qmatoms, atomcharges=atomcharges)
#QMMM_SP_ORCAcalculation.run(Grad=True)

blankline()
#with MM theory
QMMM_SP_ORCAcalculation = QMMMTheory(fragment=HCl_H2O, qm_theory=ORCAQMtheory, mm_theory=PointchargeMMtheory, qmatoms=qmatoms, atomcharges=atomcharges)
#QMMM_SP_ORCAcalculation.run()

###########################
# GEOMETRY OPTIMIZATION   #
###########################
blankline()
#QM only optimization

#Creating ORCA theory object without fragment information
ORCAQMtheory = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

print("ORCAQMtheory dict:", ORCAQMtheory.__dict__)
#Defining optimization object
Opt_frag = Optimizer(fragment=HCl_H2O, theory=ORCAQMtheory, optimizer='SD')
print("Opt_frag dict:", Opt_frag.__dict__)
#Run optimization
Opt_frag.run()

###########################
# MOLECULAR DYNAMICS      #
###########################

