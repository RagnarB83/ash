from yggdrasill import *
import settings_yggdrasill
settings_yggdrasill.init() #initialize

#Test settings
path="/Users/bjornssonsu/ownCloud/PyQMMM-project/Yggdrasill-testdir"
os.chdir(path)

exit()


#PDB read in
PDB_frag = Fragment(pdbfile="xraymodel-solvionated.pdb")
print("PDB_frag:", PDB_frag)


#print(PDB_frag.__dict__)
print(PDB_frag.numatoms)



####################################
#Exception if openmm not found:
#    from simtk.openmm.app import *
#ModuleNotFoundError: No module named 'simtk'


# OPEN MM
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout, exit, stderr

# Load CHARMM files
psf = CharmmPsfFile('step5_charmm2omm.psf')
pdb = PDBFile('step5_charmm2omm.pdb')
params = CharmmParameterSet('par_all36_prot.rtf', 'top_all36_prot.prm',
                       'par_all36_lipid.rtf', 'top_all36_lipid.prm',
                       'toppar_water_ion.str')

# Create an openmm system by calling createSystem on psf
system = psf.createSystem(params, nonbondedMethod=NoCutoff,
         nonbondedCutoff=1*nanometer, constraints=HBonds)

integrator = LangevinIntegrator(300*kelvin,   # Temperature of head bath
                                1/picosecond, # Friction coefficient
                                0.002*picoseconds) # Time step

simulation = Simulation(psf.topology, system, integrator)


###############
exit()
###########################
#Basic pyscf usage
from pyscf import gto, scf

mol = gto.Mole()
mol.verbose = 5
#mol.output = 'out_h2o'
mol.atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587'''
mol.basis = 'ccpvdz'
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.kernel()

exit()

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


###########################################
# POINT-CHARGE EMBEDDED QM-SP CALCULATION #
###########################################


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



###########################
# MOLECULAR DYNAMICS      #
###########################

#Creating ORCA theory object without fragment information
#ORCAQMtheory = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

#Defining MD object

#MD_frag = MolecularDynamics(fragment=NewHCl_H2O, theory=ORCAQMtheory, ensemble='NVT', temperature=300)



