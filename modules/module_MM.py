import numpy as np
import time

from modules.module_coords import distance
from functions.functions_general import ashexit, blankline,print_time_rel,BC, load_julia_interface
import constants
import ash


# Simple nonbonded MM theory. Charges and LJ-potentials
class NonBondedTheory:
    def __init__(self, atomtypes=None, forcefield=None, charges = None, LJcombrule='geometric',
                 codeversion=None, printlevel=2):

        #If codeversion not explicity asked for then we go for defaults (that may have changed if Julia interface failed)
        if codeversion == None:
            codeversion=ash.settings_ash.settings_dict["nonbondedMM_code"]
            print("MM Codeversion not set. Using default setting: ", codeversion)
        self.codeversion=codeversion

        #Printlevel
        self.printlevel=printlevel

        #Atom types
        self.atomtypes=atomtypes
        #Read MM forcefield.
        self.forcefield=forcefield

        #
        self.numatoms = len(self.atomtypes)
        self.LJcombrule=LJcombrule

        #Todo: Delete
        # If qmatoms list passed to Nonbonded theory then we are doing QM/MM
        #self.qmatoms=qmatoms
        #print("Defining Nonbonded Theory")
        #print("qmatoms:", self.qmatoms)



        #These are charges for whole system including QM.
        self.atom_charges = charges
        #Possibly have self.mm_charges here also??


        #Initializing sigmaij and epsij arrays. Will be filled by calculate_LJ_pairpotentials
        self.sigmaij=np.zeros((self.numatoms, self.numatoms))
        self.epsij=np.zeros((self.numatoms, self.numatoms))
        self.pairarrays_assigned = False

    #Todo: Need to make active-region version of pyarray version here.
    def calculate_LJ_pairpotentials(self, qmatoms=None, actatoms=None, frozenatoms=None):
        module_init_time=time.time()
        #actatoms
        if actatoms is None:
            actatoms=[]
        if frozenatoms is None:
            frozenatoms=[]

        #Deleted combination_rule argument. Now using variable assigned to object

        combination_rule=self.LJcombrule

        #If qmatoms passed list passed then QM/MM and QM-QM pairs will be ignored from pairlist
        if self.printlevel >= 2:
            print("Inside calculate_LJ_pairpotentials")
        #Todo: Figure out if we can find out if qmatoms without being passed
        if qmatoms is None or qmatoms == []:
            qmatoms = []
            print("WARNING: qmatoms list is empty.")
            print("This is fine if this is a pure MM job.")
            print("If QM/MM job, then qmatoms list should be passed to NonBonded theory.")


        import math
        if self.printlevel >= 2:
            print("Defining Lennard-Jones pair potentials")

        #List to store pairpotentials
        self.LJpairpotentials=[]
        #New: multi-key dict instead using tuple
        self.LJpairpotdict={}
        if combination_rule == 'geometric':
            if self.printlevel >= 2:
                print("Using geometric mean for LJ pair potentials")
        elif combination_rule == 'arithmetic':
            if self.printlevel >= 2:
                print("Using geometric mean for LJ pair potentials")
        elif combination_rule == 'mixed_geoepsilon':
            if self.printlevel >= 2:
                print("Using mixed rule for LJ pair potentials")
                print("Using arithmetic rule for r/sigma")
                print("Using geometric rule for epsilon")
        elif combination_rule == 'mixed_geosigma':
            if self.printlevel >= 2:
                print("Using mixed rule for LJ pair potentials")
                print("Using geometric rule for r/sigma")
                print("Using arithmetic rule for epsilon")
        else:
            print("Unknown combination rule. Exiting")
            ashexit()

        #A large system has many atomtypes. Creating list of unique atomtypes to simplify loop
        CheckpointTime = time.time()
        self.uniqatomtypes = np.unique(self.atomtypes).tolist()
        DoAll=True
        for count_i, at_i in enumerate(self.uniqatomtypes):
            #print("count_i:", count_i)
            for count_j,at_j in enumerate(self.uniqatomtypes):
                #if count_i < count_j:
                if DoAll==True:
                    #print("at_i {} and at_j {}".format(at_i,at_j))
                    #Todo: if atom type not in dict we get a KeyError here.
                    # Todo: Add exception or add zero-entry to dict ??
                    if len(self.forcefield[at_i].LJparameters) == 0:
                        continue
                    if len(self.forcefield[at_j].LJparameters) == 0:
                        continue
                    if self.printlevel >= 3:
                        print("LJ sigma_i {} for atomtype {}:".format(self.forcefield[at_i].LJparameters[0], at_i))
                        print("LJ sigma_j {} for atomtype {}:".format(self.forcefield[at_j].LJparameters[0], at_j))
                        print("LJ eps_i {} for atomtype {}:".format(self.forcefield[at_i].LJparameters[1], at_i))
                        print("LJ eps_j {} for atomtype {}:".format(self.forcefield[at_j].LJparameters[1], at_j))
                        blankline()
                    if combination_rule=='geometric':
                        sigma=math.sqrt(self.forcefield[at_i].LJparameters[0]*self.forcefield[at_j].LJparameters[0])
                        epsilon=math.sqrt(self.forcefield[at_i].LJparameters[1]*self.forcefield[at_j].LJparameters[1])
                        if self.printlevel >=3:
                            print("LJ sigma_ij : {} for atomtype-pair: {} {}".format(sigma,at_i, at_j))
                            print("LJ epsilon_ij : {} for atomtype-pair: {} {}".format(epsilon,at_i, at_j))
                            blankline()
                    elif combination_rule=='arithmetic':
                        if self.printlevel >=3:
                            print("Using arithmetic mean for LJ pair potentials")
                            print("NOTE: to be confirmed")
                        sigma=0.5*(self.forcefield[at_i].LJparameters[0]+self.forcefield[at_j].LJparameters[0])
                        epsilon=0.5-(self.forcefield[at_i].LJparameters[1]+self.forcefield[at_j].LJparameters[1])
                    elif combination_rule=='mixed_geosigma':
                        if self.printlevel >=3:
                            print("Using geometric mean for LJ sigma parameters")
                            print("Using arithmetic mean for LJ epsilon parameters")
                            print("NOTE: to be confirmed")
                        sigma=math.sqrt(self.forcefield[at_i].LJparameters[0]*self.forcefield[at_j].LJparameters[0])
                        epsilon=0.5-(self.forcefield[at_i].LJparameters[1]+self.forcefield[at_j].LJparameters[1])
                    elif combination_rule=='mixed_geoepsilon':
                        if self.printlevel >=3:
                            print("Using arithmetic mean for LJ sigma parameters")
                            print("Using geometric mean for LJ epsilon parameters")
                            print("NOTE: to be confirmed")
                        sigma=0.5*(self.forcefield[at_i].LJparameters[0]+self.forcefield[at_j].LJparameters[0])
                        epsilon=math.sqrt(self.forcefield[at_i].LJparameters[1]*self.forcefield[at_j].LJparameters[1])
                    self.LJpairpotentials.append([at_i, at_j, sigma, epsilon])
                    #Dict using two keys (actually a tuple of two keys)
                    self.LJpairpotdict[(at_i,at_j)] = [sigma, epsilon]
                    #print(self.LJpairpotentials)
        #Takes not time so disabling time-printing
        #print_time_rel(CheckpointTime, modulename="pairpotentials")
        #Remove redundant pair potentials
        CheckpointTime = time.time()
        for acount, pairpot_a in enumerate(self.LJpairpotentials):
            for bcount, pairpot_b in enumerate(self.LJpairpotentials):
                if acount < bcount:
                    if set(pairpot_a) == set(pairpot_b):
                        del self.LJpairpotentials[bcount]
        if self.printlevel >= 2:
            #print("Final LJ pair potentials (sigma_ij, epsilon_ij):\n", self.LJpairpotentials)
            print("New: LJ pair potentials as dict:")
            print("self.LJpairpotdict:", self.LJpairpotdict)

        #Create numatomxnumatom array of eps and sigma
        blankline()
        if self.printlevel >= 2:
            print("Creating epsij and sigmaij arrays ({},{})".format(self.numatoms,self.numatoms))
            print("Will skip QM-QM ij pairs for qmatoms: ", qmatoms)
            print("Will skip frozen-frozen ij pairs")
            print("len actatoms:", len(actatoms))
            print("len frozenatoms:", len(frozenatoms))
        beginTime = time.time()

        CheckpointTime = time.time()
        # See speed-tests at /home/bjornsson/pairpot-test

        print("self.codeversion:", self.codeversion)
        if self.codeversion=="julia":
            if self.printlevel >= 2:
                print("Using PyJulia for fast sigmaij and epsij array creation")
            # Necessary for statically linked libpython
            #try:
            #    from julia.api import Julia
            #    from julia import Main
            #except:
            #   print("Problem importing Pyjulia (import julia)")
            #    print("Make sure Julia is installed and PyJulia module available")
            #    print("Also, are you using python3_ash ?")
            #    print("Alternatively, use codeversion='py' argument to NonBondedTheory to use slower Python version for array creation")
            #    ashexit()
            print("Loading Julia")
            try:
                Juliafunctions=load_julia_interface()
            except:
                print("Problem loading Julia")
                ashexit()
            # Do pairpot array for whole system
            if len(actatoms) == 0:
                print("Calculating pairpotential array for whole system")

                self.sigmaij, self.epsij = Juliafunctions.pairpot_full(self.numatoms, self.atomtypes, self.LJpairpotdict,qmatoms)
            else:
            #    #or only for active region
                print("Calculating pairpotential array for active region only")
                #pairpot_active(numatoms,atomtypes,LJpydict,qmatoms,actatoms)
                print("self.numatoms", self.numatoms)
                #print("self.atomtypes", self.atomtypes)
                print("self.LJpairpotdict", self.LJpairpotdict)
                print("qmatoms", qmatoms)
                #print("actatoms", actatoms)
                
                self.sigmaij, self.epsij = Juliafunctions.pairpot_active(self.numatoms, self.atomtypes, self.LJpairpotdict, qmatoms, actatoms)
        # New for-loop for creating sigmaij and epsij arrays. Uses dict-lookup instead
        elif self.codeversion=="py":
            if self.printlevel >= 2:
                print("Using Python version for array creation")
                print("Does not yet skip frozen-frozen atoms...to be fixed")
                #Todo: add frozen-frozen atoms skip
            #Update: Only doing half of array
            for i in range(self.numatoms):
                for j in range(i+1, self.numatoms):
                    #Skipping if i-j pair in qmatoms list. I.e. not doing QM-QM LJ calc.
                    #if all(x in qmatoms for x in (i, j)) == True:
                    #
                    if i in qmatoms and j in qmatoms:
                        #print("Skipping i-j pair", i,j, " as these are QM atoms")
                        continue
                    elif (self.atomtypes[i], self.atomtypes[j]) in self.LJpairpotdict:
                        self.sigmaij[i, j] = self.LJpairpotdict[(self.atomtypes[i], self.atomtypes[j])][0]
                        self.epsij[i, j] = self.LJpairpotdict[(self.atomtypes[i], self.atomtypes[j])][1]
                    elif (atomtypes[j], atomtypes[i]) in self.LJpairpotdict:
                        self.sigmaij[i, j] = self.LJpairpotdict[(self.atomtypes[j], self.atomtypes[i])][0]
                        self.epsij[i, j] = self.LJpairpotdict[(self.atomtypes[j], self.atomtypes[i])][1]
        else:
            print("unknown codeversion")
            ashexit()

        if self.printlevel >= 2:
            #print("self.sigmaij ({}) : {}".format(len(self.sigmaij), self.sigmaij))
            #print("self.epsij ({}) : {}".format(len(self.epsij), self.epsij))
            print("sigmaij size: {}".format(len(self.sigmaij)))
            print("epsij size: {}".format(len(self.epsij)))
        #print_time_rel(CheckpointTime, modulename="pairpot arrays", moduleindex=4)
        self.pairarrays_assigned = True
        print_time_rel(module_init_time, modulename='LJ-pairpotential arrays', moduleindex=4)


    def update_charges(self,atomlist,charges):
        print("Updating charges.")
        assert len(atomlist) == len(charges)
        for atom,charge in zip(atomlist,charges):
            self.atom_charges[atom] = charge
        #print("Charges are now:", charges)
        print("Sum of charges:", sum(charges))

    # current_coords is now used for full_coords, charges for full coords
    def run(self, current_coords=None, elems=None, charges=None, connectivity=None,
            Coulomb=True, Grad=True, qmatoms=None, actatoms=None, frozenatoms=None):
        module_init_time=time.time()
        if current_coords is None:
            print("No current_coords argument. Exiting...")
            ashexit()
        CheckpointTime = time.time()
        #If qmatoms list provided to run (probably by QM/MM object) then we are doing QM/MM
        #QM-QM pairs will be skipped in LJ

        #Testing if arrays assigned or not. If not calling calculate_LJ_pairpotentials
        #Passing qmatoms over so pairs can be skipped
        #This sets self.sigmaij and self.epsij and also self.LJpairpotentials
        #Todo: if actatoms have been defined this will be skipped in pairlist creation
        #if frozenatoms passed frozen-frozen interactions will be skipped
        if self.pairarrays_assigned is False:
            print("Calling LJ pairpot calc")
            self.calculate_LJ_pairpotentials(qmatoms=qmatoms,actatoms=actatoms)
        else:
            print("LJ pairpot arrays exist...")

        if len(self.LJpairpotentials) > 0:
            LJ=True
        #If charges not provided to run function. Use object charges
        if charges == None:
            charges=self.atom_charges

        #If coords not provided to run function. Use object coords
        #HMM. I guess we are not keeping coords as part of MMtheory?
        #if len(full_cords)==0:
        #    full_coords=

        if self.printlevel >= 2:
            print(BC.OKBLUE, BC.BOLD, "------------RUNNING NONBONDED MM CODE-------------", BC.END)
            print("Calculating MM energy and gradient")
        #initializing
        self.Coulombchargeenergy=0
        self.LJenergy=0
        self.MMGradient=[]
        self.Coulombchargegradient=[]
        self.LJgradient=[]

        #Slow Python version
        if self.codeversion=='py':
            if self.printlevel >= 2:
                print("Using slow Python MM code")
            #Sending full coords and charges over. QM charges are set to 0.
            if Coulomb==True:
                self.Coulombchargeenergy, self.Coulombchargegradient  = coulombcharge(charges, current_coords)
                if self.printlevel >= 2:
                    print("Coulomb Energy (au):", self.Coulombchargeenergy)
                    print("Coulomb Energy (kcal/mol):", self.Coulombchargeenergy * constants.harkcal)
                    print("")
                    #print("self.Coulombchargegradient:", self.Coulombchargegradient)
                blankline()
            # NOTE: Lennard-Jones should  calculate both MM-MM and QM-MM LJ interactions. Full coords necessary.
            if LJ==True:
                #LennardJones(coords, epsij, sigmaij, connectivity=[], qmatoms=[])
                #self.LJenergy,self.LJgradient = LennardJones(full_coords,self.atomtypes, self.LJpairpotentials, connectivity=connectivity)
                self.LJenergy,self.LJgradient = LennardJones(current_coords,self.epsij,self.sigmaij)
                #print("Lennard-Jones Energy (au):", self.LJenergy)
                #print("Lennard-Jones Energy (kcal/mol):", self.LJenergy*constants.harkcal)
            self.MMEnergy = self.Coulombchargeenergy+self.LJenergy
            if Grad==True:
                self.MMGradient = self.Coulombchargegradient+self.LJgradient
        #Combined Coulomb+LJ Python version. Slow
        elif self.codeversion=='py_comb':
            print("not active")
            ashexit()
            self.MMenergy, self.MMgradient = LJCoulpy(current_coords, self.atomtypes, charges, self.LJpairpotentials,
                                                          connectivity=connectivity)
        elif self.codeversion=='f2py':
            if self.printlevel >= 2:
                print("Using Fortran F2Py MM code")
            try:
                #print(os.environ.get("LD_LIBRARY_PATH"))
                import LJCoulombv1
            except:
                print("Fortran library LJCoulombv1 not found! Make sure you have run the installation script.")
            self.MMEnergy, self.MMGradient, self.LJenergy, self.Coulombchargeenergy =\
                LJCoulomb(current_coords, self.epsij, self.sigmaij, charges, connectivity=connectivity)
        elif self.codeversion=='f2pyv2':
            if self.printlevel >= 2:
                print("Using fast Fortran F2Py MM code v2")
            try:
                import LJCoulombv2
                print(LJCoulombv2.__doc__)
                print("----------")
            except:
                print("Fortran library LJCoulombv2 not found! Make sure you have run the installation script.")
            self.MMEnergy, self.MMGradient, self.LJenergy, self.Coulombchargeenergy =\
                LJCoulombv2(current_coords, self.epsij, self.sigmaij, charges, connectivity=connectivity)
        elif self.codeversion=='julia':
            if self.printlevel >= 2:
                print("Using fast Julia version, v1")
            # Necessary for statically linked libpython
            try:
                Juliafunctions=load_julia_interface()
            except:
                print("Problem loading Julia")
                print("Problem importing Pyjulia (import julia)")
                print("Make sure Julia is installed and PyJulia module available")
                print("Also, are you using python3_ash ?")
                print("Alternatively, use codeversion='py' argument to NonBondedTheory to use slower Python version for array creation")
                ashexit()

            print_time_rel(CheckpointTime, modulename="from run to just before calling ")
            self.MMEnergy, self.MMGradient, self.LJenergy, self.Coulombchargeenergy =\
                Juliafunctions.LJcoulombchargev1c(charges, current_coords, self.epsij, self.sigmaij, connectivity)
            print_time_rel(CheckpointTime, modulename="from run to done julia")
        else:
            print("Unknown version of MM code")
            ashexit()

        if self.printlevel >= 2:
            print("Lennard-Jones Energy (au):", self.LJenergy)
            print("Lennard-Jones Energy (kcal/mol):", self.LJenergy * constants.harkcal)
            print("Coulomb Energy (au):", self.Coulombchargeenergy)
            print("Coulomb Energy (kcal/mol):", self.Coulombchargeenergy * constants.harkcal)
            print("MM Energy:", self.MMEnergy)
        if self.printlevel >= 3:
            print("self.MMGradient:", self.MMGradient)

        if self.printlevel >= 2:
            print(BC.OKBLUE, BC.BOLD, "------------ENDING NONBONDED MM CODE-------------", BC.END)
        print_time_rel(module_init_time, modulename='NonbondedTheory run', moduleindex=2)
        return self.MMEnergy, self.MMGradient


#MMAtomobject used to store LJ parameter and possibly charge for MM atom with atomtype, e.g. OT
class AtomMMobject:
    def __init__(self, atomcharge=None, LJparameters=None, element=None):
        self.atomcharge = atomcharge
        self.LJparameters = LJparameters
        self.element=element
    def add_charge(self, atomcharge=None):
        self.atomcharge = atomcharge
    def add_LJparameters(self, LJparameters=None):
        self.LJparameters=LJparameters
    def add_element(self,element=None):
        self.element=element

#Makes more sense to store this here. Simplifies ASH inputfile import.
def MMforcefield_read(file):
    print("Reading forcefield file:", file)
    MM_forcefield = {}
    atomtypes=[]
    MM_forcefield["residues"]=[]
    with open(file) as f:
        for line in f:
            if '#' not in line:
                #Now reading residue type (fragmenttypes)
                if line.startswith("resid"):
                    if 'atomtypes' in line:
                        residname=line.split()[0].replace("_atomtypes","")
                        MM_forcefield["residues"].append(residname)
                        #adding atomtypes for residue
                        lsplit=line.split()
                        MM_forcefield[residname+'_atomtypes']=lsplit[1:]
                    if 'charges' in line:
                        residname=line.split()[0].replace("_charges","")
                        lsplit=line.split()
                        MM_forcefield[residname+'_charges']=[float(i) for i in lsplit[1:]]                 
                    if 'elements' in line:
                        #adding elements for residue
                        lsplit=line.split()
                        residname=line.split()[0].replace("_elements","")
                        MM_forcefield[residname+'_elements']=lsplit[1:]
                    #if line.startswith("charges"):
                    #    lsplit=line.split()
                    #    for c in lsplit:
                    #        MM_forcefield[atomtype].add_charge(atomcharge=c)
                    #    MM_forcefield[residname]=lsplit[1:]     
                
                if 'combination_rule' in line:
                    combrule=line.split()[-1]
                    print("Found combination rule defintion in forcefield file:", combrule)
                    MM_forcefield["combination_rule"]=combrule
                #This 
                if line.startswith("charge") == True:
                    print("Found charge definition in forcefield file:", ' '.join(line.split()[:]))
                    atomtype=line.split()[1]
                    if atomtype not in MM_forcefield.keys():
                        MM_forcefield[atomtype]=AtomMMobject()
                    charge=float(line.split()[2])
                    MM_forcefield[atomtype].add_charge(atomcharge=charge)
                    # TODO: Charges defined are currently not used I think
                if 'LennardJones_i_sigma' in line:
                    print("Found LJ single-atom sigma definition in forcefield file:", ' '.join(line.split()[:]))
                    atomtype=line.split()[1]
                    if atomtype not in MM_forcefield.keys():
                        MM_forcefield[atomtype] = AtomMMobject()
                    sigma_i=float(line.split()[2])
                    eps_i=float(line.split()[3])
                    MM_forcefield[atomtype].add_LJparameters(LJparameters=[sigma_i,eps_i])
                if 'LennardJones_i_R0' in line:
                    print("Found LJ single-atom R0 definition in forcefield file:", ' '.join(line.split()[:]))
                    atomtype=line.split()[1]
                    R0tosigma=0.5**(1/6)
                    if atomtype not in MM_forcefield.keys():
                        MM_forcefield[atomtype] = AtomMMobject()
                    sigma_i=float(line.split()[2])*R0tosigma
                    eps_i=float(line.split()[3])
                    MM_forcefield[atomtype].add_LJparameters(LJparameters=[sigma_i,eps_i])
                #if 'element' in line:
                #    atomtype=line.split()[1]
                #    if atomtype not in MM_forcefield.keys():
                #        MM_forcefield[atomtype] = AtomMMobject()
                #    el=line.split()[2]
                #    MM_forcefield[atomtype].add_element(element=el)
                if 'LennardJones_ij' in line:
                    print("Found LJ pair definition in forcefield file")
                    atomtype_i=line.split()[1]
                    atomtype_j=line.split()[2]
                    sigma_ij=float(line.split()[3])
                    eps_ij=float(line.split()[4])
                    print("This is incomplete. Exiting")
                    ashexit()
                    # TODO: Need to finish this. Should replace LennardJonespairpotentials later
    return MM_forcefield




#UFF dictionary with parameters
#Taken from oldmolcrys/old-solvshell and originally from Chemshell
# Element: [R0,eps]. R0 in Angstrom and eps in kcal/mol
UFFdict={'H': [2.886, 0.044], 'He': [2.362, 0.056], 'Li': [2.451, 0.025], 'Be': [2.745, 0.085], 'B': [4.083, 0.18],
         'C': [3.851, 0.105], 'N': [3.66, 0.069], 'O': [3.5, 0.06], 'F': [3.364, 0.05], 'Ne': [3.243, 0.042],
         'Na': [2.983, 0.03], 'Mg': [3.021, 0.111], 'Al': [4.499, 0.505], 'Si': [4.295, 0.402], 'P': [4.147, 0.305],
         'S': [4.035, 0.274], 'Cl': [3.947, 0.227], 'Ar': [3.868, 0.185], 'K': [3.812, 0.035], 'Ca': [3.399, 0.238],
         'Sc': [3.295, 0.019], 'Ti': [3.175, 0.017], 'V': [3.144, 0.016], 'Cr': [3.023, 0.015], 'Mn': [2.961, 0.013],
         'Fe': [2.912, 0.013], 'Co': [2.872, 0.014], 'Ni': [2.834, 0.015], 'Cu': [3.495, 0.005], 'Zn': [2.763, 0.124],
         'Ga': [4.383, 0.415], 'Ge': [4.28, 0.379], 'As': [4.23, 0.309], 'Se': [4.205, 0.291], 'Br': [4.189, 0.251],
         'Kr': [4.141, 0.22], 'Rb': [4.114, 0.04], 'Sr': [3.641, 0.235], 'Y': [3.345, 0.072], 'Zr': [3.124, 0.069],
         'Nb': [3.165, 0.059], 'Mo': [3.052, 0.056], 'Tc': [2.998, 0.048], 'Ru': [2.963, 0.056], 'Rh': [2.929, 0.053],
         'Pd': [2.899, 0.048], 'Ag': [3.148, 0.036], 'Cd': [2.848, 0.228], 'In': [4.463, 0.599], 'Sn': [4.392, 0.567],
         'Sb': [4.42, 0.449], 'Te': [4.47, 0.398], 'I': [4.5, 0.339], 'Xe': [4.404, 0.332], 'Cs': [4.517, 0.045],
         'Ba': [3.703, 0.364], 'La': [3.522, 0.017], 'Ce': [3.556, 0.013], 'Pr': [3.606, 0.01], 'Nd': [3.575, 0.01],
         'Pm': [3.547, 0.009], 'Sm': [3.52, 0.008], 'Eu': [3.493, 0.008], 'Gd': [3.368, 0.009], 'Tb': [3.451, 0.007],
         'Dy': [3.428, 0.007], 'Ho': [3.409, 0.007], 'Er': [3.391, 0.007], 'Tm': [3.374, 0.006], 'Yb': [3.355, 0.228],
         'Lu': [3.64, 0.041], 'Hf': [4.141, 0.072], 'Ta': [3.17, 0.081], 'W': [3.069, 0.067], 'Re': [2.954, 0.066],
         'Os': [3.12, 0.037], 'Ir': [2.84, 0.073], 'Pt': [2.754, 0.08], 'Au': [3.293, 0.039], 'Hg': [2.705, 0.385],
         'Tl': [4.347, 0.68], 'Pb': [4.297, 0.663], 'Bi': [4.37, 0.518], 'Po': [4.709, 0.325], 'At': [4.75, 0.284],
         'Rn': [4.765, 0.248], 'Fr': [4.9, 0.05], 'Ra': [3.677, 0.404], 'Ac': [3.478, 0.033], 'Th': [3.396, 0.026],
         'Pa': [3.424, 0.022], 'U': [3.395, 0.022], 'Np': [3.424, 0.019], 'Pu': [3.424, 0.016], 'Am': [3.381, 0.014],
         'Cm': [3.326, 0.013], 'Bk': [3.339, 0.013], 'Cf': [3.313, 0.013], 'Es': [3.299, 0.012], 'Fm': [3.286, 0.012],
         'Md': [3.274, 0.011], 'No': [3.248, 0.011], 'Lr': [3.236, 0.011]}

#CHARMM-GAAMP for small drug-like molecules
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5997559/#SD1
#REST IS UFF
#CHARMMXXXdict={'H': [2.886, 0.044],  'B': [4.083, 0.18],
#         'C': [3.851, 0.105], 'N': [3.66, 0.069], 'O': [3.5, 0.06], 'F': [3.364, 0.05], 
#         'Na': [2.983, 0.03], 'Mg': [3.021, 0.111], 'Al': [4.499, 0.505], 'Si': [4.295, 0.402], 'P': [4.147, 0.305],
#         'S': [4.035, 0.274], 'Cl': [3.947, 0.227]}

#Modified UFF dictionary with no LJ on H.
#Deprecated. To be deleted.
UFF_modH_dict={'H': [0.000, 0.000], 'He': [2.362, 0.056], 'Li': [2.451, 0.025], 'Be': [2.745, 0.085], 'B': [4.083, 0.18],
         'C': [3.851, 0.105], 'N': [3.66, 0.069], 'O': [3.5, 0.06], 'F': [3.364, 0.05], 'Ne': [3.243, 0.042],
         'Na': [2.983, 0.03], 'Mg': [3.021, 0.111], 'Al': [4.499, 0.505], 'Si': [4.295, 0.402], 'P': [4.147, 0.305],
         'S': [4.035, 0.274], 'Cl': [3.947, 0.227], 'Ar': [3.868, 0.185], 'K': [3.812, 0.035], 'Ca': [3.399, 0.238],
         'Sc': [3.295, 0.019], 'Ti': [3.175, 0.017], 'V': [3.144, 0.016], 'Cr': [3.023, 0.015], 'Mn': [2.961, 0.013],
         'Fe': [2.912, 0.013], 'Co': [2.872, 0.014], 'Ni': [2.834, 0.015], 'Cu': [3.495, 0.005], 'Zn': [2.763, 0.124],
         'Ga': [4.383, 0.415], 'Ge': [4.28, 0.379], 'As': [4.23, 0.309], 'Se': [4.205, 0.291], 'Br': [4.189, 0.251],
         'Kr': [4.141, 0.22], 'Rb': [4.114, 0.04], 'Sr': [3.641, 0.235], 'Y': [3.345, 0.072], 'Zr': [3.124, 0.069],
         'Nb': [3.165, 0.059], 'Mo': [3.052, 0.056], 'Tc': [2.998, 0.048], 'Ru': [2.963, 0.056], 'Rh': [2.929, 0.053],
         'Pd': [2.899, 0.048], 'Ag': [3.148, 0.036], 'Cd': [2.848, 0.228], 'In': [4.463, 0.599], 'Sn': [4.392, 0.567],
         'Sb': [4.42, 0.449], 'Te': [4.47, 0.398], 'I': [4.5, 0.339], 'Xe': [4.404, 0.332], 'Cs': [4.517, 0.045],
         'Ba': [3.703, 0.364], 'La': [3.522, 0.017], 'Ce': [3.556, 0.013], 'Pr': [3.606, 0.01], 'Nd': [3.575, 0.01],
         'Pm': [3.547, 0.009], 'Sm': [3.52, 0.008], 'Eu': [3.493, 0.008], 'Gd': [3.368, 0.009], 'Tb': [3.451, 0.007],
         'Dy': [3.428, 0.007], 'Ho': [3.409, 0.007], 'Er': [3.391, 0.007], 'Tm': [3.374, 0.006], 'Yb': [3.355, 0.228],
         'Lu': [3.64, 0.041], 'Hf': [4.141, 0.072], 'Ta': [3.17, 0.081], 'W': [3.069, 0.067], 'Re': [2.954, 0.066],
         'Os': [3.12, 0.037], 'Ir': [2.84, 0.073], 'Pt': [2.754, 0.08], 'Au': [3.293, 0.039], 'Hg': [2.705, 0.385],
         'Tl': [4.347, 0.68], 'Pb': [4.297, 0.663], 'Bi': [4.37, 0.518], 'Po': [4.709, 0.325], 'At': [4.75, 0.284],
         'Rn': [4.765, 0.248], 'Fr': [4.9, 0.05], 'Ra': [3.677, 0.404], 'Ac': [3.478, 0.033], 'Th': [3.396, 0.026],
         'Pa': [3.424, 0.022], 'U': [3.395, 0.022], 'Np': [3.424, 0.019], 'Pu': [3.424, 0.016], 'Am': [3.381, 0.014],
         'Cm': [3.326, 0.013], 'Bk': [3.339, 0.013], 'Cf': [3.313, 0.013], 'Es': [3.299, 0.012], 'Fm': [3.286, 0.012],
         'Md': [3.274, 0.011], 'No': [3.248, 0.011], 'Lr': [3.236, 0.011]}

#Alternative H parameters:
# Polar H on nitrogens  (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5997559/#SD1)
# -0.01028 (eps)		0.59895 (Rmin  /2 ???)
#  0.0046 (eps  ?)      0.224500  (Rmin/2 )



#Fast LJ-Coulomb via Fortran and f2PY
#Outdated, to be removed
def LJCoulomb(coords,epsij, sigmaij, charges, connectivity=None):
    #print("Inside LJCoulomb")
    #Todo: Avoid calling import everytime in the future...
    import LJCoulombv1
    #print(LJCoulombv1.__doc__)
    numatoms=len(coords)
    rc=9999.5
    grad = np.zeros((numatoms,3))
    penergy, LJenergy, coulenergy, grad = LJCoulombv1.ljcoulegrad(coords, rc, epsij, sigmaij, charges, grad, dim=3, natom=numatoms)
    return penergy, grad, LJenergy, coulenergy

#Fast LJ-Coulomb via Fortran and f2PY
#Outdated, to be removed
def LJCoulombv2(coords,epsij, sigmaij, charges, connectivity=None):
    #print("Inside LJCoulomb")
    #Todo: Avoid calling import everytime in the future...
    import LJCoulombv2
    #print(LJCoulombv2.__doc__)
    numatoms=len(coords)
    #rc: threshold for ignoring LJ interaction
    rc=9999.5
    grad = np.zeros((numatoms,3))
    #Calling Fortran function
    penergy, LJenergy, coulenergy, grad = LJCoulombv2.ljcoulegrad(coords, rc, epsij, sigmaij, charges, grad, dim=3, natom=numatoms)
    return penergy, grad, LJenergy, coulenergy

#Slow Lennard-Jones function
#Outdated.
def LennardJones(coords, epsij, sigmaij, connectivity=None, qmatoms=None):
    print("Inside Python Lennard-Jones function")
    #print("qmatoms:", qmatoms)
    #print("QM atom pairs are skipped if qmatoms list provided")
    #print("connectivity: ", connectivity)
    #print("Calculating LJ pairs based on connectivity if present")
    #print("Note: This means that if two LJ sites are part of same molecular fragment then LJ is not calculated")
    #print("Note: Not correct behaviour for CHARMM/Amber etc")
    #print("Note: Will give correct behaviour for molcrys-QM/MM as QM fragment will not interact with itself")


    #if len(connectivity)==0:
    #    print("Warning!. No connectivity list present. Will treat all LJ pairs.")
    #else:
    #    print(len(connectivity)," connectivity lists present")

    atomlist=list(range(0, len(coords)))
    #LJ energy
    energy=0
    #LJ gradient
    gradient = np.zeros((len(coords), 3))
    #Iterating over atom i
    for i in atomlist:
        #Iterating over atom j
        for j in atomlist:
            #Skipping if same atom
            if i != j:
                #Skipping identical pairs
                if i < j:
                    pairdistance = distance(coords[i], coords[j])
                    #print("sigma, eps, pairdistance", sigma,eps,pairdistance)
                    V_LJ=4*epsij[i,j]*((sigmaij[i,j]/pairdistance)**12-(sigmaij[i,j]/pairdistance)**6)
                    energy+=V_LJ
                    #Typo in http://localscf.com/localscf.com/LJPotential.aspx.html ??
                    #Using http://www.courses.physics.helsinki.fi/fys/moldyn/lectures/L4.pdf
                    #Check this: http://people.virginia.edu/~lz2n/mse627/notes/Potentials.pdf
                    LJgrad_const=(24*epsij[i,j]*((sigmaij[i,j]/pairdistance)**6-2*(sigmaij[i,j]/pairdistance)**12))*(1/(pairdistance**2))
                    gr=np.array([(coords[i][0] - coords[j][0])*LJgrad_const, (coords[i][1] - coords[j][1])*LJgrad_const,
                                     (coords[i][2] - coords[j][2])*LJgrad_const])
                    gradient[i] += gr
                    gradient[j] -= gr
    #Convert gradient from kcal/mol per Å to hartree/Bohr
    final_gradient=gradient * (1/constants.harkcal) / constants.ang2bohr
    #print("LJ gradient (hartree/Bohr):", final_gradient)
    #Converg energy from kcal/mol to hartree
    final_energy=energy*(1/constants.harkcal)
    print("LJ energy (hartree)", final_energy)

    return final_energy,final_gradient

#TODO: Do we always calculate charge if atoms are connected? Need connectivity for CHARMM/Amber expressions
#Coulomb energy and gradient in Bohrs
def coulombcharge(charges, coords):
    #Converting list to numpy array and converting to bohr
    coords_b=np.array(coords)*constants.ang2bohr
    #Coulomb energy
    energy=0
    #Initialize Coulomb gradient
    gradient = np.zeros((len(coords_b), 3))
    blankline()
    #Iterating over atom i
    for count_i,charge_i in enumerate(charges):
        #Iterating over atom j
        for count_j,charge_j in enumerate(charges):
            #Skipping if same atom
            if count_i != count_j:
                #Skipping identical pairs
                if count_i < count_j:
                    pairdistance=distance(coords_b[count_i],coords_b[count_j])
                    pairenergy=(charge_i*charge_j)/pairdistance
                    energy+=pairenergy
                    #Using electric field expression from: http://www.physnet.org/modules/pdf_modules/m115.pdf
                    Efield_pair_hat=np.array([(coords_b[count_i][0]-coords_b[count_j][0])/pairdistance,
                                              (coords_b[count_i][1]-coords_b[count_j][1])/pairdistance,
                                              (coords_b[count_i][2]-coords_b[count_j][2])/pairdistance ])
                    #Doing ij pair and storing contribution for each
                    Efield_pair_j=(Efield_pair_hat*charge_j)/(pairdistance**2)
                    Efield_pair_i = (Efield_pair_hat * charge_i) / (pairdistance**2)
                    gradient[count_i] += -1 * Efield_pair_j*charge_i
                    gradient[count_j] -= -1 * Efield_pair_i*charge_j
    return energy,gradient

#Combined Lennard-Jones and Coulomb
#Terribly written

def LJCoulpy(coords,atomtypes, charges, LJPairpotentials, connectivity=None):
    print("Inside LJCoulpy function")
    print("Calculating LJ pairs based on connectivity")
    if len(connectivity)==0:
        print("Warning!. No connectivity list present. Will treat all LJ pairs.")
    else:
        print(len(connectivity)," connectivity lists present")

    atomlist=list(range(0, len(coords)))
    #LJ energy
    energy=0
    #LJ gradient
    gradient = np.zeros((len(coords), 3))
    #Iterating over atom i
    for i in atomlist:
        #Iterating over atom j
        for j in atomlist:
            #Skipping if same atom
            if i != j:
                #Skipping identical pairs
                if i < j:

                    #Coulomb part
                    pairdistance_b=distance(coords_b[i],coords_b[j])
                    pairenergy=(charges[i]*charges[j])/pairdistance_b
                    Coulenergy+=pairenergy
                    #Using electric field expression from: http://www.physnet.org/modules/pdf_modules/m115.pdf
                    Efield_pair_hat=np.array([(coords_b[i][0]-coords_b[j][0])/pairdistance_b,
                                              (coords_b[i][1]-coords_b[j][1])/pairdistance_b,
                                              (coords_b[i][2]-coords_b[j][2])/pairdistance_b ])
                    #Doing ij pair and storing contribution for each
                    Efield_pair_j=(Efield_pair_hat*charges[j])/(pairdistance_b**2)
                    Efield_pair_i = (Efield_pair_hat * charges[i]) / (pairdistance_b**2)
                    Coulgradient[i] += -1 * Efield_pair_j*charges[i]
                    Coulgradient[j] -= -1 * Efield_pair_i*charges[j]


                    for l in LJPairpotentials:
                        #print("l:", l)
                        #This checks if i-j pair exists in LJPairpotentials list:
                        if set([atomtypes[i], atomtypes[j]]) == set([l[0],l[1]]):
                        #if atomtypes[i] in l and atomtypes[j] in l:
                            #print("COUNTING!!! unless...")
                            #Now checking connectivity for whether we should calculate LJ energy for pair or not
                            skip=False
                            for conn in connectivity:
                                #print("conn:", conn)
                                #If i,j in same list
                                if all(x in conn for x in [i, j]) == True:
                                    #print("Atoms connected. skipping ")
                                    skip=True
                                    continue
                            if skip == False:
                                #print("i : {}  and j : {}".format(i,j))
                                #print("atomtype_i : {}  and atomtype_j : {}".format(atomtypes[i],atomtypes[j]))
                                sigma=l[2]
                                eps=l[3]
                                #pairdistance = distance(coords[i], coords[j])
                                pairdistance=pairdistance_b*constants.bohr2ang
                                #print("sigma, eps, pairdistance", sigma,eps,pairdistance)
                                V_LJ=4*eps*((sigma/pairdistance)**12-(sigma/pairdistance)**6)
                                #print("V_LJ: {} kcal/mol  V_LJ: {} au:".format(V_LJ,V_LJ/constants.harkcal))
                                LJenergy+=V_LJ
                                #print("energy: {} kcal/mol  energy: {} au:".format(energy, energy / constants.harkcal))
                                #print("------------------------------")
                                #Typo in http://localscf.com/localscf.com/LJPotential.aspx.html ??
                                #Using http://www.courses.physics.helsinki.fi/fys/moldyn/lectures/L4.pdf
                                #TODO: Equation needs to be double-checked for correctness. L4.pdf equation ambiguous
                                #Check this: http://people.virginia.edu/~lz2n/mse627/notes/Potentials.pdf
                                LJgrad_const=(24*eps*((sigma/pairdistance)**6-2*(sigma/pairdistance)**12))*(1/(pairdistance**2))
                                #print("LJgrad_const ", LJgrad_const)
                                #print("LJgrad_const:", LJgrad_const)
                                gr=np.array([(coords[i][0] - coords[j][0])*LJgrad_const, (coords[i][1] - coords[j][1])*LJgrad_const,
                                     (coords[i][2] - coords[j][2])*LJgrad_const])
                                #print("gr:", gr)
                                LJgradient[i] += gr
                                LJgradient[j] -= gr
    #Convert gradient from kcal/mol per Å to hartree/Bohr
    LJfinal_gradient=LJgradient * (1/constants.harkcal) / constants.ang2bohr
    #print("LJ gradient (hartree/Bohr):", LJfinal_gradient)
    #Converg energy from kcal/mol to hartree
    LJfinal_energy=LJenergy*(1/constants.harkcal)
    print("LJ energy (hartree)", LJfinal_energy)

    #Coulomb
    print("Coulenergy : ", Coulenergy)

    final_energy = LJfinal_energy+Coulenergy

    final_gradient = LJfinal_gradient + Coulgradient

    return final_energy,final_gradient
