# YGGDRASILL - A GENERAL COMPCHEM AND QM/MM ENVIRONMENT

from constants import *
from elstructure_functions import *
import os
import glob
from functions_solv import *
from functions_coords import *
from functions_ORCA import *
from functions_general import *
import settings_yggdrasill
from functions_MM import *
from functions_optimization import *

def print_yggdrasill_header():
    programversion = 0.1
    #http://asciiflow.com
    #https://textik.com/#91d6380098664f89
    #https://www.gridsagegames.com/rexpaint/

    ascii_banner="""                                                                                                                                                                                                                                                                           
██╗   ██╗ ██████╗  ██████╗ ██████╗ ██████╗  █████╗ ███████╗██╗██╗     ██╗     
╚██╗ ██╔╝██╔════╝ ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██╔════╝██║██║     ██║     
 ╚████╔╝ ██║  ███╗██║  ███╗██║  ██║██████╔╝███████║███████╗██║██║     ██║     
  ╚██╔╝  ██║   ██║██║   ██║██║  ██║██╔══██╗██╔══██║╚════██║██║██║     ██║     
   ██║   ╚██████╔╝╚██████╔╝██████╔╝██║  ██║██║  ██║███████║██║███████╗███████╗
   ╚═╝    ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚══════╝
"""
    print(BC.WARNING,"----------------------------------------------------------------------------------",BC.END)
    print(BC.WARNING,"----------------------------------------------------------------------------------",BC.END)
    print(BC.OKBLUE,ascii_banner,BC.END)
    print(BC.WARNING,BC.BOLD,"YGGDRASILL version", programversion,BC.END)
    print(BC.WARNING,"A COMPCHEM AND QM/MM ENVIRONMENT", BC.END)
    print(BC.WARNING,"----------------------------------------------------------------------------------",BC.END)
    print(BC.WARNING,"----------------------------------------------------------------------------------",BC.END)


#Numerical frequencies class
class NumericalFrequencies:
    def __init__(self, fragment, theory, npoint=2, displacement=0.0005, hessatoms=[], numcores=1 ):
        self.numcores=numcores
        self.fragment=fragment
        self.theory=theory
        self.coords=fragment.coords
        self.elems=fragment.elems
        self.numatoms=len(self.elems)
        #Hessatoms list is allatoms (if not defined), otherwise the atoms provided and thus a partial Hessian is calculated.
        self.allatoms=list(range(0,self.numatoms))
        if hessatoms==[]:
            self.hessatoms=self.allatoms
        else:
            self.hessatoms=hessatoms
        self.npoint = npoint
        self.displacement=displacement
        self.displacement_bohr = self.displacement *constants.bohr2ang

    def run(self):
        print("Starting Numerical Frequencies job for fragment")
        print("System size:", self.numatoms)
        print("Hessian atoms:", self.hessatoms)
        if self.hessatoms != self.allatoms:
            print("This is a partial Hessian.")
        if self.npoint ==  1:
            print("One-point formula used (forward difference)")
        elif self.npoint == 2:
            print("Two-point formula used (central difference)")
        else:
            print("Unknown npoint option. npoint should be set to 1 (one-point) or 2 (two-point formula).")
            exit()
        print("Displacement: {:3.3f} Å ({:3.3f} Bohr)".format(self.displacement,self.displacement_bohr))
        blankline()
        print("Starting geometry:")
        #Converting to numpy array
        #TODO: get rid list->np-array conversion
        current_coords_array=np.array(self.coords)
        print_coords_all(current_coords_array, self.elems)
        blankline()

        #Looping over each atom and each coordinate to create displaced geometries
        #Only displacing atom if in hessatoms list. i.e. possible partial Hessian
        list_of_displaced_geos=[]
        list_of_displacements=[]
        for atom_index in range(0,len(current_coords_array)):
            if atom_index in self.hessatoms:
                for coord_index in range(0,3):
                    val=current_coords_array[atom_index,coord_index]
                    #Displacing in + direction
                    current_coords_array[atom_index,coord_index]=val+self.displacement
                    y = current_coords_array.copy()
                    list_of_displaced_geos.append(y)
                    list_of_displacements.append([atom_index, coord_index, '+'])
                    if self.npoint == 2:
                        #Displacing  - direction
                        current_coords_array[atom_index,coord_index]=val-self.displacement
                        y = current_coords_array.copy()
                        list_of_displaced_geos.append(y)
                        list_of_displacements.append([atom_index, coord_index, '-'])
                    #Displacing back
                    current_coords_array[atom_index, coord_index] = val

        #Looping over geometries and creating inputfiles.
        freqinputfiles=[]
        for disp, geo in zip(list_of_displacements,list_of_displaced_geos):
            atom_disp=disp[0]
            if disp[1] == 0:
                crd='x'
            elif disp[1] == 1:
                crd = 'y'
            elif disp[1] == 2:
                crd = 'z'
            drection=disp[2]
            displacement_jobname='Numfreq-Disp-'+'Atom'+str(atom_disp)+crd+drection
            print("Displacing Atom: {} Coordinate: {} Direction: {}".format(atom_disp, crd, drection))

            if type(self.theory)==ORCATheory:
                create_orca_input_plain(displacement_jobname, self.elems, geo, self.theory.orcasimpleinput,
                                    self.theory.orcablocks, self.theory.charge, self.theory.mult, Grad=True)
            elif type(self.theory)==QMMMTheory:
                print("QM/MM Theory for Numfreq in progress")
                exit()
            elif type(self.theory)==xTBTheory:
                print("xtb for Numfreq not implemented yet")
                exit()
            else:
                print("theory not implemented for numfreq yet")
                exit()
            freqinputfiles.append(displacement_jobname)
        #freqinplist is in order of atom, x/y/z-coordinates, direction etc.
        #e.g. Numfreq-Disp-Atom0x+,  Numfreq-Disp-Atom0x-, Numfreq-Disp-Atom0y+  etc.

        #Adding initial geometry to freqinputfiles list
        if type(self.theory) == ORCATheory:
            create_orca_input_plain('Originalgeo', self.elems, current_coords_array, self.theory.orcasimpleinput,
                                self.theory.orcablocks, self.theory.charge, self.theory.mult, Grad=True)
        elif type(self.theory) == QMMMTheory:
            print("QM/MM theory for Numfreq in progress")
        else:
            print("theory not implemented for numfreq yet")
            exit()
        freqinputfiles.append('Originalgeo')

        #Run all inputfiles in parallel by multiprocessing
        blankline()
        print("Starting Displacement calculations.")
        if type(self.theory) == ORCATheory:
            run_inputfiles_in_parallel(self.theory.orcadir, freqinputfiles, self.numcores)
        elif type(self.theory) == QMMMTheory:
            print("QM/MM theory for Numfreq in progress")
        else:
            print("theory not implemented for numfreq yet")
            exit()

        #Grab energy and gradient of original geometry. Only used for onepoint formula
        original_grad = gradientgrab('Originalgeo' + '.engrad')

        #If partial Hessian remove non-hessatoms part of gradient:
        #Get partial matrix by deleting atoms not present in list.
        original_grad=get_partial_matrix(self.allatoms, self.hessatoms, original_grad)
        original_grad_1d = np.ravel(original_grad)

        #Initialize Hessian
        hesslength=3*len(self.hessatoms)
        hessian=np.zeros((hesslength,hesslength))

        #Twopoint-formula Hessian. pos and negative directions come in order
        if self.npoint == 2:
            count=0; hessindex=0
            for file in freqinputfiles:
                if file != 'Originalgeo':
                    count+=1
                    if count == 1:
                        if type(self.theory) == ORCATheory:
                            grad_pos = gradientgrab(file + '.engrad')
                        else:
                            print("theory not implemented for numfreq yet")
                            exit()
                        # If partial Hessian remove non-hessatoms part of gradient:
                        grad_pos = get_partial_matrix(self.allatoms, self.hessatoms, grad_pos)
                        grad_pos_1d = np.ravel(grad_pos)
                    elif count == 2:
                        #Getting grad as numpy matrix and converting to 1d
                        if type(self.theory) == ORCATheory:
                            grad_neg=gradientgrab(file+'.engrad')
                        else:
                            print("theory not implemented for numfreq yet")
                            exit()
                        # If partial Hessian remove non-hessatoms part of gradient:
                        grad_neg = get_partial_matrix(self.allatoms, self.hessatoms, grad_neg)
                        grad_neg_1d = np.ravel(grad_neg)
                        Hessrow=(grad_pos_1d - grad_neg_1d)/(2*self.displacement_bohr)
                        hessian[hessindex,:]=Hessrow
                        grad_pos_1d=0
                        grad_neg_1d=0
                        count=0
                        hessindex+=1
                    else:
                        print("Something bad happened")
                        exit()
                blankline()

        #Onepoint-formula Hessian
        elif self.npoint == 1:
            for index,file in enumerate(freqinputfiles):
                #Skipping original geo
                if file != 'Originalgeo':
                    #Getting grad as numpy matrix and converting to 1d
                    if type(self.theory) == ORCATheory:
                        grad=gradientgrab(file+'.engrad')
                    else:
                        print("theory not implemented for numfreq yet")
                        exit()
                    # If partial Hessian remove non-hessatoms part of gradient:
                    grad = get_partial_matrix(self.allatoms, self.hessatoms, grad)
                    grad_1d = np.ravel(grad)
                    Hessrow=(grad_1d - original_grad_1d)/self.displacement_bohr
                    hessian[index,:]=Hessrow

        #Symmetrize Hessian by taking average of matrix and transpose
        symm_hessian=(hessian+hessian.transpose())/2
        self.hessian=symm_hessian

        #Write Hessian to file
        with open("Hessian", 'w') as hfile:
            hfile.write(str(hesslength)+' '+str(hesslength)+'\n')
            for row in self.hessian:
                rowline=' '.join(map(str, row))
                hfile.write(str(rowline)+'\n')
            blankline()
            print("Wrote Hessian to file: Hessian")
        #Write ORCA-style Hessian file
        write_ORCA_Hessfile(self.hessian, self.coords, self.elems, self.fragment.list_of_masses, self.hessatoms)

        #Project out Translation+Rotational modes
        #TODO

        #Diagonalize Hessian
        print("self.fragment.list_of_masses:", self.fragment.list_of_masses)
        print("self.elems:", self.elems)
        # Get partial matrix by deleting atoms not present in list.
        hesselems = get_partial_list(self.allatoms, self.hessatoms, self.elems)
        hessmasses = get_partial_list(self.allatoms, self.hessatoms, self.fragment.list_of_masses)
        print("hesselems", hesselems)
        print("hessmasses:", hessmasses)
        self.frequencies=diagonalizeHessian(self.hessian,hessmasses,hesselems)[0]

        #Print out normal mode output. Like in Chemshell or ORCA
        blankline()
        print("Normal modes:")
        print("Eigenvectors will be printed here")
        blankline()
        #Print out Freq output. Maybe print normal mode compositions here instead???
        printfreqs(self.frequencies,len(self.hessatoms))

        #Print out thermochemistry
        thermochemcalc(self.frequencies,self.hessatoms, self.fragment, self.theory.mult, temp=298.18,pressure=1)

        #TODO: https://pages.mtu.edu/~msgocken/ma5630spring2003/lectures/diff/diff/node6.html


#Molecular dynamics class
class MolecularDynamics:
    def __init__(self, fragment, theory, ensemble, temperature):
        self.fragment=fragment
        self.theory=theory
        self.ensemble=ensemble
        self.temperature=temperature
    def run(self):
        print("Molecular dynamics is not ready yet")
        exit()

def print_time_rel(timestampA,modulename=''):
    secsA=time.time()-timestampA
    minsA=secsA/60
    print("-------------------------------------------------------------------")
    print("Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secsA, minsA ))
    print("-------------------------------------------------------------------")

def print_time_rel_and_tot(timestampA,timestampB, modulename=''):
    secsA=time.time()-timestampA
    minsA=secsA/60
    #hoursA=minsA/60
    secsB=time.time()-timestampB
    minsB=secsB/60
    #hoursB=minsB/60
    print("-------------------------------------------------------------------")
    print("Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secsA, minsA ))
    print("Total Walltime: {:3.1f} seconds, {:3.1f} minutes.".format(secsB, minsB ))
    print("-------------------------------------------------------------------")

def print_time_rel_and_tot_color(timestampA,timestampB, modulename=''):
    secsA=time.time()-timestampA
    minsA=secsA/60
    #hoursA=minsA/60
    secsB=time.time()-timestampB
    minsB=secsB/60
    #hoursB=minsB/60
    print(BC.WARNING,"-------------------------------------------------------------------", BC.END)
    print(BC.WARNING,"Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secsA, minsA ), BC.END)
    print(BC.WARNING,"Total Walltime: {:3.1f} seconds, {:3.1f} minutes.".format(secsB, minsB ), BC.END)
    print(BC.WARNING,"-------------------------------------------------------------------", BC.END)

#Theory classes

# Different MM theories

# Simple nonbonded MM theory. Charges and LJ-potentials
class NonBondedTheory:
    def __init__(self, charges = [], atomtypes=[], LJ=False, forcefield=[], LJcombrule='geometric'):
        #These are charges for whole system including QM.
        self.atom_charges = charges
        #Possibly have self.mm_charges here also??

        #Read MM forcefield.
        self.forcefield=forcefield

        #Atom types
        self.atomtypes=atomtypes

        #If atomtypes and forcefield both defined then calculate pairpotentials
        if len(self.atomtypes) > 0:
            if len(self.forcefield) > 0:
                self.calculate_LJ_pairpotentials(LJcombrule)

    def calculate_LJ_pairpotentials(self,combination_rule='geometric'):
        import math
        print("Defining Lennard-Jones pair potentials")
        #Printlevel. Todo: Make more general
        printsetting='normal'
        #List to store pairpotentials
        self.LJpairpotentials=[]
        #New: multi-key dict instead using tuple
        self.LJpairpotdict={}
        if combination_rule == 'geometric':
            print("Using geometric mean for LJ pair potentials")
        elif combination_rule == 'arithmetic':
            print("Using geometric mean for LJ pair potentials")
        elif combination_rule == 'mixed_geoepsilon':
            print("Using mixed rule for LJ pair potentials")
            print("Using arithmetic rule for r/sigma")
            print("Using geometric rule for epsilon")
        elif combination_rule == 'mixed_geosigma':
            print("Using mixed rule for LJ pair potentials")
            print("Using geometric rule for r/sigma")
            print("Using arithmetic rule for epsilon")
        else:
            print("Unknown combination rule. Exiting")
            exit()

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
                    if printsetting=='Debug':
                        print("LJ sigma_i {} for atomtype {}:".format(self.forcefield[at_i].LJparameters[0], at_i))
                        print("LJ sigma_j {} for atomtype {}:".format(self.forcefield[at_j].LJparameters[0], at_j))
                        print("LJ eps_i {} for atomtype {}:".format(self.forcefield[at_i].LJparameters[1], at_i))
                        print("LJ eps_j {} for atomtype {}:".format(self.forcefield[at_j].LJparameters[1], at_j))
                        blankline()
                    if combination_rule=='geometric':
                        sigma=math.sqrt(self.forcefield[at_i].LJparameters[0]*self.forcefield[at_j].LJparameters[0])
                        epsilon=math.sqrt(self.forcefield[at_i].LJparameters[1]*self.forcefield[at_j].LJparameters[1])
                        if printsetting == 'Debug':
                            print("LJ sigma_ij : {} for atomtype-pair: {} {}".format(sigma,at_i, at_j))
                            print("LJ epsilon_ij : {} for atomtype-pair: {} {}".format(epsilon,at_i, at_j))
                            blankline()
                    elif combination_rule=='arithmetic':
                        if printsetting == 'Debug':
                            print("Using arithmetic mean for LJ pair potentials")
                            print("NOTE: to be confirmed")
                        sigma=0.5*(self.forcefield[at_i].LJparameters[0]+self.forcefield[at_j].LJparameters[0])
                        epsilon=0.5-(self.forcefield[at_i].LJparameters[1]+self.forcefield[at_j].LJparameters[1])
                    elif combination_rule=='mixed_geosigma':
                        if printsetting == 'Debug':
                            print("Using geometric mean for LJ sigma parameters")
                            print("Using arithmetic mean for LJ epsilon parameters")
                            print("NOTE: to be confirmed")
                        sigma=math.sqrt(self.forcefield[at_i].LJparameters[0]*self.forcefield[at_j].LJparameters[0])
                        epsilon=0.5-(self.forcefield[at_i].LJparameters[1]+self.forcefield[at_j].LJparameters[1])
                    elif combination_rule=='mixed_geoepsilon':
                        if printsetting == 'Debug':
                            print("Using arithmetic mean for LJ sigma parameters")
                            print("Using geometric mean for LJ epsilon parameters")
                            print("NOTE: to be confirmed")
                        sigma=0.5*(self.forcefield[at_i].LJparameters[0]+self.forcefield[at_j].LJparameters[0])
                        epsilon=math.sqrt(self.forcefield[at_i].LJparameters[1]*self.forcefield[at_j].LJparameters[1])
                    self.LJpairpotentials.append([at_i, at_j, sigma, epsilon])
                    #Dict using two keys (actually a tuple of two keys)
                    self.LJpairpotdict[(at_i,at_j)] = [sigma, epsilon]
                    #print(self.LJpairpotentials)
        print_time_rel(CheckpointTime)
        #Remove redundant pair potentials
        CheckpointTime = time.time()
        for acount, pairpot_a in enumerate(self.LJpairpotentials):
            for bcount, pairpot_b in enumerate(self.LJpairpotentials):
                if acount < bcount:
                    if set(pairpot_a) == set(pairpot_b):
                        del self.LJpairpotentials[bcount]
        print_time_rel(CheckpointTime)
        print("Final LJ pair potentials (sigma_ij, epsilon_ij):\n", self.LJpairpotentials)
        print("New: LJ pair potentials as dict:")
        print("self.LJpairpotdict:", self.LJpairpotdict)

        #Create numatomxnumatom array of eps and sigma
        #Todo: rewrite in Fortran like in Abin.
        numatoms=len(self.atomtypes)
        self.sigmaij = np.zeros((numatoms, numatoms))
        self.epsij = np.zeros((numatoms, numatoms))
        print("Creating epsij and sigmaij arrays")
        beginTime = time.time()

        CheckpointTime = time.time()
        for i in range(numatoms):
            for j in range(numatoms):
                for ljpot in self.LJpairpotentials:
                    if self.atomtypes[i] == ljpot[0] and self.atomtypes[j] == ljpot[1]:
                        #print("Here")
                        self.sigmaij[i, j] = ljpot[2]
                        self.epsij[i, j] = ljpot[3]
                    elif self.atomtypes[j] == ljpot[0] and self.atomtypes[i] == ljpot[1]:
                        #print("here 2")
                        self.sigmaij[i, j] = ljpot[2]
                        self.epsij[i, j] = ljpot[3]
        print("self.sigmaij:", self.sigmaij)
        print("self.epsij:", self.epsij)
        print_time_rel(CheckpointTime)

    def update_charges(self,charges):
        print("Updating charges.")
        self.atom_charges = charges
        print("Charges are now:", charges)
    #Provide specific coordinates (MM region) and charges (MM region) upon run
    def run(self, full_coords=[], mm_coords=[], charges=[], connectivity=[], LJ=True, Coulomb=True, Grad=True, version='py'):

        #If charges not provided to run function. Use object charges
        if len(charges)==0:
            charges=self.atom_charges

        #If coords not provided to run function. Use object coords
        #HMM. I guess we are not keeping coords as part of MMtheory?
        #if len(full_cords)==0:
        #    full_coords=

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING NONBONDED MM CODE-------------", BC.END)
        print("Calculating MM energy and gradient")
        #initializing
        self.Coulombchargeenergy=0
        self.LJenergy=0
        self.MMGradient=[]
        self.Coulombchargegradient=[]
        self.LJgradient=[]

        #Slow Python version or fast Fortran version
        if version=='py':
            print("Using slow Python MM code")
            #Sending full coords and charges over. QM charges are set to 0.
            if Coulomb==True:
                self.Coulombchargeenergy, self.Coulombchargegradient  = coulombcharge(charges, full_coords)
                print("Coulomb Energy (au):", self.Coulombchargeenergy)
                print("Coulomb Energy (kcal/mol):", self.Coulombchargeenergy * constants.harkcal)
                print("")
                print("self.Coulombchargegradient:", self.Coulombchargegradient)
                blankline()
            # NOTE: Lennard-Jones should  calculate both MM-MM and QM-MM LJ interactions. Full coords necessary.
            if LJ==True:
                self.LJenergy,self.LJgradient = LennardJones(full_coords,self.atomtypes, self.LJpairpotentials, connectivity=connectivity)
                print("Lennard-Jones Energy (au):", self.LJenergy)
                print("Lennard-Jones Energy (kcal/mol):", self.LJenergy*constants.harkcal)
            self.MMEnergy = self.Coulombchargeenergy+self.LJenergy

            if Grad==True:
                self.MMGradient = self.Coulombchargegradient+self.LJgradient

        elif version=='f2py':
            print("Using fast Fortran F2Py MM code")
            try:
                import LJCoulombv1
                print(LJCoulombv1.__doc__)
                print("----------")
            except:
                print("Fortran library LJCoulombv1 not found! Make sure you have run the installation script.")
            self.MMEnergy, self.MMGradient, self.LJenergy, self.Coulombchargeenergy =\
                LJCoulomb(full_coords, self.epsij, self.sigmaij, charges, connectivity=connectivity)
        elif version=='f2pyv2':
            print("Using fast Fortran F2Py MM code v2")
            try:
                import LJCoulombv2
                print(LJCoulombv2.__doc__)
                print("----------")
            except:
                print("Fortran library LJCoulombv1 not found! Make sure you have run the installation script.")
            self.MMEnergy, self.MMGradient, self.LJenergy, self.Coulombchargeenergy =\
                LJCoulombv2(full_coords, self.epsij, self.sigmaij, charges, connectivity=connectivity)

        else:
            print("Unknown version of MM code")

        print("Lennard-Jones Energy (au):", self.LJenergy)
        print("Lennard-Jones Energy (kcal/mol):", self.LJenergy * constants.harkcal)
        print("Coulomb Energy (au):", self.Coulombchargeenergy)
        print("Coulomb Energy (kcal/mol):", self.Coulombchargeenergy * constants.harkcal)
        print("MM Energy:", self.MMEnergy)
        print("self.MMGradient:", self.MMGradient)

        print(BC.OKBLUE, BC.BOLD, "------------ENDING NONBONDED MM CODE-------------", BC.END)
        return self.MMEnergy, self.MMGradient

#QM/MM theory object.
#Required at init: qm_theory and qmatoms. Fragment not. Can come later
class QMMMTheory:
    def __init__(self, qm_theory, qmatoms, fragment='', mm_theory="" , atomcharges="", embedding="Elstat", printlevel=3):
        print(BC.WARNING,BC.BOLD,"------------Defining QM/MM object-------------", BC.END)
        #Theory level definitions
        self.printlevel=printlevel
        self.qm_theory=qm_theory
        self.qm_theory_name = self.qm_theory.__class__.__name__
        self.mm_theory=mm_theory
        self.mm_theory_name = self.mm_theory.__class__.__name__
        if self.mm_theory_name == "str":
            self.mm_theory_name="None"
        print("QM-theory:", self.qm_theory_name)
        print("MM-theory:", self.mm_theory_name)

        #Embedding type: mechanical, electrostatic etc.
        self.embedding=embedding
        print("Embedding:", self.embedding)
        self.qmatoms = qmatoms
        #If fragment object has been defined
        if fragment != '':
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
            self.connectivity=fragment.connectivity

            # Region definitions
            self.allatoms=list(range(0,len(self.elems)))

            self.qmcoords=[self.coords[i] for i in self.qmatoms]
            self.qmelems=[self.elems[i] for i in self.qmatoms]
            self.mmatoms=listdiff(self.allatoms,self.qmatoms)

            self.mmcoords=[self.coords[i] for i in self.mmatoms]
            self.mmelems=[self.elems[i] for i in self.mmatoms]
            print("List of all atoms:", self.allatoms)
            print("QM region:", self.qmatoms)
            print("MM region", self.mmatoms)
            blankline()

            #List of QM and MM labels
            self.hybridatomlabels=[]
            for i in self.allatoms:
                if i in self.qmatoms:
                    self.hybridatomlabels.append('QM')
                elif i in self.mmatoms:
                    self.hybridatomlabels.append('MM')

            print("atomcharges:", atomcharges)
            # Charges defined for regions
            self.qmcharges=[atomcharges[i] for i in self.qmatoms]
            print("self.qmcharges:", self.qmcharges)
            self.mmcharges=[atomcharges[i] for i in self.mmatoms]
            print("self.mmcharges:", self.mmcharges)

        if mm_theory != "":
            if self.embedding=="Elstat":
                print("X")
                #Setting QM charges to 0 since electrostatic embedding
                self.charges=[]
                for i,c in enumerate(atomcharges):
                    if i in self.mmatoms:
                        self.charges.append(c)
                    else:
                        self.charges.append(0.0)
                mm_theory.update_charges(self.charges)
                print("Charges of QM atoms set to 0 (since Electrostatic Embedding):")
                for i in self.allatoms:
                    if i in qmatoms:
                        print("QM atom {} charge: {}".format(i, self.charges[i]))
                    else:
                        print("MM atom {} charge: {}".format(i, self.charges[i]))
            blankline()

    def run(self, current_coords=[], elems=[], Grad=False, nprocs=1):
        print(BC.WARNING, BC.BOLD, "------------RUNNING QM/MM MODULE-------------", BC.END)
        print("QM Module:", self.qm_theory_name)
        print("MM Module:", self.mm_theory_name)
        #If no coords provided to run (from Optimizer or NumFreq or MD) then use coords associated with object.
        if len(current_coords) != 0:
            pass
        else:
            current_coords=self.coords

        if self.embedding=="Elstat":
            PC=True
        else:
            PC=False
        #Updating QM coords and MM coords.
        #TODO: Should we use different name for updated QMcoords and MMcoords here??
        self.qmcoords=[current_coords[i] for i in self.qmatoms]
        self.mmcoords=[current_coords[i] for i in self.mmatoms]
        if self.qm_theory_name=="ORCATheory":
            #Calling ORCA theory, providing current QM and MM coordinates.
            if Grad==True:
                if PC==True:
                    self.QMEnergy, self.QMgradient, self.PCgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                                                         current_MM_coords=self.mmcoords,
                                                                                         MMcharges=self.mmcharges,
                                                                                         qm_elems=self.qmelems, mm_elems=self.mmelems,
                                                                                         Grad=True, PC=True, nprocs=nprocs)
                else:
                    self.QMEnergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=True, PC=False, nprocs=nprocs)
            else:
                self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=False, PC=PC, nprocs=nprocs)
        elif self.qm_theory_name == "Psi4Theory":
            print("recently implemented")
            #Calling Psi4 theory, providing current QM and MM coordinates.
            if Grad==True:
                print("Grad true")
                if PC==True:
                    print("Grad. PC-embedding true. not rady")
                    exit()
                    #self.QMEnergy, self.QMgradient, self.PCgradient = self.qm_theory.run(current_coords=self.qmcoords,
                    #                                                                     current_MM_coords=self.mmcoords,
                    #                                                                     MMcharges=self.mmcharges,
                    #                                                                     qm_elems=self.qmelems, mm_elems=self.mmelems,
                    #                                                                     Grad=True, PC=True, nprocs=nprocs)
                elif PE==True:
                    print("Grad. PolEmbedding true. not ready")
                    exit()
                else:
                    print("grad. mechh embedding. not ready")
                    exit()
                    self.QMEnergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=True, PC=False, nprocs=nprocs)
            else:
                print("grad false.")
                if PC == True:
                    print("PC embed true. not ready")
                    self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=False, PC=PC, nprocs=nprocs)
                elif PE==True:
                    print("PolEmbed true")

                else:
                    print("mech true", not ready)
                    exit()


        elif self.qm_theory_name == "xTBTheory":
            print("not yet implemented")
        elif self.qm_theory_name == "DaltonTheory":
            print("not yet implemented")
        elif self.qm_theory_name == "NWChemtheory":
            print("not yet implemented")
        else:
            print("invalid QM theory")

        # MM theory
        if self.mm_theory_name == "NonBondedTheory":
            self.MMEnergy, self.MMGradient= self.mm_theory.run(full_coords=self.coords, mm_coords=self.mmcoords, charges=self.charges, connectivity=self.connectivity)
            #self.MMEnergy=self.mm_theory.MMEnergy
            #if Grad==True:
            #    self.MMGrad = self.mm_theory.MMGrad
            #    print("self.MMGrad:", self.MMGrad)
        else:
            self.MMEnergy=0

        #Final QM/MM Energy
        self.QM_MM_Energy= self.QMEnergy+self.MMEnergy
        blankline()
        print("{:<20} {:>20.12f}".format("QM energy: ",self.QMEnergy))
        print("{:<20} {:>20.12f}".format("MM energy: ", self.MMEnergy))
        print("{:<20} {:>20.12f}".format("QM/MM energy: ", self.QM_MM_Energy))
        blankline()

        #Final QM/MM gradient. Combine QM gradient, MM gradient and PC-gradient (elstat MM gradient from QM code).
        #First combining QM and PC gradient to one.
        if Grad == True:
            self.QM_PC_Gradient = np.zeros((len(self.allatoms), 3))
            qmcount=0;pccount=0
            for i in self.allatoms:
                if i in self.qmatoms:
                    self.QM_PC_Gradient[i]=self.QMgradient[qmcount]
                    qmcount+=1
                else:
                    self.QM_PC_Gradient[i] = self.PCgradient[pccount]
                    pccount += 1
            #Now assemble final QM/MM gradient
            self.QM_MM_Gradient=self.QM_PC_Gradient+self.MMGradient

            if self.printlevel==3:
                print("QM+PC gradient (au/Bohr):")
                print_coords_all(self.QM_PC_Gradient, self.elems, self.allatoms)
                blankline()
                print("MM gradient (au/Bohr):")
                print_coords_all(self.MMGradient, self.elems, self.allatoms)
                blankline()
                print("Total QM/MM gradient (au/Bohr):")
                print_coords_all(self.QM_MM_Gradient, self.elems,self.allatoms)
            print(BC.WARNING,BC.BOLD,"------------ENDING QM/MM MODULE-------------",BC.END)
            return self.QM_MM_Energy, self.QM_MM_Gradient
        else:
            return self.QM_MM_Energy



#ORCA Theory object. Fragment object is optional. Only used for single-points.
class ORCATheory:
    def __init__(self, orcadir, fragment='', charge='', mult='', orcasimpleinput='',
                 orcablocks='', extraline='', brokensym=None, HSmult=None, atomstoflip=[]):
        self.orcadir = orcadir
        if fragment != '':
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge!='':
            self.charge=int(charge)
        if mult!='':
            self.mult=int(mult)
        self.orcasimpleinput=orcasimpleinput
        self.orcablocks=orcablocks
        self.extraline=extraline
        self.brokensym=brokensym
        self.HSmult=HSmult
        self.atomstoflip=atomstoflip
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old ORCA files")
        try:
            os.remove(self.inputfilename+'.gbw')
            os.remove(self.inputfilename + '.ges')
            os.remove(self.inputfilename + '.prop')
            os.remove(self.inputfilename + '.uco')
            os.remove(self.inputfilename + '_property.txt')
            #os.remove(self.inputfilename + '.out')
            for tmpfile in glob.glob("self.inputfilename*tmp"):
                os.remove(tmpfile)
        except:
            pass
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=[], current_MM_coords=[], MMcharges=[], qm_elems=[],
            mm_elems=[], elems=[], Grad=False, PC=False, nprocs=1 ):
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING ORCA INTERFACE-------------", BC.END)
        #Coords provided to run or else taken from initialization.
        if len(current_coords) != 0:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems == []:
            if elems == []:
                qm_elems=self.elems
            else:
                qm_elems = elems

        #Create inputfile with generic name
        self.inputfilename="orca-input"
        print("Creating inputfile:", self.inputfilename+'.inp')
        print("ORCA input:")
        print(self.orcasimpleinput)
        print(self.extraline)
        print(self.orcablocks)
        if PC==True:
            print("Pointcharge embedding is on!")
            create_orca_pcfile(self.inputfilename, mm_elems, current_MM_coords, MMcharges)
            if self.brokensym==True:
                create_orca_input_pc(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        self.charge, self.mult, extraline=self.extraline, HSmult=self.HSmult,
                                     atomstoflip=self.atomstoflip)
            else:
                create_orca_input_pc(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        self.charge, self.mult, extraline=self.extraline)
        else:
            if self.brokensym == True:
                create_orca_input_plain(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        self.charge,self.mult, extraline=self.extraline, HSmult=self.HSmult,
                                     atomstoflip=self.atomstoflip)
            else:
                create_orca_input_plain(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        self.charge,self.mult, extraline=self.extraline)

        #Run inputfile using ORCA parallelization. Take nprocs argument.
        #print(BC.OKGREEN, "------------Running ORCA calculation-------------", BC.END)
        print(BC.OKGREEN, "ORCA Calculation started.", BC.END)
        # Doing gradient or not.
        if Grad == True:
            run_orca_SP_ORCApar(self.orcadir, self.inputfilename + '.inp', nprocs=nprocs, Grad=True)
        else:
            run_orca_SP_ORCApar(self.orcadir, self.inputfilename + '.inp', nprocs=nprocs)
        #print(BC.OKGREEN, "------------ORCA calculation done-------------", BC.END)
        print(BC.OKGREEN, "ORCA Calculation done.", BC.END)

        #Now that we have possibly run a BS-DFT calculation, turning Brokensym off for future calcs (opt, restart, etc.)
        #TODO: Possibly use different flag for this???
        self.brokensym=False

        #Check if finished. Grab energy and gradient
        outfile=self.inputfilename+'.out'
        engradfile=self.inputfilename+'.engrad'
        pcgradfile=self.inputfilename+'.pcgrad'
        if checkORCAfinished(outfile) == True:
            self.energy=finalenergygrab(outfile)

            if Grad == True:
                self.grad=gradientgrab(engradfile)
                if PC == True:
                    #Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                    self.pcgrad=pcgradientgrab(pcgradfile)
                    print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                    return self.energy, self.grad, self.pcgrad
                else:
                    print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                    return self.energy, self.grad

            else:
                print("Single-point ORCA energy:", self.energy)
                print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                return self.energy
        else:
            print(BC.FAIL,"Problem with ORCA run", BC.END)
            print(BC.OKBLUE,BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
            exit()



#Psi4 Theory object. Fragment object is optional. Only used for single-points.
#PSI4 runmode:
#   : library means that Yggdrasill will load Psi4 libraries and run psi4 directly
#   : inputfile means that Yggdrasill will create Psi4 inputfile and run a separate psi4 executable
#psi4dir only necessary for inputfile-based userinterface
#printsetting is by default set to 'File. Change to something else for stdout print
# PE: Polarizable embedding (CPPE). Pass pe_modulesettings dict as well
class Psi4Theory:
    def __init__(self, fragment='', charge='', mult='', printsetting='False', psi4settings='', psi4functional='',
                 runmode='library', psi4dir='', pe=False, potfile=''):

        self.printsetting=printsetting
        print("printsetting:", printsetting)
        print("self.printsetting:", self.printsetting)
        self.runmode=runmode
        #CPPE Polarizable Embedding options
        self.pe=pe
        #Potfile from user or passed on via QM/MM Theory object ?
        self.potfile=potfile
        if self.runmode != 'library':
            try:
                self.psi4dir = psi4dir
            except:
                print("For Psi4 runmode=inputfile interface, the psi4dir variable needs to included")
                exit()

        if fragment != '':
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge!='':
            self.charge=int(charge)
        if mult!='':
            self.mult=int(mult)
        self.psi4settings=psi4settings
        self.psi4functional=psi4functional
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old Psi4 files")
        try:
            os.remove('psi4output.dat')
            os.remove('timer.dat')
        except:
            pass
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=[], current_MM_coords=[], MMcharges=[], qm_elems=[],
            mm_elems=[], elems=[], Grad=False, PC=False, nprocs=1 ):

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING PSI4 INTERFACE-------------", BC.END)

        #Coords provided to run or else taken from initialization.
        if len(current_coords) != 0:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems == []:
            if elems == []:
                qm_elems=self.elems
            else:
                qm_elems = elems


        #PSI4 runmode:
        #   : library means that Yggdrasill will load Psi4 libraries and run psi4 directly
        #   : inputfile means that Yggdrasill will create Psi4 inputfile and run a separate psi4 executable

        if self.runmode=='library':
            try:
                import psi4
            except:
                print(BC.FAIL,"Problem importing psi4. Make sure psi4 has been installed as part of same Python as Yggdrasill", BC.END)
                print(BC.WARNING,"If problematic, switch to inputfile based Psi4 interface: NOT YET READY", BC.END)
                exit()

            #Printing to output or not:
            #print("self.printsetting:", self.printsetting)
            if self.printsetting:
                print("Printsetting = True. Printing output to stdout...")
            else:
                print("Printsetting = False. Printing output to file: psi4output.dat ")
                psi4.core.set_output_file('psi4output.dat', False)

            #Creating Psi4 molecule object using lists and manual information
            print("elems:", elems)
            print("current_coords:", current_coords)
            print("current_MM_coords:", current_MM_coords)
            print("MMcharges:", MMcharges)
            psi4molfrag = psi4.core.Molecule.from_arrays(
                elez=elemstonuccharges(elems),
                fix_com=True,
                fix_orientation=True,
                fix_symmetry='c1',
                molecular_charge=self.charge,
                molecular_multiplicity=self.mult,
                geom=current_coords)
            psi4.activate(psi4molfrag)


            #Adding MM charges as pointcharges
            if PC==True:
                psi4.Chrgfield = QMMM()
                #Mmcoords are input in Bohr
                for mmcharge,mmcoord in zip(MMcharges,current_MM_coords):
                    psi4.Chrgfield.extern.addCharge(mmcharge, mmcoord[0]*constants.ang2bohr,
                                                    mmcoord[1]*constants.ang2bohr, mmcoord[2]*constants.ang2bohr)
                psi4.set_global_option_python('EXTERN', Chrgfield.extern)

            #Setting inputvariables
            #Todo: make memory psi4-interface variable ?
            psi4.set_memory('500 MB')

            #Changing charge and multiplicity
            #psi4molfrag.set_molecular_charge(self.charge)
            #psi4molfrag.set_multiplicity(self.mult)

            #Reading dict object with basic settings and passing to Psi4
            psi4.set_options(self.psi4settings)

            #Reading module options dict and passing to Psi4
            #TODO: Make one for SCF, CC, PCM etc.
            #psi4.set_module_options(modulename, moduledict)

            #Reading PE module options if PE=True
            if self.pe==True:
                print(BC.OKGREEN,"Polarizable Embedding Option On! Using CPPE module inside Psi4", BC.END)
                print(BC.WARNING, "Potfile: ", self.potfile, BC.END)
                try:
                    if os.path.exists(self.potfile):
                        pass
                    else:
                        print(BC.FAIL, "Potfile: ", self.potfile, "does not exist!", BC.END)
                        exit()
                except:
                    exit()
                psi4.set_module_options('pe', {'potfile' : self.potfile})

            #Controlling OpenMP parallelization. Controlled here, not via OMP_NUM_THREADS etc.
            psi4.set_num_threads(nprocs)

            #Running energy or energy+gradient. Currently hardcoded to SCF-DFT jobs
            #TODO: Support pointcharges and PE embedding
            if Grad==True:
                grad=psi4.gradient('scf', dft_functional=self.psi4functional)
                self.gradient=np.array(grad)
                self.energy = psi4.variable("CURRENT ENERGY")
            else:
                self.energy=psi4.energy('scf', dft_functional=self.psi4functional)

            #TODO: write in error handling here

            print(BC.OKBLUE, BC.BOLD, "------------ENDING PSI4-INTERFACE-------------", BC.END)

            if Grad == True:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy, self.gradient
            else:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy

        #INPUT-FILE BASED INTERFACE: TODO: finish
        else:
            print("PSI4 Run Mode: Inputfile based")
            print("Not complete yet...")
            exit()
            #Create Psi4 inputfile with generic name
            self.inputfilename="orca-input"
            print("Creating inputfile:", self.inputfilename+'.inp')
            print("ORCA input:")
            print(self.orcasimpleinput)
            print(self.extraline)
            print(self.orcablocks)
            if PC==True:
                print("Pointcharge embedding is on!")
                create_psi4_pcfile(self.inputfilename, mm_elems, current_MM_coords, MMcharges)
                create_psi4_input_pc(self.inputfilename, qm_elems, current_coords, self.psi4settings,
                                        self.charge, self.mult)
            else:
                create_psi4_input_plain(self.inputfilename, qm_elems, current_coords, self.psi4settings,
                                        self.charge,self.mult)


            #Run inputfile using Psi4 parallelization. Take nprocs argument.
            print(BC.OKGREEN, "Psi4 Calculation started.", BC.END)
            # Doing gradient or not.
            if Grad == True:
                run_orca_SP_Psi4par(self.psi4dir, self.inputfilename + '.inp', nprocs=nprocs, Grad=True)
            else:
                run_orca_SP_Psi4par(self.psi4dir, self.inputfilename + '.inp', nprocs=nprocs)
            #print(BC.OKGREEN, "------------ORCA calculation done-------------", BC.END)
            print(BC.OKGREEN, "Psi4 Calculation done.", BC.END)

            #Check if finished. Grab energy and gradient
            outfile=self.inputfilename+'.out'
            engradfile=self.inputfilename+'.engrad'
            pcgradfile=self.inputfilename+'.pcgrad'
            if checkPsi4finished(outfile) == True:
                self.energy=finalenergygrab(outfile)

                if Grad == True:
                    self.grad=gradientgrab(engradfile)
                    if PC == True:
                        #Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                        self.pcgrad=pcgradientgrab(pcgradfile)
                        print(BC.OKBLUE,BC.BOLD,"------------ENDING PSI4-INTERFACE-------------", BC.END)
                        return self.energy, self.grad, self.pcgrad
                    else:
                        print(BC.OKBLUE,BC.BOLD,"------------ENDING PSI4-INTERFACE-------------", BC.END)
                        return self.energy, self.grad

                else:
                    print("Single-point PSI4 energy:", self.energy)
                    print(BC.OKBLUE,BC.BOLD,"------------ENDING PSI4-INTERFACE-------------", BC.END)
                    return self.energy
            else:
                print(BC.FAIL,"Problem with Psi4 run", BC.END)
                print(BC.OKBLUE,BC.BOLD, "------------ENDING PSI4-INTERFACE-------------", BC.END)
                exit()




# Fragment class
class Fragment:
    def __init__(self, coordsstring=None, xyzfile=None, pdbfile=None, coords=None, elems=None):
        print("Defining new Yggdrasill fragment object")
        self.energy = None
        self.elems=[]
        self.coords=[]
        self.connectivity=[]
        self.atomcharges = []
        #TODO: Not sure if we use or not
        self.atomtypes = []
        if coords is not None:
            self.coords=coords
            self.elems=elems
        #If coordsstring given, read elems and coords from it
        elif coordsstring is not None:
            self.add_coords_from_string(coordsstring)
        #If xyzfile argument, run read_xyzfile
        elif xyzfile is not None:
            self.read_xyzfile(xyzfile)
        elif pdbfile is not None:
            self.read_pdbfile(pdbfile)
        if coords is not None:
            self.nuccharge = nucchargelist(self.elems)
            self.numatoms = len(self.coords)
            self.atomlist = list(range(0, self.numatoms))
            self.allatoms = self.atomlist
            self.mass = totmasslist(self.elems)
            self.list_of_masses = list_of_masses(self.elems)
    #Add coordinates from geometry string. Will replace.
    def add_coords_from_string(self, coordsstring):
        print("Getting coordinates from string:", coordsstring)
        if len(self.coords)>0:
            print("Fragment already contains coordinates")
            print("Adding extra coordinates")
        coordslist=coordsstring.split('\n')
        for count, line in enumerate(coordslist):
            if len(line)> 1:
                self.elems.append(line.split()[0])
                self.coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        self.nuccharge = nucchargelist(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(0, self.numatoms))
        self.allatoms = self.atomlist
        self.mass = totmasslist(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
        self.calc_connectivity()
    #Replace coordinates by providing elems and coords lists.
    def replace_coords(self, elems, coords):
        print("Replacing coordinates in fragment.")
        self.elems=elems
        self.coords=coords
        self.nuccharge = nucchargelist(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(0, self.numatoms))
        self.allatoms = self.atomlist
        self.mass = totmasslist(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
        self.calc_connectivity()
    def delete_coords(self):
        self.coords=[]
        self.elems=[]
        self.connectivity=[]
    def add_coords(self, elems,coords):
        print("Adding coordinates to fragment.")
        if len(self.coords)>0:
            print("Fragment already contains coordinates")
            print("Adding extra coordinates")
        self.elems = self.elems+elems
        self.coords = self.coords+coords
        self.nuccharge = nucchargelist(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(0, self.numatoms))
        self.allatoms = self.atomlist
        self.mass = totmasslist(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
        self.calc_connectivity()
    def print_coords(self):
        print("Defined coordinates (Å):")
        print_coords_all(self.coords,self.elems)
    #Read XYZ file
    def read_pdbfile(self,filename):
        print("Reading coordinates from PDBfile \"{}\" into fragment".format(filename))
        residuelist=[]
        #If elemcolumn found
        elemcol=[]
        #TODO: Are there different PDB formats?
        with open(filename) as f:
            for line in f:
                if 'ATOM' in line:
                    self.coords.append([float(line.split()[6]), float(line.split()[7]), float(line.split()[8])])
                    elemcol.append(line.split()[-1])
                    residuelist.append(line.split()[3])
        if len(elemcol) != len(self.coords):
            print("did not find same number of elements as coordinates")
            print("Need to define elements in some other way")
            exit()
        else:
            self.elems=elemcol

        self.nuccharge = nucchargelist(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(0, self.numatoms))
        self.allatoms = self.atomlist
        self.mass = totmasslist(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
        self.calc_connectivity()

    #Read XYZ file
    def read_xyzfile(self,filename):
        print("Reading coordinates from XYZfile {} into fragment".format(filename))
        with open(filename) as f:
            for count,line in enumerate(f):
                if count == 0:
                    self.numatoms=int(line.split()[0])
                if count > 1:
                    if len(line) > 3:
                        self.elems.append(line.split()[0])
                        self.coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        self.nuccharge = nucchargelist(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(0, self.numatoms))
        self.allatoms = self.atomlist
        self.mass = totmasslist(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
        self.calc_connectivity()
    # Get coordinates for specific atoms (from list of atom indices)
    def get_coords_for_atoms(self, atoms):
        subcoords=[self.coords[i] for i in atoms]
        subelems=[self.elems[i] for i in atoms]
        return subcoords,subelems
    #Calculate connectivity (list of lists) of coords
    def calc_connectivity(self, conndepth=99, scale=None, tol=None ):
        print("Calculating connectivity of fragment...")
        print("Number of atoms:", len(self.coords))

        if len(self.coords) > 10000:
            print("Atom number > 10K. Connectivity calculation could take a while")

        if scale == None:
            scale=settings_yggdrasill.scale
            tol=settings_yggdrasill.tol
            print("Using global scale and tol parameters")

        print("Scale:", scale)
        print("Tol:", tol)
        # Calculate connectivity by looping over all atoms
        found_atoms = []
        fraglist = []
        count = 0
        #Todo: replace by Fortran code? Pretty slow for 10K atoms
        timestampA=time.time()
        for atom in range(0, len(self.elems)):
            if atom not in found_atoms:
                count += 1
                members = get_molecule_members_loop_np2(self.coords, self.elems, conndepth, scale,
                                                        tol, atomindex=atom)
                if members not in fraglist:
                    fraglist.append(members)
                    found_atoms += members
        print_time_rel(timestampA)
        #flat_fraglist = [item for sublist in fraglist for item in sublist]
        self.connectivity=fraglist
        #Calculate number of atoms in connectivity list of lists
        conn_number_sum=0
        for l in self.connectivity:
            conn_number_sum+=len(l)
        if self.numatoms != conn_number_sum:
            print("Connectivity problem")
            exit()
        self.connected_atoms_number=conn_number_sum
    def update_atomcharges(self, charges):
        self.atomcharges = charges
    #Adding fragment-type info (used by molcrys, identifies whether atom is mainfrag, counterfrag1 etc.)
    def add_fragment_type_info(self,fragmentobjects):
        # Create list of fragment-type label-list
        self.fragmenttype_labels = []
        for i in self.atomlist:
            for count,fobject in enumerate(fragmentobjects):
                if i in fobject.flat_clusterfraglist:
                    self.fragmenttype_labels.append(count)
    def write_xyzfile(self,xyzfilename="Fragment-xyzfile.xyz"):

        with open(xyzfilename, 'w') as ofile:
            ofile.write(str(len(self.elems)) + '\n')
            ofile.write("title" + '\n')
            for el, c in zip(self.elems, self.coords):
                line = "{:4} {:12.6f} {:12.6f} {:12.6f}".format(el, c[0], c[1], c[2])
                ofile.write(line + '\n')
        print("Wrote XYZ file:", xyzfilename)
    #Print system-fragment information to file. Default name of file: "fragment-info
    def print_system(self,filename='fragment-info.txt'):
        print("Printing fragment information to disk:", filename)

        with open(filename, 'w') as outfile:
            outfile.write("Fragment: \n")
            outfile.write("Num elems: {}\n".format(len(self.elems)))
            outfile.write("Num coords: {}\n".format(len(self.coords)))
            outfile.write("Num atoms: {}\n".format(self.numatoms))
            outfile.write("\n")
            outfile.write("Index Atom            x             y             z           charge        fragment-type\n")
            outfile.write("-----------------------------------------------------------------------------------------\n")
            #TODO: Add residue-fraglist-number as last column
            for at, el, coord, charge, label in zip(self.atomlist, self.elems,self.coords,self.atomcharges, self.fragmenttype_labels):
                line="{:6} {:6}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f} {:6d}\n".format(at, el,coord[0], coord[1], coord[2], charge, label)
                outfile.write(line)
            outfile.write("elems: {}\n".format(self.elems))
            outfile.write("coords: {}\n".format(self.coords))
            #outfile.write("list of masses: {}\n".format(self.list_of_masses))
            outfile.write("atomcharges: {}\n".format(self.atomcharges))
            outfile.write("Sum of atomcharges: {}\n".format(sum(self.atomcharges)))
            outfile.write("connectivity: {}\n".format(self.connectivity))
    def set_energy(self,energy):
        self.energy=float(energy)

class xTBTheory:
    def __init__(self, xtbdir, fragment=None, charge=None, mult=None, xtbmethod=None):
        self.xtbdir = xtbdir
        self.fragment=fragment
        self.coords=fragment.coords
        self.elems=fragment.elems
        self.charge=charge
        self.mult=mult
        self.xtbmethod=xtbmethod
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old xTB files")
        try:
            os.remove('xtb-inpfile.xyz')
            os.remove('xtb-inpfile.out')
            os.remove('gradient')
            os.remove('charges')
            os.remove('energy')
            #TODO: Add restart function so that xtbrestart is not always deleted
            os.remove('xtbrestart')
        except:
            pass
    def run(self, current_coords=[], current_MM_coords=[], MMcharges=[], qm_elems=[],
                mm_elems=[], elems=[], Grad=False, PC=False, nprocs=1):
        print("------------STARTING XTB INTERFACE-------------")
        #Create XYZfile with generic name for xTB to run
        inputfilename="xtb-inpfile"
        print("Creating inputfile:", inputfilename+'.xyz')
        #What coordinates to work with
        if len(current_coords) != 0:
            pass
        else:
            current_coords=self.coords
        #Using current_coords from now on
        numatoms=len(current_coords)
        self.cleanup()
        #Todo: xtbrestart possibly. needs to be optional
        write_xyzfile(self.elems, current_coords, inputfilename)



        #Run inputfile. Take nprocs argument. Orcadir argument??
        print("------------Running xTB-------------")
        print("...")
        if Grad==True:
            run_xtb_SP_serial(self.xtbdir, self.xtbmethod, inputfilename + '.xyz', self.charge, self.mult, Grad=True)
        else:
            run_xtb_SP_serial(self.xtbdir, self.xtbmethod, inputfilename+'.xyz', self.charge, self.mult)


        print("------------xTB calculation done-------------")
        #Check if finished. Grab energy
        if Grad==True:
            self.energy,self.grad=xtbgradientgrab('gradient',numatoms)
            print("------------ENDING XTB-INTERFACE-------------")
            return self.energy, self.grad
        else:
            outfile=inputfilename+'.out'
            self.energy=xtbfinalenergygrab(outfile)
            print("------------ENDING XTB-INTERFACE-------------")
            return self.energy




# TEST: Run multiple  QM/MM Engrad calculations in parallel

 #       for disp, geo in zip(list_of_displacements,list_of_displaced_geos):
 #           atom_disp=disp[0]
 #           if disp[1] == 0:
 #               crd='x'
 #           elif disp[1] == 1:
 #               crd = 'y'
 #           elif disp[1] == 2:
 #               crd = 'z'
 #           drection=disp[2]
 #           displacement_jobname='Numfreq-Disp-'+'Atom'+str(atom_disp)+crd+drection
 #           print("Displacing Atom: {} Coordinate: {} Direction: {}".format(atom_disp, crd, drection))

#Called from run_QMMM_SP_in_parallel. Runs
def run_QM_MM_SP(list):
    orcadir=list[0]
    current_coords=list[1]
    theory=list[2]
    #label=list[3]
    #Create new dir (name of label provided
    #Cd dir
    theory.run(Grad=True)

def run_QMMM_SP_in_parallel(orcadir, list_of__geos, list_of_labels, QMMMtheory, numcores):
    import multiprocessing as mp
    blankline()
    print("Number of CPU cores: ", numcores)
    print("Number of geos:", len(list_of__geos))
    print("Running snapshots in parallel")
    pool = mp.Pool(numcores)
    results = pool.map(run_QM_MM_SP, [[orcadir,geo, QMMMtheory ] for geo in list_of__geos])
    pool.close()
    print("Calculations are done")



#MMAtomobject used to store LJ parameter and possibly charge for MM atom with atomtype, e.g. OT
class AtomMMobject:
    def __init__(self, atomcharge=None, LJparameters=[]):
        sf="dsf"
        self.atomcharge = atomcharge
        self.LJparameters = LJparameters
    def add_charge(self, atomcharge=None):
        self.atomcharge = atomcharge
    def add_LJparameters(self, LJparameters=None):
        self.LJparameters=LJparameters

#Makes more sense to store this here. Simplifies Yggdrasill inputfile import.
def MMforcefield_read(file):
    print("Reading forcefield file:", file)
    MM_forcefield = {}
    atomtypes=[]
    with open(file) as f:
        for line in f:
            if 'combination_rule' in line:
                combrule=line.split()[-1]
                print("Found combination rule defintion in forcefield file:", combrule)
                MM_forcefield["combination_rule"]=combrule
            if 'charge' in line:
                print("Found charge definition in forcefield file:", ' '.join(line.split()[:]))
                atomtype=line.split()[1]
                if atomtype not in MM_forcefield.keys():
                    MM_forcefield[atomtype]=AtomMMobject()
                charge=float(line.split()[2])
                MM_forcefield[atomtype].add_charge(atomcharge=charge)
                # TODO: Charges defined are currently not used I think
            if 'LennardJones_i_sigma' in line:
                #TODO: need to have it ignore commented-outl ines
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
                print("R0tosigma conversion", R0tosigma)
                if atomtype not in MM_forcefield.keys():
                    MM_forcefield[atomtype] = AtomMMobject()
                sigma_i=float(line.split()[2])*R0tosigma
                eps_i=float(line.split()[3])
                MM_forcefield[atomtype].add_LJparameters(LJparameters=[sigma_i,eps_i])
            if 'LennardJones_ij' in line:
                print("Found LJ pair definition in forcefield file")
                atomtype_i=line.split()[1]
                atomtype_j=line.split()[2]
                sigma_ij=float(line.split()[3])
                eps_ij=float(line.split()[4])
                print("This is incomplete. Exiting")
                exit()
                # TODO: Need to finish this. Should replace LennardJonespairpotentials later
    return MM_forcefield


