import subprocess as sp
import shutil
import time
import numpy as np
import os
import sys
import glob
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.constants import ang2bohr, harkcal
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader,pygrep
from ash.functions.functions_parallel import check_OpenMPI
import ash.settings_ash

#Interface to Dice: SHCI, QMC (single-det or SHCI multi-det) and NEVPT2

#TODO: Remove need for second-iteration print-det in AFQMC-SHCI
#Should be fixed but need to check
#TODO: Test SHCI initial orbitals option, requires parameter options 
#TODO: fix nevpt2
#TODO: Add proper frozen core to mch object setup (instead of relying on nmax threshold)
#Straightforward: https://pyscf.org/user/mcscf.html#frozen-orbital-mcscf 
#Add also to pyscf part

class DiceTheory:
    def __init__(self, dicedir=None, pyscftheoryobject=None, filename='input.dat', printlevel=2, numcores=1, 
                moreadfile=None, initial_orbitals='MP2', CAS_AO_labels=None,memory=20000, frozencore=True,
                SHCI=False, NEVPT2=False, AFQMC=False,  
                SHCI_stochastic=True, SHCI_PTiter=200, SHCI_sweep_iter= [0,3],
                SHCI_DoRDM=False, SHCI_sweep_epsilon = [ 5e-3, 1e-3 ], SHCI_macroiter=0,
                SHCI_davidsonTol=5e-05, SHCI_dE=1e-08, SHCI_maxiter=9, SHCI_epsilon2=1e-07, SHCI_epsilon2Large=1000,
                SHCI_targetError=0.0001, SHCI_sampleN=200, SHCI_nroots=1,
                SHCI_cas_nmin=1.999, SHCI_cas_nmax=0.0, SHCI_active_space=None, SHCI_active_space_range=None,
                Dice_SHCI_direct=None, fcidumpfile=None, Dice_refdeterminant=None,
                QMC_trialWF=None, QMC_SHCI_numdets=1000, QMC_dt=0.005, QMC_nsteps=50, QMC_nblocks=1000, QMC_nwalkers_per_proc=5):

        self.theorynamelabel="Dice"
        self.theorytype="QM"
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")
        

        if dicedir == None:
            print(BC.WARNING, f"No dicedir argument passed to {self.theorynamelabel}Theory. Attempting to find dicedir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.dicedir=ash.settings_ash.settings_dict["dicedir"]
            except KeyError:
                print(BC.WARNING,"Found no dicedir variable in settings_ash module either.",BC.END)
                try:
                    self.dicedir = os.path.dirname(os.path.dirname(shutil.which('Dice')))
                    print(BC.OKGREEN,"Found Dice in PATH. Setting dicedir to:", self.dicedir, BC.END)
                except:
                    print(BC.FAIL,"Found no Dice executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.dicedir = dicedir
        #Check if dir exists
        if os.path.exists(self.dicedir):
            print("Dice directory:", self.dicedir)
        else:
            print(f"Chosen Dice directory : {self.dicedir} does not exist. Exiting...")
            ashexit()
        #Check for PySCFTheory object 
        if pyscftheoryobject is None:
            if Dice_SHCI_direct == None:
                print("Error:No pyscftheoryobject was provided. This is required")
                ashexit()
        
        #Check if conflicting options selected
        if NEVPT2 is True and AFQMC is True:
            print("NEVPT2 and AFQMC can not both be True")
            ashexit()
        if NEVPT2 is False and AFQMC is False and SHCI is False:
            print("Either NEVPT2, SHCI or AFQMC option needs to be True")
            ashexit()
        if QMC_trialWF == 'SHCI' and SHCI is False:
            print("QMC_trialWF='SHCI' requires SHCI to be True, turning on.")
            SHCI=True
        if NEVPT2 == True and SHCI is False:
            print("NEVPT2 requires SHCI to be True, turning on.")
            SHCI=True
        if SHCI is True and AFQMC is True and QMC_trialWF != 'SHCI':
            print("SHCI and AFQMC True but QMC_trialWF != \'SHCI\'.")
            print("Probably not what you wanted. Either set SHCI to False or QMC_trialWF=\'SHCI\' ")
            ashexit()

        #Path to Dice binary
        self.dice_binary=self.dicedir+"/bin/Dice"
        #Put Dice script dir in path
        sys.path.insert(0, self.dicedir+"/scripts")

        #Now import pyscf, shciscf (plugin), qmcutils (Dice dir scripts)
        if SHCI == True and Dice_SHCI_direct != True:
            self.load_pyscf()
            self.load_shciscf()
        self.load_qmcutils()

        if numcores > 1:
            try:
                print(f"MPI-parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
                check_OpenMPI()
            except:
                print("Problem with mpirun")
                ashexit()
        
        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.numcores=numcores
        self.pyscftheoryobject=pyscftheoryobject
        self.NEVPT2=NEVPT2
        self.AFQMC=AFQMC
        self.SHCI=SHCI
        self.Dice_SHCI_direct=Dice_SHCI_direct
        self.moreadfile=moreadfile
        self.fcidumpfile=fcidumpfile
        self.Dice_refdeterminant=Dice_refdeterminant
        self.memory=memory #Memory in MB (total) assigned to PySCF mcscf object
        self.initial_orbitals=initial_orbitals #Initial orbitals to be used (unless moreadfile option)
        self.CAS_AO_labels=CAS_AO_labels  #Used only if AVAS-CASSCF, DMET-CASSCF initial_orbitals option
        #SHCI options
        if self.SHCI is True:
            self.SHCI_stochastic=SHCI_stochastic
            self.SHCI_PTiter=SHCI_PTiter
            self.SHCI_sweep_iter=SHCI_sweep_iter
            self.SHCI_DoRDM=SHCI_DoRDM
            self.SHCI_sweep_epsilon = SHCI_sweep_epsilon
            self.SHCI_macroiter=SHCI_macroiter
            self.SHCI_davidsonTol=SHCI_davidsonTol
            self.SHCI_dE=SHCI_dE
            self.SHCI_maxiter=SHCI_maxiter
            self.SHCI_epsilon2=SHCI_epsilon2
            self.SHCI_epsilon2Large=SHCI_epsilon2Large
            self.SHCI_targetError=SHCI_targetError
            self.SHCI_sampleN=SHCI_sampleN
            self.SHCI_nroots=SHCI_nroots
            self.SHCI_cas_nmin=SHCI_cas_nmin
            self.SHCI_cas_nmax=SHCI_cas_nmax
            self.SHCI_active_space=SHCI_active_space #Alternative to SHCI_cas_nmin/SHCI_cas_nmax
            self.SHCI_active_space_range=SHCI_active_space_range #Alternative (suitable when el-number changes)
        #QMC options
        self.QMC_trialWF=QMC_trialWF
        self.QMC_SHCI_numdets=QMC_SHCI_numdets
        self.frozencore=frozencore
        self.QMC_dt=QMC_dt
        self.QMC_nsteps=QMC_nsteps
        self.QMC_nblocks=QMC_nblocks
        self.QMC_nwalkers_per_proc=QMC_nwalkers_per_proc
        #If SHCI is used as trial WF we turn off PT stage (timeconsuming)
        if self.AFQMC is True and self.QMC_trialWF == 'SHCI':
            print("AFQMC with SHCI trial WF. Turning off PT stage (not needed)")
            self.SHCI_stochastic=True #otherwise deterministic PT happens
            self.SHCI_PTiter=0 # PT skipped with this
        #Print stuff
        print("Printlevel:", self.printlevel)
        print("Memory (MB):", self.memory)
        print("Num cores:", self.numcores)
        print("PySCF object:", self.pyscftheoryobject)
        print("SHCI:", self.SHCI)
        print("NEVPT2:", self.NEVPT2)
        print("AFQMC:", self.AFQMC)
        print("Frozencore:", self.frozencore)
        print("moreadfile:", self.moreadfile)
        print("Initial orbitals:", self.initial_orbitals)
        print("CAS_AO_labels:", self.CAS_AO_labels)
        print("Dice_SHCI_direct:", self.Dice_SHCI_direct)
        if self.Dice_SHCI_direct is True:
            print("FCIDUMP file:", self.fcidumpfile)
            print("Dice Reference det. string:", self.Dice_refdeterminant)
        if self.SHCI is True:
            print("SHCI_stochastic", self.SHCI_stochastic)
            print("SHCI_PTiter", self.SHCI_PTiter)
            print("SHCI_sweep_iter", self.SHCI_sweep_iter)
            print("SHCI_DoRDM", self.SHCI_DoRDM)
            print("SHCI_sweep_epsilon", self.SHCI_sweep_epsilon)
            print("SHCI DavidsonTol:", self.SHCI_davidsonTol)
            print("SHCI dE:", self.SHCI_dE)
            print("SHCI_maxiter:", self.SHCI_maxiter)
            print("SHCI_macroiter", self.SHCI_macroiter)
            print("SHCI_epsilon2:", self.SHCI_epsilon2)
            print("SHCI_epsilon2Large:", self.SHCI_epsilon2Large)
            print("SHCI_targetError:", self.SHCI_targetError)
            print("SHCI_sampleN:", self.SHCI_sampleN)
            print("SHCI_nroots:", self.SHCI_nroots)
            print("SHCI_active_space_range:", self.SHCI_active_space_range)
            print("SHCI_active_space:", self.SHCI_active_space)
            print("SHCI CAS NO nmin", self.SHCI_cas_nmin)
            print("SHCI CAS NO nmax", self.SHCI_cas_nmax)
        if self.AFQMC is True:
            print("QMC_trialWF:", self.QMC_trialWF)
            if self.QMC_trialWF == 'SHCI':
                print("QMC_SHCI_numdets:", self.QMC_SHCI_numdets)
            print("QMC settings:")
            print("QMC_dt:", self.QMC_dt)
            print("Number of steps per block (QMC_nsteps):", self.QMC_nsteps)
            print("Number of blocks (QMC_nblocks):", self.QMC_nblocks)
            print("Number of walkers per proc:", self.QMC_nwalkers_per_proc)
    
    def load_pyscf(self):
        try:
            import pyscf
        except:
            print(BC.FAIL, "Problem importing pyscf. Make sure pyscf has been installed: pip install pyscf", BC.END)
            ashexit(code=9)
        self.pyscf=pyscf
        print("\nPySCF version:", self.pyscf.__version__)
    def load_shciscf(self):
        #SHCI pyscf plugin
        try:
            from pyscf.shciscf import shci
            self.shci=shci
        except ModuleNotFoundError:
            print("Problem importing pyscf.shciscf (PySCF interface module to Dice)")
            print("See: https://github.com/pyscf/shciscf on how to install shciscf module for pyscf")
            print("Most likely: pip install git+https://github.com/pyscf/shciscf")
            ashexit()
        except ImportError:
            print("Create settings.py file (see path in pyscf import error message, probably above) and add this:")
            m=f"""
SHCIEXE = "{self.dicedir}/bin/Dice"
SHCISCRATCHDIR= "."
MPIPREFIX=""
            """
            print(m)
            ashexit()
    def load_qmcutils(self):
        try:
            import QMCUtils
            self.QMCUtils=QMCUtils
        except ModuleNotFoundError as me:
            print("ModuleNotFoundError:")
            print("Exception message:", me)
            print("Either: QMCUtils requires another module (pandas?). Please install it via conda or pip.")
            print("or: Problem importing QMCUtils. Dice directory is probably incorrectly set")
            ashexit()
    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleaning up Dice temporary files")
        if self.SHCI is True:
            bkpfiles=glob.glob('*.bkp')
            for bkpfile in bkpfiles:
                os.remove(bkpfile)
    def determine_frozen_core(self,elems):
        print("Determining frozen core based on system list of elements")
        #Main elements 
        FC_elems={'H':0,'He':0,'Li':0,'Be':0,'B':2,'C':2,'N':2,'O':2,'F':2,'Ne':2,
        'Na':2,'Mg':2,'Al':10,'Si':10,'P':10,'S':10,'Cl':10,'Ar':10,
        'K':10,'Ca':10,'Sc':10,'Ti':10,'V':10,'Cr':10,'Mn':10,'Fe':10,'Co':10,'Ni':10,'Cu':10,'Zn':10,
        'Ga':18,'Ge':18,'As':18,'Se':18, 'Br':18, 'Kr':18}
        #NOTE: To be updated for 4d TM row etc
        num_el=0
        for el in elems:
            num_el+=FC_elems[el]
        self.frozen_core_el=num_el
        self.frozen_core_orbs=int(num_el/2)
        print("Total frozen electrons in system:", self.frozen_core_el)
        print("Total frozen orbitals in system:", self.frozen_core_orbs)

    def grab_num_dets(self):
        grab=False
        numdet=0
        with open("output.dat") as f:
            for line in f:
                if 'Performing final tigh' in line:
                    grab=False
                if grab is True:
                    if len(line.split()) == 7:
                        numdet=int(line.split()[3])
                if 'Iter Root       Eps1   #Var. Det.               Ener' in line:
                    grab=True
        return numdet

    #Run Dice-SHCI from FCIDUMP job without pyscf
    #Assumes you have an external FCIDUMP file
    #TODO: Currently the refdeterminant string has to be set manually. TO FIX
    def run_Dice_SHCI_from_FCIDUMP(self):
        #Create inputfile input.dat
        nocc=self.SHCI_active_space[0] #how many electrons in active space
        #refdeterminant="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"
        #what orbital indices the electrons occupy in the set of MOs in the FCIDUMP file
        if self.Dice_refdeterminant == None:
            print("Error: reference determinant string required!")
            ashexit()
        sweepschedule=""
        for it,eps in zip(self.SHCI_sweep_iter,self.SHCI_sweep_epsilon):
            sweepschedule=sweepschedule+f"{it} {eps}\n"
        sweepschedule=os.linesep.join([s for s in sweepschedule.splitlines() if s])
        #Inputfile creation
        inputstring=f"""

# reference determinant
nocc {nocc}
{self.Dice_refdeterminant} 
end

orbitals {self.fcidumpfile}

# Variational
schedule
{sweepschedule}
end
davidsontol {self.SHCI_davidsonTol}
dE {self.SHCI_dE}
maxiter {self.SHCI_maxiter}

# PT
epsilon2 {self.SHCI_epsilon2}
epsilon2large {self.SHCI_epsilon2Large}
sampleN {self.SHCI_sampleN}
seed 200
targeterror {self.SHCI_targetError}

# Misc
noio
        """
        with open("input.dat", 'w') as f:
            f.write(inputstring)
        
        # Call Dice directly
        self.call_dice_directly()
        #Read energy and determinants from outputfile: output.dat
        enresult = pygrep("PTEnergy:","output.dat")
        print("enresult:", enresult)
        self.energy = enresult[1]
        self.error = enresult[-1]
        self.num_var_determinants = self.grab_num_dets()
        print("Number of variational determinants:", self.num_var_determinants)

    # call_dice_directly : Call Dice directly if inputfile and FCIDUMP file already exists
    def call_dice_directly(self):
        module_init_time=time.time()
        print("Calling Dice executable directly")
        #For calling Dice directly when needed
        print(f"Running Dice with ({self.numcores} MPI processes)")
        with open('output.dat', "w") as outfile:
            sp.call(['mpirun', '-np', str(self.numcores), self.dice_binary, self.filename], stdout=outfile)
        print_time_rel(module_init_time, modulename='Dice-SHCI-direct-run', moduleindex=2)
    
    #Set up initial orbitals
    #This returns a set of MO-coeffs and occupations either from checkpointfile or from MP2/CC/SHCI job
    def setup_initial_orbitals(self):
        module_init_time=time.time()
        print("\n INITIAL ORBITAL OPTION")
        if len(self.pyscftheoryobject.mf.mo_occ) == 2:
            totnumborb=len(self.pyscftheoryobject.mf.mo_occ[0])
        else:
            totnumborb=len(self.pyscftheoryobject.mf.mo_occ)
        print(f"There are {totnumborb} orbitals in the system")
        #READ ORBITALS OR DO natural orbitals with MP2/CCSD/CCSD(T)
        if self.moreadfile == None:
            print("No checkpoint file given (moreadfile option).")
            print(f"Will calculate PySCF {self.initial_orbitals} natural orbitals to use as input in Dice CAS job")
            if self.initial_orbitals not in ['MP2','CCSD','CCSD(T)', 'SHCI', 'AVAS-CASSCF', 'DMET-CASSCF','CASSCF']:
                print("Error: Unknown initial_orbitals choice. Exiting.")
                ashexit()
            print("Options are: MP2, CCSD, CCSD(T), SHCI, AVAS-CASSCF, DMET-CASSCF")
            #Option to do small-eps SHCI step 
            if self.initial_orbitals == 'SHCI':
                print("SHCI initial orbital option")
                print("First calculating MP2 natural orbitals, then doing SHCI-job")
                #Call pyscftheory method for MP2,CCSD and CCSD(T)
                MP2nat_occupations, MP2nat_mo_coefficients = self.pyscftheoryobject.calculate_natural_orbitals(self.pyscftheoryobject.mol,
                                                                self.pyscftheoryobject.mf, method='MP2')
                self.setup_active_space(occupations=MP2nat_occupations)
                self.setup_SHCI_job(verbose=5, rdmoption=True) #Creates the self.mch CAS-CI/CASSCF object with RDM True
                self.SHCI_object_set_mos(mo_coeffs=MP2nat_mo_coefficients) #Sets the MO coeffs of mch object              
                self.SHCI_object_run() #Runs the self.mch object
                #NOTE: Only worry is that we create self.mch object here and then again later
                #TODO: Control eps and PT for SHCI job
                #TODO: We probably want SHCI-var+PT for an accurate RDM. Too expensive?
                #TODO: No RDM for stochastic so we have to switch to deterministic RDM
                #????
                print("Done with SHCI-run for initial orbital step")
                print("Now making natural orbitals from SHCI WF")
                occupations, mo_coefficients = self.pyscf.mcscf.addons.make_natural_orbitals(self.mch)
                print("SHCI natural orbital occupations:", occupations)
            elif self.initial_orbitals == 'AVAS-CASSCF' or self.initial_orbitals == 'DMET-CASSCF':
                print("Calling calculate_natural_orbitals using AVAS/DMET method")

                if self.CAS_AO_labels == None:
                    print("Error: CAS_AO_labels are messing, provide keyword to DiceTheory!")
                    ashexit()

                occupations, mo_coefficients = self.pyscftheoryobject.calculate_natural_orbitals(self.pyscftheoryobject.mol,
                                                                self.pyscftheoryobject.mf, method=self.initial_orbitals, 
                                                                CAS_AO_labels=self.CAS_AO_labels)
            else:
                print("Calling nat-orb option in pyscftheory")
                #Call pyscftheory method for MP2,CCSD and CCSD(T)
                occupations, mo_coefficients = self.pyscftheoryobject.calculate_natural_orbitals(self.pyscftheoryobject.mol,
                                                                self.pyscftheoryobject.mf, method=self.initial_orbitals)

        else:
            print("Will read MOs from checkpoint file:", self.moreadfile)
            if '.chk' not in self.moreadfile:
                print("Error: not a PySCF chkfile")
                ashexit()
            mo_coefficients = self.pyscf.lib.chkfile.load(self.moreadfile, 'mcscf/mo_coeff')
            occupations = self.pyscf.lib.chkfile.load(self.moreadfile, 'mcscf/mo_occ')
            print("Chk-file occupations:", occupations)
            print("Length of occupations array:", len(occupations))
            if len(occupations) != totnumborb:
                print("Occupations array length does NOT match length of MO coefficients in PySCF object")
                print("Is basis different? Exiting")
                ashexit()

        #Check if occupations are sensible (may be nonsense if CCSD/CAS calc failed)
        if True in [i > 2.00001 for i in occupations]:
            print("Problem. Occupation array contains occupations larger than 2.0. Something went wrong (bad convergence?)")
            ashexit()
        if True in [i < 0.0 for i in occupations]:
            print("Problem. Occupation array contains negative occupations. Something possibly wrong (bad convergence?)")
            print("Continuing but these orbitals may be bad")
        #    ashexit()
        print("Initial orbital step complete")
        print("----------------------------------")
        print()
        print_time_rel(module_init_time, modulename='Dice-Intial-orbital-step', moduleindex=2)
        return mo_coefficients, occupations

    #Determine active space based on either natural occupations of initial orbitals or 
    def setup_active_space(self, occupations=None):
        print("Inside setup_active_space")
        with np.printoptions(precision=3, suppress=True):
            print("Occupations:", occupations)

        # If SHCI_active_space defined then this takes priority over natural occupations
        if self.SHCI_active_space != None:
            print("Active space given as user-input: active_space=", self.SHCI_active_space)
            # Number of orbital and electrons from active_space keyword!
            self.nelec=self.SHCI_active_space[0]
            self.norb=self.SHCI_active_space[1]
        elif self.SHCI_active_space_range != None:
            #Convenvient when we have the orbitals we want but we can't define active_space because the electron-number changes (IEs)
            #TODO: Problem: Occupations depend on the system CHK file we read in. Electrons not matching anymore
            print("Active space range:", self.SHCI_active_space_range)
            self.norb = len(occupations[self.SHCI_active_space_range[0]:self.SHCI_active_space_range[1]])
            self.nelec = round(sum(occupations[self.SHCI_active_space_range[0]:self.SHCI_active_space_range[1]]))
            print(f"Selected active space from range: CAS({self.nelec},{self.norb})")      
        else:
            print(f"SHCI Active space determined from {self.initial_orbitals} NO threshold parameters: SHCI_cas_nmin={self.SHCI_cas_nmin} and SHCI_cas_nmax={self.SHCI_cas_nmax}")
            print("Note: Use active_space keyword if you want to select active space manually instead")
            # Determing active space from natorb thresholds
            nat_occs_for_thresholds=[i for i in occupations if i < self.SHCI_cas_nmin and i > self.SHCI_cas_nmax]
            self.norb = len(nat_occs_for_thresholds)
            self.nelec = round(sum(nat_occs_for_thresholds))
            print(f"To get this same active space in another calculation you can also do: SHCI_active_space=[{self.nelec},{self.norb}]")
            indices_for_thresholds=[i for i,j in enumerate(occupations) if j < self.SHCI_cas_nmin and j > self.SHCI_cas_nmax]
            print("indices_for_thresholds:", indices_for_thresholds)
            firstMO_index=indices_for_thresholds[0]
            lastMO_index=indices_for_thresholds[-1]
            print(f"To get this same active space in another calculation you can also do: SHCI_active_space_range=[{firstMO_index},{lastMO_index}]")
        #Check if active space still not defined and exit if so
        if self.norb == None:
            print("No active space has been defined!")
            print("You probably need to provide SHCI_active_space keyword!")
            ashexit()

    #Setup a SHCI CAS-CI or CASSCF job using the SHCI-PySCF interface
    def setup_SHCI_job(self,verbose=5, rdmoption=None):
        print(f"\nDoing SHCI-CAS calculation with {self.nelec} electrons in {self.norb} orbitals!")
        print("SHCI_macroiter:", self.SHCI_macroiter)

        #Turn RDM creation on or off
        if rdmoption is False:
            #Pick user-selected (default: False)
            dordm=self.SHCI_DoRDM
        else:
            #Probably from initial-orbitals call (RDM True)
            dordm=rdmoption
            print("RDM option:", dordm)

        if self.SHCI_macroiter == 0:
            print("This is single-iteration CAS-CI via pyscf and SHCI")

            self.mch = self.pyscf.mcscf.CASCI(self.pyscftheoryobject.mf,self.norb, self.nelec)
            print("Turning off canonicalization step in mcscf object")
            self.mch.canonicalization = False
        else:
            print("This is CASSCF via pyscf and SHCI (orbital optimization)")
            #self.mch = self.shci.SHCISCF(self.pyscftheoryobject.mf, self.norb, self.nelec)
            self.mch = self.pyscf.mcscf.CASSCF(self.pyscftheoryobject.mf,self.norb, self.nelec)
        #Settings
        self.mch.fcisolver = self.shci.SHCI(self.pyscftheoryobject.mol)
        self.mch.fcisolver.mpiprefix = f'mpirun -np {self.numcores}'
        self.mch.fcisolver.stochastic = self.SHCI_stochastic
        self.mch.fcisolver.nPTiter = self.SHCI_PTiter
        self.mch.fcisolver.sweep_iter = self.SHCI_sweep_iter
        self.mch.fcisolver.DoRDM = dordm
        self.mch.fcisolver.sweep_epsilon = self.SHCI_sweep_epsilon
        self.mch.fcisolver.davidsonTol = self.SHCI_davidsonTol
        self.mch.fcisolver.dE = self.SHCI_dE
        self.mch.fcisolver.maxiter = self.SHCI_maxiter
        self.mch.fcisolver.epsilon2 = self.SHCI_epsilon2
        self.mch.fcisolver.epsilon2Large = self.SHCI_epsilon2Large
        self.mch.fcisolver.targetError = self.SHCI_targetError
        self.mch.fcisolver.sampleN = self.SHCI_sampleN
        self.mch.fcisolver.nroots = self.SHCI_nroots
        self.mch.verbose=verbose
        #Setting memory
        self.mch.max_memory=self.memory
        print("Memory in pyscf object set to:", self.mch.max_memory)
        #CASSCF iterations
        self.mch.max_cycle_macro = self.SHCI_macroiter        

    #This sets MO coefficients of mch object unless None (then)
    def SHCI_object_set_mos(self,mo_coeffs=None):
        print("Updating MO coefficients in mch object")
        self.mch.mo_coeffs=mo_coeffs

    #Run the defined pyscf mch object
    def SHCI_object_run(self, write_det_CASCI=False,numdets=None):
        module_init_time=time.time()
        #Run SHCISCF object created above
        print("Running Dice via SHCISCF interface in pyscf")
        print("Dice output can be monitored in output.dat on local scratch")
        #Run CAS-CI/CASSCF object

        # For SHCI-AFQMC we need dets.bin file so instead of calling
        # we do a dryrun, write inputfile, add to Dice inputfile and then run
        if write_det_CASCI is True:
            print("Doing dryun")
            #Do Dryrun here where we write out Dice input file and then run
            self.shci.dryrun(self.mch)
            print("Writing SHCIConfFile")
            self.shci.writeSHCIConfFile(self.mch.fcisolver, self.mch.nelecas, False)
            print("Adding dets printout")
            with open(self.mch.fcisolver.configFile, 'a') as f:
                f.write(f'writebestdeterminants {numdets}\n\n')
            #Run dice
            print("Executing SHCI")
            self.shci.executeSHCI(self.mch.fcisolver)
            print("SHCI-step done")
            print("SHCI-trial WF energy (variational-stage only):", self.mch.e_tot)
        else:
            #Run like normal
            self.mch.run()
            #Get final energy
            self.energy = self.mch.e_tot

        #Grab number of determinants from Dice output
        self.num_var_determinants = self.grab_num_dets()
        print("Number of variational determinants:", self.num_var_determinants)
        print_time_rel(module_init_time, modulename='Dice-SHCI-run', moduleindex=2)
        #TODO
        #Grab actual number of stochastic PT iterations taken

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)

        #Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        #Cleanup before run.
        self.cleanup()

        #Run PySCF to get integrals and MOs. This would probably only be an SCF
        if self.Dice_SHCI_direct != True:
            self.pyscftheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)

        #Get frozen-core
        if self.frozencore is True:
            if self.Dice_SHCI_direct == None:
                self.determine_frozen_core(qm_elems)
        else:
            self.frozen_core_orbs=0

        # NOW RUNNING
        #NEVPT2
        if self.NEVPT2 is True:
            #NOTE: Requires fixes to getDets function in QMCUtils.py
            #TODO: Look at example. https://github.com/sanshar/Dice/blob/master/examples/NEVPT2/N2/n2nevpt.py
            print("Running Dice NEVPT2 calculation on multiconfigurational WF")

            #Calling SHCI run to get the self.mch object
            print("First running SHCI CAS-CI/CASSCF step")
            mo_coeffs, occupations = self.setup_initial_orbitals() #Returns mo-coeffs and occupations of initial orbitals
            self.setup_active_space(occupations=occupations) #This will define self.norb and self.nelec active space
            self.setup_SHCI_job() #Creates the self.mch CAS-CI/CASSCF object
            self.SHCI_object_set_mos(mo_coeffs=mo_coeffs) #Sets the MO coeffs of mch object              
            self.SHCI_object_run() #Runs the self.mch object
            #NOTE: Pretty sure we have to do full CASSCF here
            
            print(f"Now running NEVPT2 on SHCI reference WF: CAS({self.nelec},{self.norb})")
            module_init_time=time.time()
            self.QMCUtils.run_nevpt2(self.mch, nelecAct=self.nelec, numAct=self.norb, norbFrozen=self.frozen_core_orbs,
               integrals="FCIDUMP.h5", nproc=numcores, seed=None, vmc_root=self.dicedir,
               fname="nevpt2.json", foutname='nevpt2.out', nroot=0,
               spatialRDMfile=None, spinRDMfile=None, stochasticIterNorms=1000,
               nIterFindInitDets=100, numSCSamples=10000, stochasticIterEachSC=100,
               fixedResTimeNEVPT_Ene=False, epsilon=1.0e-8, efficientNEVPT_2=True,
               determCCVV=True, SCEnergiesBurnIn=50, SCNormsBurnIn=50,
               diceoutfile="output.dat")
            print_time_rel(module_init_time, modulename='Dice-NEVPT2-run', moduleindex=2)
            #TODO: Grab energy from function call
            self.energy=0.0
            print("Final Dice NEVPT2 energy:", self.energy)
        #AFQMC
        elif self.AFQMC is True:
            print("Running Dice AFQMC")
            #SHCI trial wavefunction AFQMC
            if self.QMC_trialWF == 'SHCI':
                print("Multi-determinant trial WF option via SHCI is on!")

                mo_coeffs, occupations = self.setup_initial_orbitals() #Returns mo-coeffs and occupations of initial orbitals
                self.setup_active_space(occupations=occupations) #This will define self.norb and self.nelec active space
                self.setup_SHCI_job() #Creates the self.mch CAS-CI/CASSCF object
                self.SHCI_object_set_mos(mo_coeffs=mo_coeffs) #Sets the MO coeffs of mch object              
                self.SHCI_object_run(write_det_CASCI=True, numdets=self.QMC_SHCI_numdets) #Runs the self.mch object with dets-printout

                #Get dets.bin file
                #print("\nRunning SHCI (via PySCFTheory object) once again to write dets.bin")
                #self.run_and_write_dets(self.QMC_SHCI_numdets)
                print("SHCI trial wavefunction prep complete.")
                if self.QMC_SHCI_numdets > self.num_var_determinants:
                    print(f"Error: QMC_SHCI_numdets ({self.QMC_SHCI_numdets}) larger than SHCI-calculated determinants ({self.num_var_determinants})")
                    print("Increase SHCI-WF size: e.g. by increasing active space(SHCI_cas_nmin, SHCI_cas_nmax or SHCI_active_space) or reducing thresholds (SHCI_sweep_epsilon)")
                    ashexit()
                print(f"{self.num_var_determinants} variational determinants were calculated by SHCI")
                print(f"{self.QMC_SHCI_numdets} variational determinants were written to disk (dets.bin)")
                print(f"{self.QMC_SHCI_numdets} determinants will be used in multi-determinant AFQMC job")

                #Phaseless AFQMC with hci trial
                module_init_time=time.time()
                e_afqmc, err_afqmc = self.QMCUtils.run_afqmc_mc(self.mch, mpi_prefix=f"mpirun -np {numcores}",
                                norb_frozen=self.frozen_core_orbs, chol_cut=1e-5,
                                ndets=self.QMC_SHCI_numdets, nroot=0, seed=None,
                                dt=self.QMC_dt, steps_per_block=self.QMC_nsteps, nwalk_per_proc=self.QMC_nwalkers_per_proc,
                                nblocks=self.QMC_nblocks, ortho_steps=20, burn_in=50,
                                cholesky_threshold=1.0e-3, weight_cap=None, write_one_rdm=False,
                                use_eri=False, dry_run=False)
                e_afqmc=e_afqmc[0] 
                err_afqmc=err_afqmc[0]
                print_time_rel(module_init_time, modulename='Dice-AFQMC-run', moduleindex=2)
            #Single determinant trial wavefunction
            else:
                print("Using single-determinant WF from PySCF object")
                #Phaseless AFQMC with simple mf trial
                module_init_time=time.time()
                e_afqmc, err_afqmc = self.QMCUtils.run_afqmc_mf(self.pyscftheoryobject.mf, mpi_prefix=f"mpirun -np {numcores}",
                    norb_frozen=self.frozen_core_orbs, chol_cut=1e-5, seed=None, dt=self.QMC_dt,
                    steps_per_block=self.QMC_nsteps, nwalk_per_proc=self.QMC_nwalkers_per_proc, nblocks=self.QMC_nblocks,
                    ortho_steps=20, burn_in=50, cholesky_threshold=1.0e-3,
                    weight_cap=None, write_one_rdm=False, dry_run=False)
                print_time_rel(module_init_time, modulename='Dice-AFQMC-run', moduleindex=2)
            self.energy=e_afqmc
            self.error=err_afqmc
            ##Analysis
            print("Final Dice AFQMC energy:", self.energy)
            if self.error != None:
                print(f"Error: {self.error} Eh ({self.error*harkcal} kcal/mol)")
            else:
                print(f"Error: Not available (problem with blocking?)")
        #Dice-SHCI without pyscf interface. Can be useful
        elif self.Dice_SHCI_direct is True:
            print("Running SHCI option via Dice without pyscf and SHCI plugin")
            self.run_Dice_SHCI_from_FCIDUMP()
            print("Final Dice SHCI energy:", self.energy)
            print("Final Dice SHCI PT error:", self.error)
        #Just SHCI via PySCF
        else:
            if self.SHCI is True:
                print("Regular SHCI option is active.")
                mo_coeffs, occupations = self.setup_initial_orbitals() #Returns mo-coeffs and occupations of initial orbitals
                self.setup_active_space(occupations=occupations) #This will define self.norb and self.nelec active space
                self.setup_SHCI_job() #Creates the self.mch CAS-CI/CASSCF object
                self.SHCI_object_set_mos(mo_coeffs=mo_coeffs) #Sets the MO coeffs of mch object              
                self.SHCI_object_run() #Runs the self.mch object
                print("Final Dice SHCI energy:", self.energy)

        print("Dice is finished")
        #Cleanup Dice scratch stuff (big files)
        self.cleanup()

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        return self.energy


