import subprocess as sp
import shutil
import time
import numpy as np
import os
import sys
import glob
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.constants import ang2bohr, harkcal
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader,pygrep,pygrep2
from ash.functions.functions_parallel import check_OpenMPI
import ash.settings_ash

#Interface to Block: Block2 primarily via PySCF and also directly via FCIdump
# Possibly later both Block 1.5 and Stackblock via PySCF

#Block 2 docs: https://block2.readthedocs.io
#BLock 1.5 docs: https://pyscf.org/Block/with-pyscf.html


class BlockTheory:
    def __init__(self, blockdir=None, pyscftheoryobject=None, blockversion='Block2', filename='input.dat', printlevel=2, numcores=1, 
                moreadfile=None, initial_orbitals='MP2', memory=20000, frozencore=True, fcidumpfile=None,
                Block_direct=False, maxM=1000, tol=1e-10):

        self.theorynamelabel="Block"
        self.theorytype="QM"
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        print("Using Blockversion:", blockversion)
        
        if blockdir == None:
            print(BC.WARNING, f"No blockdir argument passed to {self.theorynamelabel}Theory. Attempting to find blockdir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.blockdir=ash.settings_ash.settings_dict["blockdir"]
            except KeyError:
                print(BC.WARNING,"Found no blockdir variable in settings_ash module either.",BC.END)
                try:
                    self.blockdir = os.path.dirname(os.path.dirname(shutil.which('Block')))
                    print(BC.OKGREEN,"Found Block in PATH. Setting blockdir to:", self.blockdir, BC.END)
                except:
                    print(BC.FAIL,"Found no Block executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.blockdir = blockdir
        #Check if dir exists
        if os.path.exists(self.blockdir):
            print("Block directory:", self.blockdir)
        else:
            print(f"Chosen Block directory : {self.blockdir} does not exist. Exiting...")
            ashexit()
        #Check for PySCFTheory object 
        if pyscftheoryobject is None:
            print("Error:No pyscftheoryobject was provided. This is required")
            ashexit()

        #Path to Block binary
        self.block_binary=self.blockdir+"/bin/block"
        #Put Block script dir in path
        #sys.path.insert(0, self.blockdir+"/scripts")

        #Now import block
        self.load_pyscf()
        self.load_dmrgscf()
        #self.load_block_15()


        #dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
        #dmrgscf.settings.MPIPREFIX = ''


        if numcores > 1:
            try:
                print(f"MPI-parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
                check_OpenMPI()
            except:
                print("Problem with mpirun")
                ashexit()
        
        #Printlevel
        self.blockversion=blockversion
        self.printlevel=printlevel
        self.filename=filename
        self.numcores=numcores
        #SETTING NUMCORES by setting prefix
        self.dmrgscf.settings.MPIPREFIX = f'mpirun -n {self.numcores} --bind-to none'
        self.pyscftheoryobject=pyscftheoryobject

        self.moreadfile=moreadfile
        self.Block_direct=Block_direct
        self.maxM=maxM
        self.tol=tol
        self.fcidumpfile=fcidumpfile
        self.frozencore=frozencore
        self.memory=memory #Memory in MB (total) assigned to PySCF mcscf object
        self.initial_orbitals=initial_orbitals #Initial orbitals to be used (unless moreadfile option)
        #Print stuff
        print("Printlevel:", self.printlevel)
        print("Memory (MB):", self.memory)
        print("Num cores:", self.numcores)
        print("PySCF object:", self.pyscftheoryobject)

        print("Frozencore:", self.frozencore)
        print("moreadfile:", self.moreadfile)
        print("Initial orbitals:", self.initial_orbitals)
        if self.Block_direct is True:
            print("FCIDUMP file:", self.fcidumpfile)
    
    
    def load_pyscf(self):
        try:
            import pyscf
        except:
            print(BC.FAIL, "Problem importing pyscf. Make sure pyscf has been installed: pip install pyscf", BC.END)
            ashexit(code=9)
        self.pyscf=pyscf
        print("\nPySCF version:", self.pyscf.__version__)
    def load_dmrgscf(self):
        #DMRGSCF pyscf plugin
        try:
            from pyscf import dmrgscf
            self.dmrgscf=dmrgscf
        except ModuleNotFoundError:
            print("Problem importing dmrgscf (PySCF interface module to Block)")
            print("See: https://github.com/pyscf/dmrgscf/ on how to install shciscf module for pyscf")
            print("Most likely: pip install git+https://github.com/pyscf/dmrgscf")
            ashexit()
        except ImportError:
            print("Create settings.py file (see path in pyscf import error message, probably above) and add this:")
            m=f"""
BLOCKEXE = "{self.blockdir}/bin/Block"
BLOCKEXE_COMPRESS_NEVPT = "" #path to block.spin_adapted
BLOCKSCRATCHDIR = "." #path to scratch dir. Best to keep at . (but make sure we execute on scratch)
MPIPREFIX = "" # mpi-prefix. Leave blank
            """
            print(m)
            ashexit()

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleaning up Block temporary files")
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

    #Grabbing Dice variational energy
    def grab_var_energy(self):
        grab=False
        with open("output.dat") as f:
            for line in f:
                if len(line.split()) < 3:
                    grab=False
                if grab is True:
                    if len(line.split()) == 3:
                        energy=float(line.split()[1])
                if 'Root             Energy' in line:
                    grab=True
        return energy

    def grab_important_dets(self):
        grab=False
        det_strings=[]
        with open("output.dat") as f:
            for line in f:
                if len(line.split()) < 2:
                    grab=False
                if grab is True:
                    if len(line.split()) > 10:
                        det_strings.append(line)
                if ' Det     weight  Determinant string' in line:
                    grab=True
        return det_strings
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

    # call_block_directly : Call block directly if inputfile and FCIDUMP file already exists
    def call_block_directly(self):
        module_init_time=time.time()
        print("Calling Block executable directly")
        print(f"Running Block with ({self.numcores} MPI processes)")
        with open('output.dat', "w") as outfile:
            sp.call(['mpirun', '-np', str(self.numcores), self.block_binary, self.filename], stdout=outfile)
        print_time_rel(module_init_time, modulename='Block-direct-run', moduleindex=2)
    
    #Set up initial orbitals
    #This returns a set of MO-coeffs and occupations either from checkpointfile or from MP2/CC/SHCI job
    def setup_initial_orbitals(self, elems):
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
            if self.initial_orbitals not in ['canMP2','MP2','DFMP2', 'DFMP2relax', 'CCSD','CCSD(T)', 'SHCI', 'AVAS-CASSCF', 'DMET-CASSCF','CASSCF']:
                print("Error: Unknown initial_orbitals choice. Exiting.")
                ashexit()
            print("Options are: MP2, CCSD, CCSD(T), SHCI, AVAS-CASSCF, DMET-CASSCF")
            #Option to do small-eps SHCI step 
            if self.initial_orbitals == 'SHCI':
                print("SHCI initial orbital option")
                print("First calculating MP2 natural orbitals, then doing SHCI-job")
                #Call pyscftheory method for MP2,CCSD and CCSD(T)
                MP2nat_occupations, MP2nat_mo_coefficients = self.pyscftheoryobject.calculate_natural_orbitals(self.pyscftheoryobject.mol,
                                                                self.pyscftheoryobject.mf, method='MP2', elems=elems)
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
                                                                CAS_AO_labels=self.CAS_AO_labels, elems=elems)
            else:
                print("Calling nat-orb option in pyscftheory")
                #Call pyscftheory method for MP2,CCSD and CCSD(T)
                occupations, mo_coefficients = self.pyscftheoryobject.calculate_natural_orbitals(self.pyscftheoryobject.mol,
                                                                self.pyscftheoryobject.mf, method=self.initial_orbitals,
                                                                elems=elems)

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
            print("Warning! Occupation array contains occupations larger than 2.0. Something possibly wrong (bad convergence?)")
            print("Continuing but these orbitals may be bad")
            #ashexit()
        if True in [i < 0.0 for i in occupations]:
            print("Warning!. Occupation array contains negative occupations. Something possibly wrong (bad convergence?)")
            print("Continuing but these orbitals may be bad")
        #    ashexit()
        print("Initial orbital step complete")
        print("----------------------------------")
        print()
        print_time_rel(module_init_time, modulename='Dice-Initial-orbital-step', moduleindex=2)
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
            indices_for_thresholds=[i for i,j in enumerate(occupations) if j < self.SHCI_cas_nmin and j > self.SHCI_cas_nmax]
            firstMO_index=indices_for_thresholds[0]
            lastMO_index=indices_for_thresholds[-1]
            self.norb = len(nat_occs_for_thresholds)
            self.nelec = round(sum(nat_occs_for_thresholds))
            print(f"To get this same active space in another calculation you can also do: \nSHCI_active_space=[{self.nelec},{self.norb}]")
            print(f"or: \nSHCI_active_space_range=[{firstMO_index},{lastMO_index}]")
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
        important_dets = self.grab_important_dets()
        print("Most important determinants:")
        print("    Det     weight  Determinant string")
        print("State :   0")
        print(*important_dets)
        print()
        #Grab actual number of stochastic PT iterations taken
        ref_energy=float(pygrep("Given Ref. Energy", "output.dat")[-1])
        pt_energies=pygrep2("PTEnergy", "output.dat")
        var_energy=self.grab_var_energy()
        det_PT_energy=float(pt_energies[0].split()[-1])
        stoch_PT_energy=float(pt_energies[1].split()[1])
        print(f"Dice ref. energy: {ref_energy} Eh")
        print(f"Dice variational energy: {var_energy} Eh")
        print(f"Dice PT energy (deterministic): {det_PT_energy} Eh")
        print(f"Dice PT energy (stochastic): {stoch_PT_energy} Eh")
        print_time_rel(module_init_time, modulename='Dice-SHCI-run', moduleindex=2)
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
        if self.Block_direct != True:
            self.pyscftheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)

        #Get frozen-core
        if self.frozencore is True:
            if self.Dice_SHCI_direct == None:
                self.determine_frozen_core(qm_elems)
        else:
            self.frozen_core_orbs=0

        # NOW RUNNING
        #TODO: Distinguish between Block2, Stackblock, Block1.5. 
        #TODO: Distinguish between direct run using FCIDUMP also
        if self.blockversion =='Block2':
            mc = self.dmrgscf.DMRGSCF(self.pyscftheoryobject.mf, self.norb, self.nelec, maxM=self.maxM, tol=self.tol)
            #mc.fcisolver.runtimeDir = lib.param.TMPDIR
            #mc.fcisolver.scratchDirectory = lib.param.TMPDIR
            #mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 4)) #Needs more
            mc.fcisolver.memory = int(self.memory / 1000) # mem in GB

            mc.canonicalization = True
            mc.natorb = True
            mc.kernel(coeff)

            mo_coeffs, occupations = self.setup_initial_orbitals(elems) #Returns mo-coeffs and occupations of initial orbitals
            self.setup_active_space(occupations=occupations) #This will define self.norb and self.nelec active space
            self.setup_SHCI_job() #Creates the self.mch CAS-CI/CASSCF object
            self.SHCI_object_set_mos(mo_coeffs=mo_coeffs) #Sets the MO coeffs of mch object              
            self.SHCI_object_run() #Runs the self.mch object
        elif self.blockversion =='Stackblock':
            #???
            ashexit()
        elif self.blockversion =='Block1_5':
            #??
            ashexit()
        print("Block is finished")
        #Cleanup Block scratch stuff (big files)
        self.cleanup()

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        return self.energy


