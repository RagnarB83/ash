import subprocess as sp
import shutil
import time
import numpy as np
import os
import sys
import glob
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.constants import ang2bohr, harkcal
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
from ash.functions.functions_parallel import check_OpenMPI
import ash.settings_ash

#Interface to Dice: SHCI, QMC (single-det or SHCI multi-det) and NEVPT2

#TODO: Remove need for second-iteration print-det
#TODO: fix nevpt2

class DiceTheory:
    def __init__(self, dicedir=None, pyscftheoryobject=None, filename='input.dat', printlevel=2, numcores=1, 
                SHCI=False, NEVPT2=False, AFQMC=False, QMC_trialWF=None, frozencore=True,
                SHCI_stochastic=True, SHCI_PTiter=200, SHCI_sweep_iter= [0,3],
                SHCI_DoRDM=False, SHCI_sweep_epsilon = [ 5e-3, 1e-3 ], SHCI_macroiter=0,
                SHCI_davidsonTol=5e-05, SHCI_dE=1e-08, SHCI_maxiter=9, SHCI_epsilon2=1e-07, SHCI_epsilon2Large=1000,
                SHCI_targetError=0.0001, SHCI_sampleN=200, SHCI_nroots=1,
                SHCI_cas_nmin=1.999, SHCI_cas_nmax=0.0, SHCI_active_space=None,
                read_chkfile_name=None, Dice_SHCI_direct=None, fcidumpfile=None, refdeterminant=None,
                QMC_SHCI_numdets=1000, dt=0.005, nsteps=50, nblocks=1000, nwalkers_per_proc=5):

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
        self.read_chkfile_name=read_chkfile_name
        self.fcidumpfile=fcidumpfile
        self.refdeterminant=refdeterminant
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
        #QMC options
        self.QMC_trialWF=QMC_trialWF
        self.QMC_SHCI_numdets=QMC_SHCI_numdets
        self.frozencore=frozencore
        self.dt=dt
        self.nsteps=nsteps
        self.nblocks=nblocks
        self.nwalkers_per_proc=nwalkers_per_proc
        #If SHCI is used as trial WF we turn off PT stage (timeconsuming)
        if self.AFQMC is True and self.QMC_trialWF == 'SHCI':
            print("AFQMC with SHCI trial WF. Turning off PT stage")
            self.SHCI_stochastic=True #otherwise deterministic PT happens
            self.SHCI_PTiter=0 # PT skipped with this
        #Print stuff
        print("Printlevel:", self.printlevel)
        print("Num cores:", self.numcores)
        print("PySCF object:", self.pyscftheoryobject)
        print("SHCI:", self.SHCI)
        print("NEVPT2:", self.NEVPT2)
        print("AFQMC:", self.AFQMC)
        print("Frozencore:", self.frozencore)
        print("read_chkfile_name:", self.read_chkfile_name)
        print("Dice_SHCI_direct:", self.Dice_SHCI_direct)
        print("FCIDUMP file:", self.fcidumpfile)
        print("Reference det. string:", self.refdeterminant)
        if self.SHCI is True:
            print("SHCI_stochastic", self.SHCI_stochastic)
            print("SHCI_PTiter", self.SHCI_PTiter)
            print("SHCI_sweep_iter", self.SHCI_sweep_iter)
            print("SHCI_DoRDM", self.SHCI_DoRDM)
            print("SHCI_sweep_epsilon", self.SHCI_sweep_epsilon)
            print("SHCI_macroiter", self.SHCI_macroiter)
            print("SHCI CAS NO nmin", self.SHCI_cas_nmin)
            print("SHCI CAS NO nmax", self.SHCI_cas_nmax)
        if self.AFQMC is True:
            print("QMC_trialWF:", self.QMC_trialWF)
            if self.QMC_trialWF == 'SHCI':
                print("QMC_SHCI_numdets:", self.QMC_SHCI_numdets)
            print("QMC settings:")
            print("dt:", self.dt)
            print("Number of steps per block (nsteps):", self.nsteps)
            print("Number of blocks (nblocks):", self.nblocks)
            print("Number of walkers per proc:", self.nwalkers_per_proc)
    
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
        print("Cleaning up Dice stuff")
        #TODO: add more here
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

    #Write dets.bin file. Requires running SHCI once more to get determinants
    def run_and_write_dets(self,numdets):
        module_init_time=time.time()
        print("Calling run_and_write_dets")
        #Run once more 
        self.shci.dryrun(self.mch)
        self.shci.writeSHCIConfFile(self.mch.fcisolver, self.mch.nelecas, False)
        with open(self.mch.fcisolver.configFile, 'a') as f:
            f.write(f'writebestdeterminants {numdets}\n\n')
        self.run_shci_directly()

        print_time_rel(module_init_time, modulename='Dice-extra-step', moduleindex=2)

    def run_shci_directly(self):
        print("Calling SHCI PySCF interface")
        #Running Dice via SHCI-PySCF interface
        print("Dice output can be monitored in output.dat on local scratch")
        self.shci.executeSHCI(self.mch.fcisolver)

        #Grab number of determinants
        self.num_var_determinants= self.grab_num_dets()
        print("Number of variational determinants:", self.num_var_determinants)

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

    #Run Dice-SHCI job without pyscf
    def run_Dice_SHCI(self):
        #Create inputfile input.dat
        nocc=self.SHCI_active_space[0] #how many electrons in active space
        #refdeterminant="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"
        #what orbital indices the electrons occupy in the set of MOs in the FCIDUMP file
        sweepschedule=""
        for it,eps in zip(self.SHCI_sweep_iter,SHCI_sweep_epsilon):
            sweepschedule=sweepschedule+f"{it} {eps}\n"
        #Inputfile creation
        inputstring=f"""

# reference determinant
nocc {nocc}
{self.refdeterminant} 
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
epsilon2 {self.self.SHCI_epsilon2}
epsilon2large {self.self.SHCI_epsilon2Large}
sampleN {self.SHCI_sampleN}
seed 200
targeterror {self.SHCI_targetError}

# Misc
noio
        """
        with open("input.dat", 'w') as f:
            f.write(inputstring)
        
        # Call Dice directly
        self.run_dice_directly()

        #Read energy and determinants from outputfile: output.dat
        #TODO


    # run_dice_directly: In case we need to. Currently unused
    def run_dice_directly(self):
        print("Calling Dice executable directly")
        #For calling Dice directly when needed
        print(f"Running Dice with ({self.numcores} MPI processes)")
        with open('output.dat', "w") as outfile:
            sp.call(['mpirun', '-np', str(self.numcores), self.dice_binary, self.filename], stdout=outfile)

    #Run a SHCI CAS-CI or CASSCF job using the SHCI-PySCF interface
    def run_SHCI(self,verbose=5):
        module_init_time=time.time()
        self.nelec=None
        self.norb=None
        #READ ORBITALS OR DO MP2 natural orbitals
        if self.read_chkfile_name == None:
            print("No checkpoint file given.")
            print("Will calculate PySCF MP2 natural orbitals to use as input in Dice CAS job")
            natocc, mo_coefficients = self.pyscftheoryobject.calculate_MP2_natural_orbitals(self.pyscftheoryobject.mol,
                                                                                            self.pyscftheoryobject.mf)
            #Updating mf object with MP2-nat MO coefficients
            self.pyscftheoryobject.mf.mo_coeff=mo_coefficients
            #Updating mo-occupations with MP2-nat occupations (pointless?)
            self.pyscftheoryobject.mf.mo_occ=natocc
            print(f"SHCI Active space determined from MP2 NO threshold parameters: SHCI_cas_nmin={self.SHCI_cas_nmin} and SHCI_cas_nmax={self.SHCI_cas_nmax}")
            print("Note: Use active_space keyword if you want to select active space manually instead")
            # Determing active space from natorb thresholds
            nat_occs_for_thresholds=[i for i in natocc if i < self.SHCI_cas_nmin and i > self.SHCI_cas_nmax]
            self.norb = len(nat_occs_for_thresholds)
            self.nelec = round(sum(nat_occs_for_thresholds))
            print(f"To get this same active space in another calculation you can also do: SHCI_active_space=[{self.nelec},{self.norb}]")
        else:
            print("Will read MOs from checkpoint file")
            prevmos = self.pyscf.lib.chkfile.load(self.read_chkfile_name, 'mcscf/mo_coeff')
            #Updating mf object with MP2-nat MO coefficients
            self.pyscftheoryobject.mf.mo_coeff=prevmos
            #Updating mo-occupations with MP2-nat occupations (pointless?)
            #self.pyscftheoryobject.mf.mo_occ=natocc

        #This will override norb/nelec if defined by MP2nat orbs above
        if self.SHCI_active_space != None:
            print("Active space given as input: active_space=", self.SHCI_active_space)
            # Number of orbital and electrons from active_space keyword!
            self.nelec=self.SHCI_active_space[0]
            self.norb=self.SHCI_active_space[1]

        #Check if checkpointfile provided but active space not defined. 
        #We don't have occupations to use so an active needs to be defined
        if self.norb == None:
            print("No active space has been defined!")
            print("You probably need to provied SHCI_active_space keyword!")
            ashexit()

        
        print(f"\nDoing SHCI-CAS calculation with {self.nelec} electrons in {self.norb} orbitals!")
        print("SHCI_macroiter:", self.SHCI_macroiter)
        self.mch = self.shci.SHCISCF( self.pyscftheoryobject.mf, self.norb, self.nelec )
        self.mch.fcisolver.mpiprefix = f'mpirun -np {self.numcores}'
        self.mch.fcisolver.stochastic = self.SHCI_stochastic
        self.mch.fcisolver.nPTiter = self.SHCI_PTiter
        self.mch.fcisolver.sweep_iter = self.SHCI_sweep_iter
        self.mch.fcisolver.DoRDM = self.SHCI_DoRDM
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
        #
        if self.SHCI_macroiter == 0:
            print("SHCI_macroiter: 0. This means CAS-CI.")
            print("Turning off canonicalization step in mcscf object")
            self.mch.canonicalization = False
        #CASSCF iterations
        self.mch.max_cycle_macro = self.SHCI_macroiter

        #Run SHCISCF (ususually only 1 iteration CAS-CI, unless self.SHCI_macroiter > 0)
        print("Dice output can be monitored in output.dat on local scratch")
        self.energy = self.mch.mc1step()[0]

        #Grab number of determinants
        self.num_var_determinants= self.grab_num_dets()
        print("Number of variational determinants:", self.num_var_determinants)

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
        self.pyscftheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)

        #Get frozen-core
        if self.frozencore is True:
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
            self.run_SHCI()
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
                
                #Calling SHCI run to get the self.mch object
                self.run_SHCI()

                #Get dets.bin file
                print("\nRunning SHCI (via PySCFTheory object) once again to write dets.bin")
                self.run_and_write_dets(self.QMC_SHCI_numdets)
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
                                dt=self.dt, steps_per_block=self.nsteps, nwalk_per_proc=self.nwalkers_per_proc,
                                nblocks=self.nblocks, ortho_steps=20, burn_in=50,
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
                    norb_frozen=self.frozen_core_orbs, chol_cut=1e-5, seed=None, dt=self.dt,
                    steps_per_block=self.nsteps, nwalk_per_proc=self.nwalkers_per_proc, nblocks=self.nblocks,
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
            print("Running SHCI option via Dice without pyscf")
            self.run_Dice_SHCI()
            #TODO: Grab energy
        #Just SHCI via PySCF
        else:
            if self.SHCI is True:
                print("Standalone SHCI option is active.")
                self.run_SHCI()
                print("Final Dice SHCI energy:", self.energy)

        print("Dice is finished")
        #Cleanup Dice scratch stuff (big files)
        self.cleanup()

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        return self.energy


