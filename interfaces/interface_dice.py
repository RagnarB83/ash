import subprocess as sp
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
class DiceTheory:
    def __init__(self, dicedir=None, pyscftheoryobject=None, filename='input.dat', printlevel=2, numcores=1, 
                SHCI=False, NEVPT2=False, AFQMC=False, QMC_trialWF=None, frozencore=True,
                SHCI_stochastic=True, SHCI_PTiter=200, SHCI_sweep_iter= [0,3],
                SHCI_DoRDM=True, SHCI_sweep_epsilon = [ 5e-3, 1e-3 ], SHCI_macroiter=0,
                SHCI_davidsonTol=5e-05, SHCI_dE=1e-08, SHCI_maxiter=9, SHCI_epsilon2=1e-07, SHCI_epsilon2Large=1000,
                SHCI_targetError=0.0001, SHCI_sampleN=200, SHCI_nroots=1,
                SHCI_cas_nmin=1.999, SHCI_cas_nmax=0.0, SHCI_active_space=None,
                QMC_SHCI_numdets=1000, dt=0.005, nsteps=50, nblocks=1000, nwalkers_per_proc=5):

        self.theorynamelabel="Dice"
        self.theorytype="QM"
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")
        
        if dicedir == None:
            print(BC.WARNING, f"No dicedir argument passed to {self.theorynamelabel}Theory. Attempting to find dicedir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.dicedir=ash.settings_ash.settings_dict["dicedir"]
            except:
                print(BC.WARNING,"Found no dicedir variable in settings_ash module either.",BC.END)
                print("Exiting")
                ashexit()
        else:
            self.dicedir = dicedir

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
        #Path to Dice binary
        self.dice_binary=self.dicedir+"/bin/Dice"
        #Put Dice script dir in path
        sys.path.insert(0, dicedir+"/scripts")

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

        #Print stuff
        print("Printlevel:", self.printlevel)
        print("Num cores:", self.numcores)
        print("PySCF object:", self.pyscftheoryobject)
        print("SHCI:", self.SHCI)
        print("NEVPT2:", self.NEVPT2)
        print("AFQMC:", self.AFQMC)
        print("Frozencore:", self.frozencore)
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
            if self.QMC_trialWF is 'SHCI':
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
        except:
            print("Problem importing pyscf.sciscf")
            ashexit()
    def load_qmcutils(self):
        try:
            import QMCUtils
            self.QMCUtils=QMCUtils
        except:
            print("Problem import QMCUtils. Dice directory is probably incorrect")
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
        print("Calling run_and_write_dets")
        #Run once more 
        self.shci.dryrun(self.mch)
        self.shci.writeSHCIConfFile(self.mch.fcisolver, self.mch.nelecas, False)
        with open(self.mch.fcisolver.configFile, 'a') as f:
            f.write(f'writebestdeterminants {numdets}\n\n')
        self.run_shci_directly()

    def run_shci_directly(self):
        print("Calling SHCI PySCF interface")
        #Running Dice via SHCI-PySCF interface
        self.shci.executeSHCI(self.mch.fcisolver)

    # run_dice_directly: In case we need to. Currently unused
    def run_dice_directly(self):
        print("Calling Dice executable directly")
        #For calling Dice directly when needed
        print(f"Running Dice with ({self.numcores} MPI processes)")
        with open('output.dat', "w") as outfile:
            sp.call(['mpirun', '-np', str(self.numcores), self.dice_binary, self.filename], stdout=outfile)

    #Run a SHCI CAS-CI or CASSCF job using the SHCI-PySCF interface 
    def run_SHCI(self):
        print("Will calculate PySCF MP2 natural orbitals to use as input in Dice CAS job")
        natocc, mo_coefficients = self.pyscftheoryobject.calculate_MP2_natural_orbitals(self.pyscftheoryobject.mol,
                                                                                            self.pyscftheoryobject.mf)
        #Updating mf object with MP2-nat MO coefficients
        self.pyscftheoryobject.mf.mo_coeff=mo_coefficients
        #Updating mo-occupations with MP2-nat occupations (pointless?)
        self.pyscftheoryobject.mf.mo_occ=natocc
        if self.SHCI_active_space == None:
            print(f"SHCI Active space determined from MP2 NO threshold parameters: SHCI_cas_nmin={self.SHCI_cas_nmin} and SHCI_cas_nmax={self.SHCI_cas_nmax}")
            print("Note: Use active_space keyword if you want to select active space manually instead")
            # Determing active space from natorb thresholds
            nat_occs_for_thresholds=[i for i in natocc if i < self.SHCI_cas_nmin and i > self.SHCI_cas_nmax]
            norb = len(nat_occs_for_thresholds)
            nelec = round(sum(nat_occs_for_thresholds))
        else:
            print("Active space given as input: active_space=", self.SHCI_active_space)
            # Number of orbital and electrons from active_space keyword!
            nelec=self.SHCI_active_space[0]
            norb=self.SHCI_active_space[1]
        print(f"Doing SHCI-CAS calculation with {nelec} electrons in {norb} orbitals!")
        print("SHCI_macroiter:", self.SHCI_macroiter)
        self.mch = self.shci.SHCISCF( self.pyscftheoryobject.mf, norb, nelec )
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

        #CASSCF iterations
        self.mch.max_cycle_macro = self.SHCI_macroiter

        #Run SHCISCF (ususually only 1 iteration, so CAS-CI)
        self.energy = self.mch.mc1step()[0]


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
        print("pyscftheoryobject:", self.pyscftheoryobject.__dict__)

        #Get frozen-core
        if self.frozencore is True:
            self.determine_frozen_core(qm_elems)
        else:
            self.frozen_core_orbs=0

        # NOW RUNNING
        #NEVPT2
        if self.NEVPT2 is True:
            print("Running Dice NEVPT2 calculation on multiconfigurational WF")
            print("Not ready")
            ashexit()
            #mc=self.pmch
            self.QMCUtils.run_nevpt2(mc, nelecAct=None, numAct=None, norbFrozen=self.frozen_core_orbs,
               integrals="FCIDUMP.h5", nproc=numcores, seed=None,
               fname="nevpt2.json", foutname='nevpt2.out', nroot=0,
               spatialRDMfile=None, spinRDMfile=None, stochasticIterNorms=1000,
               nIterFindInitDets=100, numSCSamples=10000, stochasticIterEachSC=100,
               fixedResTimeNEVPT_Ene=False, epsilon=1.0e-8, efficientNEVPT_2=True,
               determCCVV=True, SCEnergiesBurnIn=50, SCNormsBurnIn=50,
               diceoutfile="dice.out")

            #TODO: Grab energy from function call
            self.energy=0.0
            print("Final Dice NEVPT2 energy:", self.energy)
        #AFQMC
        elif self.AFQMC is True:
            print("Running Dice AFQMC")
            #SHCI trial wavefunction AFQMC
            if self.QMC_trialWF == 'SHCI':
                print("Using multiconfigurational WF via SHCI")
                
                #Calling SHCI run to get 
                self.run_SHCI()
                mc=self.pyscftheoryobject.mch
                
                #Get dets.bin file
                print("Running SHCI (via PySCFTheory object) once again to write dets.bin")
                self.run_and_write_dets(self.QMC_SHCI_numdets)

                #Phaseless AFQMC with hci trial
                e_afqmc, err_afqmc = self.QMCUtils.run_afqmc_mc(mc, mpi_prefix=f"mpirun -np {numcores}",
                                norb_frozen=self.frozen_core_orbs, chol_cut=1e-5,
                                ndets=self.QMC_SHCI_numdets, nroot=0, seed=None,
                                dt=self.dt, steps_per_block=self.nsteps, nwalk_per_proc=self.nwalkers_per_proc,
                                nblocks=self.nblocks, ortho_steps=20, burn_in=50,
                                cholesky_threshold=1.0e-3, weight_cap=None, write_one_rdm=False,
                                use_eri=False, dry_run=False)
                e_afqmc=e_afqmc[0] 
                err_afqmc=err_afqmc[0]
            #Single determinant trial wavefunction
            else:
                print("Using single-determinant WF from PySCF object")
                #Phaseless AFQMC with simple mf trial
                e_afqmc, err_afqmc = self.QMCUtils.run_afqmc_mf(self.pyscftheoryobject.mf, mpi_prefix=f"mpirun -np {numcores}",
                    norb_frozen=self.frozen_core_orbs, chol_cut=1e-5, seed=None, dt=self.dt,
                    steps_per_block=self.nsteps, nwalk_per_proc=self.nwalkers_per_proc, nblocks=self.nblocks,
                    ortho_steps=20, burn_in=50, cholesky_threshold=1.0e-3,
                    weight_cap=None, write_one_rdm=False, dry_run=False)
            self.energy=e_afqmc
            self.error=err_afqmc
            ##Analysis
            print("Final Dice AFQMC energy:", self.energy)
            print(f"Error: {self.error} Eh ({self.error*harkcal} kcal/mol)")

        #Just SHCI 
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

