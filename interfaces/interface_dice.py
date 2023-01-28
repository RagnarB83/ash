import subprocess as sp
import time
import numpy as np
import os
import sys
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.constants import ang2bohr, harkcal
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
from ash.functions.functions_parallel import check_OpenMPI
import ash.settings_ash

#Interface to Dice: QMC and NEVPT2
class DiceTheory:
    def __init__(self, dicedir=None, pyscftheoryobject=None, filename='input.json', printlevel=2,
                numcores=1, nevpt2=False, AFQMC=False, trialWF=None, frozencore=False):

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

        #Put Dice script dir in path
        sys.path.insert(0, dicedir+"/scripts")
        #Import various functionality 
        import QMCUtils
        self.QMCUtils=QMCUtils

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
        self.nevpt2=nevpt2
        self.AFQMC=AFQMC
        self.trialWF=trialWF
        self.frozencore=frozencore

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("not ready")
        #print(f"Deleting old checkpoint file: {self.checkpointfilename}")
        #files=[self.checkpointfilename]
        #for file in files:
        #    try:
        #        os.remove(file)
        #    except:
        #        pass
    def determine_frozen_core(self,elems):
        print("Determining frozen core")
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

        #Run PySCF to get integrals
        #self.pyscftheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)

        #Read PySCF checkpointfile and create ipie inputfile
        if self.frozencore is True:
            self.determine_frozen_core(qm_elems)
            #self.frozen_core_orb

        #NEVPT2 or AFQMC
        if self.NEVPT2 is True:
            self.QMCUtils.run_nevpt2(mc, nelecAct=None, numAct=None, norbFrozen=None,
               integrals="FCIDUMP.h5", nproc=None, seed=None,
               fname="nevpt2.json", foutname='nevpt2.out', nroot=0,
               spatialRDMfile=None, spinRDMfile=None, stochasticIterNorms=1000,
               nIterFindInitDets=100, numSCSamples=10000, stochasticIterEachSC=100,
               fixedResTimeNEVPT_Ene=False, epsilon=1.0e-8, efficientNEVPT_2=True,
               determCCVV=True, SCEnergiesBurnIn=50, SCNormsBurnIn=50,
               vmc_root=None, diceoutfile="dice.out")
        elif self.AFQMC is True:
            #QMCUtils.run_afqmc(mc, ndets = 100, norb_frozen = norb_frozen)
            if self.trialWF == 'SHCI':
                #mc=
                #Phaseless AFQMC with hci trial
                self.QMCUtils.run_afqmc_mc(mc, vmc_root=None, mpi_prefix=None,
                                norb_frozen=0, nproc=None, chol_cut=1e-5,
                                ndets=100, nroot=0, seed=None,
                                dt=0.005, steps_per_block=50, nwalk_per_proc=5,
                                nblocks=1000, ortho_steps=20, burn_in=50,
                                cholesky_threshold=1.0e-3, weight_cap=None, write_one_rdm=False,
                                run_dir=None, scratch_dir=None, use_eri=False,
                                dry_run=False)
            else:
                #mf=
                #Phaseless AFQMC with simple mf trial
                self.QMCUtils.run_afqmc_mf(mf, vmc_root=None, mpi_prefix=None,
                    mo_coeff=None, norb_frozen=0, nproc=None,
                    chol_cut=1e-5, seed=None, dt=0.005,
                    steps_per_block=50, nwalk_per_proc=5, nblocks=1000,
                    ortho_steps=20, burn_in=50, cholesky_threshold=1.0e-3,
                    weight_cap=None, write_one_rdm=False, run_dir=None,
                    scratch_dir=None, dry_run=False)
                #General??
                #self.QMCUtils.run_afqmc(mf_or_mc, vmc_root=None, mpi_prefix=None,
                #        mo_coeff=None, ndets=100, nroot=0,
                #        norb_frozen=0, nproc=None, chol_cut=1e-5,
                #        seed=None, dt=0.005, steps_per_block=50,
                #        nwalk_per_proc=5, nblocks=1000, ortho_steps=20,
                #        burn_in=50, cholesky_threshold=1.0e-3, weight_cap=None,
                #        write_one_rdm=False, run_dir=None, scratch_dir=None, 
                #        use_eri=False, dry_run=False)
        else:
            print("Unknown Dice run option")
            ashexit()
        #Parallel
        if self.numcores > 1:
            print(f"Running Dice with MPI parallelization ({self.numcores} MPI processes)")






        else:
            print("bla")
            exit()

        print("Dice finished")
        ##Analysis
        

        print("Final Dice energy:", E_final)
        print(f"Error: {error} Eh ({error*harkcal:.2f} kcal/mol)")
        print("nsamp_ac:", nsamp_ac)
        self.energy=E_final
        self.error=error

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        return self.energy


