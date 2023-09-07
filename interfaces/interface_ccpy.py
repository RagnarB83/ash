import subprocess as sp
import shutil
import time
import numpy as np
import os
import sys
import glob
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.constants import ang2bohr, harkcal
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader,pygrep,pygrep2,find_program
from ash.functions.functions_parallel import check_OpenMPI
import ash.settings_ash

#Interface to ccpy: https://github.com/piecuch-group/ccpy
#Coupled cluster package in python



class ccpyTheory:
    def __init__(self, pyscftheoryobject=None, filename='input.dat', printlevel=2,
                moreadfile=None, initial_orbitals='MP2', memory=20000, frozencore=True, tol=1e-10, numcores=1, 
                method="CCPQ", adaptive=False, percentages=None):

        self.theorynamelabel="ccpy"
        self.theorytype="QM"
        self.analytic_hessian=False
        
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        #Check for PySCFTheory object 
        if pyscftheoryobject is None:
            print("Error: No pyscftheoryobject was provided. This is required")
            ashexit()

        #MAKING SURE WE HAVE ccpy
        try:
            import ccpy
        except ModuleNotFoundError:
            print("Error: ccpy module is not installed. Please install ccpy")
            ashexit()

        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.numcores=numcores
        #
        self.method=method # Options: 'CCSD', 'CCSDT', CCPQ
        self.adaptive=adaptive #Adaptive CC(P;Q) or not 
        self.percentages=percentages #What triples percentages to loop through

        self.pyscftheoryobject=pyscftheoryobject

        self.moreadfile=moreadfile
        self.tol=tol
        self.frozencore=frozencore
        self.memory=memory #Memory in MB (total) assigned to PySCF mcscf object
        self.initial_orbitals=initial_orbitals #Initial orbitals to be used (unless moreadfile option)

        #CCpy adaptive
        if adaptive is True and self.percentages is None:
            print("Error. Adaptive is True but no percentages provided")
            print("Setting percentages list to: [1.0]")
            self.percentages=[1.0]            

        #Print stuff
        print("Printlevel:", self.printlevel)
        print("PySCF object:", self.pyscftheoryobject)
        print("Num cores:", self.numcores)
        print("Memory (MB)", self.memory)
        print("Frozencore:", self.frozencore)
        print("moreadfile:", self.moreadfile)
        print("Initial orbitals:", self.initial_orbitals)
        print("Tolerance", self.tol)

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleaning up  temporary files")

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

        #Check symmetry in pyscf mol object
        if self.pyscftheoryobject.mol.symmetry is None:
            self.pyscftheoryobject.mol.symmetry='C1'

        #Get frozen-core
        if self.frozencore is True:
            self.determine_frozen_core(qm_elems)
        else:
            self.frozen_core_orbs=0

        #Create ccpy driver object
        from ccpy.drivers.driver import Driver, AdaptDriver
        print("self.frozen_core_orbs:",self.frozen_core_orbs)
        print("self.pyscftheoryobject.mf:", self.pyscftheoryobject.mf)
        driver = Driver.from_pyscf(self.pyscftheoryobject.mf, nfrozen=self.frozen_core_orbs)
        print("driver:",driver)
        driver.system.print_info()

        if self.adaptive is True:
            print("adaptive CC(P;Q) calculation.")
            print("self.percentages:",self.percentages)
            adaptdriver = AdaptDriver(
                    driver,
                    self.percentages,
                    full_storage=False,
                    perturbative=False,
                    pspace_analysis=False,
                )
            adaptdriver.run()

            print("CC(P) Energies:", adaptdriver.ccp_energy)
            print("CC(P;Q) Energies:", adaptdriver.ccpq_energy)
            self.CC_PQ_energies=adaptdriver.ccpq_energy
            self.energy = float(adaptdriver.ccpq_energy[-1])

        else:
            print("Non-adaptive CC calculation.")
            CCSD_corr_energy=0.0
            CCSDt_corr_energy=0.0
            HOC_energy=0.0
            total_corr_energy=0.0
            #driver.options["maximum_iterations"] = 1000 # 4 Sigma state requires ~661 iterations in left-CCSD
            #driver.options["davidson_max_subspace_size"] = 50
            if self.method.lower() == "ccsd" :
                driver.run_cc(method=self.method)
                CCSD_corr_energy=driver.correlation_energy
                total_corr_energy=driver.correlation_energy
            elif self.method.lower() == "ccsdt1":
                driver.run_cc(method=self.method)
                total_corr_energy=driver.correlation_energy            
            elif self.method.lower() == "ccsdt":
                driver.run_cc(method=self.method)
                total_corr_energy=driver.correlation_energy
            elif self.method.lower() == "ccsd(t)":
                driver.run_cc(method="ccsd")
                CCSD_corr_energy=driver.correlation_energy
                driver.run_ccp3(method="ccsd(t)")
                HOC_energy = driver.deltapq[0]["A"]
                total_corr_energy = CCSD_corr_energy + HOC_energy
            elif self.method.lower() == "cct3":
                driver.run_cc(method="ccsdt1")
                driver.run_hbar(method="ccsd")
                driver.run_leftcc(method="left_ccsd")
                driver.run_ccp3(method="cct3")
                #CC(t;3)_A
                print("CC(t;3)_A:", driver.deltapq[0]["A"])
                #CC(t;3)_D
                print("CC(t;3)_A:", driver.deltapq[0]["D"])
                HOC_energy = driver.deltapq[0]["D"]
                CCSDt_corr_energy=driver.correlation_energy
                total_corr_energy = CCSDt_corr_energy + HOC_energy
            elif self.method.lower() == "crcc23":
                driver.run_cc(method="ccsd")
                CCSD_corr_energy=driver.correlation_energy
                driver.run_hbar(method="ccsd")
                driver.run_leftcc(method="left_ccsd")
                driver.run_ccp3(method="crcc23")

                print("driver.deltapq:", driver.deltapq)
                print("driver.deltapq:[0]", driver.deltapq[0])
                print("driver.deltapq[0] D", driver.deltapq[0]["D"])
                HOC_energy=driver.deltapq[0]["D"]
                total_corr_energy = CCSD_corr_energy + HOC_energy
            elif 'eom' in self.method.lower():
                driver.run_eomcc(method=self.method.lower(), state_index=1)
            
            reference_energy = driver.system.reference_energy
            
            #driver.run_hbar(method="ccsd")
            #driver.run_guess(method="cis", multiplicity=1, nroot=10)
            #driver.run_eomcc(method="eomccsd", state_index=selected_states[1:])
            self.energy =  reference_energy + total_corr_energy
            print("driver dict:", driver.__dict__)
            print()
            print(f"Reference energy {reference_energy} Eh")
            print(f"Total correlation energy ({self.method}) {total_corr_energy} Eh")
            if CCSD_corr_energy != 0.0:
                print(f"   CCSD correlation energy {CCSD_corr_energy} Eh")
            if CCSDt_corr_energy != 0.0:
                print(f"   CCSDt correlation energy {CCSDt_corr_energy} Eh")
            if HOC_energy != 0.0:
                print(f"   HOC correlation energy {HOC_energy} Eh")
            print(f"Total energy {self.energy} Eh")



        print("ccpy is finished")
        #Cleanup scratch stuff (big files)
        self.cleanup()

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel}Theory run', moduleindex=2)
        return self.energy


