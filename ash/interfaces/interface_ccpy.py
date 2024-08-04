import time
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
from ash.functions.functions_parallel import check_OpenMPI
from ash.interfaces.interface_ORCA import read_ORCA_json_file, read_ORCA_bson_file
import numpy as np
import os

# Interface to ccpy: https://github.com/piecuch-group/ccpy
# Coupled cluster package in python

# TODO: GAMESS option once GAMESS interface available
# TODO: Change pyscftheoryobject and orcatheoryobject to theory= ?

class ccpyTheory:
    def __init__(self, pyscftheoryobject=None, orcatheoryobject=None, orca_jsonformat="json",
                 fcidumpfile=None, filename=None, printlevel=2, label="ccpy",
                frozencore=True, cc_tol=1e-8, numcores=1,
                cc_maxiter=300, cc_amp_convergence=1e-7, nact_occupied=None, nact_unoccupied=None, civecs_file=None, 
                method=None, percentages=None, states=None, roots_per_irrep=None, EOM_guess_symmetry=False,
                two_body_approx=False):

        self.theorynamelabel="ccpy"
        self.theorytype="QM"
        self.analytic_hessian=False

        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        # Check for PySCFTheory object
        if pyscftheoryobject is None and orcatheoryobject is None and fcidumpfile is None:
            print("Error: No pyscftheoryobject or orcatheoryobject was provided and fcidumpfile is none. Either option is required")
            ashexit()

        # MAKING SURE WE HAVE ccpy
        try:
            import ccpy
        except ModuleNotFoundError as e:
            print("Error: ccpy module is not installed. Please install ccpy")
            print("Error message:", e)
            ashexit()

        # Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.numcores=numcores
        #
        self.method=method.lower() # Lower-case name of method
        self.pyscftheoryobject=pyscftheoryobject
        self.orcatheoryobject=orcatheoryobject
        self.orca_jsonformat=orca_jsonformat #json or bson
        self.fcidumpfile=fcidumpfile

        self.frozencore=frozencore

        # ccpy options
        self.cc_tol=cc_tol
        self.cc_maxiter=cc_maxiter #Maximum number of iterations for CC calculation
        self.cc_amp_convergence=cc_amp_convergence

        # Adaptive CC(P;Q)
        self.adaptive=False #Initially set to False
        self.percentages=percentages #What triples percentages to loop through

        # Active space CC methods
        self.nact_occupied=nact_occupied
        self.nact_unoccupied=nact_unoccupied

        # Two-body approximation
        # Can be used for both ccp3 and adaptive CC(P;Q)
        self.two_body_approx=two_body_approx

        # CIPSI-driven methods require civecs_file (file containing CI vectors)
        self.civecs_file=civecs_file

        # EOM 
        self.states=states #List of states: states=[1, 2, 3, 4, 5, 6, 7]
        if roots_per_irrep is None:
            self.roots_per_irrep={}
        else:
            self.roots_per_irrep=roots_per_irrep
        self.EOM_guess_symmetry=EOM_guess_symmetry

        ############################
        # METHODS available
        ############################
        # Simple methods available
        # CC4 and CCSDTQ : RHF only
        self.simple_methods = ["ccd", "ccsd", "ccsdt", "ccsdtq", "cc3", "cc4"]
        # Activespace_methods requires active space
        self.activespace_methods=["ccsdt1", "cct3", "ccsdt_p"]
        # CR-CC methods (completely renormalized CC)
        self.cr_cc_methods=["crcc23", "crcc24"]
        # Adaptive methods
        self.adaptive_methods=["adaptive-cc(p;q)"]
        # CIPSI-driven methods
        self.cipsi_methods=["eccc23", "eccc24", "cipsi-cc(p;q)"]
        # EOM methods
        self.eom_methods=["eomccsd", "eomcc3", "eomccsdt", "eomccsdt(a)_star", 
                          "creom23", "ipeom2", "ipeom3",
                          "ipeomccsdta_star"]
        if self.method is None:
            print("No valid method selected")
            print("CCPy interface supports the following methods:")
            print("Simple CC methods:", self.simple_methods)
            print("Active-space CC methods:", self.activespace_methods)
            print("CR-CC methods:", self.cr_cc_methods)
            print("Adaptive CC(P;Q):", self.adaptive_methods)
            print("CIPSI-driven CC methods:", self.cipsi_methods)
            print("EOM-methods:", self.eom_methods)
            print("Not all EOM methods not supported yet")

        print("\nMethod chosen:", self.method)
        # Activespace-method: check for active space info
        if self.method in self.activespace_methods:
            if self.nact_occupied is None or self.nact_unoccupied is None:
                print(f"The active-space CC-method {self.method} requires setting an active space!")
                print("Set keywords: nact_occupied and nact_unoccupied")
                ashexit()
        # Adaptive-CC:
        if "adaptive" in self.method:
            print("adaptive found in method name")
            print("Assuming adaptive CC(P;Q) requested")
            self.adaptive=True
        if self.adaptive is True and self.percentages is None:
            print("Error. Adaptive is True but no percentages provided")
            print("Setting percentages list to: [0.0,1.0]")
            self.percentages=[0.0,1.0]
        # CIPSI-driven CC
        self.civecs_file = civecs_file
        if self.method in self.cipsi_methods and self.civecs_file is None:
            print(f" Method {self.method} requires civecs_file to be set")
            ashexit()
        # EOM
        if self.method in self.eom_methods and self.states is None:
            print("EOM-method selected but states keyword not set")
            print("Set states to be list of states to calcualate")
            ashexit()

        # Print stuff
        print("Printlevel:", self.printlevel)
        print("PySCFTheory object:", self.pyscftheoryobject)
        print("ORCATheory object:", self.orcatheoryobject)
        print("FCIDUMP file:", self.fcidumpfile)
        print("Num cores:", self.numcores)
        print("Frozencore:", self.frozencore)
        print("Tolerance", self.cc_tol)

    # Set numcores method
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

    def run_CRCC(self,driver):
        driver.run_cc(method="ccsd")
        CCSD_corr_energy=driver.correlation_energy
        driver.run_hbar(method="ccsd")
        driver.run_leftcc(method="left_ccsd")
        driver.run_ccp3(method="crcc23")
        if "24" in self.method:
            driver.run_ccp4(method="crcc24")
            HOC_energy = driver.deltap3[0]["D"] +  driver.deltap4[0]["D"] 
        else:
            HOC_energy=driver.deltap3[0]["D"]
        total_corr_energy = CCSD_corr_energy + HOC_energy

        return total_corr_energy, CCSD_corr_energy, HOC_energy

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)

        # Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # Cleanup before run.
        self.cleanup()

        # Get frozen-core
        if self.frozencore is True:
            self.determine_frozen_core(qm_elems)
        else:
            self.frozen_core_orbs=0
        print("self.frozen_core_orbs:",self.frozen_core_orbs)
        
        ##########################
        # CREATE DRIVER
        ##########################
        from ccpy.drivers.driver import Driver

        # OPTION 1: DRIVER via FCIDUMP meanfield
        if self.fcidumpfile is not None:
            print("FCIDUMP file provided:", self.fcidumpfile)
            driver = Driver.from_fcidump(self.fcidumpfile, nfrozen=self.frozen_core_orbs, charge=charge, 
                                         rohf_canonicalization="Roothaan")

        # OPTION 2: DRIVER via pyscftheoryobject
        # Run PySCF to get integrals and MOs. This would probably only be an SCF
        elif self.pyscftheoryobject is not None:
            print("PySCFTheory object provided")
            self.pyscftheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)

            # Check symmetry in pyscf mol object
            if self.pyscftheoryobject.mol.symmetry is None:
                self.pyscftheoryobject.mol.symmetry='C1'

            driver = Driver.from_pyscf(self.pyscftheoryobject.mf, nfrozen=self.frozen_core_orbs)
        # OPTION 3: DRIVER via orcatheoryobject
        # Run ORCA to get integrals and MOs. 
        elif self.orcatheoryobject is not None:
            print("ORCA object provided")
            print("Now running ORCATheory object")
            self.orcatheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)

            # JSON-file create
            from ash.interfaces.interface_ORCA import create_ORCA_json_file
            print("Creating JSON file")
            jsonfile = create_ORCA_json_file(self.orcatheoryobject.filename+'.gbw', two_el_integrals=True, format=self.orca_jsonformat)
            print("Loading integrals from JSON file")
            system, hamiltonian = load_orca_integrals( jsonfile, nfrozen=self.frozen_core_orbs)
            print("Deleting JSON file")
            os.remove(jsonfile)
            driver = Driver(system, hamiltonian, max_number_states=50)
            # Check symmetry
            #TODO

            #driver = Driver.from_pyscf(self.pyscftheoryobject.mf, nfrozen=self.frozen_core_orbs)
            #driver = 


        # Set active space in driver.system before if required
        if self.method in self.activespace_methods:
            driver.system.set_active_space(nact_occupied=self.nact_occupied, 
                                    nact_unoccupied=self.nact_unoccupied)
        # CIPSI-driven CC requires T3 and T4 excitations from CI-vectors file

        # Some DRIVER settings
        driver.options["maximum_iterations"] = self.cc_maxiter
        driver.options["energy_convergence"] = self.cc_tol
        driver.options["amp_convergence"] = self.cc_amp_convergence
        driver.options["maximum_iterations"] = self.cc_maxiter

        # Print driver info
        driver.system.print_info()

        if self.method in self.cipsi_methods:
            if self.method == "eccc23" or self.method == "cipsi-cc(p;q)":
                from ccpy.utilities.pspace import get_pspace_from_cipsi
                _, t3_excitations, _ = get_pspace_from_cipsi(self.civecs_file, driver.system, nexcit=3)
            elif self.method == "eccc24":
                from ccpy.utilities.pspace import get_triples_pspace_from_cipsi, get_quadruples_pspace_from_cipsi
                # T3 and T4
                print("Reading triples excitations from CIPSI")
                t3_excitations, _ = get_triples_pspace_from_cipsi(self.civecs_file, driver.system)
                print("Reading quadruples excitations from CIPSI")
                t4_excitations, _ = get_quadruples_pspace_from_cipsi(self.civecs_file, driver.system)

        ######################################
        # RUN
        ######################################
        GS_label=self.method #Label to use in final printing. Updated below
        if self.adaptive is True:

            print("Adaptive CC(P;Q) calculation.")
            print("self.percentages:", self.percentages)
            from ccpy.drivers.adaptive import AdaptDriver
            adaptdriver = AdaptDriver(
                    driver, percentage=self.percentages)
            adaptdriver.options["energy_tolerance"]= self.cc_tol
            adaptdriver.options["two_body_approx"] = self.two_body_approx
            adaptdriver.run()

            print("CC(P) Energies:", adaptdriver.ccp_energy)
            print("CC(P;Q) Energies:", adaptdriver.ccpq_energy)
            self.CC_PQ_energies=adaptdriver.ccpq_energy
            self.energy = float(adaptdriver.ccpq_energy[-1])

            GS_label="CC(P;Q)" #Label to use in final printing
            
        else:
            print("Non-adaptive CC calculation.")

            ########################
            # GROUND-STATE CC
            ########################
            CCSD_corr_energy=0.0
            CCSDt_corr_energy=0.0
            HOC_energy=0.0
            total_corr_energy=0.0
            if self.method in self.simple_methods:
                driver.run_cc(method=self.method)
                total_corr_energy=driver.correlation_energy
            # Perturbative methods: CCSD(T)
            elif self.method == "ccsd(t)":
                GS_label="CCSD(T)" #Label to use in final printing
                driver.run_cc(method="ccsd")
                CCSD_corr_energy=driver.correlation_energy
                driver.run_ccp3(method="ccsd(t)")
                HOC_energy = driver.deltap3[0]["A"]
                total_corr_energy = CCSD_corr_energy + HOC_energy
            # Active-space CC methods
            elif self.method == "ccsdt1":
                GS_label="CCSDt" #Label to use in final printing
                driver.run_cc(method=self.method)
                total_corr_energy=driver.correlation_energy
            elif self.method == "ccsdt_p":
                GS_label="CCSDt (via CC(P))" #Label to use in final printing
                from ccpy import Driver, get_active_triples_space
                # Obtain the list of triples excitations corresponding to the CCSDt truncation (ground-state symmetry adapted)
                t3_excitations = get_active_triples_space(driver.system, 
                                                          target_irrep=driver.system.reference_symmetry)

                # Run active-space CCSDt calculation via general CC(P) solver
                driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
            elif self.method == "cct3":
                GS_label="CC(t;3)" #Label to use in final printing
                driver.run_cc(method="ccsdt1")
                driver.run_hbar(method="ccsd")
                driver.run_leftcc(method="left_ccsd")
                driver.run_ccp3(method="cct3", two_body_approx=self.two_body_approx)
                # CC(t;3)_A
                print("CC(t;3)_A:", driver.deltap3[0]["A"])
                # CC(t;3)_D
                print("CC(t;3)_A:", driver.deltap3[0]["D"])
                HOC_energy = driver.deltap3[0]["D"]
                CCSDt_corr_energy=driver.correlation_energy
                total_corr_energy = CCSDt_corr_energy + HOC_energy
            # CR-CC methods (CR-CC(2,3) or CR-CC(2,4))
            elif self.method in self.cr_cc_methods:
                # Calling method
                total_corr_energy, CCSD_corr_energy, HOC_energy = self.run_CRCC(driver)
                GS_label="CR-CC" #Label to use in final printing
            elif self.method in self.cipsi_methods:
                print("CIPSI-driven CC chosen")
                if "ec" in self.method:
                    # Applies to both eccc23 and eccc24
                    print("EC method chosen")
                    driver.run_eccc(method="eccc2", ci_vectors_file=self.civecs_file)
                    driver.run_hbar(method="ccsd")
                    driver.run_leftcc(method="left_ccsd")
                    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations)
                elif self.method == "cipsi-cc(p;q)":
                    print("CIPSI CC(P;Q) method chosen")
                    GS_label="CIPIS-CC(P;Q)" #Label to use in final printing
                    print("Note: Only T3 excitations")
                    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
                    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
                    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
                    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations, two_body_approx=self.two_body_approx)

                CCSD_corr_energy=driver.correlation_energy

                if self.method == "eccc24":
                    GS_label="EC-CC(2,4)" #Label to use in final printing
                    driver.run_ccp4(method="ccp4", state_index=0, t4_excitations=t4_excitations)
                    HOC_energy = driver.deltap3[0]["D"] + driver.deltap4[0]["D"]
                elif self.method == "eccc23":
                    GS_label="EC-CC(2,3)" #Label to use in final printing
                    HOC_energy = driver.deltap3[0]["D"]
                elif self.method == "cipsi-cc(p;q)":
                    HOC_energy = driver.deltap3[0]["D"]

                total_corr_energy = CCSD_corr_energy + HOC_energy
            

            ########################
            # EXCITED-STATE CC
            ########################

            #TODO: missing EOM-CCT3
            #TODO: missing Active-space IP-EOMCCSD
            #TODO: Grab EOM contributions

            elif self.method in self.eom_methods:
                #Configure options
                left_method=None
                ccp3_method=None
                ipccp3_method=None
                run_GS=True # Will be done except CR-CC below (done differently)
                mult_for_guess=mult
                # IP-EOM
                #TODO EA
                if 'ip' in self.method:
                    # IP-EOMCCSD(2h-1p) 
                    # IP-EOMCCSD(3h-2p)
                    # IP-EOMCCSDT(a)*
                    gsmethod="ccsd"
                    GS_label="CCSD" #Label to use in final printing
                    guessmethod="ipcisd" #CIS is usually the guess
                    mult_for_guess=-1 # for IP should be like this

                    if '3' in self.method:
                        hbarmethod="ccsd"
                        ipeommethod="ipeom3"
                        leftipmeethod="left_ipeom3"
                    elif '2' in self.method:
                        hbarmethod="ccsd"
                        ipeommethod="ipeom2"
                        leftipmeethod="left_ipeom2"
                    elif 'star' in self.method:
                        # ipeomccsdta_star
                        hbarmethod="ccsdta"
                        ipeommethod="ipeom2"
                        leftipmeethod="left_ipeom2"
                        ipccp3_method="ipeomccsdta_star"
                # CREOM
                elif 'cr' in self.method:
                    run_GS=False
                    # Running GS CR_CC problem first
                    total_corr_energy, CCSD_corr_energy, HOC_energy = self.run_CRCC(self.method)
                    GS_label="CR-CC" #Label to use in final printing
                    guessmethod="cisd"
                    eomrunmethod="eomccsd"
                    left_method="left_ccsd"
                    ccp3_method="crcc23"
                # EOM-CCSDT(a)*
                elif 'eomccsdt(a)_star' in self.method:
                    gsmethod="ccsd"
                    GS_label="CCSD" #Label to use in final printing
                    hbarmethod="ccsdta"
                    guessmethod="cisd"
                    eomrunmethod="eomccsd" #correct
                    left_method="left_ccsd"
                    ccp3_method="eomccsdta_star"
                # EOM-CCSDT
                elif 'ccsdt' in self.method:
                    gsmethod="ccsdt"
                    hbarmethod="ccsdt"
                    guessmethod="cis" #CIS is usually the guess
                    eomrunmethod="eomccsd"
                    GS_label="CCSDT" #Label to use in final printing
                # EOM-CCSD
                elif 'ccsd' in self.method:
                    gsmethod="ccsd"
                    GS_label="CCSD" #Label to use in final printing
                    hbarmethod="ccsd"
                    guessmethod="cis" #CIS is usually the guess
                    eomrunmethod="eomccsd"
                # EOM-CC3
                elif 'cc3' in self.method:
                    gsmethod="cc3"
                    GS_label="CC3" #Label to use in final printing
                    hbarmethod="cc3"
                    guessmethod="cisd"
                    eomrunmethod="eomcc3"

                # Run Ground-state CC and HBar if required
                if run_GS:
                    driver.run_cc(method=gsmethod)
                    driver.run_hbar(method=hbarmethod)

                # Saving GS energy
                total_corr_energy=driver.correlation_energy
                GS_CC_energy =  driver.system.reference_energy + driver.correlation_energy
                # Run Guess
                driver.run_guess(method=guessmethod, multiplicity=mult_for_guess, use_symmetry=self.EOM_guess_symmetry,
                                 nact_occupied=self.nact_occupied, nact_unoccupied=self.nact_unoccupied,
                                 roots_per_irrep=self.roots_per_irrep) #roots_per_irrep={"A1": 1, "B1": 1, "B2": 0, "A2": 1}

                # Run IP-EOM-CC 
                if 'ip' in self.method:
                    driver.run_ipeomcc(method=ipeommethod, state_index=self.states)
                    driver.run_leftipeomcc(method=leftipmeethod, state_index=self.states)
                # Run EOM-CC
                else:
                    driver.run_eomcc(method=eomrunmethod, state_index=self.states)

                    # Run left EOM problem if needed
                    if left_method is not None:
                        driver.run_leftcc(method=left_method, state_index=self.states)

                # Perturbative
                if ccp3_method is not None:
                    # Compute EOMCCSDT(a)* excited-state corrections
                    driver.run_ccp3(method=ccp3_method, state_index=self.states)
                if ipccp3_method is not None:
                    # Compute IPEOMCCSDT(a)* excited-state corrections
                    driver.run_ipccp3(method=ipccp3_method, state_index=self.states)


            else:
                print("Error. Method not recognized")
                ashexit()

            # Reference energy
            reference_energy = driver.system.reference_energy

            # Total energy: Ref_energy + total_corr_energy (all corr)
            self.energy =  reference_energy + total_corr_energy
            print()
            print("-"*70)
            print("GROUND-STATE COUPLED CLUSTER RESULT")
            print("-"*70)
            print(f"Reference energy {reference_energy} Eh")
            print(f"Total correlation energy ({GS_label}) {total_corr_energy} Eh")
            # Print special contributions if defined
            if CCSD_corr_energy != 0.0:
                print(f"   CCSD correlation energy {CCSD_corr_energy} Eh")
            if CCSDt_corr_energy != 0.0:
                print(f"   CCSDt correlation energy {CCSDt_corr_energy} Eh")
            if HOC_energy != 0.0:
                print(f"   HOC correlation energy {HOC_energy} Eh")
            print(f"Total energy {self.energy} Eh")

            # PRINT EOM excitations
            if self.method in self.eom_methods:
                # Excitation energy
                excitation_energies = driver.vertical_excitation_energy
                print("EOM-CC excitation energies:", excitation_energies)
                print()
                print("List of all CC states:")
                print("-"*80)
                print(" State   Type       Total Energy (Eh)        Excitation energy (eV)")
                print("-"*80)
                print(f" {0:3d}      (GS)        {GS_CC_energy:<13.10f}")
                for i in range(0,len(driver.guess_energy)):
                    EE=excitation_energies[i]
                    print(f" {i+1:3d}      (ES)        {GS_CC_energy+EE:<13.10f}             {EE*27.211386245988:>7.4f}")

        print("ccpy is finished")

        self.driver=driver
        # Cleanup scratch stuff (big files)
        self.cleanup()

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel}Theory run', moduleindex=2)
        return self.energy


# Load integrals directly from ORCA json-file
# TODO: support bson
def load_orca_integrals(
        jsonfile, nfrozen=0, ndelete=0,
        num_act_holes_alpha=0, num_act_particles_alpha=0,
        num_act_holes_beta=0, num_act_particles_beta=0,
        use_cholesky=False, cholesky_tol=1.0e-09,
        normal_ordered=True, dump_integrals=False, sorted=True):

    # import System
    from ccpy.models.system import System
    from ccpy.models.integrals import getHamiltonian
    from ccpy.utilities.dumping import dumpIntegralstoPGFiles
    from ccpy.energy.hf_energy import calc_hf_energy, calc_hf_frozen_core_energy

    if '.json' in jsonfile:
        json_data = read_ORCA_json_file(jsonfile)
    elif '.bson' in jsonfile:
        json_data = read_ORCA_bson_file(jsonfile)
    else:
        print(f"File {jsonfile} does not have .json or .bson ending. Unknown format")
        ashexit()

    # Nuc-repulsion
    from ash.modules.module_coords import nuc_nuc_repulsion
    coords = np.array([i["Coords"] for i in json_data["Atoms"]])
    nuc_charges = np.array([i["ElementNumber"] for i in json_data["Atoms"]])
    nuclear_repulsion = nuc_nuc_repulsion(coords, nuc_charges)

    # Molecule charge and spin and symmetry 
    molecule_charge = json_data["Charge"]
    molecule_mult = json_data["Multiplicity"]
    pointgroup_symmetry = json_data["PointGroup"]

    # MO coefficients (used for 1-elec integrals)
    mos = json_data["MolecularOrbitals"]["MOs"]
    mo_coeff = np.array([m["MOCoefficients"] for m in mos])

    # Electrons
    occupations = np.array([m["Occupancy"] for m in json_data["MolecularOrbitals"]["MOs"]])
    print("Occupations:", occupations)
    mo_energies = np.array([m["OrbitalEnergy"] for m in json_data["MolecularOrbitals"]["MOs"]])
    norbitals = len(occupations)
    print("Total num orbitals:", norbitals)
    num_occ_orbs = len(np.nonzero(occupations)[0])
    print("Total num ccupied orbitals:", num_occ_orbs)
    nelectrons= int(round(sum(occupations))) #Rounding up to deal with possible non-integer occupations
    print("Number of (active) electrons:", nelectrons)
    from ash.functions.functions_elstructure import check_occupations
    WF_assignment = check_occupations(occupations)
    print("WF_assignment:", WF_assignment)

    # Orbital symmetries
    #symm_num_to_label_dict ={} #Dict
    orbital_symmetries = np.array([m["OrbitalSymLabel"] for m in json_data["MolecularOrbitals"]["MOs"]])

    # 1-electron integrals
    H = np.array(json_data["H-Matrix"])
    print("H:", H)
    # 1-elec
    from functools import reduce
    one_el = reduce(np.dot, (mo_coeff.T, H, mo_coeff))
    print("one_el:", one_el)
    # 2-electron integrals
    twoint = json_data["2elIntegrals"]
    mo_COUL_aa = np.array(json_data["2elIntegrals"][f"MO_PQRS"]["alpha/alpha"])
    mo_EXCH_aa = np.array(json_data["2elIntegrals"][f"MO_PRQS"]["alpha/alpha"])

    # Creating integral tensor
    two_el_tensor=np.zeros((norbitals,norbitals,norbitals,norbitals))

    # Processing Coulomb
    for i in mo_COUL_aa:
        two_el_tensor[int(i[0]), int(i[1]), int(i[2]), int(i[3])] = i[4]
    # Processing Exchange,  NOTE: index swap because Exchange
    for j in mo_EXCH_aa:
        two_el_tensor[int(j[0]), int(j[2]), int(j[1]), int(j[3])] = j[4]


    system = System(
        nelectrons,
        norbitals,
        molecule_mult,  # PySCF mol.spin returns 2S, not S
        nfrozen,
        ndelete=ndelete,
        point_group=pointgroup_symmetry,
        orbital_symmetries = orbital_symmetries,
        charge=molecule_charge,
        nuclear_repulsion=nuclear_repulsion,
        mo_energies=mo_energies,
        mo_occupation=occupations,
    )

    # Perform AO-to-MO transformation
    print("mo_coeff:", mo_coeff)
    print("H:", H)
    e1int = np.einsum(
        "pi,pq,qj->ij", mo_coeff, H, mo_coeff, optimize=True
    )
    # put integrals into Fortran order
    print("e1int 1:", e1int)
    e1int = np.asfortranarray(e1int)
    print("e1int 2:", e1int)
    print("two_el_tensor 1:", two_el_tensor)
    e2int = np.asfortranarray(two_el_tensor)
    print("e2int 2:", e2int)
    # Check that the HF energy calculated using the integrals matches the PySCF result
    from ccpy.interfaces.pyscf_tools import get_hf_energy
    hf_energy = get_hf_energy(e1int, e2int, system, notation="physics")
    print("hf_energy:", hf_energy)
    hf_energy += nuclear_repulsion
    print("hf_energy:", hf_energy)

    system.reference_energy = hf_energy
    system.frozen_energy = calc_hf_frozen_core_energy(e1int, e2int, system)
    print("system.frozen_energy :", system.frozen_energy )
    if dump_integrals:
        dumpIntegralstoPGFiles(e1int, e2int, system)

    return system, getHamiltonian(e1int, e2int, system, normal_ordered, sorted)