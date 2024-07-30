import time
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
from ash.functions.functions_parallel import check_OpenMPI

# Interface to ccpy: https://github.com/piecuch-group/ccpy
# Coupled cluster package in python

#TODO: GAMESS option once GAMESS interface available

class ccpyTheory:
    def __init__(self, pyscftheoryobject=None, fcidumpfile=None, filename=None, printlevel=2, label="ccpy",
                moreadfile=None, initial_orbitals='MP2', memory=20000, frozencore=True, cc_tol=1e-8, numcores=1,
                cc_maxiter=300, cc_amp_convergence=1e-7, nact_occupied=None, nact_unoccupied=None, civecs_file=None, 
                method=None, percentages=None, states=None, roots_per_irrep=None):

        self.theorynamelabel="ccpy"
        self.theorytype="QM"
        self.analytic_hessian=False

        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        # Check for PySCFTheory object
        if pyscftheoryobject is None and fcidumpfile is None:
            print("Error: No pyscftheoryobject was provided and fcidumpfile is none. Either option is required")
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
        self.fcidumpfile=fcidumpfile

        self.moreadfile=moreadfile
        self.frozencore=frozencore
        self.memory=memory #Memory in MB (total) assigned to PySCF mcscf object
        self.initial_orbitals=initial_orbitals #Initial orbitals to be used (unless moreadfile option)
        
        #ccpy options
        self.cc_tol=cc_tol
        self.cc_maxiter=cc_maxiter #Maximum number of iterations for CC calculation
        self.cc_amp_convergence=cc_amp_convergence

        # Adaptive CC(P;Q)
        self.adaptive=False #Initially set to False
        self.percentages=percentages #What triples percentages to loop through

        # Active space CC methods
        self.nact_occupied=nact_occupied
        self.nact_unoccupied=nact_unoccupied

        # CIPSI-driven methods require civecs_file (file containing CI vectors)
        self.civecs_file=civecs_file

        # EOM 
        self.states=states #List of states: states=[1, 2, 3, 4, 5, 6, 7]
        if roots_per_irrep is None:
            self.roots_per_irrep={}
        else:
            self.roots_per_irrep=roots_per_irrep

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
        self.cipsi_methods=["eccc23", "eccc24", ""]
        # EOM methods
        self.eom_methods=["eomccsd", "eomcc3", "eomccsdt", "eomccsdt(a)_star", "creom23", "ipeom2", "ipeom3",
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
            print("Setting percentages list to: [1.0]")
            self.percentages=[1.0]
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
        print("PySCF object:", self.pyscftheoryobject)
        print("FCIDUMP file:", self.fcidumpfile)
        print("Num cores:", self.numcores)
        print("Memory (MB)", self.memory)
        print("Frozencore:", self.frozencore)
        print("moreadfile:", self.moreadfile)
        print("Initial orbitals:", self.initial_orbitals)
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
        # import ccpy Driver
        from ccpy.drivers.driver import Driver

        # OPTION 1: DRIVER via FCIDUMP meanfield
        if self.fcidumpfile is not None:
            driver = Driver.from_fcidump(self.fcidumpfile, nfrozen=self.frozen_core_orbs, charge=charge, 
                                         rohf_canonicalization="Roothaan")

        # OPTION 2: DRIVER via pyscftheoryobject
        # Run PySCF to get integrals and MOs. This would probably only be an SCF
        elif self.pyscftheoryobject is not None:
            self.pyscftheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)

            print("self.pyscftheoryobject.mf:", self.pyscftheoryobject.mf)
            # Check symmetry in pyscf mol object
            if self.pyscftheoryobject.mol.symmetry is None:
                self.pyscftheoryobject.mol.symmetry='C1'

            driver = Driver.from_pyscf(self.pyscftheoryobject.mf, nfrozen=self.frozen_core_orbs)

        # Some DRIVER settings
        driver.options["maximum_iterations"] = self.cc_maxiter
        driver.options["energy_convergence"] = self.cc_tol
        driver.options["amp_convergence"] = self.cc_amp_convergence
        driver.options["maximum_iterations"] = self.cc_maxiter

        #Print driver info
        driver.system.print_info()

        # Set active space in driver before if required
        if self.method in self.activespace_methods:
            driver.set_active_space(nact_occupied=self.nact_occupied, 
                                    nact_unoccupied=self.nact_unoccupied)
        # CIPSI-driven CC requires T3 and T4 excitations from CI-vectors file
        if self.method in self.cipsi_methods:
            if self.method == "eccc23":
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
        if self.adaptive is True:

            print("Adaptive CC(P;Q) calculation.")
            print("self.percentages:", self.percentages)
            from ccpy.drivers.adaptive import AdaptDriver
            adaptdriver = AdaptDriver(
                    driver, percentage=self.percentages)
            adaptdriver.options["energy_tolerance"]= self.cc_tol
            adaptdriver.options["two_body_approx"] = True
            adaptdriver.run()

            print("CC(P) Energies:", adaptdriver.ccp_energy)
            print("CC(P;Q) Energies:", adaptdriver.ccpq_energy)
            self.CC_PQ_energies=adaptdriver.ccpq_energy
            self.energy = float(adaptdriver.ccpq_energy[-1])

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
                driver.run_cc(method="ccsd")
                CCSD_corr_energy=driver.correlation_energy
                driver.run_ccp3(method="ccsd(t)")
                HOC_energy = driver.deltap3[0]["A"]
                total_corr_energy = CCSD_corr_energy + HOC_energy
            # Active-space CC methods
            elif self.method == "ccsdt1":
                driver.run_cc(method=self.method)
                total_corr_energy=driver.correlation_energy
            elif self.method == "ccsdt_p":
                from ccpy import Driver, get_active_triples_space
                # AKA: CCSDTp
                # Obtain the list of triples excitations corresponding to the CCSDt truncation (ground-state symmetry adapted)
                t3_excitations = get_active_triples_space(driver.system, 
                                                          target_irrep=driver.system.reference_symmetry)

                # Run active-space CCSDt calculation via general CC(P) solver
                driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
                print("driver dict:", driver.__dict__)
            elif self.method == "cct3":
                driver.run_cc(method="ccsdt1")
                driver.run_hbar(method="ccsd")
                driver.run_leftcc(method="left_ccsd")
                driver.run_ccp3(method="cct3")
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
            elif self.method in self.cipsi_methods:
                print("CIPSI-driven CC chosen")
                driver.run_eccc(method="eccc2", ci_vectors_file=self.civecs_file)
                driver.run_hbar(method="ccsd")
                CCSD_corr_energy=driver.correlation_energy
                driver.run_leftcc(method="left_ccsd")
                driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations)

                if self.method == "eccc24":
                    driver.run_ccp4(method="ccp4", state_index=0, t4_excitations=t4_excitations)
                    HOC_energy = driver.deltap3[0]["D"] + driver.deltap4[0]["D"]
                else:
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
                run_GS=True
                # EOM-CCSDT
                if 'ccsdt' in self.method:
                    gsmethod="ccsdt"
                    hbarmethod="ccsdt"
                    guessmethod="cis" #CIS is usually the guess
                    eomrunmethod="eomccsd"
                # IP-EOM
                elif 'ip' in self.method:
                    # IP-EOMCCSD(2h-1p) 
                    # IP-EOMCCSD(3h-2p)
                    # IP-EOMCCSDT(a)*
                    gsmethod="ccsd"
                    guessmethod="ipcisd" #CIS is usually the guess

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
                    guessmethod="cisd"
                    eomrunmethod="eomccsd"
                    left_method="left_ccsd"
                    ccp3_method="crcc23"
                # EOM-CCSDT(a)*
                elif 'eomccsdt(a)_star' in self.method:
                    gsmethod="ccsd"
                    hbarmethod="ccsdta"
                    guessmethod="cisd"
                    eomrunmethod="eomccsd" #correct
                    left_method="left_ccsd"
                    ccp3_method="eomccsdta_star"
                # EOM-CCSD
                elif 'ccsd' in self.method:
                    gsmethod="ccsd"
                    hbarmethod="ccsd"
                    guessmethod="cis" #CIS is usually the guess
                    eomrunmethod="eomccsd"
                # EOM-CC3
                elif 'cc3' in self.method:
                    gsmethod="cc3"
                    hbarmethod="cc3"
                    guessmethod="cisd"
                    eomrunmethod="eomcc3"

                # Run Ground-state CC and HBar if required
                if run_GS:
                    driver.run_cc(method=gsmethod)
                    driver.run_hbar(method=hbarmethod)
                # Run Guess
                driver.run_guess(method=guessmethod, multiplicity=mult, 
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
            print("driver dict:", driver.__dict__)
            print()
            print(f"Reference energy {reference_energy} Eh")
            print(f"Total correlation energy ({self.method}) {total_corr_energy} Eh")
            # Print special contributions if defined
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
