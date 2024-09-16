import time
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
from ash.functions.functions_parallel import check_OpenMPI
from ash.interfaces.interface_ORCA import read_ORCA_json_file, read_ORCA_bson_file,read_ORCA_msgpack_file
import numpy as np
import os

# Interface to ccpy: https://github.com/piecuch-group/ccpy
# Coupled cluster package in python

# TODO: GAMESS option once GAMESS interface available
# TODO: Change pyscftheoryobject and orcatheoryobject to theory= ?

class ccpyTheory:
    def __init__(self, pyscftheoryobject=None, orcatheoryobject=None, orca_gbwfile=None, orca_jsonformat="msgpack",
                 fcidumpfile=None, filename=None, printlevel=2, label="ccpy", delete_json=True,
                frozencore=True, cc_tol=1e-8, numcores=1, dump_integrals=False,
                cc_maxiter=300, cc_amp_convergence=1e-7, nact_occupied=None, nact_unoccupied=None, civecs_file=None, 
                method=None, percentages=None, states=None, roots_per_irrep=None, EOM_guess_symmetry=False,
                two_body_approx=False, parallelization="OMP-and_MKL"):

        self.theorynamelabel="ccpy"
        self.theorytype="QM"
        self.analytic_hessian=False

        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        # Check for PySCFTheory object
        if pyscftheoryobject is None and orcatheoryobject is None and fcidumpfile is None and orca_gbwfile is None:
            print("Error: No pyscftheoryobject, orcatheoryobject, orca_gbwfile or fcidumpfile was provided. One of these options is required")
            ashexit()

        # orcatheoryobject vs orca_gbwfile
        if orcatheoryobject is not None and orca_gbwfile is not None:
            print("Error: Both orcatheoryobject and orca_gbwfile provided. Only one option is allowed")
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
        self.orca_gbwfile=orca_gbwfile
        self.orca_jsonformat=orca_jsonformat #json, bson or msgpack
        self.fcidumpfile=fcidumpfile
        self.orca_mo_coeff=None #Populated later

        self.frozencore=frozencore
        self.dump_integrals=dump_integrals
        self.delete_json=delete_json

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

        self.parallelization=parallelization
        print("Setting up parallelization (options are: 'MKL', 'OMP' 'OMP-and-MKL)")
        if self.parallelization == 'MKL':
            print(f"Setting MKL threads to {self.numcores} and OMP threads to 1")
            os.environ['MKL_NUM_THREADS'] = str(self.numcores)
            os.environ['OMP_NUM_THREADS'] = str(1)
        elif self.parallelization == 'OMP':
            print(f"Setting OMP threads to {self.numcores} and MKL threads to 1")
            os.environ['MKL_NUM_THREADS'] = str(1)
            os.environ['OMP_NUM_THREADS'] = str(self.numcores) 
        elif self.parallelization == 'OMP-and-MKL':
            print(f"Setting MKL and OMP threads to {self.numcores}")
            os.environ['MKL_NUM_THREADS'] = str(self.numcores)
            os.environ['OMP_NUM_THREADS'] = str(self.numcores)
        else:
            print("Unknown parallelization method choice.")
            print("Exiting")

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
        # Main elements
        FC_elems={'H':0,'He':0,'Li':0,'Be':0,'B':2,'C':2,'N':2,'O':2,'F':2,'Ne':2,
                        'Na':2,'Mg':2,'Al':10,'Si':10,'P':10,'S':10,'Cl':10,'Ar':10,
                        'K':10,'Ca':10,'Sc':10,'Ti':10,'V':10,'Cr':10,'Mn':10,'Fe':10,
                        'Co':10,'Ni':10,'Cu':10,'Zn':10,
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

    def run_density(self, state_index=0):
        print("Now running rdm1 calc for state:", state_index)

        # Check if hbar has been run
        if self.driver.flag_hbar is False:
            print("Error: Hbar calculation has not been run")
            print("Call theory.driver.run_hbar(method=\"METHODNAME\")")
            ashexit()
        # Check if L-list is only None 
        if all(x is None for x in self.driver.L):
            print("Error: No L-vector found in driver. Cannot run RDM1 calculation")
            print("Call theory.driver.run_leftcc(method=\"left_METHODNAME\", state_index=[0])")
            ashexit()

        # Run RDM1 calc
        self.driver.run_rdm1(state_index=[state_index])

        rdm1 = self.driver.rdm1[0][0]
        rdm1a_matrix = np.concatenate( (np.concatenate( (rdm1.a.oo, rdm1.a.ov * 0.0), axis=1),
                                            np.concatenate( (rdm1.a.vo * 0.0, rdm1.a.vv), axis=1)), axis=0)
        rdm1b_matrix = np.concatenate( (np.concatenate( (rdm1.b.oo, rdm1.b.ov * 0.0), axis=1),
                                            np.concatenate( (rdm1.b.vo * 0.0, rdm1.b.vv), axis=1)), axis=0)
        rdm_matrix = rdm1a_matrix + rdm1b_matrix

        # For frozen-core case we manually add the frozen-core part to RDM
        if self.frozen_core_orbs > 0:
            print("Found frozen core, adding rdm contribution")
            nmo = self.frozen_core_orbs + rdm_matrix.shape[0]
            final_rdm_matrix=np.zeros((nmo,nmo))
            # frozen core diagonal is 2.0
            for i in range(self.frozen_core_orbs):
                final_rdm_matrix[i,i]=2.0
            final_rdm_matrix[self.frozen_core_orbs:,self.frozen_core_orbs:] = rdm_matrix
        else:
            final_rdm_matrix=rdm_matrix
        return final_rdm_matrix

    def make_natural_orbitals(self,rdm_matrix, mo_coeffs=None, get_AO_basis=True):

        # Diagonalize RDM in MO basis
        print("Diagonalizing RDM to get natural orbitals")
        natocc, natorb_MO = np.linalg.eigh(rdm_matrix)
        natocc = np.flip(natocc)
        natorb_MO = np.flip(natorb_MO, axis=1)
        print("Natural orbitals")
        print("-"*30)
        print("NO-index      Occupation")
        print("-"*30)
        for i,nocc in enumerate(natocc):
            print(f"    {i:<3d}         {nocc:7.4f}")
        print()
        print("Natural orbitals in MO basis:", natorb_MO)
        self.natorb_MO = natorb_MO
        self.natorb_AO=None

        if get_AO_basis:
            print("get_AO_basis True")
            print("Will convert NOs from MO-basis into AO-basis")
            if mo_coeffs is None:
                print("No mo_coeffs provided to make_natural_orbitals")
                print("Attempting to find some")
                if self.pyscftheoryobject is not None:
                    print("pyscftheoryobject found. Taking...")
                    mo_coeffs = self.pyscftheoryobject.mf.mo_coeff
                elif self.orca_mo_coeff is not None:
                    print("orca_mo_coeff found. Taking...")
                    mo_coeffs = self.orca_mo_coeff

            # Get the active natorbs in AO basis
            self.natorb_AO = np.dot(mo_coeffs, natorb_MO)
            print("natorb in AO", self.natorb_AO)

        return natocc, self.natorb_AO

    # Writing Molden file
    def write_molden_file(self,occupations,mo_coeffs,mo_energies=None,label="molden"):
        print("write_molden_file function")
        if self.pyscftheoryobject is not None:
            print("Pyscftheoryobject found. Using pyscf functionality")
            from pyscf.tools import molden
            print("Writing orbitals to disk as Molden file", f'{label}.molden')
            if mo_energies is None:
                print("No MO energies. Setting to 0.0")
                mo_energies = np.array([0.0 for i in occupations])

            with open(f'{label}.molden', 'w') as f1:
                molden.header(self.pyscftheoryobject.mol, f1)
                molden.orbital_coeff(self.pyscftheoryobject.mol, f1, mo_coeffs, 
                                     ene=mo_energies, occ=occupations)
            return f'{label}.molden'

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
        # OPTION 3: DRIVER via ORCA GBW file
        elif self.orca_gbwfile is not None:
            print("ORCA GBW file provided")
            # JSON-file create
            from ash.interfaces.interface_ORCA import create_ORCA_json_file
            print("Creating JSON file")
            jsonfile = create_ORCA_json_file(self.orca_gbwfile, two_el_integrals=True, format=self.orca_jsonformat)
            print_time_rel(module_init_time, modulename='create json done', moduleindex=3)
            print("Loading integrals from JSON file")
            system, hamiltonian, mo_coeff = load_orca_integrals(jsonfile, nfrozen=self.frozen_core_orbs, 
                                                      dump_integrals=self.dump_integrals)
            # Saving MO coefficients from ORCA
            self.orca_mo_coeff = mo_coeff
            print("Deleting JSON file")
            if self.delete_json is True:
                os.remove(jsonfile)
            driver = Driver(system, hamiltonian, max_number_states=50)
        # OPTION 4: DRIVER via orcatheoryobject
        # Run ORCA to get integrals and MOs. 
        elif self.orcatheoryobject is not None:
            print("ORCA object provided")
            print("Now running ORCATheory object")
            self.orcatheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)
            print_time_rel(module_init_time, modulename='orcatheory run done', moduleindex=3)
            # JSON-file create
            from ash.interfaces.interface_ORCA import create_ORCA_json_file
            print("Creating JSON file")
            jsonfile = create_ORCA_json_file(self.orcatheoryobject.gbwfile, two_el_integrals=True, format=self.orca_jsonformat)
            print_time_rel(module_init_time, modulename='create json done', moduleindex=3)
            print("Loading integrals from JSON file")
            system, hamiltonian = load_orca_integrals(jsonfile, nfrozen=self.frozen_core_orbs, 
                                                      dump_integrals=self.dump_integrals)
            print("Deleting JSON file")
            if self.delete_json is True:
                os.remove(jsonfile)
            driver = Driver(system, hamiltonian, max_number_states=50)

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

            self.driver=adaptdriver.driver
            self.adaptdriver=adaptdriver
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

        if self.adaptive is False:
            self.driver=driver
        print("ccpy is finished")

        # Cleanup scratch stuff (big files)
        self.cleanup()

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel}Theory run', moduleindex=2)
        return self.energy


# Load integrals directly from ORCA json-file
# TODO: get json rohf bug fixed
# TODO: add canonicalization ?
def load_orca_integrals(
        jsonfile, nfrozen=0, ndelete=0, convert_UHF_to_ROHF=True, 
        normal_ordered=True, dump_integrals=False, sorted=True):

    module_init_time=time.time()
    # import System
    from ccpy.models.system import System
    from ccpy.models.integrals import getHamiltonian
    from ccpy.utilities.dumping import dumpIntegralstoPGFiles
    from ccpy.energy.hf_energy import calc_hf_energy, calc_hf_frozen_core_energy

    if '.json' in jsonfile:
        json_data = read_ORCA_json_file(jsonfile)
    elif '.bson' in jsonfile:
        json_data = read_ORCA_bson_file(jsonfile)
    elif '.msgpack' in jsonfile:
        json_data = read_ORCA_msgpack_file(jsonfile)
    else:
        print(f"File {jsonfile} does not have .json or .bson ending. Unknown format")
        ashexit()
    print_time_rel(module_init_time, modulename='load_orca_integrals until after readjson complete', moduleindex=3)
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
    mo_coeff = np.transpose(mo_coeff)
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
    orbital_symmetries = [m["OrbitalSymLabel"] for m in json_data["MolecularOrbitals"]["MOs"]]

    if WF_assignment == "UHF":
        if convert_UHF_to_ROHF:
            print("convert_UHF_to_ROHF is True")
            print("Will hack UHF WF into ROHF")
            rohf_num_orbs= int(len(occupations)/2)
            norbitals=rohf_num_orbs
            alpha_occupations = occupations[0:rohf_num_orbs]
            beta_occupations = occupations[rohf_num_orbs:]
            # Hacking occupations
            new_occupations = []
            for i in range(0,rohf_num_orbs):
                if beta_occupations[i] == 1.0:
                    new_occupations.append(2.0)
                elif alpha_occupations[i] == 1.0:
                    new_occupations.append(1.0)
                else:
                    new_occupations.append(0.0)
            print("New dummy ROHF occupations:", new_occupations)
            occupations=new_occupations
            WF_assignment="ROHF"
            # Now proceeding as if were ROHF
            # Half of MO coefficients
            print("mo_coeff shape:", mo_coeff.shape)
            mo_coeff = mo_coeff[:rohf_num_orbs,:rohf_num_orbs]
            print("mo_coeff shape", mo_coeff.shape)
            orbital_symmetries = orbital_symmetries[0:rohf_num_orbs]
            mo_energies = mo_energies[0:rohf_num_orbs]

    # 1-electron integrals
    H = np.array(json_data["H-Matrix"])
    # Perform AO-to-MO transformation
    e1int = np.einsum("pi,pq,qj->ij", mo_coeff, H, mo_coeff, optimize=True)
    # put integrals in Fortran order
    e1int = np.asfortranarray(e1int)

    print_time_rel(module_init_time, modulename='load_orca_integrals: 1-el int done', moduleindex=3)
    # 2-electron integrals
    mo_COUL_aa = np.array(json_data["2elIntegrals"][f"MO_PQRS"]["alpha/alpha"])
    mo_EXCH_aa = np.array(json_data["2elIntegrals"][f"MO_PRQS"]["alpha/alpha"])
    two_el_tensor=np.zeros((norbitals,norbitals,norbitals,norbitals))

    # Processing Coulomb
    for i in mo_COUL_aa:
        p = int(i[0]); q = int(i[1]); r = int(i[2]); s = int(i[3])
        two_el_tensor[p, q, r, s] = i[4]
        two_el_tensor[p, q, s, r] = i[4]
        two_el_tensor[q, p, r, s] = i[4]
        two_el_tensor[q, p, s, r] = i[4]
        two_el_tensor[r, s, p, q] = i[4]
        two_el_tensor[s, r, p, q] = i[4]
        two_el_tensor[r, s, q, p] = i[4]
        two_el_tensor[s, r, q, p] = i[4]

    # Processing Exchange,  NOTE: index swap because Exchange
    for j in mo_EXCH_aa:
        p = int(j[0])
        r = int(j[1]) #index swap because Exchange (PRQS)
        q = int(j[2])
        s = int(j[3])
        two_el_tensor[p, q, r, s] = j[4]
        two_el_tensor[p, q, s, r] = j[4]
        two_el_tensor[q, p, r, s] = j[4]
        two_el_tensor[q, p, s, r] = j[4]
        two_el_tensor[r, s, p, q] = j[4]
        two_el_tensor[s, r, p, q] = j[4]
        two_el_tensor[r, s, q, p] = j[4]
        two_el_tensor[s, r, q, p] = j[4]

    e2int = np.transpose(two_el_tensor,(0,2,1,3))
    e2int = np.asfortranarray(e2int)
    print_time_rel(module_init_time, modulename='load_orca_integrals 2elint done', moduleindex=3)
    # Creating ccpy system
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
        mo_occupation=occupations)

    # Check that the HF energy calculated using the integrals matches the PySCF result
    from ccpy.interfaces.pyscf_tools import get_hf_energy
    hf_energy = get_hf_energy(e1int, e2int, system, notation="physics")
    print("hf_energy:", hf_energy)
    hf_energy += nuclear_repulsion
    print("hf_energy (with nuc_repuls):", hf_energy)

    system.reference_energy = hf_energy
    system.frozen_energy = calc_hf_frozen_core_energy(e1int, e2int, system)
    print("system.frozen_energy :", system.frozen_energy )
    if dump_integrals:
        print("Dumping integrals")
        dumpIntegralstoPGFiles(e1int, e2int, system)

    print_time_rel(module_init_time, modulename='load_orca_integrals', moduleindex=3)
    return system, getHamiltonian(e1int, e2int, system, normal_ordered, sorted), mo_coeff