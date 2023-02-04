import time

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader
import ash.modules.module_coords
import os
import glob
import numpy as np
from functools import reduce
import scipy

#PySCF Theory object.
# TODO: PE: Polarizable embedding (CPPE). Not completely active in PySCF 1.7.1. Bugfix required I think
#TODO: Add support for AVAS
#TODO: Support for creating mf object from FCIDUMP: https://pyscf.org/_modules/pyscf/tools/fcidump.html


class PySCFTheory:
    def __init__(self, printsetting=False, printlevel=2, numcores=1, 
                  scf_type=None, basis=None, functional=None, gridlevel=5, symmetry=False,
                  pe=False, potfile='', filename='pyscf', memory=3100, conv_tol=1e-8, verbose_setting=4, 
                  CC=False, CCmethod=None, CC_direct=False, frozen_core_setting='Auto',
                  CAS=False, CASSCF=False, active_space=None, CAS_nocc_a=None, CAS_nocc_b=None,
                  frozen_virtuals=None, FNO=False, FNO_thresh=None, 
                  read_chkfile_name=None, write_chkfile_name=None,
                  PyQMC=False, PyQMC_nconfig=1, PyQMC_method='DMC'):

        self.theorytype="QM"
        print_line_with_mainheader("PySCFTheory initialization")
        #Exit early if no SCF-type
        if scf_type is None:
            print("Error: You must select an scf_type, e.g. 'RHF', 'UHF', 'RKS', 'UKS'")
            ashexit()
        if basis is None:
            print("Error: You must give a basis set (basis keyword)")
            ashexit()
        if functional is not None:
            print(f"Functional keyword: {functional} chosen. DFT is on!")
            if scf_type == 'RHF':
                print("Changing RHF to RKS")
                scf_type='RKS'
            if scf_type == 'UHF':
                print("Changing UHF to UKS")
                scf_type='UKS'
        else:
            if scf_type == 'RKS' or scf_type == 'UKS':
                print("Error: RKS/UKS chosen but no functional. Exiting")
                ashexit()
        if CC is True and CCmethod is None:
            print("Error: Need to choose CCmethod, e.g. 'CCSD', 'CCSD(T)")
            ashexit()
        if CASSCF is True and CAS is False:
            CAS=True
        #Printlevel
        self.printlevel=printlevel
        self.memory=memory
        self.filename=filename
        self.printsetting=printsetting
        #CPPE Polarizable Embedding options
        self.pe=pe
        #Potfile from user or passed on via QM/MM Theory object ?
        self.potfile=potfile

        #
        self.scf_type=scf_type
        self.basis=basis
        self.functional=functional
        self.CC=CC
        self.CCmethod=CCmethod
        self.CC_direct=CC_direct
        self.FNO=FNO
        self.FNO_thresh=FNO_thresh
        self.frozen_core_setting=frozen_core_setting
        self.frozen_virtuals=frozen_virtuals
        self.gridlevel=gridlevel

        self.conv_tol=conv_tol
        self.verbose_setting=verbose_setting
        self.read_chkfile_name=read_chkfile_name
        self.write_chkfile_name=write_chkfile_name
        self.symmetry=symmetry
        #CAS
        self.CAS=CAS
        self.CASSCF=CASSCF
        self.active_space=active_space
        self.CAS_nocc_a=CAS_nocc_a
        self.CAS_nocc_b=CAS_nocc_b

        #PyQMC
        self.PyQMC=PyQMC
        self.PyQMC_nconfig=PyQMC_nconfig #integer. number of configurations in guess
        self.PyQMC_method=PyQMC_method # DMC or VMC
        if self.PyQMC is True:
            self.postSCF=True
            self.load_pyqmc()

        #Natural orbital threshold parameterds to determined active space
        # Was used by Dice. Could also be used for DMRG and other near-FCI-type methods
        #self.cas_nmin=cas_nmin
        #self.cas_nmax=cas_nmax
        #Whether job is SCF (HF/DFT) only or a post-SCF method like CC or CAS 
        self.postSCF=False
        if self.CAS is True:
            self.postSCF=True
            if self.active_space == None or len(self.active_space) != 2:
                print("active_space must be defined as a list of 2 numbers (M electrons in N orbitals)")
                ashexit()
            #print("CAS_nocc_a:", self.CAS_nocc_a)
            #print("CAS_nocc_b:", self.CAS_nocc_b)
        if self.CC is True:
            self.postSCF=True

        #Are we doing an initialSCF calculation or not
        #Generally yes, unless PyQMC, More options will come here
        if self.PyQMC is True:
            self.SCF=False
            self.postSCF=True
        else:
            self.SCF=True

        #Attempting to load pyscf
        self.load_pyscf()
        self.set_numcores(numcores)

        #PySCF scratch dir. Todo: Need to adapt
        #print("Setting PySCF scratchdir to ", os.getcwd())

        #Print the options
        print("SCF:", self.SCF)
        print("SCF-type:", self.scf_type)
        print("Post-SCF:", self.postSCF)
        print("Symmetry:", self.symmetry)
        print("conv_tol:", self.conv_tol)
        print("Grid level:", self.gridlevel)
        print("verbose_setting:", self.verbose_setting)
        print("Basis:", self.basis)
        print("Functional:", self.functional)
        print("Coupled cluster:", self.CC)
        print("CC method:", self.CCmethod)
        print("CC direct:", self.CC_direct)
        print("FNO-CC:", self.FNO)
        print("FNO_thresh:", self.FNO_thresh)

        print("Frozen_core_setting:", self.frozen_core_setting)
        print("Frozen_virtual orbitals:",self.frozen_virtuals)
        print("Polarizable embedding:", self.pe)
        print()
        print("CAS:", self.CAS)
        print("CASSCF:", self.CASSCF)
        print("PyQMC:", self.PyQMC)
        print()
        #TODO: Restart settings for PySCF

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
        #Create list of frozen orbital indices
        self.frozen_core_orbital_indices=[i for i in range(0,self.frozen_core_orbs)]
        print("List of frozen orbital indices:", self.frozen_core_orbital_indices)


    def load_pyscf(self):
        try:
            import pyscf
        except:
            print(BC.FAIL, "Problem importing pyscf. Make sure pyscf has been installed: pip install pyscf", BC.END)
            ashexit(code=9)
        self.pyscf=pyscf
        print("\nPySCF version:", self.pyscf.__version__)
        from pyscf.tools import molden
        self.pyscf_molden=molden
        #Always importing MP2 for convenience
        from pyscf.mp.dfump2_native import DFMP2
        self.pyscf_dmp2=DFMP2
        #And mcsdcf (natural orbitals creation)
        from pyscf import mcscf
        self.mcscf=mcscf
        #And CC
        from pyscf import cc
        self.pyscf_cc=cc
        from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
        from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
        self.ccsd_t_lambda=ccsd_t_lambda
        self.ccsd_t_rdm=ccsd_t_rdm
        #TODO: Needs to be revisited
        if self.pe==True:
            #import pyscf.solvent as solvent
            #from pyscf.solvent import pol_embed
            import cppe
    def load_pyqmc(self):
        try:
            import pyqmc.api as pyq
            self.pyqmc=pyq
        except:
            print(BC.FAIL, "Problem importing pyqmc.api. Make sure pyqmc has been installed: pip install pyqmc", BC.END)
            ashexit(code=9)
    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
        print("Setting numcores to: ", self.numcores)
        #Controlling OpenMP parallelization.
        self.pyscf.lib.num_threads(numcores)
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old PySCF files")
        try:
            os.remove('timer.dat')
            os.remove(self.filename+'.dat')
        except:
            pass
    def write_orbitals_to_Moldenfile(self,mol, orbitals, occupations, label="orbs"):
        print("Writing orbitals to disk as Molden file")
        self.pyscf_molden.from_mo(mol, f'pyscf_{label}.molden', orbitals, occ=occupations)

    #Deprecated
    #def calculate_MP2_natural_orbitals(self,mol, mf):
        
        # MP2 natural occupation numbers and natural orbitals
    #    natocc, natorb = self.pyscf_dmp2(mf.to_uhf()).make_natorbs()
    #    print("MP2 natural orbital occupations:", natocc)
        #Writing to disk as Molden file
    #    self.write_orbitals_to_Moldenfile(mol, natorb,natocc, label="MP2nat")

        #Choosing MO-coeffients to be
    #    if self.scf_type == 'RHF' or self.scf_type == 'RKS':
    #        mo_coefficients=natorb              
    #    else:
    #        mo_coefficients=[natorb,natorb]
    #    return natocc, mo_coefficients
    def calculate_natural_orbitals(self,mol, mf, method='CCSD'):
        #ALTERNATIVE: https://github.com/pyscf/pyscf/issues/466
        #https://github.com/pyscf/pyscf/blob/7f4f66b37337c5c3a9c2ff94de44861266394032/pyscf/mcscf/test/test_addons.py

        if method =='MP2':
            # MP2 natural occupation numbers and natural orbitals
            #natocc, natorb = self.pyscf_dmp2(mf.to_uhf()).make_natorbs() Old
            mp2 = self.pyscf.mp.MP2(mf).run()
            natocc, natorb = self.mcscf.addons.make_natural_orbitals(mp2)
        elif method =='FCI':
        #TODO: FCI https://github.com/pyscf/pyscf/blob/master/examples/fci/14-density_matrix.py
        # FCI solver
        cisolver = self.pyscf.fci.FCI(mol, myhf.mo_coeff)
        e, fcivec = cisolver.kernel()
        # Spin-traced 1-particle density matrix
        norb = myhf.mo_coeff.shape[1]
        # 6 alpha electrons, 4 beta electrons because spin = nelec_a-nelec_b = 2
        nelec_a = 6
        nelec_b = 4
        dm1 = cisolver.make_rdm1(fcivec, norb, (nelec_a,nelec_b))

        elif method == 'CCSD':
            ccsd = self.pyscf_cc.CCSD(mf)
            print("Running CCSD")
            ccsd.run()
            natocc, natorb = self.mcscf.addons.make_natural_orbitals(ccsd)
        elif method == 'CCSD(T)':
            #No CCSD(T) object in pyscf. So manual approach.
            mycc = self.pyscf_cc.CCSD(mf).run()
            eris = mycc.ao2mo()
            #Make rdm for ccsd(t)
            conv, l1, l2 = self.ccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2)
            rdm1 = self.ccsd_t_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2, eris=eris, ao_repr=True)
            S = mf.get_ovlp()
            # Slight difference for restricted vs. unrestriced case
            if isinstance(rdm1, tuple):
                Dm = rdm1[0]+rdm1[1]
            elif isinstance(rdm1, np.ndarray):
                if np.ndim(rdm1) == 3:
                    Dm = rdm1[0]+rdm1[1]
                elif np.ndim(rdm1) == 2:
                    Dm = rdm1
                else:
                    raise ValueError(
                        "rdm1 passed to is a numpy array," +
                        "but it has the wrong number of dimensions: {}".format(np.ndim(rdm1)))
            else:
                raise ValueError(
                    "\n\tThe rdm1 generated by method_obj.make_rdm1() was a {}."
                    "\n\tThis type is not supported, please select a different method and/or "
                    "open an issue at https://github.com/pyscf/pyscf/issues".format(type(rdm1))
                )
            # Diagonalize the DM in AO (using Eqn. (1) referenced above)
            A = reduce(np.dot, (S, Dm, S))
            w, v = scipy.linalg.eigh(A, b=S)
            # Flip NOONs (and NOs) since they're in increasing order
            natocc = np.flip(w)
            natorb = np.flip(v, axis=1)

        print(f"{method} natural orbital occupations:", natocc)
        #Choosing MO-coeffients to be
        if self.scf_type == 'RHF' or self.scf_type == 'RKS':
            mo_coefficients=natorb              
        else:
            mo_coefficients=[natorb,natorb]
        return natocc, mo_coefficients

    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None ):

        module_init_time=time.time()

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING PYSCF INTERFACE-------------", BC.END)

        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for PYSCFTheory.run method", BC.END)
            ashexit()

        #If pe and potfile given as run argument
        if pe is not False:
            self.pe=pe
        if potfile is not None:
            self.potfile=potfile

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

        #MOL OBJECT
        #Defining pyscf mol object and populating 
        self.mol = self.pyscf.gto.Mole()
        #Mol system printing. Hardcoding to 3 as otherwise too much PySCF printing
        self.mol.verbose = 3
        coords_string=ash.modules.module_coords.create_coords_string(qm_elems,current_coords)
        self.mol.atom = coords_string
        self.mol.symmetry = self.symmetry
        self.mol.charge = charge; self.mol.spin = mult-1
        #PYSCF basis object: https://sunqm.github.io/pyscf/tutorial.html
        #Object can be string ('def2-SVP') or a dict with element-specific keys and values
        self.mol.basis=self.basis
        #Memory settings
        self.mol.max_memory = self.memory
        #BUILD mol object
        self.mol.build()
        ###########

        #Polarizable embedding option
        if self.pe==True:
            print(BC.OKGREEN, "Polarizable Embedding Option On! Using CPPE module inside PySCF", BC.END)
            print(BC.WARNING, "Potfile: ", self.potfile, BC.END)
            try:
                if os.path.exists(self.potfile):
                    pass
                else:
                    print(BC.FAIL, "Potfile: ", self.potfile, "does not exist!", BC.END)
                    ashexit()
            except:
                ashexit()
            # TODO: Adapt to RKS vs. UKS etc.
            self.mf = self.pyscf.solvent.PE(self.pyscf.scf.RKS(self.mol), self.potfile)
        #Regular job
        else:
            if PC is True:
                # QM/MM pointcharge embedding
                #mf = mm_charge(dft.RKS(mol), [(0.5, 0.6, 0.8)], MMcharges)
                if self.scf_type == 'RKS':
                    self.mf = self.pyscf.qmmm.mm_charge(dft.RKS(self.mol), current_MM_coords, MMcharges)
                else:
                    print("Error. scf_type other than RKS and PC True not ready")
                    ashexit()
            else:
                if self.scf_type == 'RKS':
                    self.mf = self.pyscf.scf.RKS(self.mol)
                elif self.scf_type == 'UKS':
                    self.mf = self.pyscf.scf.UKS(self.mol)
                elif self.scf_type == 'RHF':
                    self.mf = self.pyscf.scf.RHF(self.mol)
                elif self.scf_type == 'UHF':
                    self.mf = self.pyscf.scf.UHF(self.mol)

        #Printing settings.
        if self.printsetting==True:
            print("Printsetting = True. Printing output to stdout...")
            #np.set_printoptions(linewidth=500) TODO: not sure
        else:
            print("Printsetting = False. Printing to:", self.filename )
            self.mf.stdout = open(self.filename+'.out', 'w')

        #DFT
        if self.functional is not None:
            #Setting functional
            self.mf.xc = self.functional
            #TODO: libxc vs. xcfun interface control here
            #mf._numint.libxc = xcfun

            #Grid setting
            self.mf.grids.level = self.gridlevel

        self.mf.conv_tol = self.conv_tol
        #Control printing here. TOdo: make variable
        self.mf.verbose = self.verbose_setting

        #FROZEN ORBITALS in CC
        if self.CC:
            #Frozen-core settings
            if self.frozen_core_setting == 'Auto':
                self.determine_frozen_core(elems)
            elif self.frozen_core_setting == None or self.frozen_core_setting == 'None':
                print("Warning: No core-orbitals will be frozen in the CC calculation.")
                self.frozen_core_orbital_indices=None
            else:
                print("Manual user frozen core:", frozen_orbs)
                self.frozen_core_orbital_indices=self.frozen_core_setting
            #Optional frozen virtuals also
            if self.frozen_virtuals is not None:
                print(f"Frozen virtuals option active. Will freeze orbitals {self.frozen_virtuals}.")
                self.frozen_orbital_indices = self.frozen_core_orbital_indices + self.frozen_virtuals
            else:
                self.frozen_orbital_indices=self.frozen_core_orbital_indices
            print("Final frozen-orbital list (core and virtuals):", self.frozen_orbital_indices)
        
        
        ##############################
        #RUNNING
        ##############################
        print()
        #####################
        #SCF STEP
        #####################
        if self.SCF is True:
            print("Running SCF")
            self.mf.verbose=self.verbose_setting+1
            if self.write_chkfile_name != None:
                self.mf.chkfile = self.write_chkfile_name
            else:
                self.mf.chkfile = "scf.chk"
            print("Will write checkpointfile:", self.mf.chkfile )

            #SCF from chkpointfile orbitals if specfied
            if self.read_chkfile_name != None:
                print("Will read guess orbitals from checkpointfile:", self.read_chkfile_name)
                #self.mf.chkfile = self.read_chkfile_name
                #self.mf.init_guess = 'chk'
                #dm = self.mf.from_chk(self.mol, self.read_chkfile_name)
                #e_tot, e_cas, fcivec, mo, mo_energy = casscf.kernel(prevmos)
                #scf_result = self.mf.run()
                self.mf.__dict__.update(self.pyscf.scf.chkfile.load(self.read_chkfile_name, 'scf'))
                dm = self.mf.make_rdm1()
                scf_result = self.mf.run(dm)
                
            else:
                print("Starting SCF from default guess orbitals")
                #SCF starting from default guess orbitals
                scf_result = self.mf.run()


            
            print("SCF energy:", scf_result.e_tot)
            print("SCF energy components:", scf_result.scf_summary)
            if self.scf_type == 'RHF' or self.scf_type == 'RKS':
                num_scf_orbitals_alpha=len(scf_result.mo_occ)
                print("Total num. orbitals:", num_scf_orbitals_alpha)
            else:
                num_scf_orbitals_alpha=len(scf_result.mo_occ[0])
                print("Total num. orbitals:", num_scf_orbitals_alpha)

            #Get SCFenergy as initial total energy
            self.energy = scf_result.e_tot
        #####################
        #COUPLED CLUSTER
        #####################
        if self.CC is True:
            print("Coupled cluster is on !")
            #Default MO coefficients None (unless MP2natorbs option below)
            mo_coefficients=None

            #Optional MP2 natural orbitals
            if self.FNO is True:
                print("FNO is True")
                print("MP2 natural orbitals on!")
                print("Will calculate MP2 natural orbitals to use as input in CC job")
                natocc, mo_coefficients = self.calculate_MP2_natural_orbitals(self.mol,self.mf)

                #Optional natorb truncation if FNO_thresh is chosen
                if self.FNO_thresh is not None:
                    print("FNO thresh option chosen:", self.FNO_thresh)
                    num_small_virtorbs=len([i for i in natocc if i < self.FNO_thresh])
                    print("Num. virtual orbitals below threshold:", num_small_virtorbs)
                    #List of frozen orbitals
                    virt_frozen= [num_scf_orbitals_alpha-i for i in range(1,num_small_virtorbs+1)][::-1]
                    print("List of frozen virtuals:", virt_frozen)            
                    print("List of frozen virtuals:", virt_frozen)            
                    self.frozen_orbital_indices = self.frozen_orbital_indices + virt_frozen
        
            #CCSD-part as RCCSD or UCCSD
            print()
            print("All frozen_orbital_indices:", self.frozen_orbital_indices)
            print("Total number of frozen orbitals:", len(self.frozen_orbital_indices))
            print("Total number of orbitals:", num_scf_orbitals_alpha)
            print("Number of active orbitals:", num_scf_orbitals_alpha-len(self.frozen_orbital_indices))
            print()
            print("Now starting CCSD calculation")
            if self.scf_type == "RHF":
                cc = self.pyscf_cc.CCSD(self.mf, self.frozen_orbital_indices,mo_coeff=mo_coefficients)
            elif self.scf_type == "UHF":
                cc = self.pyscf_cc.UCCSD(self.mf,self.frozen_orbital_indices,mo_coeff=mo_coefficients)
                
            elif self.scf_type == "RKS":
                print("Warning: CCSD on top of RKS determinant")
                cc = self.pyscf_cc.CCSD(self.mf.to_rhf(), self.frozen_orbital_indices,mo_coeff=mo_coefficients)
            elif self.scf_type == "UKS":
                print("Warning: CCSD on top of UKS determinant")
                cc = self.pyscf_cc.UCCSD(self.mf.to_uhf(),self.frozen_orbital_indices,mo_coeff=mo_coefficients)
            
            #Switch to integral-direct CC if user-requested
            #NOTE: Faster but only possible for small/medium systems
            cc.direct = self.CC_direct
            
            result = cc.run()
            print("Reference energy:", result.e_hf)
            #CCSD energy
            self.energy = result.e_tot
            #(T) part
            if self.CCmethod == 'CCSD(T)':
                print("Calculating triples ")
                et = cc.ccsd_t()
                print("Triples energy:", et)
                self.energy = result.e_tot + et
                print("Final CCSD(T) energy:", self.energy)
        #####################
        #PyQMC
        #####################
        elif self.PyQMC is True:
            print("PyQMC is on!")
            configs = self.pyqmc.initial_guess(self.mol,self.PyQMC_nconfig)
            wf, to_opt = self.pyqmc.generate_wf(self.mol,self.mf)
            pgrad_acc = self.pyqmc.gradient_generator(self.mol,wf, to_opt)
            wf, optimization_data = self.pyqmc.line_minimization(wf, configs, pgrad_acc)
            #DMC, untested
            if self.PyQMC_method == 'DMC':
                configs, dmc_data = self.pyqmc.rundmc(wf, configs)
            #VMC. untested
            elif self.PyQMC_method == 'VMC':
                #More options possible
                df, configs = vmc(wf,configs)
        #####################
        #CAS-CI and CASSCF
        #####################

        elif self.CAS is True:
            print("CAS run is on!")
            #First run SCF
            scf_result = self.mf.run()
            print(f"Now running CAS job with active space of {self.active_space[0]} electrons in {self.active_space[1]} orbitals")
            
            #Initial orbitals for CAS-CI or CASSCF
            #Default MP2 natural orbitals unless we read-in chkfile
            if self.read_chkfile_name == None:
                print("Will calculate MP2 natural orbitals to use as input in CAS job")
                natocc, natorbs = self.calculate_MP2_natural_orbitals(self.mol,self.mf)
            else:
                print("read_chkfile_name option was specified")
                print("This means that SCF-orbitals are ignored and we will read MO coefficients from file:", self.read_chkfile_name)
            
            if self.CASSCF is True:
                print("Doing CASSCF (orbital optimization)")
                #TODO: Orbital option for starting CASSCF calculation
                casscf = self.mcscf.CASSCF(self.mf, self.active_space[1], self.active_space[0])
                casscf.verbose=self.verbose_setting
                if self.write_chkfile_name != None:
                    casscf.chkfile = self.write_chkfile_name
                else:
                    casscf.chkfile = "casscf.chk"
                print("Will write checkpointfile:", casscf.chkfile )

                #CASSCF from chkpointfile orbitals if specfied
                if self.read_chkfile_name != None:
                    prevmos = self.pyscf.lib.chkfile.load(self.read_chkfile_name, 'mcscf/mo_coeff')
                    e_tot, e_cas, fcivec, mo, mo_energy = casscf.kernel(prevmos)
                else:
                    #CASSCF starting from MP2 natural orbitals
                    e_tot, e_cas, fcivec, mo, mo_energy = casscf.kernel(natorbs)

                print("CASSCF run done")
            else:
                print("Doing CAS-CI (no orbital optimization)")
                casci = self.mcscf.CASCI(self.mf, self.active_space[1], self.active_space[0])
                casci.verbose=self.verbose_setting
                #CAS-CI from chkpointfile orbitals if specfied
                if self.read_chkfile_name != None:
                    prevmos = self.pyscf.lib.chkfile.load(self.read_chkfile_name, 'mcscf/mo_coeff')
                    e_tot, e_cas, fcivec, mo, mo_energy = casci.kernel(prevmos)
                else:
                    #CAS-CI starting from MP2 natural orbitals
                    e_tot, e_cas, fcivec, mo, mo_energy = casci.kernel(natorbs)

                print("CAS-CI run done")
                print("")

            print("e_tot:", e_tot)
            print("e_cas:", e_cas)
            self.energy = e_tot

            #################
            # for ipie
            #################
            #print("CAS_nocc_a:", self.CAS_nocc_a)
            #print("CAS_nocc_b:", self.CAS_nocc_b)
            #Write to checkpoint file
            #coeff, occa, occb = zip(*self.pyscf.fci.addons.large_ci(fcivec, self.active_space[0], (self.CAS_nocc_a, self.CAS_nocc_a), 
            #    tol=1e-8, return_strs=False))
            # Need to write wavefunction to checkpoint file.
            #import h5py
            #with h5py.File("scf.chk", 'r+') as fh5:
            #    fh5['mcscf/ci_coeffs'] = coeff
            #    fh5['mcscf/occs_alpha'] = occa
            #    fh5['mcscf/occs_beta'] = occb
        else:
            print("No post-SCF job.")
        
        ##############
        #GRADIENT
        ##############
        #NOTE: only SCF supported for now
        if Grad==True:
            print("Gradient requested")
            if self.postSCF is True:
                print("Gradient for postSCF methods is not implemented in ASH interface")
                ashexit()
            if PC is True:
                print("Gradient with PC is not quite ready")
                print("Units need to be checked.")
                ashexit()
                hfg = mm_charge_grad(grad.dft.RKS(self.mf), current_MM_coords, MMcharges)
                #                grad = self.mf.nuc_grad_method()
                self.gradient = hfg.kernel()
            else:
                print("Calculating regular SCF gradient")
                #Doing regular SCF gradeitn
                grad = self.mf.nuc_grad_method()
                self.gradient = grad.kernel()
                print("Gradient calculation done")


        #TODO: write in error handling here
        print()
        print(BC.OKBLUE, BC.BOLD, "------------ENDING PYSCF INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point PySCF energy:", self.energy)
            print_time_rel(module_init_time, modulename='pySCF run', moduleindex=2)
            return self.energy, self.gradient
        else:
            print("Single-point PySCF energy:", self.energy)
            print_time_rel(module_init_time, modulename='pySCF run', moduleindex=2)
            return self.energy

