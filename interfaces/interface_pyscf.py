import time

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader
import ash.modules.module_coords
import os
import sys
import glob
import numpy as np
from functools import reduce
import scipy
import random

#PySCF Theory object.
# TODO: PE: Polarizable embedding (CPPE). Not completely active in PySCF 1.7.1. Bugfix required I think
#TODO: Support for creating mf object from FCIDUMP: https://pyscf.org/_modules/pyscf/tools/fcidump.html
#TODO: Remove PyQMC from pyscftheory. 

class PySCFTheory:
    def __init__(self, printsetting=False, printlevel=2, numcores=1, 
                  scf_type=None, basis=None, functional=None, gridlevel=5, symmetry=False,
                  dispersion=None,
                  pe=False, potfile='', filename='pyscf', memory=3100, conv_tol=1e-8, verbose_setting=4, 
                  CC=False, CCmethod=None, CC_direct=False, frozen_core_setting='Auto', cc_maxcycle=200,
                  CAS=False, CASSCF=False, active_space=None, stability_analysis=False, casscf_maxcycle=200,
                  frozen_virtuals=None, FNO=False, FNO_thresh=None, x2c=False,
                  moreadfile=None, write_chkfile_name=None,
                  PyQMC=False, PyQMC_nconfig=1, PyQMC_method='DMC',
                  AVAS=False, DMET_CAS=False, CAS_AO_labels=None,
                  cas_nmin=None, cas_nmax=None, losc=False, loscfunctional=None, LOSC_method='postSCF',
                  loscpath=None, LOSC_window=None,
                  mcpdft=False, mcpdft_functional=None, specialrun=False):

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
        self.stability_analysis=stability_analysis
        self.basis=basis
        self.functional=functional
        self.x2c=x2c
        self.CC=CC
        self.CCmethod=CCmethod
        self.CC_direct=CC_direct
        self.cc_maxcycle=cc_maxcycle
        self.FNO=FNO
        self.FNO_thresh=FNO_thresh
        self.frozen_core_setting=frozen_core_setting
        self.frozen_virtuals=frozen_virtuals
        self.gridlevel=gridlevel

        self.conv_tol=conv_tol
        self.verbose_setting=verbose_setting
        self.moreadfile=moreadfile
        self.write_chkfile_name=write_chkfile_name
        self.symmetry=symmetry
        #CAS
        self.CAS=CAS
        self.CASSCF=CASSCF
        self.active_space=active_space
        self.casscf_maxcycle=casscf_maxcycle

        #Auto-CAS options
        self.AVAS=AVAS
        self.DMET_CAS=DMET_CAS
        self.CAS_AO_labels=CAS_AO_labels
        self.cas_nmin=cas_nmin
        self.cas_nmax=cas_nmax
        #PyQMC
        self.PyQMC=PyQMC
        self.PyQMC_nconfig=PyQMC_nconfig #integer. number of configurations in guess
        self.PyQMC_method=PyQMC_method # DMC or VMC
        if self.PyQMC is True:
            self.postSCF=True
            self.load_pyqmc()
        #LOSC
        self.losc=losc
        self.loscfunctional=loscfunctional
        self.LOSC_method=LOSC_method
        self.LOSC_window=LOSC_window
        #MC-PDFT
        self.mcpdft=mcpdft
        self.mcpdft_functional=mcpdft_functional

        #Dispersion option
        #Uses: https://github.com/ajz34/vdw
        self.dispersion=dispersion

        #Special PySCF run option
        self.specialrun=specialrun

        #Whether job is SCF (HF/DFT) only or a post-SCF method like CC or CAS 
        self.postSCF=False
        if self.CAS is True:
            self.postSCF=True
            print("CAS is True. Active_space keyword should be defined unless AVAS or DMET_CAS is True.")
            if self.AVAS is True or self.DMET_CAS is True: 
                print("AVAS/DMET_CAS is True")
                if self.CAS_AO_labels is None:
                    print("AVAS/DMET_CAS requires CAS_AO_labels keyword. Specify as e.g. CAS_AO_labels=['Fe 3d', 'Fe 4d', 'C 2pz']")
                    ashexit()
            elif self.cas_nmin != None or self.cas_nmax != None:
                print("Keyword cas_nmin and cas_nmax provided")
                print("Will use together with MP2 natural orbitals to choose CAS")
            elif self.active_space == None or len(self.active_space) != 2:
                print("No option chosen for active space.")
                print("If neither AVAS,DMET_CAS or cas_nmin/cas_nmax options are chosen then")
                print("active_space must be defined as a list of 2 numbers (M electrons in N orbitals)")
                ashexit()
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
        #self.load_pyscf()
        #self.set_numcores(numcores)
        self.numcores=numcores
        if self.losc is True:
            self.load_losc(loscpath)

        #PySCF scratch dir. Todo: Need to adapt
        #print("Setting PySCF scratchdir to ", os.getcwd())

        #Print the options
        print("SCF:", self.SCF)
        print("SCF-type:", self.scf_type)
        print("x2c:", self.x2c)
        print("Post-SCF:", self.postSCF)
        print("Symmetry:", self.symmetry)
        print("conv_tol:", self.conv_tol)
        print("Grid level:", self.gridlevel)
        print("verbose_setting:", self.verbose_setting)
        print("Basis:", self.basis)
        print("SCF stability analysis:", self.stability_analysis)
        print("Functional:", self.functional)
        print("Coupled cluster:", self.CC)
        print("CC method:", self.CCmethod)
        print("CC direct:", self.CC_direct)
        print("CC maxcycles:", self.cc_maxcycle)
        print("FNO-CC:", self.FNO)
        print("FNO_thresh:", self.FNO_thresh)

        print("Frozen_core_setting:", self.frozen_core_setting)
        print("Frozen_virtual orbitals:",self.frozen_virtuals)
        print("Polarizable embedding:", self.pe)
        print()
        print("CAS:", self.CAS)
        print("CASSCF:", self.CASSCF)
        print("CASSCF maxcycles:", self.casscf_maxcycle)
        print("AVAS:", self.AVAS)
        print("DMET_CAS:", self.DMET_CAS)
        print("CAS_AO_labels (for AVAS/DMET_CAS)", self.CAS_AO_labels)
        print("PyQMC:", self.PyQMC)
        print("CAS_nmin:", self.cas_nmin)
        print("CAS_nmax:", self.cas_nmax)
        print("Active space:", self.active_space)
        print()
        print("MC-PDFT:", self.mcpdft)
        print("mcpdft_functional:", self.mcpdft_functional)
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

    def load_losc(self,loscpath):
        #Not sure if this works. Think PYTHONPATH is necessary
        if loscpath != None:
            sys.path.insert(0, loscpath)
        try:
            import pyscf_losc
            pyscf_losc=pyscf_losc
        except ModuleNotFoundError as e:
            print("Problem importing pyscf_losc")
            print("e:", e)
            ashexit()
    #def load_pyscf(self):
    #    try:
    #        import pyscf
    #    except:
    #        print(BC.FAIL, "Problem importing pyscf. Make sure pyscf has been installed: pip install pyscf", BC.END)
    #        ashexit(code=9)
    #    pyscf=pyscf
    #    print("\nPySCF version:", pyscf.__version__)
    #    import pyscf.tools
    #    pyscf_tools=pyscf.tools
    #    from pyscf.tools import molden
    #    pyscf_molden=molden
    #    #Always importing MP2 for convenience
    #    #from pyscf.mp.dfump2_native import DFMP2
    #
    #    from pyscf.mp.dfmp2_native import DFRMP2
    #    from pyscf.mp.dfump2_native import DFUMP2
    #    pyscf_dfrmp2=DFRMP2
    #    pyscf_dfump2=DFUMP2
        #from pyscf.mp.dfmp2_native import DFRMP2
        #from pyscf.mp.dfump2_native import DFUMP2
        #And mcsdcf (natural orbitals creation)
    #    from pyscf import mcscf
    #    self.mcscf=mcscf
    #    #And CC
    #    from pyscf import cc
    #    pyscf_cc=cc
    #    from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
    #    from pyscf.cc import uccsd_t_lambda
    #    from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
    #    from pyscf.cc import uccsd_t_rdm
    #    self.ccsd_t_lambda=ccsd_t_lambda
    #    self.uccsd_t_lambda=uccsd_t_lambda
    #    self.ccsd_t_rdm=ccsd_t_rdm
    #    self.uccsd_t_rdm=uccsd_t_rdm
    #    #AVAS/DMET
    #    from pyscf.mcscf import avas, dmet_cas
    #    pyscf_avas=avas
    #    pyscf_dmet_cas=dmet_cas
    #    
    #    if self.pe==True:
    #        #TODO: Needs to be revisited
    #        #import pyscf.solvent as solvent
    #        #from pyscf.solvent import pol_embed
    #        import cppe
    #    if self.mcpdft is True:
    #        try:
    #            from pyscf import mcpdft, mcdcft
    #        except ImportError:
    #            print("Problem importing mcpdft library")
    #            print("Make sure that mcpdft has been installed into this pyscf environment")
    #            print("See: https://gagliardigroup.uchicago.edu/open-software-data/mc-pdft/")
    #            ashexit()
    #        self.mcpdft_l=mcpdft
    #        self.mcdcft_l=mcdcft

    def load_pyqmc(self):
        try:
            import pyqmc.api as pyq
            self.pyqmc=pyq
        except:
            print(BC.FAIL, "Problem importing pyqmc.api. Make sure pyqmc has been installed: pip install pyqmc", BC.END)
            ashexit(code=9)
    #Set numcores method
    #def set_numcores(self,numcores):
    #    self.numcores=numcores
    #    print("Setting numcores to: ", self.numcores)
    #    #Controlling OpenMP parallelization.
    #    pyscf.lib.num_threads(numcores)
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old PySCF files")
        try:
            os.remove('timer.dat')
            os.remove(self.filename+'.dat')
        except:
            pass
    def write_orbitals_to_Moldenfile(self,mol, mo_coeffs, occupations, mo_energies, label="orbs"):
        print("Writing orbitals to disk as Molden file")
        #pyscf_molden.from_mo(mol, f'pyscf_{label}.molden', orbitals, occ=occupations)
        with open(f'pyscf_{label}.molden', 'w') as f1:
            pyscf_molden.header(mol, f1)
            pyscf_molden.orbital_coeff(mol, f1, mo_coeffs, ene=mo_energies, occ=occupations)
    
    #Write Cube files for a list of orbital indices
    def cubegen_orbital(self, mol, name, coeffs, nx=60,ny=60,nz=60):
        pyscf_tools.cubegen.orbital(mol, name, coeffs, nx=nx, ny=ny, nz=nz)
        print("cubegen_orbital: Wrote file:", name)
    #Write Cube file for density
    def cubegen_density(self, mol, name, dm, nx=60,ny=60,nz=60):
        pyscf_tools.cubegen.density(mol, name, dm, nx=nx, ny=ny, nz=nz)
        print("cubegen_density: Wrote file:", name)
    #Write Cube file for MEP
    def cubegen_mep(self, mol, name, dm, nx=60,ny=60,nz=60):
        pyscf_tools.cubegen.mep(mol, name, dm, nx=nx, ny=ny, nz=nz)
        print("cubegen_mep: Wrote file:", name)
    #Natural orbital at the RHF/UHF MP2, CCSD, CCSD(T), AVAS-CASSCF, DMET-CASSCF levels
    #Also possible to do  KS-CC or KS-MP2
    #TODO: Calling make_natural_orbitals calculated RDM1 for MP2 and CCSD but does not store it.
    #For pop analysis we would have to calculate it again.
    #Currently doing so for MP2 (cheap-ish) for pop-ana printout but not CCSD. CCSD(T) is fine since manual
    #Make CCSD and MP2 manual also?
    #Relaxed refers to DFMP2 only 
    def calculate_natural_orbitals(self,mol, mf, method='MP2', CAS_AO_labels=None, elems=None, relaxed=False, numcores=1):
        module_init_time=time.time()
        print("Inside calculate_natural_orbitals")
        print("Number of PySCF lib threads is:", pyscf.lib.num_threads())

        #Determine frozen core from element list
        self.determine_frozen_core(elems)
        self.frozen_orbital_indices=self.frozen_core_orbital_indices

        if self.scf_type == "RKS" or self.scf_type == "UKS":
            print("Warning: SCF-type of PySCF object appears to be a Kohn-Sham determinant")
            print("Kohn-Sham functional:", self.functional)
            print("Natural orbital calculation will use KS reference orbitals")
            print("First converting RKS/UKS object to RHF/UHF object")
            #Converting KS-DFT to HF object (orbitals untouched)
            if self.scf_type == "RKS":
                mf = mf.to_rhf()
            elif self.scf_type == "UKS":
                mf = mf.to_uhf()
        if 'MP2' in method:
            print("Running MP2 natural orbital calculation")
            # MP2 natural occupation numbers and natural orbitals
            #natocc, natorb = pyscf_dmp2(mf.to_uhf()).make_natorbs() Old
            
            #Simple canonical MP2 with unrelaxed density
            #TODO: could be slow for large systems
            #TODO: Later change MP2 to re-direct to DFMP2 ?
            if method == 'MP2' or method == 'canMP2':
                mp2 = pyscf.mp.MP2(mf, frozen=self.frozen_orbital_indices)
                print("Running MP2")
                mp2.run()
                print("Making MP2 density matrix")
                mp2_dm = mp2.make_rdm1()
                if self.scf_type == "RKS" or self.scf_type == "RHF" :
                    print("Mulliken analysis for RHF-MP2 density matrix")
                    self.run_population_analysis(mf, unrestricted=False, dm=mp2_dm, type='Mulliken', label='MP2')
                else:
                    print("Mulliken analysis for UHF-MP2 density matrix")
                    self.run_population_analysis(mf, unrestricted=True, dm=mp2_dm, type='Mulliken', label='MP2')
                #TODO: Fix. Slightly silly, calling make_natural_orbitals will cause dm calculation again
                natocc, natorb = self.mcscf.addons.make_natural_orbitals(mp2)
            elif method == 'DFMP2' or method =='DFMP2relax':
                #DF-MP2 scales better but syntax differs: https://pyscf.org/user/mp.html#dfmp2
                if self.scf_type == "RKS" or self.scf_type == "RHF" :
                    unrestricted=False
                    dmp2 = pyscf_dfrmp2(mf, frozen=self.frozen_orbital_indices)
                else:
                    unrestricted=True
                    dmp2 = pyscf_dfump2(mf, frozen=(self.frozen_orbital_indices,self.frozen_orbital_indices))
                #Now run DMP2 object
                dmp2.run()
                #RDMs: Unrelaxed vs. Relaxed
                if method =='DFMP2relax':
                    relaxed=True
                if relaxed is True:
                    dfmp2_dm = dmp2.make_rdm1_relaxed(ao_repr=False) #Relaxed
                    print("Mulliken analysis for restricted DF-MP2 relaxed density matrix")
                    self.run_population_analysis(mf, unrestricted=unrestricted, dm=dfmp2_dm, type='Mulliken', label='DFMP2-relaxed') 
                else:
                    dfmp2_dm = dmp2.make_rdm1_unrelaxed(ao_repr=False) #Unrelaxed
                    print("Mulliken analysis for restricted DF-MP2 unrelaxed density matrix")
                    self.run_population_analysis(mf, unrestricted=unrestricted, dm=dfmp2_dm, type='Mulliken', label='DFMP2-unrelaxed')
                   
                #Make natorbs
                #NOTE: This should not have to recalculate RDM here since provided
                #natocc, natorb = dmp2.make_natorbs(rdm1_mo=dfmp2_dm, relaxed=relaxed)
                #NOTE: Above gives weird occupations ?
                #NOTE: Slightly silly, calling make_natural_orbitals will cause dm calculation again
                natocc, natorb = self.mcscf.addons.make_natural_orbitals(dmp2)                
            #natocc, natorb = self.mcscf.addons.make_natural_orbitals(mp2)
        elif method =='FCI':
            print("Running FCI natural orbital calculation")
            print("not ready")
            exit()
            #TODO: FCI https://github.com/pyscf/pyscf/blob/master/examples/fci/14-density_matrix.py
            # FCI solver
            cisolver = pyscf.fci.FCI(mol, myhf.mo_coeff)
            e, fcivec = cisolver.kernel()
            # Spin-traced 1-particle density matrix
            norb = myhf.mo_coeff.shape[1]
            # 6 alpha electrons, 4 beta electrons because spin = nelec_a-nelec_b = 2
            nelec_a = 6
            nelec_b = 4
            dm1 = cisolver.make_rdm1(fcivec, norb, (nelec_a,nelec_b))
        elif method == 'AVAS-CASSCF':
            print("Doing AVAS and then CASSCF to get natural orbitals")
            norb_cas, nel_cas, avasorbitals = pyscf_avas.avas(self.mf, CAS_AO_labels)
            print(f"AVAS determined an active space of: CAS({nel_cas},{norb_cas})")
            print(f"Now doing CASSCF using AVAS active space (CAS({nel_cas},{norb_cas})) and AVAS orbitals")
            casscf = self.mcscf.CASSCF(mf, norb_cas, nel_cas)
            casscf.max_cycle_macro=self.casscf_maxcycle
            casscf.verbose=self.verbose_setting
            cas_result = casscf.run(avasorbitals, natorb=True)
            print("CASSCF occupations", cas_result.mo_occ)
            return cas_result.mo_occ, cas_result.mo_coeff
        elif method == 'DMET-CASSCF':
            print("Doing DMET-CAS and then CASSCF to get natural orbitals")
            print("DMET_CAS automatic CAS option chosen")
            norb_cas, nel_cas, dmetorbitals = pyscf_dmet_cas.guess_cas(mf, mf.make_rdm1(), CAS_AO_labels)
            print(f"DMET_CAS determined an active space of: CAS({nel_cas},{norb_cas})")
            print(f"Now doing CASSCF using DMET-CAS active space (CAS({nel_cas},{norb_cas})) and DMET-CAS orbitals")
            casscf = self.mcscf.CASSCF(mf, norb_cas, nel_cas)
            casscf.max_cycle_macro=self.casscf_maxcycle
            casscf.verbose=self.verbose_setting
            cas_result = casscf.run(dmetorbitals, natorb=True)
            print("CASSCF occupations", cas_result.mo_occ)
            return cas_result.mo_occ, cas_result.mo_coeff
        elif method == 'CCSD':
            print("Running CCSD natural orbital calculation")
            ccsd = pyscf_cc.CCSD(mf, frozen=self.frozen_orbital_indices)
            ccsd.max_cycle=200
            ccsd.verbose=5
            ccsd.run()
            natocc, natorb = self.mcscf.addons.make_natural_orbitals(ccsd)
        elif method == 'CCSD(T)':
            print("Running CCSD(T) natural orbital calculation")
            #No CCSD(T) object in pyscf. So manual approach. Slower algorithms
            ccsd = pyscf_cc.CCSD(mf, frozen=self.frozen_orbital_indices)
            ccsd.max_cycle=200
            ccsd.verbose=5
            ccsd.run()
            eris = ccsd.ao2mo()
            #Make RDMs for ccsd(t) RHF and UHF
            #Note: Checking type of CCSD object because if ROHF object then was automatically converted to UHF
            # and hence UCCSD
            if type(ccsd) == pyscf.cc.uccsd.UCCSD:
                print("CCSD(T) lambda UHF")
                #NOTE: No threading parallelization seen here, not sure why
                conv, l1, l2 = self.uccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2)
                rdm1 = self.uccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris, ao_repr=True)
                print("Mulliken analysis for UHF-CCSD(T) density matrix")
                self.run_population_analysis(mf, unrestricted=True, dm=rdm1, type='Mulliken', label='CCSD(T)')
            else:
                print("CCSD(T) lambda RHF")
                conv, l1, l2 = self.ccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2)
                rdm1 = self.ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris, ao_repr=True)
                print("Mulliken analysis for RHF-CCSD(T) density matrix")
                self.run_population_analysis(mf, unrestricted=False, dm=rdm1, type='Mulliken', label='CCSD(T)')

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
        with np.printoptions(precision=5, suppress=True):
            print(f"{method} natural orbital occupations:", natocc)
        #Choosing MO-coeffients to be
        if self.scf_type == 'RHF' or self.scf_type == 'RKS':
            mo_coefficients=natorb              
        else:
            mo_coefficients=[natorb,natorb]
        print_time_rel(module_init_time, modulename='calculate_natural_orbitals', moduleindex=2)
        return natocc, mo_coefficients
    
    #Population analysis, requiring mf and dm objects
    #Currently only Mulliken
    def run_population_analysis(self, mf, unrestricted=True, dm=None, type='Mulliken', label=None, verbose=3):
        import pyscf
        print()
        if label==None:
            label=''
        if type == 'Mulliken':
            if unrestricted is False:
                if dm is None:
                    dm = mf.make_rdm1()
                mulliken_pop =pyscf.scf.rhf.mulliken_pop(self.mol,dm, verbose=verbose)
                print(f"{label} Mulliken charges:", mulliken_pop[1])
            elif unrestricted is True:
                if dm is None:
                    dm = mf.make_rdm1()
                mulliken_pop =pyscf.scf.rhf.mulliken_pop(self.mol,dm, verbose=verbose)
                mulliken_spinpop = pyscf.scf.uhf.mulliken_spin_pop(self.mol,dm, verbose=verbose)
                print(f"{label} Mulliken charges:", mulliken_pop[1])
                print(f"{label} Mulliken spin pops:", mulliken_spinpop[1])
        return

    def run_stability_analysis(self):
        def stableprint(stable_i,stable_e):
            if stable_i is True:
                print("SCF WF is internally STABLE")
            else:
                print("SCF WF is internally UNSTABLE")
            if stable_e is True:
                print("SCF WF is externally STABLE")
            else:
                print("SCF WF is externally UNSTABLE")
        module_init_time=time.time()
        if self.stability_analysis is True:
            print("Doing stability analysis (to turn off: stability_analysis=False)")
            mos_i, mos_e, stable_i, stable_e =  self.mf.stability(external=True, return_status=True, verbose=5)
            stableprint(stable_i, stable_e)
            if stable_i is False:
                print("Doing internal stability analysis loop")
                self.mf = self.stability_analysis_loop(self.mf,mos_i)
                print("Doing final stability analysis with external=True")
                mos_i, mos_e, stable_i, stable_e =  self.mf.stability(external=True, return_status=True, verbose=5)
                print("Now done with stability analysis")
                stableprint(stable_i, stable_e)
            print_time_rel(module_init_time, modulename='Stability analysis', moduleindex=2)
        else:
            if self.printlevel >1:
                print("No stability analysis requested")
        
    #Stability analysis loop for any mf object
    def stability_analysis_loop(self,mf,mos,maxcyc=10):
        #Looping until internal stability is reached
        cyc=0
        stable=False
        while (not stable and cyc < maxcyc):
            print(f'Try to optimize orbitals until stable, attempt {cyc}')
            dm1 = mf.make_rdm1(mos, mf.mo_occ)
            mf = mf.run(dm1)
            print(f"Loop {cyc}. Current SCF energy: {mf.e_tot}")
            mos, _, stable, _ = mf.stability(return_status=True)
            cyc += 1
        if not stable:
            print(f'Stability Opt failed after {cyc} attempts')
            return mf
        print("Stability analysis loop succeeded in finding stable internal solution!")
        return mf

    #General run function to distinguish  specialrun and mainrun
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None):

        if self.specialrun is True:
            #Attempt at having special run method that could be run my multiprocessing
            #Create pyscf inputscript that defines mol object in script
            #Writes things that should be run etc and then executes by launching separate process
            #Not ready
            ashexit()
            energy=random.random()
            grad=np.random.random([len(elems),3])
            return energy, grad
        else:

            return self.mainrun(current_coords=current_coords, current_MM_coords=current_MM_coords, MMcharges=MMcharges, qm_elems=qm_elems,
            elems=elems, Grad=Grad, PC=PC, numcores=numcores, pe=pe, potfile=potfile, restart=restart, label=label,
            charge=charge, mult=mult)

    #Main Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def mainrun(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None,pyscf=None ):

        module_init_time=time.time()
        if self.printlevel >0:
            print(BC.OKBLUE,BC.BOLD, "------------RUNNING PYSCF INTERFACE-------------", BC.END)

        #Load pyscf
        import pyscf
        #Set PySCF threads to numcores
        pyscf.lib.num_threads(self.numcores)
        print("Number of PySCF lib threads is:", pyscf.lib.num_threads())
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
        self.mol = pyscf.gto.Mole()
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
            self.mf = pyscf.solvent.PE(pyscf.scf.RKS(self.mol), self.potfile)
        #Regular job
        else:
            if PC is True:
                # QM/MM pointcharge embedding
                #mf = mm_charge(dft.RKS(mol), [(0.5, 0.6, 0.8)], MMcharges)
                if self.scf_type == 'RKS':
                    self.mf = pyscf.qmmm.mm_charge(pyscf.dft.RKS(self.mol), current_MM_coords, MMcharges)
                else:
                    print("Error. scf_type other than RKS and PC True not ready")
                    ashexit()
            else:
                if self.scf_type == 'RKS':
                    self.mf = pyscf.scf.RKS(self.mol)
                elif self.scf_type == 'UKS':
                    self.mf = pyscf.scf.UKS(self.mol)
                elif self.scf_type == 'RHF':
                    self.mf = pyscf.scf.RHF(self.mol)
                elif self.scf_type == 'UHF':
                    self.mf = pyscf.scf.UHF(self.mol)

        #Convert non-relativistic mf object to spin-free x2c if self.x2c is True
        if self.x2c is True:
            print("x2c is True. Changing SCF object to relativistic x2c Hamiltonian")
            self.mf = self.mf.sfx2c1e()

        #Printing settings.
        if self.printsetting==True:
            if self.printlevel >1:
                print("Printing output to stdout...")
            #np.set_printoptions(linewidth=500) TODO: not sure
        else:
            self.mf.stdout = open(self.filename+'.out', 'w')
            if self.printlevel >0:
                print(f"PySCF printing to: {self.filename}.out")

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
                print("Manual user frozen core:", self.frozen_core_setting)
                self.frozen_core_orbital_indices=self.frozen_core_setting
            #Optional frozen virtuals also
            if self.frozen_virtuals is not None:
                print(f"Frozen virtuals option active. Will freeze orbitals {self.frozen_virtuals}.")
                self.frozen_orbital_indices = self.frozen_core_orbital_indices + self.frozen_virtuals
            else:
                self.frozen_orbital_indices=self.frozen_core_orbital_indices
            print("Final frozen-orbital list (core and virtuals):", self.frozen_orbital_indices)




        ##############
        #DISPERSION
        ##############
        #Modifying self.mf object
        if self.dispersion != None:
            print("Dispersion correction is active")
            try:
                import vdw
            except ModuleNotFoundError:
                print("vdw library not found. See https://github.com/ajz34/vdw")
                print("You probably have to do: pip install pyvdw")
                ashexit()
            except ImportError as e:
                print("Import Error when importing vdw")
                print("Exception message from vdw library:", e)
                print("Note: A toml library also needs to be installed: e.g. conda install toml or pip install toml")
                ashexit()

            if self.dispersion == 'D3':
                print("D3 correction on")
                from vdw import to_dftd3
                self.mf = to_dftd3(self.mf, do_grad=Grad)
            if self.dispersion == 'D4':
                print("D4 correction on")
                from vdw import to_dftd4
                self.mf = to_dftd4(self.mf, do_grad=Grad)
            elif self.dispersion == 'TS':
                print("TS correction on")
                from vdw import to_mbd
                self.mf = to_mbd(self.mf, variant="ts", do_grad=Grad)  
            elif self.dispersion == 'MBD':
                print("MBD correction on")
                from vdw import to_mbd
                self.mf = to_mbd(self.mf, variant="rsscs", do_grad=Grad) 

        
        ##############################
        #RUNNING
        ##############################
        print()
        #####################
        #SCF STEP
        #####################
        if self.SCF is True:
            if self.printlevel >1:
                print(f"Running SCF (SCF-type: {self.scf_type})")
            self.mf.verbose=self.verbose_setting+1
            if self.write_chkfile_name != None:
                self.mf.chkfile = self.write_chkfile_name
            else:
                self.mf.chkfile = "scf.chk"
            if self.printlevel >1:
                print("Will write checkpointfile:", self.mf.chkfile )

            #SCF from chkpointfile orbitals if specfied
            if self.moreadfile != None:
                print("Will read guess orbitals from checkpointfile:", self.moreadfile)
                try:
                    print("Trying to read SCF-orbitals from checkpointfile")
                    self.mf.__dict__.update(pyscf.scf.chkfile.load(self.moreadfile, 'scf'))
                    dm = self.mf.make_rdm1()
                    scf_result = self.mf.run(dm)
                    print("SCF energy:", scf_result.e_tot)
                except TypeError:
                    print("No SCF orbitals found. Could be checkpointfile from CASSCF?")
                    print("Ignoring and continuing")
                    scf_result = self.mf.run()
                    print("SCF energy:", scf_result.e_tot)

                
            else:
                if self.printlevel >1:
                    print("Starting SCF from default guess orbitals")
                #SCF starting from default guess orbitals
                scf_result = self.mf.run()
                print("SCF energy:", scf_result.e_tot)

            #Possible stability analysis
            self.run_stability_analysis()
            print("SCF energy:", scf_result.e_tot)
            if self.printlevel >1:
                print("SCF energy components:", scf_result.scf_summary)
            if self.dispersion != None:
                print(f"Dispersion contribution: {self.mf.e_vdw} (included in SCF energy)")

            #Possible population analysis (if dm=None then taken from mf object)
            if self.scf_type == 'RHF' or self.scf_type == 'RKS':
                num_scf_orbitals_alpha=len(scf_result.mo_occ)
                if self.printlevel >1:
                    print("Total num. orbitals:", num_scf_orbitals_alpha)
                if self.printlevel >1:
                    self.run_population_analysis(self.mf, dm=None, unrestricted=False, type='Mulliken', label='SCF')
            else:
                num_scf_orbitals_alpha=len(scf_result.mo_occ[0])
                if self.printlevel >1:
                    print("Total num. orbitals:", num_scf_orbitals_alpha)
                if self.printlevel >1:
                    self.run_population_analysis(self.mf, dm=None, unrestricted=True, type='Mulliken', label='SCF')

            #Get SCFenergy as initial total energy
            self.energy = scf_result.e_tot

            #LOSC
            if self.losc is True:
                print("Now doing LOSC calculation")
                # Configure LOC calculation settings.
                pyscf_losc.options.set_param("localizer", "max_iter", 1000)
                if self.loscfunctional == 'BLYP':
                    losc_func=pyscf_losc.BLYP
                elif self.loscfunctional == 'B3LYP':
                    losc_func=pyscf_losc.B3LYP
                elif self.loscfunctional == 'PBE':
                    losc_func=pyscf_losc.PBE
                elif self.loscfunctional == 'PBE0':
                    losc_func=pyscf_losc.PBE0
                elif self.loscfunctional == 'SVWN':
                    losc_func=pyscf_losc.SVWN
                elif self.loscfunctional == 'GGA':
                    losc_func=pyscf_losc.GGA
                else:
                    print("unknown loscfunctional")
                    ashexit()
                if self.functional != self.loscfunctional:
                    print("Warning: PySCF functional and LOSC-function not matching")

                #Writing regular orbitals to disk as Moldenfile
                self.write_orbitals_to_Moldenfile(self.mol, self.mf.mo_coeff, self.mf.mo_occ, self.mf.mo_energy, label="CAN-orbs")
                #Write HOMO and LUMO as cube files
                #homo_idx = self.mol.nelectron // 2 - 1
                #lumo_idx = homo_idx + 1
                #self.cubegen_orbital(self.mol, 'HOMO.cube', self.mf.mo_coeff[:,homo_idx], nx=60,ny=60,nz=60)
                #self.cubegen_orbital(self.mol, 'LUMO.cube', self.mf.mo_coeff[:,lumo_idx], nx=60,ny=60,nz=60)
                
                # Conduct the post-SCF LOC calculation
                #window=[-30,10] optional energy window
                #TODO: get output from program written to disk or as stdout
                if self.LOSC_window != None:
                    print("LOSC Energy window chosen:", self.LOSC_window)
                if self.LOSC_method=='postSCF':
                    print("postSCF LOSC_method chosen")
                    a, b, losc_data = pyscf_losc.post_scf_losc(losc_func,
                    self.mf, return_losc_data = True, orbital_energy_unit='eV', window=self.LOSC_window)
                    print("losc_data:", losc_data)
                    print("a:", a)
                    print("b:", b)
                    self.orbitalets = losc_data["C_lo"][0]
                    self.orbitalets_energies=losc_data["losc_dfa_orbital_energy"][0]/27.211386245988
                    print("orbitalets_energies:", self.orbitalets_energies)
                    self.write_orbitals_to_Moldenfile(self.mol, self.orbitalets, self.mf.mo_occ, self.orbitalets_energies, label="LOSC-orbs")
                elif self.LOSC_method=='SCF': 
                    print("SCF LOSC_method chosen")
                    #SCF-LOSC calculation
                    #Final 
                    loscmf = pyscf_losc.scf_losc(losc_func, self.mf, orbital_energy_unit='eV', window=self.LOSC_window)
                    print("loscmf:", loscmf)
                    self.loscmf=loscmf
                    self.write_orbitals_to_Moldenfile(self.mol, self.loscmf.mo_coeff, self.loscmf.mo_occ, self.loscmf.mo_energy, label="LOSC-SCF-orbs")

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
                natocc, mo_coefficients = self.calculate_natural_orbitals(self.mol,self.mf, method='MP2', elems=elems)

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
                cc = pyscf_cc.CCSD(self.mf, self.frozen_orbital_indices,mo_coeff=mo_coefficients)
            elif self.scf_type == "UHF":
                cc = pyscf_cc.UCCSD(self.mf,self.frozen_orbital_indices,mo_coeff=mo_coefficients)
                
            elif self.scf_type == "RKS":
                print("Warning: CCSD on top of RKS determinant")
                cc = pyscf_cc.CCSD(self.mf.to_rhf(), self.frozen_orbital_indices,mo_coeff=mo_coefficients)
            elif self.scf_type == "UKS":
                print("Warning: CCSD on top of UKS determinant")
                cc = pyscf_cc.UCCSD(self.mf.to_uhf(),self.frozen_orbital_indices,mo_coeff=mo_coefficients)

            #Setting CCSD maxcycles (default 200)
            cc.max_cycle=self.cc_maxcycle
            cc.verbose=5 #Shows CC iterations with 5

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
                ashexit()
                df, configs = vmc(wf,configs)
        
        #####################
        #CAS-CI and CASSCF
        #####################

        elif self.CAS is True:
            print("CAS run is on!")
            #First run SCF
            scf_result = self.mf.run()
            
            #Initial orbitals for CAS-CI or CASSCF
            #Checking for AVAS; DMET_CAS and Chkfile options. Otherwise MP2 natural orbitals.
            print("Checking for CAS initial orbital options.")
            if self.AVAS is True:
                print("AVAS automatic CAS option chosen")
                norb_cas, nel_cas, orbitals = pyscf_avas.avas(self.mf, self.CAS_AO_labels)
                print(f"AVAS determined an active space of: CAS({nel_cas},{norb_cas})")
            elif self.DMET_CAS is True:
                print("DMET_CAS automatic CAS option chosen")
                norb_cas, nel_cas, orbitals = pyscf_dmet_cas.guess_cas(self.mf, self.mf.make_rdm1(), self.CAS_AO_labels)
                print(f"DMET_CAS determined an active space of: CAS({nel_cas},{norb_cas})")
            elif self.moreadfile != None:
                print("moreadfile option was specified")
                print("This means that SCF-orbitals are ignored and we will read MO coefficients from chkfile:", self.moreadfile)
                orbitals = pyscf.lib.chkfile.load(self.moreadfile, 'mcscf/mo_coeff')
                norb_cas=self.active_space[1]
                nel_cas=self.active_space[0]
                print(f"CAS active space chosen to be: CAS({nel_cas},{norb_cas})")
            else:
                #TODO: Have this be a keyword option also instead of just else-default ?
                print("Neither AVAS, DMET_CAS or moreadfile options chosen.")
                print("Will now calculate MP2 natural orbitals to use as input in CAS job")
                natocc, orbitals = self.calculate_natural_orbitals(self.mol,self.mf, method='MP2', elems=elems)
                print("Checking if cas_nmin/cas_nmax keyword were specified")
                if self.cas_nmin != None and self.cas_nmax != None:
                    print(f"Active space will be determined from MP2 natorbs. NO threshold parameters: cas_nmin={self.cas_nmin} and cas_nmax={self.cas_nmax}")
                    print("Note: Use active_space keyword if you want to select active space manually instead")
                    # Determing active space from natorb thresholds
                    nat_occs_for_thresholds=[i for i in natocc if i < self.cas_nmin and i > self.cas_nmax]
                    norb_cas = len(nat_occs_for_thresholds)
                    nel_cas = round(sum(nat_occs_for_thresholds))
                    print(f"To get this same active space in another calculation you can also do: active_space=[{nel_cas},{norb_cas}]")
                else:
                    print("Using active_space keyword information")
                    norb_cas=self.active_space[1]
                    nel_cas=self.active_space[0]
                print(f"CAS active space chosen to be: CAS({nel_cas},{norb_cas})")

            print(f"Now running CAS job with active space of {nel_cas} electrons in {norb_cas} orbitals")
            if self.CASSCF is True:
                print("Doing CASSCF (orbital optimization)")
                if self.mcpdft is True:
                    casscf = self.mcpdft_l.CASSCF (self.mf, self.mcpdft_functional, norb_cas, nel_cas)
                else:
                    #Regular CASSCF
                    casscf = self.mcscf.CASSCF(self.mf, norb_cas, nel_cas)
                casscf.max_cycle_macro=self.casscf_maxcycle
                casscf.verbose=self.verbose_setting
                #Writing of checkpointfile
                if self.write_chkfile_name != None:
                    casscf.chkfile = self.write_chkfile_name
                else:
                    casscf.chkfile = "casscf.chk"
                if self.printlevel >1:
                    print("Will write checkpointfile:", casscf.chkfile )

                #CASSCF starting from AVAS/DMET_CAS/MP2 natural orbitals
                #Making sure that we only feed in one set of orbitals into CAS (CC is OK with alpha and beta)
                if type(orbitals) == list:
                    if len(orbitals) == 2:
                        # Assuming list of [alphorbs-array,betaorbs-array]
                        orbitals = orbitals[0]
                    else:
                        print("Something wrong with orbitals:", orbitals)
                        print("Exiting")
                        ashexit()
                if self.mcpdft is True:
                    mcpdft_result = casscf.run(orbitals, natorb=True)
                    print("E(CASSCF):", mcpdft_result.e_mcscf)
                    print(f"Eot({self.mcpdft_functional}):", mcpdft_result.e_ot)
                    print("E(tot, MC-PDFT):", mcpdft_result.e_tot)
                    print("E(ci):", mcpdft_result.e_cas)
                    print("")
                    #casscf.compute_pdft_energy_()
                    #Optional recompute with different on-top functional
                    #e_tot, e_ot, e_states = casscf.compute_pdft_energy_(otxc='tBLYP')
                    self.energy=mcpdft_result.e_tot
                else:
                    #Regular CASSCF
                    e_tot, e_cas, fcivec, mo, mo_energy = casscf.run(orbitals, natorb=True)
                    print("e_tot:", e_tot)
                    print("e_cas:", e_cas)
                    self.energy = e_tot
                print("CASSCF run done\n")
            else:
                print("Doing CAS-CI (no orbital optimization)")
                if self.mcpdft is True:
                    print("mcpdft is True")
                    casci= self.mcpdft_l.CASCI (self.mf, self.mcpdft_functional, norb_cas, nel_cas)
                    casci.verbose=self.verbose_setting
                    mcpdft_result = casci.run()
                    print("E(CASSCF):", mcpdft_result.e_mcscf)
                    print(f"Eot({self.mcpdft_functional}):", mcpdft_result.e_ot)
                    print("E(tot, MC-PDFT):", mcpdft_result.e_tot)
                    print("E(ci):", mcpdft_result.e_cas)
                    print("")
                    self.energy = mcpdft_result.e_tot
                    print("CAS-CI run done\n")
                else:
                    #Regular CAS-CI
                    casci = self.mcscf.CASCI(self.mf, norb_cas, nel_cas)
                    casci.verbose=self.verbose_setting
                    #CAS-CI from chosen orbitals above
                    e_tot, e_cas, fcivec, mo, mo_energy = casci.kernel(orbitals)
                    print("e_tot:", e_tot)
                    print("e_cas:", e_cas)
                    self.energy = e_tot
                    print("CAS-CI run done\n")
        else:
            if self.printlevel >1:
                print("No post-SCF job.")

        ##############
        #GRADIENT
        ##############
        #NOTE: only SCF supported for now
        if Grad==True:
            if self.printlevel >1:
                print("Gradient requested")
            if self.postSCF is True:
                print("Gradient for postSCF methods is not implemented in ASH interface")
                ashexit()
            if PC is True:
                print("Gradient with PC is not quite ready")
                #print("Units need to be checked.")
                ashexit()
                hfg = mm_charge_grad(grad.dft.RKS(self.mf), current_MM_coords, MMcharges)
                #                grad = self.mf.nuc_grad_method()
                self.gradient = hfg.kernel()
            else:
                if self.printlevel >1:
                    print("Calculating regular SCF gradient")
                #Doing regular SCF gradient
                if self.dispersion != None:
                    print("here")
                    print("self.mf", self.mf)
                    print("self.mf dict", self.mf.__dict__)
                    #print("self.mf version", self.mf.version)
                    #self.mf.version='bj'
                    #print("self.mf version", self.mf.version)
                    self.gradient = self.mf.nuc_grad_method().kernel()
                else:
                    self.gradient = self.mf.nuc_grad_method().kernel()
                if self.printlevel >1:
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

