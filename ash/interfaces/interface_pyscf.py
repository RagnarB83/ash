import time

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader,listdiff
import ash.modules.module_coords
from ash.modules.module_results import ASH_Results
from ash.functions.functions_elstructure import get_ec_entropy,get_entropy
import os
import sys
import glob
import numpy as np
from functools import reduce
import random
import copy

#PySCF Theory object.
#TODO: Somehow support reading in user mf object ?
#Easier now than before. However, each run calls both prepare_run and actual_run
#Can we skip prepare_run (creates mf etc.) and update coordinates only?
#TODO: PE: Polarizable embedding (CPPE). Revisit
#TODO: Dirac HF/KS
#TODO: Gradient for post-SCF methods and TDDFT

class PySCFTheory:
    def __init__(self, printsetting=False, printlevel=2, numcores=1, label="pyscf", platform='CPU', GPU_pcgrad=False,
                  scf_type=None, basis=None, basis_file=None, cartesian_basis=None, ecp=None, functional=None, gridlevel=5, symmetry='C1',
                  guess='minao', dm=None, moreadfile=None, write_chkfile_name='pyscf.chk',
                  noautostart=False, autostart=True, fcidumpfile=None,
                  soscf=False, damping=None, diis_method='DIIS', diis_start_cycle=0, level_shift=None,
                  fractional_occupation=False, scf_maxiter=50, scf_noiter=False, direct_scf=True, GHF_complex=False, collinear_option='mcol',
                  NMF=False, NMF_sigma=None, NMF_distribution='FD', stability_analysis=False,
                  BS=False, HSmult=None, atomstoflip=None,
                  TDDFT=False, tddft_numstates=10, NTO=False, NTO_states=None,
                  mom=False, mom_occindex=0, mom_virtindex=1, mom_spinmanifold=0,
                  dispersion=None, densityfit=False, auxbasis=None, sgx=False, magmom=None,
                  pe=False, potfile=None, filename='pyscf', memory=3100, conv_tol=1e-8, verbose_setting=4,
                  CC=False, CCmethod=None, CC_direct=False, frozen_core_setting='Auto', cc_maxcycle=200, cc_diis_space=6,
                  CC_density=False, cc_conv_tol_normt=1e-06, cc_conv_tol=1e-07,
                  MP2=False,MP2_DF=False,MP2_density=False, DFMP2_density_relaxed=False,
                  CAS=False, CASSCF=False, CASSCF_numstates=1, CASSCF_weights=None, CASSCF_mults=None,
                  CASSCF_wfnsyms=None, active_space=None, casscf_maxcycle=200,
                  frozen_virtuals=None, FNO=False, FNO_orbitals='MP2', FNO_thresh=None, x2c=False,
                  AVAS=False, DMET_CAS=False, CAS_AO_labels=None, APC=False, apc_max_size=(2,2),
                  cas_nmin=None, cas_nmax=None, losc=False, loscfunctional=None, LOSC_method='postSCF',
                  loscpath=None, LOSC_window=None,
                  mcpdft=False, mcpdft_functional=None):

        self.theorynamelabel="PySCF"
        self.theorytype="QM"
        self.analytic_hessian=True
        self.printlevel=printlevel
        #if self.printlevel >= 2:
        print_line_with_mainheader("PySCFTheory initialization")

        #EARLY EXITS

        #Exit early if no SCF-type
        if scf_type is None:
            print("Error: You must select an scf_type, e.g. 'RHF', 'UHF', 'GHF', 'RKS', 'UKS', 'GKS'")
            ashexit()
        if basis is None and basis_file is None and fcidumpfile is None:
            print("Error: You must either provide a basis or a basis_file keyword . Basis can a name (string) or dict (elements as keys)")
            print("basis_file should be a string of the filename containing basis set for each element, in NWChem format")
            print("Best to download basis set from https://www.basissetexchange.org/")
            ashexit()
        if basis_file is not None and not os.path.isfile(basis_file):
            print("Error: basis_file does not exist. Exiting")
            ashexit()
        if functional is not None:
            if self.printlevel >= 1:
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
        if NMF is True:
            if NMF_distribution is None:
                print("NMF requires NMF_distribution keyword. Exiting")
                ashexit()
            if NMF_sigma is None:
                print("NMF requires NMF_sigma keyword. Exiting")
                ashexit()
        if BS is True:
            if atomstoflip is None:
                print("Error: BS is True but atomstoflip is not set. This is required. Exiting")
                ashexit()
            if scf_type == 'RHF' or scf_type == 'RKS':
                print("SCF-type has to be unrestricted for BS. Exiting")
                ashexit()
            if HSmult is None:
                print("Error: BS is True but HSmult is not set. This is required. Exiting")
                ashexit()
        if CC is True and CCmethod is None:
            print("Error: Need to choose CCmethod, e.g. 'CCSD', 'CCSD(T)")
            ashexit()
        if CASSCF is True and CAS is False:
            CAS=True
            #Check
            if CASSCF_mults != None:
                if type(CASSCF_numstates) == int:
                    print("For a state-averaged CASSCF with different spin multiplicities, CASSCF_numstates must be a list")
                    print("Example: if CASSCF_mults=[1,3] you should set CASSCF_numstates=[2,4] for 2 singlet and 4 triplet states")
                    ashexit()

        #Store optional properties of pySCF run job in a dict
        self.properties ={}

        #Setting meanfield object as None
        self.mf = None

        self.label=label
        self.memory=memory
        self.filename=filename
        self.printsetting=printsetting
        self.verbose_setting=verbose_setting

        # Counter for how often pyscftheory.run is called
        self.runcalls = 0

        #CPPE Polarizable Embedding options
        self.pe=pe
        #Potfile from user or passed on via QM/MM Theory object ?
        self.potfile=potfile

        # SCF
        self.platform=platform
        self.GPU_pcgrad=GPU_pcgrad #Pointcharge gradient not on GPU by default
        if self.platform == 'GPU':
            print("Warning: GPU platform for PySCF. This requires gpu4pyscf plugin to be available")
            self.GPU_pcgrad=True
            print("Pointcharge gradient will also be performed on GPU using cupy")
        self.scf_type=scf_type
        self.stability_analysis=stability_analysis
        self.conv_tol=conv_tol
        self.direct_scf=direct_scf
        self.basis=basis #Basis set can be string or dict with elements as keys
        self.basis_file = basis_file
        self.cartesian_basis=cartesian_basis
        self.magmom=magmom
        self.ecp=ecp
        self.functional=functional
        self.x2c=x2c
        self.gridlevel=gridlevel
        self.symmetry=symmetry
        #post-SCF
        self.frozen_core_setting=frozen_core_setting
        self.frozen_virtuals=frozen_virtuals
        #CC
        self.CC=CC
        self.CCmethod=CCmethod
        self.CC_density=CC_density #CC density True or False
        self.CC_direct=CC_direct
        self.cc_maxcycle=cc_maxcycle
        self.cc_diis_space=cc_diis_space
        self.cc_conv_tol_normt=cc_conv_tol_normt
        self.cc_conv_tol=cc_conv_tol
        self.FNO=FNO
        self.FNO_thresh=FNO_thresh
        self.FNO_orbitals=FNO_orbitals

        #Orbitals
        self.moreadfile=moreadfile
        self.write_chkfile_name=write_chkfile_name
        self.noautostart=noautostart
        if autostart is False:
            self.noautostart=True
        #FCIDUMP file as read-in option
        self.fcidumpfile=fcidumpfile


        #CAS
        self.CAS=CAS
        self.CASSCF=CASSCF
        self.active_space=active_space
        self.casscf_maxcycle=casscf_maxcycle
        self.CASSCF_numstates=CASSCF_numstates
        self.CASSCF_weights=CASSCF_weights
        self.CASSCF_mults=CASSCF_mults
        self.CASSCF_wfnsyms=CASSCF_wfnsyms

        #Auto-CAS options
        self.APC=APC
        self.apc_max_size=apc_max_size
        self.AVAS=AVAS
        self.DMET_CAS=DMET_CAS
        self.CAS_AO_labels=CAS_AO_labels
        self.cas_nmin=cas_nmin
        self.cas_nmax=cas_nmax

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

        #MP2
        self.MP2=MP2
        self.MP2_DF=MP2_DF
        self.MP2_density=MP2_density
        self.DFMP2_density_relaxed=DFMP2_density_relaxed #Whether DF-MP2 density is relaxed or not

        #BS
        self.BS=BS
        self.HSmult=HSmult
        self.atomstoflip=atomstoflip

        #GHF/GKS options
        self.GHF_complex = GHF_complex #Complex or not
        self.collinear_option=collinear_option  #Options: col, ncol, mcol

        #TDDFT
        self.TDDFT=TDDFT
        self.tddft_numstates=tddft_numstates
        #NTO
        self.NTO=NTO
        self.NTO_states=NTO_states  #List of states to calculate NTOs for
        #MOM
        self.mom=mom
        self.mom_virtindex=mom_virtindex # The relative virtual orbital index to excite into. Default 1 (LUMO). Choose 2 for LUMO+1 etc.
        self.mom_occindex=mom_occindex #The relative occupied orbital index to excite from. Default 0 (HOMO). Choose -1 for HOMO-1 etc.
        self.mom_spinmanifold=mom_spinmanifold #The spin-manifold (0: alpha or 1: beta) to excited electron in. Default: 0 (alpha)

        #SCF max iterations
        if scf_noiter is True:
            print("SCF noiter keyword is True. Setting SCF maxiter to 0")
            self.scf_maxiter=0
        else:
            self.scf_maxiter=scf_maxiter

        #Guess orbitals (if None then default)
        self.guess=guess

        #Input density matrix (overrides everything else)
        self.dm=dm

        #Damping
        self.damping=damping

        #Level-shift
        self.level_shift=level_shift

        #DIIS options
        self.diis_method=diis_method
        self.diis_start_cycle=diis_start_cycle

        #Fractional occupation/smearing
        self.fractional_occupation=fractional_occupation

        #Non-aufbau mean-field (NMF)
        self.NMF=NMF
        self.NMF_distribution=NMF_distribution
        self.NMF_sigma=NMF_sigma

        #SOSCF (Newton)
        self.soscf=soscf


        #Density fitting and semi-numeric exchange options
        self.densityfit = densityfit #RI-J
        self.auxbasis = auxbasis #Aux J basis
        self.sgx=sgx #Semi-numerical exchange

        #Special PySCF run option
        #self.specialrun=specialrun

        #Whether job is SCF (HF/DFT) only or a post-SCF method like CC or CAS
        self.postSCF=False
        if self.CC is True:
            self.postSCF=True
        if self.MP2 is True:
            self.postSCF=True
        if self.TDDFT is True:
            self.postSCF=True
        if self.mom is True:
            self.postSCF=True
        if self.CAS is True:
            self.postSCF=True
            print("CAS is True. Active_space keyword should be defined unless AVAS, APC or DMET_CAS is True.")

            if self.AVAS is True or self.DMET_CAS is True:
                print("AVAS/DMET_CAS is True")
                if self.CAS_AO_labels is None:
                    print("AVAS/DMET_CAS requires CAS_AO_labels keyword. Specify as e.g. CAS_AO_labels=['Fe 3d', 'Fe 4d', 'C 2pz']")
                    ashexit()
            if self.APC is True:
                print("Ranked-orbital APC method is True")
                print("APC max size:", self.apc_max_size)
            elif self.cas_nmin != None or self.cas_nmax != None:
                print("Keyword cas_nmin and cas_nmax provided")
                print("Will use together with MP2 natural orbitals to choose CAS")
            elif self.active_space == None or len(self.active_space) != 2:
                print("No option chosen for active space.")
                print("If neither AVAS,DMET_CAS or cas_nmin/cas_nmax options are chosen then")
                print("active_space must be defined as a list of 2 numbers (M electrons in N orbitals)")
                ashexit()

            #Checking if multi-state CASSCF or not
            if type(self.CASSCF_numstates) is int:
                print("CASSCF_numstates given as integer")
                self.CASSCF_totnumstates=self.CASSCF_numstates
            elif type(self.CASSCF_numstates) is list:
                print("CASSCF_numstates given as list")
                self.CASSCF_totnumstates=sum(self.CASSCF_numstates)
            print("Total number of CASSCF states: ", self.CASSCF_totnumstates)


        #Are we doing an initial SCF calculation or not
        #Generally yes.
        #TODO: Can we skip this for CASSCF?
        self.SCF=True

        #Attempting to load pyscf
        #self.load_pyscf()
        self.numcores=numcores
        if self.losc is True:
            self.load_losc(loscpath)

        #Number of orbitals and basis functions (only setup upon run)
        self.num_basis_functions=None
        self.num_orbs=None

        #Print the options
        if self.printlevel >= 1:
            print("SCF:", self.SCF)
            print("SCF-type:", self.scf_type)
            print("x2c:", self.x2c)
            print("Post-SCF:", self.postSCF)
            print("Symmetry:", self.symmetry)
            print("conv_tol:", self.conv_tol)
            print("Grid level:", self.gridlevel)
            print("verbose_setting:", self.verbose_setting)
            print("Basis:", self.basis)
            print("Basis-file:", self.basis_file)
            print("SCF stability analysis:", self.stability_analysis)
            print("Functional:", self.functional)
            print("Coupled cluster:", self.CC)
            print("CC method:", self.CCmethod)
            print("CC direct:", self.CC_direct)
            print("CC density:", self.CC_density)
            print("CC maxcycles:", self.cc_maxcycle)
            print("CC DIIS space size:", self.cc_diis_space)
            print("FNO-CC:", self.FNO)
            print("FNO_thresh:", self.FNO_thresh)
            print("FNO orbitals:", self.FNO_orbitals)
            print("MP2:", self.MP2)
            print("MP2_DF:",self.MP2_DF)
            print("Frozen_core_setting:", self.frozen_core_setting)
            print("Frozen_virtual orbitals:",self.frozen_virtuals)
            print("Polarizable embedding:", self.pe)
            print()
            print("CAS:", self.CAS)
            print("CASSCF:", self.CASSCF)
            print("CASSCF maxcycles:", self.casscf_maxcycle)
            print("AVAS:", self.AVAS)
            print("DMET_CAS:", self.DMET_CAS)
            print("APC:", self.APC)
            print("APC max size:", self.apc_max_size)
            print("CAS_AO_labels (for AVAS/DMET_CAS)", self.CAS_AO_labels)
            print("CAS_nmin:", self.cas_nmin)
            print("CAS_nmax:", self.cas_nmax)
            print("Active space:", self.active_space)
            print()
            print("MC-PDFT:", self.mcpdft)
            print("mcpdft_functional:", self.mcpdft_functional)

    def read_fcidump_file(self,fcidumpfile):
        import pyscf.tools.fcidump
        
        #Read FCI dump and return dictionary with integrals etc
        #result = pyscf.tools.fcidump.read(fcidumpfile, verbose=True)
        #print("result:", result)
        #H1, H2, ECORE, NORB, NELEC, MS, ORBSYM, ISYM

        self.mf = pyscf.tools.fcidump.to_scf(fcidumpfile, molpro_orbsym=False)

    # Create FCIDUMP file from either mf object (provided or internal)
    def create_fcidump_file(self, mf=None, dump_from_mos=False, mo_coeff=None, 
                            filename="FCIDUMP", tol=1e-15):
        import pyscf.tools.fcidump

        # Dump from MOs if selected (can be any MO-coefficients)
        if dump_from_mos is True:
            print("Creating FCIDUMP from MOs")
            pyscf.tools.fcidump.from_mo(self.mol,filename, mo_coeff, tol=tol)
        else:
            print("Creating FCIDUMP from mf object")
            # Otherwise mf object
            if mf is None:
                print("No mf object provided. Using internal mf (self.mf)")
                mf=self.mf

            pyscf.tools.fcidump.from_scf(mf, filename, tol=tol)

            print("Created FCIDUMP file:", filename)

    def determine_frozen_core(self,elems):
        print("Determining frozen core")
        # Main elements
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
        return self.frozen_core_orbital_indices
    def load_losc(self,loscpath):
        #Not sure if this works. Think PYTHONPATH is necessary
        if loscpath != None:
            sys.path.insert(0, loscpath)
        try:
            import pyscf_losc
        except ModuleNotFoundError as e:
            print("Problem importing pyscf_losc")
            print("e:", e)
            ashexit()

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
        print("Setting numcores to: ", self.numcores)
    #    #Controlling OpenMP parallelization.
        import pyscf
        pyscf.lib.num_threads(numcores)
    #Cleanup after run.
    def cleanup(self):
        files=['timer.dat', self.filename+'.dat',self.filename+'.chk' ]
        print("Cleaning up old PySCF files")
        for f in files:
            print("Removing file:", f)
            try:
                os.remove(f)
            except:
                print("Error removing file", f)
    def print_orbital_en_and_occ(self,mo_energies=None, mo_occupations=None):
        print("\nMO energies and occupations:\n")
        header="  No.      OCC               E(Eh)               E(eV)"
        if mo_energies is None:
            mo_energies=self.mf.mo_energy
            mo_occupations=self.mf.mo_occ

        #UHF/UKS
        if mo_energies.ndim == 2:
            print("ALPHA SET")
            print(header)
            for i,(mo_en_a,mo_occ_a) in enumerate(zip(mo_energies[0], mo_occupations[0])):
                print(f"""{i:4d} {mo_occ_a:10.5f} {mo_en_a:20.10f} {mo_en_a*27.2114:20.10f}""")
            print("\nBETA SET")
            print(header)
            for j,(mo_en_b,mo_occ_b) in enumerate(zip(mo_energies[1], mo_occupations[1])):
                print(f"""{j:4d} {mo_occ_b:10.5f} {mo_en_b:20.10f} {mo_en_b*27.2114:20.10f}""")
        #RHF/RKS:
        else:
            print(header)
            for i,(mo_en_a,mo_occ_a) in enumerate(zip(mo_energies, mo_occupations)):
                print(f"""{i:4d} {mo_occ_a:10.5f} {mo_en_a:20.10f} {mo_en_a*27.2114:20.10f}""")

    def write_orbitals_to_Moldenfile(self,mol, mo_coeffs, occupations, mo_energies=None, label="orbs"):
        from pyscf.tools import molden
        print("Writing orbitals to disk as Molden file")
        if mo_energies is None:
            print("No MO energies. Setting to 0.0")
            mo_energies = np.array([0.0 for i in occupations])
        with open(f'{label}.molden', 'w') as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(mol, f1, mo_coeffs, ene=mo_energies, occ=occupations)
        return f'{label}.molden'

    #Write Cube files for a list of orbital indices
    def cubegen_orbital(self, mol, name, coeffs, nx=60,ny=60,nz=60):
        import pyscf.tools
        pyscf.tools.cubegen.orbital(mol, name, coeffs, nx=nx, ny=ny, nz=nz)
        print("cubegen_orbital: Wrote file:", name)
    #Write Cube file for density
    def cubegen_density(self, mol, name, dm, nx=60,ny=60,nz=60):
        import pyscf.tools
        pyscf.tools.cubegen.density(mol, name, dm, nx=nx, ny=ny, nz=nz)
        print("cubegen_density: Wrote file:", name)
    #Write Cube file for MEP
    def cubegen_mep(self, mol, name, dm, nx=60,ny=60,nz=60):
        import pyscf.tools
        pyscf.tools.cubegen.mep(mol, name, dm, nx=nx, ny=ny, nz=nz)
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
        #Necessary to reimport
        import pyscf
        import pyscf.cc as pyscf_cc
        import pyscf.mcscf
        print("Number of PySCF lib threads is:", pyscf.lib.num_threads())

        #Determine frozen core from element list
        self.determine_frozen_core(elems)
        self.frozen_orbital_indices=self.frozen_core_orbital_indices
        print("Frozen orbital indices:", self.frozen_orbital_indices)
        #Necessary for MP2 at least
        if self.frozen_orbital_indices == []:
            self.frozen_orbital_indices=None
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

        #TODO: rewrite so we can reuse run_MP2 method instead
        if 'MP2' in method:
            print("Running MP2 natural orbital calculation")
            import pyscf.mp
            # MP2 natural occupation numbers and natural orbitals
            #Simple canonical MP2 with unrelaxed density
            if method == 'MP2' or method == 'canMP2':
                mp2 = pyscf.mp.MP2(mf, frozen=self.frozen_orbital_indices)
                print("Running MP2")
                mp2.run()
            elif method == 'DFMP2' or method =='DFMP2relax':
                from pyscf.mp.dfmp2_native import DFRMP2
                from pyscf.mp.dfump2_native import DFUMP2
                #DF-MP2 scales better but syntax differs: https://pyscf.org/user/mp.html#dfmp2
                if self.scf_type == "RKS" or self.scf_type == "RHF" :
                    unrestricted=False
                    mp2 = DFRMP2(mf, frozen=self.frozen_orbital_indices)
                else:
                    unrestricted=True
                    mp2 = DFUMP2(mf, frozen=(self.frozen_orbital_indices,self.frozen_orbital_indices))
                #Now run DMP2 object
                mp2.run()

            #Make natorbs
            #NOTE: This should not have to recalculate RDM here since provided
            #natocc, natorb = dmp2.make_natorbs(rdm1_mo=dfmp2_dm, relaxed=relaxed)
            #NOTE: Slightly silly, calling make_natural_orbitals will cause dm calculation again
            natocc, natorb = pyscf.mcscf.addons.make_natural_orbitals(mp2)
            #natocc, natorb = self.mcscf.addons.make_natural_orbitals(mp2)
        elif method =='FCI':
            print("Running FCI natural orbital calculation")
            print("not ready")
            exit()
            #TODO: FCI https://github.com/pyscf/pyscf/blob/master/examples/fci/14-density_matrix.py
            # FCI solver
            #cisolver = pyscf.fci.FCI(mol, myhf.mo_coeff)
            #e, fcivec = cisolver.kernel()
            # Spin-traced 1-particle density matrix
            #norb = myhf.mo_coeff.shape[1]
            # 6 alpha electrons, 4 beta electrons because spin = nelec_a-nelec_b = 2
            #elec_a = 6
            #nelec_b = 4
            #dm1 = cisolver.make_rdm1(fcivec, norb, (nelec_a,nelec_b))
        elif method == 'AVAS-CASSCF':
            print("Doing AVAS and then CASSCF to get natural orbitals")
            from pyscf.mcscf import avas
            norb_cas, nel_cas, avasorbitals = avas.avas(self.mf, CAS_AO_labels)
            print(f"AVAS determined an active space of: CAS({nel_cas},{norb_cas})")
            print(f"Now doing CASSCF using AVAS active space (CAS({nel_cas},{norb_cas})) and AVAS orbitals")
            casscf = pyscf.mcscf.CASSCF(mf, norb_cas, nel_cas)
            casscf.max_cycle_macro=self.casscf_maxcycle
            casscf.verbose=self.verbose_setting
            cas_result = casscf.run(avasorbitals, natorb=True)
            print("CASSCF occupations", cas_result.mo_occ)
            return cas_result.mo_occ, cas_result.mo_coeff
        elif method == 'DMET-CASSCF':
            from pyscf.mcscf import dmet_cas
            print("Doing DMET-CAS and then CASSCF to get natural orbitals")
            print("DMET_CAS automatic CAS option chosen")
            norb_cas, nel_cas, dmetorbitals = dmet_cas.guess_cas(mf, mf.make_rdm1(), CAS_AO_labels)
            print(f"DMET_CAS determined an active space of: CAS({nel_cas},{norb_cas})")
            print(f"Now doing CASSCF using DMET-CAS active space (CAS({nel_cas},{norb_cas})) and DMET-CAS orbitals")
            casscf = pyscf.mcscf.CASSCF(mf, norb_cas, nel_cas)
            casscf.max_cycle_macro=self.casscf_maxcycle
            casscf.verbose=self.verbose_setting
            cas_result = casscf.run(dmetorbitals, natorb=True)
            print("CASSCF occupations", cas_result.mo_occ)
            return cas_result.mo_occ, cas_result.mo_coeff
        elif method == 'CCSD':
            print("Running CCSD natural orbital calculation")
            natocc,natorb,rdm1 = self.calculate_CCSD_natorbs(ccsd=None, mf=mf)
        elif method == 'CCSD(T)':
            print("Running CCSD(T) natural orbital calculation")
            #No CCSD(T) object in pyscf. So manual approach. Slower algorithms
            natocc,natorb,rdm1 = self.calculate_CCSD_T_natorbs(ccsd=None, mf=mf)

        with np.printoptions(precision=5, suppress=True):
            print(f"{method} natural orbital occupations:", natocc)
        print_time_rel(module_init_time, modulename='calculate_natural_orbitals', moduleindex=2)
        return natocc, natorb

   #Function for CCSD natural orbitals using ccsd and mf objects
    def calculate_CCSD_natorbs(self,ccsd=None, mf=None):
        import pyscf
        import pyscf.cc as pyscf_cc
        import pyscf.mcscf
        print("Running CCSD natural orbital calculation")
        if mf is None:
            mf = self.mf
        if ccsd is None:
            ccsd = pyscf_cc.CCSD(mf, frozen=self.frozen_orbital_indices)
            ccsd.max_cycle=200
            ccsd.verbose=5
            ccsd.run()
        #Request rdm and natural orbitals
        rdm = ccsd.make_rdm1(ao_repr=True)
        natocc, natorb = pyscf.mcscf.addons.make_natural_orbitals(ccsd)
        return natocc,natorb,rdm
    #Manual verbose protocol for calculating CCSD(T) natural orbitals using ccsd and mf objects
    def calculate_CCSD_T_natorbs(self,ccsd=None, mf=None):
        print("Running CCSD(T) natural orbital calculation")
        import pyscf
        import pyscf.mcscf
        import scipy
        from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
        from pyscf.cc import uccsd_t_lambda
        from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
        from pyscf.cc import uccsd_t_rdm

        if mf is None:
            mf = self.mf
        if ccsd is None:
            ccsd = pyscf.cc.CCSD(mf, frozen=self.frozen_orbital_indices)
            ccsd.max_cycle=200
            ccsd.verbose=5
            ccsd.run()

        eris = ccsd.ao2mo()

        #Make RDMs for ccsd(t) RHF and UHF
        #Note: Checking type of CCSD object because if ROHF object then was automatically converted to UHF and hence UCCSD
        print("Solving lambda equations")
        print("Warning: this step is slow and not parallelized")
        if type(ccsd) == pyscf.cc.uccsd.UCCSD:
            print("CCSD(T) lambda UHF")
            #NOTE: No threading parallelization seen here, not sure why
            conv, l1, l2 = uccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2, max_cycle=self.cc_maxcycle)
            rdm1 = uccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris, ao_repr=True)
            Dm = rdm1[0]+rdm1[1]
        else:
            print("CCSD(T) lambda RHF")
            conv, l1, l2 = ccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2, max_cycle=self.cc_maxcycle)
            rdm1 = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris, ao_repr=True)
            if np.ndim(rdm1) == 3:
                Dm = rdm1[0]+rdm1[1]
            elif np.ndim(rdm1) == 2:
                Dm = rdm1
        if conv is False:
            print("Error: CCSD(T) lambda equations failed to converge! Be very careful with the results")
            ashexit()
        else:
            print("CC lambda equations converged!")
        # Diagonalize the DM in AO basis
        S = mf.get_ovlp()
        A = reduce(np.dot, (S, Dm, S))
        w, v = scipy.linalg.eigh(A, b=S)
        # Flip NOONs (and NOs) since they're in increasing order
        natocc = np.flip(w)
        natorb = np.flip(v, axis=1)

        return natocc,natorb,rdm1

    def run_NMF_step(self):
        print("NMF smearing active. Getting NMF energy")
        print("Sigma:", self.NMF_sigma)
        E_mf = self.mf.e_tot
        print("Mean-field energy:", E_mf)
        occ = self.mf.get_occ(mo_energy_kpts=self.mf.mo_energy)
        print("occ:", occ)

        #NOTE: Not sure if this is correct yet
        if self.scf_type == 'UHF' or self.scf_type == 'UKS':
            occ_a = occ[0]
            occ_b = occ[1]
            S_a = get_entropy(occ_a)
            S_b = get_entropy(occ_b)
            print("Total entropy alpha (FD):", S_a)
            print("Total entropy beta (FD):", S_b)

            if 'fermi' in self.NMF_distribution.lower() or  self.NMF_distribution == 'FD':
                print("Fermi entropy")
                Ec_a = get_ec_entropy(occ_a, self.NMF_sigma, method='fermi')
                Ec_b = get_ec_entropy(occ_a, self.NMF_sigma, method='fermi')
                Ec = Ec_a/2 + Ec_b/2
            elif self.NMF_distribution.lower() == 'gaussian' or self.NMF_distribution == 'G':
                Ec_a = get_ec_entropy(occ_a, self.NMF_sigma, method='gaussian')
                Ec_b = get_ec_entropy(occ_b, self.NMF_sigma, method='gaussian')
                Ec = Ec_a/2 + Ec_b/2
            else:
                print("Unknown distribution")
                ashexit()
        else:
            S = get_entropy(occ)
            print("Total entropy (FD):", S)
            print("Sigma:", self.NMF_sigma)
            if 'fermi' in self.NMF_distribution.lower() or  self.NMF_distribution == 'FD':
                Ec = get_ec_entropy(occ, self.NMF_sigma, method='fermi')
            elif self.NMF_distribution.lower() == 'gaussian' or self.NMF_distribution == 'G':
                Ec = get_ec_entropy(occ, self.NMF_sigma, method='gaussian')
            else:
                print("Unknown distribution")
                ashexit()
        print("Correlation energy:", Ec)
        E_tot_NMF = E_mf + Ec
        print("Total NMF energy:", E_tot_NMF)
        return E_tot_NMF

    #Population analysis, requiring mf and dm objects
    #Currently only Mulliken
    def run_population_analysis(self, mf, unrestricted=False, dm=None, type='Mulliken', label=None, verbose=3):
        import pyscf
        print()
        print("Running population analysis")

        #TODO: gpu4pyscf errors for Mulliken pop analysis. Probably fixed later
        #For now, we return
        if self.platform == 'GPU':
            print("GPU4PySCF does not support Mulliken population analysis right now. Returning")
            #import gpu4pyscf
            #mull_pop_func = gpu4pyscf.dft.RKS.mulliken_pop
            return
        else:
            mull_pop_func = pyscf.scf.rhf.mulliken_pop
            mull_spinpop_func = pyscf.scf.uhf.mulliken_spin_pop

        if label==None:
            label=''
        if type == 'Mulliken':
            if unrestricted is False:
                if dm is None:
                    dm = mf.make_rdm1()
                #print("dm:", dm)
                #print("dm.shape:", dm.shape)
                mulliken_populations =mull_pop_func(self.mol,dm, verbose=verbose)
                print(f"{label} Mulliken charges:", mulliken_populations[1])
            elif unrestricted is True:
                if dm is None:
                    dm = mf.make_rdm1()
                #print("dm:", dm)
                #print("dm.shape:", dm.shape)
                mulliken_populations =mull_pop_func(self.mol,dm, verbose=verbose)
                mulliken_spinpopulations = mull_spinpop_func(self.mol,dm, verbose=verbose)
                print(f"{label} Mulliken charges:", mulliken_populations[1])
                print(f"{label} Mulliken spin pops:", mulliken_spinpopulations[1])
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
            if self.scf_type == 'GHF' or self.scf_type == 'GKS':
                #pyscf.ghf_stability(self.mf, verbose=None, return_status=False):
                mos, stable =  self.mf.stability(external=True, return_status=True, verbose=5)
                if stable is True:
                    print("GHF/GKS WF is stable")
                else:
                    print("GHF/GKS WF is unstable")
                return
            else:
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

    #Attempt to read specified chkfile for orbitals
    def read_chkfile(self,chkfile):
        import pyscf
        #CHKFILE LOAD
        #Loading previous orbitals from chkfile into object

        #TODO: Check if orbitals read are the correct shape for scf_type. RKS vs. UKS incompatibility etc?
        #At least print warning, but for RKS->UKS we could in principle double the mo-coeffs and modify occupations
        #For UKS->RKS we could use only one set of mo-coeffs and modify occupations
        #
        #RKS reading UKS: currently IndexError     mocc = mo_coeff[:,mo_occ>0] IndexError: boolean index did not match indexed array along dimension 1; dimension is 15 but corresponding boolean dimension is 2
        #UKS reading RKS:  currently:   nao = dma.shape[-1]    IndexError: tuple index out of range

        #TODO: Similar for GHF/GKS from UHF/UKS

        print(f"read_chkfile: Attempting to read chkfile: {chkfile}")
        if chkfile != None:
            if self.printlevel >= 1:
                print("Reading orbitals from checkpointfile:", chkfile)
            if os.path.isfile(chkfile) is False:
                print("File does not exist. Continuing!")
                return False
            try:
                self.chkfileobject = pyscf.scf.chkfile.load(chkfile, 'scf')
            except TypeError:
                if self.printlevel >= 1:
                    print("No SCF orbitals found in chkfile. Could be checkpointfile from CASSCF?")
                    print("Ignoring and continuing")
                return False
            #Check information in chkfile
            #Check if unrestricted or restricted information
            if len(self.chkfileobject["mo_occ"]) == 2:
                print("Chkfile mo occ length is 2 => Unrestricted")
                chkfile_scftype="UHF"
            elif 2.0 in self.chkfileobject["mo_occ"]:
                chkfile_scftype="RHF"
            #Checking if mismatch between chkfile info and chosen scf-type
            #TODO: In principle we could convert RKS-info from chkfile to UKS-info and vice versa
            if chkfile_scftype == "UHF":
                if self.scf_type == "RHF" or self.scf_type == "RKS":
                    print("Warning: Mismatch between SCF-type in chkfile and PySCFTheory object. Ignoring chkfile")
                    return False
            if chkfile_scftype == "RHF":
                if self.scf_type == "UHF" or self.scf_type == "UKS":
                    print("Warning: Mismatch between SCF-type in chkfile and PySCFTheory object. Ignoring chkfile")
                    return False 
            
            if chkfile_scftype == "UHF":
                #UNRESTRICTED
                if self.printlevel >= 1:
                    print("Reading unrestricted orbitals from checkpointfile")
                #unrestricted
                try:
                    sum_occ = int(sum(self.chkfileobject["mo_occ"][0]) + sum(self.chkfileobject["mo_occ"][1]))
                    num_occ = len(self.chkfileobject["mo_occ"][0])
                except AttributeError:
                    if self.printlevel >= 1:
                        print("No occupations found in chkfile. Continuing")
                    return False
                if num_occ != self.num_basis_functions:
                    print(f"Number of occupations ({num_occ}) in chkfile does not match number of basis functions ({self.num_basis_functions})")
                    print("Ignoring MOs in chkfile and continuing")
                    return False
                print("self.num_electrons:", self.num_electrons)
                print("int(sum_occ):", int(sum_occ))
                if self.num_electrons != int(sum_occ):
                    if self.printlevel >= 1:
                        print(f"Number of electrons in checkpointfile ({sum_occ}) does not match number of electrons in molecule ({self.num_electrons})")
                        print("Ignoring MOs in chkfile and continuing")
                    return False
            else:
                #RESTRICTED
                if self.printlevel >= 1:
                    print("Reading restricted orbitals from checkpointfile")
                
                num_occ = len(self.chkfileobject["mo_occ"])
                try:
                    sum_occ = int(sum(self.chkfileobject["mo_occ"]))
                except AttributeError:
                    if self.printlevel >= 1:
                        print("No occupations found in chkfile. Continuing")
                    return False
                if num_occ != self.num_basis_functions:
                    print(f"Number of occupations ({num_occ}) in chkfile does not match number of basis functions ({self.num_basis_functions})")
                    print("Ignoring MOs in chkfile and continuing")
                    return False
                if self.num_electrons != int(sum_occ):
                    if self.printlevel >= 1:
                        print(f"Number of electrons in checkpointfile ({sum_occ}) does not match number of electrons in molecule ({self.num_electrons})")
                        print("Ignoring MOs in chkfile and continuing")
                    return False
            return True


    def setup_guess(self):
        if self.printlevel >= 1:
            print("Setting up orbital guess")

        #DM SPECIFIED
        if self.dm is not None:
        #Input DM matrix specified
            if self.printlevel >= 1:
                print("DM found inside pySCFTheory object. Using this for guess")
                print("Num. basis functions:", self.num_basis_functions)
            #print("self.dm.shape:", self.dm.shape)
            #print("self.dm.shape[0]:", self.dm.shape[0])
            if self.scf_type == 'RHF' or self.scf_type == 'RKS' or self.scf_type == 'GHF' or self.scf_type == 'GKS':
                if self.dm.shape[0] != self.num_basis_functions:
                    print(f"Warning: The density matrix shape {self.dm.shape} does not match number of basis functions ({self.num_basis_functions}).")
                    print("This density matrix can not be correct. Ignoring")
                    self.dm=None
                    return None
            else:
                #UHF/UKS etc
                if self.printlevel >= 1:
                    print("Num basis functions:", self.num_basis_functions)
                    print("DM shape:", self.dm[0].shape) #DM is a tuple for unrestricted
                if self.dm[0].shape[0] != self.num_basis_functions:
                    if self.dm[0].shape[0]*2 != self.num_basis_functions:
                        print(f"Warning: The density matrix shape {self.dm[0].shape[0]} does not match number of basis functions ({self.num_basis_functions}).")
                        print("This density matrix can not be correct. Ignoring")
                        self.dm=None
                    return None
            return self.dm


        #MOREADFILE
        elif self.moreadfile != None:
            if self.printlevel >= 1:
                print("Moread: Trying to read SCF-orbitals from file")
            if '.molden' in self.moreadfile:
                mo_coefficients, occupations = pySCF_read_MOs(self.moreadfile,self)
                self.mf.mo_occ = occupations
                self.mf.mo_coeff = mo_coefficients
            else:
                self.read_chkfile(self.moreadfile)
                self.mf.__dict__.update(self.chkfileobject)
            dm = self.mf.make_rdm1()
            return dm
        #NOTHING SPECIFIED: so possible AUTOSTART
        else:
            if self.printlevel >= 1:
                print("Neither input dm or moreadfile was specified")
            #1.AUTOSTART (unless noautostart)
            if self.noautostart is False:
                if self.printlevel >= 1:
                    print(f"Autostart: Trying file: {self.filename+'.chk'}")
                #AUTOSTART: FIRST CHECKING if CHKFILE with self.filename(pyscf.chk) exists
                if self.read_chkfile(self.filename+'.chk') is True:
                    self.mf.__dict__.update(self.chkfileobject)
                    dm = self.mf.make_rdm1()
                    return dm
            #2. GUESS GENERATION (noautostart or autostart failed)
            if self.noautostart is True:
                if self.printlevel >= 1:
                    print("Autostart false enforced")
            #If noautostart is True or no checkpointfile we do the regular orbital-guess (default minao or whatever is specified)
            if self.printlevel >= 1:
                print("Using orbital guess option:", self.guess)
            if self.guess not in ['minao', 'atom', 'huckel', 'vsap','1e']:
                print("Guess option not recognized. Valid guess options are: ", ['minao', 'atom', 'huckel', 'vsap','1e'])
                ashexit()
            self.mf.init_guess = self.guess
            return None #Should be fine since we set init_guess in line above

    #LOSC calculation
    def calc_losc(self):
        import pyscf_losc
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

    def run_hessian(self):
        print("Running meanfield Hessian")
        import pyscf.hessian
        self.hessian_obj = self.mf.Hessian()
        hessian = self.hessian_obj.kernel()
        return hessian

    def run_MP2(self,frozen_orbital_indices=None, MP2_DF=None):
        print("\nInside run_MP2")
        import pyscf.mp
        print("Frozen orbital indices:",frozen_orbital_indices)

        #If no frozen-orbs (e.g. H2) then set frozen_orbital_indices to None
        if len(frozen_orbital_indices) == 0:
            frozen_orbital_indices=None

        if self.scf_type == "RKS" or self.scf_type == "RHF" :
            print("Using restricted MP2 code")
            unrestricted=False
        else:
            unrestricted=True

        #Simple canonical MP2
        if MP2_DF is False:
            print("Warning: MP2_DF keyword is False. Will run slow canonical MP2")
            print("Set MP2_DF to True for faster DF-MP2/RI-MP2")
            mp2 = pyscf.mp.MP2(self.mf, frozen=frozen_orbital_indices)
            print("Running MP2")
            mp2.run()
            MP2_energy = mp2.e_tot
            mp2_object=mp2
        else:
            print("MP2_DF is True. Will run density-fitted MP2 code")
            from pyscf.mp.dfmp2_native import DFRMP2
            from pyscf.mp.dfump2_native import DFUMP2
            if unrestricted is False:
                print("Using restricted DF-MP2 code")
                dmp2 = DFRMP2(self.mf, frozen=frozen_orbital_indices)
            else:
                print("Using unrestricted DF-MP2 code")
                dmp2 = DFUMP2(self.mf, frozen=(frozen_orbital_indices,frozen_orbital_indices))
            #Now running DMP2 object
            dmp2.run()
            MP2_energy =  dmp2.e_tot
            mp2_object=dmp2

        return MP2_energy, mp2_object

    def run_MP2_density(self, mp2object, MP2_DF=None, DFMP2_density_relaxed=None):
        print("\nInside run_MP2_density")
        import pyscf.mcscf
        if self.scf_type == "RKS" or self.scf_type == "RHF" :
            unrestricted=False
        else:
            unrestricted=True

        #Density and natural orbitals
        print("\nMP2 density option is active")
        print(f"Now calculating MP2 density matrix and natural orbitals")
        #DM
        if MP2_DF is False:
            print("Now calculating unrelaxed canonical MP2 density")
            #This is unrelaxed canonical MP2 density
            mp2_dm = mp2object.make_rdm1(ao_repr=True)
            density_type='MP2'
        else:
            #RDMs: Unrelaxed vs. Relaxed
            if DFMP2_density_relaxed is True:
                print("Calculating relaxed DF-MP2 density")
                mp2_dm = mp2object.make_rdm1_relaxed(ao_repr=True) #Relaxed
                density_type='DFMP2-relaxed'
            else:
                print("Calculating unrelaxed DF-MP2 density")
                mp2_dm = mp2object.make_rdm1_unrelaxed(ao_repr=True) #Unrelaxed
                density_type='DFMP2-unrelaxed'


        #Preserving new DM
        print("MP2 density matrix stored as dm attribute of PySCFTheory object")
        self.dm = mp2_dm

        print("Mulliken analysis for MP2 density matrix")
        self.run_population_analysis(self.mf, unrestricted=unrestricted, dm=mp2_dm, type='Mulliken', label=density_type)

        #TODO: Fix. Slightly silly, calling make_natural_orbitals will cause dm calculation again
        natocc, natorb = pyscf.mcscf.addons.make_natural_orbitals(mp2object)

        #Dipole moment
        self.get_dipole_moment(dm=mp2_dm,label="MP2")

        #Printing occupations
        print(f"\nMP2 natural orbital occupations:")
        print(natocc)
        print()
        print("NO-based polyradical metrics:")
        ash.functions.functions_elstructure.poly_rad_index_nu(natocc)
        ash.functions.functions_elstructure.poly_rad_index_nu_nl(natocc)
        ash.functions.functions_elstructure.poly_rad_index_n_d(natocc)
        print()
        molden_name=f"pySCF_MP2_natorbs"
        print(f"Writing MP2 natural orbitals to Moldenfile: {molden_name}.molden")
        self.write_orbitals_to_Moldenfile(self.mol, natorb, natocc,  label=molden_name)

        return natocc, natorb, mp2_dm

    def run_CAS(self,elems=None):
        print("\nInside run_CAS")
        import pyscf.mcscf

        ##############################################
        #First run SCF
        # required even though orbitals are not used
        ##############################################
        scf_result = self.mf.run()


        ############################
        # INITIAL ORBITALS
        # and active space selection
        ############################
        #Initial orbitals for CAS-CI or CASSCF
        #Checking for AVAS; DMET_CAS and Chkfile options. Otherwise MP2 natural orbitals.
        print("Checking for CAS initial orbital options.")
        if self.AVAS is True:
            from pyscf.mcscf import avas
            print("AVAS automatic CAS option chosen")
            norb_cas, nel_cas, orbitals = avas.avas(self.mf, self.CAS_AO_labels)
            print(f"AVAS determined an active space of: CAS({nel_cas},{norb_cas})")
        elif self.DMET_CAS is True:
            from pyscf.mcscf import dmet_cas
            print("DMET_CAS automatic CAS option chosen")
            norb_cas, nel_cas, orbitals = dmet_cas.guess_cas(self.mf, self.mf.make_rdm1(), self.CAS_AO_labels)
            print(f"DMET_CAS determined an active space of: CAS({nel_cas},{norb_cas})")
        elif self.APC is True:
            from pyscf.mcscf import apc
            print("APC automatic CAS option chosen")
            #entropies = np.random.choice(np.arange(len(self.mf.mo_occ)),len(self.mf.mo_occ),replace=False)
            #chooser = apc.Chooser(self.mf.mo_coeff,self.mf.mo_occ,entropies,max_size=self.apc_max_size)
            #norb_cas, nel_cas, orbitals, active_idx = chooser.kernel()
            myapc = apc.APC(self.mf,max_size=self.apc_max_size)
            norb_cas,nel_cas,orbitals = myapc.kernel()
            print("norb_cas:", norb_cas)
            print("nel_cas:", nel_cas)
            print("orbitals:", orbitals)
            #print("active_idx:", active_idx)
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
            print("Initial orbitals setup done")
            print()
            print("Now checking if cas_nmin/cas_nmax keyword were specified")
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
        ########################
        # CASSCF (orbital opt)
        ########################

        ##################
        # prepare CASSCF
        ##################
        if self.CASSCF is True:
            print("Doing CASSCF (orbital optimization)")
            if self.mcpdft is True:
                from pyscf import mcpdft, mcdcft
                #old: casscf = pyscf.mcpdft_l.CASSCF (self.mf, self.mcpdft_functional, norb_cas, nel_cas)
                casscf = mcpdft.CASSCF (self.mf, self.mcpdft_functional, norb_cas, nel_cas)
            else:
                #Regular CASSCF
                casscf = pyscf.mcscf.CASSCF(self.mf, norb_cas, nel_cas)
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
            ################################
            #CASSCF MULTIPLE STATES or not
            ################################
            if self.CASSCF_totnumstates > 1:
                print(f"\nMultiple CASSCF states option chosen")
                print("Creating state-average CASSCF object")
                if self.CASSCF_weights == None:
                    print("No CASSCF weights chosen (CASSCF_weights keyword)")
                    print("Settings equal weights for states")
                    weights = [1/self.CASSCF_totnumstates for i in range(self.CASSCF_totnumstates)]
                    print("Weights:", weights)
                #Different spin multiplicities for each state
                if self.CASSCF_mults != None:
                    print("CASSCF_mults keyword was specified")
                    print("Using this to set multiplicity for each state")
                    print("Total number of states:", self.CASSCF_totnumstates)
                    solvers=[]
                    print("Creating multiple FCI solvers")
                    #Disabling for now
                    #if self.CASSCF_wfnsyms == None:
                    #    print("No CASSCF_wfnsyms set. Assuming no symmetry and setting all to A")
                    #    self.CASSCF_wfnsyms=['A' for i in self.CASSCF_mults ]
                    for mult,nstates_per_mult in zip(self.CASSCF_mults,self.CASSCF_numstates):
                        #Creating new solver
                        print(f"Creating new solver for mult={mult} with {nstates_per_mult} states")
                        solver = pyscf.fci.FCI(self.mol)
                        #solver.wfnsym= wfnsym
                        #solver.orbsym= None
                        solver.nroots = nstates_per_mult
                        solver.spin = mult-1
                        solvers.append(solver)
                    print("Solvers:", solvers)
                    casscf = pyscf.mcscf.state_average_mix_(casscf, solvers, weights)
                #Or not:
                else:
                    casscf = pyscf.mcscf.state_average_(casscf, weights)
                #TODO: Check whether input orbitals can be used with this
            else:
                print("Single-state CASSCF calculation")
            ##############################
            #RUN MC-PDFT CASSCF
            ##############################
            if self.mcpdft is True:
                #Do the CASSCF calculation with on-top functional
                print("Now running MC-PDFT with on-top functional")
                mcpdft_result = casscf.run(orbitals, natorb=True)
                #mc1 = mcdcft.CASSCF (mf, 'cBLYP', 4, 4).run ()
                #print("Now running cBLYP on top")
                #mc1 = mcdcft.CASSCF (self.mf, 'cBLYP', norb_cas, nel_cas).run ()
                #print("mc1:", mc1)
                print("E(CASSCF):", mcpdft_result.e_mcscf)
                print(f"Eot({self.mcpdft_functional}):", mcpdft_result.e_ot)
                print("E(tot, MC-PDFT):", mcpdft_result.e_tot)
                print("E(ci):", mcpdft_result.e_cas)
                print("")
                #casscf.compute_pdft_energy_()
                #Optional recompute with different on-top functional
                #e_tot, e_ot, e_states = casscf.compute_pdft_energy_(otxc='tBLYP')
                self.energy=mcpdft_result.e_tot
            ##############################
            #RUN regular CASSCF
            ##############################
            else:
                #Regular CASSCF
                print("Running CASSCF object")
                print(casscf.__dict__)
                print("CASSCF FCI solver:", casscf.fcisolver.__dict__)
                casscf_result = casscf.run(orbitals, natorb=True)
                print("casscf_result:", casscf_result)
                e_tot = casscf_result.e_tot
                e_cas = casscf_result.e_cas
                print("e_tot:", e_tot)
                print("e_cas:", e_cas)
                self.energy = e_tot
            print("CASSCF run done\n")
        ##############################
        #RUN CAS-CI
        ##############################
        else:
            print("Doing CAS-CI (no orbital optimization)")
            if self.mcpdft is True:
                print("mcpdft is True")
                casci= pyscf.mcpdft_l.CASCI (self.mf, self.mcpdft_functional, norb_cas, nel_cas)
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
                casci = pyscf.mcscf.CASCI(self.mf, norb_cas, nel_cas)
                casci.verbose=self.verbose_setting
                #CAS-CI from chosen orbitals above
                e_tot, e_cas, fcivec, mo, mo_energy = casci.kernel(orbitals)
                print("e_tot:", e_tot)
                print("e_cas:", e_cas)
                self.energy = e_tot
                print("CAS-CI run done\n")

    def run_MOM(self):
        print("\nMaximum Overlap Method calculation is ON!")
        import pyscf
        # Change 1-dimension occupation number list into 2-dimension occupation number
        # list like the pattern in unrestircted calculation
        mo0 = copy.copy(self.mf.mo_coeff)
        occ = copy.copy(self.mf.mo_occ)

        if self.scf_type == 'UHF' or self.scf_type == 'UKS':
            print("UHF/UKS MOM calculation")
            print("Previous SCF MO occupations are:")
            print("Alpha:", occ[0].tolist())
            print("Beta:", occ[1].tolist())

            #The chosen spin manifold
            spinmanifold = self.mom_spinmanifold

            #Finding HOMO index
            HOMOnum = list(occ[spinmanifold]).index(0.0)-1

            #Defining the occupied orbital index to excite from (default HOMO). if mom_occindex is -1 and HOMO is 54 then used_occ_index = 54 + (-1) = 53
            used_occ_index = HOMOnum + self.mom_occindex

            #Finding LUMO index
            LUMOnum = HOMOnum + self.mom_virtindex


            print(f"HOMO (spinmanifold:{spinmanifold}) index:", HOMOnum)
            print(f"OCC MO index (spinmanifold:{spinmanifold}) to excite from:", used_occ_index)
            print(f"LUMO index to excite into: {LUMOnum} (LUMO+{self.mom_virtindex-1})")
            print("Spin manifold:", self.mom_spinmanifold)
            print("Modifying guess")
            # Assign initial occupation pattern
            occ[spinmanifold][used_occ_index]=0	 # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
            occ[spinmanifold][LUMOnum]=1	 # it is still a singlet state

            # New SCF caculation
            if self.scf_type == 'UKS':
                MOMSCF = pyscf.scf.UKS(self.mol)
                MOMSCF.xc = self.functional
            elif self.scf_type == 'UHF':
                MOMSCF = pyscf.scf.UHF(self.mol)

            # Construct new dnesity matrix with new occpuation pattern
            dm_u = MOMSCF.make_rdm1(mo0, occ)
            # Apply mom occupation principle
            MOMSCF = pyscf.scf.addons.mom_occ(MOMSCF, mo0, occ)
            # Start new SCF with new density matrix
            print("Starting new SCF with modified MO guess")
            MOMSCF.scf(dm_u)

            #delta-SCF transition energy
            trans_energy = (MOMSCF.e_tot - self.mf.e_tot)*27.211
            print()
            print("-"*40)
            print("DELTA-SCF RESULTS")
            print("-"*40)

            print()
            print(f"Ground-state SCF energy {self.mf.e_tot} Eh")
            print(f"Excited-state SCF energy {MOMSCF.e_tot} Eh")
            print()
            print(f"delta-SCF transition energy {trans_energy} eV")
            print()
            print('Alpha electron occupation pattern of ground state : %s' %(self.mf.mo_occ[0].tolist()))
            print('Beta electron occupation pattern of ground state : %s' %(self.mf.mo_occ[1].tolist()))
            print()
            print('Alpha electron occupation pattern of excited state : %s' %(MOMSCF.mo_occ[0].tolist()))
            print('Beta electron occupation pattern of excited state : %s' %(MOMSCF.mo_occ[1].tolist()))

        elif self.scf_type == 'ROHF' or self.scf_type == 'ROKS' or self.scf_type == 'RHF' or self.scf_type == 'RKS':
            print("ROHF/ROKS MOM calculation")

            #Defining the index of the HOMO
            HOMOnum = list(occ).index(0.0)-1
            #Defining the occupied orbital index to excite from (default HOMO). if mom_occindex is -1 and HOMO is 54 then used_occ_index = 54 + (-1) = 53
            used_occ_index = HOMOnum + self.mom_occindex

            #Defining the virtual orbital index to excite into(default LUMO)
            LUMOnum = HOMOnum + self.mom_virtindex
            spinmanifold = self.mom_spinmanifold
            print("Previous SCF MO occupations are:", occ.tolist())
            print("HOMO index:", HOMOnum)
            print("Occupied MO index to excite from", used_occ_index)
            print(f"LUMO index to excite into: {LUMOnum} (LUMO+{self.mom_virtindex-1})")
            print("Spin manifold:", self.mom_spinmanifold)
            print("Modifying guess")
            setocc = np.zeros((2, occ.size))
            setocc[:, occ==2] = 1

            # Assigned initial occupation pattern
            setocc[0][used_occ_index] = 0    # this excited state is originated from OCCMO(alpha) -> LUMO(alpha)
            setocc[0][LUMOnum] = 1    # it is still a singlet state
            ro_occ = setocc[0][:] + setocc[1][:]    # excited occupation pattern within RO style
            print("setocc:",setocc)
            print("ro_occ:",ro_occ)
            # New ROKS/ROHF SCF calculation
            if self.scf_type == 'ROHF' or self.scf_type == 'RHF':
                MOMSCF = pyscf.scf.ROHF(self.mol)
            elif self.scf_type == 'ROKS' or self.scf_type == 'RKS':
                MOMSCF = pyscf.scf.ROKS(self.mol)
                MOMSCF.xc = self.functional

            # Construct new density matrix with new occpuation pattern
            dm_ro = MOMSCF.make_rdm1(mo0, ro_occ)
            # Apply mom occupation principle
            MOMSCF = pyscf.scf.addons.mom_occ(MOMSCF, mo0, setocc)
            # Start new SCF with new density matrix
            print("Starting new SCF with modified MO guess")
            MOMSCF.scf(dm_ro)

            #delta-SCF transition energy in eV
            trans_energy = (MOMSCF.e_tot - self.mf.e_tot)*27.211
            print()
            print("-"*40)
            print("DELTA-SCF RESULTS")
            print("-"*40)
            print()
            print(f"Ground-state SCF energy {self.mf.e_tot} Eh")
            print(f"Excited-state SCF energy {MOMSCF.e_tot} Eh")
            print()
            print(f"delta-SCF transition energy {trans_energy} eV")
            print()
            print('Electron occupation pattern of ground state : %s' %(self.mf.mo_occ.tolist()))
            print('Electron occupation pattern of excited state : %s' %(MOMSCF.mo_occ.tolist()))
        else:
            print("Unknown scf-type for MOM")
            ashexit()

        #delta-SCF results
        self.properties["Transition_energy"] = trans_energy

        #Overlap
        # Calculate overlap between two determiant <I|F>
        #S,X = pyscf.scf.uhf.det_ovlp(self.mf.mo_coeff,MOMSCF.mo_coeff,self.mf.mo_occ,MOMSCF.mo_occ,self.mf.get_ovlp())
        #print("overlap:", S)
        #print("X:", X)

        #Transition density matrix
        #D_21 = MOMSCF.mo_coeff* adjugate(s)*self.mf.mo_occ * self.mf.mo_coeff
        #D_21 = reduce(np.dot, (MOMSCF.mo_coeff, np.transpose(S),np.transpose(self.mf.mo_coeff)))
        #D_21 = MOMSCF.mo_coeff* np.transpose(S)*self.mf.mo_coeff
        #print("D_21:",D_21)

        #charges = self.mol.atom_charges()
        #coords = self.mol.atom_coords()
        #nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        #print("nuc_charge_center:",nuc_charge_center)
        #self.mol.set_common_orig_(nuc_charge_center)
        #dip_ints = self.mol.intor('cint1e_r_sph', comp=3)
        #print("dip_ints:",dip_ints)
        #transition_dipoles = np.einsum('xij,ji->x', dip_ints, D_21)
        #print("transition_dipoles:",transition_dipoles)
        #https://github.com/pyscf/pyscf/blob/master/examples/scf/51-elecoup_mom.py

        return MOMSCF
    def run_TDDFT(self):
        print("\nInside run_TDDFT")
        import pyscf.tddft
        print(f"Now running TDDFT (Num states: {self.tddft_numstates})")
        mytd = pyscf.tddft.TDDFT(self.mf)
        mytd.nstates = self.tddft_numstates
        mytd.kernel()
        mytd.analyze()

        print("-"*40)
        print("TDDFT RESULTS")
        print("-"*40)
        #print("TDDFT transition energies (Eh):", mytd.e)
        print("TDDFT transition energies (eV):", mytd.e*27.211399)

        # Transition dipoles
        t_dip = mytd.transition_dipole()
        print("Transition dipoles:", t_dip)

        # Oscillator strengths
        osc_strengths_length = mytd.oscillator_strength(gauge='length')
        osc_strengths_vel = mytd.oscillator_strength(gauge='velocity')
        print("Oscillator strengths (length):", osc_strengths_length)
        print("Oscillator strengths (velocity):", osc_strengths_vel)

        #Storing in properties object
        self.properties["TDDFT_transition_energies"] = mytd.e*27.211399
        self.properties["TDDFT_transition_dipoles"] = t_dip
        self.properties["TDDFT_oscillator strengths"] = osc_strengths_length

        #NTO Analysis for state 1

        if self.NTO is True:
            print("\nNTO analysis for state 1")
            if type(self.NTO_states) != list:
                print("NTO_states must be a list")
                ashexit()
            print("Now doing NTO analysis for states:", self.NTO_states)
            print(f"See pySCF outputfile ({self.filename}.out) for the NTO analysis")
            from pyscf.tools import molden
            for ntostate in self.NTO_states:
                print("Doing NTO for state:", ntostate)
                NTO_weight, nto_bla = mytd.get_nto(state=ntostate, verbose=4)
                print("Writing Molden-file:", f'nto-td-{ntostate}.molden')
                molden.from_mo(self.mol, f'nto-td-{ntostate}.molden', nto_bla)

    #Set up frozen natural orbitals
    def setup_FNO(self,elems=None):
        print("FNO is True")
        if self.FNO_orbitals =='MP2':
            print("MP2 natural orbitals on!")
            print("Will calculate MP2 natural orbitals to use as input in CC job")
            natocc, mo_coefficients = self.calculate_natural_orbitals(self.mol,self.mf, method='MP2', elems=elems)
        elif self.FNO_orbitals =='CCSD':
            print("CCSD natural orbitals on!")
            print("Will calculate CCSD natural orbitals to use as input in CC job")
            natocc, mo_coefficients = self.calculate_natural_orbitals(self.mol,self.mf, method='CCSD', elems=elems)

        #Optional natorb truncation if FNO_thresh is chosen
        if self.FNO_thresh is not None:
            print("FNO thresh option chosen:", self.FNO_thresh)
            num_small_virtorbs=len([i for i in natocc if i < self.FNO_thresh])
            print("Num. virtual orbitals below threshold:", num_small_virtorbs)
            #List of frozen orbitals
            virt_frozen= [self.num_scf_orbitals_alpha-i for i in range(1,num_small_virtorbs+1)][::-1]
            print("List of frozen virtuals:", virt_frozen)
            print("List of frozen virtuals:", virt_frozen)
            self.frozen_orbital_indices = self.frozen_orbital_indices + virt_frozen
        return mo_coefficients

    def run_CC(self,mf=None, frozen_orbital_indices=None, CCmethod='CCSD(T)', CC_direct=False, mo_coefficients=None):
        print("\nInside run_CC")
        import pyscf.scf
        import pyscf.dft
        import pyscf.cc as pyscf_cc
        import pyscf.mcscf

        if mf is None:
            mf = self.mf

        #CCSD-part as RCCSD or UCCSD
        print()
        print("Frozen_orbital_indices:", frozen_orbital_indices)
        print("Total number of frozen orbitals:", len(frozen_orbital_indices))
        print("Total number of orbitals:", self.num_orbs) #Should have been set when mf.run was called
        print("Number of active orbitals:", self.num_orbs - len(frozen_orbital_indices))
        print()

        #If no frozen-orbs (e.g. H2) then set frozen_orbital_indices to None
        if len(frozen_orbital_indices) == 0:
            frozen_orbital_indices=None

        #Check mo_coefficients in case we have to do unrestricted CC (list of 2 MOCoeff ndarrays required)
        if mo_coefficients is not None:
            print("Input orbitals found for run_CC")
            if type(mo_coefficients) is np.ndarray:
                print("MO coefficients are ndarray")
                if mo_coefficients.ndim == 2:
                    if isinstance(mf, pyscf.scf.rohf.ROHF): #Works for ROKS too
                        print("Warning: Meanfield object is ROHF, but not supported by RCCSD. PySCF will convert to UHF and use UCCSD")
                        print("Warning: Duplicating MO coefficients for UCCSD")
                        mo_coefficients=[mo_coefficients,mo_coefficients]
                    elif isinstance(mf, pyscf.scf.uhf.UHF) or isinstance(mf, pyscf.dft.uks.UKS):
                        print("Warning: single set of MO coefficients found but UHF/UKS determinant used. Duplicating MO coefficients")
                        mo_coefficients=[mo_coefficients,mo_coefficients]
        else:
            print("No input orbitals used. Using SCF orbitals from mf object")
        print("Now starting CCSD calculation")
        if self.scf_type == "RHF":
            cc = pyscf_cc.CCSD(mf, frozen_orbital_indices,mo_coeff=mo_coefficients)
        elif self.scf_type == "ROHF":
            cc = pyscf_cc.CCSD(mf, frozen_orbital_indices,mo_coeff=mo_coefficients)
        elif self.scf_type == "UHF":
            cc = pyscf_cc.UCCSD(mf, frozen_orbital_indices,mo_coeff=mo_coefficients)
        elif self.scf_type == "RKS":
            print("Warning: CCSD on top of RKS determinant")
            cc = pyscf_cc.CCSD(mf.to_rhf(), frozen_orbital_indices,mo_coeff=mo_coefficients)
        elif self.scf_type == "ROKS":
            print("Warning: CCSD on top of ROKS determinant")
            cc = pyscf_cc.CCSD(mf.to_rhf(), frozen_orbital_indices,mo_coeff=mo_coefficients)
        elif self.scf_type == "UKS":
            print("Warning: CCSD on top of UKS determinant")
            cc = pyscf_cc.UCCSD(mf.to_uhf(), frozen_orbital_indices,mo_coeff=mo_coefficients)

        #Setting thresholds
        cc.conv_tol=self.cc_conv_tol
        cc.conv_tol_normt=self.cc_conv_tol_normt

        ccobject=cc
        #Checking whether CC object created is unrestricted or not
        if type(cc) == pyscf.cc.uccsd.UCCSD:
            unrestricted=True
        else:
            unrestricted=False

        #Setting CCSD maxcycles (default 200)
        cc.max_cycle=self.cc_maxcycle
        cc.verbose=5 #Shows CC iterations with 5
        cc.diis_space=self.cc_diis_space  #DIIS space size (default 6)

        #Switch to integral-direct CC if user-requested
        #NOTE: Faster but only possible for small/medium systems
        cc.direct = self.CC_direct

        ccsd_result = cc.run()
        print("Reference energy:", ccsd_result.e_hf)
        #CCSD energy (this is total energy unless Bruckner or triples are added)
        energy = ccsd_result.e_tot
        print("CCSD energy:", energy)
        corr_CCSD_energy = ccsd_result.e_tot - ccsd_result.e_hf
        print("CCSD correlation energy:", corr_CCSD_energy)

        #T1,D1 and D2 diagnostics
        if unrestricted is True:
            T1_diagnostic_alpha = pyscf.cc.ccsd.get_t1_diagnostic(cc.t1[0])
            T1_diagnostic_beta = pyscf.cc.ccsd.get_t1_diagnostic(cc.t1[1])
            T1_diagnostic_ave = np.sqrt(T1_diagnostic_alpha*T1_diagnostic_beta)
            #T1_diagnostic_ave2 = 0.5*(T1_diagnostic_alpha+T1_diagnostic_beta)
            print("T1 diagnostic (alpha):", T1_diagnostic_alpha)
            print("T1 diagnostic (beta):", T1_diagnostic_beta)
            print("T1 diagnostic (average):", T1_diagnostic_ave)
            #print("T1 diagnostic (average2):", T1_diagnostic_ave2)
            try:
                D1_diagnostic_alpha = pyscf.cc.ccsd.get_d1_diagnostic(cc.t1[0])
                D1_diagnostic_beta = pyscf.cc.ccsd.get_d1_diagnostic(cc.t1[1])
                print("D1 diagnostic (alpha):", D1_diagnostic_alpha)
                print("D1 diagnostic (beta):", D1_diagnostic_beta)

                D1_diagnostic_alpha = pyscf.cc.ccsd.get_d1_diagnostic(cc.t1[0])
                D1_diagnostic_beta = pyscf.cc.ccsd.get_d1_diagnostic(cc.t1[1])
                print("D1 diagnostic (alpha):", D1_diagnostic_alpha)
                print("D1 diagnostic (beta):", D1_diagnostic_beta)
                D2_diagnostic_alpha = pyscf.cc.ccsd.get_d2_diagnostic(cc.t2[0])
                D2_diagnostic_beta = pyscf.cc.ccsd.get_d2_diagnostic(cc.t2[1])
                print("D2 diagnostic (alpha):", D2_diagnostic_alpha)
                print("D2 diagnostic (beta):", D2_diagnostic_beta)
            except IndexError:
                print("Problem calculating D1/D2 diagnostics. Skipping")
        else:
            T1_diagnostic = pyscf.cc.ccsd.get_t1_diagnostic(cc.t1)
            print("T1 diagnostic:", T1_diagnostic)
            D1_diagnostic = pyscf.cc.ccsd.get_d1_diagnostic(cc.t1)
            print("D1 diagnostic:", D1_diagnostic)
            D2_diagnostic = pyscf.cc.ccsd.get_d2_diagnostic(cc.t2)
            print("D2 diagnostic:", D2_diagnostic)

        #Brueckner coupled-cluster wrapper, using an outer-loop algorithm.
        if 'BCCD' in CCmethod:
            print("Bruckner CC active. Now doing BCCD on top of CCSD calculation.")
            from pyscf.cc.bccd import bccd_kernel_
            mybcc = bccd_kernel_(cc, diis=True, verbose=4,canonicalization=True)
            ccobject=mybcc
            bccd_energy = mybcc.e_tot
            print("BCCD energy:", bccd_energy)

        #(T) part
        if CCmethod == 'CCSD(T)':
            print("Calculating triples ")
            et = cc.ccsd_t()
            print("Triples energy:", et)
            energy = ccsd_result.e_tot + et
            print("Final CCSD(T) energy:", energy)
        elif CCmethod == 'BCCD(T)':
            print("Calculating triples for BCCD WF")
            et = mybcc.ccsd_t()
            print("Triples energy:", et)
            energy = bccd_energy + et
            print("Final BCCD(T) energy:", energy)

        return energy, ccobject

    def run_CC_density(self,ccobject=None,mf=None):
        print("\nInside run_CC_density")
        import pyscf.mcscf
        if ccobject is None:
            print("No CC object provided. Using self.ccobject")
            ccobject = self.ccobject
        if mf is None:
            print("No mf object provided. Using self.mf")
            mf = self.mf
        #Check R vs U
        if type(ccobject) == pyscf.cc.uccsd.UCCSD:
            unrestricted=True
        elif type(ccobject) == pyscf.cc.ccsd.CCSD:
            unrestricted=False
        else:
            print("Unknown CC object found inside run_CC_density")
            ashexit()
        #Density and natural orbitals
        print("\nCC density option is active")
        print(f"Now calculating {self.CCmethod} density matrix and natural orbitals")
        if self.CCmethod == 'CCSD':
            rdm1 = ccobject.make_rdm1(ao_repr=True)
            natocc, natorb = pyscf.mcscf.addons.make_natural_orbitals(ccobject)
            print("Mulliken analysis for CCSD density matrix")
            self.run_population_analysis(mf, unrestricted=unrestricted, dm=rdm1, type='Mulliken', label='CCSD')
            self.get_dipole_moment(dm=rdm1, label="CCSD")
            molden_name=f"pySCF_CCSD_natorbs"
        elif self.CCmethod == 'BCCD':
            rdm1 = ccobject.make_rdm1(ao_repr=True)
            natocc, natorb = pyscf.mcscf.addons.make_natural_orbitals(ccobject)
            #Dipole moment
            self.get_dipole_moment(dm=rdm1, label="BCCD")
            molden_name=f"pySCF_BCCD_natorbs"
        elif self.CCmethod == 'CCSD(T)':
            natocc,natorb,rdm1 = self.calculate_CCSD_T_natorbs(ccobject,mf)
            print("Mulliken analysis for CCSD(T) density matrix")
            self.run_population_analysis(mf, unrestricted=unrestricted, dm=rdm1, type='Mulliken', label='CCSD(T)')
            dipole = self.get_dipole_moment(dm=rdm1, label="CCSD(T)")
            molden_name=f"pySCF_CCSD_T_natorbs"
        elif self.CCmethod == 'BCCD(T)':
            print("Warning: Density for BCCD(T) has not been tested")
            natocc,natorb,rdm1 = self.calculate_CCSD_T_natorbs(ccobject,mf)
            print("Mulliken analysis for BCCD(T) density matrix")
            self.run_population_analysis(mf, unrestricted=unrestricted, dm=rdm1, type='Mulliken', label='BCCD(T)')
            dipole = self.get_dipole_moment(dm=rdm1, label="BCCD(T)")
            molden_name=f"pySCF_BCCD_T_natorbs"


        #Preserving new DM
        print("Coupled cluster density matrix stored as dm attribute of PySCFTheory object")
        #print("rdm1:", rdm1)
        #print("rdm1[0] shape", rdm1[0].shape)
        self.dm = rdm1

        #Printing occupations
        print(f"\n{self.CCmethod} natural orbital occupations:")
        print(natocc)
        print()
        print("NO-based polyradical metrics:")
        ash.functions.functions_elstructure.poly_rad_index_nu(natocc)
        ash.functions.functions_elstructure.poly_rad_index_nu_nl(natocc)
        ash.functions.functions_elstructure.poly_rad_index_n_d(natocc)
        print()
        print(f"Writing {self.CCmethod} natural orbitals to Moldenfile: {molden_name}.molden")
        self.write_orbitals_to_Moldenfile(self.mol, natorb, natocc,  label=molden_name)

        return natocc, natorb, rdm1

    #Method to grab dipole moment from pyscftheory object  (assumes run has been executed)
    def get_dipole_moment(self, dm=None, label=None):
        if self.printlevel >=1:
            print("get_dipole_moment function.")

        if self.platform =="GPU":
            print("Dipole moment calculation not currently supported on GPU")
            return None

        if label == None:
            label=""
        if dm is None:
            if self.printlevel >=1:
                print("No DM provided. Using mean-field object dm")
            #MF dipole moment
            dipole = self.mf.dip_moment(unit='A.U.',verbose=self.printlevel)
            if self.printlevel >=1:
                print(f"MF Dipole moment ({label}): {dipole} A.U.")
        else:
            if self.printlevel >=1:
                print("Using provided DM")
            dipole = self.mf.dip_moment(dm=dm,unit='A.U.',verbose=self.printlevel)
            if self.printlevel >=1:
                print(f"WF Dipole moment ({label}): {dipole} A.U.")
        return dipole
    def get_polarizability_tensor(self):
        try:
            from pyscf.prop import polarizability
        except ModuleNotFoundError:
            print("pyscf polarizability requires installation of pyscf.prop module")
            print("See: https://github.com/pyscf/properties")
            print("You can install with: pip install git+https://github.com/pyscf/properties")
            ashexit()
        print("Note: pySCF will now calculate the polarizability (SCF-level)")
        polarizability = self.mf.Polarizability().polarizability()
        return polarizability
    # polarizability property now part of separate pyscf properties module


    #Create mol object (self.mol) via method
    def create_mol(self, qm_elems, current_coords, charge, mult, cartesian_basis=None):
        if self.printlevel >= 1:
            print("Creating mol object")
        import pyscf

        #Defining pyscf mol object and populating
        self.mol = pyscf.gto.Mole()

        #Mol system printing. Hardcoding to 3 as otherwise too much PySCF printing
        self.mol.verbose = 3

        coords_string=ash.modules.module_coords.create_coords_string(qm_elems,current_coords)
        self.mol.atom = coords_string
        self.mol.symmetry = self.symmetry
        self.mol.charge = charge
        self.mol.spin = mult-1

        #cartesian basis or not
        if cartesian_basis is not None:
            print("Setting cartesian basis flag to:", cartesian_basis)
            self.mol.cart = cartesian_basis
    #Update mol object with coordinates or charge/mult
    #def update_mol(self, qm_elems, current_coords, charge, mult):
    #    coords_string=ash.modules.module_coords.create_coords_string(qm_elems,current_coords)
    #    self.mol.atom = coords_string
    #    self.mol.charge = charge
    #    self.mol.spin = mult-1

    #Define basis in mol object
    def define_basis(self,elems=None):
        if self.printlevel >= 1:
            print("Defining basis set in mol object")
        import pyscf
        #PYSCF basis object: https://sunqm.github.io/pyscf/tutorial.html
        #NOTE: We should also support basis set exchange API: https://github.com/pyscf/pyscf/issues/1299
        if self.basis_file != None:
            if self.printlevel >= 1:
                print("Reading basis set from file:", self.basis_file)
            basis_dict={}
            for elem in elems:
                if self.printlevel >= 1:
                    print(f"Reading basis set for element: {elem} from file: {self.basis_file}")
                basis_per_elem=pyscf.gto.basis.load(self.basis_file, elem)
                if self.printlevel >= 3:
                    print("basis_per_elem:", basis_per_elem)
                basis_dict[elem]=basis_per_elem
            self.mol.basis=basis_dict
        else:
            if self.printlevel >= 1:
                print("Using basis set from input string")
            self.mol.basis=self.basis
        if self.printlevel >= 1:
            print("Basis set:", self.mol.basis)
        #Optional setting magnetic moments
        if self.magmom != None:
            if self.printlevel >= 1:
                print("Setting magnetic moments from user-input:", self.magmom)
            self.mol.magmom=self.magmom #Should be a list of the collinear spins of each atom
        #ECP: Can be string ('def2-SVP') or dict or a dict with element-specific keys and values
        self.mol.ecp = self.ecp
        #Memory settings
        self.mol.max_memory = self.memory
        ###########

    #Create mf object (self.mf) via method
    def create_mf(self):
        if self.printlevel >= 1:
            print("Creating pySCF mf object")
        import pyscf
        #RKS v UKS v RHF v UHF v GHF v GKS
        #TODO: Dirac HF and KS also
        if self.scf_type == 'RKS':
            self.mf = pyscf.scf.RKS(self.mol)
        elif self.scf_type == 'ROKS':
            self.mf = pyscf.scf.ROKS(self.mol)
        elif self.scf_type == 'ROHF':
            self.mf = pyscf.scf.ROHF(self.mol)
        elif self.scf_type == 'UKS':
            self.mf = pyscf.scf.UKS(self.mol)
        elif self.scf_type == 'RHF':
            self.mf = pyscf.scf.RHF(self.mol)
        elif self.scf_type == 'UHF':
            self.mf = pyscf.scf.UHF(self.mol)
        elif self.scf_type == 'GHF':
            self.mf = pyscf.scf.GHF(self.mol)
        elif self.scf_type == 'GKS':
            self.mf = pyscf.scf.GKS(self.mol)

    #Probably depreceated. Created mf for GPU.
    def create_mf_for_gpu(self):
        if self.printlevel >= 1:
            print("Creating pySCF mf object using gpu4pyscf")

        try:
            import gpu4pyscf
        except ModuleNotFoundError:
            print("gpu4pyscf library not found. Make sure it is installed. See: https://github.com/pyscf/gpu4pyscf")
            ashexit()

        if self.scf_type == 'RKS':
            from gpu4pyscf.dft import rks
            self.mf = rks.RKS(self.mol)
        elif self.scf_type == 'UKS':
            from gpu4pyscf.dft import uks
            self.mf = uks.UKS(self.mol)
        elif self.scf_type == 'RHF':
            from gpu4pyscf.scf import RHF
            self.mf = RHF(self.mol)
        elif self.scf_type == 'UHF':
            from gpu4pyscf.scf import UHF
            self.mf = UHF(self.mol)
        else:
            print("SCF-type not available for gpu4pyscf")
            ashexit()

    def set_mf_scfconv_options(self):
        if self.printlevel >= 1:
            print("Modifying mf SCF options")
        import pyscf
        #Direct SCF or conventional
        self.mf.direct_scf=self.direct_scf
        #Tolerance
        self.mf.conv_tol = self.conv_tol
        #SCF max iterations
        self.mf.max_cycle=self.scf_maxiter

        #Fractional occupation
        if self.fractional_occupation is True:
            if self.printlevel >1:
                print(f"Fractional occupation is on!")
            self.mf = pyscf.scf.addons.frac_occ(self.mf)

        #Damping
        if self.damping != None:
            if self.printlevel >1:
                print(f"Damping value: {self.damping} DIIS-start: {self.diis_start_cycle}")
            self.mf.damp = self.damping
            self.mf.diis_start_cycle=self.diis_start_cycle
        #Level shifting
        #NOTE: https://github.com/pyscf/pyscf/blob/master/examples/scf/03-level_shift.py
        #TODO: Dynamic levelshift:
        # https://github.com/pyscf/pyscf/blob/master/examples/scf/52-dynamically_control_level_shift.py
        #Possibly to apply different levelshift to alpha/beta sets
        if self.level_shift != None:
            if self.printlevel >1:
                print(f"Levelshift value: {self.level_shift}")
            self.mf.level_shift = self.level_shift
        #DIIS
        self.set_diis()

        #SOSCF/Newton
        if self.soscf is True:
            if self.printlevel >1:
                print("SOSCF is True. Turning on in meanfield object")
            self.mf = self.mf.newton()

    def set_diis(self):
        if self.platform == 'CPU':
            import pyscf
            #DIIS option
            if self.diis_method == 'CDIIS' or self.diis_method == 'DIIS':
                self.mf.DIIS = pyscf.scf.DIIS
            elif self.diis_method == 'ADIIS':
                self.mf.DIIS = pyscf.scf.ADIIS
            elif self.diis_method == 'EDIIS':
                self.mf.DIIS = pyscf.scf.EDIIS
        else:
            import gpu4pyscf
            #DIIS option
            if self.diis_method == 'CDIIS' or self.diis_method == 'DIIS':
                self.mf.DIIS = gpu4pyscf.scf.diis.DIIS
            else:
                print("For GPU platform, ADIIS, EDIIS or others are not supported")
                ashexit()

    def set_mf_smearing(self):
        import pyscf.scf.addons
        #Smearing
        if self.NMF is True:
            print("NMF smearing active. Importing pyscf_smearing module")
            from pyscf.scf.addons import smearing_
            print("Replacing mf object with smearing mf ")
            #from pyscf import __config__
            #SMEARING_METHOD = getattr(__config__, 'pbc_scf_addons_smearing_method', 'fermi')
            #print("SMEARING_METHOD:", SMEARING_METHOD)
            #exit()
            if 'fermi' in self.NMF_distribution.lower() or self.NMF_distribution.lower() == 'fd':
                smearing_keyword='fermi'
            elif 'gauss' in self.NMF_distribution.lower():
                smearing_keyword='gauss'
            else:
                print(f"Unknown smearing option ({self.NMF_distribution}). Exiting")
                ashexit()
            print("Using smearing_keyword:", smearing_keyword)
            print("Sigma:", self.NMF_sigma)
            self.mf = smearing_(self.mf, sigma=self.NMF_sigma, method=smearing_keyword)

    def set_dispersion_options(self, Grad=False):
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
                from vdw import to_dftd3, WithDFTD3
                self.D3_D4_Disp_object=WithDFTD3
                #self.mf = to_dftd3(self.mf, do_grad=Grad)
            if self.dispersion == 'D4':
                print("D4 correction on")
                from vdw import to_dftd4,WithDFTD4
                self.D3_D4_Disp_object=WithDFTD4
                #self.mf = to_dftd4(self.mf, do_grad=Grad)
            elif self.dispersion == 'TS':
                print("TS correction on")
                from vdw import to_mbd
                self.mf = to_mbd(self.mf, variant="ts", do_grad=Grad)
            elif self.dispersion == 'MBD':
                print("MBD correction on")
                from vdw import to_mbd
                self.mf = to_mbd(self.mf, variant="rsscs", do_grad=Grad)
            elif self.dispersion == 'VV10' or self.dispersion == 'NLC':
                print("Built-in VV10 NLC dispersion is on")
                self.mf.nlc='VV10'
                #TODO: Deal with grids
                #self.mf.grids.atom_grid={'H': (99,590),'F': (99,590)}
                #self.mf.grids.prune=None
                #self.mf.nlcgrids.atom_grid={'H': (50,194),'F': (50,194)}
                #self.mf.nlcgrids.prune=dft.gen_grid.sg1_prune

    def set_DF_mf_options(self,Grad=False):
        #https://pyscf.org/user/df.html
        #ASH-default gives PySCF default :optimized JK auxbasis for family if it exists,
        # otherwise an even-tempered basis is generated
        #NOTE: For DF with pure functionals pyscf is using a large JK auxbasis despite only J integrals present
        #More efficient then to specify the : 'def2-universal-jfit' (same as 'weigend') auxbasis
        #Currently left up to user

        #If SGX was selected we do DF for Coulomb and SGX for Exchange (densityfit keyword is ignored)
        if self.sgx is True:
            print("SGX option selected. Will use DF for Coulomb and SGX for Exchange")
            import pyscf.sgx
            self.mf = pyscf.sgx.sgx_fit(self.mf, auxbasis=self.auxbasis)
            self.mf.with_df.dfj = True
            self.mf.with_df.pjs = True
            if Grad is True:
                print("Warning: Gradient calculation requested. This is probably not implemented yet in PySCF.")
                print("PySCF will most likely give an error")
        #If sgx was not selected but densityfit was, we do RI-J if pure-DFT or RI-JK if hybrid-DFT/HF
        elif self.densityfit is True:
            if self.printlevel >= 1:
                print("Density fitting option is on. Turning on in meanfield object!")
            if self.auxbasis != None:
                self.mf = self.mf.density_fit(self.auxbasis)
            else:
                self.mf = self.mf.density_fit()
        else:
            if self.printlevel >= 1:
                print("No density fitting options in use")

    def set_DFT_options(self):
        if self.functional is not None:
            #Setting functional
            self.mf.xc = self.functional
            #TODO: libxc vs. xcfun interface control here
            #mf._numint.libxc = xcfun
            #Grid setting
            self.mf.grids.level = self.gridlevel

    def set_printing_option_mf(self):
        #Verbosity of pySCF mf
        self.mf.verbose = self.verbose_setting

        #Print to stdout or to file
        if self.printsetting is True:
            if self.printlevel >1:
                print("Printing output to stdout...")
            #np.set_printoptions(linewidth=500) TODO: not sure
        else:
            self.mf.stdout = open(self.filename+'.out', 'w')
            if self.printlevel >0:
                print(f"PySCF printing to: {self.filename}.out")

    def set_collinear_option(self):
        print("Collinear option in GHF/GKS is set to:", self.collinear_option)
        self.mf.collinear = self.collinear_option
        #mf._numint.spin_samples = 6
        #Have true by default?
        #TODO: Need to re-enable
        #if self.GHF_complex is True:
        #    print("GHF/GKS complex option True")
        #    dm = self.mf.get_init_guess() + 0j
        #    dm[0,:] += .1j
        #    dm[:,0] -= .1j
        #    scf_result = self.mf.kernel(dm0=dm)

    def set_frozen_core_settings(self, elems):
        print("Setting frozen-core settings")
        #Frozen-core settings
        if self.frozen_core_setting == 'Auto':
            self.determine_frozen_core(elems)
        elif self.frozen_core_setting == None or self.frozen_core_setting == 'None':
            print("Warning: No core-orbitals will be frozen in the CC/MP2 calculation.")
            self.frozen_core_orbital_indices=[]
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
        return self.frozen_orbital_indices

    def set_embedding_options(self, PC=False, MM_coords=None, MMcharges=None):
        import pyscf
        #QM/MM electrostatic embedding
        if PC is True:
            import pyscf.qmmm
            # QM/MM pointcharge embedding
            #TODO: Gaussian blur option
            print("PC True. Adding pointcharges")
            #self.mf = pyscf.qmmm.mm_charge(self.mf, MM_coords, MMcharges)

            #Newer syntax
            mm_mol = pyscf.qmmm.mm_mole.create_mm_mol(MM_coords, MMcharges)

            #Modified pyscf QM/MM routines
            import ash.interfaces.interface_pyXscf_mods
            print("self.platform:", self.platform)
            self.mf = ash.interfaces.interface_pyXscf_mods.qmmm_for_scf(self.mf, mm_mol, platform=self.platform)
            print("Here self.mf:", self.mf)

        #Polarizable embedding option
        elif self.pe is True:
            import pyscf.solvent
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

    def run_BS_SCF(self, mult=None, dm=None):
        print("\nBroken-symmetry SCF procedure")
        print(f"First converging HS mult={self.HSmult} solution")
        #HS: Changing spin to HS-spin (num unpaired els)
        self.mol.spin = self.HSmult-1

        scf_result = self.run_SCF(dm=dm)
        print("High-spin SCF energy:", scf_result.e_tot)
        s2, spinmult = self.mf.spin_square()
        print("UHF/UKS <S**2>:", s2)
        print("UHF/UKS spinmult:", spinmult)

        print(f"\nHS-calculation complete. Now flipping atoms")
        # Flip by spinflipatom string. Example: Flip the local spin of the atom ('0 Fe' in ao_labels)
        #https://pyscf.org/pyscf_api_docs/pyscf.gto.html
        #Note: search_ao_label can take either string or list of strings
        #if self.spinflipatom != None:
        #    print("Spinflipatom option:", self.spinflipatom)
        #    #Here finding first atom of element:, e.g. '0 Fe'
        #    orbindices = self.mol.search_ao_label(self.spinflipatom)
        #    print("orbindices:", orbindices)
        #    orbliststoflip=[orbindices]

        # flip by list of atom indices
        if self.atomstoflip != None:
            print("Atomstoflip option:", self.atomstoflip)
            #Converting atom indices to the spinflipatom syntax above
            spinflipatom = [f"{i} {self.mol.atom_symbol(i)}" for i in self.atomstoflip]
            print("spinflipatom:", spinflipatom)
            orbindices = self.mol.search_ao_label(spinflipatom)
            print("Orbital indices to flip:", orbindices)
            orbliststoflip=[orbindices]
        else:
            print("atomstoflip has not been set. Exiting")
            ashexit()

        #Get alpha and beta density matrices
        dma, dmb = self.mf.make_rdm1()
        #Loop over orbliststoflip to flip all atoms
        for idx_at in orbliststoflip:
            dma_at = dma[idx_at.reshape(-1,1),idx_at].copy()
            dmb_at = dmb[idx_at.reshape(-1,1),idx_at].copy()
            dma[idx_at.reshape(-1,1),idx_at] = dmb_at
            dmb[idx_at.reshape(-1,1),idx_at] = dma_at
        dm = [dma, dmb]
        print(f"\nStarting BS-SCF with spin multiplicity={mult}")
        #BS
        self.mol.spin = mult-1
        #scf_result = self.mf.run(dm)
        scf_result = self.run_SCF(dm=dm)
        s2, spinmult = self.mf.spin_square()
        print("BS SCF energy:", scf_result.e_tot)

        return scf_result

    #Independent method to run SCF using previously defined mf object and possible input dm
    def run_SCF(self,mf=None, dm=None, max_cycle=None):
        import pyscf
        import pyscf.dft
        if self.printlevel >= 1:
            print("\nInside run_SCF")
        module_init_time=time.time()
        if mf is None:
            if self.printlevel >=1:
                print("No mf object provided. Using self.mf")
            if self.mf is None:
                print("No self.mf object defined. Exiting")
                ashexit()
            else:
                mf=self.mf
        if dm is None:
            if self.printlevel >=1:
                print("No dm provided.")
        else:
            if self.printlevel >=1:
                print("DM provided. Will be used")
        #Modify max-cycle in mf object if requested
        if max_cycle is not None:
            mf.max_cycle=max_cycle
        if self.printlevel >=1:
            print("Max cycle in mf object:", mf.max_cycle)
            print("Running SCF")
        print("mf:", mf)
        scf_result = mf.run(dm)
        E_tot = scf_result.e_tot
        if self.printlevel >=1:
            print("SCF done!")
            print("E_tot:", E_tot)
        if self.functional != None:
            E_xc = scf_result.scf_summary["exc"]
            E_dmf = E_tot - E_xc
            if self.printlevel >= 1:
                print("E_dmf:", E_dmf)
                print("E_xc:", E_xc)

        #Setting number of orbitals as attribute of object
        if isinstance(self.mf, pyscf.scf.hf.RHF) or isinstance(self.mf, pyscf.dft.rks.RKS) :
            self.num_orbs = len(self.mf.mo_occ) # Restricted
        else:
            self.num_orbs = len(self.mf.mo_occ[0]) 
            
        if self.printlevel >= 1:
            print("Number of orbitals:", self.num_orbs)

        #Calculating new dm and preserving it
        if self.printlevel >= 1:
            print("Calculating density matrix from converged MO's and storing as dm attribute of PySCFTheory object")
        dm = mf.make_rdm1()
        self.dm=dm

        print_time_rel(module_init_time, modulename='pySCF run_SCF', moduleindex=2, currprintlevel=self.printlevel, currthreshold=2)
        return scf_result

    #General run function to distinguish  possible specialrun (disabled) and mainrun
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None, Hessian=False):

        self.runcalls += 1
        #Note: We have to do prepare_run each time. Mol object (with coords,basis etc.) has to be created and built.
        #Mf object then has to be built from that mol object.
        #Prepare for run (create mol object, mf object, modify mf object etc.)
        #Does not execute SCF, CC or anything
        self.prepare_run(current_coords=current_coords, elems=elems, charge=charge, mult=mult,
                            current_MM_coords=current_MM_coords,
                            MMcharges=MMcharges, qm_elems=qm_elems, Grad=Grad, PC=PC,
                            numcores=numcores, pe=pe, potfile=potfile, restart=restart, label=label)

        #Actual run
        return self.actualrun(current_coords=current_coords, current_MM_coords=current_MM_coords, MMcharges=MMcharges, qm_elems=qm_elems,
        elems=elems, Grad=Grad, PC=PC, numcores=numcores, pe=pe, potfile=potfile, restart=restart, label=label,
        charge=charge, mult=mult, Hessian=Hessian)

    def prepare_run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None):

        module_init_time=time.time()
        if self.printlevel >0:
            print(BC.OKBLUE,BC.BOLD, "------------PREPARING PYSCF INTERFACE-------------", BC.END)
            print("Object-label:", self.label)
            print("Run-label:", label)

        #Load pyscf
        import pyscf
        #Set PySCF threads to numcores
        pyscf.lib.num_threads(self.numcores)
        if self.printlevel >1:
            print("PySCF version:", pyscf.__version__)
            print("Number of PySCF lib threads is:", pyscf.lib.num_threads())
        #Checking environment variables
        try:
            print("os.environ['OMP_NUM_THREADS']:", os.environ['OMP_NUM_THREADS'])
            if os.environ['OMP_NUM_THREADS'] != '1':
                print("Warning: Environment variable OMP_NUM_THREADS should be set to 1. PySCF may not run properly in parallel")
        except:
            pass
        try:
            print("os.environ['MKL_NUM_THREADS']:", os.environ['MKL_NUM_THREADS'])
            if os.environ['MKL_NUM_THREADS'] != '1':
                print("Warning: Environment variable MKL_NUM_THREADS should be set to 1. PySCF may not run properly in parallel")
        except:
            pass

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


        #Setting number of electrons for system (used by load_chkfile etc)
        self.num_electrons  = int(ash.modules.module_coords.nucchargelist(qm_elems) - charge)
        if self.printlevel >= 1:
            print("Number of electrons:", self.num_electrons)
            print()

        #####################
        #CREATE MOL OBJECT
        #####################
        self.create_mol(qm_elems, current_coords, charge, mult, cartesian_basis=self.cartesian_basis)

        #####################
        # BASIS
        #####################
        if self.fcidumpfile is None:
            self.define_basis(elems=qm_elems)
        self.mol.build()
        self.num_basis_functions=len(self.mol.ao_labels())
        if self.printlevel >= 1:
            print("Number of basis functions:", self.num_basis_functions)

        ############################
        # CREATE MF OBJECT
        ############################
        #if self.platform == 'GPU':
        #    print("Platform is GPU")
        #    self.create_mf_for_gpu() #Creates self.mf
        #else:
        if self.fcidumpfile is None:
            self.create_mf() #Creates self.mf
        else:
            print("FCIDUMP file read-in")
            print("Creating mf object from FCIDUMPfile")
            self.read_fcidump_file(self.fcidumpfile)

        #GHF/GKS
        if self.scf_type == 'GHF' or self.scf_type == 'GKS':
            self.set_collinear_option()


        #####################
        # RELATIVITY
        #####################
        #Convert non-relativistic mf object to spin-free x2c if self.x2c is True
        if self.x2c is True:
            print("x2c is True. Changing SCF object to relativistic x2c Hamiltonian")
            self.mf = self.mf.sfx2c1e()

        ###########
        # PRINTING
        ############
        self.set_printing_option_mf()

        #####################
        #DFT
        #####################
        self.set_DFT_options()

        ###################
        #SCF CONVERGENCE
        ###################
        self.set_mf_scfconv_options()

        ###################
        #SMEARING
        ###################
        self.set_mf_smearing()

        ##############
        #DISPERSION
        ##############
        self.set_dispersion_options(Grad=Grad)

        ##############################
        #DENSITY FITTING and SGX
        ##############################
        self.set_DF_mf_options(Grad=Grad)

        ##############################
        #FROZEN ORBITALS in CC
        ##############################
        if self.CC or self.MP2:
            self.set_frozen_core_settings(qm_elems)

        ##############################
        #EMBEDDING OPTIONS
        ##############################
        self.set_embedding_options(PC=PC,MM_coords=current_MM_coords, MMcharges=MMcharges)
        if self.printlevel >1:
            print_time_rel(module_init_time, modulename='pySCF prepare', moduleindex=2)

        #############################
        # PLATFORM CHANGE
        #############################
        #Testing to convert mf object to GPU before QM/MM
        if self.platform == 'GPU':
            print("GPU platform requested. Will now convert mf object to GPU")
            self.mf = self.mf.to_gpu()


    #Actual Run
    #Assumes prepare_run has been executed
    def actualrun(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None,pyscf=None, Hessian=False ):

        module_init_time=time.time()
        #############################################################
        #RUNNING
        #############################################################

        #####################
        #SCF STEP
        #####################
        if self.SCF is True:
            if self.printlevel >1:
                print(f"Running SCF (SCF-type: {self.scf_type})")
            self.mf.verbose=self.verbose_setting+1
            #SCF-GUESS SETUP
            dm = self.setup_guess() #dm is either density matrix from guess-option, chkfile-MO or None

            #Setup Writing CHK file
            self.mf.chkfile = self.write_chkfile_name
            if self.printlevel >1:
                print(f"Will write checkpointfile: {self.mf.chkfile}")


            ###############
            # RUN SCF
            #################
            #BS via 2-step HS-SCF and then spin-flip BS:
            if self.BS is True:
                scf_result = self.run_BS_SCF(mult=mult,dm=dm)

            #Regular single-step SCF:
            else:
                #scf_result = self.mf.run()
                #NOTE: dm needs to have been created here (regardless of the type of guess)
                #scf_result = self.mf.run(dm)
                scf_result = self.run_SCF(dm=dm)

            #Possible stability analysis
            self.run_stability_analysis()
            if self.printlevel >1:
                print("\nSCF energy:", scf_result.e_tot)
                print("SCF energy components:", self.mf.scf_summary)

            #Orbital and Occupation printing
            if self.printlevel >1:
                self.print_orbital_en_and_occ()

            #Possible population analysis (if dm=None then taken from mf object)
            if self.scf_type == 'RHF' or self.scf_type == 'RKS':
                self.num_scf_orbitals_alpha=len(scf_result.mo_occ)
                if self.printlevel >1:
                    print("Total num. orbitals:", self.num_scf_orbitals_alpha)
                if self.printlevel >1:
                    self.run_population_analysis(self.mf, dm=None, unrestricted=False, type='Mulliken', label='SCF')
            elif self.scf_type == 'GHF' or self.scf_type == 'GKS':
                self.num_scf_orbitals_alpha=len(scf_result.mo_occ)
                print("GHF/GKS job")
                print("scf_result:", scf_result)
                if self.printlevel >1:
                    print("Total num. orbitals:", self.num_scf_orbitals_alpha)
                if self.printlevel >1:
                    self.mf.canonicalize(self.mf.mo_coeff, self.mf.mo_occ)
                    self.mf.analyze()
                    #self.run_population_analysis(self.mf, dm=None, unrestricted=False, type='Mulliken', label='SCF')
                    #print("GHF/GKS spinsquare:", pyscf.scf.spin_square(self.mf.mo_coeff, s=None))
                    s2, spinmult = self.mf.spin_square()
                    print("GHF/GKS <S**2>:", s2)
                    print("GHF/GKS spinmult:", spinmult)
            elif self.scf_type == 'ROHF' or self.scf_type == 'ROKS':
                #NOTE: not checked
                self.num_scf_orbitals_alpha=len(scf_result.mo_occ)
                if self.printlevel >1:
                    print("Total num. orbitals:", self.num_scf_orbitals_alpha)
                if self.printlevel >1:
                    self.run_population_analysis(self.mf, dm=None, unrestricted=False, type='Mulliken', label='SCF')
            else:
                #UHF/UKS
                self.num_scf_orbitals_alpha=len(scf_result.mo_occ[0])
                if self.printlevel >1:
                    print("Total num. orbitals:", self.num_scf_orbitals_alpha)
                if self.printlevel >1:
                    self.run_population_analysis(self.mf, dm=None, unrestricted=True, type='Mulliken', label='SCF')
                s2, spinmult = self.mf.spin_square()
                print("UHF/UKS <S**2>:", s2)
                print(f"UHF/UKS spinmult: {spinmult}\n")
            if self.printlevel >=1:
                print("SCF Dipole moment:")
            try:
                self.get_dipole_moment()
            except ValueError as e:
                print("Problem getting dipole moment from meanfield object")
                print("Error message:", e)
                print("Continuing.")
            #Dispersion correction
            if self.dispersion != None:
                if self.dispersion == "D3" or self.dispersion == "D4":
                    with_vdw = self.D3_D4_Disp_object(self.mol, xc=self.functional)
                    vdw_energy = with_vdw.eng
                    vdw_gradient = with_vdw.grad
                    print(f"{self.dispersion} dispersion energy is: {vdw_energy}")
                elif self.dispersion == "VV10" or self.dispersion == "NL":
                    print("Dispersion correction: VV10. No post-SCF step")
                    vdw_energy=0.0
                else:
                    #For TS and MBD it is calculated by the wrapper and already included in thh SCF
                    vdw_energy=0.0 #to avoid double-counting
                    print(f"{self.dispersion} dispersion energy is: {self.mf.e_vdw}")
            else:
                vdw_energy=0.0

            #NMF
            if self.NMF is True:
                E_tot_NMF = self.run_NMF_step()
                #Total energy is SCF energy + possible vdW energy
                self.energy = E_tot_NMF + vdw_energy
            else:
                #Regular
                #Total energy is SCF energy + possible vdW energy
                self.energy = self.mf.e_tot + vdw_energy


            #####################
            #LOSC: Localized orbital scaling
            #####################
            if self.losc is True:
                self.calc_losc()

        #########################################################
        # POST-SCF CALCULATIONS
        #########################################################
        if self.postSCF is True:
            print("postSCF is True")
            #####################
            #MOM
            #####################
            if self.mom is True:
                MOMSCFobj = self.run_MOM()
            #####################
            #TDDFT
            #####################
            #TDDFT (https://pyscf.org/user/tddft.html) in Tamm-Dancoff approximation
            #TODO: transition moments
            # TODO: NTO analysis.
            # TODO: Get density matrix and cube-file per state
            # TODO: Nuclear gradient
            if self.TDDFT is True:
                self.run_TDDFT()
            #####################
            #MP2
            #####################
            if self.MP2 is True:
                print("MP2 is on !")
                MP2_energy,mp2object = self.run_MP2(frozen_orbital_indices=self.frozen_orbital_indices, MP2_DF=self.MP2_DF)
                self.energy = MP2_energy

                if self.MP2_density is True:
                    self.run_MP2_density(mp2object, MP2_DF=self.MP2_DF, DFMP2_density_relaxed=self.DFMP2_density_relaxed)
            #####################
            #COUPLED CLUSTER
            #####################
            if self.CC is True:
                print("Coupled cluster is on !")
                import pyscf.cc as pyscf_cc
                #Default MO coefficients None (unless MP2natorbs option below)
                mo_coefficients=None

                #Optional Frozen natural orbital approach via MP2 natural orbitals
                #NOTE: this is not entirely correct since occupied orbitals are natural orbitals rather than frozen HF orbitals as in the original method
                #Not sure how much it matters
                if self.FNO is True:
                    mo_coefficients = self.setup_FNO(elems=qm_elems)
                #Calling CC run function
                self. CC_energy,self.ccobject = self.run_CC(self.mf,frozen_orbital_indices=self.frozen_orbital_indices, CCmethod=self.CCmethod,
                            CC_direct=self.CC_direct, mo_coefficients=mo_coefficients)
                self.energy = self.CC_energy
                #Calling CC density-run function
                if self.CC_density is True:
                    self.run_CC_density(self.ccobject,self.mf)
            #####################
            #CAS-CI and CASSCF
            #####################
            if self.CAS is True:
                print("CAS run is on!")
                self.run_CAS(elems=qm_elems)
        else:
            if self.printlevel >1:
                print("No post-SCF job.")

        ##############
        #GRADIENT
        ##############
        if Grad:
            if self.printlevel >1:
                print("Gradient requested")

            #postSCF method gradient
            #NOTE: only MOM-SCF for now
            if self.postSCF is True:
                #MOM-SCF
                if self.mom is True:
                    if self.printlevel >1:
                        print("Calculating SCF-MOM gradient")
                    self.gradient = MOMSCFobj.nuc_grad_method().kernel()
                    if self.printlevel >1:
                        print("MOM-SCF Gradient calculation done")
                else:
                    print("Gradient for postSCF methods  is not implemented in ASH interface")
                    #TODO: Enable TDDFT, CASSCF, MP2, CC gradient etc
                    ashexit()
            #Caluclate regular SCF gradient
            else:
                if self.printlevel >1:
                    print("Calculating regular SCF gradient")
                self.gradient = self.mf.nuc_grad_method().kernel()

            #Applying dispersion gradient last
            if self.dispersion != None:
                if self.dispersion == "D3" or self.dispersion == "D4":
                    self.gradient += vdw_gradient
                    if self.printlevel > 1:
                            print("vdw_gradient", vdw_gradient)

            #Pointcharge-gradient (separate)
            if PC is True:
                if self.printlevel >=1:
                    print("Calculating pointcharge gradient")
                #Make density matrix
                checkpoint=time.time()
                dm = self.mf.make_rdm1()
                print_time_rel(checkpoint, modulename='pySCF make_rdm1 for PC', moduleindex=2)
                current_MM_coords_bohr = current_MM_coords*ash.constants.ang2bohr
                checkpoint=time.time()
                self.pcgrad = pyscf_pointcharge_gradient(self.mol,np.array(current_MM_coords_bohr),np.array(MMcharges),dm, GPU=self.GPU_pcgrad)
                print_time_rel(checkpoint, modulename='pyscf_pointcharge_gradient', moduleindex=2)

            if self.printlevel >1:
                print("Gradient calculation done")

        ##############
        #HESSIAN
        ##############
        if Hessian:
            hessinfo = self.run_hessian()
            print("hessinfo:", hessinfo)
            hessian = hessinfo.transpose(0,2,1,3).reshape(3*3,3*3)
            self.hessian=hessian
            try:
                print("Attempting IR intensity calculation (requires pyscf.prop library)")
                
                from pyscf.prop.infrared.rhf import Infrared, kernel_dipderiv
                
            except ModuleNotFoundError:
                print("pyscf IR intensity requires installation of pyscf.prop module")
                print("See: https://github.com/pyscf/properties")
                print("You can install with: pip install git+https://github.com/pyscf/properties")
                ashexit()

            mf_ir = Infrared(self.mf)
            mf_ir.mf_hess=self.hessian_obj
            mf_ir.run()
            #Could run dipole derivatives directly also
            #dipderiv = kernel_dipderiv(mf_ir)
            #print("dipderiv:", dipderiv)
            #mf_ir.summary()
            #mf_ir.ir_inten
            print("mf_ir.ir_inten:", mf_ir.ir_inten)
            self.ir_intensities=mf_ir.ir_inten

        if self.printlevel >= 1:
            print()
            print(BC.OKBLUE, BC.BOLD, "------------ENDING PYSCF INTERFACE-------------", BC.END)
        if Grad == True:
            if self.printlevel >=0:
                print("Single-point PySCF energy:", self.energy)
            print_time_rel(module_init_time, modulename='pySCF actualrun', moduleindex=2)
            if PC is True:
                return self.energy, self.gradient, self.pcgrad
            else:
                return self.energy, self.gradient
        else:
            if self.printlevel >=0:
                print("Single-point PySCF energy:", self.energy)
            print_time_rel(module_init_time, modulename='pySCF actualrun', moduleindex=2)
            return self.energy


#Based on https://github.com/pyscf/pyscf/blob/master/examples/qmmm/30-force_on_mm_particles.py
#Uses pyscf mol and MM coords and charges and provided density matrix to get pointcharge gradient
def pyscf_pointcharge_gradient(mol,mm_coords,mm_charges,dm, GPU=False):
    time0=time.time()
    #Making sure density matrix is as it should
    if dm.shape[0] == 2:
        dmf = np.array(dm[0] + dm[1]) #unrestricted
    else:
        dmf=np.array(dm)

#GPU
    if GPU is True:
        import cupy
        einsumfunc = cupy.einsum
        linalg_norm_func=cupy.linalg.norm

        mm_coords_used=cupy.asarray(mm_coords)
        mm_charges_used=cupy.asarray(mm_charges)
        qm_coords = cupy.asarray(mol.atom_coords())
        qm_charges = cupy.asarray(mol.atom_charges())
        dmf=cupy.asarray(dmf)
        array_mod=cupy.asarray
#CPU
    else:
        def dummy(f): return f
        array_mod=dummy
        einsumfunc=np.einsum
        linalg_norm_func=np.linalg.norm
        mm_coords_used=mm_coords
        mm_charges_used=mm_charges
        qm_coords = mol.atom_coords()
        qm_charges = mol.atom_charges()

    print("Einsumfunc from:", einsumfunc.__module__)
    print("Time for setup 1:", time.time()-time0)
    # The interaction between QM atoms and MM particles
    # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
    dr = qm_coords[:,None,:] - mm_coords_used
    r = linalg_norm_func(dr, axis=2)
    g = einsumfunc('r,R,rRx,rR->Rx', qm_charges, mm_charges_used, dr, r**-3)
    print("Time for setup 2:", time.time()-time0)
    # The interaction between electron density and MM particles
    # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
    #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>

    for i, q in enumerate(mm_charges_used):
        with mol.with_rinv_origin(mm_coords[i]):
            v = array_mod(mol.intor('int1e_iprinv'))
        f =(einsumfunc('ij,xji->x', dmf, v) +
            einsumfunc('ij,xij->x', dmf, v.conj())) * -q
        g[i] += f
    print("Time for setup 4:", time.time()-time0)
    #Converting from Cupy to numpy
    if GPU is True:
        return cupy.asnumpy(g)
    else:
        return g


#Function to do multireference correction via pyscf-based theories: Dice or Block.
# Calculates difference w.r.t CCSD or CCSD(T)
def pyscf_MR_correction(fragment, theory=None, MLmethod='CCSD(T)'):
    print_line_with_mainheader("pyscf_MR_correction")
    print("Multireference correction via pyscf-based theories: Dice or Block. Calculates difference w.r.t CCSD(T)")
    #Checking that correct theory is provided
    if theory == None:
        print("Theory must be provided")
        ashexit()
    elif isinstance(theory,ash.DiceTheory):
        print("DiceTheory object provided")
    elif isinstance(theory,ash.BlockTheory):
        print("BlockTheory object provided")
    else:
        print("Unrecognized theory object provided. Must be DiceTheory or BlockTheory")
        ashexit()

    #Now calling Singlepoint on the HLTheory
    result_HL = ash.Singlepoint(fragment=fragment, theory=theory)

    ###################################
    #Active space CCSD or CCSD(T) via pyscf
    ###################################
    #1. Use exactly the same MO-coefficients (MP2/CC natural orbitals) as used in Dice/Block calculation
    #2. Use exactly same active space
    ###################################
    print("theory.pyscftheoryobject.mf.mo_occ:", theory.pyscftheoryobject.mf.mo_occ)
    if len(theory.pyscftheoryobject.mf.mo_occ) == 2:
        num_orbs = len(theory.pyscftheoryobject.mf.mo_occ[0]) #Assuming Unrestricted
    else:
        num_orbs = len(theory.pyscftheoryobject.mf.mo_occ) #Assuming Restricted
    print("num_orbs:",num_orbs)
    full_list = list(range(0,num_orbs)) #From 0 to last virtual orbital
    print("full_list:", full_list)
    print("Size of full orbital list:", len(full_list))
    act_list=list(range(theory.firstMO_index,theory.lastMO_index+1)) #The range that Dice-SHCI used. Generalize this to DMRGTheory also ?
    print("Size of active-space list:", len(act_list))
    print(act_list)
    frozen_orbital_indices= listdiff(full_list,act_list)
    print("Number of frozen_orbital_indices:", len(frozen_orbital_indices))
    print("Indices:", frozen_orbital_indices)
    mo_coefficients=theory.mch.mo_coeff  #The MO coefficients used by Dice/Block

    #Calling CC PySCF method direct with our mf object and the orbital indices and MO coeffs we want
    CC_energy,CC_object = theory.pyscftheoryobject.run_CC(theory.pyscftheoryobject.mf,frozen_orbital_indices=frozen_orbital_indices,
                                                          CCmethod=MLmethod,
                                                          CC_direct=False, mo_coefficients=mo_coefficients)

    print("\nCC_energy:", CC_energy)
    print("HL energy:",result_HL.energy)
    correction = result_HL.energy - CC_energy
    print(f"\nDelta (HighLevel - {MLmethod}) correction:", correction)

    return correction



#Moldenfile from PySCF checkpointfile
#Also requires geometry (not in chkfile)
def make_molden_file_PySCF_from_chkfile(fragment=None, basis=None, chkfile=None,label=""):
    import pyscf
    from pyscf.tools import molden
    print(f"Attempting to read chkfile: {chkfile}")
    if chkfile != None:
        print("Reading orbitals from checkpointfile:", chkfile)
        if os.path.isfile(chkfile) is False:
            print("File does not exist. Continuing!")
            return False
        try:
            chkfileobject = pyscf.scf.chkfile.load(chkfile, 'scf')
        except TypeError:
            print("No SCF orbitals found. Could be checkpointfile from CASSCF?")
            print("Ignoring and continuing")
            ashexit()
    print("chkfileobject", chkfileobject)
    mo_energy = chkfileobject["mo_energy"]
    mo_occ = chkfileobject["mo_occ"]
    mo_coeff = chkfileobject["mo_coeff"]

    #Creating mol
    mol = pyscf.gto.Mole()
    #Mol system printing. Hardcoding to 3 as otherwise too much PySCF printing
    mol.verbose = 3
    coords_string=ash.modules.module_coords.create_coords_string(fragment.elems,fragment.coords)
    mol.atom = coords_string
    mol.symmetry = None
    mol.charge = fragment.charge
    mol.spin = fragment.mult-1
    mol.basis=basis
    mol.build()

    print("Writing orbitals to disk as Molden file")
    molden.from_mo(mol, f'pyscf_{label}.molden', mo_coeff, occ=mo_occ)
    with open(f'pyscf_{label}.molden', 'w') as f1:
        molden.header(mol, f1)
        molden.orbital_coeff(mol, f1, mo_coeff, ene=mo_energy, occ=mo_occ)


#pySCF CCSD(T) via cheap CCSD natural orbitals from a MP2-natorb selected CCSD-active space
def pyscf_CCSD_T_natorb_selection(fragment=None, pyscftheoryobject=None, numcores=1, thresholds=[1.997,0.02],
        MP2_DF=True, DFMP2_density_relaxed=True, Do_CC_active_space=True, debug=False):

    print_line_with_mainheader("pyscf_natorb_CCSD_T_selection")

    if fragment is None:
        print("Error: No fragment provided to pyscf_natorb_CCSD_T_selection.")
        ashexit()

    if pyscftheoryobject is None:
        print("Error: No pyscftheoryobject object provided to pyscf_natorb_CCSD_T_selection. This is necessary")
        ashexit()

    if pyscftheoryobject.MP2 is True or pyscftheoryobject.CC is True:
        print("Error: pySCFTHeory object already has MP2 or CC turned on. This is not allowed for pyscf_natorb_CCSD_T_selection")
        print("The pySCFTheory object should only contain settings for an SCF mean-field object (basis set, scf_type, functional etc.)")
        ashexit()


    #Use input PySCFTheory object for MF calculation and run
    pyscfcalc = pyscftheoryobject
    result = ash.Singlepoint(fragment=fragment, theory=pyscfcalc) #Run a SP job using object

    #Define frozen core
    frozen_orbital_indices=pyscfcalc.determine_frozen_core(fragment.elems)
    print("frozen_orbital_indices:",frozen_orbital_indices)
    #Run MP2 calculation
    MP2_energy, mp2object = pyscfcalc.run_MP2(frozen_orbital_indices=frozen_orbital_indices, MP2_DF=MP2_DF)
    print("MP2_energy:", MP2_energy)
    print("mp2object:", mp2object)
    #Run MP2 density calculation and get natural orbitals
    MP2_natocc, MP2_natorb, mp2_dm = pyscfcalc.run_MP2_density(mp2object, MP2_DF=MP2_DF, DFMP2_density_relaxed=DFMP2_density_relaxed)
    print("MP2_natocc:", MP2_natocc)

    if Do_CC_active_space is True:
        #Select active space
        full_list = list(range(0,pyscfcalc.num_orbs))
        act_list = ash.select_indices_from_occupations(MP2_natocc,selection_thresholds=thresholds)
        print("Full orbital list:", full_list)
        print("Size of full orbital list:", len(full_list))
        print("Selected active orbital list:", act_list)
        print("Size of selected active-space list:", len(act_list))
        frozen_orbital_indices= listdiff(full_list,act_list)
        print("Number of frozen_orbital_indices:", len(frozen_orbital_indices))
        print("Indices:", frozen_orbital_indices)


        #Small active space CCSD calculation on MP2 natural orbitals
        CC_energy,CC_object = pyscfcalc.run_CC(pyscfcalc.mf,frozen_orbital_indices=frozen_orbital_indices,
                                                                CCmethod='CCSD', CC_direct=False, mo_coefficients=MP2_natorb)

        CC_natocc,CC_natorb,bla = pyscfcalc.calculate_CCSD_natorbs(ccsd=CC_object, mf=pyscfcalc.mf)
        print("CC_natocc:", CC_natocc)

        final_orbs=CC_natorb

        if debug is True:
            print("MP2_natorb:",MP2_natorb)
            print("CC_natorb:", CC_natorb)
            diff = CC_natorb - MP2_natorb
            print("diff:",diff)
    else:
        print("Do_CC_active_space is False. Using MP2 natural orbs instead")
        final_orbs=MP2_natorb
    #Final: CCSD(T) using CCSD-MP2-hybrid natorbs
    normal_frozen_orbital_indices=pyscfcalc.determine_frozen_core(fragment.elems)
    print("normal_frozen_orbital_indices:", normal_frozen_orbital_indices)
    CC_energy,CC_object = pyscfcalc.run_CC(pyscfcalc.mf,frozen_orbital_indices=normal_frozen_orbital_indices,
                                                            CCmethod='CCSD(T)', mo_coefficients=final_orbs)

    result = ASH_Results(label="pyscf_CCSD_T_natorb_selection", energy=CC_energy, charge=fragment.charge, mult=fragment.mult)
    result.write_to_disk(filename="ASH_pyscf.result")
    return result

#Standalone function for reading either pySCF-CHK file or Molden file and returning MO coefficients and occupations
#Used by pySCFTheory, DiceTheory and BlockTheory
def pySCF_read_MOs(moreadfile,pyscfobject):
    import pyscf
    print("Reading MOs from :", moreadfile)
    #Molden read
    if '.molden' in moreadfile:
        print("Warning: This is a Molden file. Will try to read MOs from here but this may not work")
        mol, mo_energy, mo_coefficients, occupations, irrep_labels, spins = pyscf.tools.molden.load(moreadfile)
    #Checkpoint file
    elif '.chk'  in moreadfile:
        mo_coefficients = pyscf.lib.chkfile.load(moreadfile, 'mcscf/mo_coeff')
        occupations = pyscf.lib.chkfile.load(moreadfile, 'mcscf/mo_occ')
    print("Occupations:", occupations)
    print("Length of occupations array:", len(occupations))
    if len(occupations) != pyscfobject.num_orbs:
        print("Occupations array length does NOT match length of MO coefficients in PySCF object")
        print("Is basis different? Exiting")
        ashexit()
    return mo_coefficients, occupations

def pySCF_write_Moldenfile(pyscfobject=None, label="orbs"):
    print("pySCF_write_Moldenfile function\n")
    import pyscf
    from pyscf.tools import molden

    #Early exits
    if pyscfobject is None:
        print("Error: pyscfobject must be provided")
        ashexit()

    mo_coefficients = pyscfobject.mf.mo_coeff
    occupations = pyscfobject.mf.mo_occ
    mo_energies = pyscfobject.mf.mo_energy

    print("Writing orbitals to disk as Molden file")
    molden.from_mo(pyscfobject.mol, f'pyscf_{label}.molden', mo_coefficients, occ=occupations)
    with open(f'{label}.molden', 'w') as f1:
        molden.header(pyscfobject.mol, f1)
        molden.orbital_coeff(pyscfobject.mol, f1, mo_coefficients, ene=mo_energies, occ=occupations)
    return


#Standalone density-potential inversion functions
def KS_inversion_n2v(pyscftheoryobj, dm, method='PDECO', numcores=1, opt_max_iter=200,
                     guide_components="fermi_amaldi", gtol=1e-6):
    time_init=time.time()
    print("\nKS_inversion_kspies: KS density_potential_inversion via n2v")
    try:
        import n2v
        import gbasis
        import pylibxc2
    except ModuleNotFoundError:
        print("ModuleNotFoundError:")
        print("KS_inversion_n2v requires installation of n2v module and additional packages")
        print("See https://github.com/wasserman-group/n2v for details")
        print("""\n#Install pylibxc2
pip install pylibxc2""")
        print("""\n#Install gbasis:
git clone https://github.com/theochem/gbasis.git
cd gbasis
pip install .""")
        print("""\n#Install n2v
git clone https://github.com/wasserman-group/n2v.git
cd n2v
pip install .
""")
        ashexit()

    basis="cc-pVDZ"
    pbs="cc-pVQZ"


    # Extract data for n2v.
    da, db = pyscftheoryobj.mf.make_rdm1()/2, pyscftheoryobj.mf.make_rdm1()/2
    ca, cb = pyscftheoryobj.mf.mo_coeff[:,:pyscftheoryobj.mol.nelec[0]], pyscftheoryobj.mf.mo_coeff[:, :pyscftheoryobj.mol.nelec[1]]
    ea, eb = pyscftheoryobj.mf.mo_energy, pyscftheoryobj.mf.mo_energy

    # Initialize inverter object.
    inv = n2v.Inverter( engine='pyscf' )

    inv.set_system(pyscftheoryobj.mol, basis, pbs=pbs )
    inv.Dt = [da, db]
    inv.ct = [ca, cb]
    inv.et = [ea, eb]

    # Inverter with PDECO method, guide potention v0=Fermi-Amaldi
    inv.v_pbs = np.zeros_like(inv.v_pbs)
    inv.invert(method, opt_max_iter=opt_max_iter, guide_components=guide_components, gtol=gtol)


    #For visualization
    # Build Grid
    inv.eng.grid.build_rectangular((1001,1,1))
    x = inv.eng.grid.x
    #vrest
    vrest = inv.eng.grid.to_grid(inv.v_pbs, grid='rectangular')
    # Get Hartree and Fermi-Amaldi potentials
    vH = inv.eng.grid.hartree(density=da+db, grid='rectangular')
    vFA = (1-1/(inv.nalpha + inv.nbeta)) * vH

    # Build Vxc
    vxc = vFA + vrest - vH


    return vxc

#Takes pyscfheoryobject and DM as input, solves the inversion problem and returns MO coefficients, occupations,energies and new DM
def KS_inversion_kspies(pyscftheoryobj, dm, numcores=1,
                        method='WY', WY_method='trust-exact', pbas=None,
                        ZMP_lambda=128, ZMP_levelshift=True, ZMP_LS_scaling=8,
                        ZMP_cycles=400, guide='faxc',
                        DF=True, vxc_method = 'eval_vh',
                        plot_vxc=False, vxc_coords=None, chosen_axes=None,
                        xlimit=None, ylimit=None,
                        x_axis_in_Bohr=False, plot_all_lambdas=True, plot_format='png'):
    time_init=time.time()
    print("\nKS_inversion_kspies: KS density_potential_inversion via kspies")
    try:
        #
        import kspies
        from kspies import wy, zmp, util
    except ModuleNotFoundError:
        print("density_potential_inversion requires installation of kspies module")
        print("See: https://github.com/ssnam92/KSPies")
        print("Try: pip install kspies   and pip install opt-einsum")
        ashexit()
    import pyscf
    try:
        print("Current OMP_NUM_THREADS:", os.environ['OMP_NUM_THREADS'])
    except:
        pass
    print("Setting OMP_NUM_THREADS to:", numcores)
    os.environ['OMP_NUM_THREADS'] = str(numcores)

    if pyscftheoryobj.scf_type == "RKS" or pyscftheoryobj.scf_type == "RHF" :
        print("Case: RHF/RKS. Checking dm")
        print("dm:", dm)
        print("dm.shape:", dm.shape)
    else:
        print("Case: unrestricted (UHF/UKS).")
        print("dm:", dm)
        if type(dm) is tuple:
            print("dm alpha shape:", dm[0].shape)
        else:
            print("dm shape:", dm.shape)

    #If plotting Vxc
    if plot_vxc is True:
        import matplotlib.pyplot as plt
        print("plot_vxc is True. Setting up grid")
        print("Input coordinate grid in Angstrom. Now converting to Bohrs for kspies")
        vxc_coords = np.array(vxc_coords) #Angstrom
        coords_bohr = np.array(vxc_coords)*1.88972612546 #Bohrs
        print("coords_bohr:", coords_bohr)

        plotdim=1
        axdict={'x':0,'y':1,'z':2}
        if chosen_axes is None:
            print("Error: chosen_axes is None. Not allowed. Exiting ")
            ashexit()
        elif type(chosen_axes) == int:
            print(f"chosen_axes: {chosen_axes} is integer. Assuming index (0 for x, 1 for y, 2 for z)")
            ax=chosen_axes
            print("ax:", ax)
        elif type(chosen_axes) == str:
            print(f"chosen_axes: {chosen_axes} is string. Assuming label (x, y, z)")
            ax=axdict[chosen_axes]
            print("ax:", ax)
        elif type(chosen_axes) == list:
            if len(chosen_axes)==2:
                print("2 axes provided. A 2D plot is requested")
                if type(chosen_axes[0]) == str:
                    ax1 = axdict[chosen_axes[0]]
                    ax2 = axdict[chosen_axes[1]]
                else:
                    ax1 = chosen_axes[0]
                    ax2 = chosen_axes[1]
            elif len(chosen_axes) == 1:
                print("1 axis provided as list. A 1D plot is requested")
                if type(chosen_axes[0]) == str:
                    ax = axdict[chosen_axes[0]]
                else:
                    ax = chosen_axes[0]
        else:
            print("Something is wrong")
            exit()
        if plotdim == 1:
            print("plotdim: 1")
            if x_axis_in_Bohr is True:
                print("X-axis in Bohr")
                x_values = coords_bohr[:, ax] #in Bohr
                coord_unit="a.u"
            else:
                x_values = vxc_coords[:, ax] #in Angstrom
                coord_unit="Å"
            #Empty vxc array
            vxc_values_series = []
            labels_series = []
        else:
            print("plotdim: 2")
            print("not ready")
            #TODO
            ashexit()

    #Using eval_vh for vxc
    def run_eval_vh(pyscftheoryobj, kspiesobj,coords_bohr, dm,l=0):
        dmxc = l*kspiesobj.dm-(l+1./pyscftheoryobj.mol.nelectron)*dm
        vxc = kspies.util.eval_vh(pyscftheoryobj.mol, coords_bohr,dmxc)
        return vxc


    #Density->Potential inversion by kspies
    if method == 'ZMP':
        print("Using ZMP method for density->potential inversion")
        if pyscftheoryobj.scf_type == "RKS" or pyscftheoryobj.scf_type == "RHF" :
            print("SCF-type is restricted. Using RZMP")

            #Checking
            zmp_a = zmp.RZMP(pyscftheoryobj.mol, dm)
        else:
            print("SCF-type is unrestricted. Using UZMP")
            zmp_a = zmp.UZMP(pyscftheoryobj.mol, dm)

        #DF or not
        zmp_a.with_df = DF

        #Guiding potential
        #Note: None or faxc are good options. Artifacts seen with PBE and B3LYP for H2 example
        zmp_a.guide = guide
        print("ZMP Guiding potential:", zmp_a.guide)
        #Run ZMP
        print("Running ZMP with lambda:", ZMP_lambda)
        #Max cycle option
        zmp_a.max_cycle=ZMP_cycles
        print("ZMP Max cycles:", zmp_a.max_cycle)
        print("zmp_a.level_shift:", zmp_a.level_shift)

        if ZMP_levelshift is True:
            print("ZMP_levelshift True. Now running increasing lambda iterations with levelshifting")
            zmp_a.diis_space = 30
            zmp_a.conv_tol_dm = 1e-10
            zmp_a.conv_tol_diis = 1e-7

            for l in [ ZMP_lambda/8, ZMP_lambda/4, ZMP_lambda/2, ZMP_lambda]:
            #for l in [ZMP_lambda/(ZMP_LS_scaling*ZMP_LS_scaling), ZMP_lambda/ZMP_LS_scaling,ZMP_lambda]:
                print("Lambda:", l)
                print("Levelshift (lambda*0.1):", l*0.1)
                zmp_a.level_shift = l*0.1
                zmp_a.zscf(l)

                if plot_vxc is True and plot_all_lambdas is True:
                    #dmxc = l*zmp_a.dm-(l+1./pyscftheoryobj.mol.nelectron)*dm
                    #vxc = kspies.util.eval_vh(pyscftheoryobj.mol, coords_bohr,dmxc)
                    if vxc_method == 'eval_vh':
                        vxc = run_eval_vh(pyscftheoryobj, zmp_a,coords_bohr, dm,l=l)
                        vxc_values_series.append(vxc)
                        labels_series.append(r'$\lambda$='+str(int(l)))
                    else:
                        print("not ready")

            print("Now turning levelshift off")
            zmp_a.level_shift = 0.
        print("Running ZSCF (without levelshift)")
        zmp_a.zscf(ZMP_lambda)

        if plot_vxc is True:
            #dmxc = l*zmp_a.dm-(l+1./pyscftheoryobj.mol.nelectron)*dm
            #vxc = kspies.util.eval_vh(pyscftheoryobj.mol, coords_bohr,dmxc)
            if vxc_method == 'eval_vh':
                vxc = run_eval_vh(pyscftheoryobj, zmp_a,coords_bohr, dm,l=ZMP_lambda)
                vxc_values_series.append(vxc)
                labels_series.append(r'$\lambda$='+str(l)+' (final)')
            else:
                print("not ready")
        #Get final data
        mo_coeff =  zmp_a.mo_coeff
        mo_occ =  zmp_a.mo_occ
        mo_energy =  zmp_a.mo_energy
        final_dm =  zmp_a.dm #not sure
        #P = zmp_a.make_rdm1()
        #print("P:", P)
        #print("final_dm:", final_dm)
        #print(f"Expectation value: {pyscftheoryobj.mf.energy_tot(P)} Eh")
        print(f"Expectation value: {pyscftheoryobj.mf.energy_tot(final_dm)} Eh")

        converged =  bool(zmp_a.converged) #converged or not. converting from np.bool to bool
        if converged is False:
            print("Error: ZMP failed to converge. Exiting")
            print("Probably best to enable or modify ZMP_levelshift ")
            ashexit()
        else:
            print("ZMP converged successfully!")

    elif method == 'WY':
        print("Using WY method for density->potential inversion")

        #Optional different potential basis in WY method
        if pbas is None:
            print("No pbas set. Using same pbas as in pyscftheoryobj")
            pbas = pyscftheoryobj.mol.basis
        else:
            print("pbas keyword set:", pbas)


        if pyscftheoryobj.scf_type == "RKS" or pyscftheoryobj.scf_type == "RHF" :
            mw = wy.RWY(pyscftheoryobj.mol, dm, pbas=pbas)
        else:
            mw = wy.UWY(pyscftheoryobj.mol, dm, pbas=pbas)

        #Possible optimizer switch: trust-exact, bfgs, cg
        #When trust-exact fails (large basis etc) switch to bfgs or cg
        mw.method=WY_method
        print("WY minimizer", mw.method)
        print("Guiding potential:", mw.guide)
        #exit()
        #mw.method = 'bfgs'
        #mw.pbas=pbas
        print("WY pbas:", mw.pbas)
        print("Running WY")
        mw.run()
        mw.info()

        if plot_vxc is True:
            print("dsdfds")
            #Guiding potential
            if vxc_method == 'eval_vh':
                vxc = run_eval_vh(pyscftheoryobj, mw,coords_bohr, dm,l=0)
                vxc_values_series.append(vxc)
                labels_series.append(r'WY'+' (final)')
            else:
                print("other ")
                print("Changing guiding potential to PBE")
                mw.guide = 'pbe'
                mw.tol = 1e-7
                ao2 = pyscf.dft.numint.eval_ao(mw.pmol, coords_bohr) #potential basis values on grid
                vg = kspies.util.eval_vxc(pyscftheoryobj.mol, dm, mw.guide,
                                          coords_bohr, delta=1e-8) #guiding potential on grid

                #Regularized WY
                for eta in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                    mw.reg = eta
                    mw.run()
                    vC = np.einsum('t,rt->r', mw.b, ao2)
                    mw.info()
                    vxc_values_series.append(vg+vC)
                    labels_series.append(r'$\eta$='+str(eta))

        #Get final data
        mo_coeff =  mw.mo_coeff
        mo_occ =  mw.mo_occ
        mo_energy =  mw.mo_energy
        final_dm =  mw.dm #not sure
        converged =  bool(mw.converged) #converged or not. converting from np.bool to bool
        if converged is False:
            print("Error: WY failed to converge. Exiting")
            print("You might try changing WY optimization method (WY_method keyword) to bfgs or cg")
            ashexit()
        else:
            print("WY converged successfully!")
    else:
        print("not ready yet")
        ashexit()



    #FINAL plotting
    if plot_vxc is True:
        print("Now plotting vxc")
        for vxc,label in zip(vxc_values_series, labels_series):
            plt.plot(x_values, vxc, label = label)


        #plt.legend=True
        plt.legend(shadow=True, fontsize='small')
        #plt.figure(figsize=(3,4))
        plt.xlabel(f"x ({coord_unit})")
        plt.ylabel("vxc(r)")
        if xlimit != None:
            plt.xlim(*xlimit)
        if ylimit != None:
            plt.ylim(*ylimit)

        print(f"Saving figure vxc_{method}.{plot_format}")
        plt.savefig(f"vxc_{method}.{plot_format}")

    #Get final data
    print("\nMO properties after inversion")
    pyscftheoryobj.print_orbital_en_and_occ(mo_energies=mo_energy, mo_occupations=mo_occ)
    print()

    print("\n Now returning: mo_occ, mo_energy, mo_coeff, final_dm")

    #Resetting OMP_NUM_THREADS (can mess with pySCF)
    os.environ['OMP_NUM_THREADS'] = str(1)


    print_time_rel(time_init, modulename='density_potential_inversion', moduleindex=2)
    return mo_occ, mo_energy, mo_coeff,final_dm



#Function for evaluating functional error (FE) and density error(DE) from 2 pySCFTheory objects
#ref: Reference energy or density
#FE  = E_DFA[n_ref] - E_exact[n_ref]
#DE  = E_DFA[n_DFA] - E_DFA[n_ref]
#DFA_obj: pyscf object for DFA
#DFA_obj: DFA density matrix
#DFA_DM: Reference density matrix
#REF_E: Reference energy
def DFA_error_analysis(fragment=None, DFA_obj=None, REF_obj=None, DFA_DM=None, REF_DM=None, REF_E=None, DFA_E=None,
                            inversion_method='WY', WY_method='trust-exact', numcores=1,
                            ZMP_lambda=128, ZMP_levelshift=True, ZMP_cycles=400, DF=True):
    print_line_with_mainheader("DFA_error_analysis")

    if fragment is None:
        print("Error: No fragment provided to DFA_error_analysis")
        ashexit()
    if DFA_obj is None:
        print("Error: No DFA_obj provided to DFA_error_analysis")
        ashexit()
    if REF_obj is None:
        print("Error: No REF_obj provided to DFA_error_analysis")
        ashexit()
    if DFA_DM is None:
        print("Warning: No DFA_DM matric provided to DFA_error_analysis")
        print("Now doing single-point calculation using DFA_obj to get DM")
        dfa_result = ash.Singlepoint = ash.Singlepoint(fragment=fragment, theory=DFA_obj)
        DFA_DM = DFA_obj.dm
        DFA_E = dfa_result.energy
    if REF_DM is None:
        print("Warning: No REF_DM matric provided to DFA_error_analysis")
        print("Now doing single-point calculation using REF_obj to get REF_DM")
        ref_result = ash.Singlepoint = ash.Singlepoint(fragment=fragment, theory=REF_obj)
        REF_DM = REF_obj.dm

    if REF_E is None:
        print("Warning: No REF_E (Reference energy) provided to DFA_error_analysis")
        print("Trying to see if we can get it from REF_obj")
        try:
            REF_E = ref_result.energy
        except:
            print("Not possible. Please provide REF_E")
            ashexit()


    print("DFA_obj:", DFA_obj)
    print("REF_obj:", REF_obj)
    print("DFA_DM:", DFA_DM)
    print("REF_DM:", REF_DM)
    print("REF_E:", REF_E)


    #Density potential inversion to get reference potential from reference density
    mo_occ, mo_energy, mo_coeff,ref_DM_inv = KS_inversion_kspies(REF_obj, REF_DM, method=inversion_method,
                                                                         WY_method=WY_method, numcores=numcores,
                                                ZMP_lambda=ZMP_lambda, ZMP_levelshift=ZMP_levelshift, ZMP_cycles=ZMP_cycles, DF=DF)

    #E_DFA[n_ref]
    print("Now running E_DFA using reference density")
    #Not using run_SCF anymore as we may have post-SCF contributions
    DFA_obj.dm=ref_DM_inv
    DFA_obj.scf_maxiter=0
    res = ash.Singlepoint(theory=DFA_obj, fragment=fragment)
    #scf_result_1 = DFA_obj.run_SCF(dm=ref_DM_inv, max_cycle=0)
    E_DFA_nref=res.energy
    print("E_DFA_nref:", E_DFA_nref)
    #E_DFA[n_DFA]
    #Disabled run because we may have post-SCF contributions
    #print("Now running E_DFA using DFA density")
    E_DFA_nDFA=DFA_E
    #scf_result_2 = DFA_obj.run_SCF(dm=DFA_DM)
    #E_DFA_nDFA=scf_result_2.e_tot
    #print("E_DFA_nDFA:", E_DFA_nDFA)

    FE = E_DFA_nref - REF_E
    print(f"FE: {FE} Eh")

    DE = E_DFA_nDFA - E_DFA_nref
    print(f"DE: {DE} Eh")

    return FE, DE
