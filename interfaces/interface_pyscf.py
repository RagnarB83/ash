import time

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader
import ash.modules.module_coords
#import scipy 

#PySCF Theory object.
# TODO: PE: Polarizable embedding (CPPE). Not completely active in PySCF 1.7.1. Bugfix required I think

class PySCFTheory:
    def __init__(self, printsetting=False, printlevel=2, numcores=1, 
                  scf_type=None, basis=None, functional=None, gridlevel=5, 
                  pe=False, potfile='', filename='pyscf', memory=3100, conv_tol=1e-8, verbose_setting=4, 
                  CC=False, CCmethod=None, CC_direct=False, frozen_core_setting='Auto', 
                  frozen_virtuals=None, FNO=False, FNO_thresh=None, checkpointfile=True,
                  PyQMC=False, PyQMC_nconfig=1, PyQMC_method='DMC'):

        self.theorytype="QM"
        print_line_with_mainheader("PySCFTheory initialization")
        #Exit early if no SCF-type
        if scf_type is None:
            print("Error: You must select an scf_type, e.g. 'RHF', 'UHF', 'RKS', 'UKS'")
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
        self.checkpointfile=checkpointfile
        #PyQMC
        self.PyQMC=PyQMC
        self.PyQMC_nconfig=PyQMC_nconfig #integer. number of configurations in guess
        self.PyQMC_method=PyQMC_method # DMC or VMC
        if self.PyQMC is True:
            self.load_pyqmc()

        #Attempting to load pyscf
        self.load_pyscf()
        self.set_numcores(numcores)
        
        #PySCF scratch dir. Todo: Need to adapt
        #print("Setting PySCF scratchdir to ", os.getcwd())

        #Print the options
        print("SCF-type:", self.scf_type)
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
        from pyscf.tools import molden
        self.pyscf_molden=molden
        if self.CC is True:
            from pyscf import cc
            self.pyscf_cc=cc
            from pyscf.mp.dfump2_native import DFMP2
            self.pyscf_dmp2=DFMP2

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
        #NOTE: Simplify
        #Defining pyscf mol object and populating 
        mol = self.pyscf.gto.Mole()
        #Not very verbose system printing
        mol.verbose = 3
        coords_string=ash.modules.module_coords.create_coords_string(qm_elems,current_coords)
        mol.atom = coords_string
        mol.symmetry = 1;  mol.charge = charge; mol.spin = mult-1
        #PYSCF basis object: https://sunqm.github.io/pyscf/tutorial.html
        #Object can be string ('def2-SVP') or a dict with element-specific keys and values
        mol.basis=self.basis
        #Memory settings
        mol.max_memory = self.memory
        #BUILD mol object
        mol.build()
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
            mf = self.pyscf.solvent.PE(self.pyscf.scf.RKS(mol), self.potfile)
        #Regular job
        else:
            if PC is True:
                # QM/MM pointcharge embedding
                #mf = mm_charge(dft.RKS(mol), [(0.5, 0.6, 0.8)], MMcharges)
                if self.scf_type == 'RKS':
                    mf = self.pyscf.qmmm.mm_charge(dft.RKS(mol), current_MM_coords, MMcharges)
                else:
                    print("Error. scf_type other than RKS and PC True not ready")
                    ashexit()
            else:
                if self.scf_type == 'RKS':
                    mf = self.pyscf.scf.RKS(mol)
                elif self.scf_type == 'UKS':
                    mf = self.pyscf.scf.UKS(mol)
                elif self.scf_type == 'RHF':
                    mf = self.pyscf.scf.RHF(mol)
                elif self.scf_type == 'UHF':
                    mf = self.pyscf.scf.UHF(mol)

        #Printing settings.
        if self.printsetting==True:
            print("Printsetting = True. Printing output to stdout...")
            #np.set_printoptions(linewidth=500) TODO: not sure
        else:
            print("Printsetting = False. Printing to:", self.filename )
            mf.stdout = open(self.filename, 'w')

        #DFT
        if self.functional is not None:
            #Setting functional
            mf.xc = self.functional
            #TODO: libxc vs. xcfun interface control here
            #mf._numint.libxc = xcfun

            #Grid setting
            mf.grids.level = self.gridlevel

        
        mf.conv_tol = self.conv_tol
        #Control printing here. TOdo: make variable
        mf.verbose = self.verbose_setting

        #Checkpointfile
        if self.checkpointfile is True:
            mf.chkfile = 'scf.chk'


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
        
        
        ############
        #RUNNING
        ############
        print()
        #COUPLED CLUSTER RUN
        if self.CC is True:
            print("Now running CC job")



            #SCF-part
            print("First running SCF")
            scf_result = mf.run()
            print("SCF energy:", scf_result.e_tot)
            print("SCF energy components:", scf_result.scf_summary)
            if self.scf_type == 'RHF' or self.scf_type == 'RKS':
                num_scf_orbitals_alpha=len(scf_result.mo_occ)
                print("Total num. orbitals:", num_scf_orbitals_alpha)
            else:
                num_scf_orbitals_alpha=len(scf_result.mo_occ[0])
                print("Total num. orbitals:", num_scf_orbitals_alpha)

            #Default MO coefficients None (unless MP2natorbs option below)
            mo_coefficients=None

            #Optional MP2 natural orbitals
            if self.FNO is True:
                print("FNO is True")
                print("MP2 natural orbitals on!")
                print("Will calculate MP2 natural orbitals use as input in CC job")
                #ALTERNATIVE: https://github.com/pyscf/pyscf/issues/466
                #pt = self.pyscf.mp.MP2(mf)
                #mp2_E, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
                # form the one body density matrix
                #rdm1 = pt.make_rdm1()
                #print("rdm1:", rdm1)
                # diagonalize to yield the NOs and NO occupation #s
                #occ, no = scipy.linalg.eigh(rdm1)
                # eigenvalues are sorted in ascending order so reorder
                #occ = occ[::-1]
                #no = no[:, ::-1]
                #print("no:", no)
                #ashexit()
                # MP2 natural occupation numbers and natural orbitals
                natocc, natorb = self.pyscf_dmp2(mf.to_uhf()).make_natorbs()
                print("MP2 natural orbital occupations:", natocc)
                print("Writing MP2 natural orbitals to disk as Molden file")
                self.pyscf_molden.from_mo(mol, 'pyscf_mp2nat.molden', natorb, occ=natocc)


                #Choosing MO-coeffients to be
                if self.scf_type == 'RHF' or self.scf_type == 'RKS':
                    mo_coefficients=natorb              
                else:
                    mo_coefficients=[natorb,natorb]

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
                cc = self.pyscf_cc.CCSD(mf, self.frozen_orbital_indices,mo_coeff=mo_coefficients)
            elif self.scf_type == "UHF":
                cc = self.pyscf_cc.UCCSD(mf,self.frozen_orbital_indices,mo_coeff=mo_coefficients)
                
            elif self.scf_type == "RKS":
                print("Warning: CCSD on top of RKS determinant")
                cc = self.pyscf_cc.CCSD(mf.to_rhf(), self.frozen_orbital_indices,mo_coeff=mo_coefficients)
            elif self.scf_type == "UKS":
                print("Warning: CCSD on top of UKS determinant")
                cc = self.pyscf_cc.UCCSD(mf.to_uhf(),self.frozen_orbital_indices,mo_coeff=mo_coefficients)
            
            #Switch to integral-direct CC if user-requested
            #NOTE: Faster but only possible for small/medium systems
            cc.direct = self.CC_direct
            
            result = cc.run()
            print("Reference energy:", result.e_hf)
            #(T) part
            if self.CCmethod == 'CCSD(T)':
                print("Calculating triples ")
                et = cc.ccsd_t()
                print("Triples energy:", et)
                self.energy = result.e_tot + et
                print("Final CCSD(T) energy:", self.energy)
        #PyQMC
        elif self.PyQMC is True:
            configs = self.pyqmc.initial_guess(mol,self.PyQMC_nconfig)
            wf, to_opt = self.pyqmc.generate_wf(mol,mf)
            pgrad_acc = self.pyqmc.gradient_generator(mol,wf, to_opt)
            wf, optimization_data = self.pyqmc.line_minimization(wf, configs, pgrad_acc)
            #DMC, untested
            if self.PyQMC_method == 'DMC':
                configs, dmc_data = self.pyqmc.rundmc(wf, configs)
            #VMC. untested
            elif self.PyQMC_method == 'VMC':
                #More options possible
                df, configs = vmc(wf,configs)
        
        #SCF RUN
        else:
            scf_result = mf.run()
            #Get SCFenergy as energy
            self.energy = scf_result.e_tot
            print("SCF energy components:", scf_result.scf_summary)
        
        #Grab energy and gradient
        #NOTE: only SCF supported for now
        if Grad==True:
            if PC is True:
                print("THIS IS NOT CONFIRMED TO WORK!!!!!!!!!!!!")
                print("Units need to be checked.")
                hfg = mm_charge_grad(grad.dft.RKS(mf), current_MM_coords, MMcharges)
                #                grad = mf.nuc_grad_method()
                self.gradient = hfg.kernel()
            else:
                grad = mf.nuc_grad_method()
                self.gradient = grad.kernel()


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

