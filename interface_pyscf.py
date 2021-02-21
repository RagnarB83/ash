from functions_general import BC
import functions_coords

#PySCF Theory object. Fragment object is optional. Only used for single-points.
#PySCF runmode: Library only
# PE: Polarizable embedding (CPPE). Not completely active in PySCF 1.7.1. Bugfix required I think
class PySCFTheory:
    def __init__(self, fragment='', charge='', mult='', printsetting='False', printlevel=2, pyscfbasis='', pyscffunctional='',
                 pe=False, potfile='', outputname='pyscf.out', pyscfmemory=3100, nprocs=1):


        #Printlevel
        self.printlevel=printlevel

        self.nprocs=nprocs

        self.pyscfmemory=pyscfmemory
        self.outputname=outputname
        self.printsetting=printsetting
        #CPPE Polarizable Embedding options
        self.pe=pe
        #Potfile from user or passed on via QM/MM Theory object ?
        self.potfile=potfile


        if fragment != '':
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge!='':
            self.charge=int(charge)
        if mult!='':
            self.mult=int(mult)
        self.pyscfbasis=pyscfbasis
        self.pyscffunctional=pyscffunctional
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old PySCF files")
        try:
            os.remove('timer.dat')
            os.remove('pyscfoutput.dat')
        except:
            pass
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, nprocs=None, pe=False, potfile=None, restart=False ):

        if nprocs==None:
            nprocs=self.nprocs



        print(BC.OKBLUE,BC.BOLD, "------------RUNNING PYSCF INTERFACE-------------", BC.END)

        #If pe and potfile given as run argument
        if pe is not False:
            self.pe=pe
        if potfile is not None:
            self.potfile=potfile

        #Coords provided to run or else taken from initialization.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems=self.elems
            else:
                qm_elems = elems


        try:
            import pyscf
        except:
            print(BC.FAIL, "Problem importing pyscf. Make sure pyscf has been installed: pip install pyscf", BC.END)
            exit(9)
        #PySCF scratch dir. Todo: Need to adapt
        #print("Setting PySCF scratchdir to ", os.getcwd())

        from pyscf import gto
        from pyscf import scf
        from pyscf import lib
        from pyscf.dft import xcfun
        if self.pe==True:
            import pyscf.solvent as solvent
            from pyscf.solvent import pol_embed
            import cppe

        #Defining mol object
        mol = gto.Mole()
        #Not very verbose system printing
        mol.verbose = 3
        coords_string=functions_coords.create_coords_string(qm_elems,current_coords)
        mol.atom = coords_string
        mol.symmetry = 1
        mol.charge = self.charge
        mol.spin = self.mult-1
        #PYSCF basis object: https://sunqm.github.io/pyscf/tutorial.html
        #Object can be string ('def2-SVP') or a dict with element-specific keys and values
        mol.basis=self.pyscfbasis
        #Memory settings
        mol.max_memory = self.pyscfmemory
        #BUILD mol object
        mol.build()
        if self.pe==True:
            print(BC.OKGREEN, "Polarizable Embedding Option On! Using CPPE module inside PySCF", BC.END)
            print(BC.WARNING, "Potfile: ", self.potfile, BC.END)
            try:
                if os.path.exists(self.potfile):
                    pass
                else:
                    print(BC.FAIL, "Potfile: ", self.potfile, "does not exist!", BC.END)
                    exit()
            except:
                exit()

            # TODO: Adapt to RKS vs. UKS etc.
            mf = solvent.PE(scf.RKS(mol), self.potfile)
        else:

            if PC is True:
                # QM/MM pointcharge embedding
                #mf = mm_charge(dft.RKS(mol), [(0.5, 0.6, 0.8)], MMcharges)
                mf = mm_charge(dft.RKS(mol), current_MM_coords, MMcharges)

            else:
                #TODO: Adapt to RKS vs. UKS etc.
                mf = scf.RKS(mol)
                #Verbose printing. TODO: put somewhere else
            mf.verbose=4


        #Printing settings.
        if self.printsetting==True:
            print("Printsetting = True. Printing output to stdout...")
            #np.set_printoptions(linewidth=500) TODO: not sure
        else:
            print("Printsetting = False. Printing to:", self.outputname )
            mf.stdout = open(self.outputname, 'w')


        #TODO: Restart settings for PySCF

        #Controlling OpenMP parallelization.
        lib.num_threads(nprocs)

        #Setting functional
        mf.xc = self.pyscffunctional
        #TODO: libxc vs. xcfun interface control here
        #mf._numint.libxc = xcfun

        mf.conv_tol = 1e-8
        #Control printing here. TOdo: make variable
        mf.verbose = 4



        #RUN ENERGY job. mf object should have been wrapped by PE or PC here
        result = mf.run()
        self.energy = result.e_tot
        print("SCF energy components:", result.scf_summary)
        
        #if self.pe==True:
        #    print(mf._pol_embed.cppe_state.summary_string)

        #Grab energy and gradient
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
        print(BC.OKBLUE, BC.BOLD, "------------ENDING PYSCF INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point PySCF energy:", self.energy)
            return self.energy, self.gradient
        else:
            print("Single-point PySCF energy:", self.energy)
            return self.energy

