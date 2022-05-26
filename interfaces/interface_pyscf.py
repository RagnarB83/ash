import time

from ash.functions.functions_general import ashexit, BC,print_time_rel
import ash.modules.module_coords

#PySCF Theory object.
#PySCF runmode: Library only
# PE: Polarizable embedding (CPPE). Not completely active in PySCF 1.7.1. Bugfix required I think
class PySCFTheory:
    def __init__(self, printsetting=False, printlevel=2, pyscfbasis='', pyscffunctional='',
                 pe=False, potfile='', filename='pyscf', pyscfmemory=3100, numcores=1):

        #Indicate that this is a QMtheory
        self.theorytype="QM"
        
        #Printlevel
        self.printlevel=printlevel

        self.numcores=numcores

        self.pyscfmemory=pyscfmemory
        self.filename=filename
        self.printsetting=printsetting
        #CPPE Polarizable Embedding options
        self.pe=pe
        #Potfile from user or passed on via QM/MM Theory object ?
        self.potfile=potfile

        self.pyscfbasis=pyscfbasis
        self.pyscffunctional=pyscffunctional
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

        if numcores==None:
            numcores=self.numcores


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


        try:
            import pyscf
        except:
            print(BC.FAIL, "Problem importing pyscf. Make sure pyscf has been installed: pip install pyscf", BC.END)
            ashexit(code=9)
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
        coords_string=ash.modules.module_coords.create_coords_string(qm_elems,current_coords)
        mol.atom = coords_string
        mol.symmetry = 1
        mol.charge = charge
        mol.spin = mult-1
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
                    ashexit()
            except:
                ashexit()

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
            print("Printsetting = False. Printing to:", self.filename+'.out' )
            mf.stdout = open(self.filename+'.out', 'w')


        #TODO: Restart settings for PySCF

        #Controlling OpenMP parallelization.
        lib.num_threads(numcores)

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
            print_time_rel(module_init_time, modulename='pySCF run', moduleindex=2)
            return self.energy, self.gradient
        else:
            print("Single-point PySCF energy:", self.energy)
            print_time_rel(module_init_time, modulename='pySCF run', moduleindex=2)
            return self.energy

