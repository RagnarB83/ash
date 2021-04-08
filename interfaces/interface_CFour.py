import subprocess as sp
from functions_general import BC, pygrep
import shutil
import settings_ash

#CFour Theory object. Fragment object is optional. Used??
class CFourTheory:
    def __init__(self, cfourdir=None,fragment=None, charge=None, mult=None, printlevel=2, cfouroptions=None,nprocs=1,
                 filename='cfourjob',specialbasis=None, ash_basisfile='def2-SVP'):
                 #basis=None, method=None, reference='RHF', frozen_core='ON',
                #memory=3100, , guessoption='MOREAD',propoption='OFF',cc_prog='ECC',scf_conv=10,lineq_conv=10,
                #cc_maxcyc=300,symmetry='OFF',stabilityanalysis='OFF'):
        self.printlevel=printlevel
        self.charge=charge
        self.mult=mult
        self.nprocs=nprocs
        self.filename=filename
        
        #Default Cfour settings
        self.basis='SPECIAL' #this is default and preferred
        self.method='CCSD(T)'
        self.memory=4
        self.memory_unit='GB'
        self.reference='UHF'
        self.frozen_core='ON'
        self.guessoption='MOREAD'
        self.propoption='OFF'
        self.cc_prog='ECC'
        self.scf_conv=12
        self.lineq_conv=10
        self.cc_maxcyc=300
        self.scf_maxcyc=400
        self.symmetry='OFF'
        self.stabilityanalysis='OFF'
        self.specialbasis=[]
        #Overriding default
        #self.basis='SPECIAL' is preferred (element-specific basis definitions) but can be overriden like this
        if 'BASIS' in cfouroptions: self.basis=cfouroptions['BASIS']
        if 'CALC' in cfouroptions: self.method=cfouroptions['CALC']
        if 'MEMORY' in cfouroptions: self.memory=cfouroptions['MEMORY']
        if 'MEM_UNIT' in cfouroptions: self.memory_unit=cfouroptions['MEM_UNIT']
        if 'REFERENCE' in cfouroptions: self.reference=cfouroptions['REFERENCE']
        if 'FROZEN_CORE' in cfouroptions: self.frozen_core=cfouroptions['FROZEN_CORE']
        if 'GUESS' in cfouroptions: self.guessoption=cfouroptions['GUESS']
        if 'PROP' in cfouroptions: self.propoption=cfouroptions['PROP']
        if 'CC_PROG' in cfouroptions: self.cc_prog=cfouroptions['CC_PROG']
        if 'SCF_CONV' in cfouroptions: self.scf_conv=cfouroptions['SCF_CONV']
        if 'SCF_MAXCYC' in cfouroptions: self.scf_maxcyc=cfouroptions['SCF_MAXCYC']
        if 'LINEQ_CONV' in cfouroptions: self.lineq_conv=cfouroptions['LINEQ_CONV']
        if 'CC_MAXCYC' in cfouroptions: self.cc_maxcyc=cfouroptions['CC_MAXCYC']
        if 'SYMMETRY' in cfouroptions: self.symmetry=cfouroptions['SYMMETRY']
        if 'HFSTABILITY' in cfouroptions: self.stabilityanalysis=cfouroptions['HFSTABILITY']        
        
        #Getting special basis dict etc
        if self.basis=='SPECIAL':
            if specialbasis != None:
                #Dictionary of element:basisname entries
                self.specialbasis = specialbasis
            else:
                print("basis option is: SPECIAL (default) but no specialbasis dictionary provided. Please provide this (specialbasis keyword).")
                exit()
        else:
            self.specialbasis=[]
        
        #Copying ASH basis file to dir if requested
        if ash_basisfile != None:
            #ash_basisfile
            print("Copying ASH basis-file {} from {} to current directory".format(ash_basisfile,settings_ash.ashpath+'/basis-sets/cfour/'))
            shutil.copyfile(settings_ash.ashpath+'/basis-sets/cfour/'+ash_basisfile, 'GENBAS')





        if cfourdir == None:
            # Trying to find xcfour in path
            print("cfourdir keyword argument not provided to CfourTheory object. Trying to find xcfour in PATH")
            try:
                self.cfourdir = os.path.dirname(shutil.which('xcfour'))
                print("Found xcfour in path. Setting cfourdir.")
            except:
                print("Found no xcfour executable in path. Exiting... ")
                exit()
        else:
            self.cfourdir = cfourdir
        
        #Clean-up of possible old Cfour files before beginning
        #TODO: Skip cleanup of chosen files?
        self.cfour_clean_full()
        
    def cfour_call(self):
        with open(self.filename+'.out', 'w') as ofile:
            process = sp.run([self.cfourdir + '/xcfour'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    def cfour_clean_full(self):
        print("Cleaning up old Cfour files")
        files=['MOABCD', 'MOINTS', 'JOBARC', 'NEWMOS', 'BASINFO.DATA', 'den.dat', 'DIPOL', 'DPTDIPOL', 'DPTEFG', 'ERREX', 'EFG','FILES', 'GAMLAM', 'IIII', 'JAINDX',
               'NEWFOCK', 'NTOTAL', 'NATMOS', 'MOLDEN', 'MOLDEN_NAT', 'MOLECULE.INP', 'MOL', 'JMOLplot', 'OPTARC', 'THETA', 'VPOUT']
        for file in files:
            try:
                os.remove(file)
            except:
                pass
    def cfour_grabenergy(self):
        #Other things to possibly grab in future:
        #HF-SCF energy
        #CCSD correlation energy
        linetograb="The final electronic energy"
        energystringlist=pygrep(linetograb,self.filename+'.out')
        try:
            energy=float(energystringlist[-2])
        except:
            print("Problem reading energy from Cfour outputfile. Check:", self.filename+'.out')
            exit()
        return energy
    def cfour_grabgradient(self):
        atomcount=0
        with open('GRD') as grdfile:
            for i,line in enumerate(grdfile):
                if i==0:
                    numatoms=int(line.split()[0])
                    gradient=np.zeros((numatoms,3))
                if i>numatoms:
                    gradient[atomcount,0] = float(line.split()[1])
                    gradient[atomcount,1] = float(line.split()[2])
                    gradient[atomcount,2] = float(line.split()[3])
                    atomcount+=1
        return gradient    
    def cfour_grab_spinexpect(self):
        linetograb="Expectation value of <S**2>"
        s2line=pygrep(linetograb,self.filename+'.out')
        try:
            S2=float(s2line[-1][0:-1])
        except:
            S2=None
        return S2
    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, nprocs=None, restart=False):

        if nprocs == None:
            nprocs = self.nprocs

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING CFOUR INTERFACE-------------", BC.END)


        # Coords provided to run or else taken from initialization.
        # if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords = self.coords

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems = self.elems
            else:
                qm_elems = elems

        #Grab energy and gradient
        #TODO: No qm/MM yet. need to check if possible in CFour
        if Grad==True:
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(qm_elems,current_coords):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                inpfile.write("""*CFOUR(CALC={},BASIS={},COORD=CARTESIAN,REF={},CHARGE={}\nMULT={},FROZEN_CORE={},MEM_UNIT={},MEMORY={},SCF_MAXCYC={}\n\
GUESS={},PROP={},CC_PROG={},SCF_CONV={}\n\
LINEQ_CONV={},CC_MAXCYC={},SYMMETRY={},HFSTABILITY={},DERIV_LEVEL=1)\n\n""".format(
                    self.method,self.basis,self.reference,self.charge,self.mult,self.frozen_core,self.memory_unit,self.memory,self.scf_maxcyc,self.guessoption,self.propoption,
                    self.cc_prog,self.scf_conv,self.lineq_conv,self.cc_maxcyc,self.symmetry,self.stabilityanalysis))
                for el in qm_elems:
                    inpfile.write("{}:{}\n".format(el.upper(),self.specialbasis[el]))
                inpfile.write("\n")
            self.cfour_call()
            self.energy=self.cfour_grabenergy()
            self.S2=self.cfour_grab_spinexpect()
            self.gradient=self.cfour_grabgradient()
        else:
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(qm_elems,current_coords):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                inpfile.write("""*CFOUR(CALC={},BASIS={},COORD=CARTESIAN,REF={},CHARGE={}\nMULT={},FROZEN_CORE={},MEM_UNIT={},MEMORY={},SCF_MAXCYC={}\n\
GUESS={},PROP={},CC_PROG={},SCF_CONV={}\n\
LINEQ_CONV={},CC_MAXCYC={},SYMMETRY={},HFSTABILITY={})\n\n""".format(
                    self.method,self.basis,self.reference,self.charge,self.mult,self.frozen_core,self.memory_unit,self.memory,self.scf_maxcyc,self.guessoption,self.propoption,
                    self.cc_prog,self.scf_conv,self.lineq_conv,self.cc_maxcyc,self.symmetry,self.stabilityanalysis))
                #for specbas in self.specialbasis.items():
                for el in qm_elems:
                    if len(self.specialbasis) > 0:
                        inpfile.write("{}:{}\n".format(el.upper(),self.specialbasis[el]))
                inpfile.write("\n")
            self.cfour_call()
            self.energy=self.cfour_grabenergy()
            self.S2=self.cfour_grab_spinexpect()

        #Full cleanup (except OLDMOS and GRD)
        self.cfour_clean_full()

        print(BC.OKBLUE, BC.BOLD, "------------ENDING CFOUR INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point CFour energy:", self.energy)
            print("Single-point CFour gradient:", self.gradient)
            return self.energy, self.gradient
        else:
            print("Single-point CFour energy:", self.energy)
            return self.energy

