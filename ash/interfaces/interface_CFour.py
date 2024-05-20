import subprocess as sp
import shutil
import os
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, pygrep, print_time_rel,writestringtofile
import ash.settings_ash

CFour_basis_dict={'DZ':'PVDZ', 'TZ':'PVTZ', 'QZ':'PVQZ', '5Z':'PV5Z', 'ADZ':'AUG-PVDZ', 'ATZ':'AUG-PVTZ', 'AQZ':'AUG-PVQZ',
                'A5Z':'AUG-PV5Z'}

#CFour Theory object.
class CFourTheory:
    def __init__(self, cfourdir=None, printlevel=2, cfouroptions=None, numcores=1,
                 filename='cfourjob', specialbasis=None, ash_basisfile=None, basisfile=None, label="CFour",
                 parallelization='MKL', frozen_core_settings='Auto', DBOC=False, clean_cfour_files=True):

        self.theorynamelabel="CFour"
        self.analytic_hessian=True
        #Indicate that this is a QMtheory
        self.theorytype="QM"

        self.label=label
        self.printlevel=printlevel
        self.numcores=numcores
        self.filename=filename

        #Type of parallelization. Options: 'MKL' or 'MPI.
        #MPI not yet implemented.
        self.parallelization=parallelization
        self.clean_cfour_files=clean_cfour_files

        #Default Cfour settings
        self.basis='SPECIAL' #this is default and preferred
        self.CALC='CCSD(T)'
        self.memory=4
        self.memory_unit='GB'
        self.reference='RHF'
        self.frozen_core='ON'
        self.frozen_core_settings=frozen_core_settings #ASH default is 'Auto'
        self.guessoption='MOREAD'
        self.propoption='OFF'
        self.cc_prog='ECC'
        self.ABCDTYPE='AOBASIS'
        self.scf_conv=12
        self.lineq_conv=10
        self.cc_maxcyc=300
        self.scf_maxcyc=400
        self.symmetry='OFF'
        self.stabilityanalysis='OFF'
        self.specialbasis=[]
        self.EXTERN_POT='OFF' #Pointcharge potential off by default
        self.DBOC=DBOC
        self.FIXGEOM='OFF' #Off by default. May be turned on by run-method
        self.BRUECKNER='OFF'
        #Overriding default
        #self.basis='SPECIAL' is preferred (element-specific basis definitions) but can be overriden like this
        if 'BASIS' in cfouroptions: self.basis=cfouroptions['BASIS']
        if 'BRUECKNER' in cfouroptions: self.BRUECKNER=cfouroptions['BRUECKNER']
        if 'CALC' in cfouroptions: self.CALC=cfouroptions['CALC']
        if 'MEMORY' in cfouroptions: self.memory=cfouroptions['MEMORY']
        if 'MEM_UNIT' in cfouroptions: self.memory_unit=cfouroptions['MEM_UNIT']
        if 'REF' in cfouroptions: self.reference=cfouroptions['REF']
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
        if 'ABCDTYPE' in cfouroptions: self.ABCDTYPE=cfouroptions['ABCDTYPE']

        #Changing ABCDTYPE algorithm if not possible
        if self.CALC == 'CCSDT' or self.CALC == 'CCSDTQ' or self.CALC == 'CCSDT(Q)':
            if self.ABCDTYPE == 'AOBASIS':
                print("Warning: ABCDTYPE=AOBASIS not possible for higher-order CC (CCSDT and beyond). Changing to ABCDTYPE=STANDARD")
                self.ABCDTYPE='STANDARD'

        #Printing
        print("BASIS:", self.basis)
        print("CALC:", self.CALC)
        print("MEMORY:", self.memory)
        print("MEM_UNIT:", self.memory_unit)
        print("REFERENCE:", self.reference)
        print("FROZEN_CORE:", self.frozen_core)
        print("GUESS:", self.guessoption)
        print("PROP:", self.propoption)
        print("CC_PROG:", self.cc_prog)
        print("ABCDTYPE:", self.ABCDTYPE)
        print("SCF_CONV:", self.scf_conv)
        print("SCF_MAXCYC:", self.scf_maxcyc)
        print("LINEQ_CONV:", self.lineq_conv)
        print("CC_MAXCYC:", self.cc_maxcyc)
        print("BRUECKNER:",self.BRUECKNER)
        print("SYMMETRY:", self.symmetry)
        print("HFSTABILITY:", self.stabilityanalysis)


        #Getting special basis dict etc
        if self.basis=='SPECIAL':
            if specialbasis != None:
                #Dictionary of element:basisname entries
                self.specialbasis = specialbasis
            else:
                print("basis option is: SPECIAL (default) but no specialbasis dictionary provided. Please provide this (specialbasis keyword).")
                ashexit()
        else:
            self.specialbasis=[]


        if cfourdir == None:
            # Trying to find xcfour in path
            print("cfourdir keyword argument not provided to CFourTheory object. Trying to find xcfour in PATH")
            try:
                self.cfourdir = os.path.dirname(shutil.which('xcfour'))
                print("Found xcfour in path. Setting cfourdir to:", self.cfourdir)
            except:
                print("Found no xcfour executable in path. Exiting... ")
                ashexit()
        else:
            self.cfourdir = cfourdir

        #Copying ASH basis file from ASH-dir to current dir if requested
        if ash_basisfile != None:
            #ash_basisfile
            print("Copying ASH basis-file {} from {} to current directory".format(ash_basisfile,ash.settings_ash.ashpath+'/databases/basis-sets/cfour/'))
            shutil.copyfile(ash.settings_ash.ashpath+'/databases/basis-sets/cfour/'+ash_basisfile, 'GENBAS')
        #Copying basis-file from any dir to current dir
        elif basisfile != None:
            print(f"Copying basis-file {basisfile} to current directory as GENBAS")
            shutil.copyfile(basisfile, 'GENBAS')
        else:
            print("No ASH basis-file provided. Copying GENBAS from CFour directory.")
            try:
                shutil.copyfile(self.cfourdir+'/../basis/GENBAS', 'GENBAS')
            except shutil.SameFileError:
                pass
            try:
                shutil.copyfile(self.cfourdir+'/../basis/ECPDATA', 'ECPDATA')
            except shutil.SameFileError:
                pass



        #Clean-up of possible old Cfour files before beginning
        #TODO: Skip cleanup of chosen files?
        self.cleanup()
    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cfour_call(self):
        print("Calling CFour via xcfour executable")
        with open(self.filename+'.out', 'w') as ofile:
            if self.parallelization == 'MKL':
                print(f"MKL parallelization is active. Using MKL_NUM_THREADS={self.numcores}")
                os.environ['MKL_NUM_THREADS'] = str(self.numcores)
                process = sp.run([f"{self.cfourdir}/xcfour"], env=os.environ, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            elif self.parallelization == 'MPI':
                print(f"MPI parallelization active. Will use {self.numcores} MPI processes. (OMP and MKL disabled)")
                print("Note. Assumes Cfour compilation with MPI support with CFOUR_NUM_CORES variable used.")
                os.environ['MKL_NUM_THREADS'] = str(1)
                os.environ['OMP_NUM_THREADS'] = str(1)
                process = sp.run([f"{self.cfourdir}/xcfour"], env=os.environ, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)


    def cleanup(self):
        print("Cleaning up old CFOUR files")
        #Problematic, since it removes by globbing
        #print("Cleaning up old CFour files using xwipeout")
        #sp.run([self.cfourdir + '/xwipeout'])
        files=["THETA", "NTOTAL", "MOINTS", "NATMOS", "DIPOL", "FILES", "IIII", "JMOLplot", "MOABCD", "JOBARC", "OPTARC", "NEWMOS", "EFG", "BASINFO.DATA", "VPOUT", "OLDMOS", "NEWFOCK", "MOL",
"JAINDX", "GAMLAM", "ECPDATA", "den.dat"]
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
            ashexit()
        return energy
    def cfour_grabgradient(self,file,numatoms,symmetry=False):
        atomcount=0
        grab=False
        gradient=np.zeros((numatoms,3))
        with open(file) as f:
            for line in f:
                if '  Molecular gradient norm' in line:
                    grab = False
                if grab is True:
                    if '#' in line:
                        if 'x' not in line:
                            if 'y' not in line:
                                gradient[atomcount,0] = float(line.split()[-3])
                                gradient[atomcount,1] = float(line.split()[-2])
                                gradient[atomcount,2] = float(line.split()[-1])
                                atomcount+=1
                if '                            Molecular gradient' in line:
                    grab=True
        return gradient
    def cfour_grabPCgradient(self,file,numpcs):
        pccount=0
        grab=False
        pcgradient=np.zeros((numpcs,3))
        with open(file) as f:
            for line in f:
                if '  Molecular gradient norm' in line:
                    grab = False
                if grab is True:
                    if 'XP' in line:
                            pcgradient[pccount,0] = float(line.split()[-3])
                            pcgradient[pccount,1] = float(line.split()[-2])
                            pcgradient[pccount,2] = float(line.split()[-1])
                            pccount+=1
                if '                            Molecular gradient' in line:
                    grab=True
        return pcgradient

    def cfour_grabhessian(self,numatoms,hessfile="FCMFINAL"):
        hessdim=3*numatoms
        hessian=np.zeros((hessdim,hessdim))
        i=0; j=0
        with open(hessfile) as f:
            for num,line in enumerate(f):
                if num > 0:
                    l = line.split()
                    if j == hessdim:
                        i+=1;j=0
                    for val in l:
                        hessian[i,j] = val
                        j+=1
        return hessian
    def cfour_grab_spinexpect(self):
        linetograb="Expectation value of <S**2>"
        s2line=pygrep(linetograb,self.filename+'.out')
        try:
            S2=float(s2line[-1][0:-1])
        except:
            S2=None
        return S2

    #Determines Frozen core seetings to apply
    def determine_frozen_core(self,elems):
        print("Determining frozen core")
        print("frozen_core_settings options are: Auto, None or CFour")
        print("Auto uses ASH frozen core settings (mimics ORCA settings)")
        print("CFour uses default CFour frozen core settings (not good for 3d metals)")
        #Frozen core settings
        FC_elems={'H':0,'He':0,'Li':0,'Be':0,'B':2,'C':2,'N':2,'O':2,'F':2,'Ne':2,
        'Na':2,'Mg':2,'Al':10,'Si':10,'P':10,'S':10,'Cl':10,'Ar':10,
        'K':10,'Ca':10,'Sc':10,'Ti':10,'V':10,'Cr':10,'Mn':10,'Fe':10,'Co':10,'Ni':10,'Cu':10,'Zn':10,
        'Ga':18,'Ge':18,'As':18,'Se':18, 'Br':18, 'Kr':18}

        if self.frozen_core == 'OFF' or self.frozen_core == None or self.frozen_core_settings == None:
            print("Frozen core requested OFF. CFour will run all-electron calculations")
            self.frozencore_string=f"FROZEN_CORE=OFF"
        else:
            print("Frozen core is ON!")
            if self.frozen_core_settings == 'Auto':
                print("Auto frozen-core settings requested")
                num_el=0
                for el in elems:
                    num_el+=FC_elems[el]
                frozen_core_el=num_el
                frozen_core_orbs=int(num_el/2)
                print("Total frozen electrons in system:", frozen_core_el)
                print("Total frozen orbitals in system:", frozen_core_orbs)
                self.frozencore_string=f"DROPMO=1>{frozen_core_orbs}"
            elif self.frozen_core_settings == 'CFour':
                print("CFour settings requested")
                self.frozencore_string=f"FROZEN_CORE=ON"
            else:
                print("Unknown option for frozen_core_settings")
                ashexit()

    # Method to grab dipole moment from a CFour outputfile (assumes run has been executed)
    def get_dipole_moment(self):
        return grab_dipole_moment(self.filename+'.out')
    # Method to grab polarizability tensor from a CFour outputfile (assumes run has been executed)
    def get_polarizability_tensor(self):
        return grab_polarizability_tensor(self.filename+'.out')
    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            Hessian=False, DBOC=False,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None, charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING CFOUR INTERFACE-------------", BC.END)

        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for CFourTheory.run", BC.END)
            ashexit()

        #Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        if self.DBOC is True:
            DBOC=True


        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        if PC is True:
            self.EXTERN_POT='ON'
            #Turning symmetry off
            self.symmetry='OFF'
            print("Warning: PC=True. FIXGEOM turned on")
            self.FIXGEOM='ON'

            #Create pcharge file
            with open("pcharges", "w") as pfile:
                pfile.write(f"{len(MMcharges)}\n")
                for mmcharge,mmcoord in zip(MMcharges,current_MM_coords):
                    pfile.write(f"{ash.constants.ang2bohr*mmcoord[0]} {ash.constants.ang2bohr*mmcoord[1]} {ash.constants.ang2bohr*mmcoord[2]} {mmcharge}\n")

        #FROZEN CORE SETTINGS
        self.determine_frozen_core(qm_elems)

        #Grab energy and gradient
        #HESSIAN JOB
        if Hessian is True:
            print("CFour Hessian calculation on!")
            print("Warning: Hessian=True FIXGEOM turned on.")
            self.FIXGEOM='ON'
            print("Warning: Hessian=True, symmetry turned off.")
            self.symmetry='OFF'

            if self.propoption != 'OFF':
            #    #TODO: Check whether we can avoid this limitation
                print("Warning: Cfour property keyword can not be active when doing Hessian. Turning off")
                self.propoption = 'OFF'
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(qm_elems,current_coords):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                inpfile.write(f"""*CFOUR(CALC={self.CALC},BASIS={self.basis},COORD=CARTESIAN,UNITS=ANGSTROM\n\
REF={self.reference},CHARGE={charge},MULT={mult},{self.frozencore_string}\n\
MEM_UNIT={self.memory_unit},MEMORY={self.memory},SCF_MAXCYC={self.scf_maxcyc}\n\
GUESS={self.guessoption},PROP={self.propoption},CC_PROG={self.cc_prog},ABCDTYPE={self.ABCDTYPE}\n\
SCF_CONV={self.scf_conv},EXTERN_POT={self.EXTERN_POT},FIXGEOM={self.FIXGEOM}\n\
LINEQ_CONV={self.lineq_conv},CC_MAXCYC={self.cc_maxcyc},BRUECKNER={self.BRUECKNER},SYMMETRY={self.symmetry}\n
HFSTABILITY={self.stabilityanalysis},VIB=ANALYTIC)\n\n""")
                for el in qm_elems:
                    if len(self.specialbasis) > 0:
                        inpfile.write("{}:{}\n".format(el.upper(),self.specialbasis[el]))
                inpfile.write("\n")
            #Calling CFour
            self.cfour_call()
            self.energy=self.cfour_grabenergy()
            print("Reading CFour Hessian from file")
            self.hessian = self.cfour_grabhessian(len(qm_elems),hessfile="FCMFINAL")
        #ENERGY+GRADIENT JOB
        elif Grad==True:
            print("Warning: Grad=True. FIXGEOM turned on.")
            self.FIXGEOM='ON'
            print("Warning: Grad=True, symmetry turned off.")
            self.symmetry='OFF'

            if self.propoption != 'OFF':
                #TODO: Check whether we can avoid this limitation
                print("Warning: Cfour property keyword can not be active when doing gradient. Turning off")
                self.propoption = 'OFF'
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(qm_elems,current_coords):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                inpfile.write(f"""*CFOUR(CALC={self.CALC},BASIS={self.basis},COORD=CARTESIAN,UNITS=ANGSTROM\n\
REF={self.reference},CHARGE={charge},MULT={mult},{self.frozencore_string}\n\
MEM_UNIT={self.memory_unit},MEMORY={self.memory},SCF_MAXCYC={self.scf_maxcyc}\n\
GUESS={self.guessoption},PROP={self.propoption},CC_PROG={self.cc_prog},ABCDTYPE={self.ABCDTYPE}\n\
SCF_CONV={self.scf_conv},EXTERN_POT={self.EXTERN_POT},FIXGEOM={self.FIXGEOM}\n\
LINEQ_CONV={self.lineq_conv},CC_MAXCYC={self.cc_maxcyc},BRUECKNER={self.BRUECKNER},SYMMETRY={self.symmetry}\n\
HFSTABILITY={self.stabilityanalysis},DERIV_LEVEL=1)\n\n""")
                for el in qm_elems:
                    if len(self.specialbasis) > 0:
                        inpfile.write("{}:{}\n".format(el.upper(),self.specialbasis[el]))
                inpfile.write("\n")

            #Calling CFour
            self.cfour_call()
            #Grabbing energy and gradient
            self.energy=self.cfour_grabenergy()
            self.S2=self.cfour_grab_spinexpect()
            self.gradient=self.cfour_grabgradient(self.filename+'.out',len(qm_elems))
            #PCgradient
            if PC is True:
                self.pcgradient = self.cfour_grabPCgradient(self.filename+'.out',len(MMcharges))

        #DIAGONAL BORN-OPPENHEIMER JOB
        elif DBOC is True:
            if self.propoption != 'OFF':
                print("Warning: Cfour property keyword can not be active when doing DBOC calculation. Turning off")
                self.propoption = 'OFF'
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(qm_elems,current_coords):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')

                inpfile.write(f"""*CFOUR(CALC={self.CALC},BASIS={self.basis},COORD=CARTESIAN,UNITS=ANGSTROM\n\
REF={self.reference},CHARGE={charge},MULT={mult},{self.frozencore_string}\n\
MEM_UNIT={self.memory_unit},MEMORY={self.memory},SCF_MAXCYC={self.scf_maxcyc}\n\
GUESS={self.guessoption},PROP={self.propoption},CC_PROG={self.cc_prog},ABCDTYPE={self.ABCDTYPE}\n\
SCF_CONV={self.scf_conv},EXTERN_POT={self.EXTERN_POT},FIXGEOM={self.FIXGEOM}\n\
LINEQ_CONV={self.lineq_conv},CC_MAXCYC={self.cc_maxcyc},BRUECKNER={self.BRUECKNER},SYMMETRY={self.symmetry}\n\
HFSTABILITY={self.stabilityanalysis},DBOC=ON)\n\n""")
                #for specbas in self.specialbasis.items():
                for el in qm_elems:
                    if len(self.specialbasis) > 0:
                        inpfile.write("{}:{}\n".format(el.upper(),self.specialbasis[el]))
                inpfile.write("\n")
            self.cfour_call()
            self.energy=self.cfour_grabenergy()
            self.S2=self.cfour_grab_spinexpect()
        #ENERGY JOB
        else:
            if self.propoption != 'OFF':
                print("Warning: density requested. FIXGEOM turned on to prevent orientation change")
                print("Also EXTERN_POT turned on to mimic dummy PC-job for same reason")
                self.FIXGEOM='ON'
                self.EXTERN_POT='ON'
                #Write dummy PC file to disk
                writestringtofile("0", "pcharges")
            else:
                self.FIXGEOM='OFF'
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(qm_elems,current_coords):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                inpfile.write(f"""*CFOUR(CALC={self.CALC},BASIS={self.basis},COORD=CARTESIAN,UNITS=ANGSTROM\n\
REF={self.reference},CHARGE={charge},MULT={mult},{self.frozencore_string}\n\
MEM_UNIT={self.memory_unit},MEMORY={self.memory},SCF_MAXCYC={self.scf_maxcyc}\n\
GUESS={self.guessoption},PROP={self.propoption},CC_PROG={self.cc_prog},ABCDTYPE={self.ABCDTYPE}\n\
SCF_CONV={self.scf_conv},EXTERN_POT={self.EXTERN_POT},FIXGEOM={self.FIXGEOM}\n\
LINEQ_CONV={self.lineq_conv},CC_MAXCYC={self.cc_maxcyc},BRUECKNER={self.BRUECKNER},SYMMETRY={self.symmetry}\n\
HFSTABILITY={self.stabilityanalysis})\n\n""")
                #for specbas in self.specialbasis.items():
                for el in qm_elems:
                    if len(self.specialbasis) > 0:
                        inpfile.write("{}:{}\n".format(el.upper(),self.specialbasis[el]))
                inpfile.write("\n")
            self.cfour_call()
            self.energy=self.cfour_grabenergy()
            self.S2=self.cfour_grab_spinexpect()

        #Full cleanup
        if self.clean_cfour_files is True:
            print("clean_cfour_files is True")
            self.cleanup()

        print(BC.OKBLUE, BC.BOLD, "------------ENDING CFOUR INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point CFour energy:", self.energy)
            print("Single-point CFour gradient:", self.gradient)
            print_time_rel(module_init_time, modulename='CFour run', moduleindex=2)
            if PC is True:
                return self.energy, self.gradient, self.pcgradient
            else:
                return self.energy, self.gradient
        else:
            print("Single-point CFour energy:", self.energy)
            print_time_rel(module_init_time, modulename='CFour run', moduleindex=2)
            return self.energy

#CFour DBOC correction on fragment. Either provide CFourTheory object or use default settings
# Either provide fragment or provide coords and elems
def run_CFour_DBOC_correction(coords=None, elems=None, charge=None, mult=None, method='CCSD',basis='TZ',
                              fragment=None, theory=None, openshell=False, numcores=1):
    init_time = time.time()
    if fragment is None:
        fragment = ash.Fragment(coords=coords, elems=elems, charge=charge,mult=mult)
    if openshell is True:
        ref='UHF'
    print("\nNow running CFour DBOC correction")
    #CFour Theory
    cfouroptions = {
    'CALC':method,
    'BASIS':CFour_basis_dict[basis],
    'REF':'RHF',
    'FROZEN_CORE':'ON',
    'MEM_UNIT':'MB',
    'MEMORY':3100,
    'CC_PROG':'ECC',
    'SCF_CONV':10,
    'LINEQ_CONV':10,
    'CC_MAXCYC':300,
    'SYMMETRY':'OFF',
    'HFSTABILITY':'OFF',
    }
    if theory is None:
        theory = CFourTheory(cfouroptions=cfouroptions, DBOC=True,numcores=numcores)
    else:
        theory.DBOC=True
    theory.cleanup()
    ash.Singlepoint(theory=theory, fragment=fragment)
    theory.cleanup()

    dboc_correction = None
    with open(theory.filename+'.out', 'r') as outfile:
        for line in outfile:
            if 'The total diagonal Born-Oppenheimer correction (DBOC) is:' in line:
                if 'a.u.' in line:
                    dboc_correction = float(line.split()[-2])
    print("Diagonal Born-Oppenheimer correction (DBOC):", dboc_correction, "au")
    print_time_rel(init_time, modulename='run_CFour_DBOC_correction', moduleindex=2)
    return dboc_correction


#CFour HLC correction on fragment. Either provide CFourTheory object or use default settings
# Calculates HLC - CCSD(T) correction, e.g. CCSDT - CCSD(T) energy
# Either use fragment or provide coordinates and elements
def run_CFour_HLC_correction(coords=None, elems=None, charge=None, mult=None, fragment=None,theory=None, method='CCSDT',
                             basis='TZ', ref='RHF', openshell=False, numcores=1, cc_prog='VCC', abcdtype='AOBASIS'):
    init_time = time.time()
    if fragment is None:
        fragment = ash.Fragment(coords=coords, elems=elems, charge=charge,mult=mult)
    if openshell is True:
        ref='UHF'
    print("\nNow running CFour HLC correction")
    #CFour Theory
    cfouroptions = {
    'CALC': method,
    'BASIS':CFour_basis_dict[basis],
    'REF':ref,
    'FROZEN_CORE':'ON',
    'MEM_UNIT':'MB',
    'MEMORY':3100,
    'CC_PROG':cc_prog,
    'ABCDTYPE':abcdtype,
    'SCF_CONV':10,
    'LINEQ_CONV':10,
    'CC_MAXCYC':300,
    'SYMMETRY':'OFF',
    'HFSTABILITY':'OFF',
    }
    if theory is None:
        #High-level calc
        theory = CFourTheory(cfouroptions=cfouroptions, numcores=numcores, filename='CFour_HLC_HL')
        print("Now running CFour HLC calculation with", method, "method and", basis, "basis")
        result_HL = ash.Singlepoint(theory=theory,fragment=fragment)
        theory.cleanup()
        #CCSD(T) calc
        theory.method='CCSD(T)'
        theory.filename='CFour_HLC_ccsd_t'
        print("Changing method in CFourTheory object to CCSD(T)")
        print("Now running CFour CCSD(T) calculation")
        result_ccsd_t = ash.Singlepoint(theory=theory,fragment=fragment)
        theory.cleanup()

        delta_corr = result_HL.energy - result_ccsd_t.energy

        print("High-level CFour CCSD(T)-> Highlevel correction:", delta_corr, "au")
    else:
        #Running HL calculation provided
        theory.cleanup()
        theory.filename='CFour_HLC_HL'

        print("Now running CFour HLC calculation")
        result_big = ash.Singlepoint(theory=theory,fragment=fragment)
        theory.cleanup()
        #Changing CALC level to CCSD(T)
        theory.method='CCSD(T)'
        theory.filename='CFour_HLC_ccsd_t'
        print("Changing method in CFourTheory object to CCSD(T)")
        print("Now running CFour CCSD(T) calculation on fragment")
        result_ccsd_t = ash.Singlepoint(theory=theory,fragment=fragment)
        theory.cleanup()

        delta_corr = result_big.energy - result_ccsd_t.energy
        print("High-level CFour CCSD(T) -> Highlevel correction:", delta_corr, "au")
    print_time_rel(init_time, modulename='run_CFour_HLC_correction', moduleindex=2)
    return delta_corr

#Function to create a correct Molden file from CFour
#CFour creates both MOLDEN (SCF) and MOLDEN_NAT (corr WF)
#Issue: CFour Molden format is non-standard (skips beta orbitals, normalization and prints Cartesian d-functions etc)
#This ugly function uses molden2aim to do the conversion
#TODO: Do this in ASH directly instead of using molden2aim at some point
def convert_CFour_Molden_file(moldenfile, molden2aimdir=None, printlevel=2):
    print("convert_CFour_Molden_file")

    moldenfile_basename=os.path.basename(moldenfile).split('.')[0]

    #Finding molden2aim in PATH. Present in ASH (May require compilation)
    ashpath=os.path.dirname(ash.__file__)
    molden2aim=ashpath+"/external/Molden2AIM/src/"+"molden2aim.exe"
    if os.path.isfile(molden2aim) is False:
        print("Did not find {}. Did you compile it ? ".format(molden2aim))
        print("Go into dir:", ashpath+"/external/Molden2AIM/src")
        print("Compile using gfortran or ifort:")
        print("gfortran -O3 edflib.f90 edflib-pbe0.f90 molden2aim.f90 -o molden2aim.exe")
        print("ifort -O3 edflib.f90 edflib-pbe0.f90 molden2aim.f90 -o molden2aim.exe")
        ashexit()
    else:
        print("Found molden2aim.exe: ", molden2aim)

    #Write configuration file for molden2aim
    with open("m2a.ini", 'w') as m2afile:
        string = """########################################################################
    #  In the following 8 parameters,
    #     >0:  always performs the operation without asking the user
    #     =0:  asks the user whether to perform the operation
    #     <0:  always neglect the operation without asking the user
    molden= 1           ! Generating a standard Molden file in Cart. function
    wfn= -1              ! Generating a WFN file
    wfncheck= -1         ! Checking normalization for WFN
    wfx= -1              ! Generating a WFX file (not implemented)
    wfxcheck= -1         ! Checking normalization for WFX (not implemented)
    nbo= -1              ! Generating a NBO .47 file
    nbocheck= -1         ! Checking normalization for NBO's .47
    wbo= -1              ! GWBO after the .47 file being generated

    ########################################################################
    #  Which quantum chemistry program is used to generate the MOLDEN file?
    #  1: ORCA, 2: CFOUR, 3: TURBOMOLE, 4: JAGUAR (not supported),
    #  5: ACES2, 6: MOLCAS, 7: PSI4, 8: MRCC, 9: NBO 6 (> ver. 2014),
    #  0: other programs, or read [Program] xxx from MOLDEN.
    #
    #  If non-zero value is given, [Program] xxx in MOLDEN will be ignored.
    #
    program=2

    ########################################################################
    #  For ECP: read core information from Molden file
    #<=0: if the total_occupation_number is smaller than the total_Za, ask
    #     the user whether to read core information
    # >0: always search and read core information
    rdcore=0

    ########################################################################
    #  Which orbirals will be printed in the WFN/WFX file?
    # =0: print only the orbitals with occ. number > 5.0d-8
    # <0: print only the orbitals with occ. number > 0.1 (debug only)
    # >0: print all the orbitals
    iallmo=0

    ########################################################################
    #  Used for WFX only
    # =0: print "UNKNOWN" for Energy and Virial Ratio
    # .ne. 0: print 0.0 for Energy and 2.0 for Virial Ratio
    unknown=1

    ########################################################################
    #  Print supporting information or not
    # =0: print; .ne. 0: do not print
    nosupp=0

    ########################################################################
    #  The following parameters are used only for debugging.
    clear=1            ! delete temporary files (1) or not (0)

    ########################################################################
    """
        m2afile.write(string)


    #Write Molden2aim input file
    mol2aiminput=['', moldenfile, '', '']
    m2aimfile = open("mol2aim.inp", "w")
    for mline in mol2aiminput:
        m2aimfile.write(mline+'\n')
    m2aimfile.close()

    #Run molden2aim
    m2aimfile = open('mol2aim.inp')
    p = sp.Popen(molden2aim, stdin=m2aimfile, stderr=sp.STDOUT)
    p.wait()

    print(f"Created new Molden file (via molden2aim): {moldenfile_basename}_new.molden")
    print("This file can be correctly read by Multiwfn")



#CFour dipole moment output when density requested but not gradient
def grab_dipole_moment_density_job(outfile):
    dipole_moment = []
    grab=False
    with open(outfile) as f:
        for line in f:
            if grab is True:
                if 'Components of second moment' in line:
                    grab=False
                if '        X =' in line:
                    dipole_moment.append(float(line.split()[2]))
                    dipole_moment.append(float(line.split()[5]))
                    dipole_moment.append(float(line.split()[8]))
            if 'Properties computed from the correlated density matrix' in line:
                grab=True
    return dipole_moment


# For CFour engrad job
def grab_dipole_moment(outfile):
    dipole_moment = []
    grab=False
    with open(outfile) as f:
        for line in f:
            if grab is True:
                if '  Conversion factor used:' in line:
                    grab=False
                if ' x ' in line:
                    dipole_moment.append(float(line.split()[1]))
                if ' y ' in line:
                    dipole_moment.append(float(line.split()[1]))
                if ' z ' in line:
                    dipole_moment.append(float(line.split()[1]))
            if ' Total dipole moment' in line:
                grab=True
    return dipole_moment

# For CFour engrad job with PROP=2
#HF, CCSD, CCSD(T) polarizability
#NOTE: Not well tested. might requires ECC (VCC and NCC fail)
def grab_polarizability_tensor(outfile):
    pz_tensor = np.zeros((3,3))
    grab=False
    count=0
    with open(outfile) as f:
        for line in f:
            if grab is True:
                if ' HF-SCF Polarizability' in line:
                    return pz_tensor
                if len(line.split()) == 4:
                    pz_tensor[count,0]=float(line.split()[1])
                    pz_tensor[count,1]=float(line.split()[2])
                    pz_tensor[count,2]=float(line.split()[3])
                    count+=1
            #CC polarizability output
            if 'Polarizability Tensor' in line:
                if 'CCSD' in line:
                    grab=True
    return pz_tensor
