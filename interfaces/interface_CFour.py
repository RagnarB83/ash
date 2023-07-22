import subprocess as sp
import shutil
import os
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, pygrep, print_time_rel
import ash.settings_ash


#CFour Theory object.
class CFourTheory:
    def __init__(self, cfourdir=None, printlevel=2, cfouroptions=None, numcores=1,
                 filename='cfourjob', specialbasis=None, ash_basisfile=None):
        
        #Indicate that this is a QMtheory
        self.theorytype="QM"
        
        self.printlevel=printlevel
        self.numcores=numcores
        self.filename=filename
        
        #Default Cfour settings
        self.basis='SPECIAL' #this is default and preferred
        self.method='CCSD(T)'
        self.memory=4
        self.memory_unit='GB'
        self.reference='RHF'
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
        
        #Printing
        print("BASIS:", self.basis)
        print("CALC:", self.method)
        print("MEMORY:", self.memory)
        print("MEM_UNIT:", self.memory_unit)
        print("REFERENCE:", self.reference)
        print("FROZEN_CORE:", self.frozen_core)
        print("GUESS:", self.guessoption)
        print("PROP:", self.propoption)
        print("CC_PROG:", self.cc_prog)
        print("SCF_CONV:", self.scf_conv)
        print("SCF_MAXCYC:", self.scf_maxcyc)
        print("LINEQ_CONV:", self.lineq_conv)
        print("CC_MAXCYC:", self.cc_maxcyc)
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
            print("cfourdir keyword argument not provided to CfourTheory object. Trying to find xcfour in PATH")
            try:
                self.cfourdir = os.path.dirname(shutil.which('xcfour'))
                print("Found xcfour in path. Setting cfourdir to:", cfourdir)
            except:
                print("Found no xcfour executable in path. Exiting... ")
                ashexit()
        else:
            self.cfourdir = cfourdir

        #Copying ASH basis file to dir if requested
        if ash_basisfile != None:
            #ash_basisfile
            print("Copying ASH basis-file {} from {} to current directory".format(ash_basisfile,ash.settings_ash.ashpath+'/basis-sets/cfour/'))
            shutil.copyfile(ash.settings_ash.ashpath+'/basis-sets/cfour/'+ash_basisfile, 'GENBAS')
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
    def cfour_call(self,Grad=False):
        with open(self.filename+'.out', 'w') as ofile:
            #export CFOUR_NUM_CORES=1
            process = sp.run([self.cfourdir + '/xcfour'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    
        #Running gradient calculation. Produces file GRD containing gradient
        with open("cfour_grad.out", 'w') as gfile:
            process = sp.run([self.cfourdir + '/xvdint'], check=True, stdout=gfile, stderr=gfile, universal_newlines=True)


    
    def cleanup(self):
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
            ashexit()
        return energy
    def cfour_grabgradient(self,file,numatoms):
        atomcount=0
        grab=False
        gradient=np.zeros((numatoms,3))
        with open(file) as f:
            for line in f:
                if '  Molecular gradient norm' in line:
                    grab = False
                if grab is True:
                    if '#' in line:
                        gradient[atomcount,0] = float(line.split()[2])
                        gradient[atomcount,1] = float(line.split()[3])
                        gradient[atomcount,2] = float(line.split()[4])
                        atomcount+=1
                if '                            Molecular gradient' in line:
                    grab=True
        return gradient
        #GRD may contain multiple gradients for different methods
        #with open('GRD') as grdfile:
        #    for i,line in enumerate(grdfile):
        #        if i==0:
        #            numatoms=int(line.split()[0])
        #            gradient=np.zeros((numatoms,3))
        #        if i>0:
        #            gradient[atomcount,0] = float(line.split()[1])
        #            gradient[atomcount,1] = float(line.split()[2])
        #            gradient[atomcount,2] = float(line.split()[3])
        #            atomcount+=1
  
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

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        #Grab energy and gradient
        #TODO: No qm/MM yet. need to check if possible in CFour
        if Grad==True:
            if self.propoption != 'OFF':
                #TODO: Check whether we can avoid this limitation
                print("Warning: Cfour property keyword can not be active when doing gradient. Turning off")
                self.propoption == 'OFF'
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(qm_elems,current_coords):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                print("self.propoption:",self.propoption)
                inpfile.write(f"""*CFOUR(CALC={self.method},BASIS={self.basis},COORD=CARTESIAN,UNITS=ANGSTROM,REF={self.reference},CHARGE={charge}\nMULT={mult},FROZEN_CORE={self.frozen_core},MEM_UNIT={self.memory_unit},MEMORY={self.memory},SCF_MAXCYC={self.scf_maxcyc}\n\
GUESS={self.guessoption},PROP={self.propoption},CC_PROG={self.cc_prog},SCF_CONV={self.scf_conv}\n\
LINEQ_CONV={self.lineq_conv},CC_MAXCYC={self.cc_maxcyc},SYMMETRY={self.symmetry},HFSTABILITY={self.stabilityanalysis},DERIV_LEVEL=1)\n\n""")
                for el in qm_elems:
                    if len(self.specialbasis) > 0:
                        inpfile.write("{}:{}\n".format(el.upper(),self.specialbasis[el]))
                inpfile.write("\n")
            
            #Calling CFour
            self.cfour_call(Grad=True)
            #Grabbing energy and gradient
            self.energy=self.cfour_grabenergy()
            self.S2=self.cfour_grab_spinexpect()
            self.gradient=self.cfour_grabgradient(self.filename+'.out',len(qm_elems))
        else:
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(qm_elems,current_coords):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                inpfile.write(f"""*CFOUR(CALC={self.method},BASIS={self.basis},COORD=CARTESIAN,UNITS=ANGSTROM,REF={self.reference},CHARGE={charge}\nMULT={mult},FROZEN_CORE={self.frozen_core},MEM_UNIT={self.memory_unit},MEMORY={self.memory},SCF_MAXCYC={self.scf_maxcyc}\n\
GUESS={self.guessoption},PROP={self.propoption},CC_PROG={self.cc_prog},SCF_CONV={self.scf_conv}\n\
LINEQ_CONV={self.lineq_conv},CC_MAXCYC={self.cc_maxcyc},SYMMETRY={self.symmetry},HFSTABILITY={self.stabilityanalysis})\n\n""")
                #for specbas in self.specialbasis.items():
                for el in qm_elems:
                    if len(self.specialbasis) > 0:
                        inpfile.write("{}:{}\n".format(el.upper(),self.specialbasis[el]))
                inpfile.write("\n")
            self.cfour_call()
            self.energy=self.cfour_grabenergy()
            self.S2=self.cfour_grab_spinexpect()

        #Full cleanup (except OLDMOS and GRD)
        self.cleanup()

        print(BC.OKBLUE, BC.BOLD, "------------ENDING CFOUR INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point CFour energy:", self.energy)
            print("Single-point CFour gradient:", self.gradient)
            print_time_rel(module_init_time, modulename='CFour run', moduleindex=2)
            return self.energy, self.gradient
        else:
            print("Single-point CFour energy:", self.energy)
            print_time_rel(module_init_time, modulename='CFour run', moduleindex=2)
            return self.energy

