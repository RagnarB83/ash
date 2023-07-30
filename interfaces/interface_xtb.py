import os
import shutil
import numpy as np
import subprocess as sp
import time

import ash.constants
import ash.settings_solvation
import ash.settings_ash
from ash.functions.functions_general import ashexit, blankline,reverse_lines, print_time_rel,BC, print_line_with_mainheader,print_if_level
import ash.modules.module_coords
from ash.modules.module_coords import elemstonuccharges, check_multiplicity, check_charge_mult


#Now supports 2 runmodes: 'library' (fast Python C-API) or 'inputfile'
#TODO: QM/MM pointcharges for library
#TODO: THis should be a general interface so remove settings_solvation calls.
#TODO: xtb. Need to check how compatible threading and multiprocessing of xtb is


class xTBTheory:
    def __init__(self, xtbdir=None, xtbmethod='GFN1', runmode='inputfile', numcores=1, printlevel=2, filename='xtb_',
                 maxiter=500, electronic_temp=300, label=None, accuracy=0.1, hardness_PC=1000, solvent=None):

        self.theorynamelabel="xTB"
        self.theorytype="QM"
        self.analytic_hessian=False

        #Hardness of pointcharge. GAM factor. Big number means PC behaviour
        self.hardness=hardness_PC

        #Accuracy (0.1 it quite tight)
        self.accuracy=accuracy

        #Printlevel
        self.printlevel=printlevel
        
        #Controlling output in xtb-library
        if self.printlevel >= 3:
            self.verbosity = "full" #Full output
        elif self.printlevel == 2:
            self.verbosity = "minimal" #  SCC iterations
        elif self.printlevel <= 1:
            self.verbosity = "muted" #nothing


        #Label to distinguish different xtb objects
        self.label=label


        self.numcores=numcores
        self.filename=filename
        self.xtbmethod=xtbmethod
        self.maxiter=maxiter
        self.runmode=runmode
        
        self.electronic_temp=electronic_temp
        
        print_line_with_mainheader("xTB INTERFACE")
        print("Runmode:", self.runmode)
        print("xTB method:", self.xtbmethod)
        #Parallelization for both library and inputfile runmode
        print("xTB object numcores:", self.numcores)
        #NOTE: Setting OMP_NUM_THREADS should be sufficient for performance. MKL threading should be handled by xTB
        os.environ["OMP_NUM_THREADS"] = str(self.numcores)
        #os.environ["MKL_NUM_THREADS"] = "1"

        #New library version. interface via conda: xtb-python
        if self.runmode=='library':
            print("Using new library-based xTB interface")
            print("Importing xtb-python library")
            try:
                from xtb.interface import Calculator, Param
            except:
                print("Problem importing xTB library. Have you installed : conda install -c conda-forge xtb-python  ?")
                ashexit(code=9)

            # Creating variable and setting to None. Replaced by run
            self.calcobject=None
            print("xTB method:", self.xtbmethod)

            #Creating solvent object if solvent requested
            if solvent != None:
                from xtb.utils import get_solvent, Solvent
                self.solvent_object = get_solvent(solvent)
                if self.solvent_object == None:
                    print("Unknown solvent. Not found in xtb.utils.get_solvent")
                    ashexit()
            else:
                self.solvent_object=None
        #Inputfile
        elif self.runmode=='inputfile':
            if xtbdir == None:
                print(BC.WARNING, "No xtbdir argument passed to xTBTheory. Attempting to find xtbdir variable inside settings_ash", BC.END)
                try:
                    print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                    self.xtbdir=ash.settings_ash.settings_dict["xtbdir"]
                except:
                    print(BC.WARNING,"Found no xtbdir variable in ash.settings_ash module either.",BC.END)
                    try:
                        self.xtbdir = os.path.dirname(shutil.which('xtb'))
                        print(BC.OKGREEN,"Found xtb in path. Setting xtbdir to:", self.xtbdir, BC.END)
                    except:
                        print("Found no xtb executable in path. Exiting... ")
                        ashexit()
            else:
                self.xtbdir = xtbdir

            #Solvent line to be passed to run-call
            if solvent != None:
                self.solvent_line="--alpb {}".format(solvent)
            else:
                self.solvent_line=""

        else:
            print("unknown runmode. exiting")
            ashexit()

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    #Cleanup after run.
    def cleanup(self):
        if self.printlevel >= 2:
            print("Cleaning up old xTB files")
        files=[self.filename + '.xyz',self.filename + '.out','xtbopt.xyz','xtbopt.log','xtbrestart','molden.input','charges','pcgrad','wbo','xtbinput','pcharge','xtbtopo.mol']
        
        for file in files:
            try:
                os.remove(file)
            except:
                pass

    #Do an xTB-Numfreq Hessian instead of ASH optimization. Useful for gas-phase chemistry (avoids too much ASH printout
    def Hessian(self, fragment=None, Hessian=None, numcores=None, label=None, charge=None, mult=None):
        module_init_time=time.time()
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING INTERNAL xTB Hessian-------------", BC.END)

        if fragment == None:
            print("No fragment provided to xTB Hessian. Exiting")
            ashexit()
        else:
            print("Fragment provided to Hessian")
        #
        current_coords=fragment.coords
        elems=fragment.elems

        #Check charge/mult
        charge,mult = check_charge_mult(charge, mult, self.theorytype, fragment, "xTBTheory.Hessian", theory=self)

        if numcores==None:
            numcores=self.numcores


        if self.printlevel >= 2:
            print("Creating inputfile:", self.filename+'.xyz')

        #Check if mult is sensible
        check_multiplicity(elems,charge,mult)
        if self.runmode=='inputfile':
            #Write xyz_file
            ash.modules.module_coords.write_xyzfile(elems, current_coords, self.filename, printlevel=self.printlevel)

            #Run inputfile.
            if self.printlevel >= 2:
                print("------------Running xTB-------------")
                print("Running xtB using {} cores".format(numcores))
                print("...")

            run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, 
                                    Hessian=True, maxiter=self.maxiter, electronic_temp=self.electronic_temp, 
                                    accuracy=self.accuracy, printlevel=self.printlevel, numcores=numcores)
            if self.printlevel >= 2:
                print("------------xTB calculation done-----")

            print("xtb Hessian calculation done")
            hessian = xtbhessiangrab(len(elems))
            print_time_rel(module_init_time, modulename='xtB Hessian-run', moduleindex=2)

            #Also setting Hessian of fragment
            fragment.hessian=hessian
            return hessian
            
        else:
            print("Only runmode='inputfile allowed for xTBTheory.Opt(). Exiting")
            ashexit()

    #Do an xTB-optimization instead of ASH optimization. Useful for gas-phase chemistry (avoids too much ASH printout
    def Opt(self, fragment=None, Grad=None, Hessian=None, numcores=None, label=None, charge=None, mult=None):
        module_init_time=time.time()
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING INTERNAL xTB OPTIMIZATION-------------", BC.END)

        if fragment == None:
            print("No fragment provided to xTB Opt. Exiting")
            ashexit()
        else:
            print("Fragment provided to Opt")
        #
        current_coords=fragment.coords
        elems=fragment.elems

        #Check charge/mult
        charge,mult = check_charge_mult(charge, mult, self.theorytype, fragment, "xTBTheory.Opt", theory=self)

        if numcores==None:
            numcores=self.numcores


        if self.printlevel >= 2:
            print("Creating inputfile:", self.filename+'.xyz')

        #Check if mult is sensible
        check_multiplicity(elems,charge,mult)
        if self.runmode=='inputfile':
            #Write xyz_file
            ash.modules.module_coords.write_xyzfile(elems, current_coords, self.filename, printlevel=self.printlevel)

            #Run inputfile.
            if self.printlevel >= 2:
                print("------------Running xTB-------------")
                print("Running xtB using {} cores".format(numcores))
                print("...")

            run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, 
                                    Opt=True, maxiter=self.maxiter, electronic_temp=self.electronic_temp, 
                                    accuracy=self.accuracy, printlevel=self.printlevel, numcores=numcores)

            if self.printlevel >= 2:
                print("------------xTB calculation done-----")

            print("Grabbing optimized coordinates")
            #Grab optimized coordinates from filename.xyz
            opt_elems,opt_coords = ash.modules.module_coords.read_xyzfile("xtbopt.xyz")
            fragment.replace_coords(fragment.elems,opt_coords)

            #return
            #TODO: Check if xtB properly converged or not 
            #Regardless take coordinates and go on. Possibly abort if xtb completely
        else:
            print("Only runmode='inputfile allowed for xTBTheory.Opt(). Exiting")
            ashexit()
            #Update coordinates in someway
        print("ASH fragment updated:", fragment)
        fragment.print_coords()
        #Writing out fragment file and XYZ file
        fragment.print_system(filename='Fragment-optimized.ygg')
        fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')

        #Printing internal coordinate table
        ash.modules.module_coords.print_internal_coordinate_table(fragment)
        print_time_rel(module_init_time, modulename='xtB Opt-run', moduleindex=2)
        return 


    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, printlevel=None,
                elems=None, Grad=False, PC=False, numcores=None, label=None, charge=None, mult=None):
        module_init_time=time.time()

        if self.runmode == 'library':
            from xtb.interface import Calculator, Param

        if MMcharges is None:
            MMcharges=[]

        if numcores is None:
            numcores=self.numcores

        if self.printlevel >= 2:
            print("------------STARTING XTB INTERFACE-------------")
            print("Object-label:", self.label)
            print("Run-label:", label)
        #Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for xTBTheory.run method", BC.END)
            ashexit()

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        #Since xTB will stupidly run even when number of unp. electrons and num-electrons don't match
        check_multiplicity(qm_elems,charge,mult)
        if self.runmode=='inputfile':
            if self.printlevel >=2:
                print("Using inputfile-based xTB interface")
            #TODO: Add restart function so that xtbrestart is not always deleted
            #Create XYZfile with generic name for xTB to run
            #inputfilename="xtb-inpfile"
            if self.printlevel >= 2:
                print("Creating inputfile:", self.filename+'.xyz')
            num_qmatoms=len(current_coords)
            num_mmatoms=len(MMcharges)

            #self.cleanup()
            #Todo: xtbrestart possibly. needs to be optional
            ash.modules.module_coords.write_xyzfile(qm_elems, current_coords, self.filename,printlevel=self.printlevel)


            #Run inputfile.
            if self.printlevel >= 2:
                print("------------Running xTB-------------")
                print("Running xtB using {} cores".format(self.numcores))
                print("...")
            if Grad==True:
                #print("Grad is True")
                if PC==True:
                    #print("PC is true")
                    create_xtb_pcfile_general(current_MM_coords, MMcharges, hardness=self.hardness)
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, printlevel=self.printlevel,
                                      Grad=True, maxiter=self.maxiter, electronic_temp=self.electronic_temp, accuracy=self.accuracy, numcores=numcores)
                else:
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, maxiter=self.maxiter, printlevel=self.printlevel,
                                  Grad=True, electronic_temp=self.electronic_temp, accuracy=self.accuracy, solvent=self.solvent_line, numcores=numcores)
            else:
                if PC==True:
                    create_xtb_pcfile_general(current_MM_coords, MMcharges, hardness=self.hardness)
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, maxiter=self.maxiter, printlevel=self.printlevel,
                                      electronic_temp=self.electronic_temp, accuracy=self.accuracy, solvent=self.solvent_line, numcores=numcores)
                else:
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, maxiter=self.maxiter, printlevel=self.printlevel,
                                      electronic_temp=self.electronic_temp, accuracy=self.accuracy, solvent=self.solvent_line, numcores=numcores)

            if self.printlevel >= 2:
                print("------------xTB calculation done-----")
            #Check if finished. Grab energy
            if Grad==True:
                self.energy,self.grad=xtbgradientgrab(num_qmatoms)
                if PC==True:
                    # Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                    self.pcgrad = xtbpcgradientgrab(num_mmatoms)
                    if self.printlevel >= 2:
                        print("xtb energy :", self.energy)
                        print("------------ENDING XTB-INTERFACE-------------")
                    print_time_rel(module_init_time, modulename='xTB run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
                    return self.energy, self.grad, self.pcgrad
                else:
                    if self.printlevel >= 2:
                        print("xtb energy :", self.energy)
                        print("------------ENDING XTB-INTERFACE-------------")
                    print_time_rel(module_init_time, modulename='xTB run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
                    return self.energy, self.grad
            else:
                outfile=self.filename+'.out'
                self.energy=xtbfinalenergygrab(outfile)
                if self.printlevel >= 2:
                    print("xtb energy :", self.energy)
                    print("------------ENDING XTB-INTERFACE-------------")
                print_time_rel(module_init_time, modulename='xTB run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
                return self.energy
        
        elif self.runmode =='library':
            if self.printlevel >= 1:
                print("------------Running xTB (library)-------------")
            #Converting Angstroms to Bohr
            coords_au=np.array(current_coords)*ash.constants.ang2bohr
            #Converting element-symbols to nuclear charges
            qm_elems_numbers=np.array(elemstonuccharges(qm_elems))
            assert len(coords_au) == len(qm_elems_numbers)
            #Choosing method
            if self.xtbmethod == 'GFN2':
                if self.printlevel >= 2:
                    print("Using GFN2 parameterization")
                param_method=Param.GFN2xTB
            elif self.xtbmethod == 'GFN1':
                if self.printlevel >= 2:
                    print("Using GFN1 parameterization")
                param_method=Param.GFN1xTB
            elif self.xtbmethod == 'GFN0':
                if self.printlevel >= 2:
                    print("Using GFN0 parameterization")
                param_method=Param.GFN0xTB
            elif self.xtbmethod == 'GFNFF':
                if self.printlevel >= 2:
                    print("Using GFNFF parameterization")
                    print("warning: experimental")
                param_method=Param.GFNFF
            elif self.xtbmethod == 'IPEA':
                if self.printlevel >= 2:
                    print("Using IPEA parameterization")
                param_method=Param.IPEAxTB
            else:
                print("unknown xtbmethod")
                ashexit()

            #Creating calculator using Hamiltonian and coordinates
            #Setting charge and mult

            #first run call: create new object containing coordinates and settings
            if self.calcobject == None:
                print("Creating new xTB calc object")
                self.calcobject = Calculator(param_method, qm_elems_numbers, coords_au, charge=charge, uhf=mult-1)
                self.calcobject.set_verbosity(self.verbosity)
                self.calcobject.set_electronic_temperature(self.electronic_temp)
                self.calcobject.set_max_iterations(self.maxiter)
                self.calcobject.set_accuracy(self.accuracy)
                #Solvent
                if self.solvent_object != None:
                    print("Setting solvent to:", self.solvent_object)
                    self.calcobject.set_solvent(self.solvent_object)
            #next run calls: only update coordinates
            else:
                if self.printlevel >= 2:
                    print("Updating coordinates in xTB calcobject")
                self.calcobject.update(coords_au)

            #QM/MM pointcharge field
            #calc.
            if PC==True:
                if self.printlevel >= 2:
                    print("Using PointCharges")
                mmcharges=np.array(MMcharges)
                #print("Setting external point charges")
                #print("num MM charges", len(MMcharges))
                #print(MMcharges)
                #print("num MM coords", len(current_MM_coords))
                #print(current_MM_coords)
                MMcoords_au=np.array(current_MM_coords)*ash.constants.ang2bohr
                #print(MMcoords_au)
                #NOTE: Are these element nuclear charges or what ?
                numbers=np.array([9999 for i in MMcharges])
                #print("numbers:", numbers)
                self.calcobject.set_external_charges(numbers,mmcharges,MMcoords_au)

            #Run
            #TODO: Can we turn off gradient calculation somewhere?
            if self.printlevel >= 2:
                print("Running xtB using {} cores".format(self.numcores))
            res = self.calcobject.singlepoint()
            if self.printlevel >= 2:
                print("------------xTB calculation done-------------")
            if Grad == True:
                if self.printlevel >= 2:
                    print("Grad is True")
                self.energy = res.get_energy()
                self.grad =res.get_gradient()
                if self.printlevel >= 2:
                    print("xtb energy :", self.energy)
                if PC == True:
                    #pcgrad
                    #get pcgrad
                    print("pc grad is not yet implemented for runmode library ")
                    print("If you were trying to use xtb in a QM/MM object then you have to switch to runmode='inputfile' instead")
                    ashexit()

                    #_gradient = np.zeros((len(self), 3))
                    #_lib.xtb_getGradient(self._env, self._res, _cast("double*", _gradient))
                    #TODO: Create get_PCgradient function in the Python API
                    #TODO: Wait for xtb-python to be updated to use xtb version 6.5.1
                    self.pcgrad =res.get_PCgradient()
                    print("self.pcgrad:", self.pcgrad)
                    print("here")
                    ashexit()
                    print("------------ENDING XTB-INTERFACE-------------")
                    print_time_rel(module_init_time, modulename='xTBlib run', moduleindex=2)
                    return self.energy, self.grad, self.pcgrad
                else:
                    if self.printlevel >= 2:
                        print("------------ENDING XTB-INTERFACE-------------")
                    if self.printlevel >= 2:
                        print_time_rel(module_init_time, modulename='xTBlib run', moduleindex=2)
                    return self.energy, self.grad

            else:
                #NOTE: Gradient has still been calculated but is ignored. Not sure how to turn off
                self.energy = res.get_energy()
                if self.printlevel >= 2:
                    print("xtb energy :", self.energy)
                print("------------ENDING XTB-INTERFACE-------------")
                print_time_rel(module_init_time, modulename='xTBlib run', moduleindex=2)
                return self.energy

#Grab Final single point energy
def xtbfinalenergygrab(file):
    Energy=None
    with open(file) as f:
        for line in f:
            if 'TOTAL ENERGY' in line:
                Energy=float(line.split()[-3])
    return Energy


#Grab Hessian from xtb Hessian file
def xtbhessiangrab(numatoms):
    hessdim=numatoms*3
    hessarray2d=np.zeros((hessdim, hessdim))
    i=0; j=0
    with open('hessian') as f:
        for line in f:
            if '$hessian' not in line:
                l = line.split()
                if j == hessdim:
                    i+=1;j=0
                for val in l:
                    hessarray2d[i,j] = val
                    j+=1
    return hessarray2d
#Grab gradient and energy from gradient file
def xtbgradientgrab(numatoms):
    grab=False
    gradient = np.zeros((numatoms, 3))
    count=0
    #Converting Fortran D exponent to E
    t = 'string'.maketrans('D', 'E')
    #Reading file backwards so adding to gradient backwards too
    row=numatoms-1
    #Read file in reverse
    with open('gradient') as f:
        for line in reverse_lines(f):
            if '  cycle =' in line:
                energy=float(line.split()[6])
                return energy, gradient
            if count==numatoms:
                grab=False
            if grab==True:
                gradient[row] = [float( line.split()[0].translate(t)), float(line.split()[1].translate(t)),
                                 float(line.split()[2].translate(t))]
                count+=1
                row-=1
            if '$end' in line:
                grab=True


def xtbVIPgrab(file):
    with open(file) as f:
        for line in f:
            if 'delta SCC IP (eV):' in line:
                VIP=float(line.split()[-1])
    return VIP

def xtbVEAgrab(file):
    with open(file) as f:
        for line in f:
            if 'delta SCC EA' in line:
                VEA=float(line.split()[-1])
    return VEA

# Run xTB single-point job
def run_xtb_SP_serial(xtbdir, xtbmethod, xyzfile, charge, mult, Grad=False, Opt=False, Hessian=False, maxiter=500, electronic_temp=300, accuracy=0.1, solvent=None, printlevel=2, numcores=1):
    
    if solvent == None:
        solvent_line=""
    else:
        solvent_line=solvent

    basename = xyzfile.split('.')[0]
    uhf=mult-1
    #Writing xtbinputfile to disk so that we use ORCA-style PCfile and embedding
    with open('xtbinput', 'w') as xfile:
        xfile.write('$embedding\n')
        xfile.write('interface=orca\n')
        xfile.write('end\n')

    if 'GFN2' in xtbmethod.upper():
        xtbflag = 2
    elif 'GFN1' in xtbmethod.upper():
        xtbflag = 1
    elif 'GFN0' in xtbmethod.upper():
        xtbflag = 0
    else:
        print("Unknown xtbmethod chosen. Exiting...")
        ashexit()
    
    if Grad==True:
        command_list=[xtbdir + '/xtb', basename+'.xyz', '--gfn', str(xtbflag), '--grad', '--chrg', str(charge), '--uhf', str(uhf), '--iterations', str(maxiter),
                              '--etemp', str(electronic_temp), '--acc', str(accuracy), '--parallel', str(numcores), '--input', 'xtbinput', str(solvent_line)  ]
    elif Opt == True:
        command_list=[xtbdir + '/xtb', basename+'.xyz', '--gfn', str(xtbflag), '--opt', '--chrg', str(charge), '--uhf', str(uhf), '--iterations', str(maxiter),
                              '--etemp', str(electronic_temp), '--acc', str(accuracy), '--parallel', str(numcores), '--input', 'xtbinput', str(solvent_line)  ]    
    elif Hessian == True:
        try:
            os.remove("hessian")
        except:
            pass
        command_list=[xtbdir + '/xtb', basename+'.xyz', '--gfn', str(xtbflag), '--hess', '--chrg', str(charge), '--uhf', str(uhf), '--iterations', str(maxiter),
                              '--etemp', str(electronic_temp), '--acc', str(accuracy), '--parallel', str(numcores), '--input', 'xtbinput', str(solvent_line)  ]    
    else:
        command_list=[xtbdir + '/xtb', basename + '.xyz', '--gfn', str(xtbflag), '--chrg', str(charge), '--uhf', str(uhf), '--iterations', str(maxiter),
                      '--etemp', str(electronic_temp), '--acc', str(accuracy), '--parallel', str(numcores), '--input', 'xtbinput', str(solvent_line)]
    if printlevel > 1:
        print("Running xtb with these arguments:", command_list)

    #Catching errors best we can
    try:
        with open(basename+'.out', 'w') as ofile:
            process = sp.run(command_list, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            if process.returncode == 0:
                print_if_level(f"xTB job succeeded.",printlevel,2)
                return
    except sp.CalledProcessError:
        print("xTB subprocess gave error.")
        if Hessian == True:
            if os.path.exists("hessian"):
                print("Hessian file was still created, ignoring error and continuing.")
                return
            else:
                print("Hessian file was not created. Check xtb output for error")
                ashexit()
        else:
            #Some other error. Restarting without xtbrestart (in case a bad one) and trying again.
            print("Something went wrong with xTB. ")
            #TODO: Check for SCF convergence?
            print("Removing xtbrestart MO-file and trying to run again")
            try:
                os.remove("xtbrestart")
            except FileNotFoundError:
                print("Nof xtbrestart file present")
            shutil.copyfile(basename+'.out', basename+'_firstrun.out')
            try:
                with open(basename+'.out', 'w') as ofile:
                    process = sp.run(command_list, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
                if process.returncode == 0:
                    return
            except:
                print("Still an xtb problem. Exiting. Check xtb outputfile")
                ashexit()
    else:
        print("some other error")
        print("process:", process)
        print("process returncode", process.returncode)
        ashexit()



# Run GFN-xTB single-point job (for multiprocessing execution) for both state A and B (e.g. VIE calc)
#Takes 1 argument: line with xyzfilename and the xtb options.
#Runs inside separate dir
def run_gfnxtb_SPVIE_multiproc(line):
    basename=line.split()[0].split('.')[0]
    xyzfile=line.split()[0]
    #Create dir for snapshot
    os.mkdir(basename)
    os.chdir(basename)
    #Copy xyzfile into it
    shutil.copyfile('../'+xyzfile, './'+xyzfile)
    #Copy pointcharge file into dir as pcharge
    shutil.copyfile('../'+basename+'.pc', './pcharge')
    os.listdir()
    #Silly way of getting arguments from line-string again.
    gfnoption=line.split()[1]
    chargeA=line.split()[2]
    uhfA=line.split()[3]
    chargeB=line.split()[4]
    uhfB=line.split()[5]
    with open(basename+'_StateA.out', 'w') as ofile:
        process = sp.run([ash.settings_solvation.xtbdir + '/xtb', xyzfile, '--gfn', gfnoption, '--chrg', chargeA, '--uhf', uhfA ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    with open(basename+'_StateB.out', 'w') as ofile:
        process = sp.run([ash.settings_solvation.xtbdir + '/xtb', xyzfile, '--gfn', gfnoption, '--chrg', chargeB, '--uhf', uhfB ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    os.chdir('..')

# Run xTB VIP single-point job (for multiprocessing execution)
#Takes 1 argument: line with xyzfilename and the xtb options
#PROBLEM: IPEA option has convergence issues for occasional snapshots.
#DISCOURAGED
def run_xtb_VIP_multiproc(line):
    basename=line.split()[0].split('.')[0]
    xyzfile=line.split()[0]
    #Create dir for snapshot
    os.mkdir(basename)
    os.chdir(basename)
    shutil.copyfile('../'+xyzfile, './'+xyzfile)
    chargeseg1=line.split()[1]
    chargeseg2=line.split()[2]
    uhfseg1=line.split()[3]
    uhfseg2=line.split()[4]
    ipseg=line.split()[5]
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([ash.settings_solvation.xtbdir + '/xtb', xyzfile, chargeseg1, chargeseg2, uhfseg1, uhfseg2, ipseg], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    os.chdir('..')

#Using IPEA-xtB method for IP calculations
def run_xtb_VIP(xyzfile, charge, mult):
    basename = xyzfile.split('.')[0]
    uhf=mult-1
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([ash.settings_solvation.xtbdir + '/xtb', basename+'.xyz', '--vip', '--chrg', str(charge), '--uhf', str(uhf) ], check=True, stdout=ofile, universal_newlines=True)


#def run_inputfile_xtb(xyzfile, xtbmethod, chargeA, multA, chargeB, multB):
#    blankline()
#    print("Launching xTB job in serial")
#    print("Number of CPU cores: ", mp.cpu_count())
#    print("XYZ file:", xyzfiles)
#    run_xTB_SP
#    print("Calculations is done")

#TODO: Deal with pcharge pointcharge file.
def run_inputfiles_in_parallel_xtb(xyzfiles, xtbmethod, chargeA, multA, chargeB, multB):
    import multiprocessing as mp
    blankline()
    NumCoresToUse=ash.settings_solvation.NumCores
    print("Launching xTB jobs in parallel")
    print("OMP_NUM_THREADS:", os.environ['OMP_NUM_THREADS'])
    xTBCoresRestriction = False
    if xTBCoresRestriction==True:
        NumCoresToUse=8
        print("xTBCoresRestriction Active!")
        print("Restricting multiprocessing cores to:", NumCoresToUse)
    print("Number of CPU cores: ", NumCoresToUse)
    blankline()
    print("Number of XYZ files:", len(xyzfiles))
    print("Running snapshots in parallel")
    #Create lines to serve as arguments to run_xtb_SP_multiproc
    inputlines=[]
    uhfA=multA-1
    uhfB=multB-1
    pool = mp.Pool(NumCoresToUse)
    if 'GFN' in xtbmethod.upper():
        print("GFN xTB flag")
        print("Will do 2 calculations for State A and State B")
        print("StateA: Charge: {} Mult: {}".format(chargeA, multA))
        print("StateB: Charge: {} Mult: {}".format(chargeB, multB))
        if 'GFN2' in xtbmethod.upper():
            xtbflag=2
        elif 'GFN1' in xtbmethod.upper():
            xtbflag=1
        elif 'GFN0' in xtbmethod.upper():
            xtbflag=0
        for xyzfile in xyzfiles:
            #Passing line with all info to run_gfnxtb_SPVIE_multiproc. Charge/Mult separated in function
            line="{} {} {} {} {} {}".format(xyzfile, xtbflag, chargeA, uhfA, chargeB, uhfB)
            inputlines.append(line)
        results = pool.map(run_gfnxtb_SPVIE_multiproc, [l for l in inputlines])
    elif 'VIP' or 'VEA' or 'VIPEA' in xtbmethod.upper():
        print("IP/EA option. Will do VIP/VEA calculation")
        if 'VIP' in xtbmethod.upper():
            print("VIP xtB flag!")
            xtbflag='--vip'
        elif 'VEA' in xtbmethod.upper():
            print("VEA xtB flag!")
            xtbflag = '--vea'
        elif 'VIPEA' in xtbmethod.upper():
            print("VIPEA xtB flag!")
            xtbflag = '--vipea'
        for xyzfile in xyzfiles:
            line = "{} --chrg {} --uhf {} {}".format(xyzfile, chargeA, uhfA, xtbflag )
            inputlines.append(line)

        results = pool.map(run_xtb_VIP_multiproc, [l for l in inputlines])

    pool.close()
    print("xTB Calculations are done")


#Create xTB pointcharge file based on provided list of elems and coords (MM region elems and coords) and charges for solvent unit.
#Assuming elems and coords list are in regular order, e.g. for TIP3P waters: O H H O H H etc.
#Using Bohrs for xTB. Will be renamed to pcharge when copied to dir.
#Hardness parameter removes the damping used by xTB.
def create_xtb_pcfile_solvent(name,elems,coords,solventunitcharges,bulkcorr=False):
    #Creating list of pointcharges based on solventunitcharges and number of elements provided
    #Modifying
    pchargelist=solventunitcharges*int(len(elems)/len(solventunitcharges))
    bohr2ang=ash.constants.bohr2ang
    hardness=200
    #https://xtb-docs.readthedocs.io/en/latest/pcem.html
    with open(name+'.pc', 'w') as pcfile:
        pcfile.write(str(len(elems))+'\n')
        for p,c in zip(pchargelist,coords):
            line = "{} {} {} {} {}".format(p, c[0]/bohr2ang, c[1]/bohr2ang, c[2]/bohr2ang, hardness)
            pcfile.write(line+'\n')

#General xtb pointchargefile creation
#Using ORCA-style format: pc-coords in Å
def create_xtb_pcfile_general(coords,pchargelist,hardness=1000):
    bohr2ang=ash.constants.bohr2ang
    #https://xtb-docs.readthedocs.io/en/latest/pcem.html
    with open('pcharge', 'w') as pcfile:
        pcfile.write(str(len(pchargelist))+'\n')
        for p,c in zip(pchargelist,coords):
            line = "{} {} {} {} {}".format(p, c[0], c[1], c[2], hardness)
            pcfile.write(line+'\n')


#Grab pointcharge gradient (Eh/Bohr) from xtb pcgrad file
def xtbpcgradientgrab(numatoms):
    gradient = np.zeros((numatoms, 3))
    with open('pcgrad') as pgradfile:
        for count,line in enumerate(pgradfile):
            val_x=float(line.split()[0])
            val_y = float(line.split()[1])
            val_z = float(line.split()[2])
            #gradient[count-1] = [val_x,val_y,val_z]
            gradient[count] = [val_x,val_y,val_z]
    return gradient

#Grab xTB charges. Assuming default xTB charges that are inside file charges
def grabatomcharges_xTB():
    charges=[]
    with open('charges') as file:
        for line in file:
            charges.append(float(line.split()[0]))
    return charges
