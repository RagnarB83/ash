import os
import shutil
import numpy as np
import subprocess as sp
import time

import constants
import settings_solvation
import settings_ash
from functions.functions_general import ashexit, blankline,reverse_lines, print_time_rel,BC, print_line_with_mainheader
import modules.module_coords
from modules.module_coords import elemstonuccharges, check_multiplicity, check_charge_mult


#Now supports 2 runmodes: 'library' (fast Python C-API) or 'inputfile'
#
#TODO: THis should be a general interface so remove settings_solvation calls.
#TODO: xtb. Need to combine OMP-parallelization of xtb and multiprocessing if possible


class xTBTheory:
    def __init__(self, xtbdir=None, xtbmethod='GFN1', runmode='inputfile', numcores=1, printlevel=2, filename='xtb_',
                 maxiter=500, electronic_temp=300, label=None, accuracy=0.1, hardness_PC=1000, solvent=None):

        #Indicate that this is a QMtheory
        self.theorytype="QM"

        #SOlvent
        #TODO: Only available for inputfile interface right now
        if solvent != None:
            self.solvent_line="--alpb {}".format(solvent)
        else:
            self.solvent_line=""
        #Hardness of pointcharge. GAM factor. Big number means PC behaviour
        self.hardness=hardness_PC

        #Accuracy (0.1 it quite tight)
        self.accuracy=accuracy

        #Printlevel
        self.printlevel=printlevel
        self.verbosity=printlevel-1
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
            self.Calculator=Calculator
            self.Param=Param

            # Creating variable and setting to None. Replaced by run
            self.calcobject=None
            print("xTB method:", self.xtbmethod)
        #OLD library.
        elif self.runmode=='oldlibrary':
            print("Using old library-based xTB interface")
            print("Loading library...")
            # Load xtB library and ctypes datatypes that run uses
            try:
                #import xtb_interface_library
                import interfaces.interface_xtb_library
                self.xtbobject = interfaces.interface_xtb_library.XTBLibrary()
            except:
                print("Problem importing xTB library. Check that the library dir (containing libxtb.so) is available in LD_LIBRARY_PATH.")
                print("e.g. export LD_LIBRARY_PATH=/path/to/xtb_6.X.X/lib64:$LD_LIBRARY_PATH")
                print("Or that the MKL library is available and loaded")
                ashexit(code=9)
            from ctypes import c_int, c_double
            #Needed for complete interface?:
            # from ctypes import Structure, c_int, c_double, c_bool, c_char_p, c_char, POINTER, cdll, CDLL
            self.c_int = c_int
            self.c_double = c_double
        elif self.runmode=='inputfile':
            if xtbdir == None:
                print(BC.WARNING, "No xtbdir argument passed to xTBTheory. Attempting to find xtbdir variable inside settings_ash", BC.END)
                try:
                    print("settings_ash.settings_dict:", settings_ash.settings_dict)
                    self.xtbdir=settings_ash.settings_dict["xtbdir"]
                except:
                    print(BC.WARNING,"Found no xtbdir variable in settings_ash module either.",BC.END)
                    try:
                        self.xtbdir = os.path.dirname(shutil.which('xtb'))
                        print(BC.OKGREEN,"Found xtb in path. Setting xtbdir to:", self.xtbdir, BC.END)
                    except:
                        print("Found no xtb executable in path. Exiting... ")
                        ashexit()
            else:
                self.xtbdir = xtbdir
        else:
            print("unknown runmode. exiting")
            ashexit()

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
            modules.module_coords.write_xyzfile(elems, current_coords, self.filename, printlevel=self.printlevel)

            #Run inputfile.
            if self.printlevel >= 2:
                print("------------Running xTB-------------")
                print("Running xtB using {} cores".format(numcores))
                print("...")

            
            run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, 
                                      Opt=True, maxiter=self.maxiter, electronic_temp=self.electronic_temp, accuracy=self.accuracy)

            if self.printlevel >= 2:
                print("------------xTB calculation done-----")

            print("Grabbing optimized coordinates")
            #Grab optimized coordinates from filename.xyz
            opt_elems,opt_coords = modules.module_coords.read_xyzfile("xtbopt.xyz")
            fragment.replace_coords(fragment.elems,opt_coords)

            return
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
        modules.module_coords.print_internal_coordinate_table(fragment)
        print_time_rel(module_init_time, modulename='xtB Opt-run', moduleindex=2)
        return 


    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, printlevel=None,
                elems=None, Grad=False, PC=False, numcores=None, label=None, charge=None, mult=None):
        module_init_time=time.time()


        # #Verbosity change. May be changed in run (e.g. by Numfreq)
        # if printlevel != None:
        #     if printlevel< 2:
        #         self.verbosity=0
        #         print("setting verb to 0")
        #     else:
        #         self.verbosity=1
        # else:
        #     self.verbosity=self.printlevel-1

        if MMcharges is None:
            MMcharges=[]

        if numcores is None:
            numcores=self.numcores

        if self.printlevel >= 2:
            print("------------STARTING XTB INTERFACE-------------")

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
        #we wil this little test here
        timeA=time.time()
        check_multiplicity(qm_elems,charge,mult)
        print_time_rel(timeA, modulename='check_multiplicity', moduleindex=2)
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
            modules.module_coords.write_xyzfile(qm_elems, current_coords, self.filename,printlevel=self.printlevel)


            #Run inputfile.
            if self.printlevel >= 2:
                print("------------Running xTB-------------")
                print("Running xtB using {} cores".format(self.numcores))
                print("...")
            if Grad==True:
                print("Grad is True")
                if PC==True:
                    print("PC is true")
                    create_xtb_pcfile_general(current_MM_coords, MMcharges, hardness=self.hardness)
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, 
                                      Grad=True, maxiter=self.maxiter, electronic_temp=self.electronic_temp, accuracy=self.accuracy)
                else:
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, maxiter=self.maxiter,
                                  Grad=True, electronic_temp=self.electronic_temp, accuracy=self.accuracy, solvent=self.solvent_line)
            else:
                if PC==True:
                    create_xtb_pcfile_general(current_MM_coords, MMcharges, hardness=self.hardness)
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, maxiter=self.maxiter,
                                      electronic_temp=self.electronic_temp, accuracy=self.accuracy, solvent=self.solvent_line)
                else:
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, maxiter=self.maxiter,
                                      electronic_temp=self.electronic_temp, accuracy=self.accuracy, solvent=self.solvent_line)

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
                    print_time_rel(module_init_time, modulename='xTB run', moduleindex=2)
                    return self.energy, self.grad, self.pcgrad
                else:
                    if self.printlevel >= 2:
                        print("xtb energy :", self.energy)
                        print("------------ENDING XTB-INTERFACE-------------")
                    print_time_rel(module_init_time, modulename='xTB run', moduleindex=2)
                    return self.energy, self.grad
            else:
                outfile=self.filename+'.out'
                self.energy=xtbfinalenergygrab(outfile)
                if self.printlevel >= 2:
                    print("xtb energy :", self.energy)
                    print("------------ENDING XTB-INTERFACE-------------")
                print_time_rel(module_init_time, modulename='xTB run', moduleindex=2)
                return self.energy
        
        elif self.runmode =='library':
            print("------------Running xTB (library)-------------")
            #Converting Angstroms to Bohr
            coords_au=np.array(current_coords)*constants.ang2bohr
            #Converting element-symbols to nuclear charges
            qm_elems_numbers=np.array(elemstonuccharges(qm_elems))
            assert len(coords_au) == len(qm_elems_numbers)
            print("Number of xTB atoms:", len(coords_au))
            #Choosing method
            if self.xtbmethod == 'GFN2':
                print("Using GFN2 parameterization")
                param_method=self.Param.GFN2xTB
            elif self.xtbmethod == 'GFN1':
                print("Using GFN1 parameterization")
                param_method=self.Param.GFN1xTB
            elif self.xtbmethod == 'GFN0':
                print("Using GFN0 parameterization")
                param_method=self.Param.GFN0xTB
            elif self.xtbmethod == 'GFNFF':
                print("Using GFNFF parameterization")
                print("warning: experimental")
                param_method=self.Param.GFNFF
            elif self.xtbmethod == 'IPEA':
                print("Using IPEA parameterization")
                param_method=self.Param.IPEAxTB
            else:
                print("unknown xtbmethod")
                ashexit()

            #Creating calculator using Hamiltonian and coordinates
            #Setting charge and mult
            #NOTE: New calculator object in every Opt/MD iteration is unnecessary
            #TODO: change

            #first run call: create new object containing coordinates and settings
            if self.calcobject == None:
                print("Creating new xTB calc object")
                self.calcobject = self.Calculator(param_method, qm_elems_numbers, coords_au, charge=charge, uhf=mult-1)
                self.calcobject.set_verbosity(self.verbosity)
                self.calcobject.set_electronic_temperature(self.electronic_temp)
                self.calcobject.set_max_iterations(self.maxiter)
                self.calcobject.set_accuracy(self.accuracy)
            #nextt run calls: only update coordinates
            else:
                print("Updating coordinates in xTB calcobject")
                self.calcobject.update(coords_au)

            #QM/MM pointcharge field
            #calc.
            if PC==True:
                print("Using PointCharges")
                mmcharges=np.array(MMcharges)
                #print("Setting external point charges")
                #print("num MM charges", len(MMcharges))
                #print(MMcharges)
                #print("num MM coords", len(current_MM_coords))
                #print(current_MM_coords)
                MMcoords_au=np.array(current_MM_coords)*constants.ang2bohr
                #print(MMcoords_au)
                #NOTE: Are these element nuclear charges or what ?
                numbers=np.array([9999 for i in MMcharges])
                #print("numbers:", numbers)
                self.calcobject.set_external_charges(numbers,mmcharges,MMcoords_au)

            #Run
            #TODO: Can we turn off gradient calculation somewhere?
            print("Running xtB using {} cores".format(self.numcores))
            res = self.calcobject.singlepoint()
            print("------------xTB calculation done-------------")
            if Grad == True:
                print("Grad is True")
                self.energy = res.get_energy()
                self.grad =res.get_gradient()
                if self.printlevel >= 2:
                    print("xtb energy :", self.energy)
                if PC == True:
                    #pcgrad
                    #get pcgrad
                    print("pc grad is not yet implemented. ")
                    ashexit()
                    print("------------ENDING XTB-INTERFACE-------------")
                    print_time_rel(module_init_time, modulename='xTBlib run', moduleindex=2)
                    return self.energy, self.grad, self.pcgrad
                else:
                    print("------------ENDING XTB-INTERFACE-------------")
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

        elif self.runmode=='oldlibrary':

            if PC==True:
                print("Pointcharge-embedding on but xtb-runmode is library!")
                print("The xtb library-interface is not yet ready for QM/MM calculations")
                print("Use runmode='inputfile' for now")
                ashexit()


            #Hard-coded options. Todo: revisit
            options = {
                "print_level": 1,
                "parallel": 0,
                "accuracy": 0.1,
                "electronic_temperature": 300.0,
                "gradient": True,
                "restart": True,
                "ccm": True,
                "max_iterations": 30,
                "solvent": "none",
            }

            #Using the xtbobject previously defined
            num_qmatoms=len(current_coords)
            #num_mmatoms=len(MMcharges)
            nuc_charges=np.array(modules.module_coords.elemstonuccharges(qm_elems), dtype=self.c_int)

            #Converting coords to numpy-array and then to Bohr.
            current_coords_bohr=np.array(current_coords)*constants.ang2bohr
            positions=np.array(current_coords_bohr, dtype=self.c_double)
            args = (num_qmatoms, nuc_charges, positions, options, 0.0, 0, "-")
            print("------------Running xTB-------------")
            if self.xtbmethod=='GFN1':
                results = self.xtbobject.GFN1Calculation(*args)
            elif self.xtbmethod=='GFN2':
                results = self.xtbobject.GFN2Calculation(*args)
            else:
                print("Unknown xtbmethod.")
                ashexit()
            print("------------xTB calculation done-------------")
            if Grad==True:
                self.energy = float(results['energy'])
                self.grad = results['gradient']
                print("xtb energy:", self.energy)
                #print("self.grad:", self.grad)
                print("------------ENDING XTB-INTERFACE-------------")
                print_time_rel(module_init_time, modulename='xTB run', moduleindex=2)
                return self.energy, self.grad
            else:
                self.energy = float(results['energy'])
                print("xtb energy:", self.energy)
                print("------------ENDING XTB-INTERFACE-------------")
                print_time_rel(module_init_time, modulename='xTB run', moduleindex=2)
                return self.energy
        else:
            print("Unknown option to xTB interface")
            ashexit()



#Grab Final single point energy
def xtbfinalenergygrab(file):
    with open(file) as f:
        for line in f:
            if 'TOTAL ENERGY' in line:
                Energy=float(line.split()[-3])
    return Energy

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
def run_xtb_SP_serial(xtbdir, xtbmethod, xyzfile, charge, mult, Grad=False, Opt=False, maxiter=500, electronic_temp=300, accuracy=0.1, solvent=None):
    
    if solvent != None:
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
                              '--etemp', str(electronic_temp), '--acc', str(accuracy), '--input', 'xtbinput', str(solvent_line)  ]
    elif Opt == True:
        command_list=[xtbdir + '/xtb', basename+'.xyz', '--gfn', str(xtbflag), '--opt', '--chrg', str(charge), '--uhf', str(uhf), '--iterations', str(maxiter),
                              '--etemp', str(electronic_temp), '--acc', str(accuracy), '--input', 'xtbinput', str(solvent_line)  ]    
    else:
        command_list=[xtbdir + '/xtb', basename + '.xyz', '--gfn', str(xtbflag), '--chrg', str(charge), '--uhf', str(uhf), '--iterations', str(maxiter),
                      '--etemp', str(electronic_temp), '--acc', str(accuracy), '--input', 'xtbinput', str(solvent_line)]
    print("Running xtb with these arguments:", command_list)
    
    with open(basename+'.out', 'w') as ofile:
        process = sp.run(command_list, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

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
        process = sp.run([settings_solvation.xtbdir + '/xtb', xyzfile, '--gfn', gfnoption, '--chrg', chargeA, '--uhf', uhfA ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    with open(basename+'_StateB.out', 'w') as ofile:
        process = sp.run([settings_solvation.xtbdir + '/xtb', xyzfile, '--gfn', gfnoption, '--chrg', chargeB, '--uhf', uhfB ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
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
        process = sp.run([settings_solvation.xtbdir + '/xtb', xyzfile, chargeseg1, chargeseg2, uhfseg1, uhfseg2, ipseg], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    os.chdir('..')

#Using IPEA-xtB method for IP calculations
def run_xtb_VIP(xyzfile, charge, mult):
    basename = xyzfile.split('.')[0]
    uhf=mult-1
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([settings_solvation.xtbdir + '/xtb', basename+'.xyz', '--vip', '--chrg', str(charge), '--uhf', str(uhf) ], check=True, stdout=ofile, universal_newlines=True)


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
    NumCoresToUse=settings_solvation.NumCores
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
    bohr2ang=constants.bohr2ang
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
    bohr2ang=constants.bohr2ang
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
