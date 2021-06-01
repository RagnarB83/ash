import os
import sys
import shutil
import numpy as np
import subprocess as sp
import time

import constants
import settings_solvation
from functions_general import blankline,reverse_lines, print_time_rel
import module_coords


#xTB functions: primarily for inputfile-based interface. Library-interfaces is in interface_xtb.py


# https://github.com/grimme-lab/xtb/blob/master/python/xtb/interface.py
#Now supports 2 runmodes: 'library' (fast Python C-API) or 'inputfile'
#
#TODO: THis should be a general interface so remove settings_solvation calls.
#TODO: xtb. Need to combine OMP-parallelization of xtb and multiprocessing if possible
#TODO: Currently doing multiprocessing over all 8*2=16 snapshots. First A, then B.
#TODO: Might not be a need to do first A then B since a ROHF-type Hamiltonian
#TODO. Could parallelize over all 32 calculations. However, we are currently using 24 cores so...


class xTBTheory:
    def __init__(self, xtbdir=None, fragment=None, charge=None, mult=None, xtbmethod=None, runmode='inputfile', nprocs=1, printlevel=2, filename='xtb_',
                 maxiter=500):

        #Printlevel
        self.printlevel=printlevel

        if xtbmethod is None:
            print("xTBTheory requires xtbmethod keyword to be set")
            exit(1)

        self.nprocs=nprocs
        if fragment != None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        self.charge=charge
        self.mult=mult
        self.filename=filename
        self.xtbmethod=xtbmethod
        self.maxiter=maxiter
        self.runmode=runmode
        if self.runmode=='library':
            print("Using library-based xTB interface")
            print("Loading library...")
            os.environ["OMP_NUM_THREADS"] = str(nprocs)
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            # Load xtB library and ctypes datatypes that run uses
            try:
                #import xtb_interface_library
                import interface_xtb_library
                self.xtbobject = interface_xtb_library.XTBLibrary()
            except:
                print("Problem importing xTB library. Check that the library dir (containing libxtb.so) is available in LD_LIBRARY_PATH.")
                print("e.g. export LD_LIBRARY_PATH=/path/to/xtb_6.2.3/lib64:$LD_LIBRARY_PATH")
                print("Or that the MKL library is available and loaded")
                exit(9)
            from ctypes import c_int, c_double
            #Needed for complete interface?:
            # from ctypes import Structure, c_int, c_double, c_bool, c_char_p, c_char, POINTER, cdll, CDLL
            self.c_int = c_int
            self.c_double = c_double
        else:
            if xtbdir == None:
                # Trying to find xtbdir in path
                print("xtbdir argument not provided to xTBTheory object. Trying to find xtb in path")
                try:
                    self.xtbdir = os.path.dirname(shutil.which('xtb'))
                    print("Found xtb in path. Setting xtbdir.")
                except:
                    print("Found no xtb executable in path. Exiting... ")
            else:
                self.xtbdir = xtbdir
    #Cleanup after run.
    def cleanup(self):
        if self.printlevel >= 2:
            print("Cleaning up old xTB files")
        list_files=[]
        list_files.append(self.filename + '.xyz')
        list_files.append(self.filename + '.out')
        list_files.append('xtbrestart')
        list_files.append('molden.input')
        list_files.append('chargess')
        list_files.append('pcgrad')
        list_files.append('wbo')
        list_files.append('xtbinput')
        list_files.append('pcharge')
        list_files.append('xtbtopo.mol')
        
        for file in list_files:
            try:
                os.remove(file)
            except:
                pass
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
                elems=None, Grad=False, PC=False, nprocs=None):
        module_init_time=time.time()
        if MMcharges is None:
            MMcharges=[]

        if nprocs is None:
            nprocs=self.nprocs

        if self.printlevel >= 2:
            print("------------STARTING XTB INTERFACE-------------")
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


        #Parallellization
        #Todo: this has not been confirmed to work
        #Needs to be done before library-import??
        print("nprocs:", nprocs)
        os.environ["OMP_NUM_THREADS"] = str(nprocs)
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
            self.cleanup()
            #Todo: xtbrestart possibly. needs to be optional
            module_coords.write_xyzfile(qm_elems, current_coords, self.filename,printlevel=self.printlevel)

            #Run inputfile. Take nprocs argument.
            if self.printlevel >= 2:
                print("------------Running xTB-------------")
                print("...")
            if Grad==True:
                if PC==True:
                    create_xtb_pcfile_general(current_MM_coords, MMcharges)
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', self.charge, self.mult, Grad=True, maxiter=self.maxiter)
                else:
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', self.charge, self.mult, maxiter=self.maxiter,
                                  Grad=True)
            else:
                if PC==True:
                    create_xtb_pcfile_general(current_MM_coords, MMcharges)
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', self.charge, self.mult, maxiter=self.maxiter)
                else:
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', self.charge, self.mult, maxiter=self.maxiter)

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
        elif self.runmode=='library':

            if PC==True:
                print("Pointcharge-embedding on but xtb-runmode is library!")
                print("The xtb library-interface is not yet ready for QM/MM calculations")
                print("Use runmode='inputfile' for now")
                exit(1)


            #Hard-coded options. Todo: revisit
            options = {
                "print_level": 1,
                "parallel": 0,
                "accuracy": 1.0,
                "electronic_temperature": 300.0,
                "gradient": True,
                "restart": False,
                "ccm": True,
                "max_iterations": 30,
                "solvent": "none",
            }

            #Using the xtbobject previously defined
            num_qmatoms=len(current_coords)
            #num_mmatoms=len(MMcharges)
            nuc_charges=np.array(module_coords.elemstonuccharges(qm_elems), dtype=self.c_int)

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
                exit()
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
            exit()



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
    return VIP

# Run xTB single-point job
def run_xtb_SP_serial(xtbdir, xtbmethod, xyzfile, charge, mult, Grad=False, maxiter=500):
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
        exit()
    with open(basename+'.out', 'w') as ofile:
        if Grad==True:
            process = sp.run([xtbdir + '/xtb', basename+'.xyz', '--gfn', str(xtbflag), '--grad', '--chrg', str(charge), '--uhf', '--iterations', str(maxiter),
                              str(uhf), '--input', 'xtbinput' ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        else:
            process = sp.run(
                [xtbdir + '/xtb', basename + '.xyz', '--gfn', str(xtbflag), '--chrg', str(charge), '--uhf', str(uhf), '--iterations', str(maxiter),
                 '--input', 'xtbinput'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
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
def create_xtb_pcfile_general(coords,pchargelist):
    #Creating list of pointcharges based on solventunitcharges and number of elements provided
    bohr2ang=constants.bohr2ang
    hardness=1000
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
            gradient[count-1] = [val_x,val_y,val_z]
    return gradient

#Grab xTB charges. Assuming default xTB charges that are inside file charges
def grabatomcharges_xTB():
    charges=[]
    with open('charges') as file:
        for line in file:
            charges.append(float(line.split()[0]))
    return charges
