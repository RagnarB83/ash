import subprocess as sp
import os
import shutil
import time
import numpy as np

import ash.settings_ash
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader

#MRCC Theory object.
class MRCCTheory:
    def __init__(self, mrccdir=None, filename='mrcc', printlevel=2,
                mrccinput=None, numcores=1, parallelization='OMP-and-MKL'):

        print_line_with_mainheader("MRCCTheory initialization")

        if mrccinput is None:
            print("MRCCTheory requires a mrccinput keyword")
            ashexit()

        if mrccdir == None:
            print(BC.WARNING, "No mrccdir argument passed to MRCCTheory. Attempting to find mrccdir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.mrccdir=ash.settings_ash.settings_dict["mrccdir"]
            except KeyError:
                print(BC.WARNING,"Found no mrccdir variable in settings_ash module either.",BC.END)
                try:
                    self.mrccdir = os.path.dirname(shutil.which('dmrcc'))
                    print(BC.OKGREEN,"Found dmrcc in PATH. Setting mrccdir to:", self.mrccdir, BC.END)
                except:
                    print(BC.FAIL,"Found no dmrcc executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.mrccdir = mrccdir


        #Indicate that this is a QMtheory
        self.theorytype="QM"

        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.mrccinput=mrccinput
        self.numcores=numcores

        #Parallelization strategy: 'OMP', 'OMP-and-MKL' or 'MPI'
        self.parallelization=parallelization

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("MRCC cleanup not yet implemented.")

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING MRCC INTERFACE-------------", BC.END)
        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for MRCCTheory.run method", BC.END)
            ashexit()

        print("Running MRCC object.")
        print("Job label:", label)
        print("Creating inputfile: MINP")
        print("MRCC input:")
        print(self.mrccinput)

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
        #TODO: No qm/MM yet. need to check if possible in MRCC

        if Grad==True:
            write_mrcc_input(self.mrccinput,charge,mult,qm_elems,current_coords,numcores,Grad=True)
            run_mrcc(self.mrccdir,self.filename+'.out',self.parallelization,numcores)
            self.energy=grab_energy_mrcc(self.filename+'.out')
            self.gradient = grab_gradient_mrcc(self.filename+'.out',len(qm_elems))

        else:
            write_mrcc_input(self.mrccinput,charge,mult,qm_elems,current_coords,numcores)
            run_mrcc(self.mrccdir,self.filename+'.out',self.parallelization,numcores)
            self.energy=grab_energy_mrcc(self.filename+'.out')

        #TODO: write in error handling here
        print(BC.OKBLUE, BC.BOLD, "------------ENDING MRCC INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point MRCC energy:", self.energy)
            print("MRCC gradient:", self.gradient)
            print_time_rel(module_init_time, modulename='MRCC run', moduleindex=2)
            return self.energy, self.gradient
        else:
            print("Single-point MRCC energy:", self.energy)
            print_time_rel(module_init_time, modulename='MRCC run', moduleindex=2)
            return self.energy

def run_mrcc(mrccdir,filename,parallelization,numcores):
    with open(filename, 'w') as ofile:
        #process = sp.run([mrccdir + '/dmrcc'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

        if parallelization == 'OMP':
            print(f"OMP parallelization is active. Using OMP_NUM_THREADS={numcores}")
            os.environ['OMP_NUM_THREADS'] = str(numcores)
            os.environ['MKL_NUM_THREADS'] = str(1)
            process = sp.run([mrccdir + '/dmrcc'], env=os.environ, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        elif parallelization == 'OMP-and-MKL':
            print(f"OMP-and-MKL parallelization is active. Both OMP_NUM_THREADS and MKL_NUM_THREADS set to: {numcores}")
            os.environ['OMP_NUM_THREADS'] = str(numcores)
            os.environ['MKL_NUM_THREADS'] = str(numcores)
            process = sp.run([mrccdir + '/dmrcc'], env=os.environ, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        elif parallelization == 'MPI':
            print(f"MPI parallelization active. Will use {numcores} MPI processes. (OMP and MKL disabled)")
            os.environ['MKL_NUM_THREADS'] = str(1)
            os.environ['OMP_NUM_THREADS'] = str(1)
            process = sp.run([mrccdir + '/dmrcc'], env=os.environ, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#TODO: Gradient option
#NOTE: Now setting ccsdthreads and ptthreads to number of cores
def write_mrcc_input(mrccinput,charge,mult,elems,coords,numcores,Grad=False):
    with open("MINP", 'w') as inpfile:
        inpfile.write(mrccinput + '\n')
        inpfile.write(f'ccsdthreads={numcores}\n')
        inpfile.write(f'ptthreads={numcores}\n')
        inpfile.write('unit=angs\n')
        inpfile.write('charge={}\n'.format(charge))
        inpfile.write('mult={}\n'.format(mult))
        #If Grad true set density to first-order. Gives properties and gradient
        if Grad is True:
            #dens=2 for RHF
            if "calc=RHF" in mrccinput:
                inpfile.write('dens=2\n')
            else:
                inpfile.write('dens=1\n')
        #inpfile.write('dens=2\n')
        inpfile.write('geom=xyz\n')
        inpfile.write('{}\n'.format(len(elems)))
        inpfile.write('\n')
        for el,c in zip(elems,coords):
            inpfile.write('{}   {} {} {}\n'.format(el,c[0],c[1],c[2]))
        inpfile.write('\n')

def grab_energy_mrcc(outfile):
    #Option 1. Grabbing all lines containing energy in outputfile. Take last entry.
    # CURRENT Option 2: grab energy from iface file. Higher level WF entry should be last
    with open("iface") as f:
        for line in f:
            if 'ENERGY' in line:
                energy=float(line.split()[5])
    return energy


def grab_gradient_mrcc(file,numatoms):
    grab=False
    atomcount=0
    gradient=np.zeros((numatoms,3))
    with open(file) as f:
        for line in f:
            if grab is True:
                if len(line.split())==5:
                    gradient[atomcount,0] = float(line.split()[-3])
                    gradient[atomcount,1] = float(line.split()[-2])
                    gradient[atomcount,2] = float(line.split()[-1])
                    atomcount+=1
            if ' Molecular gradient [au]:' in line:
                grab=True
    return gradient
