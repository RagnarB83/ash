import os
import shutil
import numpy as np
import subprocess as sp
import time

import ash.constants
import ash.settings_solvation
import ash.settings_ash
from ash.functions.functions_general import ashexit, blankline,reverse_lines, print_time_rel,BC, print_line_with_mainheader
import ash.modules.module_coords
from ash.modules.module_coords import elemstonuccharges, check_multiplicity, check_charge_mult


#BIGDFT

class BigDFTTheory:
    def __init__(self, printlevel=2, filename='bigdft_', maxiter=500, electronic_temp=300, label=None,
                 hgrid=0.4, functional=None, threads=1, mpiprocs=1, numcores=1, use_gpu=False):

        #Indicate that this is a QMtheory
        self.theorytype="QM"

        #Printlevel
        self.printlevel=printlevel
        
        #Label to distinguish different objects
        self.label=label


        self.numcores=numcores
        self.filename=filename
        self.maxiter=maxiter
        
        
        print_line_with_mainheader("BigDFT INTERFACE")

        #Parallelization for both library and inputfile runmode
        print("BigDFT object numcores:", self.numcores)

        try:
            from BigDFT import Calculators as calc
        except:
            print("Problem importing BigDFT library. Have you installed it correctly ?")
            ashexit(code=9)

        #???
        #reload(calc)
        #Specifying 
        if threads == 1 and mpiprocs == 1 and numcores == 1:
            print(f"Threads: {threads} MPIprocs:{mpiprocs} and numcores:{numcores}")
            print("Using the default of 1 OMP thread")
            self.study = calc.SystemCalculator(omp=1)
        elif mpiprocs != 1 and threads == 1:
            print("Setting MPI procs to ", mpiprocs)
            self.study = calc.SystemCalculator(omp=1,mpi_run=f'mpirun -np {mpiprocs}')
        elif threads != 1 and mpiprocs == 1:
            print("Setting Threads to ", threads)
            self.study = calc.SystemCalculator(omp=threads)
        elif threads != 1 and mpiprocs != 1:
            print(f"Setting Threads: {threads} and MPIprocs:{mpiprocs}")
            self.study = calc.SystemCalculator(omp=threads,mpi_run=f'mpirun -np {mpiprocs}')
        elif numcores != 1:
            print("Numcores set. Setting MPI-processes equal to numcores (MPI is faster than threading)", numcores)
            self.study = calc.SystemCalculator(omp=1,mpi_run=f'mpirun -np {mpiprocs}')
        else:
            print("Unknown parallelization option")
            ashexit()
        print("self.study:", self.study)

        if functional is None:
            print("functional keyword not set. Exiting")
            ashexit()
        #Define inputobject
        from BigDFT import Inputfiles as I
        self.inp=I.Inputfile()

        #Settings
        self.inp.set_hgrid(hgrid)
        #self.inp.set_rmult([3.5,9.0])
        self.inp.set_xc(functional)
        
        self.inp["perf"]={}

        if use_gpu is True:
            print("use_gpu:", use_gpu)
            from BigDFT import InputActions
            InputActions.use_gpu_acceleration(self.inp)

        print("BigDFT input object:", self.inp)

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    #Cleanup after run.
    def cleanup(self):
        if self.printlevel >= 2:
            print("Cleaning up old BigDFT files")
        files= []
        
        for file in files:
            try:
                os.remove(file)
            except:
                pass

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, printlevel=None,
                elems=None, Grad=False, PC=False, numcores=None, label=None, charge=None, mult=None):
        module_init_time=time.time()

        if MMcharges is None:
            MMcharges=[]


        #NOTE: CHECK
        if numcores is None:
            numcores=self.numcores

        if self.printlevel >= 2:
            print("------------STARTING BIGDFT INTERFACE-------------")
            print("Object-label:", self.label)
            print("Run-label:", label)
        #Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        #Checking if charge and mult has been provided and sensible
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for BigDFTTheory.run method", BC.END)
            ashexit()   
        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems
        check_multiplicity(qm_elems,charge,mult)
        #Write coordinates to disk (necessary ??)
        ash.modules.module_coords.write_xyzfile(qm_elems, current_coords, self.filename, printlevel=self.printlevel)
        self.inp.set_atomic_positions(f'{self.filename}.xyz')

        if Grad is True:
            print("Grad is True")
            self.inp["perf"]["calculate_forces"] = True

            print(" self.inp:",  self.inp)


        print("------------Running BigDFT-------------")
        #Call BigDFT run
        #NOTE: Parallelization. OMP_NUM_THREADS ???
        result = self.study.run(input=self.inp)

        self.energy = result.energy

        #Check if finished. Grab energy
        if Grad==True:
            self.gradient = grab_gradient_bigdft(len(qm_elems))
            print("BigDFT energy :", self.energy)
            print("------------ENDING BIGDFT-INTERFACE-------------")
            print_time_rel(module_init_time, modulename='BigDFT run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.energy, self.gradient
        else:
            print("BigDFT energy :", self.energy)
            print("------------ENDING BIGDFT-INTERFACE-------------")
            print_time_rel(module_init_time, modulename='BIGDFT run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.energy



def grab_gradient_bigdft(numatoms):
    grab=False
    gradient=np.zeros((numatoms,3))
    i=0
    with open("forces_posinp.xyz") as f:
        for line in f:
            if grab is True:
                gradient[i,0] = -1*float(line.split()[1])
                gradient[i,1] = -1*float(line.split()[2])
                gradient[i,2] = -1*float(line.split()[3])
                i+=1
            if line.startswith(' forces'):
                grab=True
    return gradient