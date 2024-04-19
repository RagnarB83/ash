import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
import ash.settings_ash
from ash.functions.functions_parallel import check_OpenMPI

#Example theory file to demonstrate how to write a very basic interface for a QM-code in ASH

# First we define the class: It should typically be called XXXTheory to be consistent with the ASH naming scheme
# The __init__ method is flexible but must contain keyword arguments filename, printlevel, label, numcores.
#Other input options are flexible

class SomeDummyTheory:
    def __init__(self, some_input_option=None, filename="somedummytheory", printlevel=2, label="SomeDummyTheory", numcores=1):

        #Early exits, check if no user input etc.
        if some_input_option is None:
            print(BC.FAIL, "Some kind of input is required. Exiting.", BC.END)
            ashexit()

        #Set input variables as attributes
        self.some_input_option = some_input_option
        self.printlevel=printlevel
        self.numcores=numcores
        self.label=label

        #Some hardcoded attributes
        self.theorynamelabel="SomeDummyTheory" #Name of theory, sometimes displayed
        self.analytic_hessian=False
        #Store optional properties of job run job in a dict
        self.properties ={}

        #Check if executable is available in PATH or if Python library can be loaded
        #--------DO THAT HERE--------

        #Check parallelization option. If MPI, check if MPI library is available
        #--------DO THAT HERE--------

        #Set number of cores for code (mpirun, OMP_NUM_THREADS, MKL_NUM_THREADS etc.) using self.numcores
        #--------DO THAT HERE--------
    
    #Set numcores method (if an ASH job-function wants to change number of cores)
    def set_numcores(self,numcores):
        self.numcores=numcores
        #NOTE: add other code necessary for updating number of cores here
    
    #Cleanup (called by some ASH job-functions)
    def cleanup(self):
        print(f"{self.theorynamelabel} cleanup not yet implemented.")
        #NOTE: add useful cleanup code here (e.g. remove )
    
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        #Timing initialization
        module_init_time=time.time()
        #Optional numcores handling
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)
        print(f"Creating inputfile: {self.filename}")
        print(f"{self.theorynamelabel} input:")
        print(self.some_input_option)

        ###########################################
        #Prepare input QM and MM coordinates here
        ###########################################

        #Create inputfile (if used) and write coordinates to disk if needed
        #--------DO THAT HERE--------
        #EXAMPLES: 
        #inputfile_creator(elems=elems, coords=current_coords, some_input_option=self.some_input_option, 
        #                  Grad=Grad, filename=self.filename,charge=charge, mult=mult)
        #write_xyzfile(dummy_elem_list, dummy_coords, f"{system_xyzfile}", printlevel=1)
        
        #If PC (pointcharges, i.e. QM/MM) is True then deal with input pointcharge handling to program herehere
        #if PC is True:
            #--------DO THAT HERE--------
            #e.g. write PC coordinates and charges to inputfile or separate file
        
        
        # Call a function that runs the program here
        #--------DO THAT HERE--------
        #EXAMPLE:
        #run_program(self.filename, self.numcores)
        
        #Grab energy and gradient from outputfile or Python library object here
        #--------DO THAT HERE--------
        energy=None
        gradient=None
        pcgradient=None
        #energy, gradient = some_function_that_grabs_E_and_G(self.filename)
        
        #For QM/MM then also grab pointcharge gradient
        #if PC is True:
        #    pcgradient = pcgradient_grab(self.filename)

        #Optional: add properties from output to self.properties dict
        
        if self.printlevel >= 1:
            print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        
        
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        #Print timings for run method (this info also goes into global Timings object)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2, 
                       currprintlevel=self.printlevel, currthreshold=1)

        #Return energy or energy,gradient or energy,gradient,pcgradient
        if Grad:
            if PC:
                return energy, gradient, pcgradient
            else:
                return energy, gradient
        else:
            return energy


#################################################
# Independent QM-program functions
# stuff that does not need to be in the class
#################################################

#def inputfile_creator(elems=None, coords=None, some_input_option=None, 
#                      Grad=False, filename=None,charge=None, mult=None):

    #--------DO SOMETHING HERE--------

#    return