import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader, \
writestringtofile, check_program_location
import ash.settings_ash
from ash.functions.functions_parallel import check_OpenMPI

# ReSpectTheory object.

class ReSpectTheory:
    def __init__(self, respectdir=None, filename='respect', printlevel=2, label="ReSpect",
                numcores=1, scf_inputkeywords=None, jobtype=None, 
                jobtype_inputkeywords=None):

        self.theorynamelabel="ReSpectTheory"
        self.label=label
        self.theorytype="QM"
        self.analytic_hessian=False
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        self.respectdir=check_program_location(respectdir,'respectdir','respect')
        #
        self.filename=filename
        self.jobtype=jobtype
        self.scf_inputkeywords=scf_inputkeywords
        self.jobtype_inputkeywords=jobtype_inputkeywords
        #
        if scf_inputkeywords is None:
            print("Error: You must pass a dictionary to scf_inputkeywords option")
            ashexit()
        if jobtype is None:
            print("jobtype is None. This means ReSpect will only carry out an SCF calculation")
        else:
            print("jobtype is", jobtype)
            if jobtype_inputkeywords is None:
                print(f"Error: jobtype {jobtype} selected. You must pass a dictionary to jobtype_inputkeywords option")
                ashexit()

        # Checking OpenMPI
        if numcores != 1:
            print(f"Parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
            print("parallelization:", self.parallelization)
            if self.parallelization == 'MPI':
                print("Parallelization is MPI. Checking availability of OpenMPI")
                check_OpenMPI()

        # Printlevel
        self.printlevel=printlevel
        self.numcores=numcores

    # Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    #def cleanup(self):

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        # Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)
        print(f"{self.theorynamelabel} input:")

        # Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # Delete old scratchdir for ReSpect if present and create new one
        try:
            shutil.rmtree("respect_calc_scratch")
        except:
            pass
        os.mkdir('respect_calc_scratch')

        # Create inputfile
        print("Creating ReSpect inputfile")
        create_respect_inputfile(self.filename,qm_elems, current_coords, charge,mult, jobtype=self.jobtype,
                                 scf_inputkeywords=self.scf_inputkeywords, jobtype_inputkeywords=self.jobtype_inputkeywords, )

        # Always run SCF-job first
        print("Running ReSpect SCF calculation")
        run_respect(respectdir=self.respectdir, jobtype='scf', inputfile=self.filename, numcores=self.numcores, scratchdir=f'{os.getcwd()}/respect_calc_scratch')

        #Grab energy, gradient
        self.energy, self.gradient = grab_energy_gradient(f"{self.filename}.out_scf", Grad=Grad)
        #Grab properties

        # if jobtype has been set
        if self.jobtype is not None:
            print("Running ReSpect property calculation:", self.jobtype)
            if isinstance(self.jobtype, list):
                print("Multiple jobtypes requested. Will loop through them")
                for job in self.jobtype:
                    print("Running job:", job)
                    run_respect(respectdir=self.respectdir, jobtype=self.jobtype, inputfile=self.filename, numcores=self.numcores, scratchdir='/respect_calc_scratch')
            else:
                run_respect(respectdir=self.respectdir, jobtype=self.jobtype, inputfile=self.filename, numcores=self.numcores, scratchdir='./respect_calc_scratch')

        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        # Grab gradient if calculated
        if Grad is True:
            return self.energy, self.gradient
        else:
            return self.energy

# jobtype: scf, gt
def run_respect(respectdir=None, jobtype='scf', inputfile='', numcores=1, scratchdir='.'):

    process = sp.run([respectdir + f'/respect', f'--{jobtype}', f'--inp={inputfile}', f"--nt={numcores}", f"--scratch={scratchdir}"], 
                     check=True, stdout=ofile, stderr=ofile, universal_newlines=True)


def grab_energy_gradient(filename, Grad=False):
    gradient=None
    if Grad:
        pass
    else:
        with open(filename) as f:
            for line in f:
                if '  --- Total energy             --- =' in line:
                    energy = float(line.split()[-1])
                    break
    return energy, gradient

def create_respect_inputfile(filename,elems, coords, charge,mult, scf_inputkeywords=None, jobtype_inputkeywords=None, jobtype=None):

    indent="  "
    f = open(f"{filename}.inp", "w")
    f.write("# Respect inputfile created by ASH\n")
    f.write("# SCF inputblock\n")
    f.write("scf:\n")
    f.write("\n")
    # Geometry block
    f.write(f"{indent}geometry:\n")
    for el,c in zip(elems,coords):
        f.write(f"{indent}{indent}{el}  {c[0]} {c[2]} {c[2]} \n")
    f.write(f"{indent}charge:        {charge}\n")
    f.write(f"{indent}multiplicity:        {mult}\n")
    for s_k,s_val in scf_inputkeywords.items():
        f.write(f"    {s_k}: {s_val}\n")
    if jobtype is not None:
        for job_k, job_val in jobtype_inputkeywords.items():
            f.write(f"#{job_k} input block\n")
            f.write(f"{job_k}:\n")
            for subj_k,subj_val in job_val.items():
                f.write(f"{indent}{subj_k}: {subj_val}\n")

    f.close()
