import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader, writestringtofile
import ash.settings_ash
from ash.functions.functions_parallel import check_OpenMPI

# Turbomole Theory object.

class TurbomoleTheory:
    def __init__(self, TURBODIR=None, turbomoledir=None, filename='XXX', printlevel=2, label="Turbomole",
                numcores=1, parallelization='SMP', functional=None, gridsize="m4", scfconf=7, symmetry="c1", rij=True,
                basis=None, jbasis=None, scfiterlimit=50, maxcor=500, ricore=500,
                mp2=False):

        self.theorynamelabel="Turbomole"
        self.label=label
        self.theorytype="QM"
        self.analytic_hessian=False
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        #
        self.scfiterlimit=scfiterlimit
        self.functional=functional
        self.symmetry=symmetry
        self.scfconf=scfconf
        self.gridsize=gridsize
        self.basis=basis
        self.jbasis=jbasis
        self.maxcor=maxcor
        self.ricore=ricore
        self.rij=rij
        self.mp2=mp2
        self.parallelization=parallelization

        # Basis set
        if basis is None:
            print(BC.WARNING, f"No basis set provided to {self.theorynamelabel}Theory. Exiting...", BC.END)
            ashexit()
        else:
            self.basis=basis

        # MP2
        if self.mp2 is True:
            self.rij=False
            self.dft=False
            if functional is not None:
                print("Error: MP2 is True but a functional was provided. Exiting...")
                ashexit()

            self.turbo_scf_exe="dscf"
            self.filename_scf="dscf"

            self.turbo_mp2_exe="ricc2"
            self.filename_mp2="ricc2"

            self.turbo_exe_grad="grad"
        # DFT
        elif functional is not None:
            self.dft=True
            print("Functional provided. Choosing Turbomole executables to be ridft and rdgrad")
            if rij is True:
                self.turbo_scf_exe="ridft"
                self.turbo_exe_grad="rdgrad"
                self.filename_scf="ridft"
                self.filename_grad="rdgrad"
            else:
                self.turbo_scf_exe="dscf"
                self.turbo_exe_grad="grad"
                self.filename_scf="dscf"
                self.filename_grad="grad"      
        print("self.turbo_scf_exe:", self.turbo_scf_exe)
        print("jbasis:", jbasis)
        # Checking for ridft and jbas
        if self.turbo_scf_exe =="ridft" and jbasis is None:
            print("No jbasis provided for ridft. Exiting...")
            ashexit()
        else:
            self.jbasis=jbasis

        # Checking OpenMPI
        if numcores != 1:
            print(f"Parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
            print("parallelization:", self.parallelization)
            if self.parallelization == 'MPI':
                print("Parallelization is MPI. Checking availability of OpenMPI")
                check_OpenMPI()

        # Finding Turbomole
        if TURBODIR is not None:
            #self.turbomoledir = TURBODIR
            self.TURBODIR=TURBODIR
        elif turbomoledir is None:
            print(BC.WARNING, f"No turbomoledir argument passed to {self.theorynamelabel}Theory. Attempting to find turbomoledir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.turbomoledir=ash.settings_ash.settings_dict["turbomoledir"]
            except:
                print(BC.WARNING,"Found no turbomoledir variable in settings_ash module either.",BC.END)
                try:
                    self.turbomoledir = os.path.dirname(shutil.which('ridft'))
                    print(BC.OKGREEN,"Found ridft (Turbomol executable) in PATH. Setting turbomoledir to:", self.turbomoledir, BC.END)
                except:
                    print(BC.FAIL,"Found no ridft executable in PATH. Exiting... ", BC.END)
                    ashexit()
            self.TURBODIR = os.path.dirname(os.path.dirname(self.turbomoledir))
        else:
            self.turbomoledir = turbomoledir
            self.TURBODIR = os.path.dirname(os.path.dirname(self.turbomoledir))

        # Setting environment variable TURBODIR (for basis set )
        os.environ['TURBODIR'] = self.TURBODIR
        print("TURBODIR:", self.TURBODIR)

        # Printlevel
        self.printlevel=printlevel
        self.numcores=numcores

    # Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        files=['coord','control','energy','gradient', 'auxbasis', 'basis', 'mos', 'ridft.out', 'rdgrad.out', 'ricc2.out', 'statistics']
        for f in files:
            if os.path.exists(f):
                os

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)
        print(f"{self.theorynamelabel} input:")


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

        # Delete old files
        files=['coord','control','energy','gradient']
        for f in files:
            if os.path.exists(f):
                os.remove(f)

        # Create coord file
        create_coord_file(qm_elems,current_coords)

        #
        create_control_file(functional=self.functional, gridsize=self.gridsize, scfconf=self.scfconf, dft=self.dft,
                            symmetry="c1", basis=self.basis, jbasis=self.jbasis, rij=self.rij, mp2=self.mp2,
                            scfiterlimit=self.scfiterlimit, maxcor=self.maxcor, ricore=self.ricore)
        #################
        # Run Turbomole
        #################

        print("Running Turbomole executable:", self.turbo_scf_exe)
        # SCF-energy only
        run_turbo(self.turbomoledir,self.filename_scf, exe=self.turbo_scf_exe, parallelization=self.parallelization,
                  numcores=self.numcores)
        self.energy = grab_energy_from_energyfile()
        print("SCF Energy:", self.energy)

        # MP2 energy only
        if self.mp2 is True:
            print("MP2 is True. Running:", self.turbo_mp2_exe)
            run_turbo(self.turbomoledir,self.filename_mp2, exe=self.turbo_mp2_exe, parallelization=self.parallelization,
                  numcores=self.numcores)
            mp2_corr_energy = grab_energy_from_energyfile(column=4)
            print("MP2 correlation energy:", mp2_corr_energy)
            self.energy += mp2_corr_energy
            print("Total MP2 energy:", self.energy)

        # GRADIENT
        if Grad is True:
            print("Running Turbomole-gradient executable")
            print("self.turbo_exe_grad:", self.turbo_exe_grad)
            print("self.filename_grad:", self.filename_grad)
            run_turbo(self.turbomoledir,self.filename_grad, exe=self.turbo_exe_grad, parallelization=self.parallelization,
                  numcores=self.numcores)
            self.gradient = grab_gradient(len(current_coords))

        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        # Grab gradient if calculated
        if Grad is True:
            return self.energy, self.gradient
        else:
            return self.energy


################################
# Independent Turbomole functions
################################

def create_coord_file(elems,coords):
    ang2bohr=1.88972612546
    with open('coord', 'w') as coordfile:
        coordfile.write("$coord\n")
        for i in range(len(elems)):
            coordfile.write(f"{coords[i][0]*ang2bohr} {coords[i][1]*ang2bohr} {coords[i][2]*ang2bohr} {elems[i]}\n")
        coordfile.write("$end\n")

def create_control_file(functional="lh12ct-ssifpw92", gridsize="m4", scfconf="7", symmetry="c1", rij=True, dft=True, mp2=False,
                        basis="def2-SVP", jbasis="def2-SVP", scfiterlimit=30, maxcor=500, ricore=500):

#Skipping orb section for now
#$closed shells
# a       1-7                                    ( 2 )

    controlstring=f"""
$title
$symmetry {symmetry}
$coord    file=coord
$atoms
    basis ={basis}
    jbas  ={jbasis}
$basis    file=basis
$scfmo   file=mos
$scfiterlimit       {scfiterlimit}
$scfdamp   start=0.300  step=0.050  min=0.100
$scfdump
$scfdiis
$maxcor    {maxcor} MiB  per_core
$scforbitalshift  automatic=.1
$energy    file=energy
$grad    file=gradient
$scfconv   {scfconf}
"""

    if dft is True:
        controlstring += f"""$dft
    functional   {functional}
    gridsize   {gridsize}"""

    if mp2 is True:
        controlstring += f"""$denconv .1d-6
$ricc2
mp2"""


    if rij is True:
        controlstring += f"""$ricore      {maxcor}
$rij"""

    controlstring+="\n$end"

    writestringtofile(controlstring, 'control')


def run_turbo(turbomoledir,filename, exe="ridft", numcores=1, parallelization=None):
    print(f"Running executable {exe} and writing to output {filename}.out")

    with open(filename+'.out', 'w') as ofile:
        if numcores >1:
            if parallelization == 'MPI':
                print("Parallelization is MPI")
                print(f"Warning: make sure that turbomoledir ({turbomoledir}) is set to the correct path for MPI parallelization")
                os.environ['PARA_ARCH'] = 'MPI'
                os.environ['PARNODES'] = str(numcores)
                process = sp.run(['mpirun', '-np', str(numcores), turbomoledir + f'/{exe}'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            elif parallelization == 'SMP':
                print("Parallelization is SMP")
                os.environ['PARA_ARCH'] = 'SMP'
                print(f"Warning: make sure that turbomoledir ({turbomoledir}) is set to the correct path for SMP parallelization")
                process = sp.run([turbomoledir + f'/{exe}'], check=True, stdout=ofile, stderr=ofile, universal_new=True)
        else:
            process = sp.run([turbomoledir + f'/{exe}'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

def grab_energy_from_energyfile(column=1):
    energy = None
    with open('energy', 'r') as energyfile:
        for line in energyfile:
            if '$end' in line:
                return energy
            if "$energy" not in line:
                energy = float(line.split()[column])
    return energy

def grab_gradient(numatoms):
    gradient = np.zeros((numatoms,3))
    with open('gradient', 'r') as gradfile:
        gradlines = gradfile.readlines()
    counter=0
    for i,line in enumerate(gradlines):
        if '$end' in line:
            break
        if i > 4:
            gradient[counter] = [float(j.replace('D','E')) for j in line.split()]
            counter+=1

    return gradient