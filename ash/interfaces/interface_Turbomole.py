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
    def __init__(self, turbodir=None, filename='XXX', printlevel=2, label="Turbomole",
                numcores=1, functional=None, gridsize="m4", scfconf=7, symmetry="c1",
                basis=None, jbasis=None, scfiterlimit=50, maxcor=500, ricore=500):

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

        # Basis set
        if basis is None:
            print(BC.WARNING, f"No basis set provided to {self.theorynamelabel}Theory. Exiting...", BC.END)
            ashexit()
        else:
            self.basis=basis

        # Turbomole executable to use
        if functional is not None:
            print("Functional provided. Choosing Turbomole executable to be ridft")
            #TODO: choose RI or not
            self.turbo_exe="ridft"
            self.filename="ridft"

        print("self.turbo_exe:", self.turbo_exe)
        print("jbasis:", jbasis)
        # Checking for ridft and jbas
        if self.turbo_exe =="ridft" and jbasis is None:
            print("No jbasis provided for ridft. Exiting...")
            ashexit()
        else:
            self.jbasis=jbasis


        # Checking OpenMPI
        if numcores != 1:
            print(f"Parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
            check_OpenMPI()

        # Finding Turbomole
        if turbodir is None:
            print(BC.WARNING, f"No turbodir argument passed to {self.theorynamelabel}Theory. Attempting to find turbodir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.turbodir=ash.settings_ash.settings_dict["turbodir"]
            except:
                print(BC.WARNING,"Found no turbodir variable in settings_ash module either.",BC.END)
                try:
                    self.turbodir = os.path.dirname(shutil.which('ridft'))
                    print(BC.OKGREEN,"Found ridft (Turbomol executable) in PATH. Setting turbodir to:", self.turbodir, BC.END)
                except:
                    print(BC.FAIL,"Found no ridft executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.turbodir = turbodir

        # Printlevel
        self.printlevel=printlevel
        self.numcores=numcores

    # Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print(f"{self.theorynamelabel} cleanup not yet implemented.")

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

        # Create coord file
        create_coord_file(qm_elems,current_coords)

        #
        create_control_file(functional=self.functional, gridsize=self.gridsize, scfconf=self.scfconf, 
                            symmetry="c1", basis=self.basis, jbasis=self.jbasis, 
                            scfiterlimit=self.scfiterlimit, maxcor=self.maxcor, ricore=self.ricore)

        # Run Turbomole
        print("Running Turbomole executable:", self.turbo_exe)
        run_turbo(self.turbodir,self.filename, exe=self.turbo_exe, 
                  numcores=self.numcores)

        # Grab energy
        exit()
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Grab gradient if calculated
        if Grad is True:
            #Grab gradient
            self.gradient = grab_gradient_NWChem(self.filename+'.out',len(current_coords))
            #Grab PCgradient from separate file
            if PC is True:
                self.pcgradient = grab_pcgradient_NWChem(f'{self.filename}.bqforce.dat',len(MMcharges))
                print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
                return self.energy, self.gradient, self.pcgradient
            else:
                print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
                return self.energy, self.gradient
        #Returning energy without gradient
        else:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
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

def create_control_file(functional="lh12ct-ssifpw92", gridsize="m4", scfconf="7", symmetry="c1",
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
$dft
    functional   {functional}
    gridsize   {gridsize}
$scfconv   {scfconf}
$ricore      {ricore}
$rij
$jbas    file=auxbasis
$end"""

    writestringtofile(controlstring, 'control')


def run_turbo(turbodir,filename, exe="ridft", numcores=1):
    print(f"Running executable {exe} and writing to output {filename}.out")
    with open(filename+'.out', 'w') as ofile:
        #if numcores >1:
        #    process = sp.run(['mpirun', '-np', str(numcores), turbodir + '/nwchem', filename+'.nw'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        #else:
        process = sp.run([turbodir + '/ridft'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
