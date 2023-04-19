import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
import ash.settings_ash
from ash.modules.module_coords import write_xyzfile
from ash.functions.functions_parallel import check_OpenMPI

#deMon2k interface

#FILES needed in dir: BASIS, AUXIS, deMon.inp, executable
# Optional: ECPS and MCPS

#TODO: GUESS control

#deMon2k Theory object.
class deMon2kTheory:
    def __init__(self, demondir=None, filename='deMon', binary_name='binary', printlevel=2, numcores=1, 
                functional='PBE', scf_type='UKS', basis_name='cc-pVDZ', auxis_name='GEN-A3*'):

        self.theorynamelabel="deMon2k"
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        #EARLY EXITS

        #Checking OpenMPI
        if numcores != 1:
            print(f"Parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
            check_OpenMPI()
        #Finding deMon2k
        if demondir == None:
            print(BC.WARNING, f"No demondir argument passed to {self.theorynamelabel}Theory. Attempting to find demondir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.demondir=ash.settings_ash.settings_dict["demondir"]
                self.binary_name=binary_name
            except:
                print(BC.WARNING,"Found no demondir variable in settings_ash module either.",BC.END)
                try:
                    print(f"Looking for {binary_name}")
                    self.demondir = os.path.dirname(shutil.which(binary_name))
                    print(BC.OKGREEN,f"Found {binary_name} in PATH. Setting demondir to:", self.demondir, BC.END)
                    self.binary_name=binary_name
                except:
                    print(BC.FAIL,f"Found no {binary_name} executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.demondir = demondir
        
        #Indicate that this is a QMtheory
        self.theorytype="QM"

        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        #Defining inputfile
        self.numcores=numcores
        self.functional=functional
        self.basis_name=basis_name
        self.auxis_name=auxis_name
        self.scf_type=scf_type


    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup():
        print(f"self.theorynamelabel cleanup not yet implemented.")

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
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
        print(f"Creating inputfile: {self.filename}.inp")
        print(f"{self.theorynamelabel} input:")
        #TODO: echo final inputfile instead (after modification)

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

        #Case: QM/MM deMon2k job
        if PC is True:
            print("pc true")
            exit()
        else:
            #No QM/MM
            #Write xyz-file with coordinates
            write_xyzfile(qm_elems, current_coords, f"{self.filename}", printlevel=1)
            #Write simple deMon2k input
            write_deMon2k_input(elems, current_coords, jobname='ash', filename=self.filename,
                                    functional=self.functional, Grad=Grad, charge=charge, mult=mult,
                                    scf_type=self.scf_type,
                                    basis_name=self.basis_name, auxis_name=self.auxis_name)

        #Check for BASIS and AUXIS FILES before calling
        print("Checking if BASIS file exists in current dir")
        if os.path.isfile("BASIS") is True:
            print(f"File exists in current directory: {os.getcwd()}")
        else:
            print("No file found. Trying parent dir")
            if os.path.isfile("../BASIS") is True:
                print("Found file in parent dir. Copying to current dir:", os.getcwd())
                shutil.copy(f"../BASIS", f"./BASIS")
        print("Checking if AUXIS file exists in current dir")
        if os.path.isfile("AUXIS") is True:
            print(f"File exists in current directory: {os.getcwd()}")
        else:
            print("No file found. Trying parent dir")
            if os.path.isfile("../AUXIS") is True:
                print("Found file in parent dir. Copying to current dir:", os.getcwd())
                shutil.copy(f"../AUXIS", f"./AUXIS")
        #Run deMon2k
        run_deMon2k(self.demondir,self.binary_name,self.filename,numcores=self.numcores)

        #Grab energy
        self.energy=grab_energy_demon2k(self.filename+'.out',method=self.method)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        
        #Grab gradient if calculated
        if Grad is True:
            #Grab gradient
            self.gradient = grab_gradient_deMon2k(f'ash-{self.filename}-1_0.xyz',len(current_coords))
            #Grab PCgradient from separate file
            if PC is True:
                print("not ready")

                #self.pcgradient = grab_pcgradient_deMon2k(f'{self.filename}.bqforce.dat',len(MMcharges))
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
# Independent deMon2k functions
################################

def run_deMon2k(demondir,bin_name,filename,numcores=1):
    with open(filename+'.out', 'w') as ofile:
        if numcores >1:
            process = sp.run(["mpirun", "--bind-to", "none", f"-np", f"{str(numcores)}", f"{demondir}/{bin_name}", f"{filename}.inp"], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        else:
            process = sp.run([demondir + f'/{bin_name}', filename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#Regular deMon2k input
def write_deMon2k_input(elems, coords, jobname='ash', filename='deMon', scf_type=None, tolerance=1e-8,
                        functional=None, Grad=True, charge=None, mult=None,
                        basis_name=None, auxis_name=None):
    #Energy or Energy+gradient
    if Grad is True:
        jobdirective='ENERGY_FORCE'
    else:
        jobdirective='ENERGY'
    
    #deMon2k
    with open(f"{filename}.inp", 'w') as inpfile:

        ####################
        #GLOBAL
        ####################
        inpfile.write(f'TITLE {jobname}\n')
        inpfile.write(f'CHARGE {charge}\n')
        inpfile.write(f'MULTI {mult}\n')
        inpfile.write(f'#\n')
        inpfile.write(f'SCFTYPE {scf_type} TOL={tolerance}\n')        
        inpfile.write(f'VXCTYPE {functional}\n')
        if Grad is True:
            inpfile.write(f'DYNAMICS INT=1, MAX=0, STEP=0\n')
            inpfile.write(f'TRAJECTORY FORCES\n')
            inpfile.write(f'VELOCITIES ZERO\n')
            inpfile.write(f'PRINT MD OPT\n')
        inpfile.write(f'#\n')
        #GUESS RESTART
        #POPULATION MULLIKEN
        #inpfile.write(f'PRINT MOS\n')
        #inpfile.write(f'VISUALIZATION MOLDEN FULL\n')
        inpfile.write(f'#\n')
        inpfile.write(f'# --- GEOMETRY ---\n')
        inpfile.write(f'#\n')
        inpfile.write(f'#\n')
        inpfile.write(f'GEOMETRY CARTESIAN ANGSTROM\n')
        for e,c in zip(elems,coords):
            inpfile.write(f'{e} {c[0]} {c[1]} {c[2]}\n')
        #BASIS stuff
        inpfile.write(f'AUXIS ({auxis_name})\n')
        inpfile.write(f'BASIS ({basis_name})\n')

#Grab deMon2k energy
def grab_energy_demon2k(outfile,method=None):
    energy=None
    grabline=" TOTAL ENERGY                ="
    with open(outfile) as f:
        for line in f:
            if grabline in line:
                energy=float(line.split()[-1])
    return energy

#Grab gradient from deMon2K MD-type output
def grab_gradient_deMon2k(outfile,numatoms):
    grad_grab=False
    gradient=np.zeros((numatoms,3))
    atomcount=0
    with open(outfile) as o:
        for line in o:
            if grad_grab is True:
                if len(line.split()) == 5 and 'ATOM' not in line:
                    gradient[atomcount,0] = float(line.split()[-3])
                    gradient[atomcount,1] = float(line.split()[-2])
                    gradient[atomcount,2] = float(line.split()[-1])
                    atomcount+=1
            if ' ACCELERATIONS' in line:
                grad_grab=False
            if ' GRADIENTS OF TIME STEP 0' in line:
                grad_grab=True
            if atomcount == numatoms:
                 grad_grab=False
    return gradient
