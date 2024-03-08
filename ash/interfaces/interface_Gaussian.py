import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
import ash.settings_ash

# Very basic interface to Gaussian.

class GaussianTheory:
    def __init__(self, gaussiandir=None, gauss_executable='g16', filename='gaussian', file_extension='.com', printlevel=2, label="Gaussian",
                gaussianinput=None, memory='800MB', numcores=1):

        self.theorynamelabel="Gaussian"
        self.label=label
        self.theorytype="QM"
        self.analytic_hessian=False
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        if gaussianinput is None:
            print(f"{self.theorynamelabel}Theory requires a gaussianinput keyword")
            ashexit()

        # Parallelization: Gaussian uses shared-memory parallelism (controlled via nprocshared). No support for Linda yet.
        # Numcores will be used to set nprocshared in Gaussian input file.

        # Finding Gaussian
        self.gauss_executable=gauss_executable
        self.file_extension=file_extension
        if gaussiandir is None:
            print(BC.WARNING, f"No gaussiandir argument passed to {self.theorynamelabel}Theory. Attempting to find gaussiandir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.gaussiandir=ash.settings_ash.settings_dict["gaussiandir"]
            except:
                print(BC.WARNING,"Found no gaussiandir variable in settings_ash module either.",BC.END)
                print("Searching for g16 executable in PATH")
                try:
                    self.gaussiandir = os.path.dirname(shutil.which('g16'))
                    print(BC.OKGREEN,"Found g16 in PATH. Setting gaussiandir to:", self.gaussiandir, BC.END)
                    self.gauss_executable='g16'
                except:
                    print("Did not find g16. Searching for g09 executable in PATH")
                    try:
                        self.gaussiandir = os.path.dirname(shutil.which('g09'))
                        print(BC.OKGREEN,"Found g09 in PATH. Setting gaussiandir to:", self.gaussiandir, BC.END)
                        self.gauss_executable='g09'
                    except:
                        print(BC.FAIL,"Found no gaussian executable in PATH. Exiting... ", BC.END)
                        ashexit()
        else:
            self.gaussiandir = gaussiandir
            print("Gaussian dir provided:", self.gaussiandir)
            print(f"Making sure gauss_executable {self.gauss_executable} is in PATH.")

        # Setting Gaussian environment variables
        os.environ['GAUSS_EXEDIR'] = self.gaussiandir
        print("Setting GAUSS_EXEDIR to:", self.gaussiandir)
        os.environ['GAUSS_SCRDIR'] = '.'
        print("Setting GAUSS_SCRDIR to: ", os.getcwd())

        # CHECKS if input contains disallowed keywords
        if 'opt' in gaussianinput.lower():
            print(BC.FAIL, f"Error. You should not input an optimization keyword in gaussianinput-string. Exiting...", BC.END)
            ashexit()
        if 'freq' in gaussianinput.lower():
            print(BC.FAIL, f"Error. You should not input a freq keyword in gaussianinput-string. Exiting...", BC.END)
            ashexit()
        if 'force' in gaussianinput.lower():
            print(BC.FAIL, f"Error. You should not input a force keyword in gaussianinput-string. Exiting...", BC.END)
            ashexit()
        if 'scan' in gaussianinput.lower():
            print(BC.FAIL, f"Error. You should not input a scan keyword in gaussianinput-string. Exiting...", BC.END)
            ashexit()

        # Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.gaussianinput=gaussianinput
        self.memory=memory
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
        print(f"Creating inputfile: {self.filename}{self.file_extension}")
        print(f"{self.theorynamelabel} input:")
        print(self.gaussianinput)

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

        # Write inputfile
        write_Gaussian_input(self.gaussianinput,charge,mult,qm_elems,current_coords, memory=self.memory, MMcharges=MMcharges, MMcoords=current_MM_coords,
                Grad=Grad, PC=PC, filename=self.filename, file_extension=self.file_extension, numcores=self.numcores)

        # Run Gaussian
        print(self.gauss_executable)
        run_Gaussian(self.gaussiandir,gauss_exe=self.gauss_executable, filename=self.filename, file_extension=self.file_extension)

        # Grab energy
        self.energy_dict=grab_energy_gaussian(self.filename+'.log')
        for k,v in self.energy_dict.items():
            if v is not None:
                print(f"{k}: {v}")
        self.energy=self.energy_dict['total_energy']

        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Grab gradient if calculated
        if Grad is True:
            # Grab gradient
            self.gradient = grab_gradient_Gaussian(self.filename+'.log',len(current_coords))
            # Grab PCgradient from separate file
            if PC is True:
                print("A gradient calculation with PCs has been requested. Unfortunately, point charge gradients are not available")
                ashexit()
                #self.pcgradient = grab_pcgradient_Gaussian(f'{self.filename}.bqforce.dat',len(MMcharges))
                print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
                return self.energy, self.gradient, self.pcgradient
            else:
                print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
                return self.energy, self.gradient
        # Returning energy without gradient
        else:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy


################################
# Independent Gaussian functions
################################

def run_Gaussian(gaussiandir, gauss_exe=None, file_extension=None, filename='gaussian'):
    # Note: using .out to grab any stdout and stderr from program. Default .log is used for actual output
    with open(filename+'.out', 'w') as ofile:
        process = sp.run([gaussiandir + f'/{gauss_exe}', filename+file_extension], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

def write_Gaussian_input(gaussianinput,charge,mult,elems,coords, filename='gaussian', file_extension=None, memory='800MB', numcores=1,
                        PC = False, Grad=True, MMcharges=None, MMcoords=None):

    # Grad
    if Grad is True:
        gaussianinput += ' force'

    # Avoid reorientation of molecule
    if 'nosym' not in gaussianinput:
        gaussianinput += ' nosymm'

    # Pointcharge embedding
    if PC is True:
        gaussianinput += ' charge'

    with open(f"{filename}{file_extension}", 'w') as inpfile:
        # Todo: Gaussian-header lines
        inpfile.write(f'%mem={memory}\n')
        inpfile.write(f'%chk={filename}.chk\n')
        inpfile.write(f'%NProcShared={numcores}\n')

        # Turning off symmetry by default (important for ASH opt and QM/MM).
        # TODO: Re-enable symmetry if wanted by user? Useful for using symmetry in expensive CC calculations
        inpfile.write(gaussianinput+'\n')
        inpfile.write('\n') #empty line
        inpfile.write('ASH-created Gaussian input\n') #Title line
        inpfile.write('\n') #empty line
        inpfile.write(f'{charge} {mult}\n') #empty line
        for el,c in zip(elems,coords):
            inpfile.write(f'{el } {c[0]} {c[1]} {c[2]}\n')

        inpfile.write('\n')

        # Pointcharge embedding
        if PC is True:
            for MMc,MMxyz in zip(MMcharges,MMcoords):
                inpfile.write(f'{MMxyz[0]} {MMxyz[1]} {MMxyz[2]} {MMc}\n')
        inpfile.write('\n')


# Grab Gaussian energy
def grab_energy_gaussian(outfile):
    energy_dict = {'total_energy': None, 'scf_energy': None,
                   'mp2_energy': None, 'mp2corr_energy': None,
                   'ccsd_energy': None, "ccsd_t_energy": None}

    # Note: Grabbing all possible SCF, MP2, CCSD, CCSD(T) energies
    # Setting total-energy to the last found energy

    with open(outfile) as f:
        for line in f:
            if 'SCF Done:' in line:
                energy_dict["scf_energy"]=float(line.split()[4])
                energy_dict["total_energy"]=energy_dict["scf_energy"]
            if ' E2 =' in line:
                energy_dict["mp2corr_energy"]=float(line.split()[2])
                energy_dict["mp2_energy"]=float(line.split()[-1])
                energy_dict["total_energy"]=energy_dict["mp2_energy"]
            if 'E(CORR)=' in line:
                energy_dict["ccsd_energy"]=float(line.split()[3])
                energy_dict["total_energy"]=energy_dict["ccsd_energy"]
            if 'CCSD(T)=' in line:
                energy_dict["ccsd_t_energy"]=float(line.split()[-1])
                energy_dict["total_energy"]=energy_dict["ccsd_t_energy"]
    return energy_dict


def grab_gradient_Gaussian(outfile, numatoms):
    grad_grab = False
    gradient = np.zeros((numatoms, 3))
    atomcount = 0
    with open(outfile) as o:
        for i, line in enumerate(o):
            if grad_grab is True:
                if 'Number' in line or '----' in line:
                    continue
                gradient[atomcount, 0] = -1 * float(line.split()[-3])
                gradient[atomcount, 1] = -1 * float(line.split()[-2])
                gradient[atomcount, 2] = -1 * float(line.split()[-1])
                atomcount += 1
            if 'Cartesian' in line:
                grad_grab = False
            if ' Center     Atomic                   Forces (Hartrees/Bohr)' in line:
                grad_grab = True
            if atomcount == numatoms:
                grad_grab = False
    return gradient

# QM/MM: TODO

#Grab PC gradient from Gaussian written file
# def grab_pcgradient_Gaussian(pcgradfile,numpc):
#     pc_gradient=np.zeros((numpc,3))
#     pccount=0
#     with open(pcgradfile) as o:
#         for i,line in enumerate(o):
#             if '#' not in line:
#                 pc_gradient[pccount,0] = float(line.split()[0])
#                 pc_gradient[pccount,1] = float(line.split()[1])
#                 pc_gradient[pccount,2] = float(line.split()[2])
#                 pccount+=1
#     if pccount != numpc:
#         print("Problem grabbing PC gradient from file:", pcgradfile)
#         ashexit()
#     return pc_gradient

# # Write PC-coords and charges in Å
# def create_Gaussian_pcfile_general(coords,pchargelist,filename='gaussian'):
#     with open(filename+'.pc', 'w') as pcfile:
#         pcfile.write(str(len(pchargelist))+'\n')
#         pcfile.write('\n')
#         for p,c in zip(pchargelist,coords):
#             line = "{} {} {} {}".format(p, c[0], c[1], c[2])
#             pcfile.write(line+'\n')
