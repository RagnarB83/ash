import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
import ash.settings_ash
from ash.functions.functions_parallel import check_OpenMPI

# Reasonably flexible NWChem interface: NWChem input should be specified by nwcheminput multi-line string.
# Everything except geometry
# Works for RHF,UHF,RKS,UKS, MP2, CCSD(T) w/wo TCE. PC available: energy and gradient
# TODO: Constrained DFT
# TODO: GPU support for CC
# TODO: ADFT

# NWChem Theory object.
class NWChemTheory:
    def __init__(self, nwchemdir=None, filename='nwchem', openshell=False, printlevel=2, label="NWChem",
                nwcheminput=None, method='scf', tce=False, numcores=1):

        self.theorynamelabel="NWChem"
        self.label=label
        self.theorytype="QM"
        self.analytic_hessian=False
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        if nwcheminput is None:
            print(f"{self.theorynamelabel}Theory requires a nwcheminput keyword")
            ashexit()

        #Checking OpenMPI
        if numcores != 1:
            print(f"Parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
            check_OpenMPI()
        #Finding NWChem
        if nwchemdir == None:
            print(BC.WARNING, f"No nwchemdir argument passed to {self.theorynamelabel}Theory. Attempting to find nwchemdir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.nwchemdir=ash.settings_ash.settings_dict["nwchemdir"]
            except:
                print(BC.WARNING,"Found no nwchemdir variable in settings_ash module either.",BC.END)
                try:
                    self.nwchemdir = os.path.dirname(shutil.which('nwchem'))
                    print(BC.OKGREEN,"Found nwchem in PATH. Setting nwchemdir to:", self.nwchemdir, BC.END)
                except:
                    print(BC.FAIL,"Found no nwchem executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.nwchemdir = nwchemdir

        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.nwcheminput=nwcheminput
        self.numcores=numcores
        self.openshell=openshell
        self.method=method
        self.tce=tce

    #Set numcores method
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
        print(f"Creating inputfile: {self.filename}.nw")
        print(f"{self.theorynamelabel} input:")
        print(self.nwcheminput)

        #Force openshell if mult > 1
        if mult > 1:
            self.openshell=True

        # if dft input then method is DFT
        if 'dft' in self.nwcheminput:
            print("DFT job detected. method set to dft instead of scf")
            self.method='dft'
        elif 'ccsd' in self.method:
            if self.openshell is True:
                print("Regular CC module not available for open-shell. Switching to TCE module")
                self.tce = True
        #Otherwise
        else:
            #If not set then probably HF => scf
            if self.method == None:
                self.method='scf'


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

        #Write PC to disk
        if PC is True:
            print("pc true")
            create_NWChem_pcfile_general(current_MM_coords,MMcharges, filename=self.filename)
            pcfile=self.filename+'.pc'
        else:
            pcfile=None

        #Write inputfile
        write_NWChem_input(self.nwcheminput,charge,mult,qm_elems,current_coords, method=self.method,
                Grad=Grad, PCfile=pcfile, filename=self.filename, openshell=self.openshell, tce=self.tce)

        #Run NWChem
        run_NWChem(self.nwchemdir,self.filename,numcores=self.numcores)

        #Grab energy
        self.energy=grab_energy_nwchem(self.filename+'.out',method=self.method, tce=self.tce)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        #Grab gradient if calculated
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
# Independent NWChem functions
################################

def run_NWChem(nwchemdir,filename,numcores=1):
    with open(filename+'.out', 'w') as ofile:
        if numcores >1:
            process = sp.run(['mpirun', '-np', str(numcores), nwchemdir + '/nwchem', filename+'.nw'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        else:
            process = sp.run([nwchemdir + '/nwchem', filename+'.nw'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

def write_NWChem_input(nwcheminput,charge,mult,elems,coords, filename='nwchem',
    PCfile=None, Grad=True, method='scf', openshell=False, tce=False):
    if Grad is True:
        jobdirective='gradient'
    else:
        jobdirective='energy'
    with open(f"{filename}.nw", 'w') as inpfile:
        inpfile.write('#NWChem input\n')
        inpfile.write(f'echo\n')
        inpfile.write(f'start {filename}\n')
        inpfile.write(f'charge {charge}\n')
        #Turning off symmetry by default (important for ASH opt and QM/MM).
        #TODO: Re-enable symmetry if wanted by user? Useful for using symmetry in expensive CC calculations
        inpfile.write(f'geometry nocenter noautosym units angstrom noautoz print\n')
        for el,c in zip(elems,coords):
            inpfile.write(f'{el } {c[0]} {c[1]} {c[2]}\n')
        inpfile.write(f'end\n')
        #Pointcharge embedding
        if PCfile != None:
            inpfile.write(f'bq\n')
            #Loading PC file. Telling NWChem about format
            inpfile.write(f'load {PCfile} format 2 3 4 1\n')
            #PC forces
            inpfile.write(f'force\n')
            inpfile.write(f'end\n')

        if openshell is True:
            #HF or post-HF case
            if method == 'scf' or method =='mp2' or 'ccsd' in method or 'tce' in method:
                inpfile.write(f'scf\n')
                inpfile.write(f'uhf doublet\n')
                inpfile.write(f'nopen {mult-1}\n')
                inpfile.write(f'end\n')
            #DFT
            elif method == 'dft':
                inpfile.write(f'dft\n')
                inpfile.write(f'odft\n')
                inpfile.write(f'mult {mult}\n')
                inpfile.write(f'end\n')
            else:
                print("unknown method")
                ashexit()
        inpfile.write(nwcheminput)
        #Write job directive
        if tce is True:
            inpfile.write(f"task tce {jobdirective}\n")
        else:
            inpfile.write(f"task {method} {jobdirective}\n")
        inpfile.write('\n')

#Grab NWChem energy
def grab_energy_nwchem(outfile,method=None,tce=False):
    energy=None
    if method == 'scf':
        grabline='Total SCF energy ='
    elif method == 'mp2':
        grabline='Total MP2 energy'
    elif method == 'ccsd' and tce is True:
        grabline='CCSD total energy'
    elif method == 'ccsd(t)' and tce is True:
        grabline='CCSD(T) total energy'
    elif method == 'ccsd':
        grabline='Total CCSD energy:'
    elif method == 'ccsd(t)':
        grabline='Total CCSD(T) energy:'
    elif method == 'dft':
        grabline='Total DFT energy ='
    else:
        print("Unknown method")
        ashexit()
    with open(outfile) as f:
        for line in f:
            if grabline in line:
                energy=float(line.split()[-1])
    return energy

def grab_gradient_NWChem(outfile,numatoms):
    grad_grab=False
    gradient=np.zeros((numatoms,3))
    atomcount=0
    with open(outfile) as o:
        for i,line in enumerate(o):
            if grad_grab is True:
                if 'x' in line:
                    continue
                gradient[atomcount,0] = float(line.split()[-3])
                gradient[atomcount,1] = float(line.split()[-2])
                gradient[atomcount,2] = float(line.split()[-1])
                atomcount+=1
            if 'Net gradient' in line:
                grad_grab=False
            if 'atom               coordinates                        gradient' in line:
                grad_grab=True
            if atomcount == numatoms:
                 grad_grab=False
    return gradient


#Grab PC gradient from NWchem written file
def grab_pcgradient_NWChem(pcgradfile,numpc):
    pc_gradient=np.zeros((numpc,3))
    pccount=0
    with open(pcgradfile) as o:
        for i,line in enumerate(o):
            if '#' not in line:
                pc_gradient[pccount,0] = float(line.split()[0])
                pc_gradient[pccount,1] = float(line.split()[1])
                pc_gradient[pccount,2] = float(line.split()[2])
                pccount+=1
    if pccount != numpc:
        print("Problem grabbing PC gradient from file:", pcgradfile)
        ashexit()
    return pc_gradient

# Write PC-coords and charges in Å
def create_NWChem_pcfile_general(coords,pchargelist,filename='nwchem'):
    with open(filename+'.pc', 'w') as pcfile:
        pcfile.write(str(len(pchargelist))+'\n')
        pcfile.write('\n')
        for p,c in zip(pchargelist,coords):
            line = "{} {} {} {}".format(p, c[0], c[1], c[2])
            pcfile.write(line+'\n')
