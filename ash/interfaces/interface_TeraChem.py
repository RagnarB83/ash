import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
from ash.modules.module_coords import write_xyzfile
import ash.settings_ash

# TeraChem Theory object.
class TeraChemTheory:
    def __init__(self, terachemdir=None, filename='terachem', printlevel=2, label="TeraChem",
                teracheminput=None, numcores=1):

        self.theorynamelabel="TeraChem"
        self.theorytype="QM"
        self.analytic_hessian=False
        self.label=label
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        if teracheminput is None:
            print(f"{self.theorynamelabel}Theory requires a teracheminput keyword")
            ashexit()
        self.terachemdir=None
        if terachemdir == None:
            print(BC.WARNING, f"No terachemdir argument passed to {self.theorynamelabel}Theory. Attempting to find terachemdir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.terachemdir=ash.settings_ash.settings_dict["terachemdir"]
            except:
                print(BC.WARNING,"Found no terachemdir variable in settings_ash module either.",BC.END)
                try:
                    self.terachemdir = os.path.dirname(shutil.which('terachem'))
                    print(BC.OKGREEN,"Found terachem in PATH. Setting terachemdir to:", self.terachemdir, BC.END)
                except:
                    print(BC.FAIL,"Found no terachem executable in PATH. Exiting... ", BC.END)
                    #ashexit()
        else:
            self.terachemdir = terachemdir

        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.teracheminput=teracheminput
        self.numcores=numcores
    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup():
        print(f"self.theorynamelabel cleanup not yet implemented.")

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)
        print(f"Creating inputfile: {self.filename}.in")
        print(f"{self.theorynamelabel} input:")
        print(self.teracheminput)

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

        # Write coordinates to disk as XYZ-file
        write_xyzfile(qm_elems, current_coords, "terachem", printlevel=1)

        # Write PC to disk
        if PC is True:
            create_terachem_pcfile_general(current_MM_coords,MMcharges, filename=self.filename)

        # Grab energy and gradient
        if Grad:
            if PC is True:
                write_terachem_input(self.teracheminput,charge,mult,qm_elems,current_coords,
                    Grad=True, PCfile=self.filename+'.pc', xyzfilename=self.filename+'.xyz',filename=self.filename)
            else:
                write_terachem_input(self.teracheminput,charge,mult,qm_elems,current_coords,Grad=True, filename=self.filename)

            # Run Terachem
            run_terachem(self.terachemdir,self.filename)

            self.energy=grab_energy_terachem(self.filename+'.out')
            if PC is True:
                self.gradient,self.pcgradient = grab_gradient_terachem(self.filename+'.out',len(current_coords), numpc=len(MMcharges))
            else:
                self.gradient,self.pcgradient = grab_gradient_terachem(self.filename+'.out',len(current_coords))
        else:
            if PC is True:
                write_terachem_input(self.teracheminput,charge,mult,qm_elems,current_coords,Grad=False, PCfile=self.filename+'.pc')
            else:
                write_terachem_input(self.teracheminput,charge,mult,qm_elems,current_coords,Grad=False)

            #Run Terachem
            run_terachem(self.terachemdir,self.filename)

            self.energy=grab_energy_terachem(self.filename+'.out')

        # PC-correction
        # Terachem includes PC-PC interaction, we need to correct energy and pcgradient
        if PC:
            print("Terachem energy (before correction)", self.energy)
            # pc_pc_energy = ash.modules.module_coords.nuc_nuc_repulsion(current_MM_coords, MMcharges)
            # print("old PC-PC energy:", pc_pc_energy)
            # TODO: Benchmark how fast this is, add Julia option
            #Note: Using cupy by default, requires GPU (requirement for Terachem anyway)
            curr_time = time.time()
            pc_selfen, pc_selfgrad = ash.modules.module_MM.coulombcharge(MMcharges, current_MM_coords, mode="cupy")
            print_time_rel(curr_time, modulename=f'PC-E+G correction', moduleindex=2)
            print("PC-PC self-energy:", pc_selfen)
            self.energy = self.energy - pc_selfen
            print("Terachem energy (after PC-PC selfenergy subtraction)", self.energy)
            if Grad:
                self.pcgradient = self.pcgradient - pc_selfgrad

        # TODO: write in error handling here
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        if Grad:
            print(f"Single-point {self.theorynamelabel} energy:", self.energy)
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            if PC is True:
                return self.energy, self.gradient, self.pcgradient
            else:
                return self.energy, self.gradient
        else:
            print(f"Single-point {self.theorynamelabel} energy:", self.energy)
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy

def run_terachem(terachemdir,filename):
    with open(filename+'.out', 'w') as ofile:
        process = sp.run([terachemdir + '/terachem', filename+'.in'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#functional,basis,charge,mult,elems,coords,cutoff=1e-8,Grad=True
def write_terachem_input(teracheminput,charge,mult,elems,coords,xyzfilename="terachem.xyz", filename='terachem',
    PCfile=None, Grad=True):
    pckeyword="no"
    if PCfile is not None:
        pckeyword=PCfile
    joboption="energy"
    if Grad is True:
        joboption="gradient"
    with open(f"{filename}.in", 'w') as inpfile:
        inpfile.write('#Terachem input\n')
        inpfile.write(f'coordinates {xyzfilename}\n')
        inpfile.write(f'charge {charge}\n')
        inpfile.write(f'spinmult {mult}\n')
        for key in teracheminput:
          inpfile.write('{0} {1}\n'.format(key, teracheminput[key]))
        inpfile.write(f'run {joboption}\n')
        inpfile.write(f'pointcharges {pckeyword}\n')
        inpfile.write(f'timings yes\n')
        inpfile.write('end\n')
        inpfile.write('\n')


def grab_energy_terachem(outfile):
    #Option 1. Grabbing all lines containing energy in outputfile. Take last entry.
    # CURRENT Option 2: grab energy from iface file. Higher level WF entry should be last
    energy=None
    with open(outfile) as f:
        for line in f:
            if 'FINAL ENERGY:' in line:
                energy=float(line.split()[2])
    return energy


def grab_gradient_terachem(outfile,numatoms,numpc=None):
    grad_grab=False
    pcgrad_grab=False

    gradient=np.zeros((numatoms,3))
    pc_gradient=None
    if numpc is not None:
        pc_gradient=np.zeros((numpc,3))
    atomcount=0
    pccount=0
    with open(outfile) as o:
        for i,line in enumerate(o):
            if grad_grab is True:
                if '------' in line:
                    continue
                if len(line.split()) == 3:
                    gradient[atomcount,0] = float(line.split()[0])
                    gradient[atomcount,1] = float(line.split()[1])
                    gradient[atomcount,2] = float(line.split()[2])
                    atomcount+=1
                else:
                    continue
            if pcgrad_grab is True:
                if len(line.split()) == 3:
                    pc_gradient[pccount,0] = float(line.split()[0])
                    pc_gradient[pccount,1] = float(line.split()[1])
                    pc_gradient[pccount,2] = float(line.split()[2])
                    pccount+=1
                if pccount == numpc:
                    pcgrad_grab=False
            if 'Net gradient' in line:
                grad_grab=False
                pcgrad_grab=False
            if 'dE/dX' in line:
                grad_grab=True
            if '------- MM / Point charge part' in line:
                pcgrad_grab=True
            if atomcount == numatoms:
                 grad_grab=False
    return gradient, pc_gradient


# pc-coords in Å
def create_terachem_pcfile_general(coords,pchargelist,filename='terachem'):
    with open(filename+'.pc', 'w') as pcfile:
        pcfile.write(str(len(pchargelist))+'\n')
        pcfile.write('\n')
        for p,c in zip(pchargelist,coords):
            line = "{} {} {} {}".format(p, c[0], c[1], c[2])
            pcfile.write(line+'\n')
