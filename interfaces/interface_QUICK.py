import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader

#QUICK Theory object.
#TODO: Add pointcharges to input and grab PC gradients
class QUICKTheory:
    def __init__(self, quickdir=None, filename='quick', printlevel=2,
                quickinput=None, numcores=1):

        self.theorynamelabel="QUICK"
        print_line_with_mainheader("QUICKTheory initialization")

        if quickinput is None:
            print(f"{self.theorynamelabel}Theory requires a quickinput keyword")
            ashexit()
        self.quickdir=None
        if quickdir == None:
            print(BC.WARNING, f"No quickdir argument passed to {self.theorynamelabel}Theory. Attempting to find quickdir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", settings_ash.settings_dict)
                self.quickdir=settings_ash.settings_dict["quickdir"]
            except:
                print(BC.WARNING,"Found no quickdir variable in settings_ash module either.",BC.END)
                try:
                    self.quickdir = os.path.dirname(shutil.which('quick.cuda'))
                    print(BC.OKGREEN,"Found quick.cuda in PATH. Setting quickdir to:", self.quickdir, BC.END)
                except:
                    print(BC.FAIL,"Found no quick.cuda executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.quickdir = quickdir

        #Indicate that this is a QMtheory
        self.theorytype="QM"

        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.quickinput=quickinput
        self.numcores=numcores
    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup():
        print("QUICK cleanup not yet implemented.")
    #TODO: Parallelization is enabled most easily by OMP_NUM_THREADS AND MKL_NUM_THREADS. NOt sure if we can control this here


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
            print(BC.FAIL, "Error. charge and mult has not been defined for QUICKTheory.run method", BC.END)
            ashexit()

        print("Job label:", label)
        print("Creating inputfile: quick.inp")
        print("QUICK input:")
        print(self.quickinput)

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
        if Grad==True:
            if PC is True:
                write_quick_input(self.quickinput,charge,mult,qm_elems,current_coords,Grad=True, pc_coords=current_MM_coords,pc_values=MMcharges)
            else:
                write_quick_input(self.quickinput,charge,mult,qm_elems,current_coords,Grad=True)
            
            #Run QUICK
            run_quick(self.quickdir,self.filename+'.out')

            self.energy=grab_energy_quick(self.filename+'.out')
            if PC is True:
                self.gradient,pcgradient = grab_gradient_quick(self.filename+'.out',len(current_coords), numpc=len(MMcharges))
            else:
                self.gradient,self.pcgradient = grab_gradient_quick(self.filename+'.out',len(current_coords))
            print("self.gradient", self.gradient)
            print("pcgradient:", self.pcgradient)
        else:
            write_quick_input(self.quickinput,charge,mult,qm_elems,current_coords,Grad=False)
            run_quick(self.quickdir,self.filename+'.out')
            self.energy=grab_energy_quick(self.filename+'.out')

        #TODO: write in error handling here
        print(BC.OKBLUE, BC.BOLD, "------------ENDING QUICK INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point QUICK energy:", self.energy)
            print_time_rel(module_init_time, modulename='QUICK run', moduleindex=2)
            if PC is True:
                return self.energy, self.gradient, self.pcgradient
            else:
                return self.energy, self.gradient
        else:
            print("Single-point QUICK energy:", self.energy)
            print_time_rel(module_init_time, modulename='QUICK run', moduleindex=2)
            return self.energy

#NOT tested
def run_quick(quickdir,filename):
    process = sp.run([quickdir + '/quick.cuda'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#functional,basis,charge,mult,elems,coords,cutoff=1e-8,Grad=True
def write_quick_input(quickinputline,charge,mult,elems,coords,pc_coords=None, pc_values=None, Grad=True):
    pckeyword=""
    gradkeyword=""
    if Grad is True:
        gradkeyword="GRADIENT"
    if pc_coords is not None:
        pckeyword="EXTCHARGES"
    with open("quick.in", 'w') as inpfile:
        #inpfile.write(f"{functional} BASIS={basis} {gradkeyword} CHARGE={charge} cutoff={cutoff}" + '\n')
        inpfile.write(f"{quickinputline} {gradkeyword} CHARGE={charge} {pckeyword}")
        inpfile.write('\n')
        inpfile.write('\n')
        for el,c in zip(elems,coords):
            inpfile.write('{}   {} {} {}\n'.format(el,c[0],c[1],c[2]))
        inpfile.write('\n')
        if pc_coords is not None:
            for c,v in zip(pc_coords,pc_values):
                inpfile.write(f'{c[0]} {c[1]} {c[2]} {v}\n')
        inpfile.write('\n')

def grab_energy_quick(outfile):
    #Option 1. Grabbing all lines containing energy in outputfile. Take last entry.
    # CURRENT Option 2: grab energy from iface file. Higher level WF entry should be last
    energy=None
    with open(outfile) as f:
        for line in f:
            if ' TOTAL ENERGY' in line:
                energy=float(line.split()[-1])
                print(energy)
    return energy


def grab_gradient_quick(outfile,numatoms,numpc=None):
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
                if ' COORDINATE    XYZ            GRADIENT' in line:
                    continue
                if 'X' in line:
                    gradient[atomcount,0] = float(line.split()[-1])
                elif 'Y' in line:
                    gradient[atomcount,1] = float(line.split()[-1])
                elif 'Z' in line:
                    gradient[atomcount,2] = float(line.split()[-1])
                    atomcount+=1
                else:
                    continue
            if pcgrad_grab is True:
                if ' COORDINATE    XYZ            GRADIENT' in line:
                    continue
                if 'X' in line:
                    pc_gradient[pccount,0] = float(line.split()[-1])
                elif 'Y' in line:
                    pc_gradient[pccount,1] = float(line.split()[-1])
                elif 'Z' in line:
                    pc_gradient[pccount,2] = float(line.split()[-1])
                    pccount+=1
                else:
                    continue
                if pccount == numpc:
                    pcgrad_grab=False
            if ' ANALYTICAL GRADIENT:' in line:
                grad_grab=True
            if ' POINT CHARGE GRADIENT:' in line:
                pcgrad_grab=True
            if atomcount == numatoms:
                 grad_grab=False
    return gradient, pc_gradient