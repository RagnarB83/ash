import subprocess as sp
import os
import shutil
import time
import re
import pprint
import json

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader

#PyMBE Theory object.
class PyMBETheory:
    def __init__(self, pymbedir=None, filename='pymbe_', printlevel=2,
                pymbedict=None,pymbeinput=None, numcores=1):

        print_line_with_mainheader("PyMBETheory initialization")

        if pymbeinput is None and pymbedict is None:
            print("PyMBTheory requires a pymbeinput or pymbedict keyword")
            ashexit()

        if pymbedir == None:
            print(BC.WARNING, "No pymbedir argument passed to PyMBETheory. Attempting to find pymbedir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", settings_ash.settings_dict)
                self.pymbedir=settings_ash.settings_dict["pymbedir"]
            except:
                print(BC.WARNING,"Found no pymbedir variable in settings_ash module either.",BC.END)
                ashexit()
                #try:
                #    self.pymbedir = os.path.dirname(shutil.which('dmrcc'))
                #    print(BC.OKGREEN,"Found dmrcc in PATH. Setting mrccdir to:", self.mrccdir, BC.END)
                #except:
                #    print(BC.FAIL,"Found no dmrcc executable in PATH. Exiting... ", BC.END)
                #    ashexit()
        else:
            self.pymbedir = pymbedir

        #Checking if pyscf is present
        #TODO: Avoid importing since PyMBE 
        print("Checking if pyscf exists")
        #try:
        #    import pyscf
        #    print("PySCF is present")
        #except:
        #    print("PyMBE requires installation of pyscf")
        #    print("Try e.g. : pip install pyscf")
        #    ashexit()
        #Checking if mpi4py is present
        print("Checking if mpi4py exists")
        try:
            import mpi4py
            print("mpi4py is present")
        except:
            print("PyMBE requires installation of mpi4py Python library.")
            print("Try e.g. : conda install mpi4py")
            ashexit()

        #Indicate that this is a QMtheory
        self.theorytype="QM"

        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.pymbeinput=pymbeinput
        self.pymbedict=pymbedict
        self.numcores=numcores

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("PyMBE cleanup not yet implemented.")
    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if Grad==True:
            print("Grad not available")
            ashexit()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING PyMBE INTERFACE-------------", BC.END)
        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for PyMBE.run method", BC.END)
            ashexit()

        print("Running PyMBE object.")
        #TODO: Need to finish parallelization
        print("Job label:", label)
        print("Creating inputfile: input")
        print("PyMBE input:")
        if self.pymbeinput is not None:
            print(self.pymbeinput)
        if self.pymbedict is not None:
            print(self.pymbedict)
        
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
        
        #Write inputfile
        write_pymbe_input(pymbeinput=self.pymbeinput,pymbedict=self.pymbedict,
                charge=charge,mult=mult,elems=qm_elems,coords=current_coords)

        #Needed for PyMBE run
        print("Setting environment variable PYTHONHASHSEED to 0.")
        os.environ["PYTHONHASHSEED"] = "0"
        
        #Running
        run_pymbe(self.pymbedir,self.filename+'.out', numcores=numcores)
        print()
        
        #Printing results
        print("PyMBE results")
        mbe_results_dict=grab_results_pymbe()
        pprint.pprint(mbe_results_dict)
        print()
        #print("mbe_results_dict:", mbe_results_dict)
        print("Final MBE-FCI results:\n")
        print("MBE order  Total E         Corr. E")
        for totE,corrE,label in zip(mbe_results_dict["tot_energies"],mbe_results_dict["corr_energies"],mbe_results_dict["labels"]):
            print(f"{label:>5s} {totE:14.7f} {corrE:14.7f}")
        print()
        
        #Setting final energy as last
        self.energy=mbe_results_dict["MBE total energy"]
        #TODO: write in error handling here
        print(BC.OKBLUE, BC.BOLD, "------------ENDING PyMBE INTERFACE-------------", BC.END)
        print("Single-point PyMBE energy:", self.energy)
        print_time_rel(module_init_time, modulename='PyMBE run', moduleindex=2)
        return self.energy




def run_pymbe(pymbedir,filename, numcores=1):
    with open(filename, 'w') as ofile:
        if numcores == 1:
            print("Running PyMBE using 1 core")
            process = sp.run([pymbedir + '/src/main.py'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        else:
            print(f"Running PyMBE using {numcores} cores using external MPI4PY mpiexec command")
            process = sp.run(['mpiexec','-np', str(numcores), pymbedir + '/src/main.py'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#Silly: Writing Python-script inputfile to disk
def write_pymbe_input(coords=None, elems=None, pymbeinput=None, pymbedict=None,charge=None,mult=None,):
    with open("input", 'w') as inpfile:
        inpfile.write("#!/usr/bin/env python\n")
        inpfile.write("# -*- coding: utf-8 -*\n\n")
        inpfile.write("atom = '''\n")
        for el,c in zip(elems,coords):
            inpfile.write('{}   {} {} {}\n'.format(el,c[0],c[1],c[2]))
        inpfile.write("'''\n")
        #Write inputstring
        unpaired_els=mult-1
        if pymbeinput is not None:
            for inpline in pymbeinput.split('\n'):
                if 'system' in inpline:
                    #Adding spin info
                    systemline=inpline.replace('}',f', \'charge\':{charge}, \'spin\':{unpaired_els} }}')
                    inpfile.write(systemline + '\n')
                else:
                    inpfile.write(inpline + '\n')
        #Write dictionary to file input
        if pymbedict is not None:
            for subd_key,subd_val in pymbedict.items():
                if 'system' in subd_key:
                    #Adding spin info
                    systemline=str(subd_val).replace('}',f', \'charge\':{charge}, \'spin\':{unpaired_els} }}')
                    inpfile.write(str(subd_key)+' = '+systemline + '\n')
                else:
                    inpfile.write(str(subd_key)+' = '+str(subd_val) + '\n')
        inpfile.write('\n')


#Grab results from the PymBE output dir
def grab_results_pymbe():
    grab_E=False
    final_dict={'tot_energies':[], 'corr_energies':[], 'labels':[]}
    #TODO: Only supporting 1 root right now
    with open("output/pymbe.results") as f:
        for line in f:
            if 'basis set           =' in line:
                final_dict["basis-set"] = line.split()[3]
            if 'frozen core         =' in line:
                final_dict["frozen-core"] = line.split()[3]
            if 'system size         =' in line:
                final_dict["system-size"] = line.split()[3:8]
            if 'state (mult.)       =' in line:
                final_dict["state-mult"] = line.split()[3:5]
            if 'orbitals            =' in line:
                final_dict["orbitals"] = line.split()[2:4]
            if 'FCI solver          =' in line:
                final_dict["FCI-solver"] = line.split()[3]
            if 'expansion model  =' in line:
                final_dict["expansion-model"] = line.split()[8]
            if 'reference        =' in line:
                final_dict["reference"] = line.split()[7]
            if 'reference space  =' in line:
                final_dict["reference-space"] = line.split()[12:16]
            if 'reference orbs.  =' in line:
                final_dict["reference-orbs"] = line.split()[9]
            if 'base model       =' in line:
                final_dict["base-model"] = line.split()[8]
            
            if ' Hartree-Fock energy     =' in line:
                final_dict["HF_E"] = float(line.split()[-1])
            if 'base model energy       =' in line:
                final_dict["Base model energy"] = float(line.split()[-1])
            if 'MBE total energy        =' in line:
                final_dict["MBE total energy"] = float(line.split()[-1])
            if ' wave funct. symmetry' in line:
                final_dict["symmetry"] = line.split()[-1]
            #Final energy table:
            if grab_E is True:
                if '         ref     |' in line:
                    final_dict["labels"].append('ref')
                    final_dict["tot_energies"].append(float(line.split()[2]))
                    final_dict["corr_energies"].append(float(line.split()[4]))
                if re.match("          [0-9]      \|",line) is not None:
                    final_dict["labels"].append(line.split()[0])
                    final_dict["tot_energies"].append(float(line.split()[2]))
                    final_dict["corr_energies"].append(float(line.split()[4]))
            if 'MBE-FCI energy (root = 0)' in line:
                grab_E=True
    return final_dict
