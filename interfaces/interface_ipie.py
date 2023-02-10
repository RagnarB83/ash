import subprocess as sp
import time
import numpy as np
import os
import shutil
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.constants import ang2bohr, harkcal
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
from ash.functions.functions_parallel import check_OpenMPI

class ipieTheory:
    def __init__(self, pyscftheoryobject=None, filename='input.json', printlevel=2,
                numcores=1, numblocks_skip=5, dt=0.005, nwalkers=800, nsteps=25, blocks=20,
                frozencore=False, checkpointfilename='scf.chk', mcscf=False):

        self.theorynamelabel="ipie"
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")
        
        try:
            self.ipie_exe = os.path.dirname(shutil.which('ipie'))+'/ipie'
            self.pyscf_to_ipie_exe = os.path.dirname(shutil.which('pyscf_to_ipie.py'))+'/pyscf_to_ipie.py'
        except:
            print("Problem finding executables: ipie and 'pyscf_to_ipie.py' scripts in PATH. Did you install correctly? ")
            ashexit()
        try:
            import pyblock
        except ModuleNotFoundError: 
             print("Problem importing pyblock. Pleas install pyblock: pip install pyblock")
             ashexit()
        if numcores > 1:
            try:
                print(f"MPI-parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
                check_OpenMPI()
            except:
                print("Problem with mpirun")
                ashexit()

        #Indicate that this is a QMtheory
        self.theorytype="QM"
        
        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.numcores=numcores
        self.pyscftheoryobject=pyscftheoryobject

        #
        self.numblocks_skip=numblocks_skip
        self.dt=dt
        self.nwalkers=nwalkers
        self.nsteps=nsteps
        self.blocks=blocks
        self.frozencore=frozencore
        self.checkpointfilename=checkpointfilename
        self.mcscf=mcscf
        print("AFQMC settings:")
        print("MCSCF:", self.mcscf)
        print("Num walkers:", self.nwalkers)
        print("Num blocks:", self.blocks)
        print("Num steps:", self.nsteps)
        print("Timestep:", self.dt)
        print("Number of blocks to skip:", self.numblocks_skip)

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print(f"Deleting old checkpoint file: {self.checkpointfilename}")
        files=[self.checkpointfilename]
        for file in files:
            try:
                os.remove(file)
            except:
                pass
    def determine_frozen_core(self,elems):
        print("Determining frozen core")
        #Main elements 
        FC_elems={'H':0,'He':0,'Li':0,'Be':0,'B':2,'C':2,'N':2,'O':2,'F':2,'Ne':2,
        'Na':2,'Mg':2,'Al':10,'Si':10,'P':10,'S':10,'Cl':10,'Ar':10,
        'K':10,'Ca':10,'Sc':10,'Ti':10,'V':10,'Cr':10,'Mn':10,'Fe':10,'Co':10,'Ni':10,'Cu':10,'Zn':10,
        'Ga':18,'Ge':18,'As':18,'Se':18, 'Br':18, 'Kr':18}
        #NOTE: To be updated for 4d TM row etc
        num_el=0
        for el in elems:
            num_el+=FC_elems[el]
        self.frozen_core_el=num_el
        self.frozen_core_orbs=int(num_el/2)
        print("Total frozen electrons in system:", self.frozen_core_el)
        print("Total frozen orbitals in system:", self.frozen_core_orbs)

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, numcores=None, restart=False, label=None,
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

        #Cleanup before run.
        #Delete old PySCF checkpointfile, otherwise checkpointfile might contain multiple stuff
        self.cleanup()


        #Run PySCF to get integrals
        self.pyscftheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)

        #Read PySCF checkpointfile and create ipie inputfile
        if self.frozencore is True:
            self.determine_frozen_core(qm_elems)
            print("Calling pyscf_to_ipie_exe")
            if self.mcscf is True:
                print("MCSCF option true.")
                sp.call([self.pyscf_to_ipie_exe ,'-i', self.checkpointfilename, '-j', self.filename, '--mcscf', '--frozen-core', str(self.frozen_core_orbs)])
            else:
                sp.call([self.pyscf_to_ipie_exe ,'-i', self.checkpointfilename, '-j', self.filename, '--frozen-core', str(self.frozen_core_orbs)])
        else:
            if self.mcscf is True:
                print("MCSCF option true.")
                sp.call([self.pyscf_to_ipie_exe ,'-i', self.checkpointfilename, '-j', self.filename, '--mcscf'])
            else:
                sp.call([self.pyscf_to_ipie_exe ,'-i', self.checkpointfilename, '-j', self.filename])

        #Modify inputfile
        f = open(self.filename,'r')
        new=[]
        for line in f.readlines():
            if 'blocks' in line:
                new.append(f"\"blocks\": {self.blocks},\n")
            elif 'nwalkers' in line:
                new.append(f"\"nwalkers\": {self.nwalkers},\n")
            elif 'dt' in line:
                new.append(f"\"dt\": {self.dt},\n")
            elif 'nsteps' in line:
                new.append(f"\"nsteps\": {self.nsteps},\n")
            elif 'trial' in line:
                new.append(line)
                new.append(f"\"calculate_variational_energy\": true,\n")
            else:
                new.append(line)
        f.close()
        g = open(self.filename,'w')
        for i in new:
            g.write(i)
        g.close()

        #Parallel
        if self.numcores > 1:
            print(f"Running ipie with MPI parallelization ({self.numcores} MPI processes)")
            with open('output.dat', "w") as outfile:
                sp.call(['mpirun', '-np', str(self.numcores), self.ipie_exe, self.filename], stdout=outfile)
        #Serial
        else:
            print("Running ipie in serial")
            with open('output.dat', "w") as outfile:
                sp.call([self.ipie_exe, self.filename], stdout=outfile)

        print("ipie finished")
        ##Analysis
        with open('output.dat', "a") as outfile:
            sp.call(['reblock.py','-b', str(self.numblocks_skip), '-f', 'output.dat'], stdout=outfile)
        
        E_final, error, nsamp_ac = grab_ipie_energy('output.dat')

        print("Final ipie AFQMC energy:", E_final)
        print(f"Error: {error} Eh ({error*harkcal:.2f} kcal/mol)")
        print("nsamp_ac:", nsamp_ac)
        self.energy=E_final
        self.error=error

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        return self.energy


def grab_ipie_energy(outputfile):
    grab=False
    with open (outputfile) as ofile:
        for line in ofile:
            if grab is True:
                E_final=float(line.split()[0])
                error=float(line.split()[1])
                nsamp_ac=int(line.split()[2])
            if 'ETotal_ac  ETotal_error_ac' in line:
                grab=True
    return E_final, error, nsamp_ac