import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
import ash.settings_ash
from ash.modules.module_coords import write_xyzfile
from ash.functions.functions_parallel import check_OpenMPI

#Reasonably flexible CP2K interface: CP2K input should be specified by cp2kinput multi-line string.
#cp2k.sopt vs cp2k.ssmp vs cp2k_shell.ssmp


#CP2K Theory object.
class CP2KTheory:
    def __init__(self, cp2kdir=None, filename='cp2k', printlevel=2, basis_dict=None, potential_dict=None,
                cell_length=10, functional=None, psolver=None, potential_file='POTENTIAL', basis_file='BASIS_MOLOPT',
                method='QUICKSTEP', numcores=1, periodic_val=None):

        self.theorynamelabel="CP2K"
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        #EARLY EXITS
        if basis_dict is None:
            print("basis_dict keyword is required")
            ashexit()
        if potential_dict is None:
            print("potential_dict keyword is required")
            ashexit()
        if functional is None:
            print("functional keyword is required")
            ashexit()

        #Checking OpenMPI
        if numcores != 1:
            print(f"ORCA parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version (for the ORCA version) is available in your environment")
            check_OpenMPI()
        #Finding CP2K
        if cp2kdir == None:
            print(BC.WARNING, f"No cp2kdir argument passed to {self.theorynamelabel}Theory. Attempting to find cp2kdir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.cp2kdir=ash.settings_ash.settings_dict["cp2kdir"]
            except:
                print(BC.WARNING,"Found no cp2kdir variable in settings_ash module either.",BC.END)
                try:
                    self.cp2kdir = os.path.dirname(shutil.which('cp2k.sopt'))
                    print(BC.OKGREEN,"Found cp2k.sopt in PATH. Setting cp2kdir to:", self.cp2kdir, BC.END)
                except:
                    print(BC.FAIL,"Found no cp2k executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.cp2kdir = cp2kdir
        
        #Indicate that this is a QMtheory
        self.theorytype="QM"

        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        #Defining inputfile
        self.basis_dict=basis_dict
        self.potential_dict=potential_dict
        self.cell_length=cell_length
        self.functional=functional
        self.basis_file=basis_file
        self.potential_file=potential_file
        self.psolver=psolver
        self.numcores=numcores
        self.method=method
        self.periodic_val=periodic_val


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

        #Case: QM/MM CP2K job
        if PC is True:
            print("pc true")
            #TODO: Let's write a dummy PSF-file that contains atom indices, names, masses and pointcharge values
            #TODO: Need also to read in coordinates (XYZ or PDB)
            #TODO: Then specify in CP2K input what atoms are QM and what are MM
            #create_CP2K_pcfile_general(current_MM_coords,MMcharges, filename=self.filename)
            #pcfile=self.filename+'.pc'

            #1.Write coordinate file for whole system
            write_xyzfile(qm_elems, current_coords, f"{self.filename}", printlevel=1)
            #2. Write dummy PSF file for whole system that contains MM charges?
            #2b. Alternative write dummy PDB-file for whole system that contains MM charges in some column

            #3. Write CP2K QM/MM inputfile
            coupling='COULOMB' #Regular elstat-embedding. Gaussian smearing also possible, see GEEP
            #Dictionary with QM-atom indices (for full system), grouped by element
            qm_kind_dict={'C':[1,2,3]}
            write_CP2K_input(method='QMMM', jobname='ash', center_coords=True,
                             basis_dict=self.basis_dict, potential_dict=self.potential_dict,
                             functional=self.functional, restartfile=None,
                             PCfile=None, Grad=Grad, filename='cp2k', charge=charge, mult=mult,
                             periodic_val=self.periodic_val, cell_length=self.cell_length, basis_file=self.basis_file, 
                             potential_file=self.potential_file,
                             psolver=self.psolver, coupling=coupling, qm_kind_dict=qm_kind_dict)
        else:
            #No QM/MM
            #Write xyz-file with coordinates
            write_xyzfile(qm_elems, current_coords, f"{self.filename}", printlevel=1)
            #Write simple CP2K input
            write_CP2K_input(method=self.method, jobname='ash', center_coords=True,
                             basis_dict=self.basis_dict, potential_dict=self.potential_dict,
                             functional=self.functional, restartfile=None,
                             PCfile=None, Grad=Grad, filename='cp2k', charge=charge, mult=mult,
                             periodic_val=self.periodic_val, cell_length=self.cell_length, basis_file=self.basis_file, potential_file=self.potential_file,
                             psolver=self.psolver)

        #Delete old forces file if present
        try:
            os.remove(f'ash-{self.filename}-1_0.xyz')
        except:
            pass

        #Run CP2K
        run_CP2K(self.cp2kdir,self.filename,numcores=self.numcores)

        #Grab energy
        self.energy=grab_energy_cp2k(self.filename+'.out',method=self.method)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        
        #Grab gradient if calculated
        if Grad is True:
            #Grab gradient
            self.gradient = grab_gradient_CP2K(f'ash-{self.filename}-1_0.xyz',len(current_coords))
            #Grab PCgradient from separate file
            if PC is True:
                print("not ready")

                #self.pcgradient = grab_pcgradient_CP2K(f'{self.filename}.bqforce.dat',len(MMcharges))
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
# Independent CP2K functions
################################

def run_CP2K(cp2kdir,filename,numcores=1):
    with open(filename+'.out', 'w') as ofile:
        if numcores >1:
            process = sp.run(['mpirun', '-np', str(numcores), cp2kdir + '/cp2k.sopt', filename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        else:
            process = sp.run([cp2kdir + '/cp2k.sopt', filename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#Regular CP2K input
def write_CP2K_input(method='QUICKSTEP', jobname='ash', center_coords=True,
                    basis_dict=None, potential_dict=None, functional=None, restartfile=None,
                    PCfile=None, Grad=True, filename='cp2k', charge=None, mult=None,
                    periodic_val=None, cell_length=10, basis_file='BASIS_MOLOPT', potential_file='POTENTIAL',
                    psolver='wavelet', 
                    coupling='COULOMB', qm_kind_dict=None):
    #Energy or Energy+gradient
    if Grad is True:
        jobdirective='ENERGY_FORCE'
    else:
        jobdirective='ENERGY'
    
    #Make sure we center coordinates if wavelet
    if psolver == 'wavelet':
        print("psolver is wavelet. Coordinates must be centered")
        center_coords=True

    #CP2K wants coordinates centered (currently default by ASH), can be avoided but:
    #NOTE: The CP2K wavelet solver requires atoms to be centered. Force projection would be required
    #NOTE: The MT  solver does not requires this. But not as good? Seems to be faster
    #User should not input &global but should have force-eval
    with open(f"{filename}.inp", 'w') as inpfile:

        ####################
        #GLOBAL
        ####################
        inpfile.write(f'&GLOBAL\n')
        inpfile.write(f'  PROJECT {jobname}\n')
        inpfile.write(f'  RUN_TYPE {jobdirective}\n')
        inpfile.write(f'  PRINT_LEVEL MEDIUM\n')
        inpfile.write(f'&END GLOBAL\n\n')

        ####################
        #FORCE_EVAL
        ####################
        inpfile.write(f'&FORCE_EVAL\n')
        inpfile.write(f'  METHOD {method}\n')
        inpfile.write(f'  &PRINT\n')
        inpfile.write(f'    &FORCES\n')
        inpfile.write(f'      FILENAME {filename}\n')
        inpfile.write(f'    &END FORCES\n')
        inpfile.write(f'  &END PRINT\n\n')

        ##########
        #DFT
        ##########
        inpfile.write(f'  &DFT\n')
        inpfile.write(f'    CHARGE {charge}\n')
        if mult > 1:
            inpfile.write(f'    UKS\n')
        inpfile.write(f'    MULTIPLICITY {mult}\n')
        inpfile.write(f'    BASIS_SET_FILE_NAME {basis_file}\n')
        inpfile.write(f'    POTENTIAL_FILE_NAME {potential_file}\n')
        if restartfile != None:
            inpfile.write(f'    WFN_RESTART_FILE_NAME {restartfile}\n')
        #POISSON
        inpfile.write(f'    &POISSON\n')
        inpfile.write(f'      PERIODIC {periodic_val}\n')
        inpfile.write(f'      PSOLVER {psolver}\n')
        inpfile.write(f'    &END POISSON\n')
        #SCF

        #XC
        inpfile.write(f'    &XC\n')
        inpfile.write(f'      &XC_FUNCTIONAL {functional}\n')
        inpfile.write(f'      &END XC_FUNCTIONAL\n')
        inpfile.write(f'    &END XC\n')

        inpfile.write(f'  &END DFT\n\n')

        #QM/MM
        if method == 'QMMM':
            #MM
            inpfile.write(f'    &MM\n')
            inpfile.write(f'      &FORCEFIELD\n')
            inpfile.write(f'        PARM_FILE_NAME {PCfile}\n')
            inpfile.write(f'        PARMTYPE CHM\n')
            inpfile.write(f'        VDW_SCALE14 0.0\n')
            inpfile.write(f'        EI_SCALE14 0.0\n')
            inpfile.write(f'      &END FORCEFIELD\n')
            inpfile.write(f'      &POISSON\n')            
            inpfile.write(f'        &EWALD\n')
            inpfile.write(f'          GMAX 25 25 25\n')
            inpfile.write(f'        &END EWALD\n')
            inpfile.write(f'      &END POISSON\n') 
            inpfile.write(f'    &END MM\n')
            #QM/MM
            inpfile.write(f'    &QMMM\n')
            inpfile.write(f'      ECOUPL {coupling}\n')
            inpfile.write(f'      &CELL \n')
            inpfile.write(f'        ABC 25.0 25.0 25.0 \n')
            inpfile.write(f'      &END CELL\n')
            #Write QM_KIND blocks
            for qm_el,indices in qm_kind_dict.items():
                inpfile.write(f'      &QM_KIND {qm_el}\n')
                indices_string = ' '.join([str(i) for i in indices])
                inpfile.write(f'          MM_INDEX {indices_string}\n')
                inpfile.write(f'      &END QM_KIND\n')
            inpfile.write(f'    &END QMMM\n')

        ##########
        #SUBSYS
        ##########
        inpfile.write(f'  &SUBSYS\n')

        #CELL BLOCK
        inpfile.write(f'    &CELL\n')
        inpfile.write(f'      ABC {cell_length} {cell_length} {cell_length}\n')
        inpfile.write(f'      PERIODIC {periodic_val}\n')
        inpfile.write(f'    &END CELL\n')

        #KIND: basis and potentail for each element
        for el in basis_dict.keys():
            inpfile.write(f'    &KIND {el}\n')
            inpfile.write(f'      ELEMENT {el}\n')
            inpfile.write(f'      BASIS_SET {basis_dict[el]}\n')
            inpfile.write(f'      POTENTIAL {potential_dict[el]}\n')
            inpfile.write(f'    &END KIND\n')
        inpfile.write(f'\n')
        #TOPOLOGY BLOCK
        inpfile.write(f'    &TOPOLOGY\n')
        if center_coords is True:
            inpfile.write(f'      &CENTER_COORDINATES\n')
            inpfile.write(f'      &END\n')
        #if method == 'QMMM':
        #    inpfile.write(f'      CONN_FILE_FORMAT CHM\n')
        #    inpfile.write(f'      CONN_FILE_NAME  ./{filename}.psf\n')
        inpfile.write(f'      COORD_FILE_FORMAT xyz\n')
        inpfile.write(f'      COORD_FILE_NAME  ./{filename}.xyz\n')
        inpfile.write(f'    &END\n')
        inpfile.write(f'  &END SUBSYS\n')

        inpfile.write(f'&END FORCE_EVAL\n')

#Grab CP2K energy
def grab_energy_cp2k(outfile,method=None):
    energy=None
    grabline=" ENERGY| Total"
    with open(outfile) as f:
        for line in f:
            if grabline in line:
                energy=float(line.split()[-1])
    return energy

#TODO: CP2K may append to this file if run again. Delete ?
def grab_gradient_CP2K(outfile,numatoms):
    grad_grab=False
    gradient=np.zeros((numatoms,3))
    atomcount=0
    with open(outfile) as o:
        for line in o:
            if grad_grab is True:
                if len(line.split()) == 6:
                    gradient[atomcount,0] = -1*float(line.split()[-3])
                    gradient[atomcount,1] = -1*float(line.split()[-2])
                    gradient[atomcount,2] = -1*float(line.split()[-1])
                    atomcount+=1
            if ' SUM OF' in line:
                grad_grab=False
            if ' # Atom   Kind   Element' in line:
                grad_grab=True
            if atomcount == numatoms:
                 grad_grab=False
    return gradient


#Grab PC gradient from NWchem written file
def grab_pcgradient_CP2K(pcgradfile,numpc):
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
def create_CP2K_pcfile_general(coords,pchargelist,filename='cp2k'):
    with open(filename+'.pc', 'w') as pcfile:
        pcfile.write(str(len(pchargelist))+'\n')
        pcfile.write('\n')
        for p,c in zip(pchargelist,coords):
            line = "{} {} {} {}".format(p, c[0], c[1], c[2])
            pcfile.write(line+'\n')