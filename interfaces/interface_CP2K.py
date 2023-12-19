import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
import ash.settings_ash
from ash.modules.module_coords import write_xyzfile, write_pdbfile
from ash.functions.functions_parallel import check_OpenMPI

#Reasonably flexible CP2K interface: CP2K input should be specified by cp2kinput multi-line string.
#Executables: cp2k.psmp vs. cp2k.sopt vs cp2k.ssmp vs cp2k_shell.ssm

#CP2K Theory object.
#CP2K embedding options: coupling='COULOMB' (regular elstat-embed) or 'GAUSSIAN' (GEEP) or S-WAVE
#periodic_type: 'XYZ' or 'NONE'
class CP2KTheory:
    def __init__(self, cp2kdir=None, filename='cp2k', printlevel=2, basis_dict=None, potential_dict=None, label="CP2K",
                cell_length=10, functional=None, psolver=None, potential_file='POTENTIAL', basis_file='BASIS_MOLOPT',
                method='QUICKSTEP', numcores=1, periodic_val=None, periodic_type='XYZ',
                coupling='COULOMB', GEEP_num_gauss=12):

        self.theorytype="QM"
        self.theorynamelabel="CP2K"
        self.label=label
        self.analytic_hessian=False
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
            print(f"Parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
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
                    print("Looking for cp2k.psmp")
                    self.cp2kdir = os.path.dirname(shutil.which('cp2k.psmp'))
                    print(BC.OKGREEN,"Found cp2k.psmp in PATH. Setting cp2kdir to:", self.cp2kdir, BC.END)
                    self.cp2k_bin_name='cp2k.psmp'
                except:
                    try:
                        print("Looking for cp2k.sopt")
                        self.cp2kdir = os.path.dirname(shutil.which('cp2k.sopt'))
                        print(BC.OKGREEN,"Found cp2k.sopt in PATH. Setting cp2kdir to:", self.cp2kdir, BC.END)
                        self.cp2k_bin_name='cp2k.sopt'
                    except:
                        print(BC.FAIL,"Found no cp2k executable in PATH. Exiting... ", BC.END)
                        ashexit()
        else:
            self.cp2kdir = cp2kdir
        

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
        self.coupling=coupling
        self.GEEP_num_gauss=GEEP_num_gauss

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
            if self.coupling == 'COULUMB':
                print("Error: Coupling option is COULOMB (regular electrostatic embedding)")
                print("This option is not compatible with CP2K GPW/GAPW option which is the main reason to use CP2K")
                print("Set coupling to GAUSS or S-WAVE instead")
                print("Exiting")
                ashexit()

            #TODO: Let's write a dummy PSF-file that contains atom indices, names, masses and pointcharge values
            #TODO: Need also to read in coordinates (XYZ or PDB)
            #TODO: Then specify in CP2K input what atoms are QM and what are MM
            #create_CP2K_pcfile_general(current_MM_coords,MMcharges, filename=self.filename)
            #pcfile=self.filename+'.pc'

            #1.Write coordinate file for whole system
            #write_xyzfile(qm_elems, current_coords, f"{self.filename}", printlevel=1)
            

            #Created PDB-file
            #print("len elems:", len(elems))
            print("len qm_elems:", len(qm_elems))
            print("len current_coords:", len(current_coords))
            print("len current_coords:", len(current_coords))
            print("len current_MM_coords:", len(current_MM_coords))
            print("len MMcharges:", len(MMcharges))
            #exit()

            #Write dummy PDB-file for CP2K
            #Warning: Special one for CP2K. Will contain QM-atoms,MM-atoms,link-atoms,dipole charges etc
            pdbfile="dummy_cp2k.pdb"
            #Create dummy fragment
            #QM-atoms (with link-atoms) + MM-pointcharges (dipole charges etc)
            dummy_elem_list = qm_elems + ['X']*len(MMcharges) 
            #print("dummy_elem_list:", dummy_elem_list)
            print("len dummy_elem_list:", len(dummy_elem_list))
            dummy_coord_list = np.concatenate((current_coords,current_MM_coords),axis=0)
            #print("dummy_coord_list:", dummy_coord_list)
            print("len dummy_coord_list:", len(dummy_coord_list))
            dummy_fragment = ash.Fragment(elems=dummy_elem_list, coords=dummy_coord_list)
            dummy_atomnames=qm_elems + ['F']*len(MMcharges)
            dummy_resnames=['QM']*len(qm_elems) +  ['MM']*len(MMcharges)  #Dummy residue names
            dummy_residlabels =[1]*len(qm_elems) +  [2]*len(MMcharges)  #Dummy resid labels
            
            #Proper
            dummy_charges = [0.0]*len(qm_elems) + MMcharges
            #Testing. Doing this avoids CP2K blow-up so the problem is the charges
            #dummy_charges = [0.0]*len(qm_elems) + [0.0]*len(MMcharges)  #Setting everything to zero for testing
            
            print("len dummy_atomnames:", len(dummy_atomnames))
            print("len dummy_resnames:", len(dummy_resnames))
            print("len dummy_residlabels:", len(dummy_residlabels))
            print("dummy_fragment numatoms", dummy_fragment.numatoms)
            print("len dummy_fragment coords", len(dummy_fragment.coords))
            print("len dummy_fragment elems", len(dummy_fragment.elems))
            write_pdbfile(dummy_fragment, outputname="dummy_cp2k", openmmobject=None, atomnames=dummy_atomnames, 
                          resnames=dummy_resnames, residlabels=dummy_residlabels, segmentlabels=None, dummyname='DUM',
                          charges_column=dummy_charges)
            

            #3. Write CP2K QM/MM inputfile
            #coupling='COULOMB' #Regular elstat-embedding. Gaussian smearing also possible, see GEEP
            #Dictionary with QM-atom indices (for full system), grouped by element
            #qm_kind_dict={'C':[1,2,3]}
            qm_kind_dict={}
            for el in set(qm_elems):
                #CP2K starts counting from 1 
                qm_kind_dict[el]= [i+1 for i, x in enumerate(qm_elems) if x == el]
            print("qm_kind_dict:",qm_kind_dict)
            write_CP2K_input(method='QMMM', jobname='ash', center_coords=True, qm_elems=qm_elems,
                             basis_dict=self.basis_dict, potential_dict=self.potential_dict,
                             functional=self.functional, restartfile=None, mgrid_commensurate=True,
                             PCfile=None, Grad=Grad, filename='cp2k', charge=charge, mult=mult,
                             periodic_val=self.periodic_val, cell_length=self.cell_length, basis_file=self.basis_file, 
                             potential_file=self.potential_file,
                             psolver=self.psolver, coupling=self.coupling, GEEP_num_gauss=self.GEEP_num_gauss,
                             qm_kind_dict=qm_kind_dict,
                             pdbfile=pdbfile)
        else:
            #No QM/MM
            #Write xyz-file with coordinates
            write_xyzfile(qm_elems, current_coords, f"{self.filename}", printlevel=1)
            #Write simple CP2K input
            write_CP2K_input(method=self.method, jobname='ash', center_coords=True, qm_elems=qm_elems,
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

        #Check for BASIS and POTENTIAL FILES before calling
        print("Checking if POTENTIAL file exists in current dir")
        if os.path.isfile("POTENTIAL") is True:
            print(f"File exists in current directory: {os.getcwd()}")
        else:
            print("No file found. Trying parent dir")
            if os.path.isfile("../POTENTIAL") is True:
                print("Found file in parent dir. Copying to current dir:", os.getcwd())
                shutil.copy(f"../POTENTIAL", f"./POTENTIAL")
            else:
                print("No file found in parent dir. Using GTHpotential file from ASH. Copying to dir as POTENTIAL")
                shutil.copyfile(ash.settings_ash.ashpath+'/basis-sets/cp2k/GTH_POTENTIALS', './POTENTIAL')
        #TODO: Need to support other basis sets (called something else than BASIS_MOLOPT)
        print("Checking if BASIS_MOLOPT file exists in current dir")
        if os.path.isfile("BASIS_MOLOPT") is True:
            print(f"File exists in current directory: {os.getcwd()}")
        else:
            print("No file found. Trying parent dir")
            if os.path.isfile("../BASIS_MOLOPT") is True:
                print("Found file in parent dir. Copying to current dir:", os.getcwd())
                shutil.copy(f"../BASIS_MOLOPT", f"./BASIS_MOLOPT")
            else:
                print("No file found in parent dir. Using basis set file from ASH. Copying to dir as BASIS_MOLOPT")
                shutil.copyfile(ash.settings_ash.ashpath+'/basis-sets/cp2k/BASIS_MOLOPT', './BASIS_MOLOPT')

        #Run CP2K
        run_CP2K(self.cp2kdir,self.cp2k_bin_name,self.filename,numcores=self.numcores)

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
                exit()
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

def run_CP2K(cp2kdir,bin_name,filename,numcores=1):
    with open(filename+'.out', 'w') as ofile:
        if numcores >1:
            process = sp.run(["mpirun", "--bind-to", "none", f"-np", f"{str(numcores)}", f"{cp2kdir}/{bin_name}", f"{filename}.inp"], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        else:
            process = sp.run([cp2kdir + f'/{bin_name}', filename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#Regular CP2K input
def write_CP2K_input(method='QUICKSTEP', jobname='ash-CP2K', center_coords=True, qm_elems=None,
                    basis_dict=None, potential_dict=None, functional=None, restartfile=None,
                    PCfile=None, Grad=True, filename='cp2k', charge=None, mult=None,
                    periodic_val=None, mgrid_commensurate=False,
                    periodic_type="XYZ", cell_length=10, basis_file='BASIS_MOLOPT', potential_file='POTENTIAL',
                    psolver='wavelet', 
                    coupling='COULOMB', GEEP_num_gauss=12,
                    qm_kind_dict=None, mm_ewald_type='NONE', mm_ewald_alpha=0.35, mm_ewald_gmax="21 21 21",
                    pdbfile=None):
    #
    first_atom=1
    last_atom=len(qm_elems)
    
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
        inpfile.write(f'      PERIODIC {periodic_type}\n') #NOTE
        inpfile.write(f'      PSOLVER {psolver}\n')
        inpfile.write(f'    &END POISSON\n')
        #SCF

        #MGRID
        inpfile.write(f'    &MGRID\n')
        inpfile.write(f'      COMMENSURATE {mgrid_commensurate}\n')
        inpfile.write(f'    &END MGRID\n')

        #PRINT stuff
        inpfile.write(f'    &PRINT\n')
        inpfile.write(f'      &MO\n')
        inpfile.write(f'        EIGENVALUES .TRUE.\n')
        inpfile.write(f'      &END MO\n')
        inpfile.write(f'    &END PRINT\n')

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
            inpfile.write(f'        DO_NONBONDED FALSE\n')
            inpfile.write(f'      &END FORCEFIELD\n')
            inpfile.write(f'      &POISSON\n')            
            inpfile.write(f'        &EWALD\n')
            inpfile.write(f'          EWALD_TYPE {mm_ewald_type}\n') 
            #mm_ewald_type=None would turn off MM-MM periodic interactions
            inpfile.write(f'          ALPHA {mm_ewald_alpha}\n')
            inpfile.write(f'          GMAX {mm_ewald_gmax}\n')
            inpfile.write(f'        &END EWALD\n')
            inpfile.write(f'      &END POISSON\n') 
            inpfile.write(f'    &END MM\n')
            #QM/MM
            inpfile.write(f'    &QMMM\n')
            inpfile.write(f'      ECOUPL {coupling}\n')
            if coupling == 'GAUSS':
                inpfile.write(f'      USE_GEEP_LIB {GEEP_num_gauss}\n')
                
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
        inpfile.write(f'      PERIODIC {periodic_type}\n')
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
        if method == 'QMMM':
            inpfile.write(f'      COORD_FILE_NAME {pdbfile}\n') #PDB-file of whole system with charges
            inpfile.write(f'      COORD_FILE_FORMAT PDB\n')
            inpfile.write(f'      CHARGE_EXTENDED TRUE\n') #Read charges from col 81
            inpfile.write(f'      CONNECTIVITY OFF\n') #No read or generate bonds 
            inpfile.write(f'      &GENERATE\n')
            inpfile.write(f'           &ISOLATED_ATOMS\n') #Topology of isolated atoms
            inpfile.write(f'               LIST {first_atom}..{last_atom}\n')
            inpfile.write(f'           &END\n')
            inpfile.write(f'      &END GENERATE\n')
        else:
            inpfile.write(f'      COORD_FILE_FORMAT xyz\n')
            inpfile.write(f'      COORD_FILE_NAME  ./{filename}.xyz\n')
        inpfile.write(f'    &END TOPOLOGY\n')
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


#Grab PC gradient
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
