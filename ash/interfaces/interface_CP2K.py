import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
import ash.settings_ash
from ash.modules.module_coords import write_xyzfile, write_pdbfile,cubic_box_size,bounding_box_dimensions
from ash.functions.functions_parallel import check_OpenMPI

# Dictionary of element radii in Angstrom for use with CP2K for GEEP embedding
# Warning:Parameters for O, H, K and Cl found online. Rest are guestimates.
# Note: Also seen 1.2 used for O
element_radii_for_cp2k = {'H': 0.44, 'He': 0.44, 'Li': 0.6, 'Be': 0.6,
                          'B': 0.78, 'C': 0.78, 'N': 0.78, 'O': 0.78,
                          'F': 0.78, 'Ne': 0.78, 'Na': 1.58, 'Mg': 1.58,
                          'Al': 1.67, 'Si': 1.67, 'P': 1.67, 'S': 1.67,
                          'Cl': 1.67, 'Ar': 1.67, 'K': 1.52, 'Ca': 1.6,
                          'Sc': 1.6, 'Ti': 1.6, 'V': 1.6, 'Cr': 1.6, 'Mn': 1.6,
                          'Fe': 1.6, 'Co': 1.6, 'Ni': 1.6, 'Cu': 1.6,
                          'Zn': 1.6, 'Br': 1.6, 'Mo': 1.7}

# CP2K Theory object.
# CP2K embedding options: coupling='COULOMB' (regular elstat-embed) or 'GAUSSIAN' (GEEP) or S-WAVE
# Periodic: True (periodic_type='XYZ) or False (periodic_type='NONE')
# NOTE: Currently not supporting 2D or 1D periodicity
# Psolvers: 'PERIODIC', 'MT', 'MULTIPOLE' or 'wavelet'
# For Periodic=True: 'PERIODIC', 'wavelet' or 'IMPLICIT'
# For Periodic=False: 'MT' or 'wavelet' are good options

# MT Solver (noPBC): Cell should be 2 as large as charge density. MT incompatible with GEEP and PBC
# Wavelet Solver (noPBC or PBC): Cell can be smaller. Compatible with GEEP. Also PBC
# Periodic Solver (only PBC): Cell can be smaller. Compatible with GEEP. Fast

# Basis_method='GAPW' : all-electron or pseudopotential. More stable forces, might be more expensive. Not all features available
# 'GPW' : only pseudopotential. Can be more efficient.
# 'XTB'
class CP2KTheory:
    def __init__(self, cp2kdir=None, cp2k_bin_name=None, filename='cp2k', printlevel=2, basis_dict=None, potential_dict=None, label="CP2K",
                periodic=False, periodic_type='XYZ', qm_periodic_type=None, xtb_periodic=False, cell_dimensions=None, cell_vectors=None,
                qm_cell_dims=None, qm_cell_shift_par=6.0, wavelet_scf_type=40,
                functional=None, psolver='wavelet', potential_file='POTENTIAL', basis_file='BASIS',
                basis_method='GAPW', ngrids=4, cutoff=250, rel_cutoff=60,
                method='QUICKSTEP', numcores=1, parallelization='OMP', mixed_mpi_procs=None, mixed_omp_threads=None,
                center_coords=True, scf_maxiter=50, outer_scf_maxiter=10, scf_convergence=1e-6, eps_default=1e-10,
                coupling='GAUSSIAN', GEEP_num_gauss=6, MM_radius_scaling=1, mm_radii=None,
                OT=True, OT_minimizer='DIIS', OT_preconditioner='FULL_ALL',
                OT_linesearch='3PNT', outer_SCF=True, outer_SCF_optimizer='SD', OT_energy_gap=0.08):

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

        #NOTE: We still define a cell even though we may not be periodic
        #If no cell provided: CONTINUE and guess cell size later
        if cell_dimensions is None and cell_vectors is None:
            print("Warning: Neither cell_dimensions or cell_vectors have been provided.")
            print("This is not good but ASH will continue and try to guess the cell size from the QM-coordinates")

        if cell_dimensions is not None and cell_vectors is not None:
            print("Error: cell_dimensions and cell_vectors can not both be provided")
            ashexit()
        #PERIODIC logic
        if periodic is True:
            print("Periodic is True")
            self.periodic_type=periodic_type
            print("Periodic type:", self.periodic_type)
            if psolver.upper() == 'MT':
                print("Error: For periodic simulations the Poisson solver (psolver) can not be MT.")
                ashexit()
        else:
            print("Periodic is False")
            self.periodic_type='NONE'
            print("PERIODIC_TYPE:", self.periodic_type)

        #Parallelization
        self.numcores=numcores
        #Type of parallelization strategy. 'OMP','MPI','Mixed'
        self.parallelization=parallelization
        self.mixed_mpi_procs=mixed_mpi_procs #Mixed only:
        self.mixed_omp_threads=mixed_omp_threads #Mixed only:

        #Finding CP2K dir and binaries
        self.cp2kdir, self.cp2k_bin_name = find_cp2k(cp2kdir,cp2k_bin_name)

        #Checking OpenMPI
        if self.numcores != 1:
            print(f"Parallel job requested with numcores: {self.numcores}")
            print(f"Parallelization strategy: {self.parallelization}")
            if parallelization == 'Mixed':
                if self.mixed_mpi_procs is None or self.mixed_omp_threads is None:
                    print("Error: mixed_mpi_procs and mixed_omp_threads keywords must be provided for Mixed parallelization")
                    ashexit()
                if 'psmp' not in self.cp2k_bin_name:
                    print("Error: Mixed parallelization only possible with cp2k.psmp")
                    ashexit()
                print("Mixed parallelization requested. Will use", self.mixed_mpi_procs, "MPI processes and", self.mixed_omp_threads, "OpenMP threads")
            elif parallelization == 'OMP':
                if 'ssmp' not in self.cp2k_bin_name:
                    if 'psmp' not in self.cp2k_bin_name:
                        print("Error: OMP parallelization requires either ssmp or psmp binary")
                        ashexit()
                print(f"OpenMP parallelization requested. 1 CP2K process will be launched with {numcores} OpenMP threads")
            elif parallelization == 'MPI':
                if 'popt' not in self.cp2k_bin_name:
                    if 'psmp' not in self.cp2k_bin_name:
                        print("Error: MPI parallelization requires either popt or psmp binary")
                        ashexit()
                print("MPI parallelization requested. Will use", self.numcores, "MPI processes")
                print("Make sure that the correct OpenMPI version is available in your environment")
                check_OpenMPI()
            else:
                print("Unknown parallelization choice")
                ashexit()
        else:
            print("Numcores=1. No parallelization of CP2K requested")
            self.parallelization=None

        # Printlevel
        self.printlevel=printlevel
        self.filename=filename

        # Methods
        self.method=method #Can be QUICKSTEP or QMMM
        self.basis_method = basis_method #GAPW or GPW
        # Basis and PP stuff
        self.basis_dict=basis_dict
        self.basis_file=basis_file
        self.potential_dict=potential_dict
        self.potential_file=potential_file

        # Periodic options and cell
        self.periodic=periodic
        self.psolver=psolver
        self.wavelet_scf_type=wavelet_scf_type
        self.qm_periodic_type=qm_periodic_type
        self.xtb_periodic=xtb_periodic # Boolean, xtB Ewald True or False
        # self.cell_length=cell_length #Total cell length (full system including MM if QM/MM)
        self.cell_dimensions=cell_dimensions #Cell dimensions. For full system
        self.cell_vectors=cell_vectors #Cell vectors. For full system
        self.qm_cell_dims=qm_cell_dims # Optional QM-cell dims (only if QM/MM)
        self.qm_cell_shift_par=qm_cell_shift_par #Shift of QM-cell size if estimated from QM-coords
        self.functional=functional
        self.center_coords=center_coords

        #SCF onvergence stuff
        self.OT=OT
        self.OT_minimizer=OT_minimizer
        self.OT_preconditioner=OT_preconditioner
        self.OT_linesearch=OT_linesearch
        self.outer_SCF=outer_SCF
        self.outer_SCF_optimizer=outer_SCF_optimizer
        self.OT_energy_gap=OT_energy_gap

        #Grid stuff
        self.ngrids=ngrids
        self.cutoff=cutoff
        self.rel_cutoff=rel_cutoff
        self.scf_convergence=scf_convergence
        self.scf_maxiter=scf_maxiter
        self.outer_scf_maxiter=outer_scf_maxiter
        self.eps_default=eps_default

        #QM/MM
        self.coupling=coupling
        self.GEEP_num_gauss=GEEP_num_gauss
        self.MM_radius_scaling=MM_radius_scaling #Scale the MM radii by this factor
        self.mm_radii=mm_radii #Optional dictionary of MM radii for GEEP. If not provided then element_radii_for_cp2k

        print("CP2KTheory settings:\n")
        print("")
        print("CP2K directory:", self.cp2kdir)
        print("CP2K binary name:", self.cp2k_bin_name)
        print("Method:", self.method)
        print("Functional:", self.functional)
        print("Basis method:", self.basis_method)
        print("Basis file:", self.basis_file)
        print("Potential file:", self.potential_file)
        print("Cutoff:", self.cutoff)
        print("Relative cutoff:", self.rel_cutoff)
        print("")
        print("Periodic:", self.periodic)
        print("Periodic type:", self.periodic_type)
        print("QM periodic type:", self.qm_periodic_type)
        print("XTB periodic:", self.xtb_periodic)
        print("Cell dimensions:", self.cell_dimensions)
        print("Cell vectors:", self.cell_vectors)
        print("QM cell dimensions:", self.qm_cell_dims)
        print("QM cell shift par:", self.qm_cell_shift_par)
        print("Wavelet SCF type:", self.wavelet_scf_type)
        print("")
        print("Printlevel:", self.printlevel)
        print("Parallelization:", self.parallelization)
        print("Numcores:", self.numcores)
        print("")
        print("SCF convergence:", self.scf_convergence)
        print("SCF maxiter:", self.scf_maxiter)
        print("Outer SCF maxiter:", self.outer_scf_maxiter)
        print("EPS default:", self.eps_default)
        print("OT:", self.OT)
        print("OT minimizer:", self.OT_minimizer)
        print("OT preconditioner:", self.OT_preconditioner)
        print("OT linesearch:", self.OT_linesearch)
        print("Outer SCF:", self.outer_SCF)
        print("Outer SCF optimizer:", self.outer_SCF_optimizer)
        print("OT energy gap:", self.OT_energy_gap)


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

        #PERIODIC and QM/MM
        if self.periodic is True:
            print("Periodic CP2K calculation will be carried out")
            if PC is True:
                print("QM/MM is on")
                #Periodic True and QM/MM: Need to check qm_periodic_type
                if self.qm_periodic_type is None:
                    print("Periodic and QM/MM is both on but qm_periodic_type is None")
                    print("Setting qm_periodic_type to XYZ (sets periodicity in QM-cell)")
                    self.qm_periodic_type='XYZ'

        print("QM periodic type:", self.qm_periodic_type)
        print("Poisson solver", self.psolver)

        print_time_rel(module_init_time, modulename=f'CP2K run-prep1', moduleindex=2)
        #Case: QM/MM CP2K job
        if PC is True:
            print("PC true")
            if self.coupling == 'COULUMB':
                print("Error: Coupling option is COULOMB (regular electrostatic embedding)")
                print("This option is not compatible with CP2K GPW/GAPW option which is the main reason to use CP2K")
                print("Set coupling to GAUSS or S-WAVE instead")
                print("Exiting")
                ashexit()
            elif self.coupling == 'GAUSS':
                print("Using Gaussian GEEP embedding")
                print("Number of Gaussians used:", self.GEEP_num_gauss)
                print("Scaling MM radii by factor:", self.MM_radius_scaling)
            else:
                print("Unknown coupling option. Exiting")
                ashexit()
            #
            if self.cell_dimensions is None and self.cell_vectors is None:
                print("Error: cell_dimensions and cell_vectors have not been provided")
                print("This is required for a QM/MM job. Exiting")
                ashexit()
            elif self.cell_dimensions is not None:
                print("cell_dimensions:", self.cell_dimensions)
            if self.cell_vectors is not None:
                print("cell_vectors:", self.cell_vectors)

            print_time_rel(module_init_time, modulename=f'CP2K run-prep2', moduleindex=2)
            #QM-CELL
            if self.qm_cell_dims is None:
                print("Warning: QM-cell box dimensions have not been set by user (qm_cell_dims keyword)")
                print("Now estimating QM-cell box dimensions from QM-coordinates.")
                #If doing non-periodic wavelet then we must have a cubic box.
                if self.psolver == 'wavelet':
                    print("Poisson solver is wavelet. Making cubic box")
                    qm_cell = cubic_box_size(current_coords)
                    qm_box_dims=np.array([qm_cell,qm_cell,qm_cell])
                else:
                    print("Poisson solver not wavelet. Using non-cubic box")
                    qm_box_dims = bounding_box_dimensions(current_coords)
                print(f"QM-box size (based on QM-coords): {qm_box_dims} Angstrom")
                print(f"Adding shift of {self.qm_cell_shift_par} Angstrom (qm_cell_shift_par keyword)")
                qm_box_dims=np.around(qm_box_dims + self.qm_cell_shift_par,1)
                print(f"Setting QM-cell cubic box dimensions to {qm_box_dims} Angstrom")
                self.qm_cell_dims=qm_box_dims

            #1.Writing special XYZ-file coordinate-file (not PDB-file, coordinate precision is problematic)
            #NOTE: Coordinates reordered: QM-coords first, then MM-coords
            #Applies to elem-list also and charges-list
            dummy_elem_list = qm_elems + mm_elems

            dummy_coords = np.concatenate((current_coords,current_MM_coords),axis=0)
            dummy_charges = [0.0]*len(qm_elems) + MMcharges
            system_xyzfile="system_cp2k"
            print_time_rel(module_init_time, modulename=f'CP2K run-prep3a', moduleindex=2)
            write_xyzfile(dummy_elem_list, dummy_coords, f"{system_xyzfile}", printlevel=1)
            print_time_rel(module_init_time, modulename=f'CP2K run-prep3b', moduleindex=2)
            #Telling CP2K which atoms are QM
            #Dictionary with QM-atom indices (for full system), grouped by element
            qm_kind_dict={}
            for el in set(qm_elems):
                #CP2K starts counting from 1
                qm_kind_dict[el]= [i+1 for i, x in enumerate(qm_elems) if x == el]
            print("qm_kind_dict:",qm_kind_dict)
            #MM-kind list
            #NOTE: This controls Gaussian radii for GEEP. Currently using same parameters for element
            #Could be made more flexible
            mm_kind_list = list(set(mm_elems))
            print("mm_kind_list:",mm_kind_list)

            #TODO: Unclear whether dipole-charges added by ASH will work at all for CP2K GEEP
            #If not then we should detect this and exit and tell user to turn off in QMMMTheory

            #2. Write charges.inc file containing charge definitions
            #File will be included in CP2K inputfile
            charge_lines = [f"{d}\n" for d in dummy_charges]
            all_charges_lines=["&CHARGES\n"] + charge_lines + ["&END CHARGES\n"]
            with open("charges.inc", 'w') as incfile:
                incfile.writelines(all_charges_lines)

            print_time_rel(module_init_time, modulename=f'CP2K run-prep4c', moduleindex=2)
            #3. Write CP2K QM/MM inputfile
            write_CP2K_input(method='QMMM', jobname='ash', center_coords=self.center_coords, qm_elems=qm_elems,
                             basis_dict=self.basis_dict, potential_dict=self.potential_dict,
                             basis_method=self.basis_method, wavelet_scf_type=self.wavelet_scf_type,
                             functional=self.functional, restartfile=None, mgrid_commensurate=True,
                             Grad=Grad, filename='cp2k', charge=charge, mult=mult,
                             coordfile=system_xyzfile,
                             cell_dimensions=self.cell_dimensions,
                             cell_vectors=self.cell_vectors,
                             qm_cell_dims=self.qm_cell_dims, qm_periodic_type=self.qm_periodic_type,
                             xtb_periodic=self.xtb_periodic,
                             basis_file=self.basis_file,
                             potential_file=self.potential_file, periodic_type=self.periodic_type,
                             psolver=self.psolver, coupling=self.coupling, GEEP_num_gauss=self.GEEP_num_gauss,
                             MM_radius_scaling=self.MM_radius_scaling, mm_radii=self.mm_radii,
                             qm_kind_dict=qm_kind_dict, mm_kind_list=mm_kind_list,
                             scf_convergence=self.scf_convergence, eps_default=self.eps_default,
                             scf_maxiter=self.scf_maxiter, outer_scf_maxiter=self.outer_scf_maxiter,
                             ngrids=self.ngrids, cutoff=self.cutoff, rel_cutoff=self.rel_cutoff, printlevel=self.printlevel,
                             OT=self.OT, OT_minimizer=self.OT_minimizer, OT_preconditioner=self.OT_preconditioner, OT_linesearch=self.OT_linesearch,
                             outer_SCF=self.outer_SCF, outer_SCF_optimizer=self.outer_SCF_optimizer, OT_energy_gap=self.OT_energy_gap)
        else:
            #No QM/MM
            #QM-CELL
            if self.cell_dimensions is None:
                print("Warning: cell dimensions have not been set by user")
                print("Now estimating cell box dimensions from the system oordinates.")
                if self.psolver == 'wavelet':
                    print("Poisson solver is wavelet. Making cubic box")
                    qm_cell = cubic_box_size(current_coords)
                    qm_box_dims=np.array([qm_cell,qm_cell,qm_cell])
                else:
                    print("Poission solver not wavelet. Using non-cubic box")
                    qm_box_dims = bounding_box_dimensions(current_coords)
                print(f"Cell box size (based on coords): {qm_box_dims} Angstrom")
                print(f"Adding shift of {self.qm_cell_shift_par} Angstrom (qm_cell_shift_par keyword)")
                qm_box_dims=np.around(qm_box_dims + self.qm_cell_shift_par,1)
                print(f"Setting Cell box dimensions to {qm_box_dims} Angstrom")
                self.cell_dimensions=list(qm_box_dims)+[90.0,90.0,90.0]

            #Write xyz-file with coordinates
            system_xyzfile="system_cp2k"
            write_xyzfile(qm_elems, current_coords, f"{system_xyzfile}", printlevel=1)

            #Write simple CP2K input
            write_CP2K_input(method=self.method, jobname='ash', center_coords=self.center_coords, qm_elems=qm_elems,
                             basis_dict=self.basis_dict, potential_dict=self.potential_dict,
                             basis_method=self.basis_method, wavelet_scf_type=self.wavelet_scf_type,
                             functional=self.functional, restartfile=None,
                             Grad=Grad, filename='cp2k', charge=charge, mult=mult,
                             coordfile=system_xyzfile, scf_convergence=self.scf_convergence, eps_default=self.eps_default,
                             scf_maxiter=self.scf_maxiter, outer_scf_maxiter=self.outer_scf_maxiter,
                             periodic_type=self.periodic_type,
                             xtb_periodic=self.xtb_periodic,
                             cell_dimensions=self.cell_dimensions,
                             cell_vectors=self.cell_vectors,
                             basis_file=self.basis_file, potential_file=self.potential_file,
                             psolver=self.psolver, printlevel=self.printlevel,
                             OT=self.OT, OT_minimizer=self.OT_minimizer, OT_preconditioner=self.OT_preconditioner, OT_linesearch=self.OT_linesearch,
                             outer_SCF=self.outer_SCF, outer_SCF_optimizer=self.outer_SCF_optimizer, OT_energy_gap=self.OT_energy_gap)

        #Delete old forces file if present
        try:
            os.remove(f'ash-{self.filename}-1_0.xyz')
        except:
            pass
        print_time_rel(module_init_time, modulename=f'CP2K run-prep5', moduleindex=2)
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
                shutil.copyfile(ash.settings_ash.ashpath+'/databases/basis-sets/cp2k/GTH_POTENTIALS', './POTENTIAL')
        print("Checking if BASIS file exists in current dir")
        if os.path.isfile("BASIS") is True:
            print(f"File exists in current directory: {os.getcwd()}")
        else:
            print("No file found. Trying parent dir")
            if os.path.isfile("../BASIS") is True:
                print("Found file in parent dir. Copying to current dir:", os.getcwd())
                shutil.copy(f"../BASIS", f"./BASIS")
            else:
                print("No file found in parent dir. Using basis set file from ASH. Copying to dir as BASIS")
                shutil.copyfile(ash.settings_ash.ashpath+'/databases/basis-sets/cp2k/BASIS_MOLOPT', './BASIS')
        print_time_rel(module_init_time, modulename=f'CP2K run-prep6', moduleindex=2)
        #Timing for Run-prep
        print_time_rel(module_init_time, modulename=f'CP2K run-prep', moduleindex=2)

        #Run CP2K
        if self.parallelization == 'Mixed':
            run_CP2K(self.cp2kdir,self.cp2k_bin_name,self.filename,numcores=self.numcores,paramethod=self.parallelization,
                     mixed_mpi_procs=self.mixed_mpi_procs, mixed_omp_threads=self.mixed_omp_threads)
        else:
            run_CP2K(self.cp2kdir,self.cp2k_bin_name,self.filename,numcores=self.numcores,paramethod=self.parallelization)


        #Grab energy
        self.energy=grab_energy_cp2k(self.filename+'.out',method=self.method)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        #Grab gradient if calculated
        if Grad is True:
            #Grab gradient
            self.gradient = grab_gradient_CP2K(f'ash-{self.filename}-1_0.xyz',len(current_coords))
            #Grab PCgradient from file
            if PC is True:
                self.pcgradient = grab_pcgradient_CP2K(f'ash-{self.filename}-1_0.xyz',len(MMcharges),len(current_coords))
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

def run_CP2K(cp2kdir,bin_name,filename,numcores=1, paramethod='MPI', mixed_omp_threads=None, mixed_mpi_procs=None):
    with open(filename+'.out', 'w') as ofile:
        if numcores >1:
            print("paramethod:", paramethod)
            if paramethod == 'MPI':
                #MPI-parallel setting OMP_NUM_THREDS to 1
                print(f"Launching MPI-parallel CP2K using {numcores} MPI processes")
                os.environ["OMP_NUM_THREADS"] = str(1)
                process = sp.run(["mpirun", "--bind-to", "none", f"-np", f"{str(numcores)}", f"{cp2kdir}/{bin_name}", f"{filename}.inp"], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            elif paramethod == 'OMP':
                #OpenMP-parallel, no-MPI
                print(f"Launching OpenMP parallel CP2K using {numcores} OpenMP threads")
                os.environ["OMP_NUM_THREADS"] = str(numcores)
                process = sp.run([cp2kdir + f'/{bin_name}', filename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            elif paramethod == 'Mixed':
                #Mixed MPI and OpenMP-parallel
                print(f"Launching Mixed MPI/OpenMP parallel CP2K using {numcores} cores in total")
                print(f"Will use {mixed_mpi_procs} MPI processes and {mixed_omp_threads} OpenMP threads per process for a total of {mixed_mpi_procs*mixed_omp_threads} cores")
                if mixed_mpi_procs*mixed_omp_threads != numcores:
                    print(f"Error: mixed_mpi_procs*mixed_omp_threads={mixed_mpi_procs}*{mixed_omp_threads} is not equal to numcores={numcores}")
                    ashexit()
                os.environ["OMP_NUM_THREADS"] = str(mixed_omp_threads)
                process = sp.run(["mpirun", "--bind-to", "none", f"-np", f"{str(mixed_mpi_procs)}", f"{cp2kdir}/{bin_name}", f"{filename}.inp"], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            else:
                print("Unknown parallelization method")
                ashexit()
        else:
            #Serial: No MPI, setting OMP_NUM_THREDS to 1
            print("Launching serial CP2K")
            os.environ["OMP_NUM_THREADS"] = str(1)
            process = sp.run([cp2kdir + f'/{bin_name}', filename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#Regular CP2K input
def write_CP2K_input(method='QUICKSTEP', jobname='ash-CP2K', center_coords=True, qm_elems=None,
                    basis_dict=None, potential_dict=None, functional=None, restartfile=None,
                    Grad=True, filename='cp2k', system_coord_file_format="XYZ",
                    coordfile=None,
                    charge=None, mult=None, basis_method='GAPW',
                    mgrid_commensurate=False, scf_maxiter=50, outer_scf_maxiter=10,
                    scf_guess='RESTART', scf_convergence=1e-6, eps_default=1e-10,
                    periodic_type="XYZ", cell_dimensions=None, cell_vectors=None,
                    qm_cell_dims=None, qm_periodic_type=None, xtb_periodic=False, basis_file='BASIS', potential_file='POTENTIAL',
                    psolver='wavelet', wavelet_scf_type=40,
                    ngrids=4, cutoff=250, rel_cutoff=60,
                    coupling='GAUSSIAN', GEEP_num_gauss=6, MM_radius_scaling=1, mm_radii=None,
                    qm_kind_dict=None, mm_kind_list=None,
                    mm_ewald_type='NONE', mm_ewald_alpha=0.35, mm_ewald_gmax="21 21 21", printlevel=2,
                    OT=False, OT_minimizer='DIIS', OT_preconditioner='FULL_ALL', OT_linesearch='3PNT',
                    outer_SCF=False, outer_SCF_optimizer='DIIS', OT_energy_gap=0.08):
    if method == 'QMMM':
        if mm_radii == None:
            print("No user MM radii provided. Will use default radii from internal dict (element_radii_for_cp2k).")
            mm_radii=element_radii_for_cp2k
            print("Radii will be scaled by factor:", MM_radius_scaling)
        else:
            print("User MM radii provided:", mm_radii)
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
        print("psolver is wavelet. Coordinates must be centered. Enforcing this by adding CENTER_COORDINATES to input")
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
        #SCF: Control GUESS etc
        inpfile.write(f'    &SCF\n')
        inpfile.write(f'      SCF_GUESS {scf_guess}\n')
        inpfile.write(f'      MAX_SCF {scf_maxiter}\n')
        inpfile.write(f'      EPS_SCF {scf_convergence}\n')
        if outer_SCF is True:
            inpfile.write(f'      &OUTER_SCF\n')
            inpfile.write(f'            OPTIMIZER {outer_SCF_optimizer}\n')
            inpfile.write(f'            MAX_SCF {outer_scf_maxiter}\n')
            inpfile.write(f'      &END OUTER_SCF\n')
        if OT is True:
            #Warning default OT settings here are supposedly expensive
            inpfile.write(f'      &OT \n')
            inpfile.write(f'            MINIMIZER {OT_minimizer}\n') #DIIS or CG
            inpfile.write(f'            PRECONDITIONER {OT_preconditioner}\n') # FULL_SINGLE_INVERSE or FULL_KINETIC
            inpfile.write(f'            LINESEARCH {OT_linesearch}\n') #NONE, 2PNT, 3PNT, GOLD
            inpfile.write(f'            ENERGY_GAP {OT_energy_gap}\n') #0.08 (default), 0.001, 0.002
            inpfile.write(f'      &END OT\n')
        inpfile.write(f'    &END SCF\n')
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
        if psolver == 'wavelet':
            inpfile.write(f'      &WAVELET {psolver}\n')
            inpfile.write(f'         SCF_TYPE {wavelet_scf_type}\n')
            inpfile.write(f'      &END WAVELET {psolver}\n')
        inpfile.write(f'    &END POISSON\n')
        #QS
        inpfile.write(f'    &QS\n')
        inpfile.write(f'      METHOD {basis_method}\n') #NOTE
        if basis_method == 'XTB':
            inpfile.write(f'      &XTB\n') #NOTE
            inpfile.write(f'          CHECK_ATOMIC_CHARGES F\n')
            inpfile.write(f'          DO_EWALD  {xtb_periodic}\n') #NOTE
            inpfile.write(f'          USE_HALOGEN_CORRECTION T\n') #NOTE
            inpfile.write(f'      &END XTB\n') #NOTE
        inpfile.write(f'      EPS_DEFAULT {eps_default}\n') #NOTE
        inpfile.write(f'    &END QS\n')

        #MGRID
        inpfile.write(f'    &MGRID\n')
        inpfile.write(f'      NGRIDS {ngrids}\n')
        inpfile.write(f'      CUTOFF {cutoff}\n')
        inpfile.write(f'      REL_CUTOFF {rel_cutoff}\n')
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
            inpfile.write(f'        DO_ELECTROSTATICS FALSE\n') #This turns off CP2K PC-PC interactions (not QM-PC)
            inpfile.write(f'        DO_NONBONDED FALSE\n') #This turns off CP2K LJ interactions
            inpfile.write(f'        @INCLUDE charges.inc\n') #including charges.inc file containing all system charges
            inpfile.write(f'      &END FORCEFIELD\n')
            #NOTE: POISSSON and EWALD section below is necessary even though we turn off MM-MM interactions
            #mm_ewald_type=None would turn off MM-MM periodic interactions
            inpfile.write(f'      &POISSON\n')
            inpfile.write(f'        &EWALD\n')
            inpfile.write(f'          EWALD_TYPE {mm_ewald_type}\n')
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
                #Unclear whether useful or not
                #inpfile.write(f'      NOCOMPATIBILITY\n')
            #QM-region cell
            inpfile.write(f'      &CELL \n')
            inpfile.write(f'        ABC {qm_cell_dims[0]} {qm_cell_dims[1]} {qm_cell_dims[2]} \n')
            inpfile.write(f'        PERIODIC {qm_periodic_type}  \n')
            inpfile.write(f'      &END CELL\n')
            #Write MM_KIND blocks
            #Necessary for GEEP. Radii used in embedding.
            for mm_el in mm_kind_list:
                if printlevel >2:
                    print("Assuming MM radius for element:", mm_el, "of:", mm_radii[mm_el], "Angstrom")
                    print("MM-radius after scaling", mm_radii[mm_el]*MM_radius_scaling, "Angstrom")
                #Note: CP2K converts this to Bohrs internally (printed as Bohrs in output)
                inpfile.write(f'      &MM_KIND {mm_el}\n')
                inpfile.write(f'          RADIUS {mm_radii[mm_el]*MM_radius_scaling}\n')
                inpfile.write(f'      &END MM_KIND\n')
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
        #This should be the total system cell size
        if cell_dimensions != None:
            inpfile.write(f'      ABC {cell_dimensions[0]} {cell_dimensions[1]} {cell_dimensions[2]}\n')
            inpfile.write(f'      ALPHA_BETA_GAMMA {cell_dimensions[3]} {cell_dimensions[4]} {cell_dimensions[5]}\n')
        elif cell_vectors != None:
            inpfile.write(f'      A {cell_vectors[0][0]} {cell_vectors[0][1]} {cell_vectors[0][2]}\n')
            inpfile.write(f'      B {cell_vectors[1][0]} {cell_vectors[1][1]} {cell_vectors[1][2]}\n')
            inpfile.write(f'      C {cell_vectors[2][0]} {cell_vectors[2][1]} {cell_vectors[2][2]}\n')
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
            inpfile.write(f'      COORD_FILE_FORMAT {system_coord_file_format}\n') #File-format: XYZ or PDB (pdb is bad)
            inpfile.write(f'      COORD_FILE_NAME {coordfile}.xyz\n') #Coord-file of whole system with charges (ideally XYZ)
            #inpfile.write(f'      CHARGE_EXTENDED TRUE\n') #Read charges from col 81
            inpfile.write(f'      CONNECTIVITY OFF\n') #No read or generate bonds
            inpfile.write(f'      &GENERATE\n')
            inpfile.write(f'           &ISOLATED_ATOMS\n') #Topology of isolated atoms
            inpfile.write(f'               LIST {first_atom}..{last_atom}\n')
            inpfile.write(f'           &END\n')
            inpfile.write(f'      &END GENERATE\n')
        else:
            inpfile.write(f'      COORD_FILE_FORMAT {system_coord_file_format}\n')
            inpfile.write(f'      COORD_FILE_NAME  ./{coordfile}.xyz\n')
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
def grab_pcgradient_CP2K(pcgradfile,numpc,numatoms):
    pc_gradient=np.zeros((numpc,3))
    pccount=0; atomcount=0
    atomgrab=False; pcgrad_grab=False
    with open(pcgradfile) as o:
        for i,line in enumerate(o):
            if '#' not in line:
                if 'SUM OF ATOMIC' in line:
                    return pc_gradient
                if pcgrad_grab is True:
                    pc_gradient[pccount,0] = -1*float(line.split()[-3])
                    pc_gradient[pccount,1] = -1*float(line.split()[-2])
                    pc_gradient[pccount,2] = -1*float(line.split()[-1])
                    pccount+=1
                if atomgrab is True:
                    atomcount+=1
            #Beginning pcgrad_grab
            if atomcount == numatoms:
                pcgrad_grab=True
            #Beginning atomcount
            if ' # Atom   Kind   Element' in line:
                atomgrab=True


def find_cp2k(cp2kdir, cp2k_bin_name):
    #List of binaries to search for in this order unless specified
    cp2k_binaries=["cp2k.psmp", "cp2k.popt", "cp2k.ssmp","cp2k.sopt"]
    if cp2kdir == None:
        #No cp2kdir
        print(BC.WARNING, f"No cp2kdir argument passed to CP2KTheory. Attempting to find cp2kdir variable inside settings_ash", BC.END)
        try:
            print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
            cp2kdir=ash.settings_ash.settings_dict["cp2kdir"]
            print("Found cp2kdir variable in settings_ash module:", cp2kdir)
        except:
            print(BC.WARNING,"Found no cp2kdir variable in settings_ash module either.",BC.END)
            print("Now searching for binary in path")
            if cp2k_bin_name == None:
                print(f"cp2k_bin_name variable not set. Will search for multiple binaries in PATH, using this order: {cp2k_binaries}")
            else:
                print("cp2k_bin_name variable set:", cp2k_bin_name)
                print(f"Searching for {cp2k_bin_name} in dir")
                cp2k_binaries=[cp2k_bin_name]

            for bin in cp2k_binaries:
                if shutil.which(bin) is not None:
                    print(BC.OKGREEN,"Found cp2k binary:", bin, BC.END)
                    cp2k_bin_name=bin
                    cp2kdir = os.path.dirname(shutil.which(bin))
                    return cp2kdir, cp2k_bin_name
    #If cp2kdir provided or found above
    if cp2kdir != None:
        #cp2kdir provided. Searching
        print("cp2kdir found:", cp2kdir)
        if cp2k_bin_name == None:
            print("cp2k_bin_name variable not set. Will search for multiple binaries in dir")
            print("Using this order: ", cp2k_binaries)
        else:
            print("cp2k_bin_name variable set:", cp2k_bin_name)
            print(f"Searching for {cp2k_bin_name} in dir")
            cp2k_binaries=[cp2k_bin_name]
        #Searching for binaries in dir
        for bin in cp2k_binaries:
            if os.path.isfile(cp2kdir+'/'+bin) is True:
                print(BC.OKGREEN,"Found cp2k binary:", bin, BC.END)
                cp2k_bin_name=bin
                return cp2kdir, cp2k_bin_name

    print("Error: Unsuccessful at finding cp2k binaries and/or directory")
    print("Note: Make sure the cp2k binaries are in your PATH and named correctly")
    ashexit()
    return
