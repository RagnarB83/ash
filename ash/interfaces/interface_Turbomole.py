import subprocess as sp
import os
import shutil
import time
import numpy as np
import pathlib
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader, writestringtofile
from ash.modules.module_coords import nucchargelist
from ash.modules.module_coords_PBC import cell_vectors_to_params, cell_params_to_vectors
import ash.settings_ash
from ash.functions.functions_parallel import check_OpenMPI

# Turbomole Theory object.

class TurbomoleTheory:
    def __init__(self, TURBODIR=None, turbomoledir=None, filename='XXX', printlevel=2, label="Turbomole", uff=False,
                numcores=1, parallelization='SMP', functional=None, dispersion=None, gridsize="m4", scfconv=7, symmetry="c1", rij=True,
                basis=None, jbasis=None, scfiterlimit=50, maxcor=500, ricore=500, controlfile=None,skip_control_gen=False,
                mp2=False, pointcharge_type=None, pc_gaussians=None,
                periodic=False, periodic_cell_vectors=None, PBC_dimension=3,
                periodic_cell_dimensions=None, kpoint_values=[1,1,1]):

        self.theorynamelabel="Turbomole"
        self.label=label
        self.theorytype="QM"
        self.analytic_hessian=True
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        #
        self.scfiterlimit=scfiterlimit
        self.functional=functional
        self.dispersion=dispersion
        self.symmetry=symmetry
        self.scfconv=scfconv
        self.gridsize=gridsize
        self.basis=basis
        self.jbasis=jbasis
        self.maxcor=maxcor
        self.ricore=ricore
        self.rij=rij
        self.mp2=mp2
        self.parallelization=parallelization
        self.mpi_is_setup=False
        self.smp_is_setup=False
        self.uff=uff

        # controlfile from user
        self.controlfile=controlfile
        # Skip controlfile  gen (assumes that the control file is present in dir)
        self.skip_control_gen=skip_control_gen

        #Special pointcharges options in Turbomole
        self.pointcharge_type=pointcharge_type
        self.pc_gaussians=pc_gaussians
        # if pointcharge_type is None then regular pointcharges
        # if pointcharge_type is 'gaussians' then we have smeared Gaussian charges. This will require pc_gaussians array to be defined (list of Gaussian alpha values for all charges)
        # if pointcharge_type is 'mxrank=Z' where Z is max multipole rank then we are doing point-multipole embedding. TODO: input not yet ready
        # if pointcharge_type is 'pe'. Polarizable embedding. TODO: not yet ready

        # UFF
        if self.uff:
            print("Initializing Turbomole UFF option")
            self.skip_control_gen=True
            self.turbo_scf_exe="uff"
            self.filename_scf="uff"
        # not-UFF, i.e. QM
        else:
            print("Initializing Turbomole QM")
            # QM controfile or Basis set check
            if controlfile is None:
                print("No controlfile provided. This requires basis keyword to be provided")
                if basis is None:
                    print(BC.WARNING, f"No basis set provided to {self.theorynamelabel}Theory. Exiting...", BC.END)
                    ashexit()
        self.basis=basis

        # PBC
        self.periodic=periodic
        self.PBC_dimension=PBC_dimension # PBC dimension 1:1D, 2:2D, 3:3D
        self.periodic_cell_vectors=None # initially
        self.kpoint_values=kpoint_values # k-point kpoint_values: [1,1,1] for gamma point in all directions
        self.cellderiv=False # Boolean for calculating cell derivate or not. default False
        if self.periodic:
            print("PBC enabled")
            self.cellderiv=True
            if periodic_cell_vectors is None and periodic_cell_dimensions is None:
                print("Error: for periodic calculations, you must specify either periodic_cell_vectors or  periodic_cell_dimensions")
                ashexit()
                # Convert to cell vectors
                self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)
            elif periodic_cell_vectors is not None:
                self.periodic_cell_vectors = periodic_cell_vectors
                self.periodic_cell_dimensions = cell_vectors_to_params(periodic_cell_vectors)
            elif periodic_cell_dimensions is not None:
                self.periodic_cell_dimensions = periodic_cell_dimensions
                self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)

            print("Cell vectors:", self.periodic_cell_vectors)
            print("Cell dimensions:", self.periodic_cell_dimensions)

        # User controlfile
        if self.controlfile is not None:
            if self.rij is True:
                self.turbo_scf_exe="ridft"
                self.turbo_exe_grad="rdgrad"
                self.filename_scf="ridft"
                self.filename_grad="rdgrad"
            else:
                self.turbo_scf_exe="dscf"
                self.turbo_exe_grad="grad"
                self.filename_scf="dscf"
                self.filename_grad="grad"
        # MP2
        elif self.mp2 is True:
            self.rij=False
            self.dft=False
            if functional is not None:
                print("Error: MP2 is True but a functional was provided. Exiting...")
                ashexit()

            self.turbo_scf_exe="dscf"
            self.filename_scf="dscf"

            self.turbo_mp2_exe="ricc2"
            self.filename_mp2="ricc2"

            self.turbo_exe_grad="grad"
        # DFT
        elif functional is not None:
            self.dft=True
            print("Functional provided. Choosing Turbomole executables to be ridft and rdgrad")
            print("Dispersion correction:", self.dispersion)
            if self.periodic:
                self.turbo_scf_exe="riper"
                self.turbo_exe_grad="riper"
                self.filename_scf="riper"
                self.filename_grad="riper"

            elif rij is True:
                self.turbo_scf_exe="ridft"
                self.turbo_exe_grad="rdgrad"
                self.filename_scf="ridft"
                self.filename_grad="rdgrad"
            else:
                self.turbo_scf_exe="dscf"
                self.turbo_exe_grad="grad"
                self.filename_scf="dscf"
                self.filename_grad="grad"
            print("jbasis:", jbasis)
            # Checking for ridft and jbas
            if self.turbo_scf_exe =="ridft" and jbasis is None:
                print("No jbasis provided for ridft. Exiting...")
                ashexit()
            else:
                self.jbasis=jbasis
        # UFF
        elif self.uff:
            print("UFF..")
        # else
        else:
            print("Error: No controlfile provided, not MP2, not DFT (no functional provided). Unclear what type of calculation this is. Exiting.")
            ashexit()

        # Checking OpenMPI
        if numcores != 1:
            print(f"Parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
            print("parallelization:", self.parallelization)
            if self.parallelization == 'MPI':
                print("Parallelization is MPI.")

        # Finding Turbomole
        if TURBODIR is not None:
            #self.turbomoledir = TURBODIR
            self.TURBODIR=TURBODIR
        elif turbomoledir is None:
            print(BC.WARNING, f"No turbomoledir argument passed to {self.theorynamelabel}Theory. Attempting to find turbomoledir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.turbomoledir=ash.settings_ash.settings_dict["turbomoledir"]
            except:
                print(BC.WARNING,"Found no turbomoledir variable in settings_ash module either.",BC.END)
                try:
                    bindir=os.path.dirname(shutil.which('ridft'))
                    self.turbomoledir = pathlib.Path(bindir).parent
                    print(BC.OKGREEN,"Found ridft (Turbomol executable) in PATH. Setting turbomoledir to parentdir of that dir:", self.turbomoledir, BC.END)
                except:
                    print(BC.FAIL,"Found no ridft executable in PATH. Exiting... ", BC.END)
                    ashexit()
            self.TURBODIR = os.path.dirname(self.turbomoledir)
        else:
            self.turbomoledir = turbomoledir
            self.TURBODIR = os.path.dirname(self.turbomoledir)

        # Setting environment variable TURBODIR (for basis set )
        os.environ['TURBODIR'] = self.TURBODIR

        # Printlevel
        self.printlevel=printlevel
        self.numcores=numcores

        # Get sysname once
        self.run_sysname()

        # Counter for how often TurbomoleTheory.run is called
        self.runcalls=0

    # Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores

    def cleanup(self):
        files=['coord','control','energy','gradient', 'auxbasis', 'basis', 'mos', 'ridft.out', 'rdgrad.out', 'ricc2.out', 'statistics']
        for f in files:
            if os.path.exists(f):
                os
    # Update cell using either periodic_cell_vectors or periodic_cell_dimensions
    def update_cell(self,periodic_cell_vectors=None, periodic_cell_dimensions=None):
        print("Updating cell vectors")
        if periodic_cell_vectors is not None:
            self.periodic_cell_vectors = periodic_cell_vectors
            self.periodic_cell_dimensions = cell_vectors_to_params(periodic_cell_vectors)
        elif periodic_cell_dimensions is not None:
            self.periodic_cell_dimensions=periodic_cell_dimensions
            self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)

    def get_cell_gradient(self):
        return self.cell_gradient

    def setup_mpi(self,numcores):
        print("Setting up MPI for Turbomole")
        print("TURBODIR:", self.TURBODIR)
        os.environ['PARA_ARCH'] = 'MPI'
        os.environ['PARNODES'] = str(numcores)
        print("PARA_ARCH has been set to: MPI")
        print("PARNODES has been set to ", numcores)
        #self.sysname=sp.run([f'{self.TURBODIR}/scripts/sysname'], stdout=sp.PIPE).stdout.decode('utf-8').replace("\n","")
        #print("sysname is now", self.sysname)
        os.environ['PATH']=f"{self.TURBODIR}/bin/{self.sysname}" + os.pathsep+os.environ['PATH']
        print("PATH:", os.environ['PATH'])
        self.run_sysname()
        self.mpi_is_setup=True

    def setup_smp(self,numcores):
        print("Setting up SMP for Turbomole")
        print("TURBODIR:", self.TURBODIR)
        os.environ['PARA_ARCH'] = 'SMP'
        os.environ['PARNODES'] = str(numcores)
        print("PARA_ARCH has been set to: SMP")
        print("PARNODES has been set to ", numcores)
        os.environ['PATH']=f"{self.TURBODIR}/bin/{self.sysname}" + os.pathsep+os.environ['PATH']
        print("PATH:", os.environ['PATH'])
        self.run_sysname()
        self.smp_is_setup=True

    def run_sysname(self):
        print("Running sysname script to find out system architecture")
        self.sysname=sp.run([f'{self.TURBODIR}/scripts/sysname'], stdout=sp.PIPE).stdout.decode('utf-8').replace("\n","")
        print("sysname is now", self.sysname)

    def run_turbo(self,filename, exe="ridft", numcores=1, parallelization=None):
        print(f"Running executable {exe} and writing to output {filename}.out")
        with open(filename+'.out', 'w') as ofile:
            if numcores >1:
                if parallelization == 'MPI':
                    print("Parallelization is MPI")
                    if self.mpi_is_setup is False:
                        self.setup_mpi(numcores)
                    print("Now running Turbomole using binaries in dir:", f"{self.TURBODIR}/bin/{self.sysname}")
                    process = sp.run([f"{self.TURBODIR}/bin/{self.sysname}" + f'/{exe}'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
                elif parallelization == 'SMP':
                    print("Parallelization is SMP")
                    if self.smp_is_setup is False:
                        self.setup_smp(numcores)
                    print("Now running Turbomole using binaries in dir:", f"{self.TURBODIR}/bin/{self.sysname}")
                    process = sp.run([f"{self.TURBODIR}/bin/{self.sysname}" + f'/{exe}'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
                else:
                    print("Error: parallelization method not recognized. Choose either 'MPI' or 'SMP'. Exiting...")
                    ashexit()
            else:
                print("Running in serial mode")
                process = sp.run([f"{self.TURBODIR}/bin/{self.sysname}" + f'/{exe}'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None, Hessian=False,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores is None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)


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

        # Delete old files (except control)
        files=['energy','gradient']
        for f in files:
            if os.path.exists(f):
                os.remove(f)

        # Create coord file
        create_coord_file(qm_elems,current_coords)

        # Skip control file generation
        if self.skip_control_gen is True:
            print("Skipping control file generation")
        # Create controlfile
        elif self.controlfile is None:
            # Delete old controlfile
            if os.path.exists('control'):
                os.remove('control')

            print("Creating controlfile")
            numelectrons = int(nucchargelist(qm_elems) - charge)
            create_control_file(runcalls=self.runcalls, functional=self.functional, dispersion=self.dispersion,gridsize=self.gridsize, scfconv=self.scfconv, dft=self.dft,
                            symmetry="c1", basis=self.basis, jbasis=self.jbasis, rij=self.rij, mp2=self.mp2, 
                            periodic=self.periodic, PBC_dimension=self.PBC_dimension,cell_vectors=self.periodic_cell_vectors,kpoint_values=self.kpoint_values,
                            cellderiv=self.cellderiv,
                            scfiterlimit=self.scfiterlimit, maxcor=self.maxcor, ricore=self.ricore, charge=charge, mult=mult,
                            pcharges=MMcharges, pccoords=current_MM_coords, pointcharge_type=self.pointcharge_type, pc_gaussians=self.pc_gaussians,
                            numelectrons=numelectrons)
        # User-controlled controlfile
        else:
            print("controlfile option chosen: ", self.controlfile)
            if os.path.isfile(self.controlfile) is False:
                print(f"Error: File {self.controlfile} does not exist!")
                ashexit()
            print("Copying file to: control")
            shutil.copyfile(self.controlfile, './' + 'control')

        #################
        # Run Turbomole
        #################

        print("Running Turbomole executable:", self.turbo_scf_exe)

        # Check for control file unless UFF
        if self.uff is False:
            if os.path.isfile("control") is False:
                print("No control file present. Exiting")
                ashexit()

        # Run energy only (SCF for DFT/WFT. or UFF)
        self.run_turbo(self.filename_scf, exe=self.turbo_scf_exe, parallelization=self.parallelization,
                  numcores=self.numcores)
        # Updating runcalls (this will also make sure that mos file is read in next run)
        self.runcalls+=1

        if self.uff:
            print("Grabbing UFF energy and gradient")
            if os.path.isfile("uffenergy") is False:
                print("Error: No uffenergy file created. Something went wrong with the Turbomole run. Check Turbomole output files for more info. Exiting...")
            self.energy = grab_energy_from_energyfile(file="uffenergy")
            print("UFF Energy:", self.energy)
            # Gradient
            self.gradient = grab_gradient(len(current_coords), file="uffgradient")
            print("self.gradient:", self.gradient)

        else:
            # Check if energy file has been created
            if os.path.isfile("energy") is False:
                print("Error: No energy file created. Something went wrong with the Turbomole run. Check Turbomole output files for more info. Exiting...")
                ashexit()

            self.energy = grab_energy_from_energyfile()
            print("SCF Energy:", self.energy)

        # MP2 energy only
        if self.mp2 is True:
            print("MP2 is True. Running:", self.turbo_mp2_exe)
            self.run_turbo(self.turbomoledir,self.filename_mp2, exe=self.turbo_mp2_exe, parallelization=self.parallelization, numcores=self.numcores)
            mp2_corr_energy = grab_energy_from_energyfile(column=4)
            print("MP2 correlation energy:", mp2_corr_energy)
            self.energy += mp2_corr_energy
            print("Total MP2 energy:", self.energy)

        # GRADIENT
        if Grad is True and self.uff is False:

            # Run gradient calc unless riper
            if self.periodic:
                print("Turbomole RIPER has already computed gradient")
                # Now grab gradient
                self.gradient = grab_gradient(len(current_coords))
                # Now grab cell gradient
                self.cell_gradient = grab_cellgrad(file="control")
            else:
                print("Running Turbomole-gradient executable")
                print("self.turbo_exe_grad:", self.turbo_exe_grad)
                print("self.filename_grad:", self.filename_grad)
                self.run_turbo(self.filename_grad, exe=self.turbo_exe_grad, parallelization=self.parallelization,
                    numcores=self.numcores)
                # Now grab gradient
                self.gradient = grab_gradient(len(current_coords))

            if PC:
                self.pcgradient = grab_pcgradient(len(MMcharges))

        # HESSIAN
        if Hessian is True:
            print("Running Turbomole-Hessian executable: aoforce")
            self.run_turbo("aoforce", exe="aoforce", parallelization=self.parallelization,
                  numcores=self.numcores)

            self.hessian=turbomole_grabhessian(len(current_coords),hessfile="hessian")
            print("Hessian:", self.hessian)

        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        # Grab gradient if calculated
        if Grad is True:
            if PC:
                return self.energy, self.gradient, self.pcgradient
            else:
                return self.energy, self.gradient
        else:
            return self.energy


################################
# Independent Turbomole functions
################################

# We always pass Angstrom but may choose to write coord file in Angstrom or Bohr, or fract (untested)
def create_coord_file(elems,coords, write_unit='BOHR', periodic_info=None, filename="coord"):
    if write_unit.upper() == "BOHR":
        conversion_factor=1.88972612546
        unit_string=""
    elif write_unit.upper() == "ANGSTROM":
        conversion_factor=1.0
        unit_string="angs"
    elif write_unit.upper() == "FRACT":
        conversion_factor=1.0
        unit_string="fract"
    else:
        print("Error")
        ashexit()
    with open(filename, 'w') as coordfile:
        coordfile.write(f"$coord {unit_string}\n")
        for i in range(len(elems)):
            if write_unit.upper() == "FRACT":
                coordfile.write(f"{coords[i][0]/periodic_info[0]} {coords[i][1]/periodic_info[1]} {coords[i][2]/periodic_info[2]} {elems[i]}\n")
            else:
                coordfile.write(f"{coords[i][0]*conversion_factor} {coords[i][1]*conversion_factor} {coords[i][2]*conversion_factor} {elems[i]}\n")
        # PBC
        if periodic_info is not None:
            coordfile.write(f"$periodic 3\n")
            coordfile.write(f"$cell {unit_string}\n")
            coordfile.write(f"{periodic_info[0]} {periodic_info[1]} {periodic_info[2]} {periodic_info[3]} {periodic_info[4]} {periodic_info[5]}\n")
        coordfile.write("$end\n")

def create_control_file(runcalls=None, functional="lh12ct-ssifpw92", dispersion=None, gridsize="m4", scfconv="7", symmetry="c1", rij=True, dft=True, mp2=False,
                        basis="def2-SVP", jbasis="def2-SVP", scfiterlimit=30, maxcor=500, ricore=500, charge=None, mult=None, 
                        periodic=False, PBC_dimension=3, cell_vectors=None, kpoint_values=[1,1,1], cellderiv=False,
                        pcharges=None, pccoords=None, pointcharge_type=None, pc_gaussians=None, numelectrons=None):
    if pccoords is not None:
        pccoords=pccoords*1.88972612546

    # MO-line. First assuming to be empty unless runcalls > 0(turbomole will do an EHT guess automatically)
    mosline=""

    # Guess-line
    ehtline=f"$eht charge={charge} unpaired={mult-1}"

    # Closed vs. open-shell. Han dles occupations and MO-files
    if mult == 1:
        print("Case closed-shell. Writing closed-shell occupation in control file.")
        shellsection=f"""$closed shells
a       1-{int(numelectrons/2)}                                    ( 2 )"""
        # If not first call then we read file mos (close-shell MO file).
        if runcalls > 0:
            print("Making sure mos-file from previous run is read in new control file")
            mosline = "$scfmo   file=mos"
        else:
            print("First call. No mos file will be read")
    else:
        print("Case open-shell. Guessing occupation to be written in $uhf section of control file.")
        num_a_electrons = int((numelectrons + mult - 1) / 2)
        num_b_electrons = int((numelectrons - mult + 1) / 2 )
        print("Assuming num_a_electrons:", num_a_electrons)
        print("Assuming num_b_electrons:", num_b_electrons)
        shellsection=f"""$uhf
$alpha shells
a       1-{num_a_electrons}                                    ( 1 )
$beta shells
a       1-{num_b_electrons}                                    ( 1 )
"""
        if runcalls > 0:
            print("Making sure alpha and beta mo-file from previous run are read in new control file")
            mosline = """$uhfmo_alpha    file=alpha
$uhfmo_beta    file=beta"""
        else:
            print("First call. No alpha/beta MO files will be read")
    # Now defining big control string.

    controlstring=f"""
$title
$symmetry {symmetry}
$coord    file=coord
$atoms
    basis ={basis}
    jbas  ={jbasis}
$basis    file=basis
{ehtline}
{mosline}
{shellsection}
$scfiterlimit       {scfiterlimit}
$scfdamp   start=0.300  step=0.050  min=0.100
$scfdump
$scfdiis
$maxcor    {maxcor} # MiB  per_core
$scforbitalshift  automatic=.1
$energy    file=energy
$grad    file=gradient
$scfconv   {scfconv}
"""
    if periodic is True:
        controlstring += f"""$periodic {PBC_dimension}
$lattice angs
  {cell_vectors[0,0]} {cell_vectors[0,1]} {cell_vectors[0,2]}
  {cell_vectors[1,0]} {cell_vectors[1,1]} {cell_vectors[1,2]}
  {cell_vectors[2,0]} {cell_vectors[2,1]} {cell_vectors[2,2]}
$kpoints
  nkpoints {kpoint_values[0]} {kpoint_values[1]} {kpoint_values[2]}
\n"""
        if cellderiv:
            controlstring += f"$optcell \n"

    if dft is True:
        controlstring += f"""$dft
    functional   {functional}
    gridsize   {gridsize}\n"""
    #Dispersion
    if dispersion is not None:
        if 'D3' in dispersion.upper():
            if '0' in dispersion.upper() or 'ZERO' in dispersion.upper():
                controlstring += "$disp3\n"
            else:
                controlstring += "$disp3 -bj\n"            
        elif 'D2' in dispersion.upper():
            controlstring += "$disp\n"
        elif 'D4' in dispersion.upper():
            controlstring += "$disp4\n"
        
    if mp2 is True:
        controlstring += f"""\n$denconv .1d-6
$ricc2
mp2"""


    if rij is True:
        controlstring += f"""\n$ricore      {maxcor}
$rij"""

    # Point charge handling
    if pcharges is not None:
        if pointcharge_type is None:
            pointcharge_type=""
        controlstring += f"""\n$point_charges file=pointcharges\n"""
        controlstring += f"""$point_charge_gradients file=pc_gradient\n"""
        controlstring += f"""$drvopt
   point charges on"""

        pcstring=""

        for i,(charge,pcoord) in enumerate(zip(pcharges,pccoords)):
            if pointcharge_type == "gaussians":
                alpha = pc_gaussians[i]
                pcstring += f"""{pcoord[0]} {pcoord[1]} {pcoord[2]} {charge} {alpha}\n"""
            elif 'mxrank' in pointcharge_type:
                print("mxrank pointcharge_type chosen. Not ready")
                exit()
                #if '2' in pointcharge_type:
                #    pcstring += f"""{pcoord[0]} {pcoord[1]} {pcoord[2]} {charge} \n"""            
                #elif '3' in pointcharge_type:
                #    pcstring += f"""{pcoord[0]} {pcoord[1]} {pcoord[2]} {charge} \n"""      
            else:
                pcstring += f"""{pcoord[0]} {pcoord[1]} {pcoord[2]} {charge} \n"""
        with open('pointcharges', 'w') as pcfile:
            pcfile.write(f"$point_charges {pointcharge_type} \n")
            pcfile.write(pcstring)
            pcfile.write("$end\n")
    controlstring+="\n$end"

    writestringtofile(controlstring, 'control')


def grab_energy_from_energyfile(file="energy", column=1):
    energy = None
    with open(file, 'r') as energyfile:
        for line in energyfile:
            if '$end' in line:
                return energy
            if "$energy" not in line:
                energy = float(line.split()[column])
    return energy

# Fails if multiple SCF cycles are present in file (happens for riper)
def grab_gradient_old(numatoms,file="gradient"):
    gradient = np.zeros((numatoms,3))
    with open(file, 'r') as gradfile:
        gradlines = gradfile.readlines()
    counter=0
    for i,line in enumerate(gradlines):
        print("line:", line)
        if '$end' in line:
            break
        if i > numatoms+1:
            gradient[counter] = [float(j.replace('D','E')) for j in line.split()]
            counter+=1
    return gradient

def grab_gradient(numatoms,file="gradient"):
    gradient = np.zeros((numatoms,3))
    with open(file, 'r') as gradfile:
        gradlines = gradfile.readlines()
    #Reverse lines
    gradlines.reverse()
    # Setting counter to be numatoms
    counter=numatoms
    #Read lines in reverse
    for i,line in enumerate(gradlines):
        # Break when done
        if counter == 0:
            break
        # Grab gradient
        if '$end' not in line:
            gradient[counter-1] = [float(j.replace('D','E')) for j in line.split()]
            counter-=1
    return gradient

def grab_pcgradient(numpc,filename="pc_gradient"):
    pcgradient = np.zeros((numpc,3))
    with open(filename, 'r') as gradfile:
        gradlines = gradfile.readlines()
    counter=0
    for i,line in enumerate(gradlines):
        if '$end' in line:
            break
        if i > 0:
            pcgradient[counter] = [float(j.replace('D','E')) for j in line.split()]
            counter+=1

    return pcgradient

def turbomole_grabhessian(numatoms,hessfile="hessian"):
    hessdim=3*numatoms
    hessian=np.zeros((hessdim,hessdim))
    i=0
    with open(hessfile) as f:
        for line in f:
            if '$' not in line:
                vals = line.split()[2:]
                for n,v in enumerate(vals):
                    hessian[i,n] = float(v)
                i = int(line.split()[0])-1
    return hessian

# Usually in controlfile
def grab_cellgrad(file="control"):
    cellgrad=np.zeros((3,3))
    grab=False
    lines=[]
    with open(file) as f:
        for line in f:
            if grab is True and '$end' in line:
                grab=False
            if grab:
                lines.append(line)
            if '$gradlatt' in line:
                grab=True
    cellgrad[0,0] = float(lines[-3].replace('D','E').split()[0])
    cellgrad[0,1] = float(lines[-3].replace('D','E').split()[1])
    cellgrad[0,2] = float(lines[-3].replace('D','E').split()[2])
    cellgrad[1,0] = float(lines[-2].replace('D','E').split()[0])
    cellgrad[1,1] = float(lines[-2].replace('D','E').split()[1])
    cellgrad[1,2] = float(lines[-2].replace('D','E').split()[2])
    cellgrad[2,0] = float(lines[-1].replace('D','E').split()[0])
    cellgrad[2,1] = float(lines[-1].replace('D','E').split()[1])
    cellgrad[2,2] = float(lines[-1].replace('D','E').split()[2])
    return cellgrad