import subprocess as sp
import os
import shutil
import time
import numpy as np
import pathlib
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader, writestringtofile
import ash.settings_ash
from ash.functions.functions_parallel import check_OpenMPI

# Turbomole Theory object.

class TurbomoleTheory:
    def __init__(self, TURBODIR=None, turbomoledir=None, filename='XXX', printlevel=2, label="Turbomole",
                numcores=1, parallelization='SMP', functional=None, gridsize="m4", scfconf=7, symmetry="c1", rij=True,
                basis=None, jbasis=None, scfiterlimit=50, maxcor=500, ricore=500, controlfile=None,skip_control_gen=False,
                mp2=False, pointcharge_type=None, pc_gaussians=None):

        self.theorynamelabel="Turbomole"
        self.label=label
        self.theorytype="QM"
        self.analytic_hessian=True
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        #
        self.scfiterlimit=scfiterlimit
        self.functional=functional
        self.symmetry=symmetry
        self.scfconf=scfconf
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

        # Basis set check
        if controlfile is None:
            print("No controlfile provided. This requires basis to be provided")
            if basis is None:
                print(BC.WARNING, f"No basis set provided to {self.theorynamelabel}Theory. Exiting...", BC.END)
                ashexit()
        self.basis=basis

        # User controlfile
        if self.controlfile is not None:
            #Assuming
            self.turbo_scf_exe="ridft"
            self.turbo_exe_grad="rdgrad"
            self.filename_scf="ridft"
            self.filename_grad="rdgrad"
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
            if rij is True:
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
        print("self.turbo_scf_exe:", self.turbo_scf_exe)
        # Checking OpenMPI
        if numcores != 1:
            print(f"Parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
            print("parallelization:", self.parallelization)
            if self.parallelization == 'MPI':
                print("Parallelization is MPI. Checking availability of OpenMPI")
                #exit()
                #check_OpenMPI()

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
        print("TURBODIR:", self.TURBODIR)

        # Printlevel
        self.printlevel=printlevel
        self.numcores=numcores

    # Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        files=['coord','control','energy','gradient', 'auxbasis', 'basis', 'mos', 'ridft.out', 'rdgrad.out', 'ricc2.out', 'statistics']
        for f in files:
            if os.path.exists(f):
                os
    def setup_mpi(self,numcores):
        print("Setting up MPI for Turbomole")
        print("TURBODIR:", self.TURBODIR)
        os.environ['PARA_ARCH'] = 'MPI'
        os.environ['PARNODES'] = str(numcores)
        print("PARA_ARCH has been set to: MPI")
        print("PARNODES has been set to ", numcores)
        self.sysname=sp.run(['sysname'], stdout=sp.PIPE).stdout.decode('utf-8').replace("\n","")
        print("sysname is now", self.sysname)
        os.environ['PATH']=f"{self.TURBODIR}/bin/{self.sysname}" + os.pathsep+os.environ['PATH']
        print("PATH:", os.environ['PATH'])
        self.mpi_is_setup=True

    def setup_smp(self,numcores):
        print("Setting up SMP for Turbomole")
        print("TURBODIR:", self.TURBODIR)
        os.environ['PARA_ARCH'] = 'SMP'
        os.environ['PARNODES'] = str(numcores)
        print("PARA_ARCH has been set to: SMP")
        print("PARNODES has been set to ", numcores)
        self.sysname=sp.run(['sysname'], stdout=sp.PIPE).stdout.decode('utf-8').replace("\n","")
        print("sysname is now", self.sysname)
        os.environ['PATH']=f"{self.TURBODIR}/bin/{self.sysname}" + os.pathsep+os.environ['PATH']
        print("PATH:", os.environ['PATH'])
        self.smp_is_setup=True

    def run_turbo(self,turbomoledir,filename, exe="ridft", numcores=1, parallelization=None):
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
                #process = sp.run([turbomoledir + f'/{exe}'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
                self.sysname=sp.run(['sysname'], stdout=sp.PIPE).stdout.decode('utf-8').replace("\n","")
                process = sp.run([f"{self.TURBODIR}/bin/{self.sysname}" + f'/{exe}'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None, Hessian=False,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores is None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        #Checking if charge and mult has been provided
        if charge is None or mult is None:
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
            create_control_file(functional=self.functional, gridsize=self.gridsize, scfconf=self.scfconf, dft=self.dft,
                            symmetry="c1", basis=self.basis, jbasis=self.jbasis, rij=self.rij, mp2=self.mp2,
                            scfiterlimit=self.scfiterlimit, maxcor=self.maxcor, ricore=self.ricore, charge=charge, mult=mult,
                            pcharges=MMcharges, pccoords=current_MM_coords, pointcharge_type=self.pointcharge_type, pc_gaussians=self.pc_gaussians)
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
        if os.path.isfile("control") is False:
            print("No control file present. Exiting")
            ashexit()
        # SCF-energy only
        self.run_turbo(self.turbomoledir,self.filename_scf, exe=self.turbo_scf_exe, parallelization=self.parallelization,
                  numcores=self.numcores)
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
        if Grad is True:
            print("Running Turbomole-gradient executable")
            print("self.turbo_exe_grad:", self.turbo_exe_grad)
            print("self.filename_grad:", self.filename_grad)
            self.run_turbo(self.turbomoledir,self.filename_grad, exe=self.turbo_exe_grad, parallelization=self.parallelization,
                  numcores=self.numcores)
            self.gradient = grab_gradient(len(current_coords))

            if PC:
                self.pcgradient = grab_pcgradient(len(MMcharges))

        # HESSIAN
        if Hessian is True:
            print("Running Turbomole-Hessian executable: aoforce")
            self.run_turbo(self.turbomoledir,"aoforce", exe="aoforce", parallelization=self.parallelization,
                  numcores=self.numcores)

            self.hessian=None
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

def create_control_file(functional="lh12ct-ssifpw92", gridsize="m4", scfconf="7", symmetry="c1", rij=True, dft=True, mp2=False,
                        basis="def2-SVP", jbasis="def2-SVP", scfiterlimit=30, maxcor=500, ricore=500, charge=None, mult=None,
                        pcharges=None, pccoords=None, pointcharge_type=None, pc_gaussians=None):
    if pccoords is not None:
        pccoords=pccoords*1.88972612546

    ehtline=f"$eht charge={charge} unpaired={mult-1}"


#Skipping orb section for now
#$closed shells
# a       1-7                                    ( 2 )

    controlstring=f"""
$title
$symmetry {symmetry}
$coord    file=coord
$atoms
    basis ={basis}
    jbas  ={jbasis}
$basis    file=basis
{ehtline}
$scfmo   file=mos
$scfiterlimit       {scfiterlimit}
$scfdamp   start=0.300  step=0.050  min=0.100
$scfdump
$scfdiis
$maxcor    {maxcor} # MiB  per_core
$scforbitalshift  automatic=.1
$energy    file=energy
$grad    file=gradient
$scfconv   {scfconf}
"""

    if dft is True:
        controlstring += f"""$dft
    functional   {functional}
    gridsize   {gridsize}"""

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


def grab_energy_from_energyfile(column=1):
    energy = None
    with open('energy', 'r') as energyfile:
        for line in energyfile:
            if '$end' in line:
                return energy
            if "$energy" not in line:
                energy = float(line.split()[column])
    return energy

def grab_gradient(numatoms):
    gradient = np.zeros((numatoms,3))
    with open('gradient', 'r') as gradfile:
        gradlines = gradfile.readlines()
    counter=0
    for i,line in enumerate(gradlines):
        if '$end' in line:
            break
        if i > numatoms+1:
            gradient[counter] = [float(j.replace('D','E')) for j in line.split()]
            counter+=1

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