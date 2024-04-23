import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
from ash.modules.module_coords import elematomnumbers
import ash.settings_ash

# Basic interface to MNDO
# Only support for energy and gradient + QM/MM energy+ gradient for now
# TODO: check parallelization

# MNDOTheory object.


class MNDOTheory:
    def __init__(self, mndodir=None, filename='mndo', method=None, printlevel=2, label="MNDO",
                 numcores=1, restart_option=True, diis=False, guess_option=0,
                 scfconv=6):

        self.theorynamelabel="MNDO"
        self.label=label
        self.theorytype="QM"
        self.analytic_hessian=False
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        if method is None:
            print(f"{self.theorynamelabel}Theory requires a method keyword")
            ashexit()

        #Finding MNDO
        if mndodir == None:
            print(BC.WARNING, f"No mndodir argument passed to {self.theorynamelabel}Theory. Attempting to find mndodir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.mndodir=ash.settings_ash.settings_dict["mndodir"]
            except:
                print(BC.WARNING,"Found no mndodir variable in settings_ash module either.",BC.END)
                try:
                    self.mndodir = os.path.dirname(shutil.which('mndo2020'))
                    print(BC.OKGREEN,"Found mndo2020 in PATH. Setting mndodir to:", self.mndodir, BC.END)
                except:
                    print(BC.FAIL,"Found no mndo2020 executable in PATH. Exiting...", BC.END)
                    print("See https://mndo.kofo.mpg.de about MNDO licenses")
                    ashexit()
        else:
            self.mndodir = mndodir

        self.printlevel=printlevel
        self.filename=filename
        self.method=method
        self.numcores=numcores

        # Parallelization: no threading used it seems
        # Set OMP_NUM_THREADS to numcores
        # os.environ['OMP_NUM_THREADS'] = str(self.numcores)
        #Some MPI support in MNDO, but we have not tested

        # Whether to automatically save DM and read DM from file or not
        # Should give fewer SCF iterations but more IO
        self.restart_option=restart_option

        # DIIS: Default off (activated by MNDO itself if SCF-problems).
        # For convergence issues, set to True (DIIS activated from beginning)
        self.diis=diis
        self.guess_option=guess_option
        self.scfconv=scfconv #10**(-scfconv) eV. MNDO default is 6 i.e. 1E-6 eV => 3.67E-08 Eh

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
        if numcores is None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)
        print(f"Creating inputfile: {self.filename}.inp")
        print(f"{self.theorynamelabel} input:")
        print("MNDO method:", self.method)

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

        print_time_rel(module_init_time, modulename=f'mndo prep-before-write', moduleindex=3)
        # Write inputfile
        write_mndo_input(self.method,self.filename,qm_elems,current_coords,charge,mult, PC=PC, Grad=Grad, scfconv=self.scfconv,
                         MMcharges=MMcharges, MMcoords=current_MM_coords, restart=self.restart_option, diis=self.diis, guess_option=self.guess_option)

        print_time_rel(module_init_time, modulename=f'mndo prep-run', moduleindex=3)
        # Run MNDO
        run_MNDO(self.mndodir,self.filename)
        print_time_rel(module_init_time, modulename=f'mndo run-done', moduleindex=3)

        # Grab energy
        if PC is True:
            self.energy, self.gradient, self.pcgradient = grab_energy_gradient_mndo(f"{self.filename}.out", len(current_coords), Grad=Grad, PC=True, numpc=len(MMcharges))
        else:
            self.energy, self.gradient, self.pcgradient = grab_energy_gradient_mndo(f"{self.filename}.out", len(current_coords), Grad=Grad, PC=False)
        if self.energy is None:
            print("MNDO failed to calculate an energy. Exiting. Check MNDO outputfile for errors.")
            ashexit()
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Grab gradient if calculated
        if Grad is True:
            # Grab PCgradient from separate file
            if PC is True:
                print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
                return self.energy, self.gradient, self.pcgradient
            else:
                print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
                return self.energy, self.gradient
        # Returning energy without gradient
        else:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy

def write_mndo_input(method,filename,elems,coords,charge,mult, PC=False, MMcharges=None, MMcoords=None, Grad=False, restart=True, diis=False, guess_option=0, scfconv=6):
    #init_time=time.time()
    mndo_methods={'ODM3':-23,'ODM2':-22,'MNDO/d':-10,'OM3':-8,
                    'PM3':-7,'OM2':-6,'OM1':-5,'AM1':-2,
                    'MNDOC': -1, 'MNDO':0, 'MINDO/3':1, 'CNDO/2':2,
                    'SCC-DFTB':5, 'SCC-DFTB_jorgensen':6}
    if Grad is True:
        jobtype=-2
    else:
        jobtype=-1

    if mult > 1:
        openshellkwstring="iuhf=-1"
    else:
        openshellkwstring=""

    # Open file
    f = open(f"{filename}.inp", "w")

    # List to keep inputlines
    inputlines=[]

    # iop is method number
    # igeom=1 cartesian
    # jop=-2 for gradient calc
    # iform=2 for free-format
    # ksym=0 no symmetry, nprint for energy output, mprint for gradient output

    # Guess
    ktrial_line=f"ktrial={guess_option}"

    # Note: ipub=1 saves fort.11 file with density-matrix
    # Note: ktrial=11 loads fort.11 as SCF-guess.
    # Should be faster, although more IO would be performed
    restartline=""
    if restart is True:
        restartline="ipubo=1"
        ktrial_line="ktrial=11"

    # DIIS settings
    diisline="idiis=0" #idiis=0 means DIIS off by default but will be turned on if SCF-problems
    if diis is True:
        #Activate DIIS from beginning
        diisline="idiis=1"

    # f.write(f"iop={mndo_methods[method]}  jop={jobtype} {openshellkwstring}  iform=1 igeom=1 +\n")
    # f.write(f"kitscf=300 kharge={charge} {restartline} {ktrial_line} {diisline} +\n")
    # f.write(f"iscf={scfconv} imult={mult} nprint=-4  mprint=0 ksym=0 +\n")
    inputlines.append(f"iop={mndo_methods[method]}  jop={jobtype} {openshellkwstring}  iform=1 igeom=1 +\n")
    inputlines.append(f"kitscf=300 kharge={charge} {restartline} {ktrial_line} {diisline} +\n")
    inputlines.append(f"iscf={scfconv} imult={mult} nprint=-4  mprint=5 ksym=0 +\n")
    #print_time_rel(init_time, modulename=f'time1', moduleindex=3)
    # PC-part
    if PC is True:
        # mminp=2 pointcharges
        # numatom=X number of PCs
        # mmcoup=2 elstat embedding. mmcoup=3 is MMpol
        # mmlink=2 linkatoms elstat option. mmlink=2 (sees all atoms, fine since we do charge-shifting anyway)
        # mmfile=1 read PCs from file nb2o
        # ipsana=1 analytic derivative
        #f.write(f"mminp=2 numatm={len(MMcharges)} mmcoup=2 mmlink=2 nlink=0 ipsana=1\n")
        inputlines.append(f"mminp=2 numatm={len(MMcharges)} mmcoup=2 mmlink=2 nlink=0 ipsana=1\n")
    else:
        #f.write("\n")
        inputlines.append("\n")
    #f.write("MNDO label\n")
    #f.write("\n")
    inputlines.append("MNDO label\n\n")

    for el,c in zip(elems,coords):
        atomnum=elematomnumbers[el.lower()]
        #f.write(f"{atomnum} {c[0]} 0 {c[1]} 0 {c[2]} 0\n")
        inputlines.append(f"{atomnum} {c[0]} 0 {c[1]} 0 {c[2]} 0\n")
    #f.write("0 0.0 0 0.0 0 0.0 0\n")
    inputlines.append("0 0.0 0 0.0 0 0.0 0\n")

    #print_time_rel(init_time, modulename=f'time3', moduleindex=3)
    for line in inputlines:
        f.write(line)


    f.close()
    #Now appending PCs to file if using
    if PC is True:
        pcdata = np.column_stack((MMcoords, MMcharges))
        #Fast appending to file
        ash.functions.functions_general.fast_nparray_write(pcdata, float_format="%-12.7f %-12.7f %-12.7f %-8.4f", filename=f"{filename}.inp", writemode="a")
        #np.savetxt(f, pcdata, fmt='%-12.7f%-12.7f%-12.7f%-8.4f')
        #Old:
        #for q,pc_c in zip(MMcharges,MMcoords):
        #    f.write(f"{pc_c[0]} {pc_c[1]} {pc_c[2]} {q}\n")
        #print_time_rel(init_time, modulename=f'time4b', moduleindex=3)
        #np.savetxt(f, pcdata, fmt='%f %f %f %f')
        #f.writelines([f"{i[0]} {i[1]} {i[2]} {i[3]}\n" for i in pcdata])
    #print_time_rel(init_time, modulename=f'time4c', moduleindex=3)



def run_MNDO(mndodir,filename):
    print("Running MNDO")
    infile=open(f"{filename}.inp")
    ofile=open(filename+'.out', 'w')
    sp.run([f"{mndodir}/mndo2020"], stdin=infile, stdout=ofile)
    infile.close()
    ofile.close()


def grab_energy_gradient_mndo(file, numatoms, Grad=False, PC=False, numpc=None):
    energy=None
    gradient=None
    pcgradient=None
    if Grad is True:
        gradient=np.zeros((numatoms,3))
        if PC is True:
            pcgradient=np.zeros((numpc,3))
    atomcount=0
    pccount=0
    grab=False
    pcgrab=False
    grad_convfactor=1185.821047110700
    with open(file) as f:
        for line in f:
            if 'SCF TOTAL ENERGY' in line:
                energy = float(line.split()[-2])/27.211386245988
            if grab:
                if len(line.split()) > 2:
                    if 'X' not in line:
                        if 'COORDINATES' not in line:
                            gradient[atomcount,0]=float(line.split()[-3])/grad_convfactor
                            gradient[atomcount,1]=float(line.split()[-2])/grad_convfactor
                            gradient[atomcount,2]=float(line.split()[-1])/grad_convfactor
                            atomcount+=1
                            if atomcount == numatoms:
                                grab=False
            if pcgrab:
                if len(line.split()) > 2:
                    if 'X' not in line and 'COORDINATES' not in line:
                        pcgradient[pccount,0]=float(line.split()[-3])/grad_convfactor
                        pcgradient[pccount,1]=float(line.split()[-2])/grad_convfactor
                        pcgradient[pccount,2]=float(line.split()[-1])/grad_convfactor
                        pccount+=1
                        if pccount == numpc:
                            break
            if Grad:
                if 'TIME FOR GRADIENT EVALUATION' in line:
                    grab=True
                if '     CARTESIAN GRADIENT NORM' in line:
                    grab=False
            if PC is True:
                if '     GRADIENT CONTRIBUTIONS TO EXTERNAL POINT CHARGES' in line:
                    pcgrab=True
                if 'CARTESIAN QM+MM GRADIENT NORM' in line:
                    pcgrab=False

    if Grad is True and PC is True:
        if np.any(pcgradient) is False:
            print("Error: PCgradient array from MNDO output is zero. Something went wrong.")
            ashexit()

    return energy,gradient,pcgradient
