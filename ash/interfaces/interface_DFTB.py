import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader,check_program_location
from ash.modules.module_coords import elematomnumbers, write_xyzfile
import ash.settings_ash

# Basic interface to DFTB+

class DFTBTheory():
    def __init__(self, dftbdir=None, hamiltonian="XTB", xtb_method="GFN2-xTB", printlevel=2, label="DFTB",
                 numcores=1, slaterkoster_dict=None, maxmom_dict=None, Gauss_blur_width=0.0,
                 ThirdOrderFull=False, ThirdOrder=False):

        self.theorynamelabel="DFTB"
        self.label=label
        self.theorytype="QM"
        self.analytic_hessian=False
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")


        # Finding DFTB
        self.dftbdir = check_program_location(dftbdir,"dftbdir", "dftb+")

        self.printlevel=printlevel
        self.hamiltonian=hamiltonian
        self.xtb_method=xtb_method
        self.numcores=numcores
        self.slaterkoster_dict=slaterkoster_dict
        self.Gauss_blur_width=Gauss_blur_width

        # Third-order
        self.ThirdOrderFull=ThirdOrderFull
        self.ThirdOrder=ThirdOrder


        if hamiltonian != "XTB":
            print("DFTB Hamiltonian is not XTB. SlaterKoster files are requires")
            if self.slaterkoster_dict is None:
                print("Error: No dictionary of Slater-Koster files provided (slaterkoster_dict keyword). This is necessary")
                ashexit()

        if maxmom_dict is None:
            print("Warning: No maxmom_dict (dictionary of Maximum Angular Momenta for each element) provided")
            print("ASH will guess the maxmoms for each element before running")
            self.maxmom_dict={'H':'s', 'B':'p', 'C':'p', 'N':'p', 'O':'p', 'F':'p', 
                                       'Al':'p', 'Si':'p', 'P':'p', 'S':'p', 'Cl':'p'}
            print("self.maxmom_dict:", self.maxmom_dict)
        else:
            self.maxmom_dict=maxmom_dict

    # Set numcores method
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
        print(f"{self.theorynamelabel} input:")
        if self.hamiltonian.upper() == "XTB":
            print("XTB method:", self.xtb_method)

        # xTB and PC does work within DFTB
        if PC is True:
            print("Pointcharge-included DFTB calculation")
            print(self.hamiltonian)
            if self.hamiltonian.upper() == "XTB":
                print("Error: Pointcharge-calculations not possible with XTB Hamiltonian")
                ashexit()

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

        print_time_rel(module_init_time, modulename=f'DFTB prep-before-write', moduleindex=3)
        # Write XYZ-file
        xyzfilename="dftb_"
        write_xyzfile(qm_elems, current_coords, xyzfilename)
        # Write inputfile
        write_DFTB_input(self.hamiltonian,self.xtb_method,xyzfilename+'.xyz',qm_elems,current_coords,charge,mult, PC=PC, Grad=Grad,
                         slaterkoster_dict=self.slaterkoster_dict, maxmom_dict=self.maxmom_dict, MMcharges=MMcharges, MMcoords=current_MM_coords,
                         Gauss_blur_width=self.Gauss_blur_width, ThirdOrderFull=self.ThirdOrderFull, ThirdOrder=self.ThirdOrder)

        print_time_rel(module_init_time, modulename=f'DFTB prep-run', moduleindex=3)
        # Run DFTB
        run_DFTB(self.dftbdir, inputfile="dftb_in.hsd", outputfile="dftb+.out")
        print_time_rel(module_init_time, modulename=f'DFTB run-done', moduleindex=3)

        # Grab energy
        if PC is True:
            self.energy, self.gradient, self.pcgradient = grab_energy_gradient_DFTB("detailed.out", len(current_coords), Grad=Grad, PC=True, numpc=len(MMcharges))
        else:
            self.energy, self.gradient, self.pcgradient = grab_energy_gradient_DFTB("detailed.out", len(current_coords), Grad=Grad, PC=False)
        if self.energy is None:
            print("DFTB failed to calculate an energy. Exiting. Check DFTB outputfile for errors.")
            ashexit()
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        print("gradient:", self.gradient)
        print("pcgradient:", self.pcgradient)

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
#
def write_DFTB_input(hamiltonian,xtbmethod,xyzfilename, elems,coords,charge,mult, PC=False, MMcharges=None, MMcoords=None, Grad=False, SCC=True,
                     slaterkoster_dict=None, maxmom_dict=None, Gauss_blur_width=0.0, ThirdOrderFull=False, ThirdOrder=False):

    # Open file
    f = open("dftb_in.hsd", "w")

    # List to keep inputlines
    inputlines=[]

    # Geometry
    geo1="Geometry = xyzFormat {\n"
    geo2=f"<<< '{xyzfilename}' \n}}\n"

    inputlines.append(geo1)
    inputlines.append(geo2)

    # Method
    method1=f"Hamiltonian = {hamiltonian} {{"+"\n"
    inputlines.append(method1)
    if 'XTB' in hamiltonian.upper():
        method2=f"Method = '{xtbmethod}'"+'\n}\n'
        inputlines.append(method2)
    else:
    # PC
        if PC:
            numPC=len(MMcharges)
            # Write PCs to disk
            create_pcfile("PCcharges.dat",MMcoords,MMcharges)
            inputlines.append(f"ElectricField = " + "{ \n")
            inputlines.append(f"  PointCharges = " + "{ \n")
            inputlines.append(f"    GaussianBlurWidth [Angstrom] = {Gauss_blur_width}\n")
            inputlines.append(f"    CoordsAndCharges [Angstrom] = DirectRead " + "{ \n")
            inputlines.append(f"    Records = {numPC}\n")
            inputlines.append(f"    File = 'PCcharges.dat'\n")
            inputlines.append('    }\n')
            inputlines.append('  }\n')
            inputlines.append('}\n')
        # SCC
        if SCC is True:
            SCCkeyword="Yes"
        else:
            SCCkeyword="No"
        
        if ThirdOrderFull is True:
            ThirdOrderFullkeyword="Yes"
        else:
            ThirdOrderFullkeyword="No"
        if ThirdOrder is True:
            ThirdOrderkeyword="Yes"
        else:
            ThirdOrderkeyword="No"
        inputlines.append(f"  Scc = {SCCkeyword}"+'\n')
        inputlines.append(f"  ThirdOrderFull = {ThirdOrderFullkeyword}"+'\n')
        inputlines.append(f"  ThirdOrder = {ThirdOrderkeyword}"+'\n')
        # SlaterKosterFiles
        inputlines.append("  SlaterKosterFiles {"+'\n')
        for atompair,fpath in slaterkoster_dict.items():
            inputlines.append(f"    {atompair} = '{fpath}'"+'\n')
        inputlines.append('  }\n')
        # MaxAngularMoment
        inputlines.append('  MaxAngularMomentum {\n')
        for el in list(set(elems)):
            inputlines.append(f'    {el} = "{maxmom_dict[el]}"\n')
        inputlines.append('  }\n')

        inputlines.append('}\n')

    #Options
    optionline="Options { WriteDetailedOut = Yes }\n"
    parserline="ParserOptions { ParserVersion = 10 }\n"
    inputlines.append(optionline)
    inputlines.append(parserline)
    # Forces
    if Grad:
        forcesline="Analysis { PrintForces = Yes }\n"
        inputlines.append(forcesline)



    for line in inputlines:
        f.write(line)

    f.close()


def run_DFTB(DFTBdir, inputfile="dftb_in.hsd", outputfile="dftb+.out"):
    print("Running DFTB")
    infile=open(inputfile)
    ofile=open(outputfile, 'w')
    sp.run([f"{DFTBdir}/dftb+"], stdin=infile, stdout=ofile)
    ofile.close()


def grab_energy_gradient_DFTB(file, numatoms, Grad=False, PC=False, numpc=None):
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
    with open(file) as f:
        for line in f:
            if 'Total energy:' in line:
                energy = float(line.split()[2])
            if grab:
                if len(line.split()) > 2:
                    gradient[atomcount,0]=-1*float(line.split()[-3])
                    gradient[atomcount,1]=-1*float(line.split()[-2])
                    gradient[atomcount,2]=-1*float(line.split()[-1])
                    atomcount+=1
                    if atomcount == numatoms:
                        grab=False
            if pcgrab:
                if len(line.split()) > 2:
                    pcgradient[pccount,0]=-1*float(line.split()[-3])
                    pcgradient[pccount,1]=-1*float(line.split()[-2])
                    pcgradient[pccount,2]=-1*float(line.split()[-1])
                    pccount+=1
                    if pccount == numpc:
                        break
            if Grad:
                if 'Total Forces' in line:
                    grab=True
                if 'Maximal derivative component:' in line:
                    grab=False
            if PC is True:
                if 'Forces on external charges' in line:
                    pcgrab=True
                if 'Dipole moment:' in line:
                    pcgrab=False

    if Grad is True and PC is True:
        if np.any(pcgradient) is False:
            print("Error: PCgradient array from DFTB output is zero. Something went wrong.")
            ashexit()

    return energy,gradient,pcgradient

def create_pcfile(filename,coords,pchargelist):
    #https://xtb-docs.readthedocs.io/en/latest/pcem.html
    with open(filename, 'w') as pcfile:
        #pcfile.write(str(len(pchargelist))+'\n')
        for p,c in zip(pchargelist,coords):
            line = "{} {} {} {}".format(c[0], c[1], c[2], p)
            pcfile.write(line+'\n')
