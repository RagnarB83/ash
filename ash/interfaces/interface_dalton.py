import subprocess as sp
import os
import glob
import shutil

import ash.settings_ash
from ash.functions.functions_general import ashexit, BC,print_time_rel, check_program_location
from ash.modules.module_coords import elemstonuccharges
from ash.modules.module_theory import QMTheory

class DaltonTheory (QMTheory):
    def __init__(self, daltondir=None, filename='dalton', printlevel=2,
                numcores=1, pe=False, potfile=None, label="Dalton", dalton_input=None, loprop=False,
                basis_name=None, basis_dir=None, scratchdir='.'):

        self.theorytype="QM"
        self.theorynamelabel="Dalton"
        self.analytic_hessian=False

        self.daltondir=check_program_location(daltondir,'daltondir','dalton')
        if basis_name is not None:
            self.basis_name=basis_name
        else:
            print("Please provide basis_name to DaltonTheory object")
            ashexit()

        self.filename=filename

        # Directory where basis sets are. If not defined, ASH will assume basis directory is one dir up
        self.basis_dir=basis_dir

        #For loprop
        self.loprop=loprop

        # Used to write name in MOLECULE.INP. Not necessary?
        self.moleculename="None"

        # Dalton input as a multi-line string
        self.dalton_input=dalton_input

        # Label to distinguish different Dalton objects
        self.label=label
        # Printlevel
        self.printlevel=printlevel
        # Setting numcores of object
        self.numcores=numcores

        # Setting energy to 0.0 for now
        self.energy=0.0

        self.pe=pe

        #Scratch dir
        os.environ['DALTON_TMPDIR'] = scratchdir

        print("Note: Dalton assumes mult=1 for even electrons and mult=2 for odd electrons.")
        if self.printlevel >=2:
            print("")
            print("Creating DaltonTheory object")
            print("Dalton dir:", self.daltondir)
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old Dalton files")
        list=[]
        #list.append(self.filename + '.gbw')
        for file in list:
            try:
                os.remove(file)
            except:
                pass
        # os.remove(self.filename + '.out')
        try:
            for tmpfile in glob.glob("self.filename*tmp"):
                os.remove(tmpfile)
        except:
            pass
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, qm_elems=None, elems=None, mm_elems=None, Grad=False, numcores=None,
            pe=None, potfile='', restart=False, label=None, charge=None, mult=None ):

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING DALTON INTERFACE-------------", BC.END)
        if pe is None:
            pe=self.pe

        print("pe: ", pe)
        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
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

        if numcores is None:
            numcores=self.numcores
        print("Running Dalton object with {} cores available".format(numcores))

        # DALTON.INP
        print("Creating inputfile: DALTON.INP")
        with open("DALTON.INP",'w') as dalfile:
            for substring in self.dalton_input.split('\n'):
                dalfile.write(substring+'\n')
                if 'DALTON' in substring:
                    if pe is True:
                        dalfile.write(".PEQM\n")
        # Copying to .dal as well for flexibility (dalton wrapper execution)
        shutil.copy("DALTON.INP",self.filename+".dal")

        # Write the ugly MOLECULE.INP
        uniq_elems=set(qm_elems)

        with open("MOLECULE.INP",'w') as molfile:
            molfile.write('BASIS\n')
            molfile.write('{}\n'.format(self.basis_name))
            molfile.write('{}\n'.format(self.moleculename))
            molfile.write('------------------------\n')
            molfile.write('AtomTypes={} NoSymmetry Angstrom Charge={}\n'.format(len(uniq_elems),charge))
            for uniqel in uniq_elems:
                nuccharge=float(elemstonuccharges([uniqel])[0])
                num_elem=qm_elems.count(uniqel)
                molfile.write('Charge={} Atoms={}\n'.format(nuccharge,num_elem))
                for el,coord in zip(qm_elems,current_coords):
                    if el == uniqel:
                        molfile.write('{}    {} {} {}\n'.format(el,coord[0],coord[1],coord[2]))

        # Copying to .mol as well for flexibility (dalton wrapper execution)
        shutil.copy("MOLECULE.INP",self.filename+".mol")
        # POTENTIAL FILE
        # Renaming potfile as POTENTIAL.INP
        if pe is True:
            os.rename(potfile,'POTENTIAL.INP')
            print("Rename potential file {} as POTENTIAL.INP".format(potfile))

        print("Charge: {}  Mult: {}".format(charge, mult))

        print("Launching Dalton")
        print("daltondir:", self.daltondir)
        if self.basis_dir is None:
            print("No Dalton basis_dir provided. Attempting to set BASDIR via daltondir:")
            os.environ['BASDIR'] = self.daltondir+"/../basis"
            print("BASDIR:", os.environ['BASDIR'])
        else:
            print("Using basis_dir:", self.basis_dir)
            os.environ['BASDIR'] = self.basis_dir

        if self.loprop:
            run_dalton_diff(self.daltondir,self.filename)
        else:
            run_dalton_serial(self.daltondir,self.filename)

        #Grab final energy
        #TODO: Support more than DFT energies
        #ALSO: Grab excitation energies etc
        with open(self.filename+'.out') as outfile:
            for line in outfile:
                if 'Final DFT energy:' in line:
                    self.energy=float(line.split()[-1])


        if self.energy==0.0:
            print("Problem. Energy not found in output:", self.filename+'.out')
            ashexit()
        if Grad:
            print("no grad yet")
            exit()
            return self.energy, self.gradient
        else:
            return self.energy


def run_dalton_serial(daltondir,filename):
    #exit()
    with open(filename+'.out', 'w') as ofile:

        process = sp.run([daltondir + '/dalton.x'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#dalton -get "AOONEINT AOPROPER" hf h2o
def run_dalton_diff(daltondir,filename):
    #exit()
    args=['-get','AOONEINT AOPROPER',filename,filename]
    with open(filename+'.out', 'w') as ofile:

        process = sp.run([daltondir + '/dalton',*args], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)


#NOTE: loprop requires pip install loprop   but also pip install daltools

def loprop_run(archive):
    try:
        import loprop
    except ModuleNotFoundError:
        print("loprop not installed. Please install loprop via pip install loprop")
        ashexit()
    try:
        import daltools
    except ModuleNotFoundError:
        print("daltools not installed. Please install daltools via pip install daltools")
        ashexit()

    with open('loprop.out', 'w') as ofile:
        args = ['-f',archive,'-l','0','-a','1']
        process = sp.run(['loprop',*args], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

    #Creates potential file
