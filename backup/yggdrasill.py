# YGGDRASILL - A GENERAL COMPCHEM AND QM/MM ENVIRONMENT

from constants import *
from elstructure_functions import *
import os
from functions_solv import *
from functions_coords import *
from functions_ORCA import *
from functions_general import *
import settings_yggdrasill
from functions_MM import *

def print_yggdrasill_header():
    programversion = 0.1
    #http://asciiflow.com
    #https://textik.com/#91d6380098664f89
    #https://www.gridsagegames.com/rexpaint/

    ascii_banner="""
      X          X
      X          X
      XX        XX     X
       XX  X    X     XX
        XXXX   XX    XX
  XX      XXX XX  X XX
   XX    X  XXX   XXX
    XX   X  XX     X
      XXXXX  X    XX
        XXXX X  XX
           XXXXX
           XXXXX
           XXXXX
           XXXXX
           XXXXX
           XXXXX
"""
    print(BC.WARNING,"--------------------------------------------------",BC.END)
    print(BC.WARNING,"--------------------------------------------------",BC.END)
    print(BC.WARNING,BC.BOLD,"YGGDRASILL version", programversion,BC.END)
    print(BC.WARNING,"A GENERAL COMPCHEM AND QM/MM ENVIRONMENT", BC.END)
    print(BC.WARNING,BC.BOLD,ascii_banner,BC.END)
    print(BC.WARNING,"--------------------------------------------------",BC.END)
    print(BC.WARNING,"--------------------------------------------------",BC.END)

#Optimizer class
class Optimizer:
    def __init__(self, fragment, theory, optimizer):
        self.fragment=fragment
        self.theory=theory
        self.optimizer=optimizer
    def run(self):
        print(BC.WARNING, BC.BOLD, "------------STARTING OPTIMIZER-------------", BC.END)
        print("Running Optimizer")
        #Launch opt job here
        maxiter=50
        #Initial geo
        coords=self.fragment.coords
        elems=self.fragment.elems
        for i in range(1,maxiter):
            print("Geometry optimization step", i)
            #Running E+G theory job.
            self.theory.run(coords=coords, elems=elems, Grad=True)
            print("here")
            exit()
def steepest_descent(coords):
    scaling=0.01





#Theory classes

# Different MM theories

# Simple nonbonded MM theory. Charges and LJ-potentials
class NonBondedTheory:
    def __init__(self, charges = "", LJ=False):
        self.atom_charges = charges
    def update_charges(self,charges):
        self.atom_charges = charges
    def run(self):
        self.Coulombchargeenergy = coulombcharge(self.atom_charges)
        self.LJenergy = 0
        self.MMEnergy=self.Coulombchargeenergy+self.LJenergy
        #Todo: Add LJ part
        return

class QMMMTheory:
    def __init__(self, fragment, qm_theory, qmatoms, mm_theory="" , atomcharges=""):
        print(BC.WARNING,BC.BOLD,"------------STARTING QM/MM INTERFACE-------------", BC.END)
        #Theory level definitions
        self.qm_theory=qm_theory
        self.qm_theory_name = self.qm_theory.__class__.__name__
        self.mm_theory=mm_theory
        self.mm_theory_name = self.mm_theory.__class__.__name__
        if self.mm_theory_name == "str":
            self.mm_theory_name="None"
        print("QM-theory:", self.qm_theory_name)
        print("MM-theory:", self.mm_theory_name)
        self.fragment=fragment
        #System coords and elems
        self.coords=fragment.coords
        self.elems=fragment.elems
        # Region definitions
        self.allatoms=list(range(0,len(self.elems)))
        self.qmatoms=qmatoms
        print("QM region:", self.qmatoms)
        self.qmcoords=[self.coords[i] for i in self.qmatoms]
        self.qmelems=[self.elems[i] for i in self.qmatoms]
        self.mmatoms=listdiff(self.allatoms,self.qmatoms)
        self.mmcoords=[self.coords[i] for i in self.mmatoms]
        self.mmelems=[self.elems[i] for i in self.mmatoms]
        # Charges defined for regions
        self.qmcharges=[atomcharges[i] for i in self.qmatoms]
        self.mmcharges=[atomcharges[i] for i in self.mmatoms]
        if mm_theory != "":
            #Setting QM charges to 0 since electrostatic embedding
            self.charges=[]
            for i,c in enumerate(atomcharges):
                if i in self.mmatoms:
                    self.charges.append(c)
                else:
                    self.charges.append(0.0)
            mm_theory.update_charges(self.charges)
            blankline()
            print("Setting charges of QM atoms to 0:")
            for i in self.allatoms:
                if i in qmatoms:
                    print("QM atom {} charge: {}".format(i, self.charges[i]))
                else:
                    print("MM atom {} charge: {}".format(i, self.charges[i]))
        blankline()

    def run(self, Grad=False):

        if self.qm_theory_name=="ORCATheory":
            print(BC.OKBLUE,BC.BOLD,"------------QM/MM ORCA-INTERFACE-------------", BC.END)

            #Create inputfile with generic name
            inputfilename="orca-input"
            print("Creating ORCA inputfile:", inputfilename+'.inp')
            create_orca_input_pc(inputfilename,self.qmelems,self.qmcoords,
                                    self.qm_theory.orcasimpleinput,self.qm_theory.orcablocks,
                                    self.qm_theory.charge,self.qm_theory.mult)
            #Pointcharge file
            create_orca_pcfile(inputfilename, self.mmelems, self.mmcoords, self.mmcharges)
            #Run inputfile. Take nprocs argument. Orcadir argument??
            print(BC.OKGREEN,"------------Running ORCA-------------", BC.END)
            print("Calculation started at:")
            printDate()
            print("...")
            #Doing gradient or not.
            if Grad==True:
                run_orca_SP_ORCApar(self.qm_theory.orcadir, inputfilename+'.inp', nprocs=1, Grad=True)
            else:
                run_orca_SP_ORCApar(self.qm_theory.orcadir, inputfilename+'.inp', nprocs=1)
            print("Calculation ended at:")
            printDate()
            print(BC.OKGREEN, "------------ORCA calculation done-------------", BC.END)
            #Check if finished. Grab energy
            outfile=inputfilename+'.out'
            if checkORCAfinished(outfile) == True:
                self.QMEnergy=finalenergygrab(outfile)
                blankline()
                print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
            else:
                print("Problem with ORCA run")
                print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                exit()
            blankline()
        elif self.qm_theory_name == "xTBTheory":
            print("not yet implemented")
        elif self.qm_theory_name == "Psi4Theory":
            print("not yet implemented")
        elif self.qm_theory_name == "DaltonTheory":
            print("not yet implemented")
        elif self.qm_theory_name == "NWChemtheory":
            print("not yet implemented")
        else:
            print("invalid QM theory")

        # MM theory
        if self.mm_theory_name == "NonBondedTheory":
            self.mm_theory.run()
            self.MMEnergy=self.mm_theory.MMEnergy
        else:
            self.MMEnergy=0

        #Final QM/MM Energy
        self.QM_MM_Energy= self.QMEnergy+self.MMEnergy
        blankline()
        print("ORCA energy:", self.QMEnergy)
        print("MM energy:", self.MMEnergy)
        print("QM/MM total energy:", self.QM_MM_Energy)
        blankline()
        print(BC.WARNING,BC.BOLD,"------------ENDING QM/MM INTERFACE-------------",BC.END)


class xTBTheory:
    def __init__(self, xtbdir, fragment, charge, mult, xtbmethod):
        self.xtbdir = xtbdir
        self.fragment=fragment
        self.coords=fragment.coords
        self.elems=fragment.elems
        self.charge=charge
        self.mult=mult
        self.xtbmethod=xtbmethod
    def run(self):
        print("------------STARTING XTB INTERFACE-------------")
        #Create XYZfile with generic name for xTB to run
        inputfilename="xtb-inpfile"
        print("Creating inputfile:", inputfilename+'.xyz')
        write_xyzfile(self.elems, self.coords, inputfilename)
        #Run inputfile. Take nprocs argument. Orcadir argument??
        print("------------Running xTB-------------")
        print("...")
        run_xtb_SP_serial(self.xtbdir, xtbmethod, inputfilename+'.xyz', self.charge, self.mult)
        print("------------xTB calculation done-------------")
        #Check if finished. Grab energy
        outfile=inputfilename+'.out'
        xtbfinalenergygrab(outfile)
        print("------------ENDING XTB-INTERFACE-------------")
        #if checkORCAfinished(outfile) == True:
        #    self.energy=finalenergygrab(outfile)
        ##    print("ORCA energy:", self.energy)
        #else:
        #    print("Problem with ORCA run")
        #    exit()

class ORCATheory:
    def __init__(self, orcadir, fragment='', charge='', mult='', orcasimpleinput='', orcablocks=''):
        self.orcadir = orcadir
        if fragment != '':
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        self.charge=int(charge)
        self.mult=int(mult)
        self.orcasimpleinput=orcasimpleinput
        self.orcablocks=orcablocks
    def run(self, coords='', elems='', Grad=False):
        print("------------STARTING ORCA INTERFACE-------------")
        #Create inputfile with generic name
        inputfilename="orca-input"
        print("Creating inputfile:", inputfilename+'.inp')
        create_orca_input_plain(inputfilename,self.elems,self.coords,self.orcasimpleinput,self.orcablocks,self.charge,self.mult)
        #Run inputfile. Take nprocs argument. Orcadir argument??
        print("------------Running ORCA-------------")
        print("Calculation started at:")
        printDate()
        print("...")
        # Doing gradient or not.
        if Grad == True:
            run_orca_SP_ORCApar(self.orcadir, inputfilename + '.inp', nprocs=1, Grad=True)
        else:
            run_orca_SP_ORCApar(self.orcadir, inputfilename + '.inp', nprocs=1)
        print("Calculation ended at:")
        printDate()
        print("------------ORCA calculation done-------------")
        #Check if finished. Grab energy
        outfile=inputfilename+'.out'
        if checkORCAfinished(outfile) == True:
            self.energy=finalenergygrab(outfile)
            print("ORCA energy:", self.energy)
            print("------------ENDING ORCA-INTERFACE-------------")
        else:
            print("Problem with ORCA run")
            print("------------ENDING ORCA-INTERFACE-------------")
            exit()

# Fragment class
class Fragment:
    def __init__(self, coordsstring="", xyzfile=""):
        self.elems=[]
        self.coords=[]
        self.connectivity=[]
        #If coordsstring given, read elems and coords from it
        if coordsstring != "":
            self.coords_from_string(coordsstring)
        #If xyzfile argument, run read_xyzfile
        if len(xyzfile) > 1:
            self.read_xyzfile(xyzfile)
        self.nuccharge = nucchargelist(self.elems)
        self.mass = totmasslist(self.elems)
    #Add coordinates from geometry string. Will replace.
    def coords_from_string(self, coordsstring):
        coordslist=coordsstring.split('\n')
        for count, line in enumerate(coordslist):
            if len(line)> 1:
                self.elems.append(line.split()[0])
                self.coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        self.numatoms=len(self.elems)
    #Replace coordinates by providing elems and coords lists.
    def replace_coords(self, elems, coords):
        self.elems=elems
        self.coords=coords
        self.numatoms=len(self.elems)
    def add_coords(self, elems,coords):
        self.elems = self.elems+elems
        self.coords = self.coords+coords
    #Read XYZ file
    def read_xyzfile(self,filename):
        with open(filename) as f:
            for count,line in enumerate(f):
                if count == 0:
                    self.numatoms=int(line.split()[0])
                if count > 1:
                    self.elems.append(line.split()[0])
                    self.coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    #Calculate connectivity (list of lists) of coords
    def calc_connectivity(self):
        self.atomlist=list(range(0,self.numatoms))
        self.connectivity = []
        #Going through each atom and getting recursively connected atoms
        testlist=self.atomlist
        #Removing atoms from atomlist until empty
        while len(testlist) > 0:
            for index in testlist:
                wholemol=get_molecule_members_loop(self.coords, self.elems, index, settings_yggdrasill.conndepth, settings_yggdrasill.scale, settings_yggdrasill.tol)
                if wholemol in self.connectivity:
                    continue
                else:
                    self.connectivity.append(wholemol)
                    for i in wholemol:
                        testlist.remove(i)
        #Calculate number of atoms in connectivity list of lists
        conn_number_sum=0
        for l in self.connectivity:
            conn_number_sum+=len(l)
        if self.numatoms != conn_number_sum:
            print("Connectivity problem")
            exit()
        self.connected_atoms_number=conn_number_sum

