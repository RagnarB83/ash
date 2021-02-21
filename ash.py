# ASH - A GENERAL COMPCHEM AND QM/MM ENVIRONMENT

#Python libraries
import os
import shutil
import numpy as np
import copy
import subprocess as sp
import glob
import sys
import inspect
import time

import constants
import elstructure_functions




#ASH
from functions_general import blankline,BC,listdiff,print_time_rel,print_time_rel_and_tot,pygrep,printdebug,read_intlist_from_file
import functions_coords
from functions_coords import get_molecules_from_trajectory,eldict_covrad,write_pdbfile
import functions_parallel
from functions_parallel import Singlepoint_parallel,run_QMMM_SP_in_parallel
from functions_freq import AnFreq,NumFreq,approximate_full_Hessian_from_smaller,calc_rotational_constants
#Spinprojection
from functions_spinprojection import SpinProjectionTheory
#Surface
from functions_surface import calc_surface,calc_surface_fromXYZ,read_surfacedict_from_file
import settings_ash
from ash_header import print_ash_header
#QMcode interfaces
from interface_ORCA import ORCATheory
from interface_Psi4 import Psi4Theory
from interface_dalton import DaltonTheory
from interface_pyscf import PySCFTheory
from interface_MRCC import MRCCTheory
from interface_CFour import CFourTheory
#MM: external and internal
from interface_OpenMM import OpenMMTheory
from functions_MM import NonBondedTheory,UFFdict,UFF_modH_dict,LJCoulpy,coulombcharge,LennardJones,LJCoulombv2,LJCoulomb,MMforcefield_read
#QM/MM
from functions_QMMM import QMMMTheory
from functions_polembed import PolEmbedTheory
#Solvation
import functions_solv
# Geometry optimization
from functions_optimization import SimpleOpt,BernyOpt
from interface_geometric import geomeTRICOptimizer
#Workflows, benchmarking etc
import workflows
import highlevel_workflows
from workflows import ReactionEnergy,thermochemprotocol_reaction,thermochemprotocol_single
import benchmarking
from benchmarking import run_benchmark
#Other interfaces
import interface_crest


#Julia dependency
#Current behaviour: We try to import, if not possible then we continue
load_julia = True
if load_julia is True:
    try:
        print("Import PyJulia interface")
        from julia.api import Julia
        from julia import Main
        #Hungarian package needs to be installed
        try:
            from julia import Hungarian
        except:
            print("Problem loading Julia packages: Hungarian")
        ashpath = os.path.dirname(ash.__file__)
        #Various Julia functions
        print("Loading Julia functions")
        Main.include(ashpath + "/functions_julia.jl")
    except:
        print("Problem importing Pyjulia")
        print("Make sure Julia is installed, PyJulia within Python, Pycall within Julia, Julia packages have been installed and you are using python-jl")
        print("Python routines will be used instead when possible")
        #TODO: We should here set a variable that would pick py version of routines instead

#############################################################

# ASH Fragment class
class Fragment:
    def __init__(self, coordsstring=None, fragfile=None, xyzfile=None, pdbfile=None, chemshellfile=None, coords=None, elems=None, connectivity=None,
                 atomcharges=None, atomtypes=None, conncalc=True, scale=None, tol=None, printlevel=2, charge=None,
                 mult=None, label=None, readchargemult=False):
        #Label for fragment (string). Useful for distinguishing different fragments
        self.label=label

        #Printlevel. Default: 2 (slightly verbose)
        self.printlevel=printlevel

        #New. Charge and mult attribute of fragment. Useful for workflows
        self.charge = charge
        self.mult = mult

        if self.printlevel >= 2:
            print("New ASH fragment object")
        self.energy = None
        self.elems=[]
        self.coords=[]
        self.connectivity=[]
        self.atomcharges = []
        self.atomtypes = []
        self.Centralmainfrag = []
        self.formula = None
        if atomcharges is not None:
            self.atomcharges=atomcharges
        if atomtypes is not None:
            self.atomtypes=atomtypes
        #Hessian. Can be added by Numfreq/Anfreq job
        self.hessian=[]

        # Something perhaps only used by molcrys but defined here. Needed for print_system
        # Todo: revisit this
        self.fragmenttype_labels=[]
        #Here either providing coords, elems as lists. Possibly reading connectivity also
        if coords is not None:
            #self.add_coords(coords,elems,conn=conncalc)
            #Adding coords as list of lists. Possible conversion from numpy array below.
            self.coords=[list(i) for i in coords]
            self.elems=elems
            self.update_attributes()
            #If connectivity passed
            if connectivity is not None:
                conncalc=False
                self.connectivity=connectivity
            #If connectivity requested (default for new frags)
            if conncalc==True:
                self.calc_connectivity(scale=scale, tol=tol)
        #If coordsstring given, read elems and coords from it
        elif coordsstring is not None:
            self.add_coords_from_string(coordsstring, scale=scale, tol=tol)
        #If xyzfile argument, run read_xyzfile
        elif xyzfile is not None:
            self.read_xyzfile(xyzfile, readchargemult=readchargemult,conncalc=conncalc)
        elif pdbfile is not None:
            self.read_pdbfile(pdbfile, conncalc=conncalc)
        elif chemshellfile is not None:
            self.read_chemshellfile(chemshellfile, conncalc=conncalc)
        elif fragfile is not None:
            self.read_fragment_from_file(fragfile)
    def update_attributes(self):
        self.nuccharge = functions_coords.nucchargelist(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(0, self.numatoms))
        #Unnecessary alias ? Todo: Delete
        self.allatoms = self.atomlist
        self.mass = functions_coords.totmasslist(self.elems)
        self.list_of_masses = functions_coords.list_of_masses(self.elems)
        #Elemental formula
        self.formula = functions_coords.elemlisttoformula(self.elems)
        #Pretty formula without 1
        self.prettyformula = self.formula.replace('1','')

        if self.printlevel >= 2:
            print("Fragment numatoms: {} Formula: {}  Label: {}".format(self.numatoms,self.prettyformula,self.label))

    #Add coordinates from geometry string. Will replace.
    #Todo: Needs more work as elems and coords may be lists or numpy arrays
    def add_coords_from_string(self, coordsstring, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Getting coordinates from string:", coordsstring)
        if len(self.coords)>0:
            if self.printlevel >= 2:
                print("Fragment already contains coordinates")
                print("Adding extra coordinates")
        coordslist=coordsstring.split('\n')
        for count, line in enumerate(coordslist):
            if len(line)> 1:
                self.elems.append(line.split()[0])
                self.coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        self.update_attributes()
        self.calc_connectivity(scale=scale, tol=tol)
    #Replace coordinates by providing elems and coords lists. Optional: recalculate connectivity
    def replace_coords(self, elems, coords, conn=False, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Replacing coordinates in fragment.")
        
        self.elems=elems
        # Adding coords as list of lists. Possible conversion from numpy array below.
        self.coords = [list(i) for i in coords]
        self.update_attributes()
        if conn==True:
            self.calc_connectivity(scale=scale, tol=tol)
    def delete_coords(self):
        self.coords=[]
        self.elems=[]
        self.connectivity=[]
    def add_coords(self, elems,coords,conn=True, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Adding coordinates to fragment.")
        if len(self.coords)>0:
            if self.printlevel >= 2:
                print("Fragment already contains coordinates")
                print("Adding extra coordinates")
        print(elems)
        print(type(elems))
        self.elems = self.elems+list(elems)
        self.coords = self.coords+coords
        self.update_attributes()
        if conn==True:
            self.calc_connectivity(scale=scale, tol=tol)
    def print_coords(self):
        if self.printlevel >= 2:
            print("Defined coordinates (Å):")
        functions_coords.print_coords_all(self.coords,self.elems)
    #Read Amber coordinate file?
    def read_amberinpcrdfile(self,filename,conncalc=False):
        #Todo: finish
        pass
    #Read GROMACS coordinates file
    def read_grofile(self,filename,conncalc=False):
        #Todo: finish
        pass
    #Read CHARMM? coordinate file?
    def read_charmmfile(self,filename,conncalc=False):
        #Todo: finish
        pass
    def read_chemshellfile(self,filename,conncalc=False, scale=None, tol=None):
        #Read Chemshell fragment file (.c ending)
        if self.printlevel >= 2:
            print("Reading coordinates from Chemshell file \"{}\" into fragment".format(filename))
        try:
            elems, coords = functions_coords.read_fragfile_xyz(filename)
        except FileNotFoundError:
            print("File {} not found".format(filename))
            exit()
        self.coords = coords
        self.elems = elems

        self.update_attributes()
        if conncalc is True:
            self.calc_connectivity(scale=scale, tol=tol)
        else:
            # Read connectivity list
            print("Not reading connectivity from file")

    #Read PDB file
    def read_pdbfile(self,filename,conncalc=True, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Reading coordinates from PDBfile \"{}\" into fragment".format(filename))
        residuelist=[]
        #If elemcolumn found
        elemcol=[]
        #Not atomtype but atomname
        atom_name=[]
        atomindex=[]
        residname=[]

        #TODO: Check. Are there different PDB formats?
        #used this: https://cupnet.net/pdb-format/
        try:
            with open(filename) as f:
                for line in f:
                    if 'ATOM' in line:
                        atomindex.append(float(line[6:11].replace(' ','')))
                        atom_name.append(line[12:16].replace(' ',''))
                        residname.append(line[17:20].replace(' ',''))
                        residuelist.append(line[22:26].replace(' ',''))
                        coords_x=float(line[30:38].replace(' ',''))
                        coords_y=float(line[38:46].replace(' ',''))
                        coords_z=float(line[46:54].replace(' ',''))
                        self.coords.append([coords_x,coords_y,coords_z])
                        elem=line[76:78].replace(' ','')
                        if len(elem) != 0:
                            if len(elem)==2:
                                #Making sure second elem letter is lowercase
                                elemcol.append(elem[0]+elem[1].lower())
                            else:
                                elemcol.append(elem)    
                        #self.coords.append([float(line.split()[6]), float(line.split()[7]), float(line.split()[8])])
                        #elemcol.append(line.split()[-1])
                        #residuelist.append(line.split()[3])
                        #atom_name.append(line.split()[3])
                    if 'HETATM' in line:
                        print("HETATM line in file found. Please rename to ATOM")
                        exit()
        except FileNotFoundError:
            print("File {} does not exist!".format(filename))
            exit()
        if len(elemcol) != len(self.coords):
            print("len coords", len(self.coords))
            print("len elemcol", len(elemcol))            
            print("did not find same number of elements as coordinates")
            print("Need to define elements in some other way")
            exit()
        else:
            self.elems=elemcol
        self.update_attributes()
        if conncalc is True:
            self.calc_connectivity(scale=scale, tol=tol)
    #Read XYZ file
    def read_xyzfile(self,filename, scale=None, tol=None, readchargemult=False,conncalc=True):
        if self.printlevel >= 2:
            print("Reading coordinates from XYZfile {} into fragment".format(filename))
        with open(filename) as f:
            for count,line in enumerate(f):
                if count == 0:
                    self.numatoms=int(line.split()[0])
                elif count == 1:
                    if readchargemult is True:
                        self.charge=int(line.split()[0])
                        self.mult=int(line.split()[1])
                elif count > 1:
                    if len(line) > 3:
                        self.elems.append(line.split()[0])
                        self.coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        if self.numatoms != len(self.coords):
            print("Number of atoms in header not equal to number of coordinate-lines. Check XYZ file!")
            exit()
        self.update_attributes()
        if conncalc is True:
            self.calc_connectivity(scale=scale, tol=tol)
    def set_energy(self,energy):
        self.energy=float(energy)
    # Get coordinates for specific atoms (from list of atom indices)
    def get_coords_for_atoms(self, atoms):
        subcoords=[self.coords[i] for i in atoms]
        subelems=[self.elems[i] for i in atoms]
        return subcoords,subelems
    #Calculate connectivity (list of lists) of coords
    def calc_connectivity(self, conndepth=99, scale=None, tol=None, codeversion='julia' ):
        #Using py version if molecule is small. Otherwise Julia by default
        if len(self.coords) < 100:
            codeversion='py'
        elif len(self.coords) > 10000:
            if self.printlevel >= 2:
                print("Atom number > 10K. Connectivity calculation could take a while")

        
        if scale == None:
            try:
                scale = settings_ash.scale
                tol = settings_ash.tol
                if self.printlevel >= 2:
                    print("Using global scale and tol parameters from settings_ash. Scale: {} Tol: {} ".format(scale, tol ))

            except:
                scale = 1.0
                tol = 0.1
                if self.printlevel >= 2:
                    print("Exception: Using hard-coded scale and tol parameters. Scale: {} Tol: {} ".format(scale, tol ))
        else:
            if self.printlevel >= 2:
                print("Using scale: {} and tol: {} ".format(scale, tol))

        #Setting scale and tol as part of object for future usage (e.g. QM/MM link atoms)
        self.scale = scale
        self.tol = tol

        # Calculate connectivity by looping over all atoms
        timestampA=time.time()
        
        
        if codeversion=='py':
            print("Calculating connectivity of fragment using py")
            timestampB = time.time()
            fraglist = functions_coords.calc_conn_py(self.coords, self.elems, conndepth, scale, tol)
            print_time_rel(timestampB, modulename='calc connectivity py')
        elif codeversion=='julia':
            print("Calculating connectivity of fragment using julia")
            # Import Julia
            try:
                from julia.api import Julia
                from julia import Main
                timestampB = time.time()
                fraglist_temp = Main.Juliafunctions.calc_connectivity(self.coords, self.elems, conndepth, scale, tol,
                                                                      eldict_covrad)
                fraglist = []
                # Converting from numpy to list of lists
                for sublist in fraglist_temp:
                    fraglist.append(list(sublist))
                print_time_rel(timestampB, modulename='calc connectivity julia')
            except:
                print(BC.FAIL,"Problem importing Pyjulia (import julia)", BC.END)
                print("Make sure Julia is installed and PyJulia module available, and that you are using python-jl")
                print(BC.FAIL,"Using Python version instead (slow for large systems)", BC.END)
                fraglist = functions_coords.calc_conn_py(self.coords, self.elems, conndepth, scale, tol)



        if self.printlevel >= 2:
            pass
            #print_time_rel(timestampA, modulename='calc connectivity full')
        #flat_fraglist = [item for sublist in fraglist for item in sublist]
        self.connectivity=fraglist
        #Calculate number of atoms in connectivity list of lists
        conn_number_sum=0
        for l in self.connectivity:
            conn_number_sum+=len(l)
        if self.numatoms != conn_number_sum:
            print(BC.FAIL,"Connectivity problem", BC.END)
            exit()
        self.connected_atoms_number=conn_number_sum

    def update_atomcharges(self, charges):
        self.atomcharges = charges
    def update_atomtypes(self, types):
        self.atomtypes = types
    #Adding fragment-type info (used by molcrys, identifies whether atom is mainfrag, counterfrag1 etc.)
    #Old slow version below. To be deleted
    def old_add_fragment_type_info(self,fragmentobjects):
        # Create list of fragment-type label-list
        self.fragmenttype_labels = []
        for i in self.atomlist:
            for count,fobject in enumerate(fragmentobjects):
                if i in fobject.flat_clusterfraglist:
                    self.fragmenttype_labels.append(count)
    #Adding fragment-type info (used by molcrys, identifies whether atom is mainfrag, counterfrag1 etc.)
    #This one is fast
    def add_fragment_type_info(self,fragmentobjects):
        # Create list of fragment-type label-list
        combined_flat_clusterfraglist = []
        combined_flat_labels = []
        #Going through objects, getting flat atomlists for each object and combine (combined_flat_clusterfraglist)
        #Also create list of labels (using fragindex) for each atom
        for fragindex,frago in enumerate(fragmentobjects):
            combined_flat_clusterfraglist.extend(frago.flat_clusterfraglist)
            combined_flat_labels.extend([fragindex]*len(frago.flat_clusterfraglist))
        #Getting indices required to sort atomindices in ascending order
        sortindices = np.argsort(combined_flat_clusterfraglist)
        #labellist contains unsorted list of labels
        #Now ordering the labels according to the sort indices
        self.fragmenttype_labels =  [combined_flat_labels[i] for i in sortindices]

    #Molcrys option:
    def add_centralfraginfo(self,list):
        self.Centralmainfrag = list
    def write_xyzfile(self,xyzfilename="Fragment-xyzfile.xyz"):
        #Energy written to XYZ title-line if present. Otherwise: None
        with open(xyzfilename, 'w') as ofile:
            ofile.write(str(len(self.elems)) + '\n')
            if self.energy is None:
                ofile.write("Energy: None" + '\n')
            else:
                ofile.write("Energy: {:14.8f}".format(self.energy) + '\n')
            for el, c in zip(self.elems, self.coords):
                line = "{:4} {:14.8f} {:14.8f} {:14.8f}".format(el, c[0], c[1], c[2])
                ofile.write(line + '\n')
        if self.printlevel >= 2:
            print("Wrote XYZ file:", xyzfilename)
    #Print system-fragment information to file. Default name of file: "fragment.ygg
    def print_system(self,filename='fragment.ygg'):
        if self.printlevel >= 2:
            print("Printing fragment to disk:", filename)

        #Setting atomcharges, fragmenttype_labels and atomtypes to dummy lists if empty
        if len(self.atomcharges)==0:
            self.atomcharges=[0.0 for i in range(0,self.numatoms)]
        if len(self.fragmenttype_labels)==0:
            self.fragmenttype_labels=[0 for i in range(0,self.numatoms)]
        if len(self.atomtypes)==0:
            self.atomtypes=['None' for i in range(0,self.numatoms)]

        with open(filename, 'w') as outfile:
            outfile.write("Fragment: \n")
            outfile.write("Num atoms: {}\n".format(self.numatoms))
            outfile.write("Formula: {}\n".format(self.formula))
            outfile.write("Energy: {}\n".format(self.energy))
            outfile.write("\n")
            outfile.write(" Index    Atom         x                  y                  z               charge        fragment-type        atom-type\n")
            outfile.write("---------------------------------------------------------------------------------------------------------------------------------\n")
            for at, el, coord, charge, label, atomtype in zip(self.atomlist, self.elems,self.coords,self.atomcharges, self.fragmenttype_labels, self.atomtypes):
                line="{:>6} {:>6}  {:17.11f}  {:17.11f}  {:17.11f}  {:14.8f} {:12d} {:>21}\n".format(at, el,coord[0], coord[1], coord[2], charge, label, atomtype)
                outfile.write(line)
            outfile.write(
                "===========================================================================================================================================\n")
            #outfile.write("elems: {}\n".format(self.elems))
            #outfile.write("coords: {}\n".format(self.coords))
            #outfile.write("list of masses: {}\n".format(self.list_of_masses))
            outfile.write("atomcharges: {}\n".format(self.atomcharges))
            outfile.write("Sum of atomcharges: {}\n".format(sum(self.atomcharges)))
            outfile.write("atomtypes: {}\n".format(self.atomtypes))
            outfile.write("connectivity: {}\n".format(self.connectivity))
            outfile.write("Centralmainfrag: {}\n".format(self.Centralmainfrag))

    #Reading fragment from file. File created from Fragment.print_system
    def read_fragment_from_file(self, fragfile):
        if self.printlevel >= 2:
            print("Reading ASH fragment from file:", fragfile)
        coordgrab=False
        coords=[]
        elems=[]
        atomcharges=[]
        atomtypes=[]
        fragment_type_labels=[]
        connectivity=[]
        #Only used by molcrys:
        Centralmainfrag = []
        with open(fragfile) as file:
            for n, line in enumerate(file):
                if 'Num atoms:' in line:
                    numatoms=int(line.split()[-1])
                if coordgrab==True:
                    #If end of coords section
                    if '===============' in line:
                        coordgrab=False
                        continue
                    elems.append(line.split()[1])
                    coords.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                    atomcharges.append(float(line.split()[5]))
                    fragment_type_labels.append(int(line.split()[6]))
                    atomtypes.append(line.split()[7])

                if '--------------------------' in line:
                    coordgrab=True
                if 'Centralmainfrag' in line:
                    if '[]' not in line:
                        l = line.lstrip('Centralmainfrag:')
                        l = l.replace('\n','')
                        l = l.replace(' ','')
                        l = l.replace('[','')
                        l = l.replace(']','')
                        Centralmainfrag = [int(i) for i in l.split(',')]
                #Incredibly ugly but oh well
                if 'connectivity:' in line:
                    l=line.lstrip('connectivity:')
                    l=l.replace(" ", "")
                    for x in l.split(']'):
                        if len(x) < 1:
                            break
                        y=x.strip(',[')
                        y=y.strip('[')
                        y=y.strip(']')
                        try:
                            connlist=[int(i) for i in y.split(',')]
                        except:
                            connlist=[]
                        connectivity.append(connlist)
        self.elems=elems
        self.coords=coords
        self.atomcharges=atomcharges
        self.atomtypes=atomtypes
        self.update_attributes()
        self.connectivity=connectivity
        self.Centralmainfrag = Centralmainfrag


#Single-point energy function
def Singlepoint(fragment=None, theory=None, Grad=False):
    print("")
    '''
    The Singlepoint function carries out a single-point energy calculation
    :param fragment:
    :type fragment: ASH object of class Fragment
    :param theory:
    :type theory: ASH theory object
    :param Grad: whether to do Gradient or not.
    :type Grad: Boolean.
    '''
    if fragment is None or theory is None:
        print(BC.FAIL,"Singlepoint requires a fragment and a theory object",BC.END)
        exit(1)
    coords=fragment.coords
    elems=fragment.elems
    # Run a single-point energy job
    if Grad ==True:
        print(BC.WARNING,"Doing single-point Energy+Gradient job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
        # An Energy+Gradient calculation where we change the number of cores to 12
        energy,gradient= theory.run(current_coords=coords, elems=elems, Grad=True)
        print("Energy: ", energy)
        return energy,gradient
    else:
        print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)

        energy = theory.run(current_coords=coords, elems=elems)
        print("Energy: ", energy)

        #Now adding total energy to fragment
        fragment.energy=energy

        return energy

#Theory classes

# Theory object that always gives zero energy and zero gradient. Useful for setting constraints
class ZeroTheory:
    def __init__(self, fragment=None, charge=None, mult=None, printlevel=None, nprocs=1, label=None):
        self.nprocs=nprocs
        self.charge=charge
        self.mult=mult
        self.printlevel=printlevel
        self.label=label
        self.fragment=fragment
        pass
    def run(self, current_coords=None, elems=None, Grad=False, PC=False, nprocs=None ):
        self.energy = 0.0
        #Numpy object
        self.gradient = np.zeros((len(elems), 3))
        return self.energy,self.gradient


