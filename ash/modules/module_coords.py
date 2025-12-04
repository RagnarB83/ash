import math
sqrt = math.sqrt
pow = math.pow
import copy
import time
import numpy as np
import os
import subprocess as sp
#Defaultdict used by Reaction
from collections import defaultdict

from ash.functions.functions_general import ashexit, isint, listdiff, print_time_rel, BC, printdebug, print_line_with_mainheader, \
    print_line_with_subheader1, print_line_with_subheader1_end, print_line_with_subheader2, writelisttofile, load_julia_interface, \
    search_list_of_lists_for_index, natural_sort
#from ash.modules.module_singlepoint import ReactionEnergy
import ash.dictionaries_lists
import ash.settings_ash
import ash.constants
from ash.dictionaries_lists import eldict

#import ash

ashpath = os.path.dirname(ash.__file__)

#ASH Reaction class: connects list of ASH fragments and stoichiometry
#TODO: Check that the charge and multiplicity is consistent with formula. Maybe do in fragment instead?
#TODO: Check charge on both sides of reaction. Warning if different.
#TODO: Check if mult is different on both sides of reaction. Print warning

#FUNCTIONS that could interact with Reaction class:
# Singlepoint_reaction ?,
# thermochemprotocol_reaction ?
# Optimizer ? Probably not

class Reaction:
    def __init__(self, fragments, stoichiometry, label=None, unit='eV'):
        print_line_with_subheader1("New ASH reaction")

        #Reading fragments and checking for charge/mult
        self.fragments=fragments
        self.check_fragments()
        self.stoichiometry = stoichiometry
        #List of all elements in reaction
        self.elements = [item for sublist in [frag.elems for frag in fragments] for item in sublist]

        self.label=label

        self.unit=unit

        #List of energies for each fragment
        self.energies = []
        #Reaction energy
        self.reaction_energy=None

        #Keeping track of orbital-files: key: 'SCF':["frag1.gbw","frag2.gbw","frag3.gbw"], 'MP2nat':["frag1.gbw","frag2.gbw","frag3.gbw"]
        self.orbital_dictionary=defaultdict(lambda: [])
        #Keep track of various properties calculated
        self.properties=defaultdict(lambda: [])
    def reset_energies(self):
        #Reset energies etc
        self.energies = []
        self.reaction_energy=None
    def reset_all(self):
        #Reset energies etc
        self.energies = []
        self.reaction_energy=None
        self.properties=defaultdict(lambda: [])


    def check_fragments(self):
        for frag in self.fragments:
            if frag.charge == None or frag.mult == None:
                print("Error: Missing charge/mult information in fragment:",frag.formula)
                ashexit()
    def calculate_reaction_energy(self):
        if len(self.energies) == len(self.fragments):
            self.reaction_energy = ash.ReactionEnergy(list_of_energies=self.energies, stoichiometry=self.stoichiometry, unit=self.unit, silent=False,
                label=self.label)[0]
        else:
            print("Warning. Could not calculate reaction energy as we are missing energies for fragments")


# ASH Fragment class
class Fragment:
    def __init__(self, fragments=None, coordsstring=None, fragfile=None, databasefile=None, xyzfile=None, pdbfile=None, grofile=None,
                 amber_inpcrdfile=None, amber_prmtopfile=None, trexiofile=None, smiles=None,
                 chemshellfile=None, coords=None, elems=None, connectivity=None, atom=None, diatomic=None, diatomic_bondlength=None,
                 bondlength=None,
                 atomcharges=None, atomtypes=None, conncalc=False, scale=None, tol=None, printlevel=2, charge=None,
                 mult=None, label=None, readchargemult=False, use_atomnames_as_elements=False):

        # print_line_with_mainheader("Fragment")

        # Defining initial charge/mult attributes. Will be redefined
        self.charge=None
        self.mult=None

        # Setting initial dummy label. Possibly redefined below, either when reading in file or by label keyword
        self.label = None

        # Printlevel. Default: 2 (slightly verbose)
        self.printlevel = printlevel

        if self.printlevel >= 2:
            print_line_with_subheader1("New ASH fragment")
        # Minimal ASH Fragment
        if self.printlevel > 0:
            print("ASH Fragment creation")
        self.energy = None
        self.elems = []
        # self.coords=np.empty_like([],shape=(0,3))
        self.coords = np.zeros((0, 3))
        self.connectivity = []
        self.atomcharges = []
        self.atomtypes = []
        # Optional PDB-information that might be stored (if PDB-file read)
        self.pdb_atomnames = None
        self.pdb_resnames = None
        self.pdb_chainlabels = None
        self.pdb_residlabels = None
        self.pdb_conect_lines =None
        self.pdb_topology=None  #New, use OpenMM to read PDB-file and get topology
        # Atomnames in a forcefield sense
        # self.atomnames = []
        self.Centralmainfrag = []
        self.formula = None
        if atomcharges is not None:
            self.atomcharges = atomcharges
        if atomtypes is not None:
            self.atomtypes = atomtypes
        # if atomnames is not None:
        #    self.atomnames=atomnames
        # Hessian. Can be added by Numfreq/Anfreq job
        self.hessian = None

        # Something perhaps only used by molcrys but defined here. Needed for print_system
        # Todo: revisit this
        self.fragmenttype_labels = []

        # Here either providing coords, elems as lists.
        ##############################
        # NOW PROCESSING INPUT DATA
        ##############################

        # Lists of elements and coordinates provided
        if coords is not None:
            # Adding coords as list of lists (or np.array). Conversion to numpy array
            self.coords = reformat_list_to_array(coords)
            if elems is None:
                print("Error: Coords list provided but no elems list. Exiting.")
                ashexit()
            if len(elems) != len(coords):
                print(f"Error: Coords list (len {len(coords)}) and elems list ({len(elems)}) have different lengths. Exiting.")
                ashexit()
            self.elems = elems
            # self.update_attributes()
            # If connectivity passed
            if connectivity != None:
                conncalc = False
                self.connectivity = connectivity
        
        # Fragment from input fragments
        elif fragments is not None:
            print("Creating fragments by combining input fragments")
            self.elems=[]
            for f in fragments:
                self.elems += f.elems
            self.coords=np.vstack([f.coords for f in fragments])

            # Use charge/mult if provided, otherwise use 
            if charge is None:
                print("Combining charge and multiplicities from input fragments")
                try:
                    charges_fragments = [f.charge for f in fragments]
                    charge = sum(charges_fragments)
                    mults_fragments = [f.mult for f in fragments]
                    spin_fragments = [(m-1)/2 for m in mults_fragments]
                    spin = sum(spin_fragments)
                    mult = int(2*spin+1)
                except TypeError:
                    print("Charges/multiplicities not found in inputfragments.")

        # Defining an atom
        elif atom is not None:
            print("Creating Atom Fragment")
            self.elems=[atom]
            self.coords = reformat_list_to_array([[0.0,0.0,0.0]])
            #self.update_attributes()
        # Defining a diatomic
        elif diatomic is not None:
            print("Creating Diatomic Fragment from formula and bondlength")
            if bondlength is None:
                #TODO: remove diatomic_bondlength and use bondlength only
                if diatomic_bondlength is None:
                    print(BC.FAIL,"diatomic option requires bondlength to be set. Exiting!", BC.END)
                    ashexit()
                else:
                    bondlength=diatomic_bondlength
            self.elems=molformulatolist(diatomic)
            if len(self.elems) != 2:
                print(f"Problem with molecular formula diatomic={diatomic} string!")
                ashexit()
            self.coords = reformat_list_to_array([[0.0,0.0,0.0],[0.0,0.0,float(bondlength)]])
            # self.update_attributes()
        # If coordsstring given, read elems and coords from it
        elif coordsstring is not None:
            self.add_coords_from_string(coordsstring, scale=scale, tol=tol, conncalc=conncalc)
        # If smiles given, create coords from it
        elif smiles is not None:
            self.create_coords_from_smiles(smiles)
        # If xyzfile argument, run read_xyzfile
        elif xyzfile is not None:
            if not os.path.isfile(xyzfile):
                print(f"XYZ-file {xyzfile} not found. Exiting.")
                ashexit()

            self.label = xyzfile.split('/')[-1].split('.')[0]
            self.read_xyzfile(xyzfile, readchargemult=readchargemult, conncalc=conncalc)
        # PDB-file
        elif pdbfile is not None:
            self.label = pdbfile.split('/')[-1].split('.')[0]
            self.read_pdbfile_openmm(pdbfile)
            #self.read_pdbfile_old(pdbfile, conncalc=False, use_atomnames_as_elements=use_atomnames_as_elements)
        # GROMACS GRO-file
        elif grofile is not None:
            self.label = grofile.split('/')[-1].split('.')[0]
            self.read_grofile(grofile, conncalc=False)
        # Amber CRD file (requires prmtop file as well)
        elif amber_inpcrdfile is not None:
            self.label = amber_inpcrdfile.split('/')[-1].split('.')[0]
            print("Reading Amber INPCRD file")
            if amber_prmtopfile is None:
                print("amber_prmtopfile argument must be provided as well!")
                ashexit()
            self.read_amberfile(inpcrdfile=amber_inpcrdfile, prmtopfile=amber_prmtopfile, conncalc=conncalc)
        # Chemshell-file (coordinates in Bohrs)
        elif chemshellfile is not None:
            self.label = chemshellfile.split('/')[-1].split('.')[0]
            self.read_chemshellfile(chemshellfile, conncalc=conncalc)
        # ASH fragment file
        elif fragfile is not None:
            self.label = fragfile.split('/')[-1].split('.')[0]
            self.read_fragment_from_file(fragfile)
        # Trexio
        elif trexiofile is not None:
            from ash.interfaces.interface_TREXIO import read_trexio_file
            print("Reading TREXIO file:", trexiofile)
            if 'h5' in trexiofile or 'hdf5' in trexiofile:
                print("Assuming TREXIO HDF5 format based on file suffix")
                charges, coords, labels = read_trexio_file(filename=trexiofile, back_end_type="hdf5")
            elif '.text' in trexiofile:
                print("Assuming TREXIO TEXT format based on file suffix")
                charges, coords, labels = read_trexio_file(filename=trexiofile, back_end_type="text")
            else:
                print("No recognized suffix in TREXIO file found. Exiting")
                ashexit()

            self.coords = np.array(coords)
            elems = [reformat_element(el) for el in labels]
            # Converting charges to elements
            self.elems = nucchargestoelems(charges)
            # NOTE: Labels not currently used

        # Reading an XYZ-file from the ASH database
        elif databasefile is not None:
            databasepath=ashpath+"/databases/fragments/"
            xyzfile=databasepath+databasefile
            if '.xyz' not in databasefile:
                xyzfile=databasepath+databasefile+'.xyz'
            print("Reading XYZ-file from database:", xyzfile)
            self.label = xyzfile.split('/')[-1].split('.')[0]
            try:
                self.read_xyzfile(xyzfile, readchargemult=True, conncalc=conncalc)
            except FileNotFoundError:
                print(f"XYZ-file {self.label}.xyz not found in database location: {databasepath}")
                print("Files in database location:", os.listdir(databasepath))
                exit()
        # If all else fails, exit
        else:
            ashexit(errormessage="Fragment requires some kind of valid coordinate input!")
        # Label for fragment (string). Useful for distinguishing different fragments
        # This overrides label-definitions above (self.label=xyzfile etc)
        if label is not None:
            self.label = label

        # Now set charge and mult attributes of fragment from keyword arg unless None. Will override readchargemult option above if used
        if charge != None:
            self.charge = charge
        if mult != None:
            self.mult = mult

        # Now update attributes after defining coordinates, getting charge, mult
        self.update_attributes()
        if conncalc is True:
            if len(self.connectivity) == 0:
                self.calc_connectivity(scale=scale, tol=tol)

        # Constraints attributes. Used by parallel surface-scan to pass constraints along.
        # Populated by calc_surface relaxed para
        self.constraints = None

    def __repr__(self):
        print("ASH Fragment object")
        print(f"Number of Atoms in fragment: {self.numatoms}")
        print(f"Formula: {self.prettyformula}")
        print(f"Label: {self.label}")
        print(f"Charge: {self.charge} Mult: {self.mult}")
        print("Do fragment.info() for more info on fragment")
        return "ASH fragment"
    def __str__(self):
        print("ASH Fragment object")
        print(f"Number of Atoms in fragment: {self.numatoms}")
        print(f"Formula: {self.prettyformula}")
        print(f"Label: {self.label}")
        print(f"Charge: {self.charge} Mult: {self.mult}")
        print("Do fragment.info() for more info on fragment")
        return "ASH fragment"
    def info(self):
        print("ASH Fragment object")
        print(self.__dict__)
    def update_attributes(self):
        if self.printlevel >= 2:
            print("Creating/Updating fragment attributes...")
        if len(self.coords) == 0:
            print("No coordinates in fragment. Something went wrong. Exiting.")
            ashexit()
        if type(self.coords) != np.ndarray:
            print("self.coords is not a numpy array. Something is wrong. Exiting.")
            ashexit()
        self.nuccharge = nucchargelist(self.elems)
        self.nuc_charges = elemstonuccharges(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(0, self.numatoms))
        # Unnecessary alias ? Todo: Delete
        self.allatoms = self.atomlist
        self.mass = totmasslist(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
        self.masses = self.list_of_masses
        # Elemental formula
        self.formula = elemlisttoformula(self.elems)
        # Pretty formula without 1 TODO
        self.prettyformula = self.formula
        # self.prettyformula = self.formula.replace('1','')
        # Update atomtypes, atomcharges and fragmenttype_labels also if needed
        if len(self.atomcharges) == 0:
            self.atomcharges = [0.0 for i in range(0, self.numatoms)]
        elif len(self.atomcharges) < self.numatoms:
            print("\nWARNING! atomcharges list shorter than number of atoms.")
            print("Adding 0.0 entries for missing atoms.")
            self.atomcharges = self.atomcharges + [0.0 for i in range(0, self.numatoms - len(self.atomcharges))]

        if len(self.fragmenttype_labels) == 0:
            self.fragmenttype_labels = ["None" for i in range(0, self.numatoms)]
        elif len(self.fragmenttype_labels) < self.numatoms:
            print("\nWARNING! fragmenttype_labels list shorter than number of atoms.")
            print("Adding 0 entries for missing atoms.")
            self.fragmenttype_labels = self.fragmenttype_labels + [0 for i in range(0, self.numatoms - len(
                self.fragmenttype_labels))]

        if len(self.atomtypes) == 0:
            self.atomtypes = ['None' for i in range(0, self.numatoms)]
        elif len(self.atomtypes) < self.numatoms:
            print("\nWARNING! atomtypes list shorter than number of atoms.")
            print("Adding None entries for missing atoms.")
            self.atomtypes = self.atomtypes + ['None' for i in range(0, self.numatoms - len(self.atomtypes))]

        if self.printlevel >= 2:
            print("Number of Atoms in fragment: {}\nFormula: {}\nLabel: {}".format(self.numatoms, self.prettyformula, self.label))
            print("Charge: {} Mult: {}".format(self.charge, self.mult))
            print_line_with_subheader1_end()
    # Add coordinates from geometry string. Will replace.
    def add_coords_from_string(self, coordsstring, scale=None, tol=None, conncalc=False):
        if self.printlevel >= 2:
            print("Getting coordinates from string:", coordsstring)
        if len(self.coords) > 0:
            if self.printlevel >= 2:
                print("Fragment already contains coordinates")
                print("Adding extra coordinates")
        coordslist = coordsstring.split('\n')
        tempcoords = []
        for line in coordslist:
            if len(line) > 5:
                self.elems.append(reformat_element(line.split()[0]))
                # Appending to numpy array
                clist = [float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]
                tempcoords.append(clist)
        # Converting list of lists to numpy array
        self.coords = reformat_list_to_array(tempcoords)
        self.label = ''.join(self.elems)
        #self.update_attributes()
        #if conncalc is True:
        #    self.calc_connectivity(scale=scale, tol=tol)
    def create_coords_from_smiles(self, smiles):
        print("Creating coordinates from SMILES string:", smiles)
        elems, coords = smiles_to_coords(smiles)
        self.elems = elems
        self.coords = reformat_list_to_array(coords)
        self.update_attributes()
    # Replace coordinates by providing elems and coords lists. Optional: recalculate connectivity
    def replace_coords(self, elems, coords, conn=False, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Replacing coordinates in fragment.")

        self.elems = elems
        # Adding coords as list of lists. Conversion to numpy array
        # np.array([list(i) for i in coords])
        self.coords = reformat_list_to_array(coords)
        self.update_attributes()
        if conn is True:
            self.calc_connectivity(scale=scale, tol=tol)

    # Get list of atom-indices for specific elements or groups
    # Atom indices except those provided
    def get_atomindices_except(self, excludelist):
        return listdiff(self.allatoms, excludelist)

    def get_nonH_atomindices(self):
        return [index for index, el in enumerate(self.elems) if el != 'H']

    def get_atomindices_for_element(self, element):
        return [index for index, el in enumerate(self.elems) if el == element]

    def get_atomindices_except_element(self, element):
        return [index for index, el in enumerate(self.elems) if el != element]

    # Get list of lists of bonds. Used for X-H constraints for example
    def get_XH_indices(self, conncode='julia'):
        timestamp = time.time()
        scale = ash.settings_ash.settings_dict["scale"]
        tol = ash.settings_ash.settings_dict["tol"]
        Hatoms = self.get_atomindices_for_element('H')
        # Hatoms=[1,2,3,5]
        #print("H Atoms: ", str(Hatoms).strip("[]"))

        # way too slow
        if conncode == 'py':
            final_list = []
            for Hatom in Hatoms:
                connatoms = get_connected_atoms_np(self.coords, self.elems, scale, tol, Hatom)
                final_list.append(connatoms)
            return final_list
        else:
            print("Loading Julia")
            try:
                Juliafunctions = load_julia_interface()
            except:
                print("Problem loading Julia")
                ashexit()
            final_list = Juliafunctions.get_connected_atoms_forlist_julia(self.coords, self.elems, scale, tol,
                                                                          eldict_covrad, Hatoms)
        #print("final_list: ", str(final_list).strip("[]"))
        print_time_rel(timestamp, modulename='get_XH_indices', moduleindex=4)
        return final_list
        # Call connectivity routines
        # for el in self.elems:
        #    if -

    def delete_atom(self, atomindex):
        self.coords = np.delete(self.coords, atomindex, axis=0)
        # Deleting from lists
        self.elems.pop(atomindex)
        self.atomcharges.pop(atomindex)
        self.atomtypes.pop(atomindex)
        self.fragmenttype_labels.pop(atomindex)

        # Updating other attributes
        self.update_attributes()

    # Appending coordinates. Taking list of lists but appending to np array
    def add_coords(self, elems, coords, conn=True, scale=None, tol=None):

        # TODO: Check if coords is list or list of lists before proceeding
        # TODO: if np array, check if dimensions are correect before proceeding
        if self.printlevel >= 2:
            print("Adding coordinates to fragment.")
        if len(self.coords) > 0:
            if self.printlevel >= 2:
                print("Fragment already contains coordinates.")
                print("Adding extra coordinates.")
        print(elems)
        # print(type(elems))
        self.elems = self.elems + list(elems)
        self.coords = np.append(self.coords, coords, axis=0)

        self.update_attributes()
        if conn is True:
            self.calc_connectivity(scale=scale, tol=tol)

    def print_coords(self):
        if self.printlevel >= 2:
            print("Defined coordinates (Å):")
        #print_coords_all(self.coords, self.elems)
        for i,(el, c) in enumerate(zip(self.elems, self.coords)):
            line = " {:<4} {:4} {:>12.6f} {:>12.6f} {:>12.6f}".format(i,el, c[0], c[1], c[2])
            print(line)

    def print_coords_for_atoms(self,members, labels=None):
        print_coords_for_atoms(self.coords, self.elems, members, labels=labels)


    # Read Amber coordinate file? Needs to read both INPCRD and PRMTOP file. Bit messy
    def read_amberfile(self, inpcrdfile=None, prmtopfile=None, conncalc=False):
        if self.printlevel >= 2:
            print("Reading coordinates from Amber INPCRD file: '{}' and PRMTOP file: '{}' into fragment.".format(inpcrdfile,
                                                                                                            prmtopfile))
        try:
            elems, coords, box_dims = read_ambercoordinates(prmtopfile=prmtopfile, inpcrdfile=inpcrdfile)
            # NOTE: boxdims not used. Could be set as fragment variable ?
        except FileNotFoundError:
            print("File {} or {} not found".format(prmtopfile, inpcrdfile))
            ashexit()
        self.coords = reformat_list_to_array(coords)
        self.elems = elems
        #self.update_attributes()
        #if conncalc is True:
        #    self.calc_connectivity(scale=scale, tol=tol)


    # Read GROMACS coordinates file
    def read_grofile(self, filename, conncalc=False, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Reading coordinates from Gromacs GRO file '{}' into fragment".format(filename))
        try:
            elems, coords, boxdims = read_gromacsfile(filename)
            # NOTE: boxdims not used. Could be set as fragment variable ?
        except FileNotFoundError:
            print("File '{}' not found".format(filename))
            ashexit()
        self.coords = coords
        self.elems = elems
        #self.update_attributes()
        #if conncalc is True:
        #    self.calc_connectivity(scale=scale, tol=tol)

    # Read CHARMM? coordinate file?
    def read_charmmfile(self, filename, conncalc=False):
        print("not implemented yet")
        ashexit()

    # Read Chemshell fragment file (.c ending)
    def read_chemshellfile(self, filename, conncalc=False, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Reading coordinates from Chemshell file '{}' into fragment.".format(filename))
        try:
            elems, coords = read_chemshellfragfile_xyz(filename)
        except FileNotFoundError:
            print("File '{}' not found.".format(filename))
            ashexit()
        self.coords = coords
        self.elems = elems
        #self.update_attributes()
        #if conncalc is True:
        #    self.calc_connectivity(scale=scale, tol=tol)
        #else:
        #    # Read connectivity list
        #    print("Note: Not reading connectivity from file.")

    # Read PDB file
    def read_pdbfile_old(self, filename, conncalc=True, scale=None, tol=None, use_atomnames_as_elements=False):
        if self.printlevel >= 2:
            print("Reading coordinates from PDB file '{}' into fragment.".format(filename))

        self.elems, self.coords = read_pdbfile(filename, use_atomnames_as_elements=use_atomnames_as_elements)
        print("Number of atoms in PDB file:", len(self.elems))
        # Also reading PDB residue/atom/segment information
        if self.printlevel >= 2:
            print("Reading atom/residue info from PDB file '{}' into fragment.".format(filename))
        self.pdb_atomnames, self.pdb_resnames, self.pdb_residlabels, self.pdb_chainlabels, self.pdb_conect_lines = read_pdbfile_info(filename)
        print("Number of atomnames found in PDB file:", len(self.pdb_atomnames))

        if len(self.coords) != len(self.pdb_atomnames):
            print("Warning: Number of coords found in PDB file does not match number of atomnames found.")

    def read_pdbfile_openmm(self,filename):
        if self.printlevel >= 2:
            print("read_pdbfile_openmm: Reading coordinates from PDB file '{}' into fragment.".format(filename))
        try:
            import openmm.app
        except ImportError:
            print("Error: OpenMM not found. Cannot read PDB file.")
            ashexit()
        pdb = openmm.app.PDBFile(filename)
        self.coords = np.array([[i.x*10,i.y*10,i.z*10] for i in pdb.positions])
        self.elems = [atom.element.symbol for atom in pdb.topology.atoms()]

        # Topology
        self.pdb_topology = pdb.topology

    def read_xyzfile(self, filename, scale=None, tol=None, readchargemult=False, conncalc=True):
        if self.printlevel >= 2:
            print("Reading coordinates from XYZ file '{}' into fragment.".format(filename))
        coords = []
        with open(filename) as f:
            for count, line in enumerate(f):
                if count == 0:
                    self.numatoms = int(line.split()[0])
                elif count == 1:
                    if readchargemult is True:
                        if self.printlevel >= 2:
                            print("Reading charge/mult from file header.")
                        try:
                            self.charge = int(line.split()[0])
                            self.mult = int(line.split()[1])
                        except ValueError:
                            print(f"Error: XYZ-file {filename} does not have a valid charge/mult in 2nd-line of header:")
                            print("Line:", line)
                            ashexit()
                elif count > 1:
                    if len(line) > 3:
                        # Grabbing element and reformatting
                        if isint(line.split()[0]) is True:
                            # Grabbing element as atomnumber and reformatting
                            # el=dictionaries_lists.element_dict_atnum[int(line.split()[0])].symbol
                            el = reformat_element(int(line.split()[0]), isatomnum=True)
                            self.elems.append(el)
                        else:
                            el = line.split()[0]
                            self.elems.append(reformat_element(el))
                        # self.coords = np.append(self.coords,[float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
                        coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        # Convert to numpy
        self.coords = reformat_list_to_array(coords)
        if self.numatoms != len(self.coords):
            print("Number of atoms in header not equal to number of coordinate-lines. Check XYZ file!")
            ashexit()


    def set_energy(self, energy):
        self.energy = float(energy)

    def get_coordinate_center(self):
        center_x = np.mean(self.coords[:, 0])
        center_y = np.mean(self.coords[:, 1])
        center_z = np.mean(self.coords[:, 2])
        return [center_x, center_y, center_z]

    # Get coordinates for specific atoms (from list of atom indices)
    # NOTE: This also returns elements, bit silly
    def get_coords_for_atoms(self, atoms):
        subcoords = np.take(self.coords, atoms, axis=0)
        subelems = [self.elems[i] for i in atoms]
        return subcoords, subelems

    # Calculate connectivity (list of lists) of coords
    def calc_connectivity(self, conndepth=99, scale=None, tol=None, codeversion=None):
        print("Calculating connectivity.")
        # If codeversion not requested we go to default
        if codeversion is None:
            codeversion = ash.settings_ash.settings_dict["connectivity_code"]
            print("Codeversion not set. Using default setting: ", codeversion)

        # Overriding with py version if molecule is small. Faster than calling julia.
        if len(self.coords) < 1000:
            print(f"Small system ({len(self.coords)} atoms). Using py version.")
            codeversion = 'py'
        elif len(self.coords) > 10000:
            if self.printlevel >= 2:
                print("Atom number > 10K. Connectivity calculation could take a while")

        if scale is None:
            try:
                scale = ash.settings_ash.settings_dict["scale"]
                tol = ash.settings_ash.settings_dict["tol"]
                if self.printlevel >= 2:
                    print("Using global scale and tol parameters from settings_ash. Scale: {}, Tol: {} ".format(scale,
                                                                                                               tol))

            except:
                scale = 1.0
                tol = 0.1
                if self.printlevel >= 2:
                    print("Exception: Using hard-coded scale and tol parameters. Scale: {} Tol: {} ".format(scale, tol))
        else:
            if self.printlevel >= 2:
                print("Using scale: {} and tol: {} ".format(scale, tol))

        # Setting scale and tol as part of object for future usage (e.g. QM/MM link atoms)
        self.scale = scale
        self.tol = tol

        # Calculate connectivity by looping over all atoms
        timestampA = time.time()
        print("codeversion:", codeversion)
        if codeversion == 'py':
            print("Calculating connectivity of fragment using py.")
            fraglist = calc_conn_py(self.coords, self.elems, conndepth, scale, tol)
            print_time_rel(timestampA, modulename='calc connectivity py', moduleindex=4)
        elif codeversion == 'julia':
            print("Calculating connectivity of fragment using Julia.")
            try:
                Juliafunctions = load_julia_interface()
                fraglist_temp = Juliafunctions.calc_connectivity(self.coords, self.elems, conndepth, scale, tol,
                                                                 eldict_covrad)
                fraglist = []
                # Converting from numpy to list of lists
                for sublist in fraglist_temp:
                    fraglist.append(list(sublist))
                print_time_rel(timestampA, modulename='calc connectivity julia', moduleindex=4)
            except:
                print(BC.FAIL, "Problem importing Python-Julia interface.", BC.END)
                print("Make sure Julia is installed and Python-Julia interface has been set up.")
                print(BC.FAIL, "Using Python version instead (slow for large systems)", BC.END)
                # Switching default to py since Julia did not load
                ash.settings_ash.settings_dict["connectivity_code"] = "py"
                fraglist = calc_conn_py(self.coords, self.elems, conndepth, scale, tol)
                print_time_rel(timestampA, modulename='calc connectivity py', moduleindex=4)
        self.connectivity = fraglist
        # Calculate number of atoms in connectivity list of lists
        conn_number_sum = 0
        for l in self.connectivity:
            conn_number_sum += len(l)
        if self.numatoms != conn_number_sum:
            print(BC.FAIL, "Connectivity problem", BC.END)
            print("self.connectivity:", self.connectivity)
            print("conn_number_sum:", conn_number_sum)
            print("self numatoms", self.numatoms)
            ashexit()
        self.connected_atoms_number = conn_number_sum

    def update_atomcharges(self, charges):
        self.atomcharges = charges

    def update_atomtypes(self, types):
        self.atomtypes = types

    # Adding fragment-type info (used by molcrys, identifies whether atom is mainfrag, counterfrag1 etc.)
    # Old slow version below. To be deleted
    def old_add_fragment_type_info(self, fragmentobjects):
        # Create list of fragment-type label-list
        self.fragmenttype_labels = []
        for i in self.atomlist:
            for count, fobject in enumerate(fragmentobjects):
                if i in fobject.flat_clusterfraglist:
                    self.fragmenttype_labels.append(count)

    # Adding fragment-type info (used by molcrys, identifies whether atom is mainfrag, counterfrag1 etc.)
    # This one is fast
    def add_fragment_type_info(self, fragmentobjects):
        print("fragmentobjects:", fragmentobjects)
        # Create list of fragment-type label-list
        combined_flat_clusterfraglist = []
        combined_flat_labels = []
        # Going through objects, getting flat atomlists for each object and combine (combined_flat_clusterfraglist)
        # Also create list of labels (using fragindex) for each atom
        self.fragmenttypes_numatoms = []
        for fragindex, frago in enumerate(fragmentobjects):
            print("fragindex:", fragindex)
            #print("frago:", frago)
            #print(frago.__dict__)
            #print("frago.flat_clusterfraglist:", frago.flat_clusterfraglist)
            combined_flat_clusterfraglist.extend(frago.flat_clusterfraglist)
            combined_flat_labels.extend([fragindex] * len(frago.flat_clusterfraglist))
            self.fragmenttypes_numatoms.append([frago.Numatoms])
        self.fragmenttypes = len(fragmentobjects)

        # Getting indices required to sort atomindices in ascending order
        sortindices = np.argsort(combined_flat_clusterfraglist)
        # labellist contains unsorted list of labels
        # Now ordering the labels according to the sort indices
        self.fragmenttype_labels = [combined_flat_labels[i] for i in sortindices]

    # Centroid
    def get_centroid(self):
        return np.mean(self.coords, axis=0)

    # Molcrys option:
    def add_centralfraginfo(self, list):
        self.Centralmainfrag = list

    # Write PDB-file
    def write_pdbfile(self,filename="Fragment"):
        print("Fragment.write_pdbfile method called")
        filename=filename.replace(".pdb","")
        # Write PDB-file if information is available
        if self.pdb_atomnames is not None:
            print("Found PDB residue/atom/segment information stored in fragment. Writing proper PDB file.")
        else:
            print("Warning: No PDB residue/atom/segment information available (only available if Fragment was created from a PDB-file).")
            print("Will write PDB file with basic default residue/atom/segment names.")
        write_pdbfile(self, outputname=filename, atomnames=self.pdb_atomnames, chainlabels=self.pdb_chainlabels,
                      resnames=self.pdb_resnames,residlabels=self.pdb_residlabels, segmentlabels=None, conect_lines=self.pdb_conect_lines)
        return f"{filename}.pdb"

    # Create new topology from scratch if none is defined (defined automatically when reading PDB-files by OpenMM)
    def define_topology(self, scale=1.0, tol=0.1, resname="MOL"):
        try:
            import openmm.app
        except ImportError:
            print("Error: OpenMM not found. Cannot define a topology")
            ashexit()
        print("Defining new basic single-chain, multi-residue topology")
        self.pdb_topology = openmm.app.Topology()
        chain = self.pdb_topology.addChain()

        # Create connectivity by default for new topology
        if self.connectivity is None:
            self.calc_connectivity(scale=scale, tol=tol)
        elif isinstance(self.connectivity, list):
            if len(self.connectivity) == 0:
                self.calc_connectivity(scale=scale, tol=tol)

        connectivity_dict = get_connected_atoms_dict(self.coords,self.elems, scale,tol)
        #Looping over molecules defined by connectivity
        for mol in self.connectivity:
            residue = self.pdb_topology.addResidue(resname, chain)

            # Defaultdictionary to keep track of unique element-atomnames
            atomnames_dict=defaultdict(int)
            for at in mol:
                el = self.elems[at]
                atomnumber = openmm.app.Element.getBySymbol(el).atomic_number
                element = openmm.app.Element.getByAtomicNumber(atomnumber)
                # Define unique atomname
                atomnames_dict[el] += 1
                atomname = f"{el}{atomnames_dict[el]}"

                # Special handling for obvious water residues. Aids OpenMM recognition
                if atomname == "O1" and len(mol) == 3:
                    #print("atomname is O1 and 3-atom residue. Probably water")
                    #print("using atomname as O instead of O1 aids OpenMM recognition")
                    atomname="O"

                self.pdb_topology.addAtom(atomname, element, residue)

        print("Adding connectivity to PDB topology")
        ash.interfaces.interface_OpenMM.openmm_add_bonds_to_topology(self.pdb_topology, connectivity_dict)

    # Write PDB-file via OpenMM
    def write_pdbfile_openmm(self,filename="Fragment", calc_connectivity=False, pdb_topology=None,
                             skip_connectivity=False, resname="MOL"):
        print("write_pdbfile_openmm\n")
        try:
            import openmm.app
        except ImportError:
            print("Error: OpenMM not found. Cannot read PDB file.")
            ashexit()

        #Adding extension
        if '.pdb' not in filename:
            filename += ".pdb"

        if pdb_topology is not None:
            print("Using input pdb_topology")
            self.pdb_topology = pdb_topology
        elif self.pdb_topology is None:
            print("Warning: ASH Fragment has no PDB-file topology defined (required for PDB-file writing)")
            print("Now defining new topology from scratch")
            if pdb_topology is None:
                self.define_topology(resname=resname) #Creates self.pdb_topology
        else:
            print("Using pdbtopology found in ASH fragment")

        # Before writing PDB-file, request connectivity calculation so that we get correct CONECT lines for non-biomolecules
        if calc_connectivity is True:
            print("Connectivity calculation requested for Fragment")
            connectivity_dict = get_connected_atoms_dict(self.coords,self.elems, 1.0,0.1)
            print("Adding connectivity to PDB topology")
            ash.interfaces.interface_OpenMM.openmm_add_bonds_to_topology(self.pdb_topology,connectivity_dict)

        # If no_connectivity is True, we skip adding connectivity to PDB-file
        if skip_connectivity is True:
            print("skip_connectivity True: this will not write connectivity lines to PDB-file")
            # solute_resname= list(self.pdb_topology.residues())[0].name
            print("Deleting molecule bond information")
            # Setting list of bonds to empty list
            self.pdb_topology._bonds=[]
        openmm.app.PDBFile.writeFile(self.pdb_topology, self.coords, file=open(f"{filename}", 'w'))
        print(f"Wrote PDB-file: {filename}")

    def write_xyzfile(self, xyzfilename="Fragment-xyzfile.xyz", writemode='w', write_chargemult=True, write_energy=True):

        with open(xyzfilename, writemode) as ofile:
            ofile.write(str(len(self.elems)) + '\n')

            #Title line
            #Write charge,mult and energy by default. Will be None if not available
            if write_chargemult is True and write_energy is True:
                ofile.write("{} {} {}\n".format(self.charge,self.mult,self.energy))
            else:
                ofile.write("title\n")
            #elif write_chargemult is True and write_energy is True:
            #    ofile.write("{} {}\n".format(self.charge,self.mult))
            # Energy written otherwise
            #else:
            #    if self.energy is None:
            #        ofile.write("Energy: None" + '\n')
            #    else:
            #        ofile.write("Energy: {:14.8f}".format(self.energy) + '\n')

            #Coordinates
            for el, c in zip(self.elems, self.coords):
                line = "{:4} {:14.8f} {:14.8f} {:14.8f}".format(el, c[0], c[1], c[2])
                ofile.write(line + '\n')
        if self.printlevel >= 2:
            print("Wrote XYZ file: ", xyzfilename)
        return xyzfilename
    def write_XYZ_for_atoms(self,xyzfilename="Fragment-subset.xyz", atoms=None):
        subset_elems = [self.elems[i] for i in atoms]
        subset_coords = np.take(self.coords, atoms, axis=0)
        with open(xyzfilename, 'w') as ofile:
            ofile.write(str(len(subset_elems)) + '\n')
            ofile.write("title" + '\n')
            for el, c in zip(subset_elems, subset_coords):
                line = "{:4} {:>12.6f} {:>12.6f} {:>12.6f}".format(el, c[0], c[1], c[2])
                ofile.write(line + '\n')

    def write_trexio(self,filename=None,format="text"):
        from ash.interfaces.interface_TREXIO import write_trexio_file
        write_trexio_file(self, filename=filename, back_end_type=format)

    # Function to get subset-coordinates with linkatoms
    #TODO: add more options for linkatoms
    def get_subset_coords_with_linkatoms(self,qmatoms):
        conn_scale = ash.settings_ash.settings_dict["scale"]
        conn_tolerance = ash.settings_ash.settings_dict["tol"]+0.2
        boundaryatoms = ash.modules.module_coords.get_boundary_atoms(qmatoms, self.coords, self.elems, conn_scale,
                    conn_tolerance)
        # Get linkatom coordinates
        linkatoms_dict = ash.modules.module_coords.get_linkatom_positions(boundaryatoms,qmatoms, self.coords, self.elems)
        linkatoms_coords = [linkatoms_dict[pair] for pair in sorted(linkatoms_dict.keys())]
        qm_elems=[self.elems[i] for i in qmatoms]
        qm_coords_with_linkatoms = np.concatenate((np.take(self.coords,qmatoms,axis=0), linkatoms_coords), axis=0)
        qm_elems_with_linkatoms = qm_elems+ ['H' for i in linkatoms_coords]

        write_xyzfile(qm_elems_with_linkatoms, qm_coords_with_linkatoms, "qm_region_with_linkatoms")
        return qm_coords_with_linkatoms, qm_elems_with_linkatoms

    # Print system-fragment information to file. Default name of file: "fragment.ygg
    def print_system(self, filename='fragment.ygg'):
        if self.printlevel >= 2:
            print("Printing fragment to disk: ", filename)

        # Checking that lists have same length (as zip will just ignore the mismatch)
        # print("len(self.atomlist)", len(self.atomlist))
        # rint("len(self.elems)",len(self.elems) )
        # print("len(self.coords)",len(self.coords) )
        # print("len(self.atomcharges)", len(self.atomcharges) )
        # print("len(self.fragmenttype_labels)", len(self.fragmenttype_labels) )
        # print("len(self.atomtypes)", len(self.atomtypes))

        print("", )
        printdebug("len(self.atomlist): ", len(self.atomlist))
        printdebug("len(self.elems): ", len(self.elems))
        printdebug("len(self.coords): ", len(self.coords))
        printdebug("len(self.atomcharges): ", len(self.atomcharges))
        printdebug("len(self.fragmenttype_labels): ", len(self.fragmenttype_labels))
        printdebug("len(self.atomtypes): ", len(self.atomtypes))

        if (len(self.atomlist) == len(self.elems) == len(self.coords) == len(self.atomcharges) == len(self.fragmenttype_labels) == len(self.atomtypes)) is False:
            print(BC.FAIL,"Error. Missing entries in list.")
            print("Len atomlist:", len(self.atomlist))
            print("Len elems:", len(self.elems))
            print("Len coords:", len(self.coords))
            print("Len atomcharges:", len(self.atomcharges))
            print("Len atomtypes:", len(self.atomtypes))
            print("Len fragmenttype_labels:", len(self.fragmenttype_labels))
            print("fragmenttype_labels:", self.fragmenttype_labels)
            print("This should not have happened. File a bugreport", BC.END)
            ashexit()
        with open(filename, 'w') as outfile:
            outfile.write("Fragment: \n")
            outfile.write("Num atoms: {}\n".format(self.numatoms))
            outfile.write("Formula: {}\n".format(self.formula))
            outfile.write("Energy: {}\n".format(self.energy))
            if self.charge != None:
                outfile.write("charge : {}\n".format(self.charge))
            if self.mult != None:
                outfile.write("mult : {}\n".format(self.mult))
            outfile.write("\n")
            outfile.write(
                " Index    Atom         x                  y                  z               charge        fragment-type        atom-type\n")
            outfile.write(
                "---------------------------------------------------------------------------------------------------------------------------------\n")
            for at, el, coord, charge, label, atomtype in zip(self.atomlist, self.elems, self.coords, self.atomcharges,
                                                              self.fragmenttype_labels, self.atomtypes):
                label = str(label)
                line = "{:>6} {:>6}  {:17.11f}  {:17.11f}  {:17.11f}  {:14.8f} {:12s} {:>21}\n".format(at, el, coord[0],
                                                                                                       coord[1],
                                                                                                       coord[2], charge,
                                                                                                       label, atomtype)
                outfile.write(line)
            outfile.write(
                "===========================================================================================================================================\n")
            # outfile.write("elems: {}\n".format(self.elems))
            # outfile.write("coords: {}\n".format(self.coords))
            # outfile.write("list of masses: {}\n".format(self.list_of_masses))
            outfile.write("atomcharges: {}\n".format(self.atomcharges))
            outfile.write("Sum of atomcharges: {}\n".format(sum(self.atomcharges)))
            outfile.write("atomtypes: {}\n".format(self.atomtypes))
            outfile.write("connectivity: {}\n".format(self.connectivity))
            outfile.write("Centralmainfrag: {}\n".format(self.Centralmainfrag))

    # Reading fragment from file. File created from Fragment.print_system
    def read_fragment_from_file(self, fragfile):
        if self.printlevel >= 2:
            print("Reading ASH fragment from file:", fragfile)
        coordgrab = False
        coords = []
        elems = []
        atomcharges = []
        atomtypes = []
        fragment_type_labels = []
        connectivity = []
        # Only used by molcrys:
        Centralmainfrag = []
        with open(fragfile) as file:
            for n, line in enumerate(file):
                if n == 0:
                    if 'Fragment:' not in line:
                        print("This is not a valid ASH fragment file. Exiting.")
                        ashexit()
                if 'Num atoms:' in line:
                    numatoms = int(line.split()[-1])
                if 'charge :' in line:
                    self.charge=int(line.split()[-1])
                if 'mult :' in line:
                    self.mult=int(line.split()[-1])
                if coordgrab is True:
                    # If end of coords section
                    if '===============' in line:
                        coordgrab = False
                        continue
                    elems.append(line.split()[1])
                    coords.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                    atomcharges.append(float(line.split()[5]))
                    # Reading and converting to integer.
                    if line.split()[6] == 'None':
                        ftypelabel = 'None'
                    else:
                        ftypelabel = int(line.split()[6])
                    fragment_type_labels.append(ftypelabel)
                    atomtypes.append(line.split()[7])

                if '--------------------------' in line:
                    coordgrab = True
                if 'Centralmainfrag' in line:
                    if '[]' not in line:
                        l = line.lstrip('Centralmainfrag:')
                        l = l.replace('\n', '')
                        l = l.replace(' ', '')
                        l = l.replace('[', '')
                        l = l.replace(']', '')
                        Centralmainfrag = [int(i) for i in l.split(',')]
                # Incredibly ugly but oh well
                if 'connectivity:' in line:
                    l = line.lstrip('connectivity:')
                    l = l.replace(" ", "")
                    for x in l.split(']'):
                        if len(x) < 1:
                            break
                        y = x.strip(',[')
                        y = y.strip('[')
                        y = y.strip(']')
                        try:
                            connlist = [int(i) for i in y.split(',')]
                        except:
                            connlist = []
                        connectivity.append(connlist)
        self.elems = elems
        # Converting to numpy array
        self.coords = np.array(coords)
        self.atomcharges = atomcharges
        self.atomtypes = atomtypes
        self.fragmenttype_labels = fragment_type_labels
        self.update_attributes()
        self.connectivity = connectivity
        self.Centralmainfrag = Centralmainfrag


def reformat_list_to_array(l):
    # If np array already
    if type(l) == np.ndarray:
        return l
    # Reformat to np array
    elif type(l) == list:
        #Checking if input l is list of lists or not
        if any(isinstance(el, list) for el in l) is False:
            print(BC.FAIL,"Error (reformat_list_to_array): input should be a list of lists, not just a list", BC.END)
            ashexit()
        newl = np.array(l)
        return newl


# TODO: Reorganize and move to dictionaries_lists ?
# Elements and atom numbers
# elements=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
# Added M-site dummy atom
elematomnumbers = {'m': 0, 'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9, 'ne': 10,
                   'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19, 'ca': 20,
                   'sc': 21, 'ti': 22, 'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30,
                   'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39, 'zr': 40,
                   'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48, 'in': 49, 'sn': 50,
                   'sb': 51, 'te': 52, 'i': 53, 'xe': 54, 'cs': 55, 'ba': 56, 'la': 57, 'ce': 58, 'pr': 59, 'nd': 60,
                   'pm': 61, 'sm': 62, 'eu': 63, 'gd': 64, 'tb': 65, 'dy': 66, 'ho': 67, 'er': 68, 'tm': 69, 'yb': 70,
                   'lu': 71, 'hf': 72, 'ta': 73, 'w': 74, 're': 75, 'os': 76, 'ir': 77, 'pt': 78, 'au': 79, 'hg': 80,
                   'tl': 81, 'pb': 82, 'bi': 83, 'po': 84, 'at': 85, 'rn': 86, 'fr': 87, 'ra': 88, 'ac': 89, 'th': 90,
                   'pa': 91, 'u': 92, 'np': 93, 'pu': 94, 'am': 95, 'cm': 96, 'bk': 97, 'cf': 98, 'es': 99, 'fm': 100,
                   'md': 101, 'no': 102, 'lr': 103, 'rf': 104, 'db': 105, 'sg': 106, 'bh': 107, 'hs': 108, 'mt': 109,
                   'ds': 110, 'rg': 111, 'cn': 112, 'nh': 113, 'fl': 114, 'mc': 115, 'lv': 116, 'ts': 117, 'og': 118}

# Atom masses
atommasses = [1.00794, 4.002602, 6.94, 9.0121831, 10.81, 12.01070, 14.00670, 15.99940, 18.99840316, 20.1797,
              22.98976928, 24.305, 26.9815385, 28.085, 30.973762, 32.065, 35.45, 39.948, 39.0983, 40.078, 44.955908,
              47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.921595,
              78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.96, 97, 101.07, 102.9055, 106.42,
              107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.905452, 137.327, 138.90547,
              140.116, 140.90766, 144.242, 145, 150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259,
              168.93422, 173.054, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569,
              200.592, 204.38, 207.2, 208.9804, 209, 210, 222, 223, 226, 227, 232.0377, 231.03588, 238.02891, 237, 244,
              243, 247, 247, 251, 252, 257, 258, 259, 262]
# Covalent radii for elements (Alvarez) in Angstrom.
# Used for connectivity
# Added dummy atom, M
eldict_covrad = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
                 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
                 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.61,
                 'Fe': 1.52, 'Co': 1.50, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19,
                 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64,
                 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42,
                 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07,
                 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94,
                 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70,
                 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45,
                 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50, 'U': 1.96}
# Modified radii for certain elements like Na, K
eldict_covrad['Na'] = 0.0001
eldict_covrad['K'] = 0.0001
# Dummy atom M. For example the M-site on TIP4P model
eldict_covrad['M'] = 0.0


# Function to reformat element string to be correct('cu' or 'CU' become 'Cu')
# Can also convert atomic-number (isatomnum flag)
def reformat_element(elem, isatomnum=False):
    if isatomnum is True:
        el_correct = ash.dictionaries_lists.element_dict_atnum[elem].symbol
    else:
        try:
            el_correct = ash.dictionaries_lists.element_dict_atname[elem.lower()].symbol
        except KeyError:
            print("Element-string: {} not found in element-dictionary!".format(elem))
            print("This is not a valid element as defined in ASH source-file: dictionaries_lists.py")
            print("Fix element-information in coordinate-file.")
            ashexit()
    return el_correct


# Remove zero charges
def remove_zero_charges(charges, coords):
    newcharges = []
    newcoords = []
    if len(charges) != len(coords):
        print(BC.FAIL,"Something went wrong in remove_zero_charges. File a bug report", BC.END)
        ashexit()
    for charge, coord in zip(charges, coords):
        if charge != 0.0:
            newcharges.append(charge)
            newcoords.append(coord)
    return newcharges, newcoords


def print_internal_coordinate_table(fragment, actatoms=None):
    timeA = time.time()
    print("\nPrinting internal coordinate table")
    if actatoms != None:
        print("Actatoms:", actatoms)

    #If no actatoms
    if actatoms is None:
        actatoms = []
        chosen_coords = fragment.coords
        chosen_elems = fragment.elems

    #NOTE: Changing so that we calculate connectivity always regardless of availability.
    # If no connectivity in fragment then recalculate it for actatoms only
    #if len(fragment.connectivity) == 0:
    print("Connectivity needs to be calculated")

    if len(actatoms) > 0:
        chosen_coords = np.take(fragment.coords, actatoms, axis=0)
        chosen_elems = [fragment.elems[i] for i in actatoms]
    else:
        chosen_coords = fragment.coords
        chosen_elems = fragment.elems

    conndepth = 99
    scale = ash.settings_ash.settings_dict["scale"]
    tol = ash.settings_ash.settings_dict["tol"]

    if len(chosen_coords) > 1000:
        try:
            Juliafunctions = load_julia_interface()
            connectivity = Juliafunctions.calc_connectivity(chosen_coords, chosen_elems, conndepth, scale, tol,
                                                            eldict_covrad)
        except:
            print("Problem importing Python-Julia interface. Trying py-version instead.")
            connectivity = calc_conn_py(chosen_coords, chosen_elems, conndepth, scale, tol)
    else:
        #PyTHON connectivity
        connectivity = calc_conn_py(chosen_coords, chosen_elems, conndepth, scale, tol)
    print("Connectivity calculation complete.")
    #else:
    #    print("Using precalculated connectivity")
    #    connectivity = fragment.connectivity
    #    chosen_coords = fragment.coords
    #    chosen_elems = fragment.elems

    # Looping over connected fragments
    bondpairsdict = {}

    for conn_fragment in connectivity:
        # Looping over atom indices in fragment
        for atom in conn_fragment:
            connatoms = get_connected_atoms(chosen_coords, chosen_elems, ash.settings_ash.settings_dict["scale"],
                                            ash.settings_ash.settings_dict["tol"], atom)
            for conn_i in connatoms:
                #dist = distance_between_atoms(fragment=fragment, atom1=atom, atom2=conn_i)
                dist = distance(chosen_coords[atom], chosen_coords[conn_i])
                # bondpairs.append([atom,conn_i,dist])
                bondpairsdict[frozenset((atom, conn_i))] = dist

    print_line_with_subheader2("Internal coordinates")

    # Using frozenset: https://stackoverflow.com/questions/46633065/multiples-keys-dictionary-where-key-order-doesnt-matter
    print_line_with_subheader2("Bond lengths (Å):")
    for key, val in bondpairsdict.items():
        listkey = list(key)
        elA = chosen_elems[listkey[0]]
        elB = chosen_elems[listkey[1]]
        # Only print bond lengths if both atoms in actatoms list
        if not actatoms:

                print("Bond: {:8}{:4} - {:4}{:4} {:>6.3f}".format(listkey[0], elA, listkey[1], elB, val))
        else:
            #converting to full-system indices
            fullsystem_keyA=actatoms[listkey[0]]
            fullsystem_keyB=actatoms[listkey[1]]
            if fullsystem_keyA in actatoms and fullsystem_keyB in actatoms:
                print("Bond: {:8}{:4} - {:4}{:4} {:>6.3f}".format(fullsystem_keyA, elA, fullsystem_keyB, elB, val))
    print('=' * 50)
    print_time_rel(timeA, modulename='print internal coordinate table')


# Function to check if string corresponds to an element symbol or not.
# Compares in lowercase
def isElement(string):
    if string.lower() in elematomnumbers:
        return True
    else:
        return False


# Checks if list of string is list of elements or no
def isElementList(list):
    for l in list:
        if not isElement(l):
            return False
    return True


# From lists of coords,elems and atom indices, print coords with elem
def print_coords_for_atoms(coords, elems, members, labels=None):
    if labels is not None:
        if len(labels) != len(members):
            print("Problem. Length of Labels note equal to length of members list")
            ashexit()
    label=""
    for i,m in enumerate(members):
        if labels is not None:
            label=labels[i]
        print("{:>4} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}".format(label,elems[m], coords[m][0], coords[m][1], coords[m][2]))


# From lists of coords,elems and atom indices, write XYZ file coords with elem

def write_XYZ_for_atoms(coords, elems, members, name):
    subset_elems = [elems[i] for i in members]
    # subset_coords=[coords[i] for i in members]
    subset_coords = np.take(coords, members, axis=0)
    with open(name + '.xyz', 'w') as ofile:
        ofile.write(str(len(subset_elems)) + '\n')
        ofile.write("title" + '\n')
        for el, c in zip(subset_elems, subset_coords):
            line = "{:4} {:>12.6f} {:>12.6f} {:>12.6f}".format(el, c[0], c[1], c[2])
            ofile.write(line + '\n')

#Write a multi-XYZ-file, i.e. XYZ trajectory from a list with each sublist containing list of elements and np array of coords
#el_and_coords : [[['O','H','H'],np.array([[0.0, 0.0, 0.0],[0.0,0.0,1.0],[0.0,0.0,-1.0]])],etc.]
def write_multi_xyz_file(el_and_coords,numatoms,filename,label=""):
    with open(filename,"w") as f:
        for coord in el_and_coords:
            f.write(f"{numatoms}\n")
            f.write(f"{label}\n")
            for el,co in zip(coord[0],coord[1]):
                f.write(f"{el} {co[0]} {co[1]} {co[2]}\n")

# From lists of coords,elems and atom indices, print coords with elems
# If list of atom indices provided, print as leftmost column
# If list of labels provided, print as rightmost column
# If list of labels2 provided, print as rightmost column
def print_coords_all(coords, elems, indices=None, labels=None, labels2=None):
    if indices is None:
        if labels is None:
            for i in range(len(elems)):
                print(
                    "{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}".format(elems[i], coords[i][0], coords[i][1], coords[i][2]))
        else:
            if labels2 is None:
                for i in range(len(elems)):
                    print("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6}".format(elems[i], coords[i][0], coords[i][1],
                                                                               coords[i][2], labels[i]))
            else:
                for i in range(len(elems)):
                    print("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6} :>6".format(elems[i], coords[i][0], coords[i][1],
                                                                                   coords[i][2], labels[i], labels2[i]))
    else:
        if labels is None:
            for i in range(len(elems)):
                print("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}".format(indices[i], elems[i], coords[i][0],
                                                                           coords[i][1], coords[i][2]))
        else:
            if labels2 is None:
                for i in range(len(elems)):
                    print("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6}".format(indices[i], elems[i], coords[i][0],
                                                                                     coords[i][1], coords[i][2],
                                                                                     labels[i]))
            else:
                for i in range(len(elems)):
                    print("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6} {:>6}".format(indices[i], elems[i],
                                                                                           coords[i][0], coords[i][1],
                                                                                           coords[i][2], labels[i],
                                                                                           labels2[i]))


# From lists of coords,elems and atom indices, print coords with elems
# If list of atom indices provided, print as leftmost column
# If list of labels provided, print as rightmost column
# If list of labels2 provided, print as rightmost column
def write_coords_all(coords, elems, indices=None, labels=None, labels2=None, file="file", description="description"):
    f = open(file, "w")
    f.write("#{}\n".format(description))
    if indices is None:
        if labels is None:
            for i in range(len(elems)):
                f.write("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}\n".format(elems[i], coords[i][0], coords[i][1],
                                                                         coords[i][2]))

        else:
            if labels2 is None:
                for i in range(len(elems)):
                    f.write("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6}\n".format(elems[i], coords[i][0], coords[i][1],
                                                                                   coords[i][2], labels[i]))
            else:
                for i in range(len(elems)):
                    f.write(
                        "{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6} :>6\n".format(elems[i], coords[i][0], coords[i][1],
                                                                                   coords[i][2], labels[i], labels2[i]))
    else:
        if labels is None:
            for i in range(len(elems)):
                f.write("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}\n".format(indices[i], elems[i], coords[i][0],
                                                                               coords[i][1], coords[i][2]))
        else:
            if labels2 is None:
                for i in range(len(elems)):
                    f.write(
                        "{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6}\n".format(indices[i], elems[i], coords[i][0],
                                                                                     coords[i][1], coords[i][2],
                                                                                     labels[i]))
            else:
                for i in range(len(elems)):
                    f.write("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6} {:>6}\n".format(indices[i], elems[i],
                                                                                               coords[i][0],
                                                                                               coords[i][1],
                                                                                               coords[i][2], labels[i],
                                                                                               labels2[i]))

    f.close()


# Functions to get distance, angle, coordinates of fragment
def distance(A, B):
    return sqrt(pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2) + pow(A[2] - B[2], 2))  # fastest
    # return sum((v_i - u_i) ** 2 for v_i, u_i in zip(A, B)) ** 0.5 #slow
    # return np.sqrt(np.sum((A - B) ** 2)) #very slow
    # return np.linalg.norm(A - B) #VERY slow
    # return sqrt(sum((px - qx) ** 2.0 for px, qx in zip(A, B))) #slow
    # return sqrt(sum([pow((a - b),2) for a, b in zip(A, B)])) #OK
    # return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2) #Very slow
    # return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2) #faster
    # return math.sqrt(math.pow(A[0] - B[0],2) + math.pow(A[1] - B[1],2) + math.pow(A[2] - B[2],2)) #faster
    # return sqrt(sum((A-B)**2)) #slow
    # return sqrt(sum(pow((A - B),2))) does not work
    # return np.sqrt(np.power((A-B),2).sum()) #very slow
    # return sqrt(np.power((A - B), 2).sum())
    # return np.sum((A - B) ** 2)**0.5 #very slow

def angle(A, B, C):
    AB = A - B
    CB = C - B
    dot_product = np.dot(AB, CB)
    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(AB)
    magnitude2 = np.linalg.norm(CB)
    # Calculate the angle in radians
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def dihedral(A, B, C, D):
    # Calculate the vectors between adjacent atoms
    v1 = B - A
    v2 = C - B
    v3 = D - C

    # Calculate the cross products
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # Calculate the dot product
    dot = np.dot(n1, n2)
    # Handle signs correctly
    if dot < 0:
        dihedral_angle = -1 * (np.arccos(dot / (np.linalg.norm(n1) * np.linalg.norm(n2))))
    else:
        dihedral_angle = np.arccos(dot / (np.linalg.norm(n1) * np.linalg.norm(n2)))

    # Convert from radians to degrees
    dihedral_angle = dihedral_angle * 180 / np.pi
    return dihedral_angle

#User-functions
#atoms is a list of atom indices,
def distance_between_atoms(fragment=None, atoms=None):
    dist = distance(fragment.coords[atoms[0]], fragment.coords[atoms[1]])
    return dist

def angle_between_atoms(fragment=None, atoms=None):
    angle_deg = angle(fragment.coords[atoms[0]], fragment.coords[atoms[1]], fragment.coords[atoms[2]])
    return angle_deg

def dihedral_between_atoms(fragment=None, atoms=None):
    dihed_deg = dihedral(fragment.coords[atoms[0]], fragment.coords[atoms[1]], fragment.coords[atoms[2]], fragment.coords[atoms[3]])
    return dihed_deg


# TODO: clean up
def get_centroid(coords):
    sum_x = 0; sum_y = 0; sum_z = 0
    for c in coords:
        sum_x += c[0]
        sum_y += c[1]
        sum_z += c[2]
    return [sum_x / len(coords), sum_y / len(coords), sum_z / len(coords)]


# Change origin to centroid. Either use centroid of full system (default) or alternatively subset or (something else even)
def change_origin_to_centroid(fullcoords, subsetcoords=None, subsetatoms=None):
    if subsetcoords != None:
        print("Calculating centroid for the specified subset coordinates")
        centroid = get_centroid(subsetcoords)
    elif subsetatoms != None:
        print("Calculating centroid for the coordintes of specified subatoms:", subsetatoms)
        #Will grab subsetcoords
        subcoords= np.take(fullcoords, subsetatoms, axis=0)
        centroid = get_centroid(subcoords)
    else:
        print("Calculating centroid for full set of coordinates")
        centroid = get_centroid(fullcoords)

    newcoords = fullcoords - centroid
    print("Returning full coordinates with new origin at centroid")
    return newcoords


# get_solvshell function based on single point of origin. Using geometric center of molecule
def get_solvshell_origin():
    print("to finish")
    # TODO: finish get_solvshell_origin
    ashexit()


# Determine threshold for whether atoms are connected or not based on covalent radii for pair of atoms
# R_ij < scale*(rad_i + rad_j) + tol
# Uses global scale and tol parameters that may be changed at input
def threshold_conn(elA, elB, scale, tol):
    # crad=list(map(eldict_covrad.get, [elA,elB]))
    # crad=[eldict_covrad.get(key) for key in [elA,elB]]
    return scale * (eldict_covrad[elA] + eldict_covrad[elB]) + tol
    # print(crad)
    # return scale*(crad[0]+crad[1]) + tol


# Connectivity function (called by Fragment object)
def calc_conn_py(coords, elems, conndepth, scale, tol):
    found_atoms = []
    fraglist = []
    for atom in range(0, len(elems)):
        if atom not in found_atoms:
            members = get_molecule_members_loop_np2(coords, elems, conndepth, scale, tol, atomindex=atom)
            if members not in fraglist:
                fraglist.append(members)
                found_atoms += members
    return fraglist


# Get connected atoms to chosen atom index based on threshold
# Uses slow for-loop structure with distance-function call
# Don't use unless system is small
def get_connected_atoms(coords, elems, scale, tol, atomindex):
    connatoms = []
    coords_ref = coords[atomindex]
    elem_ref = elems[atomindex]
    for i, c in enumerate(coords):
        if distance(coords_ref, c) < threshold_conn(elems[i], elem_ref, scale, tol):
            if i != atomindex:
                connatoms.append(i)
    return connatoms

# Euclidean distance functions:
# https://semantive.com/pl/blog/high-performance-computation-in-python-numpy/
def einsum_mat(mat_v, mat_u):
    mat_z = mat_v - mat_u
    return np.sqrt(np.einsum('ij,ij->i', mat_z, mat_z))

def bare_numpy_mat(mat_v, mat_u):
    return np.sqrt(np.sum((mat_v - mat_u) ** 2, axis=1))

def l2_norm_mat(mat_v, mat_u):
    return np.linalg.norm(mat_v - mat_u, axis=1)

def dummy_mat(mat_v, mat_u):
    return [sum((v_i - u_i) ** 2 for v_i, u_i in zip(v, u)) ** 0.5 for v, u in zip(mat_v, mat_u)]

# Get connected atoms to chosen atom index based on threshold
# np version for calculating the euclidean distance
# https://semantive.com/pl/blog/high-performance-computation-in-python-numpy/
def get_connected_atoms_np(coords, elems, scale, tol, atomindex):
    # print("inside get conn atoms np")
    # print("atomindex:", atomindex)
    connatoms = []
    # Creating np array of the coords to compare
    compcoords = np.tile(coords[atomindex], (len(coords), 1))
    # Einsum is slightly faster than bare_numpy_mat. All distances in one go
    distances = einsum_mat(coords, compcoords)
    # Getting all thresholds as list via list comprehension.
    el_covrad_ref = eldict_covrad[elems[atomindex]]
    # Cheaper way of getting thresholds list than calling threshold_conn
    # List comprehension of dict lookup and convert to numpy. Should be as fast as can be done
    # thresholds = np.empty(len(elems))
    # for i in range(len(thresholds)):
    #    thresholds[i]=eldict_covrad[elems[i]]
    # TODO: Slowest part but hard to make faster
    thresholds = np.array([eldict_covrad[elems[i]] for i in range(len(elems))])
    # Numpy addition and multiplication done on whole array
    thresholds = thresholds + el_covrad_ref
    thresholds = thresholds * scale
    thresholds = thresholds + tol
    # Old slow way
    # thresholds=np.array([threshold_conn(elems[i], elem_ref,scale,tol) for i in range(len(elems))])
    # Getting difference of distances and thresholds
    diff = distances - thresholds
    # Getting connatoms by finding indices of diff with negative values (i.e. where distance is smaller than threshold)
    connatoms = np.where(diff < 0)[0].tolist()
    return connatoms

# Get a dictionary of atoms (values) connected to each atom (key)
def get_connected_atoms_dict(coords, elems, scale, tol):
    conndict={}
    for c in range(0,len(coords)):
        conn = get_connected_atoms_np(coords, elems, scale, tol, c)
        conn.remove(c)
        conndict[c]=conn
    return conndict

#Get connected atoms for a small list of atoms with input fragment, includes input atoms
#Used e.g. in NEB-TS
def get_conn_atoms_for_list(atoms=None, fragment=None,scale=1.0, tol=0.1):
    final_list=[]
    for atom in atoms:
        conn = ash.modules.module_coords.get_connected_atoms_np(fragment.coords, fragment.elems, scale, tol, atom)
        final_list.append(conn)
    #Flatten list
    final_list  = [item for sublist in final_list for item in sublist]
    #Remove duplicates and sort
    return np.unique(final_list).tolist()


# Numpy clever loop test.
# Either atomindex or membs has to be defined
def get_molecule_members_loop_np(coords, elems, loopnumber, scale, tol, atomindex='', membs=None):
    if membs is None:
        membs = []
        membs.append(atomindex)
        membs = get_connected_atoms_np(coords, elems, scale, tol, atomindex)
    # How often to search for connected atoms as the members list grows:
    # TODO: Need to make this better
    for i in range(loopnumber):
        for j in membs:
            conn = get_connected_atoms_np(coords, elems, scale, tol, j)
            membs = membs + conn
        membs = np.unique(membs).tolist()
    # Remove duplicates and sort
    membs = np.unique(membs).tolist()
    return membs


# Numpy clever loop test.
# Version 2 never goes through same atom

def get_molecule_members_loop_np2(coords, elems, loopnumber, scale, tol, atomindex=None, membs=None):
    if membs is None:
        membs = []
        membs.append(atomindex)
        timestampA = time.time()
        membs = get_connected_atoms_np(coords, elems, scale, tol, atomindex)
        # ash.print_time_rel(timestampA, modulename='membs first py')

    # If membs is just an integer turn into list
    if type(membs) == int:
        membs = [membs]
    finalmembs = membs

    for i in range(loopnumber):
        # Get list of lists of connatoms for each member
        newmembers = [get_connected_atoms_np(coords, elems, scale, tol, k) for k in membs]
        # print("newmembers:", newmembers)
        # ashexit()
        # Get a unique flat list
        trimmed_flat = np.unique([item for sublist in newmembers for item in sublist]).tolist()
        # print("trimmed_flat:", trimmed_flat)
        # print("finalmembs ", finalmembs)

        # Check if new atoms not previously found
        membs = listdiff(trimmed_flat, finalmembs)
        # print("membs:", membs)
        # ashexit()
        # Exit loop if nothing new found
        if len(membs) == 0:
            # print("exiting...")
            # ashexit()
            return finalmembs
        # print("type of membs:", type(membs))
        # print("type of finalmembs:", type(finalmembs))
        finalmembs += membs
        # print("finalmembs ", finalmembs)
        finalmembs = np.unique(finalmembs).tolist()
        # print("finalmembs ", finalmembs)
        # ashexit()
        # print("finalmembs:", finalmembs)
        # print("----------")
        # ash.print_time_rel(timestampA, modulename='finalmembs  py')
        # ashexit()
    return finalmembs


# Get molecule members by running get_connected_atoms function on expanding member list
# Uses loopnumber for when to stop searching.
# Does extra work but not too bad
# Uses either single atomindex or members lists
def get_molecule_members_loop(coords, elems, loopnumber, scale, tol, atomindex='', members=None):
    if members is None:
        members = []
        members.append(atomindex)
        connatoms = get_connected_atoms(coords, elems, scale, tol, atomindex)
        members = members + connatoms
    # How often to search for connected atoms as the members list grows:
    for i in range(loopnumber):
        # conn = [get_connected_atoms(coords, elems, scale,tol,j) for j in members]
        for j in members:
            conn = get_connected_atoms(coords, elems, scale, tol, j)
            members = members + conn
            # members=np.concatenate((members, conn))
        members = np.unique(members).tolist()
        members = members + conn
    # Remove duplicates and sort
    members = np.unique(members).tolist()
    return members


# Get-molecule-members with fixed recursion-depth of 4
# Efficient but limited to 4
# Updated to 5
# Maybe not so efficient after all
def get_molecule_members_fixed(coords, elems, scale, tol, atomindex='', members=None):
    print("Disabled")
    print("not so efficient")
    ashexit()
    if members is None:
        members = [atomindex]
        connatoms = get_connected_atoms(coords, elems, scale, tol, atomindex)
        members = members + connatoms
    finalmembers = members
    # How often to search for connected atoms as the members list grows:
    for j in members:
        conn = get_connected_atoms(coords, elems, scale, tol, j)
        finalmembers = finalmembers + conn
        for k in conn:
            conn2 = get_connected_atoms(coords, elems, scale, tol, k)
            finalmembers = finalmembers + conn2
            # for l in conn2:
            #    conn3 = get_connected_atoms(coords, elems, scale,tol,l)
            #    finalmembers = finalmembers + conn3
            # for m in conn3:
            #    conn4 = get_connected_atoms(coords, elems, scale, tol,m)
            #    finalmembers = finalmembers + conn4
    # Remove duplicates and sort
    finalmembers = np.unique(finalmembers).tolist()
    return finalmembers


def create_coords_string(elems, coords):
    coordsstring = ''
    for el, c in zip(elems, coords):
        coordsstring = coordsstring + el + '  ' + str(c[0]) + '  ' + str(c[1]) + '  ' + str(c[2]) + '\n'
    return coordsstring[:-1]


# Takes list of elements and gives formula
def elemlisttoformula(list):
    # This dict comprehension was slow for large systems. Using set to reduce iterations
    dict = {i: list.count(i) for i in set(list)}
    formula = ""
    for item in dict.items():
        el = item[0]
        count = item[1]
        # string=el+str(count)
        formula = formula + el + str(count)
    return formula


# From molecular formula (string, e.g. "FeCl4") to list of atoms
def molformulatolist(formulastring):
    el = ""
    diff = ""
    els = []
    atomunits = []
    numels = []
    # Read string by character backwards
    for count, char in enumerate(formulastring[::-1]):
        if isint(char):
            el = char + el
        if char.islower():
            el = char + el
            diff = char + diff
        if char.isupper():
            el = char + el
            diff = char + diff
            atomunits.append(el)
            els.append(diff)
            el = ""
            diff = ""
    for atm, element in zip(atomunits, els):
        if atm > element:
            number = atm[len(element):]
            numels.append(int(number))
        else:
            number = 1
            numels.append(int(number))
    atoms = []
    for i, j in zip(els, numels):
        for k in range(j):
            atoms.append(i)
    # Final reverse
    els.reverse()
    numels.reverse()
    atoms.reverse()
    return atoms


# Read XYZ file
def read_xyzfile(filename,printlevel=2):
    # Will accept atom-numbers as well as symbols
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                'Rb',
                'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                'Cs',
                'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
                'Ta',
                'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa',
                'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    if printlevel >= 2:
        print("Reading coordinates from XYZ file '{}'.".format(filename))
    coords = []
    elems = []
    with open(filename) as f:
        for count, line in enumerate(f):
            if count == 0:
                numatoms = int(line.split()[0])
            if count > 1:
                if len(line.strip()) > 0:
                    if isint(line.split()[0]) is True:
                        # Grabbing element as atomnumber and reformatting
                        # el=dictionaries_lists.element_dict_atnum[int(line.split()[0])].symbol
                        el = reformat_element(int(line.split()[0]), isatomnum=True)
                        elems.append(el)
                    else:
                        # Grabbing element as symbol and reformatting just in case
                        el = reformat_element(line.split()[0])
                        elems.append(el)
                    coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    if len(coords) != numatoms:
        print(BC.FAIL,"Error: Number of coordinates in XYZ-file: {} does not match header line. Exiting.".format(filename))
        ashexit()
    if len(coords) != len(elems):
        print("Number of coordinates does not match elements. Something wrong with XYZ-file?: ", filename)
        ashexit()
    return elems, coords

#Read all XYZ-files from directory
#Return fragment list
def read_xyzfiles(xyzdir,readchargemult=False, label_from_filename=True):
    print("read_xyzfiles function")
    print("Note: will read XYZ-files in directory using natural sorting")
    import glob
    filenames=[];fragments=[]
    for file in natural_sort(glob.glob(xyzdir+'/*.xyz')):
        filename=os.path.basename(file)
        filenames.append(filename)
        print("\n\nXYZ-file:", filename)
        #Creating new fragment, reading charge/mult and using filename as fragment label
        mol=ash.Fragment(xyzfile=file, readchargemult=readchargemult, label=filename)
        fragments.append(mol)
    return fragments

def set_coordinates(atoms, V, title="", decimals=8):
    """
    Print coordinates V with corresponding atoms to stdout in XYZ format.
    Parameters
    ----------
    atoms : list
        List of atomic types
    V : array
        (N,3) matrix of atomic coordinates
    title : string (optional)
        Title of molecule
    decimals : int (optional)
        number of decimals for the coordinates

    Return
    ------
    output : str
        Molecule in XYZ format

    """
    N, D = V.shape

    fmt = "{:2s}" + (" {:15." + str(decimals) + "f}") * 3

    out = list()
    out += [str(N)]
    out += [title]

    for i in range(N):
        atom = atoms[i]
        atom = atom[0].upper() + atom[1:]
        out += [fmt.format(atom, V[i, 0], V[i, 1], V[i, 2])]

    return "\n".join(out)


def print_coordinates(atoms, V, title=""):
    """
    Print coordinates V with corresponding atoms to stdout in XYZ format.

    Parameters
    ----------
    atoms : list
        List of element types
    V : array
        (N,3) matrix of atomic coordinates
    title : string (optional)
        Title of molecule

    """
    V = np.array(V)
    print(set_coordinates(atoms, V, title=title))
    return


# Write XYZfile provided list of elements and list of list of coords and filename
#Fast version. Note: list comprehension is bottleneck, unclear how to make this faster though
def write_xyzfile(elems, coords, name, printlevel=2, writemode='w', title="title"):
    #timestampA = time.time()
    #Adding headerlines to list
    header=[f"{len(elems)}\n", f"{title}\n"]
    #print("TimeA:", time.time() - timestampA)
    atomlines = [f"{el:4} {c[0]:16.12f} {c[1]:16.12f} {c[2]:16.12f}\n" for el, c in zip(elems, coords)]
    #print("TimeB:", time.time() - timestampA)
    with open(name + '.xyz', writemode) as ofile:
        ofile.writelines(header)
        ofile.writelines(atomlines)
    #print("TimeC:", time.time() - timestampA)
    if printlevel >= 2:
        print("Wrote XYZ file: ", name + '.xyz')

# Function that reads XYZ-file with multiple files, splits and return list of coordinates
# Created for splitting crest_conformers.xyz but may also be used for MD traj.
# Also grabs last word in title line. Typically an energy (has to be converted to float outside)
def split_multimolxyzfile(file, writexyz=False, skipindex=1,return_fragments=False):
    all_coords = []
    all_elems = []
    all_titles = []
    molcounter = 0
    coordgrab = False
    titlegrab = False
    coords = []
    elems = []
    fragments=[]
    with open(file) as f:
        for index, line in enumerate(f):
            if index == 0:
                numatoms = line.split()[0]
            # Grab coordinates
            if coordgrab is True:
                if len(line.split()) > 1:
                    # elems.append(line.split()[0])
                    elems.append(reformat_element(line.split()[0]))
                    coords_x = float(line.split()[1])
                    coords_y = float(line.split()[2])
                    coords_z = float(line.split()[3])
                    coords.append([coords_x, coords_y, coords_z])
                if len(coords) == int(numatoms):
                    all_coords.append(coords)
                    all_elems.append(elems)
                    if writexyz is True:
                        # Alternative option: write each conformer/molecule to disk as XYZfile
                        write_xyzfile(elems, coords, "molecule" + str(molcounter))
                    frag = Fragment(coords=coords,elems=elems,printlevel=0)
                    fragments.append(frag)
                    coords = []
                    elems = []
            # Grab title
            if titlegrab is True:
                if len(line.split()) > 0:
                    all_titles.append(line.split()[-1])
                else:
                    all_titles.append("NA")
                titlegrab = False
                coordgrab = True
            # Grabbing number of atoms from string
            if len(line.split()) > 0:
                if line.split()[0] == str(numatoms):
                    # print("Molcounter", molcounter)
                    # print("coords is", len(coords))
                    if molcounter % skipindex:
                        molcounter += 1
                        titlegrab = False
                        coordgrab = False
                    else:
                        # print("Using. molcounter", molcounter)
                        molcounter += 1
                        titlegrab = True
                        coordgrab = False
                        # ashexit()
    print(f"Found {molcounter} geometries in file: {file}")

    if return_fragments is True:
        return fragments
    else:
        return all_elems, all_coords, all_titles


# Read Tcl-Chemshell fragment file and grab elems and coords. Coordinates converted from Bohr to Angstrom
# Taken from functions_solv
def read_chemshellfragfile_xyz(fragfile):
    # removing extension from fragfile name if present and then adding back.
    pathtofragfile = fragfile.split('.')[0] + '.c'
    coords = []
    elems = []
    # TODO: Change elems and coords to numpy array instead
    grabcoords = False
    with open(pathtofragfile) as ffile:
        for line in ffile:
            if 'block = connectivity' in line:
                grabcoords = False
            if grabcoords is True:
                coords.append([float(i) * ash.constants.bohr2ang for i in line.split()[1:]])
                el = reformat_element(line.split()[0])
                elems.append(el)
            if 'block = coordinates records ' in line:
                # numatoms=int(line.split()[-1])
                grabcoords = True
        coords = reformat_list_to_array(coords)
    return elems, coords


def conv_atomtypes_elems(atomtype):
    """Convert atomtype string to element based on a dictionary.
        Hopefully captures all cases. If atomtype not found then element string assumed but reformatting so correct case

    Args:
        atomtype ([str]): [description]
    Returns:
        [str]: [description]
    """
    try:
        element = ash.dictionaries_lists.atomtypes_dict[atomtype]
        return element
    except:
        # Assume correct element but could be wrongly formatted (e.g. FE instead of Fe) so reformatting
        try:
            element = reformat_element(atomtype)
            return element
        except:
            print("Atomtype: '{}' not recognized either as valid atomtype or element. Exiting.".format(atomtype))
            print("You might have to modify the atomtype/element information in coordinate file you're reading in.")
            ashexit()


# READ PDBfile
def read_pdbfile(filename, use_atomnames_as_elements=False):
    residuelist = []
    # If elemcolumn found
    elemcol = []
    # Not atomtype but atomname
    # atom_name=[]
    # atomindex=[]
    residname = []

    coords = []
    try:
        with open(filename) as f:
            for line in f:
                # if 'ATOM ' in line or 'HETATM' in line:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # print("line:", line)
                    # atomindex=float(line[6:11].replace(' ',''))
                    atom_name = line[12:16].replace(' ', '')
                    residname.append(line[17:20].replace(' ', ''))
                    residuelist.append(line[22:26].replace(' ', ''))
                    coords_x = float(line[30:38].replace(' ', ''))
                    coords_y = float(line[38:46].replace(' ', ''))
                    coords_z = float(line[46:54].replace(' ', ''))
                    coords.append([coords_x, coords_y, coords_z])
                    elem = line[76:78].replace(' ', '').replace('\n', '')
                    # elem=elem.replace('\n','')
                    # Option to use atomnamecolumn for element information instead of element-column
                    if use_atomnames_as_elements is True:
                        elem_name = ash.dictionaries_lists.atomtypes_dict[atom_name]
                        elemcol.append(elem_name)
                    else:
                        if len(elem) != 0:
                            if len(elem) == 2:
                                # Making sure second elem letter is lowercase
                                # elemcol.append(elem[0]+elem[1].lower())
                                elemcol.append(reformat_element(elem))
                            else:
                                elemcol.append(reformat_element(elem))
                        else:
                            print("While reading line:")
                            print(line)
                            print("No element found in element-column of PDB-file")
                            print(
                                "Either fix element-column (columns 77-78) or try to use to read element-information from atomname-column:")
                            print(" Fragment(pdbfile='X', use_atomnames_as_elements=True) ")
                            ashexit()
                    # self.coords.append([float(line.split()[6]), float(line.split()[7]), float(line.split()[8])])
                    # elemcol.append(line.split()[-1])
                    # residuelist.append(line.split()[3])
                    # atom_name.append(line.split()[3])
                # if 'HETATM' in line:
                #    print("HETATM line in file found. Please rename to ATOM")
                #    ashexit()
    except FileNotFoundError:
        print("File '{}' does not exist!".format(filename))
        ashexit()
    # Create numpy array
    coords_np = reformat_list_to_array(coords)

    if len(elemcol) != len(coords):
        print("len coords", len(coords))
        print("len elemcol", len(elemcol))
        print("did not find same number of elements as coordinates")
        print("Need to define elements in some other way")
        ashexit()
    else:
        elems = elemcol
    return elems, coords_np

def read_pdbfile_info(filename, use_atomnames_as_elements=False):
    atomnames=[]
    chainlabels=[]
    residnames = []
    residlabels = []
    conect_lines = []
    try:
        with open(filename) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    #print(line)
                    atomnames.append(line[12:16].replace(' ', ''))
                    residnames.append(line[17:20].replace(' ', ''))
                    chainlabels.append(line[21:22].replace(' ', ''))
                    #Resid grab
                    #Note: Resids are integer up to 9999 but after that many programs (VMD, OpenMM) switch to a hex notation
                    # Here grabbing resid as string instead of integer in general
                    residlabel_temp=line[22:26].replace(' ', '')
                    if residlabel_temp == 'A000':
                        print("Warning: read_pdbfile_info encountered a hexadecimal notation (A000) for resid (likely due to resids > 9999). Hopefully things will be fine")
                        print(f"PDB-file: {filename}. Line: {line}")
                    residlabel=str(residlabel_temp)
                    residlabels.append(residlabel)
                if line.startswith("CONECT"):
                    conect_lines.append(line)
    except FileNotFoundError:
        print("File '{}' does not exist!".format(filename))
        ashexit()

    return atomnames, residnames, residlabels, chainlabels, conect_lines



# Read GROMACS Gro coordinate file and box info
# Read AMBERCRD file and coords and box info
# Not part of Fragment class because we don't have element information here
def read_gromacsfile(grofile):
    elems = []
    coords = []
    # TODO: Change coords to numpy array instead
    grabcoords = False
    numatoms = "unset"
    box_dims = None
    with open(grofile) as cfile:
        for i, line in enumerate(cfile):
            if i == 0:
                pass
            elif i == 1:
                numatoms = int(line.split()[0])
                print("Numatoms:", numatoms)
            elif i == numatoms + 2:
                # Last line: box dimensions
                box_dims = [10 * float(i) for i in line.split()]
                # Assuming cubic and adding 90,90,90
                box_dims.append(90.0)
                box_dims.append(90.0)
                box_dims.append(90.0)
                print("Box dimensions read: ", box_dims)
            else:
                linelist = line.split()
                # Grabbing atomtype
                atomtype = linelist[1]
                atomtype = ''.join((item for item in atomtype if not item.isdigit()))
                atomtype = atomtype.replace('\'', '')
                # Converting atomtype to element based on function above
                elem = conv_atomtypes_elems(atomtype)
                elems.append(elem)

                # If larer than 7 then GRO file contains both coords and velocities
                if len(linelist) > 7:
                    coords_x = float(linelist[-6])
                    coords_y = float(linelist[-5])
                    coords_z = float(linelist[-4])
                # If smaller then only coords
                else:
                    coords_x = float(linelist[-3])
                    coords_y = float(linelist[-2])
                    coords_z = float(linelist[-1])
                # Converting from nm to Ang
                coords.append([10 * coords_x, 10 * coords_y, 10 * coords_z])
    npcoords=reformat_list_to_array(coords)
    if len(npcoords) != len(elems):
        print(BC.FAIL,"Num coords not equal to num elems. Parsing of Gromacsfile: {} failed. BUG!".format(grofile))
        ashexit()
    return elems, npcoords, box_dims


# Read AMBERCRD file and coords and box info
# Not part of Fragment class because we don't have element information here
def read_ambercoordinates(prmtopfile=None, inpcrdfile=None):
    elems = []
    coords = []
    # TODO: Change coords to numpy array instead
    grabcoords = False
    numatoms = "unset"
    box_dims=[]
    with open(inpcrdfile) as cfile:

        for i, line in enumerate(cfile):
            if i == 0:
                pass
            elif i == 1:
                numatoms = int(line.split()[0])
                print("Numatoms: ", numatoms)
                numcoordlines = math.ceil(numatoms / 2)
                # print("numcoordlines:", numcoordlines)
            elif i == numcoordlines + 2:

                # Last line: box dimensions
                box_dims = [float(i) for i in line.split()]
                print("Box dimensions read: ", box_dims)
            else:
                linelist = line.split()
                coordvalues = []
                # Checking if values combined: e,g, -16.3842161-100.0326085
                # Then split and add
                for c in linelist:
                    if c.count('.') > 1:
                        d = c.replace('-', ' -').split()
                        coordvalues.append(float(d[0]))
                        coordvalues.append(float(d[1]))
                    else:
                        coordvalues.append(float(c))
                coords.append([coordvalues[0], coordvalues[1], coordvalues[2]])
                if len(coordvalues) == 6:
                    coords.append([coordvalues[3], coordvalues[4], coordvalues[5]])

    # Grab atom numbers and convert to elements
    grab_atomnumber = False
    with open(prmtopfile) as pfile:
        for i, line in enumerate(pfile):
            if grab_atomnumber is True:
                if 'FORMAT' not in line:
                    # reformat_element(i,isatomnum=True)
                    if '%' in line:
                        grab_atomnumber = False
                    else:
                        elems += [reformat_element(int(i), isatomnum=True) for i in line.split()]
            if '%FLAG ATOMIC_NUMBER' in line:
                grab_atomnumber = True
    if len(coords) != len(elems):
        print(BC.FAIL,f"Num coords ({len(coords)}) not equal to num elems ({len(elems)}). Parsing of Amber files: {prmtopfile} and {inpcrdfile} failed. BUG!", BC.END)
        ashexit()
    return elems, coords, box_dims

#Write Amber crd file from either ASH fragment or PDB-file
def write_amber_crdfile(fragment=None, pdbfile=None, output="amber.crd"):
    print("\nwrite_amber_crdfile")
    if fragment == None and pdbfile == None:
        print("Need to provide either fragment or pdbfile")
        ashexit()
    if fragment != None:
        elems = fragment.elems
        coords = fragment.coords
        numatoms = fragment.numatoms
    elif pdbfile != None:
        elems, coords = read_pdbfile(pdbfile)
        numatoms = len(elems)
    full_string=""
    with open(output, 'w') as ofile:
        ofile.write(f"{numatoms}\n")
        for i,coord in enumerate(coords):
            full_string+=f"{coord[0]:12.7f}{coord[1]:12.7f}{coord[2]:12.7f}"
            if i % 2 != 0:
                full_string+="\n"
        ofile.write(full_string)


# Write PDBfile proper
# Example,manual: write_pdbfile(frag, outputname="name", atomnames=openmmobject.atomnames, resnames=openmmobject.resnames, residlabels=openmmobject.resids,segmentlabels=openmmobject.segmentnames)
# Example, simple: write_pdbfile(frag, outputname="name", openmmobject=objname)
# Example, minimal: write_pdbfile(frag)
# TODO: Add option to write new hybrid-36 standard PDB file instead of current hexadecimal nonstandard fix
def write_pdbfile(fragment, outputname="ASHfragment", openmmobject=None, atomnames=None, resnames=None,
                  residlabels=None, chainlabels=None, segmentlabels=None, dummyname='DUM', charges_column=None, conect_lines=None):
    print("Writing PDB-file...")
    # Using ASH fragment
    elems = fragment.elems
    coords = fragment.coords

    # Can grab everything from OpenMMobject if provided
    # NOTE: These lists are only defined for CHARMM files currently. Not Amber or GROMACS
    if openmmobject is not None:
        atomnames = openmmobject.atomnames
        #print(atomnames)
        resnames = openmmobject.resnames
        residlabels = openmmobject.resids
        segmentlabels = openmmobject.segmentnames

    # What to choose if keyword arguments not given
    if atomnames is None or len(atomnames) == 0:
        print("Warning: using elements as atomnames")
        # Elements instead. Means VMD will display atoms properly at least
        atomnames = fragment.elems
    if resnames is None or len(resnames) == 0:
        resnames = fragment.numatoms * [dummyname]
    if chainlabels is None or len(chainlabels) == 0:
        chainlabels = fragment.numatoms * ['']
    if residlabels is None or len(residlabels) == 0:
        residlabels = fragment.numatoms * [1]
    # Note: choosing to make segment ID 3-letter-string (and then space)
    if segmentlabels is None or len(segmentlabels) == 0:
        print("Warning: no segment labels")
        segmentlabels = fragment.numatoms * ['   ']
        #segmentlabels = fragment.numatoms * ['SEG']

    if len(atomnames) > 99999:
        print("System larger than 99999 atoms. Will use hexadecimal notation for atom indices 100K and larger. ")

    if (len(atomnames) == len(coords) == len(resnames) == len(residlabels) == len(segmentlabels)) is False:
        print(BC.FAIL,"Something went wrong in write_pdbfile. Exiting. File a bug report.", BC.END)
        print("ERROR: Problem with lists...")
        print("len: atomnames", len(atomnames))
        print("len: coords", len(coords))
        print("len: resnames", len(resnames))
        print("len: residlabels", len(residlabels))
        print("len: segmentlabels", len(segmentlabels))
        print("len elems:", len(elems))
        ashexit()

    with open(outputname + '.pdb', 'w') as pfile:
        for count, (atomname, c, resname, chainlabel,resid, seg, el) in enumerate(
                zip(atomnames, coords, resnames, chainlabels,residlabels, segmentlabels, elems)):
            atomindex = count + 1
            # Convert to hexadecimal if >= 100K.
            # Note: unsupported standard but VMD will read it
            if atomindex >= 100000:

                atomindexstring = hex(count + 1)[2:]
            else:
                atomindexstring = str(atomindex)

            # Using only first 3 letters of RESname
            resname = resname[0:3]

            # Using last 4 letters of atomnmae
            atomnamestring = atomname[-4:]
            #TODO: atomname should be unique so we should add a number here ideally

            #print(atomnamestring)
            #exit()
            # Using string format from: cupnet.net/pdb-format/

            #NOTE: Changed resid from integer to string so that we can support the hex notation for resids when resids go above 9999
            resid = str(resid)

            #Optional charges column (used by CP2K)
            if charges_column != None:
                charge=charges_column[count]
                #line = "{:6s}{:>5s} {:^4s}{:1s}{:3s}{:1s}{:5d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:4s}{:>2s} {:>10.6f}".format(
                #    'ATOM', atomindexstring, atomnamestring, '', resname, '', resid, '', c[0], c[1], c[2], 1.0, 0.00,
                #    seg[0:3], el, charge)
                line = "{:6s}{:5s} {:^4s}{:1s}{:3s} {:1s}{:4s}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format(
                    'ATOM', atomindexstring, atomnamestring, '', resname, chainlabel, resid, '', c[0], c[1], c[2], 1.0, 0.00, el,charge)
            #Regular
            else:
                #line = "{:6s}{:>5s} {:^4s}{:1s}{:3s}{:>2s}{:5d}{:1s}  {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:4s}{:>2s}".format(
                #    'ATOM', atomindexstring, atomnamestring, '', resname, chainlabel, resid, '', c[0], c[1], c[2], 1.0, 0.00,
                #    seg[0:3], el)
                line = "{:6s}{:5s} {:^4s}{:1s}{:3s} {:1s}{:4s}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format(
                    'ATOM', atomindexstring, atomnamestring, '', resname, chainlabel, resid, '', c[0], c[1], c[2], 1.0, 0.00, el,'')
            pfile.write(line + '\n')
        # Write CONECT lines if provided
        if conect_lines is not None:
            for conectline in conect_lines:
                pfile.write(conectline)
    print("Wrote PDB file: ", outputname + '.pdb')
    return outputname + '.pdb'

# Calculate nuclear charge from XYZ-file
def nucchargexyz(file):
    el = []
    with open(file) as f:
        for count, line in enumerate(f):
            if count > 1:
                el.append(reformat_element(line.split()[0]))
    totnuccharge = 0
    for e in el:
        atcharge = eldict[e]
        totnuccharge += atcharge
    return totnuccharge


# Calculate total nuclear charge from list of elements
def nucchargelist(ellist):
    totnuccharge = 0
    els = []
    warning_issued=False
    for e in ellist:
        try:
            atcharge = elematomnumbers[e.lower()]
        except KeyError:
            atcharge=0.0
            if warning_issued is False:
                print("Warning: Unknown element: '{}' found in element-list".format(e))
                print("Could be dummy atom. Using nuccharge of 0.0")
                warning_issued=True
        totnuccharge += atcharge
    return totnuccharge


# get list of nuclear charges from list of elements
# Used by Psi4 and CM5calc and xTBlibrary
# aka atomic numbers, aka atom numbers
def elemstonuccharges(ellist):
    nuccharges = []
    for e in ellist:
        atcharge = elematomnumbers[e.lower()]
        nuccharges.append(atcharge)
    return nuccharges

def nucchargestoelems(chargelist):
    elems = []
    for c in chargelist:
        for key, value in elematomnumbers.items():
            if value == c:
                elems.append(key.upper())
    return elems


# Calculate molecular mass from list of atoms
def totmasslist(ellist):
    totmass = 0
    warning_issued=False
    for e in ellist:
        try:
            atcharge = int(elematomnumbers[e.lower()])
            atmass = atommasses[atcharge - 1]
        except KeyError:
            atmass = 0.0
            if warning_issued is False:
                print("Warning: Unknown element: '{}' found in element-list".format(e))
                print("Could be dummy atom. Using mass of 0.0")
                warning_issued=True

        totmass += atmass
    return totmass


# Calculate list of masses from list of elements
def list_of_masses(ellist):
    masses = []
    warning_issued=False
    for e in ellist:
        try:
            atcharge = int(elematomnumbers[e.lower()])
            atmass = atommasses[atcharge - 1]
        except KeyError:
            atmass = 0.0
            if warning_issued is False:
                print("Warning: Unknown element: '{}' found in element-list".format(e))
                print("Could be dummy atom. Using mass of 0.0")
                warning_issued=True
        masses.append(atmass)
    return masses


##########################################################################################################
#Flexible align functions that take 2 fragments or files and aligns one of them on top of the other
#Support different order, i.e. we reorder. We can either keep reorder or not
#We also support subset of atoms to align on.
##########################################################################################################

#For XYZ-files
def flexible_align_xyz(xyzfileA, xyzfileB, rotate_only=False, translate_only=False, reordering=False, reorder_method='brute', subset=None):
    print(f"Will align molecule in file {xyzfileA} onto molecule in file {xyzfileB}")
    fragmentA = Fragment(xyzfile=xyzfileA)
    fragmentB = Fragment(xyzfile=xyzfileB)

    newfragA = flexible_align(fragmentA,fragmentB, rotate_only=rotate_only, translate_only=translate_only, reordering=reordering, reorder_method=reorder_method, subset=subset)

    #Write XYZ-file for newfragA
    newfragA.write_xyzfile(f"{xyzfileA.replace('.xyz','')}_aligned.xyz")

#For PDB-files
def flexible_align_pdb(pdbfileA, pdbfileB, rotate_only=False, translate_only=False, reordering=False, reorder_method='brute', subset=None):
    print(f"Will align molecule in file {pdbfileA} onto molecule in file {pdbfileB}")
    fragmentA = Fragment(pdbfile=pdbfileA)
    fragmentB = Fragment(pdbfile=pdbfileB)

    #Call flexible align, get aligned coords as new fragA
    newfragA = flexible_align(fragmentA,fragmentB, rotate_only=rotate_only, translate_only=translate_only, reordering=reordering, reorder_method=reorder_method, subset=subset)

    #Write PDBfile. PDB-info will have been read and stored
    fragmentA.coords=newfragA.coords #Replacing coords in original fragmentA
    fragmentA.write_pdbfile(filename=f"{pdbfileA.replace('.pdb','')}_aligned") #Now write out

#For ASH fragments
def flexible_align(fragmentA, fragmentB, rotate_only=False, translate_only=False, reordering=False, reorder_method='brute', subset=None):
    print("flexible_align function")
    import geometric
    #Do chosen subset
    if subset is not None:
        print("Subset option chosen")
        if any(isinstance(el, list) for el in subset) is True:
            print("Subset is a list of lists")
            print("Subset for A:", subset[0])
            print("Subset for B:", subset[1])
            if len(subset[0]) != len(subset[1]):
                print("Length of subsets not equal. This is not allowed. Exiting.")
                ashexit()
            print("Will align using each list of indices for each fragment")
            subsetA_coords, subsetA_elems =fragmentA.get_coords_for_atoms(subset[0])
            subsetB_coords, subsetB_elems =fragmentB.get_coords_for_atoms(subset[1])

        else:
            print("Subset is a list of indices")
            print("Will align using the same indices in both fragments (will only work if both fragments have the same atom order)")
            subsetA_coords, subsetA_elems =fragmentA.get_coords_for_atoms(subset)
            subsetB_coords, subsetB_elems =fragmentB.get_coords_for_atoms(subset)

        print("subsetA_elems:", subsetA_elems)
        print("subsetA_coords:", subsetA_coords)

        print("subsetB_elems:", subsetB_elems)
        print("subsetB_coords:", subsetB_coords)

    else:
        subsetA_coords=fragmentA.coords
        subsetB_coords=fragmentB.coords

    #TODO Possible reordering
    if reordering is True:
        print("Reordering atoms in fragmentB for better alignment (may not always work)")
        print("Warning: this requires the rmsd package to be installed: pip install rmsd")
        from rmsd import reorder_brute, reorder_hungarian, reorder_inertia_hungarian, reorder_similarity, reorder_distance
        print(f"Reorder method: {reorder_method}")
        if reorder_method == 'brute':
            print("Warning: brute force method can be very slow for large systems but is very accurate")
            print("If too slow then try next (in order): inertia_hungarian, hungarian and distance")
        #Note: brute works well, hungarian fails e.g. for benzamidine example, distance works for benzamidine
        reorder_methods_dict={'brute':reorder_brute, 'hungarian':reorder_hungarian, 'inertia_hungarian':reorder_inertia_hungarian, 'similarity':reorder_similarity, 'distance':reorder_distance}
        print("Note: All reorder-method options (from rmsd pakcage): brute, hungarian, inertia_hungarian, similarity, distance")
        order = reorder(reorder_methods_dict[reorder_method], np.array(subsetA_coords), np.array(subsetB_coords),
                                np.array(subsetA_elems), np.array(subsetB_elems))
        print("Order:", order)
        subsetB_coords = subsetB_coords[order]
    else:
        print("No reordering of atoms in fragmentA")

    #Use geometric function to get translation and rotation matrices for the subsets
    trans, rot = geometric.molecule.get_rotate_translate(subsetA_coords,subsetB_coords)

    #Translate only (all atoms in A)
    if translate_only is True:
        print("Doing translation only")
        Anew = fragmentA.coords + trans
    #Rotate only (all atoms in A)
    elif rotate_only is True:
        print("Doing rotation only")
        Anew =  np.dot(fragmentA.coords, rot)
    #Trans+rot (default)
    else:
        #Apply trans+rot to all atoms in fragmentA
        Anew = np.dot(fragmentA.coords, rot) + trans

    #Create new frag
    newfrag = Fragment(elems=fragmentA.elems, coords=Anew, printlevel=0)
    print("New aligned structure")
    newfrag.print_coords()

    return newfrag


# Recommended RMSD-calc wrapper function for ASH fragments using kabsch
# Allows subset match (same set of indices or 2 sets of indices for each fragment)
# Also simpler option: heavyatomsonly=True (ignores H-atoms)
#NOTE: no reordering
def calculate_RMSD(fragmentA, fragmentB, subset=None, heavyatomsonly=False, printlevel=2,
                   write_aligned_structure=False):
    print("calculate_RMSD function")

    #Do chosen subset
    if subset is not None:
        print("Subset option chosen")
        if any(isinstance(el, list) for el in subset) is True:
            print("Subset is a list of lists")
            print("Subset for A:", subset[0])
            print("Subset for B:", subset[1])
            if len(subset[0]) != len(subset[1]):
                print("Length of subsets not equal. This is not allowed. Exiting.")
                ashexit()
            print("Will align using each list of indices for each fragment")
            subsetA_coords, subsetA_elems =fragmentA.get_coords_for_atoms(subset[0])
            subsetB_coords, subsetB_elems =fragmentB.get_coords_for_atoms(subset[1])

        else:
            print("Subset is a list of indices")
            print("Will align using the same indices in both fragments (will only work if both fragments have the same atom order)")
            subsetA_coords, subsetA_elems =fragmentA.get_coords_for_atoms(subset)
            subsetB_coords, subsetB_elems =fragmentB.get_coords_for_atoms(subset)

        if printlevel > 2:
            print("subsetA_elems:", subsetA_elems)
            print("subsetA_coords:", subsetA_coords)

            print("subsetB_elems:", subsetB_elems)
            print("subsetB_coords:", subsetB_coords)
    elif heavyatomsonly is True:
        subsetA_coords=fragmentA.coords[fragmentA.get_nonH_atomindices()]
        subsetB_coords=fragmentB.coords[fragmentB.get_nonH_atomindices()]

    else:
        subsetA_coords=fragmentA.coords
        subsetB_coords=fragmentB.coords


    #Use geometric function to get translation and rotation matrices for the subsets
    import geometric
    trans, rot = geometric.molecule.get_rotate_translate(subsetA_coords,subsetB_coords)
    Anew = np.dot(subsetA_coords, rot) + trans

    #RMSD
    rmsdval = float(np.sqrt(((Anew-subsetB_coords)**2).sum()/len(Anew)))
    #xrmsdval = kabsch_rmsd(subsetB_coords,Anew)
    
    if printlevel > 1:
        print("RMSD:", rmsdval)

    if write_aligned_structure:
        print("write_aligned_structure active")
        newfrag = Fragment(elems=fragmentA.elems, coords=Anew)
        newfrag.write_xyzfile("structA_aligned.xyz")


    return rmsdval


#####################################
# RMSD and align related functions
#####################################

def kabsch_rmsd(P, Q):
    """
    Rotate matrix P unto Q and calculate the RMSD
    """
    P = rotate(P, Q)
    return rmsd(P, Q)


def rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm
    """
    U = kabsch(P, Q)
    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch(P, Q):
    """
    The optimal rotation matrix U is calculated and then used to rotate matrix
    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
    calculated.
    Using the Kabsch algorithm with two sets of paired point P and Q,
    centered around the center-of-mass.
    Each vector set is represented as an NxD matrix, where D is the
    the dimension of the space.
    The algorithm works in three steps:
    - a translation of P and Q
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters:
    P -- (N, number of points)x(D, dimension) matrix
    Q -- (N, number of points)x(D, dimension) matrix
    Returns:
    U -- Rotation matrix
    """
    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U

def centroid(X):
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    https://en.wikipedia.org/wiki/Centroid

    C = sum(X)/len(X)

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    C : float
        centroid
    """
    C = X.mean(axis=0)
    return C


def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    """
    D = len(V[0])
    N = len(V)
    rmsd = 0.0
    for v, w in zip(V, W):
        rmsd += sum([(v[i] - w[i]) ** 2.0 for i in range(D)])
    return np.sqrt(rmsd / N)


# Turbomol coord->xyz
def coord2xyz(inputfile):
    """convert TURBOMOLE coordfile to xyz"""
    coords=[]
    elems=[]
    with open(inputfile, 'r') as f:
        coord = f.readlines()
        x = []
        y = []
        z = []
        atom = []
        for line in coord[1:-1]:
            x=float(line.split()[0]) * ash.constants.bohr2ang
            y=float(line.split()[1]) * ash.constants.bohr2ang
            z=float(line.split()[2]) * ash.constants.bohr2ang
            el = reformat_element(str(line.split()[3]))
            elems.append(el)
            coords.append([x,y,z])
    print("coords:", coords)
    print("elems", elems)
    numatoms=len(elems)
    with open("fromcoord.xyz", 'w') as cfile:
        cfile.write(f"{numatoms}\n")
        cfile.write(f"title\n")
        for e,c in zip(elems,coords):
            cfile.write(" {:13} {:20.16f} {:20.16f} {:20.16f}\n".format(e,c[0],c[1],c[2]))
    return

# Turbomole xyz->coord
def xyz2coord(inputfile):
    """convert xyz to TURBOMOLE coordfile"""
    coords=[]
    elems=[]
    with open(inputfile, 'r') as f:
        for i,line in enumerate(f):
            if i > 1:
                x=float(line.split()[1]) / ash.constants.bohr2ang
                y=float(line.split()[2]) / ash.constants.bohr2ang
                z=float(line.split()[3]) / ash.constants.bohr2ang
                el = reformat_element(str(line.split()[0]))
                elems.append(el)
                coords.append([x,y,z])
    with open("coord", 'w') as cfile:
        cfile.write(f"$coord\n")
        for e,c in zip(elems,coords):
            cfile.write("   {:20.16f} {:20.16f} {:20.16f} {:>13}\n".format(c[0],c[1],c[2],e))
        cfile.write(f"$end\n")


    return


# Get partial list by deleting elements not present in provided list of indices.
def get_partial_list(allatoms, partialatoms, l):
    newlist = copy.copy(l) #Otherwise object may be updated
    otheratoms = listdiff(allatoms, partialatoms)
    otheratoms.reverse()
    for at in otheratoms:
        del newlist[at]
    return newlist


# Old function that used scipy to do distances and Hungarian.
def scipy_hungarian(A, B):
    import scipy
    # timestampA = time.time()
    distances = scipy.spatial.distance.cdist(A, B, 'euclidean')
    # print("distances:", distances)
    # ash.print_time_rel(timestampA, modulename='scipy distances_cdist')
    # timestampA = time.time()
    indices_a, assignment = scipy.optimize.linear_sum_assignment(distances)
    # print("indices_a:", indices_a)
    # print("assignment:", assignment)
    # ash.print_time_rel(timestampA, modulename='scipy linear sum assignment')
    return assignment


# Hungarian algorithm to reorder coordinates. Uses Julia to calculates distances between coordinate-arrays A and B and then Hungarian Julia package.
def hungarian_julia(A, B):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    try:
        # Calculating distances via Julia
        # print("Here. Calling Julia distances")
        # timestampA = time.time()

        # This one is SLOW!!! For rad30 Bf3hcn example it takes 23 seconds compare to 3.8 sec for scipy. 0.8 sec for scipy for both dist and hungarian
        # distances =Juliafunctions.distance_array(A,B)
        distances = cdist(A, B, 'euclidean')
        # ash.print_time_rel(timestampA, modulename='julia distance array')
        # timestampA = time.time()
        # Julian Hungarian call. Requires Hungarian package
        try:
            Juliafunctions = load_julia_interface()
        except:
            print("Problem loading Julia.")
            ashexit()
        assignment, cost = Juliafunctions.Hungarian.hungarian(distances)

        # ash.print_time_rel(timestampA, modulename='julia hungarian')
        # timestampA = time.time()
        # Removing zeros and offsetting by 1 (Julia 1-indexing)
        final_assignment = assignment[assignment != 0] - 1

        # final_assignment = scipy_hungarian(A,B)

    except:
        print("Problem running Julia Hungarian function. Trying scipy instead.")

        ashexit()

        final_assignment = scipy_hungarian(A, B)

    return final_assignment


# Hungarian reorder algorithm
# From RMSD
def reorder_hungarian_scipy(p_atoms, q_atoms, p_coord, q_coord):
    """
    Re-orders the input atom list and xyz coordinates using the Hungarian
    method (using optimized column results)

    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    view_reorder : array
             (N,1) matrix, reordered indexes of atom alignment based on the
             coordinates of the atoms

    """

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)
    #print("unique_atoms:", unique_atoms)
    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)
    view_reorder -= 1

    for atom in unique_atoms:
        p_atom_idx, = np.where(p_atoms == atom)
        q_atom_idx, = np.where(q_atoms == atom)

        A_coord = p_coord[p_atom_idx]
        B_coord = q_coord[q_atom_idx]
        # print("A_coord:", A_coord)
        # print("B_coord:", B_coord)

        view = scipy_hungarian(A_coord, B_coord)
        view_reorder[p_atom_idx] = q_atom_idx[view]
    # print("view_reorder:", view_reorder)
    return view_reorder


def reorder_hungarian_julia(p_atoms, q_atoms, p_coord, q_coord):
    """
    Re-orders the input atom list and xyz coordinates using the Hungarian
    method (using optimized column results)

    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    view_reorder : array
             (N,1) matrix, reordered indexes of atom alignment based on the
             coordinates of the atoms

    """

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)
    print("unique_atoms: ", unique_atoms)
    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)
    view_reorder -= 1
    print("view_reorder: ", view_reorder)
    for atom in unique_atoms:
        p_atom_idx, = np.where(p_atoms == atom)
        q_atom_idx, = np.where(q_atoms == atom)

        A_coord = p_coord[p_atom_idx]
        B_coord = q_coord[q_atom_idx]

        view = hungarian_julia(A_coord, B_coord)
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder


def check_reflections(p_atoms, q_atoms, p_coord, q_coord,
                      reorder_method=reorder_hungarian_scipy,
                      rotation_method=kabsch_rmsd,
                      keep_stereo=False):
    """
    Minimize RMSD using reflection planes for molecule P and Q

    Warning: This will affect stereo-chemistry

    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    q_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    min_rmsd
    min_swap
    min_reflection
    min_review

    """

    min_rmsd = np.inf
    min_swap = None
    min_reflection = None
    min_review = None
    tmp_review = None
    swap_mask = [1, -1, -1, 1, -1, 1]
    reflection_mask = [1, -1, -1, -1, 1, 1, 1, -1]

    for swap, i in zip(AXIS_SWAPS, swap_mask):
        for reflection, j in zip(AXIS_REFLECTIONS, reflection_mask):
            if keep_stereo and i * j == -1:
                continue  # skip enantiomers

            tmp_atoms = copy.copy(q_atoms)
            tmp_coord = copy.deepcopy(q_coord)
            tmp_coord = tmp_coord[:, swap]
            tmp_coord = np.dot(tmp_coord, np.diag(reflection))
            tmp_coord -= centroid(tmp_coord)

            # Reorder
            if reorder_method is not None:
                tmp_review = reorder_method(p_atoms, tmp_atoms, p_coord, tmp_coord)
                tmp_coord = tmp_coord[tmp_review]
                tmp_atoms = tmp_atoms[tmp_review]

            # Rotation
            if rotation_method is None:
                this_rmsd = rmsd(p_coord, tmp_coord)
            else:
                this_rmsd = rotation_method(p_coord, tmp_coord)

            if this_rmsd < min_rmsd:
                min_rmsd = this_rmsd
                min_swap = swap
                min_reflection = reflection
                min_review = tmp_review

    if not (p_atoms == q_atoms[min_review]).all():
        print("error: Not aligned")
        quit()

    return min_rmsd, min_swap, min_reflection, min_review


def reorder(reorder_method, p_coord, q_coord, p_atoms, q_atoms):
    p_cent = centroid(p_coord)
    q_cent = centroid(q_coord)
    p_coord -= p_cent
    q_coord -= q_cent

    #Convert from element string to atomic number
    p_atoms = np.array([elematomnumbers[el.lower()] for el in p_atoms])
    q_atoms = np.array([elematomnumbers[el.lower()] for el in q_atoms])

    q_review = reorder_method(p_atoms, q_atoms, p_coord, q_coord)
    reorderlist = [q_review.tolist()][0]
    return reorderlist


AXIS_SWAPS = np.array([
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 1, 0],
    [2, 0, 1]])
AXIS_REFLECTIONS = np.array([
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1]])



# QM-region expand function. Finds whole fragments.
# Used by molcrys. Similar to get_solvshell function in functions_solv.py
def QMregionfragexpand(fragment=None, initial_atoms=None, radius=None):
    # If needed (connectivity ==0):
    scale = ash.settings_ash.settings_dict["scale"]
    tol = ash.settings_ash.settings_dict["tol"]
    if fragment is None or initial_atoms is None or radius is None:
        print("Provide fragment, initial_atoms and radius keyword arguments to QMregionfragexpand!")
        ashexit()
    subsetelems = [fragment.elems[i] for i in initial_atoms]
    # subsetcoords=[fragment.coords[i]for i in initial_atoms ]
    subsetcoords = np.take(fragment.coords, initial_atoms, axis=0)
    if len(fragment.connectivity) == 0:
        print("No connectivity found. Using slow way of finding nearby fragments...")
    atomlist = []

    for i, c in enumerate(subsetcoords):
        el = subsetelems[i]
        for index, allc in enumerate(fragment.coords):
            all_el = fragment.elems[index]
            if index >= len(subsetcoords):
                dist = distance(c, allc)
                if dist < radius:
                    # Get molecule members atoms for atom index.
                    # Using stored connectivity because takes forever otherwise
                    # If no connectivity
                    if len(fragment.connectivity) == 0:
                        # wholemol=get_molecule_members_loop(fragment.coords, fragment.elems, index, 1, scale, tol)
                        wholemol = get_molecule_members_loop_np2(fragment.coords, fragment.elems, 99, scale, tol,
                                                                 atomindex=index)

                    # If stored connectivity
                    else:
                        for q in fragment.connectivity:
                            # ashexit()
                            if index in q:
                                wholemol = q
                                break

                    elematoms = [fragment.elems[i] for i in wholemol]
                    atomlist = atomlist + wholemol
    atomlist = np.unique(atomlist).tolist()
    return atomlist

#Similar to QMregionfragexpand but cleaner
def cut_sphere(fragment=None, center_atom=None, radius=None):

    scale = ash.settings_ash.settings_dict["scale"]
    tol = ash.settings_ash.settings_dict["tol"]

    coords = fragment.coords
    center = fragment.coords[center_atom]

    # Calculate distances from center for each coordinate
    relative_coords = coords - center
    distances = np.linalg.norm(relative_coords, axis=1)
    inside_sphere = distances <= radius
    uncut_indices = np.where(inside_sphere)[0]
    atomlist=[]
    for uncut_index in uncut_indices:
        wholemol = get_molecule_members_loop_np2(fragment.coords, fragment.elems, 99, scale, tol,
                                                                 atomindex=uncut_index)
        atomlist = atomlist + wholemol
    atomlist = np.unique(atomlist).tolist()
    return atomlist

#Similar to QMregionfragexpand and cut_sphere but a cubic box is cut instead
def cut_cubic_box(fragment=None, center_atom=None, radius=None):

    scale = ash.settings_ash.settings_dict["scale"]
    tol = ash.settings_ash.settings_dict["tol"]

    coords = fragment.coords
    center = fragment.coords[center_atom]

    # Calculate distances from center for each coordinate
    relative_coords = coords - center
    inside_box = np.all(np.abs(relative_coords) <= radius, axis=1)
    uncut_indices = np.where(inside_box)[0]
    #cut_indices = np.where(~inside_box)[0]
    atomlist=[]
    for uncut_index in uncut_indices:
        wholemol = get_molecule_members_loop_np2(fragment.coords, fragment.elems, 99, scale, tol,
                                                                 atomindex=uncut_index)
        atomlist = atomlist + wholemol
    atomlist = np.unique(atomlist).tolist()
    return atomlist

#Function to do QM-region expansion based on QM/MM pointcharge gradient
def QMPC_fragexpand(theory=None, fragment=None, thresh=5e-4):
    if theory is None and fragment is None:
        print("QMPC_fragexpand requires fragment and theory")
        ashexit()
    if not isinstance(theory,ash.QMMMTheory):
        print("Theory is not a QMMMTheory")
        ashexit()

    #QM/MM run
    ash.Singlepoint(theory=theory, fragment=fragment, Grad=True)

    #Selection scheme based on pointcharge gradient
    pcgrad = theory.PCgradient
    large_force_indices = np.unique(np.argwhere(abs(pcgrad) > thresh)[:,0])
    #Convert pcgrad indices to system indices
    proper_largeforce_indices = large_force_indices+len(theory.qmatoms)
    #Get whole molecules
    fragment.calc_connectivity() #get connectivity

    #New expansion
    new_expansion=theory.qmatoms

    for i in proper_largeforce_indices:
        mol_index = search_list_of_lists_for_index(i,fragment.connectivity)
        molmembers = fragment.connectivity[mol_index]
        new_expansion = new_expansion + molmembers
    new_expansion = np.unique(new_expansion)

    #Print to output and disk
    print("New QM-region expansion based on pointcharge gradient selection")
    fragment.print_coords_for_atoms(new_expansion, labels=new_expansion)
    print("Writing coordinates to file: QMPC_selection.xyz")
    fragment.write_XYZ_for_atoms(xyzfilename="QMPC_selection.xyz", atoms=new_expansion)

    return new_expansion

#Function to determine the QM-MM boundary
#Note: This function was dominating QMMMTheory creation (e.g. 9.67 s / 12.41 s => 78 % for 300K system)
#Now sped up via get_connected_atoms_np. Silly
def get_boundary_atoms(qmatoms, coords, elems, scale, tol, excludeboundaryatomlist=None, unusualboundary=False):
    timeA = time.time()
    print("Determining QM-MM/HL-LL boundary")
    print("Parameters determing connectivity:")
    print("Scaling factor:", scale)
    print("Tolerance:", tol)
    if excludeboundaryatomlist is None:
        excludeboundaryatomlist = []

    print("QM atoms:", qmatoms)
    print("QM atoms to be excluded from boundary creation (excludeboundaryatomlist): ", excludeboundaryatomlist)
    # For each QM atom, do a get_conn_atoms, for those atoms, check if atoms are in qmatoms,
    # if not, then we have found an MM-boundary atom

    # TODO: Note, there can can be problems here if either scale, tol is non-ideal value (should be set in inputfile)
    # TODO: Or if eldict_covrad needs to be modified, also needs to be set in inputfile then.

    qm_mm_boundary_dict = {}
    for qmatom in qmatoms:
        # Option below to skip creating boundaryatom pair (and subsequent linkatoms) if atom index is flagged
        # Applies to rare case where QM atom is bonded to MM atom but we don't want a linkatom.
        # Example: bridging sulfide in Cys that connects to Fe4S4 and H-cluster.
        if qmatom in excludeboundaryatomlist:
            print("QMatom : {} in excludeboundaryatomlist: {}".format(qmatom, excludeboundaryatomlist))
            print("Skipping QM-MM boundary...")
            continue
        #Note: get_connected_atoms very slow
        #connatoms = get_connected_atoms(coords, elems, scale, tol, qmatom)
        connatoms = get_connected_atoms_np(coords, elems, scale, tol, qmatom)
        # Find connected atoms that are not in QM-atoms
        boundaryatom = listdiff(connatoms, qmatoms)

        if len(boundaryatom) > 1:
            print(BC.FAIL,
                  "Problem. Found more than 1 boundaryatom for QM-atom {} . This is not allowed".format(qmatom),
                  BC.END)
            print("This typically either happens when your QM-region is badly defined or a QM-atom is clashing with an MM atom")
            print("QM atom : ", qmatom)
            print("MM Boundaryatoms (connected to QM-atom based on distance) : ", boundaryatom)
            print("Please define the QM-region so that only 1 linkatom would be required.")
            print("MM Boundary atom coordinates (for debugging):")
            for b in boundaryatom:
                print(f"{b} {elems[b]} {coords[b][0]} {coords[b][1]} {coords[b][2]}")
            ashexit()
        elif len(boundaryatom) == 1:

            # Warn if QM-MM boundary is not a plain-vanilla C-C bond
            if elems[qmatom] != "C" or elems[boundaryatom[0]] != "C":
                print(BC.WARNING, "Warning: QM-MM boundary is not the ideal C-C scenario:", BC.END)
                print(BC.WARNING,
                      "QM-MM boundary: {}({}) - {}({})".format(elems[qmatom], qmatom, elems[boundaryatom[0]],
                                                               boundaryatom[0]), BC.END)
                if unusualboundary is False:
                    print(BC.WARNING,
                          "Make sure you know what you are doing (also note that ASH counts atoms from 0 not 1). Exiting.",
                          BC.END)
                    print(BC.WARNING, "To override exit, add: unusualboundary=True  to QMMMTheory object ", BC.END)
                    ashexit()
            # Adding to dict
            qm_mm_boundary_dict[qmatom] = boundaryatom[0]
    print("QM-MM boundary dictionary:", qm_mm_boundary_dict)
    print_time_rel(timeA, modulename="get_boundary_atoms")
    return qm_mm_boundary_dict


# Get linkatom positions for a list of qmatoms and the current set of coordinates
# Two methods: simple method (default) and ratio method.
# Simple method: Just use a fixed distance (default 1.09 Å)
# Ratio method: Determine by scaling QM1-MM1 distance with a ratio. Ratio can be fixed value (e.g. 0.723) or determined from equilibrium distances (not ready)
# Using linkatom distance of 1.09 Å for now as default. Makes sense for C-H link atoms.
def get_linkatom_positions(qm_mm_boundary_dict, qmatoms, coords, elems, linkatom_method='simple', linkatom_type='H',
                           linkatom_simple_distance=None, bondpairs_eq_dict=None, linkatom_ratio=0.723):
    timeA = time.time()
    print("Inside get_linkatom_positions")
    print("linkatom_type:", linkatom_type)
    print("linkatom_method:", linkatom_method)

    if linkatom_simple_distance is None:
        print("linkatom_simple_distance not set. Getting standard distance from dictionary for each element:")
    else:
        print("linkatom_simple_distance was set by user:", linkatom_simple_distance)
    #Dict of linkatom distances for different elements
    linkdistances_dict = {('C', 'H'): 1.09, ('O', 'H'): 0.98, ('N', 'H'): 0.99}
    print("Linkdatom distance dictionary:", linkdistances_dict)
    # If dictionary of linkatom-distances provided then use that instead
    if linkatom_method == 'ratio':
        if linkatom_ratio == 'Auto' and bondpairs_eq_dict is None:
            #TODO: Determine automatically somehow  
            bondpairs_eq_dict = {('C', 'H'): 1.09, ('C', 'C'): 1.522269, ('C', 'N'): 1.47, 
                                 ('C', 'O'): 1.43, ('C', 'S'): 1.81}

    # Get boundary atoms
    print("qm_mm_boundary_dict:", qm_mm_boundary_dict)
    # Get coordinates for QMX and MMX pair. Create new L coordinate that has a modified distance to QMX
    linkatoms_dict = {}
    for dict_item in qm_mm_boundary_dict.items():
        qmatom_coords = np.array(coords[dict_item[0]])
        mmatom_coords = np.array(coords[dict_item[1]])

        #Determine linkatom distance
        if linkatom_method == 'ratio':
            #print("Linkatom method: ratio")
            if linkatom_ratio == 'Auto':
                print("Automatic ratio. Determining ratio based on dict of equilibrium distances")
                #TODO
                R_eq_QM_H = bondpairs_eq_dict[(elems[dict_item[0]], linkatom_type)]
                R_eq_QM_MM = bondpairs_eq_dict[(elems[dict_item[0]], elems[dict_item[1]])]
                print("R_eq_QM_H:", R_eq_QM_H)
                print("R_eq_QM_MM:", R_eq_QM_MM)
                linkatom_ratio = R_eq_QM_H / R_eq_QM_MM
                print("Determined ratio:", linkatom_ratio)
                print("not yet ready")
                ashexit()
            r_QM1_MM1 = distance(qmatom_coords, mmatom_coords)
            # See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9314059/
            linkatom_coords = linkatom_ratio *(mmatom_coords - qmatom_coords) + qmatom_coords
            #linkatom_distance =  r_QM1_MM1 * (bondpairs_eq_dict[(elems[dict_item[0]], 'H')] / bondpairs_eq_dict[(elems[dict_item[0]], elems[dict_item[1]])])
            linkatom_distance = distance(qmatom_coords, linkatom_coords)
            print(f"Linkatom distance (QM1-L) determined to be: {linkatom_distance} (using ratio {linkatom_ratio})")
        elif linkatom_method == 'simple':
            #print("Linkatom method: simple")
            if linkatom_simple_distance is None:
                #print("linkatom_simple_distance not set. Getting standard distance from dictionary for element:", elems[dict_item[0]])
                #Getting from dict
                linkatom_distance = linkdistances_dict[(elems[dict_item[0]], linkatom_type)]
            else:
                #print("linkatom_simple_distance was set by user:", linkatom_simple_distance)
                #Getting from user
                linkatom_distance = linkatom_simple_distance
            print("Linkatom distance (QM1-L) is:", linkatom_distance)
            #Determining coords
            linkatom_coords = list(qmatom_coords + (mmatom_coords - qmatom_coords) * (
                        linkatom_distance / distance(qmatom_coords, mmatom_coords)))
        else:
            print("Invalid linkatom_method. Exiting.")
            ashexit()
        
        linkatoms_dict[(dict_item[0], dict_item[1])] = linkatom_coords
    #print_time_rel(timeA, modulename="get_linkatom_positions")
    return linkatoms_dict


# Grabbing molecules from multi-XYZ trajectory file (can be MD-file, optimization traj, nebpath traj etc).
# Creating ASH fragments for each conformer
def get_molecules_from_trajectory(file, writexyz=False, skipindex=1, conncalc=False):
    print_line_with_subheader2("Get molecules from trajectory")
    print("Finding molecules/snapshots in multi-XYZ trajectory file and creating ASH fragments...")
    print("Taking every {}th entry".format(skipindex))
    list_of_molecules = []
    all_elems, all_coords, all_titles = split_multimolxyzfile(file, writexyz=writexyz, skipindex=skipindex,return_fragments=False)
    print("Found {} molecules in file.".format(len(all_elems)))
    for i,(els, cs) in enumerate(zip(all_elems, all_coords)):
        conf = ash.Fragment(elems=els, coords=cs, conncalc=conncalc, printlevel=0, label=f"{file}_{i}")
        list_of_molecules.append(conf)

    return list_of_molecules


# Extend cell in general with original cell in center
# NOTE: Taken from functions_molcrys.
# TODO: Remove function from functions_molcrys
def cell_extend_frag(cellvectors, coords, elems, cellextpars):
    printdebug("cellextpars:", cellextpars)
    permutations = []
    for i in range(int(cellextpars[0])):
        for j in range(int(cellextpars[1])):
            for k in range(int(cellextpars[2])):
                permutations.append([i, j, k])
                permutations.append([-i, j, k])
                permutations.append([i, -j, k])
                permutations.append([i, j, -k])
                permutations.append([-i, -j, k])
                permutations.append([i, -j, -k])
                permutations.append([-i, j, -k])
                permutations.append([-i, -j, -k])
    # Removing duplicates and sorting
    permutations = sorted([list(x) for x in set(tuple(x) for x in permutations)],
                          key=lambda x: (abs(x[0]), abs(x[1]), abs(x[2])))
    # permutations = permutations.sort(key=lambda x: x[0])
    printdebug("Num permutations:", len(permutations))
    numcells = np.prod(cellextpars)
    numcells = len(permutations)
    extended = np.zeros((len(coords) * numcells, 3))
    new_elems = []
    index = 0
    for perm in permutations:
        shift = cellvectors[0:3, 0:3] * perm
        shift = shift[:, 0] + shift[:, 1] + shift[:, 2]
        # print("Permutation:", perm, "shift:", shift)
        for d, el in zip(coords, elems):
            new_pos = d + shift
            extended[index] = new_pos
            new_elems.append(el)
            # print("extended[index]", extended[index])
            # print("extended[index+1]", extended[index+1])
            index += 1
    printdebug("extended coords num", len(extended))
    printdebug("new_elems  num,", len(new_elems))
    return extended, new_elems


# From Pymol. Not sure if useful
# NOTE: also in functions_molcrys
def cellbasis(angles, edges):
    from math import cos, sin, radians, sqrt
    """
    For the unit cell with given angles and edge lengths calculate the basis
    transformation (vectors) as a 4x4 numpy.array
    """
    rad = [radians(i) for i in angles]
    basis = np.identity(4)
    basis[0][1] = cos(rad[2])
    basis[1][1] = sin(rad[2])
    basis[0][2] = cos(rad[1])
    basis[1][2] = (cos(rad[0]) - basis[0][1] * basis[0][2]) / basis[1][1]
    basis[2][2] = sqrt(1 - basis[0][2] ** 2 - basis[1][2] ** 2)
    edges.append(1.0)
    return basis * edges  # numpy.array multiplication!

# Create a molecular cluster from a periodix box based on radius and chosen atom(s)

def make_cluster_from_box(fragment=None, radius=10, center_atomindices=[0], cellparameters=None):
    print_line_with_subheader2("Make cluster from box")
    # Choosing how far to extend cell based on chosen cluster-radius
    if radius < cellparameters[0]:
        cellextension = [2, 2, 2]
    else:
        cellextension = [3, 3, 3]

    print("Cell parameters:", cellparameters)
    print("Radius: {} Å".format(radius))
    print("Cell extension used: ", cellextension)
    print("Cluster will be centered on atom indices: ", center_atomindices)

    # Extend cell
    cellvectors = cellbasis(cellparameters[3:6], cellparameters[0:3])
    ext_coords, ext_elems = cell_extend_frag(cellvectors, fragment.coords, fragment.elems, cellextension)
    print("Size of extended cell: ", len(ext_elems))
    extcellfrag = ash.Fragment(elems=ext_elems, coords=ext_coords, printlevel=2)
    # Cut cluster with radius R from extended cell, centered on atomic index. Returns list of atoms
    atomlist = QMregionfragexpand(fragment=extcellfrag, initial_atoms=center_atomindices, radius=radius)

    # Grabbing coords and elems from atomlist and creating new fragment
    clustercoords = np.take(ext_coords, atomlist, axis=0)
    clusterelems = [ext_elems[i] for i in atomlist]
    newfrag = ash.Fragment(elems=clusterelems, coords=clustercoords, printlevel=0)

    return newfrag


# Set up constraints
# def set_up_MMwater_bondconstraints(actatoms, oxygentype='OT'):
#     print("set_up_MMwater_bondconstraints")
#     print("Assuming oxygen atom type is: ", oxygentype)
#     print("Change with keyword arguement: oxygentype='XX")
#     ashexit()
#     # Go over actatoms and check if oxygen-water type

#     # Shift nested list by number e.g. shift([[1,2],[100,101]], -1)  gives : [[0,1],[99,100]]
#     # TODO: generalize
#     def shift_nested(ll, par):
#         new = []
#         for l in ll:
#             new.append([l[0] + par, l[1] + par])
#         return new

#     bondconslist = shift_nested(bondlist, -1)
#     constraints = {'bond': bondconslist}

#     return constraints


# Function to update list of atomindices after deletion of a list of atom indices (used in remove_atoms functions below)
def update_atom_indices_upon_deletion(atomlist, dellist):
    # Making sure dellist is sorted and determining highest and lowest value
    dellist.sort(reverse=True)
    lowest_atomindex = dellist[-1]
    highest_atomindex = dellist[0]
    atomlist_new = []
    for q in atomlist:
        if q in dellist:
            # These QM atoms were deleted and do not survive
            pass
        elif q < lowest_atomindex:
            # These QM atoms have lower value than loweste deleted atomindex and survive
            atomlist_new.append(q)
        elif q > highest_atomindex:
            # Shifting these indices by length of delatoms-list
            shiftpar = len(dellist)
            atomlist_new.append(q - shiftpar)
        else:
            # These atom indices are inbetween
            # Shifting depending on how many delatoms indices come before
            shiftpar = len([i for i in dellist if i < q])
            atomlist_new.append(q - shiftpar)
    return atomlist_new


def remove_atoms_from_PSF(atomindices=None, topfile=None, psffile=None, psfgendir=None):
    # Change to 1-based indexing for PSFgen
    atomindices_string = '{ ' + ' '.join(([str(i + 1) for i in atomindices])) + ' }'
    psf_script = """
# This section requires PSFGEN 1.6 to be loaded
topology {}
readpsf {}
set psffile {}
#For each delatom
set delatoms {}
for {{ set i 0}}  {{ $i < [llength $delatoms] }} {{ incr i 1 }} {{
set d [lindex $delatoms $i]
set startatoms false

#Here finding out which segname and resid for atomnumber
     set fp [open $psffile r]
     while {{-1 != [gets $fp line]}} {{
          if {{[lindex $line 1] == "NATOM" || [lindex $line 1] == "!NATOM"}} {{
                set startatoms true
          }}
             if {{ $startatoms == "true" }} {{
                if {{ [lindex $line 0] == "$d" }} {{
                  set segname [lindex $line 1]
                  set resid [lindex $line 2]
                  set atomname [lindex $line 4]
                 puts "Deleting atom $atomname (segname  $segname,  resid is $resid) from PSF information!"
                 #PSFgen 1.6 delatom command
                 delatom $segname $resid $atomname
                }}
             }}
     }}


}}
close $fp

#Needed?
#regenerate angles dihedrals

#Printing Xplor PSF file
writepsf x-plor cmap newsystem_XPLOR.psf
#writepsf charmm cmap newsystem_CHARMM.psf
writepdb new-system.pdb
    """.format(topfile, psffile, psffile, atomindices_string)

    # Creating PSF inputfile
    with open("psfinput.tcl", 'w') as f:
        f.write(psf_script)

    # Running PSFgen. Writing to stdout
    process = sp.run([psfgendir + '/psfgen', 'psfinput.tcl'])


def remove_atoms_from_system_CHARMM(fragment=None, psffile=None, topfile=None, atomindices=None, psfgendir=None,
                                    qmatoms=None, actatoms=None, offset_atom_indices=0):
    print_line_with_mainheader("remove_atoms_from_system_CHARMM")
    if fragment is None or psffile is None or topfile is None or atomindices is None:
        print("Error: remove_atoms_from_system requires keyword arguments:")
        print("fragment, psffile, topfile, atomindices")
        ashexit()

    if psfgendir is None:
        print(BC.WARNING,
              "No psfgendir argument passed to remove_atoms_from_system. Attempting to find psfgendir "
              "variable inside settings_ash.",
              BC.END)
        try:
            psfgendir = ash.settings_ash.settings_dict["psfgendir"]
        except:
            print(BC.FAIL, "Found no psfgendir variable in settings_ash module or in $PATH. Exiting.", BC.END)
            ashexit()

    print("Atoms to be deleted (0-based indexing):", atomindices)
    for a in atomindices:
        print("Atom: {}, Element: {}".format(a, fragment.elems[a]))
    print("")
    # Deleting element and coords for each atom index
    atomindices.sort(reverse=True)
    lowest_atomindex = atomindices[-1]
    highest_atomindex = atomindices[0]
    for atomindex in atomindices:
        fragment.delete_atom(atomindex)
    print("")
    print("Removed atom from fragment.")

    # Using PSFgen to create new PSF-file
    remove_atoms_from_PSF(atomindices=atomindices, topfile=topfile, psffile=psffile, psfgendir=psfgendir)
    print("")
    print("Removed atom from PSF.")
    print("Wrote new PSF-file: newsystem_XPLOR.psf")
    print("Wrote new PDB-file: new-system.pdb")
    print("")
    # Writing new fragment to disk
    fragment.write_xyzfile(xyzfilename="newfragment.xyz")
    fragment.print_system(filename='newfragment.ygg')

    # Updating provided qmatoms and actatoms lists
    if qmatoms is not None and actatoms is not None:
        print("qmatoms and actatoms lists provided to function. Will now update atomindices in these lists.")
        print("Deletion list:", atomindices)
        print("Old list of QM atoms:", qmatoms)
        print("Old list of active atoms:", actatoms)
        print("")
        new_qmatoms = update_atom_indices_upon_deletion(qmatoms, atomindices)
        new_actatoms = update_atom_indices_upon_deletion(actatoms, atomindices)

        # Possible offset of atom indices
        new_qmatoms = [i + offset_atom_indices for i in new_qmatoms]
        new_actatoms = [i + offset_atom_indices for i in new_actatoms]

        print("New list of QM atoms: ", str(new_qmatoms).strip("[]"))
        print("New list of active atoms: ", str(new_actatoms).strip("[]"))
        writelisttofile(new_qmatoms, "newqmatoms")
        writelisttofile(new_actatoms, "newactive_atoms")
    else:
        print(
            "Warning: qmatoms and actatoms not provided to function. Use qmatoms and actatoms keyword "
            "arguments if you want to update qmatoms and actatoms list.")
        print("Otherwise you have to update qmatoms and actatoms lists manually!")
    print("")
    print("remove_atoms_from_system_CHARMM: Done!")


def add_atoms_to_PSF(resgroup=None, topfile=None, psffile=None, psfgendir=None, num_added_atoms=None):
    print("Finding resgroup {} in topfile {} ".format(resgroup, topfile))
    # Checking if resgroup present in topfile
    resgroup_in_topfile = False
    grab_atoms = False
    numatoms_in_resgroup = 0
    with open(topfile) as tfile:
        for line in tfile:
            if grab_atoms is True:
                if 'RESI' in line:
                    grab_atoms = False
                if 'ATOM ' in line:
                    numatoms_in_resgroup += 1
            if resgroup in line and 'RESI' in line:
                resgroup_in_topfile = True
                grab_atoms = True

    if resgroup_in_topfile is False:
        print("Chosen resgroup: {} not in topfile: {}".format(resgroup, topfile))
        print("Add residuegroup to topology file first!")
        print("Exiting.")
        ashexit()

    print("numatoms_in_resgroup: ", numatoms_in_resgroup)
    print("num_added_atoms: ", num_added_atoms)
    if numatoms_in_resgroup != num_added_atoms:
        print("Number of ATOM entries in resgroup in {} not equal to number of added atom-coordinates.")
        print("Wrong RESgroup chosen or missing coordinates?")
        print("Exiting")
        ashexit()

    # Dummy segmentname. Can't be something existing. Using ADD1, ADD2 etc.
    import random
    import string
    dummysegname="AD" + random.choice(string.ascii_uppercase) + str(random.randint(0, 9))
    #matches = pygrep2(dummysegname, psffile)
    #segname = dummysegname + str(len(matches) + 1)
    #print("segname:", segname)
    print("dummysegname:", dummysegname)
    psf_script = """
    topology {}
    readpsf {}

    segment {} {{ residue 1 {} }}

    #Printing Xplor PSF file
    writepsf x-plor cmap newsystem_XPLOR.psf
    #writepsf charmm cmap newsystem_CHARMM.psf
    writepdb new-system.pdb
        """.format(topfile, psffile, dummysegname, resgroup)

    # Creating PSF inputfile
    with open("psfinput.tcl", 'w') as f:
        f.write(psf_script)

    # Running PSFgen. Writing to stdout
    process = sp.run([psfgendir + '/psfgen', 'psfinput.tcl'])

    return


def add_atoms_to_system_CHARMM(fragment=None, added_atoms_coordstring=None, resgroup=None, psffile=None, topfile=None,
                               psfgendir=None, qmatoms=None, actatoms=None, offset_atom_indices=0):
    print_line_with_mainheader("add_atoms_to_system")
    if fragment is None or psffile is None or topfile is None or added_atoms_coordstring is None or resgroup is None:
        print("Error: add_atoms_to_system_CHARMM requires keyword arguments:")
        print("fragment, psffile, topfile, added_atoms_coordstring, resgroup")
        ashexit()

    if psfgendir is None:
        print(BC.WARNING,
              "No psfgendir argument passed to remove_atoms_from_system. Attempting to find "
              "psfgendir variable inside settings_ash.",
              BC.END)
        try:
            psfgendir = ash.settings_ash.settings_dict["psfgendir"]
        except:
            print(BC.FAIL, "Found no psfgendir variable in settings_ash module or in $PATH. Exiting.", BC.END)
            ashexit()

    # Adding coordinates to fragment
    added_atoms_coords_list = added_atoms_coordstring.split('\n')
    added_elems = []
    added_coords = []
    for count, line in enumerate(added_atoms_coords_list):
        if len(line) > 1:
            added_elems.append(reformat_element(line.split()[0]))
            added_coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    num_added_atoms = len(added_elems)
    newatomindices = [fragment.numatoms + i for i in range(0, num_added_atoms)]
    print("newatomindices (0-based indexing):", newatomindices)
    fragment.add_coords(added_elems, added_coords, conn=False)

    # Adding atoms to PSF-file
    add_atoms_to_PSF(resgroup, topfile, psffile, psfgendir, num_added_atoms)
    print("")
    print("Added atoms to PSF.")
    print("Wrote new PSF-file: 'newsystem_XPLOR.psf'.")
    print("Wrote new PDB-file: 'new-system.pdb'")
    print("")

    # Writing new fragment to disk
    fragment.write_xyzfile(xyzfilename="newfragment.xyz")
    fragment.print_system(filename='newfragment.ygg')

    if qmatoms is not None and actatoms is not None:
        print("qmatoms and actatoms lists provided to function. Will now add atomindices to these lists.")
        new_qmatoms = qmatoms + newatomindices
        new_actatoms = actatoms + newatomindices

        # Possible offset of atom indices
        new_qmatoms = [i + offset_atom_indices for i in new_qmatoms]
        new_actatoms = [i + offset_atom_indices for i in new_actatoms]

        print("New list of QM atoms:", str(new_qmatoms).strip("[]"))
        print("New list of active atoms:", str(new_actatoms).strip("[]"))
        writelisttofile(new_qmatoms, "newqmatoms")
        writelisttofile(new_actatoms, "newactive_atoms")
    else:
        print(
            "Warning: qmatoms and actatoms not provided to function. Use qmatoms and actatoms "
            "keyword arguments if you want to update qmatoms and actatoms list.")
        print("Otherwise you have to update qmatoms and actatoms lists manually!")
    print("")

    print("")
    print("add_atoms_to_system_CHARMM: Done!")


# Get list of lists of water constraints in system (O-H,O-H,H-H) via OpenMM theory
def getwaterconstraintslist(openmmtheoryobject=None, atomlist=None, watermodel='tip3p'):
    print("Inside getwaterconstraintslist")
    if openmmtheoryobject is None or atomlist is None:
        print("getwaterconstraintslist requires openmmtheoryobject and atomlist to be set ")
        ashexit()
    if watermodel == 'tip3p' or watermodel == 'spc':
        #oxygenlabels = ['OT', 'OW', 'OWT3']
        water_resname = ['HOH','WAT','TIP']
    else:
        print("unknown watermodel")
        ashexit()

    atomtypes=openmmtheoryobject.atomtypes
    resnames=openmmtheoryobject.resnames
    elements=openmmtheoryobject.mm_elements

    if len(resnames) == 0:
        print("Error: No resnames found in OpenMMTheory object")
        ashexit()
    if len(elements) == 0:
        print("Error: No mm_elements found in OpenMMTheory object")
        ashexit()

    waterconstraints = []
    if resnames:
        for index,(rn,el) in enumerate(zip(resnames,elements)):
            # Skipping if not in atomlist
            if index not in atomlist:
                continue

            if rn in water_resname:
                if el == 'O':
                    waterconstraints.append([index, index + 1])
                    waterconstraints.append([index, index + 2])
                    waterconstraints.append([index + 1, index + 2])
    #if len(atomtypes) == 0:
    #    #NOTE: Atomtypes only defined if OpenMMTheory created from CHARMM Files
    #    # Assuming OT or OW oxygen atomtypes used if TIP3P. Assuming oxygen comes first
    #    # TODO: support more water models here. like 4-site and 5-site models
    #
    #    waterconstraints = []
    #    for index, at in enumerate(atomtypes):
    #        # Skipping if not in actatomslist
    #        if actatoms is not None:
    #            if index not in actatoms:
    #                continue
    #        if at in oxygenlabels:
    #            waterconstraints.append([index, index + 1])
    #            waterconstraints.append([index, index + 2])
    #            waterconstraints.append([index + 1, index + 2])

    return waterconstraints


#Check whether spin multiplicity is consistent with the nuclear charge and total charge
def check_multiplicity(elems,charge,mult, exit=True):
    def is_even(number):
        if number % 2 == 0:
            return True
        return False
    #From elems list calculate nuclear charge
    nuccharge=nucchargelist(elems)
    num_electrons = nuccharge - charge
    unpaired_electrons=mult-1
    result = list(map(is_even, (num_electrons,unpaired_electrons)))
    if result[0] != result[1]:
        print("The spin multiplicity {} ({} unpaired electrons) is incompatible with the total number of electrons {}".format(mult,unpaired_electrons,num_electrons))
        if exit == True:
            print("Now exiting!")
            ashexit()
        else:
            return False
#Check if charge/mult variables are not None. If None check fragment
#Only done for QM theories not MM. Passing theorytype string (e.g. from theory.theorytype if available)
def check_charge_mult(charge, mult, theorytype, fragment, jobtype, theory=None, printlevel=2):

    #Check if QM or QM/MM theory
    if theorytype == "QM":
        if charge == None or mult == None:
            if printlevel >= 2:
                print(BC.WARNING,f"Charge/mult was not provided to {jobtype}",BC.END)
            if fragment.charge != None and fragment.mult != None:
                if printlevel >= 2:
                    print(BC.WARNING,"Fragment contains charge/mult information: Charge: {} Mult: {}  Using this.".format(fragment.charge,fragment.mult), BC.END)
                    #print(BC.WARNING,"Make sure this is what you want!", BC.END)
                charge=fragment.charge; mult=fragment.mult
            else:
                print(BC.FAIL,"No charge/mult information present in fragment either. Exiting.",BC.END)
                ashexit()
    elif theorytype=="QM/MM":

        #Note: theory needs to be set
        if charge is None or mult is None:
            if printlevel >= 1:
                print(BC.WARNING,f"Warning: Charge/mult was not provided to {jobtype}",BC.END)
                print("Checking if present in QM/MM object")
            if theory.qm_charge != None and theory.qm_mult != None:
                charge=theory.qm_charge
                mult=theory.qm_mult
                if printlevel >= 1:
                    print("Found qm_charge and qm_mult attributes.")
                    print(f"Using charge={charge} and mult={mult}")
            elif fragment.charge != None and fragment.mult != None:
                print(BC.WARNING,"Fragment contains charge/mult information: Charge: {} Mult: {} Using this instead".format(fragment.charge,fragment.mult), BC.END)
                print(BC.WARNING,"Make sure this is what you want!", BC.END)
                charge=fragment.charge; mult=fragment.mult
            else:
                print(BC.FAIL,"No charge/mult information present in fragment either. Exiting.",BC.END)
                ashexit()
    elif theorytype=="ONIOM":
        print("Checking if charge/mult information present in ONIOM object")
        if theory.fullregion_charge != None and theory.fullregion_mult != None:
            print("Found fullregion_charge and fullregion_mult attributes.")
            print("All good, continuing\n")
    elif theorytype=="MM":
        #Setting charge/mult to None if MM
        charge=None; mult=None
    return charge,mult


#Get list of bad atoms based on supplied fragment and gradient
def check_gradient_for_bad_atoms(fragment=None,gradient=None, threshold=45000):
    indices=[]
    print("Checking system total gradient for bad atoms")
    print("Gradient threshold setting:", threshold)
    for i,k in enumerate(gradient):
        if any(abs(k) > threshold):
            indices.append(i)
    if len(indices) > 0:
        print("The following atoms have abnormally high values, probably due to bad atom positions:")
        print()
        print("Index    Element           Coordinates                              Gradient")
        for i in indices:
            print(f"{i:7} {fragment.elems[i]:>5} {fragment.coords[i][0]:>12.6f} {fragment.coords[i][2]:>12.6f} {fragment.coords[i][2]:>12.6f}      {gradient[i][0]:>6.3f} {gradient[i][1]:>6.3f} {gradient[i][2]:>6.3f}")
        print()
        print("These atoms may need to be constrained (e.g. if metal-cofactor) or atom positions need to be corrected before starting simulation")
    else:
        print()
        print(f"No atoms with gradients larger than threshold: {threshold}")
    return indices


#Define XH bond constraints for a given fragment and a set of atomindices (e.g. an active region)
# and an optional exclusion list (e.g. QM-region)
def define_XH_constraints(fragment, actatoms=None, excludeatoms=None, conncode='py'):
    print("Inside define_XH_constraints function")
    if actatoms == None:
        subset_elems = fragment.elems
        subset_coords = fragment.coords
        actatoms = fragment.atomlist
    else:
        subset_elems = [fragment.elems[i] for i in actatoms]
        subset_coords = np.take(fragment.coords, actatoms, axis=0)

    print(f"Defining constraints for {len(subset_elems)} atom-region")

    #Finding H-atoms (both act indices and full indices)
    tempHatoms = [index for index, el in enumerate(subset_elems) if el == 'H']
    tempHatoms_full = [actindex_to_fullindex(i,actatoms) for i in tempHatoms]
    Hatoms=[]
    if excludeatoms != None:
        print("Checking for exclude atoms")
        for th,th_f in zip(tempHatoms,tempHatoms_full):
            if th_f not in excludeatoms:
                Hatoms.append(th)
    else:
        Hatoms=tempHatoms

    #Now finding X-H pairs for active region
    #py version (slow) but good enough for a few thousand atoms
    scale = ash.settings_ash.settings_dict["scale"]
    tol = ash.settings_ash.settings_dict["tol"]
    if conncode == 'py':
        act_con_list = []
        for Hatom in Hatoms:
            connatoms = get_connected_atoms_np(subset_coords, subset_elems, scale, tol, Hatom)
            act_con_list.append(connatoms)
    #Faster Julia function
    else:
        print("Loading Julia")
        try:
            Juliafunctions = load_julia_interface()
        except:
            print("Problem loading Julia")
            ashexit()
        act_con_list = Juliafunctions.get_connected_atoms_forlist_julia(subset_coords, subset_elems, scale, tol,
                                                                        eldict_covrad, Hatoms)
    #Convert XH actregion indices to finalregion indices
    final_list=[]
    for XHpair in act_con_list:
        if len(XHpair) != 2:
            print("XHpair is strange:", XHpair)
            ashexit()
        final_list.append([actindex_to_fullindex(XHpair[0],actatoms),actindex_to_fullindex(XHpair[1],actatoms)])
    return final_list

#Simple function to convert atom indices from full system to Active region. Single index case
def fullindex_to_actindex(fullindex,actatoms):
    actindex=actatoms.index(fullindex)
    return actindex

#Simple function to convert atom indices from active region to full-system case.
def actindex_to_fullindex(actindex,actatoms):
    fullindex = actatoms[actindex]
    return fullindex


#Simple get_water constraints for fragment without doing connectivity
#Limitation: Assumes all waters from starting index to end and that waters are ordered: O H H
def simple_get_water_constraints(fragment,starting_index=None, onlyHH=False):
    print("Inside simple_get_water_constraints function")
    print("Warning: Note that water residues have to have O,H,H order and have to be at the end of the coordinate file")
    print("Starting index for first water oxygen:", starting_index)
    if starting_index == None:
        print("Error: You must provide a starting_index value!")
        ashexit()
    if fragment.elems[starting_index] != 'O':
        print("Starting atom for water fragment is not oxygen!")
        print("Make sure starting index ({}) is correct".format(starting_index))
        print("Also note that water fragments must have O H H order!")
        ashexit()
    if onlyHH is False:
        print("onlyHH is False. Will create list of O-H1, O-H2 and H1-H2 constraints")
    elif onlyHH is True:
        print("onlyHH is True. Will create list of H1-H2 constraints only")
    #
    constraints=[]
    for i in range(starting_index, fragment.numatoms):
        if fragment.elems[i] == "O":
            #X-H constraint
            if onlyHH is False:
                constraints.append([i,i+1])
                constraints.append([i,i+2])
            #H-H constraints. i.e. effectively freezing angles
            constraints.append([i+1,i+2])
    return constraints


#Function to convert Mol file to PDB-file via OpenBabel
def mol_to_pdb(file):
    #OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: mol_to_pdb requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("mol", file))
    mol.write(format='pdb', filename=os.path.splitext(file)[0]+'.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0]+'.pdb')
    return os.path.splitext(file)[0]+'.pdb'

#Function to convert SDF file to PDB-file via OpenBabel
def sdf_to_pdb(file):
    #OpenBabel
    try:
        from openbabel import openbabel
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: sdf_to_pdb requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("sdf", file))

    #Write do disk as PDB-file
    mol.write(format='pdb', filename=os.path.splitext(file)[0]+'temp.pdb', overwrite=True)
    #Read-in again (this will create a Residue)
    newmol = next(pybel.readfile("pdb", os.path.splitext(file)[0]+'temp.pdb'))
    os.remove(os.path.splitext(file)[0]+'temp.pdb')

    #Atomlabel = {0:'C1',1:'X',2:'C',3:'C',4:'C',5:'C',6:'C',7:'C',8:'C',9:'C',10:'C',11:'C',12:'C'}
    #Change atomnames (AtomIDs) to something sensible (OpenBabel does not do this by default)
    print("Creating new atomnames for PDBfile")
    #Note: currently just combining element and atomindex to get a unique atomname (otherwise Modeller will not work)
    #TODO: make something better (element-specific numbering?)
    for res in pybel.ob.OBResidueIter(newmol.OBMol):
        for i,atom in enumerate(openbabel.OBResidueAtomIter(res)):
            atomname = res.GetAtomID(atom)
            #print("atomname:", atomname)
            res.SetAtomID(atom,atomname.strip()+str(i+1))
            atomname = res.GetAtomID(atom)
            #print("atomname:", atomname)
            #res.SetAtomID(atom,Atomlabel[i])

    #Write final PDB-file
    newmol.write(format='pdb', filename=os.path.splitext(file)[0]+'.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0]+'.pdb')
    return os.path.splitext(file)[0]+'.pdb'

#Function to read in PDB-file and write new one with CONECT lines (geometry needs to be sensible)
#NOTE: Requires OpenBabel which seems unnecessary, probably better to use OpenMM functionality instead
def writepdb_with_connectivity(file):
    #OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: writepdb_with_connectivity requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("pdb", file))
    mol.write(format='pdb', filename=os.path.splitext(file)[0]+'_withcon.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0]+'_withcon.pdb')
    return os.path.splitext(file)[0]+'_withcon.pdb'

#Function to read in XYZ-file (small molecule) and create PDB-file with CONECT lines (geometry needs to be sensible)
def xyz_to_pdb_with_connectivity(file, resname="UNL"):
    print("xyz_to_pdb_with_connectivity function:")
    #OpenBabel
    try:
        from openbabel import openbabel
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: xyz_to_pdb_with_connectivity requires OpenBabel library but it could not be imported")
        print("You can install OpenBabel like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    #Read in XYZ-file
    mol = next(pybel.readfile("xyz", file))
    #Write do disk as PDB-file
    mol.write(format='pdb', filename=os.path.splitext(file)[0]+'temp.pdb', overwrite=True)
    #Read-in again (this will create a Residue)
    newmol = next(pybel.readfile("pdb", os.path.splitext(file)[0]+'temp.pdb'))

    os.remove(os.path.splitext(file)[0]+'temp.pdb')

    #Atomlabel = {0:'C1',1:'X',2:'C',3:'C',4:'C',5:'C',6:'C',7:'C',8:'C',9:'C',10:'C',11:'C',12:'C'}
    #Change atomnames (AtomIDs) to something sensible (OpenBabel does not do this by default)
    print("Creating new atomnames for PDBfile")
    #Note: currently just combining element and atomindex to get a unique atomname (otherwise Modeller will not work)
    #TODO: make something better (element-specific numbering?)
    for res in pybel.ob.OBResidueIter(newmol.OBMol):
        #Setting residue name
        res.SetName(resname)
        for i,atom in enumerate(openbabel.OBResidueAtomIter(res)):
            atomname = res.GetAtomID(atom)
            #print("atomname:", atomname)
            res.SetAtomID(atom,atomname.strip()+str(i+1))
            atomname = res.GetAtomID(atom)
            #print("atomname:", atomname)
            #res.SetAtomID(atom,Atomlabel[i])

    #Write final PDB-file
    newmol.write(format='pdb', filename=os.path.splitext(file)[0]+'.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0]+'.pdb')
    return os.path.splitext(file)[0]+'.pdb'

#Function to convert PDB-file to SMILES string
def pdb_to_smiles(fname: str) -> str:
    #OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: pdb_to_smiles requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("pdb", fname))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()

#Function to convert PDB-file to SMILES string
def smiles_to_coords(smiles_string):
    #OpenBabel
    try:
        from openbabel import pybel
        from openbabel import openbabel
    except ModuleNotFoundError:
        print("Error: smiles_to_coords requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    print("Reading SMILES by OpenBabel")
    mol = pybel.readstring("smi", smiles_string)
    print("Guessing 3D coordinates (uses MMFF94 forcefield)")
    mol.make3D()
    b_mol = mol.OBMol
    atomnums = []
    coords = []
    for atom in openbabel.OBMolAtomIter(b_mol):
        atomnums.append(atom.GetAtomicNum())
        coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    elems = [reformat_element(atn, isatomnum=True) for atn in atomnums]
    #frag = Fragment(elems=elems, coords=coords, charge=charge, mult=mult)
    return elems, coords

#Function that adds R-group to an ASH fragment
def swap_R_group(fragment=None, Rgroup=None, atomindex=None) -> Fragment:
    if fragment is None or Rgroup is None or atomindex is None:
        print("Error: add_R_group requires fragment (ASH fragment), Rgroup (string, e.g. 'Cl', 'OH', 'NO2', to be set")
        print("and atomindex  (index of the atom to be swapped/replaced)")
        ashexit()
    #Look-up R-group
    Rgroup_xyzfile=ashpath+"/databases/fragments/R-groups/"+f"{Rgroup}.xyz"
    print("\nLooking up R-group xyzfile:", Rgroup_xyzfile)
    if os.path.isfile(Rgroup_xyzfile) is False:
        print(f"Error: R-group xyz-file {Rgroup}.xyz not found in ASH database at: {ashpath}/databases/fragments/R-groups")
        print("Please add it to ASH database and try again")
        ashexit()

    print("\nFragment: ", fragment)
    print("Atom index to be swapped: ", atomindex)
    labels = ['    WILL BE SWAPPED for R-group' if i == atomindex else '' for i in range(0,fragment.numatoms)]
    print()
    print_coords_all(fragment.coords, fragment.elems, labels=labels, labels2=None)


    #TODO
    R_elems, R_coords = read_xyzfile(Rgroup_xyzfile)

    newelems = fragment.elems + R_elems
    newcoords = fragment.coords + R_coords

    newfragment = Fragment(elems=newelems, coords=newcoords, charge=fragment.charge, mult=fragment.mult)

    return newfragment


#Function that calculates box size of a molecule in a cubic box
#with optional shift
def cubic_box_size(coords, shift=0.0):
    # max and min for x,y,z coords
    max_values = np.max(coords, axis=0)
    min_values = np.min(coords, axis=0)
    #Differences for x,y,z
    span_x = max_values[0] - min_values[0]
    span_y = max_values[1] - min_values[1]
    span_z = max_values[2] - min_values[2]
    # Max span for each x,y,z
    max_span = max(span_x, span_y, span_z)
    #Optional shift
    final_span = max_span + shift
    return final_span

#More general
def bounding_box_dimensions(coordinates,shift=0.0):
    # Get max and min values for x, y, z coordinates
    max_values = np.max(coordinates, axis=0)
    min_values = np.min(coordinates, axis=0)

    # Calculate the differences along each axis to determine dimensions
    dimensions = max_values - min_values
    final_dims = dimensions + shift
    return dimensions  # Return the dimensions of the bounding box

# Combien and place 2 fragments
def combine_and_place_fragments(ref_frag, trans_frag):
    
    for displacement in [0.0, 2.0, 4.0, 6.0,8.0,10.0,12.0,14.0]:
        #Translating 2nd frag by displacement in 
        trans_frag.coords[:,-1] +=displacement
        members = get_molecule_members_loop_np2(np.vstack((ref_frag.coords,trans_frag.coords)), ref_frag.elems+trans_frag.elems, 
                                        10, 1.0, 0.4, atomindex=0, membs=None)
        if len(members) == ref_frag.numatoms:
            print("Molecules are sufficiently far apart")
            break
    
    combined_solute = Fragment(elems=ref_frag.elems+trans_frag.elems, coords = np.vstack((ref_frag.coords,trans_frag.coords)), 
                               printlevel=2)
    return combined_solute


#Simple function to combine 2 ASH fragments where one is assumed to be a solute (fewer atoms) and the other assumed to be
#some kind of solvent system (box,sphere etc.)
#Use tolerance (tol) e.g. to control how many solvent molecules around get deleted
#Currently using 0.4 as default based on threonine in acetonitrile example
def insert_solute_into_solvent(solute=None, solute2=None, solvent=None, scale=1.0, tol=0.4, write_pdb=False, write_solute_connectivity=True,
                                       solute_pdb=None, solute2_pdb=None, solvent_pdb=None, outputname="solution.pdb", write_PBC_info=True):
    print("\ninsert_solute_into_solvent\n")
    #Early exits
    if write_pdb:
        print("Write PDB option is active.")
        if solute_pdb is None or solvent_pdb is None:
            print("Error: write_pdb is active but no input solute_pdb or solvent_pdb files were provided")
            ashexit()
    if solute is None and solute_pdb is not None:
        print("No solute fragment provided but solute_pdb is set. Reading solute fragment from PDB-file")
        solute = Fragment(pdbfile=solute_pdb)
    if solute2 is None and solute2_pdb is not None:
        print("No solute2 fragment provided but solute2_pdb is set. Reading solute2 fragment from PDB-file")
        solute2 = Fragment(pdbfile=solute2_pdb)
    if solvent is None and solvent_pdb is not None:
        print("No solvent fragment provided but solvent_pdb is set. Reading solvent fragment from PDB-file")
        solvent = Fragment(pdbfile=solvent_pdb)

    #Get centers
    com_box = solvent.get_coordinate_center()

    if solute2 is  None:
        com_solute = solute.get_coordinate_center()
        #Translating solute coords
        trans_coord=np.array(com_box)-np.array(com_solute)
        solute.coords=solute.coords+trans_coord
    else:
        #Combine and Translate solute2 so that it is close to solute1
        combined_solute = combine_and_place_fragments(ref_frag=solute, trans_frag=solute2)

        #COM and trans
        com_solute = combined_solute.get_coordinate_center()
        trans_coord=np.array(com_box)-np.array(com_solute)
        #Translate combined solute fragment
        combined_solute.coords=combined_solute.coords+trans_coord


    #New Fragment by combining element lists and coordinates
    if solute2 is None:
        print("Combining solute and solvent")
        new_frag = Fragment(elems=solute.elems+solvent.elems, coords = np.vstack((solute.coords,solvent.coords)), printlevel=2)
    else:
        print("Combining combined_solute and solvent")
        new_frag = Fragment(elems=combined_solute.elems+solvent.elems, coords = np.vstack((combined_solute.coords,solvent.coords)), printlevel=2)

    new_frag.write_xyzfile(xyzfilename="solution-pre.xyz")
    #Trim by removing clashing atoms
    new_frag.printlevel=1

    #Find atoms connected to solute index 0. Uses scale and tol
    membs = get_molecule_members_loop_np2(new_frag.coords, new_frag.elems, 20, scale, tol, atomindex=0, membs=None)
    delatoms=[]
    for i in membs:
        if i >= solute.numatoms:
            delatoms.append(i)
    print("First delatoms:", delatoms)
    if solute2 is not None:
        membs2 = get_molecule_members_loop_np2(new_frag.coords, new_frag.elems, 20, scale, tol, atomindex=solute.numatoms, membs=None)
        print("membs2:", membs2)
        for j in membs2:
            if j >= solute.numatoms+solute2.numatoms:
                if j not in delatoms:
                    delatoms.append(j)
    print("Final delatoms:", delatoms)
    
    # Deleting
    delatoms.sort(reverse=True)
    print("Found clashing solvent atoms:", delatoms)
    for d in delatoms:
        new_frag.delete_atom(d)
    new_frag.printlevel=2
    print()
    print("Final fragment after removing clashing atoms:")
    new_frag.update_attributes()
    new_frag.write_xyzfile(xyzfilename="solution.xyz")

    #WRITE PDB
    if write_pdb:
        print("Write_PDB is active. Will write PDB-file of solute+solvent system for topology purposes")
        try:
            import openmm.app
        except ImportError:
            print("Error: OpenMM library not found. Please install OpenMM")
            ashexit()

        #PDB-files
        pdb1 = openmm.app.PDBFile(solute_pdb)
        solute_resname= list(pdb1.topology.residues())[0].name
        print("solute_resname:", solute_resname)
        pdb2 = openmm.app.PDBFile(solvent_pdb)
        solvent_box_vectors = pdb2.topology.getPeriodicBoxVectors()
        print("Found PBC vectors in solvent PDB-file:", solvent_box_vectors)

        #Create modeller object
        modeller = openmm.app.Modeller(pdb1.topology, pdb1.positions) #Add pdbfile1
        
        #solute2
        if solute2 is not None:
            print("Adding solute2")
            pdb_solute2 = openmm.app.PDBFile(solute2_pdb)
            solute2_resname= list(pdb_solute2.topology.residues())[0].name
            print("solute2_resname:", solute2_resname)
            modeller.add(pdb_solute2.topology, pdb_solute2.positions) #Add pdbfile2
        print("Adding solvent")
        modeller.add(pdb2.topology, pdb2.positions) #Add pdbfile2


        #Delete clashing atoms from topology
        toDelete = [r for j,r in enumerate(modeller.topology.atoms()) if j in delatoms]
        modeller.delete(toDelete)
        mergedPositions = new_frag.coords

        #Delete solute connectivity if chosen so not printed in PDB
        if write_solute_connectivity is True:
            print("Will write solute connectivity to PDB-file. Necessary for OpenMM topology recognition when bonded MM parameters are used.")
        else:
            print("Will NOT write solute connectivity to PDB-file. Necessary for OpenMM topology recognition when bonded MM parameters are NOT used.")
            print("Num bonds in topology:",  modeller.topology.getNumBonds())
            solute_bonds = [i for i in modeller.topology.bonds() if i[0].residue.name == solute_resname]
            print("Solute bonds:", solute_bonds)
            print("Deleting solute bonds")
            modeller.delete(solute_bonds)
            print("Num bonds in topology:",  modeller.topology.getNumBonds())

        #PBC info
        if write_PBC_info:
            print("write_PBC_info True: Writing PBC to header of PDB-file")
            if solvent_box_vectors is not None:
                print("PBC vectors found in solvent PDB-file:", solvent_box_vectors)
                print("Adding to solution PDB-file")
                modeller.topology.setPeriodicBoxVectors(solvent_box_vectors)

        #Write merged topology and positions to new PDB file
        ash.interfaces.interface_OpenMM.write_pdbfile_openMM(modeller.topology, mergedPositions, outputname)
    return new_frag

#Basic fast function to calculate the Coulomb energy.
#Assumes coords in Angstrom
def nuc_nuc_repulsion(coords, charges):
    charges = np.array(charges)  # Ensure charges is a numpy array
    coords_b = coords * 1.88972612546
    diff = coords_b[:, None, :] - coords_b[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    return 0.5 * np.sum(charges[:, None] * charges[None, :] / distances)

# a: the coordinates b: coordinates for 1 atom
def find_nearest_atom(a,b):
	print("Finding nearest coordinates for chosen coordinates:",b)
	idx_min = np.sum( (a-b)**2, axis=1, keepdims=True).argmin(axis=0)
	idx_min, a[idx_min]
	print("Nearest atom index in coordinates:", idx_min)
	print("Atom coordinates:", a[idx_min])
	return int(idx_min[0]), a[idx_min]

