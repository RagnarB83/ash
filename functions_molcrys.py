import numpy as np
from functions_coords import *
from functions_ORCA import *
from ash import *
import time

# Fragment-type class
class Fragmenttype:
    instances = []
    def __init__(self, name, formulastring, charge=None, mult=None):
        self.Name = name
        self.Formula = formulastring
        self.Atoms = molformulatolist(formulastring)
        self.Elements = uniq(self.Atoms)
        self.Numatoms = len(self.Atoms)
        self.Nuccharge = nucchargelist(self.Atoms)
        self.mass =totmasslist(self.Atoms)
        self.Charge = charge
        self.Mult = mult
        self.fraglist= []
        self.clusterfraglist= []
        #Keeping track of molmoms, voldict in case of DDEC
        self.molmoms=[]
        self.voldict=None

        #Current atom charges defined for fragment. Charges ordered according to something
        self.charges=[]
        #List of lists: All atom charges that have been defined for fragment. First the gasfrag, then from SP-loop etc.
        self.all_atomcharges=[]
        Fragmenttype.instances.append(self.Name)
    def change_name(self, new_name): # note that the first argument is self
        self.name = new_name # access the class attribute with the self keyword
    def add_fraglist(self, atomlist): # note that the first argument is self
        self.fraglist.append(atomlist) # access the class attribute with the self keyword
    def add_clusterfraglist(self, atomlist): # note that the first argument is self
        self.clusterfraglist.append(atomlist) # access the class attribute with the self keyword
        self.flat_clusterfraglist = [item for sublist in self.clusterfraglist for item in sublist]
    def add_charges(self,chargelist):
        self.all_atomcharges.append(chargelist)
        self.charges=chargelist
    def print_infofile(self, filename='fragmenttype-info.txt'):
        print("Printing fragment-type information to disk:", filename)
        with open(filename, 'w') as outfile:
            outfile.write("Name: {} \n".format(self.Name))
            outfile.write("Formula: {} \n".format(self.Formula))
            outfile.write("Atoms: {} \n".format(self.Atoms))
            outfile.write("Elements: {} \n".format(self.Elements))
            outfile.write("Nuccharge: {} \n".format(self.Nuccharge))
            outfile.write("Mass: {} \n".format(self.mass))
            outfile.write("Charge: {} \n".format(self.Charge))
            outfile.write("Mult: {} \n".format(self.Mult))
            outfile.write("\n")
            outfile.write("Current atomcharges: {} \n".format(self.charges))
            outfile.write("\n")
            outfile.write("All atomcharges: {} \n".format(self.all_atomcharges))
            outfile.write("\n")
            outfile.write("Molmoms: {} \n".format(self.molmoms))
            outfile.write("Voldicts: {} \n".format(self.voldict))
            outfile.write("\n")
            for al in self.all_atomcharges:
                outfile.write(' '.join([str(i) for i in al]))
                outfile.write("\n")
            outfile.write("\n")
            outfile.write("\n")
            outfile.write("Cell fraglist: {} \n".format(self.fraglist))
            outfile.write("\n")
            outfile.write("Cluster fraglist: {} \n".format(self.clusterfraglist))
            outfile.write("\n")
            outfile.write("Flat Cluster fraglist: {} \n".format(self.flat_clusterfraglist))




def print_time_rel_and_tot(timestampA,timestampB, modulename=''):
    secsA=time.time()-timestampA
    minsA=secsA/60
    hoursA=minsA/60
    secsB=time.time()-timestampB
    minsB=secsB/60
    hoursB=minsB/60
    print("-------------------------------------------------------------------")
    print("Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes, {:3.1f} hours".format(modulename, secsA, minsA, hoursA ))
    print("Total Walltime: {:3.1f} seconds, {:3.1f} minutes, {:3.1f} hours".format(secsB, minsB, hoursB ))
    print("-------------------------------------------------------------------")

#Extend cell to 3x3x3 (27 cells) so that original cell is in middle
#Loosely based on https://pymolwiki.org/index.php/Supercell
#This is used for fragment identification. Not for cluster-creation
def cell_extend_frag_withcenter(cellvectors, coords,elems):
    numcells=27
    #All permutations for centered 3x3x3 extension
    permutations = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],
                    [0, 0, -1], [0, -1, 0], [-1, 0, 0],
                    [0,1,1],[1,1,0],[1,0,1],
                    [0,-1,-1],[-1,-1,0],[-1,0,-1],
                    [0, -1, 1], [-1, 1, 0], [-1, 0, 1],
                    [0, 1, -1], [1, -1, 0], [1, 0, -1],
                    [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                    [-1, -1, 1], [-1,1,-1], [1,-1,-1], [1, 1, 1], [-1, -1, -1]]


    extended = np.zeros((len(coords) * numcells, 3))
    new_elems = []
    index = 0
    for perm in permutations:
        shift = cellvectors[0:3, 0:3] * perm
        shift = shift[:, 0] + shift[:, 1] + shift[:, 2]
        #print("Permutation:", perm, "shift:", shift)
        for d, el in zip(coords, elems):
            new_pos=d+shift
            extended[index] = new_pos
            new_elems.append(el)
            index+=1
    return extended, new_elems

#FRAGMENT DEFINE
#Define fragment-identity of each atom in unit cell. Updates mainfrag and counterfrag objects
#1. Goes through atom in cell and finds whole fragments and identifies fragment-type
#2. Using 3x3x3 extended cell with original cell in center so that we have no dangling bonds for center unitcell
#3. Find all whole fragments of the atoms in original cell but capped with atoms from extended cell
#4. For fragment-atoms outside original cell, find equivalent atoms in original cell.
#TODO: Skip step1?
def frag_define(orthogcoords,elems,cell_vectors,fragments,cell_angles=None, cell_length=None):
    blankline()
    print(BC.OKBLUE, BC.BOLD,"Frag_Define: Defining fragments of unit cell", BC.END)
    origtime=time.time()
    currtime=time.time()

    # Extend unit cell in all directions with original cell in center,
    # so that we have no dangling bonds for center unitcell
    print("Creating extended (3x3x3) unit cell for fragment identification")
    temp_extended_coords, temp_extended_elems = cell_extend_frag_withcenter(cell_vectors, orthogcoords, elems)
    # Write XYZ-file with orthogonal coordinates for 3x3xcell
    write_xyzfile(temp_extended_elems, temp_extended_coords, "temp_cell_extended_coords")
    #write XTL file for 3x3x3 cell
    #Todo: fix.
    # Need to give write_xtl fractional coords for new cell. Requires o
    #write_xtl([cell_length[0]*3,cell_length[1]*3,cell_length[2]*3], cell_angles, temp_extended_elems, temp_extended_coords, "temp_cell_extended_coords.xtl")


    blankline()
    # 1. Divide unitcell into fragments (distance-based) if whole fragments found
    print(BC.OKBLUE,"Step 1. Dividing original cell into fragments",BC.END)
    systemlist = list(range(0, len(elems)))
    print("Systemlist length:", len(systemlist))
    unassigned = []
    unassigned_formulas = []
    for i in range(len(elems)):

        printdebug("i : ", i)
        members = get_molecule_members_loop_np2(orthogcoords, elems, 99, settings_ash.scale, settings_ash.tol,
                                            atomindex=i)
        printdebug("members:" , members)
        #print("members:", members)
        el_list = [elems[i] for i in members]
        formula=elemlisttoformula(el_list)
        printdebug("el_list:", el_list)
        ncharge = nucchargelist(el_list)
        #print("Found el_list:", el_list)
        printdebug("ncharge : ", ncharge)
        Assign_Flag=False
        for fragment in fragments:
            printdebug("fragment:", fragment)
            if ncharge == fragment.Nuccharge:
                printdebug("ncharge {} is equal to fragment.Nuccharge {} ".format(ncharge, fragment.Nuccharge))
                Assign_Flag = True
                printdebug("Assign_Flag is True!")
                #Only adding members if not already added
                if members not in fragment.fraglist:
                    printdebug("members not already in fragment.fraglist. Adding")
                    fragment.add_fraglist(members)
                    for m in members:
                        try:
                            systemlist.remove(m)
                        except ValueError:
                            continue
        if Assign_Flag == False:
            printdebug("Assign_Flag is False...")
            printdebug("Could not assign members to fragment.")
            # If members list can not be assigned to fragment then we have a boundary-split
            # Assigning to unassigned
            unassigned_formulas.append(formula)
            if members not in unassigned:
                printdebug("members not in unassigned. Adding to unassigned")
                unassigned.append(members)

    for fi, fragment in enumerate(fragments):
        print("Fragment {} ({}) has {} fraglists".format(fi, fragment.Name, len(fragment.fraglist)))
        print(fragment.fraglist)
        print("")
    #Sorting and trimming unassigned list of fragments
    unassigned = np.unique(unassigned).tolist()
    print("Unassigned members", unassigned)
    print("unassigned_formulas:", unassigned_formulas)
    #Systemlist with remaining atoms
    print("systemlist:", systemlist)
    print("Systemlist length:", len(systemlist))
    blankline()
    print_time_rel_and_tot(currtime, origtime)
    currtime=time.time()

    #2.  Using extended cell find connected members of unassigned fragments
    print(BC.OKBLUE,"Step 2. Using extended cell to find connected members of unassigned fragments",BC.END)
    for m in unassigned:
        printdebug("Trying unassigned m : {} ".format(m))
        members = get_molecule_members_loop_np2(temp_extended_coords, temp_extended_elems, 99,
                                                settings_ash.scale, settings_ash.tol, membs=m)
        el_list = [temp_extended_elems[i] for i in members]
        printdebug("members:", members)
        printdebug("el_list:", el_list)
        formula = elemlisttoformula(el_list)
        print("formula:", formula)
        for fragment in fragments:
            printdebug("el_list:", el_list)
            ncharge = nucchargelist(el_list)
            if ncharge == fragment.Nuccharge:
                printdebug("Found match. ncharge is", ncharge)
                if members not in fragment.fraglist:
                    printdebug("List not already there. Adding")
                    fragment.add_fraglist(members)
                    for m in members:
                        try:
                            systemlist.remove(m)
                        except ValueError:
                            continue
            #else:
            #    print("oops WTF!! what is going on")
            #    exit()
    print("")

    # Too many fragments in fragment.fraglist because we are counting every frag with an atom inside the cell.
    #Step 3 will convert atom numbers to atom numbers for identical cell. Then identical frags can be deleted.
    print("After 2nd run:")

    for fi, fragment in enumerate(fragments):
        print("Fragment {} ({}) has {} fraglists".format(fi, fragment.Name, len(fragment.fraglist)))
        print(fragment.fraglist)
        print("")

    print("Systemlist ({}) remaining: {}".format(len(systemlist), systemlist))

    print_time_rel_and_tot(currtime, origtime)
    currtime=time.time()
    #3.  Going through fragment fraglists. Finding atoms that belong to another cell (i.e. large atom index).
    # Finding equivalent atom positions inside original cell
    #Comparing all lists and removing identical lists created by Step 2
    #Updating fraglist list inside fragment object
    #Permutations for 3x3x3 cell
    print(BC.OKBLUE,"Step 3. Finding equivalent positions of extended cell in original cell ",BC.END)

    #The permutations used in extended cell above
    permutations = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                    [0, 0, -1], [0, -1, 0], [-1, 0, 0],
                    [0, 1, 1], [1, 1, 0], [1, 0, 1],
                    [0, -1, -1], [-1, -1, 0], [-1, 0, -1],
                    [0, -1, 1], [-1, 1, 0], [-1, 0, 1],
                    [0, 1, -1], [1, -1, 0], [1, 0, -1],
                    [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                    [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [1, 1, 1], [-1, -1, -1]]
    for fragment in fragments:
        for fragindex, mfrag in enumerate(fragment.fraglist):
            for n, i in enumerate(mfrag):
                if i >= len(elems):
                    for perm in permutations:
                        if perm != [0.0, 0.0, 0.0]:
                            shift = cell_vectors[0:3, 0:3] * perm
                            shift = shift[:, 0] + shift[:, 1] + shift[:, 2]
                            shifted = temp_extended_coords[i] + shift
                            for numb, oc in enumerate(orthogcoords):
                                if abs(sum(shifted) - sum(oc)) < 0.0000001:
                                    mfrag[n] = numb
            sorted_mfrag = sorted(mfrag)
            fragment.fraglist[fragindex] = sorted_mfrag

    print_time_rel_and_tot(currtime, origtime)
    currtime=time.time()

    #Because every fragment with an atom inside original cell in step 2 gets added
    # we have duplicate fragments in fraglists, meaning we have to trim
    all=[]
    for num,fragment in enumerate(fragments):
        fragment.fraglist = sorted([list(x) for x in set(tuple(x) for x in fragment.fraglist)])
        all+=fragment.fraglist
        for frlist in fragment.fraglist:
            print("Fragment {} ({}): {}".format(num, fragment.Name, frlist))
        blankline()

    # Final check whether assignment is complete.
    all=sorted(all)
    all_flat = [item for sublist in all for item in sublist]
    if len(all_flat) != len(orthogcoords):
        print("Number of assigned atoms ({}) not matching number of atoms in cell ({}).".format(len(all_flat), len(orthogcoords)))
        print("Fragment definition incomplete")
        exit()
    else:
        print("Number of assigned atoms ({}) matches number of atoms in cell ({}).".format(len(all_flat), len(orthogcoords)))

    def find_missing(lst):
        return [x for x in range(lst[0], lst[-1] + 1) if x not in lst]
    if find_missing(all_flat) != []:
        print("Missing number in sequence.")
        print("Fragment definition incomplete")
        exit()

    print(BC.OKBLUE,"Frag_define done!",BC.END)


#From Pymol. Not sure if useful
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
    basis[1][2] = (cos(rad[0]) - basis[0][1]*basis[0][2])/basis[1][1]
    basis[2][2] = sqrt(1 - basis[0][2]**2 - basis[1][2]**2)
    edges.append(1.0)
    return basis * edges # numpy.array multiplication!

#Convert cell parameters to cell vectors. Currently only works for orthorhombic, alpha=beta=gamme=90.0
#TODO: Delete
def cellparamtovectors(cell_length,cell_angles):
    if cell_angles[0] == cell_angles[1] and cell_angles[2] == cell_angles[0] and cell_angles[0] == 90.0:
        cell_vectors=[[cell_length[0], 0.0, 0.0], [0.0, cell_length[1], 0.0],[0.0, 0.0, cell_length[2]]]
    else:
        print("Need to finish this")
        exit()
    return cell_vectors

#https://github.com/ghevcoul/coordinateTransform/blob/master/coordinateTransform.py
#Convert from fractional coordinates to orthogonal Cartesian coordinates in Angstrom
#TODO: check if correct

def fract_to_orthogonal(cellvectors, fraccoords):
    print("Inside fract_to_orthogonal")
    # Transposing cell vectors required here (otherwise nonsense for non-orthorhombic cells)
    cellvectors = np.transpose(cellvectors)
    print("Back-transposed cell_vectors used by fract_to_orthogonal:", cellvectors)
    orthog = []
    for i in fraccoords:
        x = i[0]*cellvectors[0][0] + i[1]*cellvectors[1][0] + i[2]*cellvectors[2][0]
        y = i[0]*cellvectors[0][1] + i[1]*cellvectors[1][1] + i[2]*cellvectors[2][1]
        z = i[0]*cellvectors[0][2] + i[1]*cellvectors[1][2] + i[2]*cellvectors[2][2]
        orthog.append([x, y, z])
    return orthog

#Convert from orthogonal coordinates (Å) to fractional Cartesian coordinates
#TODO: Has not been checked for correctness

def orthogonal_to_fractional(cellvectors, orthogcoords):
    print("function not tested")
    exit()
    def det3(mat):
        return ((mat[0][0] * mat[1][1] * mat[2][2]) + (mat[0][1] * mat[1][2] * mat[2][0]) + (
                    mat[0][2] * mat[1][0] * mat[2][1]) - (mat[0][2] * mat[1][1] * mat[2][0]) - (
                            mat[0][1] * mat[1][0] * mat[2][2]) - (mat[0][0] * mat[1][2] * mat[2][1]))

    fract = []
    cellParam=cellvectors
    latCnt = [x[:] for x in [[None] * 3] * 3]
    for a in range(3):
        for b in range(3):
            latCnt[a][b] = cellParam[b][a]
    detLatCnt = det3(latCnt)
    for i in orthogcoords:
        #x = i[0]*cellvectors[0][0] + i[1]*cellvectors[1][0] + i[2]*cellvectors[2][0]
        #y = i[0]*cellvectors[0][1] + i[1]*cellvectors[1][1] + i[2]*cellvectors[2][1]
        #z = i[0]*cellvectors[0][2] + i[1]*cellvectors[1][2] + i[2]*cellvectors[2][2]
        x = (det3([[i[1], latCnt[0][1], latCnt[0][2]], [i[2], latCnt[1][1], latCnt[1][2]],
                      [i[3], latCnt[2][1], latCnt[2][2]]])) / detLatCnt
        y = (det3([[latCnt[0][0], i[1], latCnt[0][2]], [latCnt[1][0], i[2], latCnt[1][2]],
                      [latCnt[2][0], i[3], latCnt[2][2]]])) / detLatCnt
        z = (det3([[latCnt[0][0], latCnt[0][1], i[1]], [latCnt[1][0], latCnt[1][1], i[2]],
                      [latCnt[2][0], latCnt[2][1], i[3]]])) / detLatCnt
        fract.append([x, y, z])
    return fract


#Extend cell in general with original cell in center
#TODO: Make syntax consistent
def cell_extend_frag(cellvectors, coords,elems,cellextpars):
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
    #Removing duplicates and sorting
    permutations = sorted([list(x) for x in set(tuple(x) for x in permutations)],key=lambda x: (abs(x[0]), abs(x[1]), abs(x[2])))
    #permutations = permutations.sort(key=lambda x: x[0])
    printdebug("Num permutations:", len(permutations))
    numcells=np.prod(cellextpars)
    numcells=len(permutations)
    extended = np.zeros((len(coords) * numcells, 3))
    new_elems = []
    index = 0
    for perm in permutations:
        shift = cellvectors[0:3, 0:3] * perm
        shift = shift[:, 0] + shift[:, 1] + shift[:, 2]
        #print("Permutation:", perm, "shift:", shift)
        for d, el in zip(coords, elems):
            new_pos=d+shift
            extended[index] = new_pos
            new_elems.append(el)
            #print("extended[index]", extended[index])
            #print("extended[index+1]", extended[index+1])
            index+=1
    printdebug("extended coords num", len(extended))
    printdebug("new_elems  num,", len(new_elems))
    return extended, new_elems

#Extend cell in all 3 directions.
#Note: original cell is not in center
#Loosely based on https://pymolwiki.org/index.php/Supercell
#TODO: Delete
def old_cell_extend_frag(cellvectors, coords,elems,cellextpars):
    print("don't use")
    exit()
    printdebug("cellextpars:", cellextpars)
    numcells=np.prod(cellextpars)
    # cellextpars: e.g. [2,2,2]
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
    #Removing duplicates and sorting
    permutations = sorted([list(x) for x in set(tuple(x) for x in permutations)],key=lambda x: (abs(x[0]), abs(x[1]), abs(x[2])))
    #permutations = permutations.sort(key=lambda x: x[0])
    print(permutations)
    print(len(permutations))
    exit()
    #print("permutations:", permutations)
    #print("cellvectors:", cellvectors)
    extended = np.zeros((len(coords) * numcells, 3))
    new_elems = []
    index = 0
    for perm in permutations:
        shift = cellvectors[0:3, 0:3] * perm
        shift = shift[:, 0] + shift[:, 1] + shift[:, 2]
        #print("Permutation:", perm, "shift:", shift)
        for d, el in zip(coords, elems):
            new_pos=d+shift
            extended[index] = new_pos
            new_elems.append(el)
            #print("extended[index]", extended[index])
            #print("extended[index+1]", extended[index+1])
            index+=1
    printdebug(extended)
    printdebug("extended coords num", len(extended))
    printdebug("new_elems  num,", len(new_elems))
    return extended, new_elems


#Read XTL file. XTL file should contain fractional coordinates o
#Grab coordinates, cell parameters
def read_xtlfile(file):
    grabcell = False
    grabfract = False
    coords=[]
    elems=[]
    with open(file) as f:
        for line in f:
            if 'EOF' in line:
                grabfract = False
            if grabcell==True:
                cell_a = float(line.split()[0])
                cell_b = float(line.split()[1])
                cell_c = float(line.split()[2])
                cell_alpha = float(line.split()[3])
                cell_beta = float(line.split()[4])
                cell_gamma = float(line.split()[5])
                grabcell=False
            if 'CELL' in line:
                grabcell=True
            if grabfract==True:
                x_coord = float(line.split()[1])
                y_coord = float(line.split()[2])
                z_coord = float(line.split()[3])
                #XTL file from VESTA can contain coordinates outside unitcell
                if x_coord < 0.0 or y_coord < 0.0 or z_coord < 0.0:
                    print("Skipping fractline in XTL file (outside cell): {} {} {}".format(x_coord,y_coord,z_coord))
                elif x_coord > 1.0 or y_coord > 1.0 or z_coord > 1.0:
                    print("Skipping fractline in XTL file (outside cell): {} {} {}".format(x_coord,y_coord,z_coord))
                else:
                    elems.append(line.split()[0])
                    coords.append([x_coord,y_coord,z_coord])

                #coords.append([float(line.split()[1]), float(line.split()[2]),
                #               float(line.split()[3])])

            if 'NAME         X           Y           Z' in line:
                grabfract=True
    #TODO: Skip lines with negative fractional coords or larger than 1.0
    return [cell_a, cell_b, cell_c],[cell_alpha, cell_beta, cell_gamma],elems,coords
#Read CIF_file
#Grab coordinates, cell parameters and symmetry operations
def read_ciffile(file):
    cell_a=0;cell_b=0;cell_c=0;cell_alpha=0;cell_beta=0;cell_gamma=0
    atomlabels=[]
    elems=[]
    firstcolumn=[]
    secondcolumn=[]
    coords=[]
    symmops=[]
    newmol=False
    fractgrab=False
    symmopgrab=False
    symmopgrab_oldsyntax=False
    cellunits=None
    with open(file) as f:
        for line in f:
            if 'cell_formula_units_Z' in line:
                cellunits=int(line.split()[-1])
                print("Found {} cell_formula_units_Z in CIF file".format(cellunits))
            if 'loop_' in line:
                fractgrab=False
            if newmol==True:
                if '_cell_length_a' in line:
                    cell_a=float(line.split()[-1].split('(')[0])
                if '_cell_length_b' in line:
                    cell_b=float(line.split()[-1].split('(')[0])
                if '_cell_length_c' in line:
                    cell_c =float(line.split()[-1].split('(')[0])
                if '_cell_angle_alpha' in line:
                    cell_alpha=float(line.split()[-1].split('(')[0])
                if '_cell_angle_beta' in line:
                    cell_beta=float(line.split()[-1].split('(')[0])
                if '_cell_angle_gamma' in line:
                        cell_gamma =float(line.split()[-1].split('(')[0])
            #symmops
            if symmopgrab==True:
                if 'space' not in line and len(line) > 2:
                    if 'x' in line:
                        symmops.append(line.split('\'')[1])
                if len(line) < 2:
                    symmopgrab=False
                if 'x' not in line:
                    symmopgrab=False
            if symmopgrab_oldsyntax==True:
                if 'x' not in line:
                    symmopgrab_oldsyntax=False
                else:
                    line2=line.replace("'","").replace(" ","").replace("\n","")
                    symmops.append(line2)
            if fractgrab == True:
                #If empty line encountered then coordinates-lines should be over
                if len(line.replace(' ','')) < 2:
                    fractgrab=False
                    print("Found all coordinates")
                elif '_atom_site' not in line:
                    if 'loop' not in line:
                        atomlabels.append(line.split()[0])
                        #Disabling since not always elems in column
                        secondcolumn.append(line.split()[1])
                        x_coord=float(line.split()[2].split('(')[0])
                        y_coord=float(line.split()[3].split('(')[0])
                        z_coord=float(line.split()[4].split('(')[0])
                        coords.append([x_coord, y_coord, z_coord])
            if 'data_' in line:
                newmol = True
            if '_atom_site_fract_z' in line:
                fractgrab=True
                print("Grabbing coordinates")
            if '_space_group_s' in line:
                symmopgrab=True
            if '_symmetry_equiv_pos_as_xyz' in line:
                symmopgrab_oldsyntax=True

    #Removing any numbers from atomlabels in order to get element information
    for atomlabel in atomlabels:
        el = ''.join([i for i in atomlabel if not i.isdigit()])
        firstcolumn.append(el)

    #Checking if first or second column contains strings that are real periodic-table elements
    if isElementList(firstcolumn):
        print("Found correct elements in 1st column")
        elems=firstcolumn
    else:
        if isElementList(secondcolumn):
            print("Found correct elements in 2nd column")
            elems = secondcolumn
        else:
            print("Found no valid element list from CIF file in either 1st or 2nd column. Check CIF-file format")
            print("firstcolumn: ", firstcolumn)
            print("secondcolumn: ", secondcolumn)
            exit()

    print("Symmetry operations found in CIF:", symmops)
    print("coords : ", coords)
    return [cell_a, cell_b, cell_c],[cell_alpha, cell_beta, cell_gamma],atomlabels,elems,coords,symmops,cellunits

#From cell parameters, fractional coordinates of asymmetric unit and symmetry operations
#create fractional coordinates for atoms of whole cell
def fill_unitcell(cell_length,cell_angles,atomlabels,elems,coords,symmops):
    fullcell=[]
    for i in symmops:
        print("symmop i:", i)
        operations_x=[];operations_y=[];operations_z=[]
        #Multoperations are unity by default. Sumoperations are 0 by default
        multoperation_x=1;sumoperation_x=0
        multoperation_y=1;sumoperation_y=0
        multoperation_z=1;sumoperation_z=0
        #Splitting by comma
        op_x=i.split(',')[0].replace(",","")
        op_y=i.split(',')[1].replace(",","")
        op_z = i.split(',')[2].replace(",", "")
        #op_z=i.split(',')[2].replace(",","").replace(" ","").replace("\n","")
        if len(op_x)==1 and len(op_y)==1 and len(op_z)==1:
            for c in coords:
                if c[0] < 0:
                    cnew_x=1+c[0]
                else:
                    cnew_x=c[0]
                if c[1] < 0:
                    cnew_y=1+c[1]
                else:
                    cnew_y=c[1]
                if c[2] < 0:
                    cnew_z = 1 + c[2]
                else:
                    cnew_z = c[2]
                fullcell.append([cnew_x,cnew_y,cnew_z])
        else:
            op_x_split=op_x.split('x')
            op_y_split = op_y.split('y')
            op_z_split = op_z.split('z')
            for xj in op_x_split:
                if len(xj) > 0:
                    if xj =='-':
                        multoperation_x=-1
                    elif xj == '+1/2':
                        sumoperation_x = 0.5
                    elif xj == '-1/2':
                        sumoperation_x = -0.5
            for yj in op_y_split:
                if len(yj) > 0:
                    if yj =='-':
                        multoperation_y=-1
                    elif yj == '+1/2':
                        sumoperation_y = 0.5
                    elif yj == '-1/2':
                        sumoperation_y = -0.5
            for zj in op_z_split:
                if len(zj) > 0:
                    if zj =='-':
                        multoperation_z=-1
                        printdebug("multoperation_z:", multoperation_z)
                    elif zj == '+1/2':
                        sumoperation_z = 0.5
                        printdebug("sumoperation_z:", sumoperation_z)
                    elif zj == '-1/2':
                        sumoperation_z = -0.5
                        printdebug("sumoperation_z:", sumoperation_z)
            for c in coords:
                c_new=[multoperation_x*c[0]+sumoperation_x,multoperation_y*c[1]+sumoperation_y,multoperation_z*c[2]+sumoperation_z]
                #Translating coordinates so always positive
                if c_new[0] < 0:
                    cnew_x=1+c_new[0]
                else:
                    cnew_x=c_new[0]
                if c_new[1] < 0:
                    cnew_y=1+c_new[1]
                else:
                    cnew_y=c_new[1]
                if c_new[2] < 0:
                    cnew_z = 1 + c_new[2]
                else:
                    cnew_z = c_new[2]
                fullcell.append([cnew_x,cnew_y,cnew_z])
    return fullcell,elems*len(symmops)


#Write fraction coordinate file for cell. Can be opened in VESTA
def write_xtl(cell_lengths,cell_angles,elems,fullcellcoords,outfile):
    """
TITLE c24 h45 fe2 n5 o4 s2
CELL
 11.996600  14.297700  18.164499  90.000000  90.000000  90.000000
SYMMETRY NUMBER 1
SYMMETRY LABEL  P1
ATOMS
NAME         X           Y           Z
"""
    sep='   '
    with open(outfile, 'w') as ofile:
        ofile.write("TITLE Fractional coordinates for whole cell\n")
        ofile.write("CELL\n")
        ofile.write(str(cell_lengths[0])+' '+str(cell_lengths[1])+' '+str(cell_lengths[2])+' '+str(cell_angles[0])+' '+str(cell_angles[1])+' '+str(cell_angles[2])+'\n')
        ofile.write("SYMMETRY NUMBER 1\n")
        ofile.write("SYMMETRY LABEL P1\n")
        ofile.write("ATOMS\n")
        ofile.write("NAME         X           Y           Z\n")
        for i,j in zip(elems,fullcellcoords):
            #ofile.write(i+sep+str(j[0])+sep+str(j[1])+sep+str(j[2])+'\n')
            ofile.write("{:4} {:12.6f} {:12.6f} {:12.6f}\n".format(i, j[0], j[1], j[2]))
        ofile.write("EOF\n")
        print("Wrote fractional coordinates to XTL file:", outfile, "(open with VESTA)")



def create_MMcluster(orthogcoords,elems,cell_vectors,sphereradius):
    print("Creating MM cluster-sphere with radius {} Å".format(sphereradius))
    print("Extending MM unit cell")
    largest_cell_length=np.amax(cell_vectors)
    print("Largest_cell_length: {} Å".format(largest_cell_length))
    #Simple equation to find out roughly how large the extended cell has to be to accommodate cluster-radius
    #Rounds up.
    #Added extra cell
    cell_expansion=math.ceil(sphereradius/largest_cell_length)+1
    print("Using cell expansion: [{},{},{}]".format(cell_expansion,cell_expansion,cell_expansion))
    extended_coords,extended_elems=cell_extend_frag(cell_vectors,orthogcoords,elems,[cell_expansion,cell_expansion,cell_expansion])
    #Write XYZ-file with orthogonal coordinates for cell
    write_xyzfile(extended_elems,extended_coords,"cell_extended_coords")
    printdebug("after extended cell")
    printdebug(len(extended_coords))
    printdebug(len(extended_elems))
    deletionlist=[]
    origin=np.array([0.0,0.0,0.0])
    comparecoords = np.tile(origin, (len(extended_coords), 1))
    print("Now cutting spherical cluster with radius {} Å from super-cell".format(sphereradius))
    # Einsum is slightly faster than bare_numpy_mat. All distances in one go
    distances = einsum_mat(extended_coords, comparecoords)
    #This for loop goes over 112504 count!!! not the best for mol-members
    for count in range(len(extended_coords)):
        #print("count:", count)
        if distances[count] > sphereradius:
            deletionlist.append(count)
    #Deleting atoms in deletion list in reverse
    extended_coords=np.delete(extended_coords, list(reversed(deletionlist)), 0)
    for d in reversed(deletionlist):
        del extended_elems[d]
    #Write XYZ-file
    write_xyzfile(extended_elems,extended_coords,"trimmedcell_extended_coords")
    return extended_coords,extended_elems

#Remove partial fragments of MM cluster
def remove_partial_fragments(coords,elems,sphereradius,fragmentobjects):
    print("Removing partial fragments from MM cluster")
    #Finding surfaceatoms
    origin=np.array([0.0,0.0,0.0])
    comparecoords = np.tile(origin, (len(coords), 1))
    distances = einsum_mat(coords, comparecoords)
    #ForFe2dimer: good values: 4.5 (5.8min), 5 (6min),6 (6.5min), 10(9.2Min)
    #Bad: 4 (5.4 min)
    thickness=5.0
    radius=sphereradius-thickness
    surfaceatoms=np.where(distances>radius)[0].tolist()
    print("Found {} surfaceatoms for outer shell of {} Å".format(len(surfaceatoms),thickness))
    #Todo: remove?
    #with open('surfaceatoms', 'w') as sfile:
    #    sfile.write('Surfaceatoms: {}'.format(surfaceatoms))
    counted=[]
    count=0
    found_atoms=[]
    fraglist=[]
    for surfaceatom in surfaceatoms:
        if surfaceatom not in found_atoms:
            count+=1
            #counted.append(surfaceatom)
            #cProfile.run('get_molecule_members_loop_np(coords, elems, 99, settings_molcrys.scale, settings_ash.tol,atomindex=surfaceatom)')
            #exit()
            #surfaceatom=0
            members=get_molecule_members_loop_np2(coords, elems, 99, settings_ash.scale,
                                                settings_ash.tol,atomindex=surfaceatom)
            #print_time_rel_and_tot(currtime, origtime)
            #currtime = time.time()
            #exit()
            if members not in fraglist:
                fraglist.append(members)
                found_atoms+=members

    #with open('fraglist', 'w') as gfile:
    #    gfile.write('fraglist: {}'.format(fraglist))
    flat_fraglist = [item for sublist in fraglist for item in sublist]
    #Todo: remove?
    with open('foundatoms', 'w') as ffile:
        ffile.write('found_atoms: {}'.format(found_atoms))
    print("len(found_atoms)", len(found_atoms))
    print("len(flat_fraglist)", len(flat_fraglist))
    print("final counted atoms:", count)

    #Going through found frags. If nuccharge of frag does not match known nuccharge it goes to deletionlist
    nuccharges=[fragmentobject.Nuccharge for fragmentobject in fragmentobjects]
    #18June 2020 update. Adding masses as another discriminator.
    masses=[fragmentobject.mass for fragmentobject in fragmentobjects]
    deletionlist=[]
    for frag in fraglist:
        el_list = [elems[i] for i in frag]
        ncharge = nucchargelist(el_list)
        mass = totmasslist(el_list)

        #Checking if valid nuccharge for fragment
        if ncharge in nuccharges:
            #Checking also if valid mass for fragment by subtracting agains known masses
            #Threshold is 0.1
            massdiffs = [abs(mass - i) for i in masses]
            if any(i <= 0.1 for i in massdiffs) is True:
                pass
            else:
                deletionlist += frag
        else:
            deletionlist+=frag

    #print("deletionlist({}(: {}".format(len(deletionlist),deletionlist))
    #with open('deletionlist', 'w') as dfile:
    #    dfile.write('deletionlist: {}'.format(deletionlist))
    deletionlist=np.unique(deletionlist).tolist()
    #print("Sorted deletionlist({}(: {}".format(len(deletionlist),deletionlist))
    #with open('sdeletionlist', 'w') as sdfile:
    #    sdfile.write('sorted deletionlist: {}'.format(deletionlist))
    #Deleting atoms in deletion list in reverse
    coords=np.delete(coords, list(reversed(deletionlist)), 0)
    for d in reversed(deletionlist):
        del elems[d]

    return coords,elems

#Updating pointcharges of fragment
def reordercluster(fragment,fragmenttype):
    print("Reordering Cluster fraglists")
    #print("fragment:", fragment)
    #print("fragmenttype:", fragmenttype)
    fraglists=fragmenttype.clusterfraglist
    #print("fraglists:", fraglists)
    frag_ref=fraglists[0]
    elems_frag_ref = np.array([fragment.elems[i] for i in fraglists[0]])
    coords_frag_ref = np.array([fragment.coords[i] for i in fraglists[0]])
    #print("Reference frag 0:", frag_ref)
    #print("elems_frag_ref:", elems_frag_ref)
    #print("coords_frag_ref:", coords_frag_ref)
    print_coords_all(coords_frag_ref, elems_frag_ref)
    #print("-----------")
    #print("length fraglists", len(fraglists))
    #print("fraglists:", fraglists)
    for fragindex,frag in enumerate(fraglists):
        #print("fragindex:", fragindex)
        #print("frag:", frag)
        if fragindex > 0:
            #print("frag:", frag)
            elems_frag=np.array([fragment.elems[i] for i in frag])
            coords_frag = np.array([fragment.coords[i] for i in frag])
            #print_coords_all(coords_frag,elems_frag)
            #print("elems_frag:", elems_frag)
            #print("coords frag:", coords_frag)
            order = reorder(reorder_hungarian, coords_frag_ref, coords_frag,
                            elems_frag_ref, elems_frag)
            #print("order:", order)
            #Using order list reshuffle frag:
            neworderfrag=[frag[i] for i in order]
            #print("neworderfrag:", neworderfrag)
            fragmenttype.clusterfraglist[fragindex]=neworderfrag
            #blankline()
            #elems_frag_new=np.array([fragment.elems[i] for i in neworderfrag])
            #coords_frag_new = np.array([fragment.coords[i] for i in neworderfrag])
            #print_coords_all(coords_frag_new, elems_frag_new)
            #exit()

#Updating pointcharges of fragment
def pointchargeupdate(fragment,fragmenttype,chargelist):
    #print("fragment:", fragment)
    #print("fragmenttype:", fragmenttype)
    #print("chargelist:", chargelist)
    fraglists=fragmenttype.clusterfraglist
    #print("fraglists:", fraglists)
    #Assemble full list of charges, updated and non-updated.
    oldfullchargelist=fragment.atomcharges
    #print("oldfullchargelist:", oldfullchargelist)
    #If chargelist has not been defined, create empty list with correct size
    if len(oldfullchargelist) == 0:
        oldfullchargelist=[None]*fragment.numatoms
        #print("oldfullchargelist:", oldfullchargelist)
    for frag in fraglists:
        #print("frag:", frag)
        elems_frag=[fragment.elems[i] for i in frag]
        #print("elems_frag:", elems_frag)
        for atomindex,charge in zip(frag,chargelist):
            #print("atomindex:", atomindex)
            #print("charge:", charge)
            oldfullchargelist[atomindex]=charge
            #print("oldfullchargelist:", oldfullchargelist)
    #Finally update full chargelist in fragment
    fragment.update_atomcharges(oldfullchargelist)
    #print("Atomcharges of object")
    #print(fragment.atomcharges)


#Calculate atomic charges for each fragment of Cluster. Assign charges to Cluster object via pointchargeupdate
# TODO: In future also calculate LJ parameters here
#ORCA-specific function
def gasfragcalc_ORCA(fragmentobjects,Cluster,chargemodel,orcadir,orcasimpleinput,orcablocks,NUMPROC,
                     brokensym=None, HSmult=None, atomstoflip=None, shortrangemodel=None):
    blankline()
    print("Now calculating atom charges for each fragment type in cluster")
    for id, fragmentobject in enumerate(fragmentobjects):
        blankline()
        print("Fragmentobject:", fragmentobject.Name)
        #Charge-model info to add to inputfile
        chargemodelline = chargemodel_select(chargemodel)

        #Call Clusterfragment and have print/write/something out coords and elems for atoms in list [0,1,2,3 etc.]
        atomlist=fragmentobject.clusterfraglist[0]
        fragcoords,fragelems=Cluster.get_coords_for_atoms(atomlist)
        write_xyzfile(fragelems, fragcoords, "fragment")
        gasfrag=Fragment(coords=fragcoords,elems=fragelems)

        print("Defined gasfrag:", gasfrag)
        print(gasfrag.__dict__)
        #Creating ORCA theory object with fragment

        #Assuming mainfrag is fragmentobject 0 and only mainfrag can be Broken-symmetry
        if id == 0:
            if brokensym==True:
                ORCASPcalculation = ORCATheory(orcadir=orcadir, fragment=gasfrag, charge=fragmentobject.Charge,
                                       mult=fragmentobject.Mult, orcasimpleinput=orcasimpleinput,
                                       orcablocks=orcablocks, extraline=chargemodelline, brokensym=brokensym, HSmult=HSmult, atomstoflip=atomstoflip)
            else:
                ORCASPcalculation = ORCATheory(orcadir=orcadir, fragment=gasfrag, charge=fragmentobject.Charge,
                                       mult=fragmentobject.Mult, orcasimpleinput=orcasimpleinput,
                                       orcablocks=orcablocks, extraline=chargemodelline)
        else:
            ORCASPcalculation = ORCATheory(orcadir=orcadir, fragment=gasfrag, charge=fragmentobject.Charge,
                                           mult=fragmentobject.Mult, orcasimpleinput=orcasimpleinput,
                                           orcablocks=orcablocks, extraline=chargemodelline)
        print("ORCASPcalculation:", ORCASPcalculation)
        print(ORCASPcalculation.__dict__)
        #Run ORCA calculation with charge-model info
        ORCASPcalculation.run(nprocs=NUMPROC)


        if chargemodel == 'DDEC3' or chargemodel == 'DDEC6':
            #Calling DDEC_calc (calls chargemol)
            atomcharges, molmoms, voldict = DDEC_calc(elems=gasfrag.elems, theory=ORCASPcalculation,
                                            ncores=NUMPROC, DDECmodel=chargemodel,molecule_spinmult=fragmentobject.Mult,
                                            calcdir="DDEC_fragment"+str(id), gbwfile="orca-input.gbw")

            print("atomcharges:", atomcharges)
            #NOTE: We are not going to derive DDEC LJ parameters here but rather at end of SP loop.
        else:
            #Grab atomic charges for fragment.
            atomcharges=grabatomcharges_ORCA(chargemodel,ORCASPcalculation.inputfilename+'.out')


        print("Elements:", gasfrag.elems)
        print("Gasloop atomcharges:", atomcharges)
        assert len(atomcharges) != 0, "Atomcharges list is empty. Something went wrong with grabbing charges"

        #Updating charges inside mainfrag/counterfrag object
        fragmentobject.add_charges(atomcharges)
        #Assign pointcharges to each atom of MM cluster.
        pointchargeupdate(Cluster,fragmentobject,atomcharges)
        #Keep backup of ORCA outputfile
        shutil.copy(ORCASPcalculation.inputfilename + '.out', fragmentobject.Name + '.out')
        shutil.copyfile(ORCASPcalculation.inputfilename + '.out', './SPloop-files/'+fragmentobject.Name+'-Gascalc' + '.out')

        if id ==0:
            shutil.copy(ORCASPcalculation.inputfilename + '.gbw', 'lastorbitals.gbw')

        #Clean up ORCA job.
        ORCASPcalculation.cleanup()
        blankline()

#Calculate atomic charges for each fragment of Cluster. Assign charges to Cluster object via pointchargeupdate
# TODO: In future also calculate LJ parameters here
def gasfragcalc_xTB(fragmentobjects,Cluster,chargemodel,xtbdir,xtbmethod,NUMPROC):
    blankline()
    print("Now calculating atom charges for each fragment type in cluster")
    print("Using default xTB charges. Ignoring chargemodel selected")
    for fragmentobject in fragmentobjects:
        blankline()
        print("Fragmentobject:", fragmentobject.Name)
        #Charge-model info to add to inputfile
        chargemodelline = chargemodel_select(chargemodel)

        #Call Clusterfragment and have print/write/something out coords and elems for atoms in list [0,1,2,3 etc.]
        atomlist=fragmentobject.clusterfraglist[0]
        fragcoords,fragelems=Cluster.get_coords_for_atoms(atomlist)
        write_xyzfile(fragelems, fragcoords, "fragment")
        gasfrag=Fragment(coords=fragcoords,elems=fragelems)

        print("Defined gasfrag:", gasfrag)
        print(gasfrag.__dict__)
        #Creating xTB theory object with fragment
        xTBSPcalculation = xTBTheory(xtbdir=xtbdir, fragment=gasfrag, charge=fragmentobject.Charge,
                                   mult=fragmentobject.Mult, xtbmethod=xtbmethod)

        print("xTBSPcalculation:", xTBSPcalculation)
        print(xTBSPcalculation.__dict__)
        #Run xTB calculation with charge-model info
        xTBSPcalculation.run(nprocs=NUMPROC)


        #Grab atomic charges for fragment.

        atomcharges=grabatomcharges_xTB()
        print("Elements:", gasfrag.elems)
        print("Gasloop atomcharges:", atomcharges)
        #Updating charges inside mainfrag/counterfrag object
        fragmentobject.add_charges(atomcharges)
        #Assign pointcharges to each atom of MM cluster.
        pointchargeupdate(Cluster,fragmentobject,atomcharges)
        #Keep backup of xTB outputfiles. Todo:
        #shutil.copy(xTBSPcalculation.inputfilename + '.out', fragmentobject.Name + '.out')
        #shutil.copyfile(xTBSPcalculation.inputfilename + '.out', './SPloop-files/'+fragmentobject.Name+'-Gascalc' + '.out')
        #Clean up xtb job. Todo:
        #xTBSPcalculation.cleanup()
        blankline()

def rmsd_list(listA,listB):
    sumsq = 0.0
    for a, b in zip(listA, listB):
        sumsq += (a-b)**2.0
    return math.sqrt(sumsq/len(listA))
