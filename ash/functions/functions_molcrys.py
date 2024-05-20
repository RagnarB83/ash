import numpy as np
import time
import math
import shutil
import copy

from ash.functions.functions_general import ashexit, blankline,uniq,printdebug,print_time_rel_and_tot,print_time_rel,BC, load_julia_interface
import ash.modules.module_coords
import ash.interfaces.interface_ORCA
import ash.interfaces.interface_xtb
from ash.interfaces.interface_xtb import grabatomcharges_xTB
from ash.modules.module_MM import UFFdict
from ash.functions.functions_elstructure import DDEC_to_LJparameters,DDEC_calc


# Function for getting indices of repeated rows in a 2d numpy array
# Used to find rows to delete
def get_indices_of_repeated_rows(a):

    # List of row-indices to delete
    to_delete = []

    unq, count = np.unique(a, axis=0, return_counts=True)
    repeated_groups = unq[count > 1]

    for repeated_group in repeated_groups:
        repeated_idx = np.argwhere(np.all(a == repeated_group, axis=1))
        #print("repeated_idx:", repeated_idx)
        #print(repeated_idx.ravel())
        for i in repeated_idx.ravel()[1:]:
            to_delete.append(i)
        #print("to_delete:", to_delete)
    return to_delete




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


#Take extended_cell coordinates, check if clashing atoms with oldcell-coordinates
#Delete clashing coordinates in extended part of extendecell coords.
def delete_clashing_atoms(extendedcell,oldcell,extendedelems,oldelems):
    oldcell=np.array(oldcell)
    #Number of decimals to compare
    decimal=8
    deletionlist=[]
    #Going through extendecell-coordinates starting after last index of old-cell
    for i in range(len(oldcell),len(extendedcell)):
        #Find if extended cell coordinates are in old cell. Mark for deletion
        result, = np.where(np.all( oldcell.round(decimals=decimal)== extendedcell[i].round(decimals=decimal), axis=1))
        if result.size >0:
            deletionlist.append(i)

    #Delete all rows in deletionlist and get final cell
    newcell = np.delete(extendedcell,deletionlist,0)

    #Get corresponding elements list
    newelems=[]
    for index,el in enumerate(extendedelems):
        if index not in deletionlist:
            newelems.append(el)

    assert len(newcell) == len(newelems), "something went wrong in delete_clashing_atoms"
    return newcell,newelems

#FRAGMENT DEFINE
#Define fragment-identity of each atom in unit cell. Updates mainfrag and counterfrag objects
#1. Goes through atom in cell and finds whole fragments and identifies fragment-type
#2. Using 3x3x3 extended cell with original cell in center so that we have no dangling bonds for center unitcell
#3. Find all whole fragments of the atoms in original cell but capped with atoms from extended cell
#4. For fragment-atoms outside original cell, find equivalent atoms in original cell.
#TODO: Skip step1?

#TODO. Problem. If atoms are precisely on the edge boundary (very symmetric crystals) then we get atoms on top of each other when we do : cell_extend_frag_withcenter
# Like for CoCN6 and maybe FeCN6 crystals. SOlution?. Delete clashing atoms???

def frag_define(orthogcoords,elems,cell_vectors,fragments,cell_angles=None, cell_length=None, scale=None, tol=None):

    if scale is None:
        scale=ash.settings_ash.settings_dict["scale"]
    if tol is None:
        tol=ash.settings_ash.settings_dict["tol"]

    blankline()
    print(BC.OKBLUE, BC.BOLD,"Frag_Define: Defining fragments of unit cell", BC.END)
    #origtime=time.time()
    #currtime=time.time()

    # Extend unit cell in all directions with original cell in center,
    # so that we have no dangling bonds for center unitcell
    print("Creating extended (3x3x3) unit cell for fragment identification")
    temp_extended_coords, temp_extended_elems = cell_extend_frag_withcenter(cell_vectors, orthogcoords, elems)

    # Write XYZ-file with orthogonal coordinates for 3x3xcell
    ash.modules.module_coords.write_xyzfile(temp_extended_elems, temp_extended_coords, "temp_cell_extended_coords-beforedel")

    #Delete duplicate entries. May happen if atoms are right on boundary
    temp_extended_coords, temp_extended_elems = delete_clashing_atoms(temp_extended_coords,orthogcoords,temp_extended_elems,elems)

    # Write XYZ-file with orthogonal coordinates for 3x3xcell
    ash.modules.module_coords.write_xyzfile(temp_extended_elems, temp_extended_coords, "temp_cell_extended_coords-afterdel")

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

    #Function that checks whether fragment in cell/cluster is the same as defined by user.
    #Metrics: Nuclear charge, Mass, or both.
    #TODO: Need to find even better metric. Something that takes connectivity into account?
    def same_fragment(fragtype=None, nuccharge=None, mass=None, formula=None):
        metric="nuccharge_and_mass"
        printdebug("metric:", metric)
        printdebug("fragtype dict:", fragtype.__dict__)

        printdebug("----------")
        printdebug("nuccharge: {} mass: {} formula: {}".format(nuccharge,mass,formula))
        printdebug("---------")
        if metric=="nuccharge_and_mass":
            printdebug("Nuccharge and mass option!")
            if nuccharge == fragtype.Nuccharge and abs(mass - fragtype.mass) <1.0 :
                printdebug("ncharge {} is equal to fragment.Nuccharge {} ".format(nuccharge, fragtype.Nuccharge))
                printdebug("mass {} is equal to fragtyp.mass {} ".format(mass, fragtype.mass))
                return True
        elif metric=="nuccharge":
            printdebug("Nucharge option!!")
            if nuccharge == fragtype.Nuccharge:
                printdebug("ncharge {} is equal to fragment.Nuccharge {} ".format(nuccharge, fragtype.Nuccharge))
                return True
        elif metric=="mass":
            printdebug("Mass option!")
            if abs(mass - fragtype.mass) <0.1 :
                printdebug("mass {} is equal to fragtyp.mass {} ".format(mass, fragtype.mass))
                return True



    for i in range(len(elems)):

        printdebug("i : ", i)
        members = ash.modules.module_coords.get_molecule_members_loop_np2(orthogcoords, elems, 99, scale, tol,
                                            atomindex=i)
        printdebug("members:" , members)
        #print("members:", members)
        el_list = [elems[i] for i in members]
        formula=ash.modules.module_coords.elemlisttoformula(el_list)
        current_mass = ash.modules.module_coords.totmasslist(el_list)
        ncharge = ash.modules.module_coords.nucchargelist(el_list)
        printdebug("Current_mass:", current_mass)
        printdebug("current formula:", formula)
        printdebug("el_list:", el_list)
        printdebug("ncharge : ", ncharge)
        Assign_Flag=False

        for fragment in fragments:
            printdebug("fragment:", fragment)
            #if ncharge == fragment.Nuccharge:
            if same_fragment(fragtype=fragment, nuccharge=ncharge, mass=current_mass, formula=formula) is True:
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
    unassigned = np.array(unassigned, dtype=object) #Changed due to numpy deprecation warning
    unassigned = np.unique(unassigned).tolist()
    print("Unassigned members size", len(unassigned))
    print("unassigned_formulas:", unassigned_formulas)
    #Systemlist with remaining atoms
    #print("systemlist:", systemlist)
    print("Systemlist length:", len(systemlist))
    blankline()
    #print_time_rel_and_tot(currtime, origtime, modulename='molcrys_frag_define_step1', moduleindex=4)
    #currtime=time.time()

    #2.  Using extended cell find connected members of unassigned fragments
    print(BC.OKBLUE,"Step 2. Using extended cell to find connected members of unassigned fragments",BC.END)
    for m in unassigned:
        printdebug("Trying unassigned m : {} ".format(m))
        members = ash.modules.module_coords.get_molecule_members_loop_np2(temp_extended_coords, temp_extended_elems, 99,
                                                scale, tol, membs=m)
        el_list = [temp_extended_elems[i] for i in members]
        current_mass = ash.modules.module_coords.totmasslist(el_list)
        printdebug("members:", members)
        printdebug("el_list:", el_list)
        printdebug("current_mass:", current_mass)
        formula = ash.modules.module_coords.elemlisttoformula(el_list)
        print("formula:", formula)
        for fragment in fragments:
            printdebug("el_list:", el_list)
            ncharge = ash.modules.module_coords.nucchargelist(el_list)
            #if ncharge == fragment.Nuccharge:
            if same_fragment(fragtype=fragment, nuccharge=ncharge, mass=current_mass, formula=formula) is True:
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
            #    ashexit()
    print("")

    # Too many fragments in fragment.fraglist because we are counting every frag with an atom inside the cell.
    #Step 3 will convert atom numbers to atom numbers for identical cell. Then identical frags can be deleted.
    print("After 2nd run:")

    for fi, fragment in enumerate(fragments):
        print("Fragment {} ({}) has {} fraglists".format(fi, fragment.Name, len(fragment.fraglist)))
        print(fragment.fraglist)
        print("")

    print("Systemlist ({}) remaining: {}".format(len(systemlist), systemlist))

    #print_time_rel_and_tot(currtime, origtime, modulename='molcrys_frag_define_step2', moduleindex=4)
    #currtime=time.time()

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

    #print_time_rel_and_tot(currtime, origtime, modulename='molcrys_frag_define_step3a', moduleindex=4)
    #currtime=time.time()

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

        def find_missing(lst):
            return [x for x in range(lst[0], lst[-1] + 1) if x not in lst]

        if len(all_flat) > 0:
            if find_missing(all_flat) != []:
                print("Missing number in sequence.")
                print("Fragment definition incomplete")

        return 1
    else:
        print("Number of assigned atoms ({}) matches number of atoms in cell ({}).".format(len(all_flat), len(orthogcoords)))
        return 0




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
    print("cell_length :", cell_length)
    print("cell_angles :", cell_angles)
    if cell_angles[0] == cell_angles[1] and cell_angles[2] == cell_angles[0] and cell_angles[0] == 90.0:
        cell_vectors=[[cell_length[0], 0.0, 0.0], [0.0, cell_length[1], 0.0],[0.0, 0.0, cell_length[2]]]
    else:
        print("Need to finish this")
        ashexit()
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
    ashexit()
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
        x = (det3([[i[0], latCnt[0][1], latCnt[0][2]], [i[1], latCnt[1][1], latCnt[1][2]],
                      [i[2], latCnt[2][1], latCnt[2][2]]])) / detLatCnt
        y = (det3([[latCnt[0][0], i[0], latCnt[0][2]], [latCnt[1][0], i[1], latCnt[1][2]],
                      [latCnt[2][0], i[2], latCnt[2][2]]])) / detLatCnt
        z = (det3([[latCnt[0][0], latCnt[0][1], i[0]], [latCnt[1][0], latCnt[1][1], i[1]],
                      [latCnt[2][0], latCnt[2][1], i[2]]])) / detLatCnt
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


#Simple super-cellexpansion
#def simple_cell_extend_frag(cellvectors, coords,elems):
#    permutations=[[0,0,0],[0,0,1]]
#    numcells=2
#    extended = np.zeros((len(coords) * numcells, 3))
#    new_elems = []
#    index = 0
#    for perm in permutations:
#        shift = cellvectors[0:3, 0:3] * perm
#        shift = shift[:, 0] + shift[:, 1] + shift[:, 2]
#        #print("Permutation:", perm, "shift:", shift)
#        for d, el in zip(coords, elems):
#            new_pos=d+shift
#            extended[index] = new_pos
#            new_elems.append(el)
#            #print("extended[index]", extended[index])
#            #print("extended[index+1]", extended[index+1])
#            index+=1
#    printdebug(extended)
#    printdebug("extended coords num", len(extended))
#    printdebug("new_elems  num,", len(new_elems))
#    return extended, new_elems

#Extend cell in all 3 directions.
#Note: original cell is not in center
#Loosely based on https://pymolwiki.org/index.php/Supercell
#TODO: Delete
def old_cell_extend_frag(cellvectors, coords,elems,cellextpars):
    print("don't use")
    ashexit()
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
    ashexit()
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

                #Now checking each coordinate if it is outside unitcell
                coord_mod = shift_fract_coord([x_coord,y_coord,z_coord])
                elems.append(line.split()[0])
                coords.append(coord_mod)
                #Note: if statements below right now just for printing
                #XTL file can contain coordinates outside unitcell
                if x_coord < 0.0 or y_coord < 0.0 or z_coord < 0.0:
                    print("Warning: Fractional atom coordinate in XTL file may be outside cell: {} {} {}".format(x_coord,y_coord,z_coord))
                    print("Modifying")
                elif x_coord > 1.0 or y_coord > 1.0 or z_coord > 1.0:
                    print("Warning: Fractional atom coordinate in XTL file may be outside cell: {} {} {}".format(x_coord,y_coord,z_coord))
                    print("Modifying")
            if 'NAME         X           Y           Z' in line:
                grabfract=True
    #TODO: Skip lines with negative fractional coords or larger than 1.0
    return [cell_a, cell_b, cell_c],[cell_alpha, cell_beta, cell_gamma],elems,coords

#Function to shift a fractional coordinate so that it is fully inside unit cell
def shift_fract_coord(a, rounddec=5):
    new = []
    #Looping over each coord
    for i in a:
        if i > 1.0:
            mod = round(i-1.0,rounddec)
        elif i < 0.0:
            mod = round(1.0+i,rounddec)
        # Set 1.0 fract coords to 0 (will create duplicates but removed later)
        elif i == 1.0:
            mod = round(0.0, rounddec)
        #No shift required but rounding up number
        else:
            mod=round(i, rounddec)
        new.append(mod)
    return new


# Read CIF_file
# Grab coordinates, cell parameters and symmetry operations
def read_ciffile(file):
    cell_a=0;cell_b=0;cell_c=0;cell_alpha=0;cell_beta=0;cell_gamma=0
    atomlabels=[]
    elems=[]
    firstcolumn=[]
    secondcolumns=[]
    coords=[]
    symmops=[]
    newmol=False
    fractgrab=False
    symmopgrab=False
    symmopgrab_oldsyntax=False
    cellunits=None
    atomsitecolumns=[]
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
                    if 'x' in line or 'X' in line:
                        symmops.append(line.split('\'')[1])
                if len(line) < 2:
                    symmopgrab=False
                if 'x' not in line:
                    if 'X' not in line:
                        symmopgrab=False
            if symmopgrab_oldsyntax==True:
                if 'x' not in line:
                    symmopgrab_oldsyntax=False
                else:
                    #Removing number from beginning if present
                    if line[0].isdigit():
                        line=line[1:]
                    line2=line.replace("'","").replace(" ","").replace("\n","")
                    symmops.append(line2)
            if fractgrab == True:
                #If empty line encountered then coordinates-lines should be over
                if len(line.replace(' ','')) < 2:
                    fractgrab=False
                    print("Found all coordinates")
                elif '#' in line:
                    fractfrab=False
                    print("Found all coordinates")
                elif '_atom_site' not in line:
                    if 'loop' not in line:
                        atomlabelcolumn=int(atomsitecolumns.index("_atom_site_label"))
                        #Grabbing x,y,z columns
                        xcolumn=int(atomsitecolumns.index("_atom_site_fract_x"))
                        ycolumn=int(atomsitecolumns.index("_atom_site_fract_y"))
                        zcolumn=int(atomsitecolumns.index("_atom_site_fract_z"))
                        atomlabels.append(line.split()[atomlabelcolumn])
                        x_coord=float(line.split()[xcolumn].split('(')[0])
                        y_coord=float(line.split()[ycolumn].split('(')[0])
                        z_coord=float(line.split()[zcolumn].split('(')[0])
                        coords.append([x_coord, y_coord, z_coord])

                        #Disabled since reading atomsitecolumns option should be more robust
                        #Disabling since not always elems in column
                        secondcol=line.split()[1]
                        secondcolumns.append(secondcol)
                        #If second-column is proper float then this is fract_x, else trying next
                        #if is_string_float_withdecimal(secondcol.split('(')[0]):
                        #    print(secondcol.split('(')[0])
                         #   x_coord=float(line.split()[1].split('(')[0])
                         #   y_coord=float(line.split()[2].split('(')[0])
                         #   z_coord=float(line.split()[3].split('(')[0])
                         #   coords.append([x_coord, y_coord, z_coord])
                        #else:
                        #    x_coord=float(line.split()[2].split('(')[0])
                        #    y_coord=float(line.split()[3].split('(')[0])
                        #    z_coord=float(line.split()[4].split('(')[0])
                        #    coords.append([x_coord, y_coord, z_coord])
            if 'data_' in line:
                newmol = True
            if '_atom_site_' in line:
                atomsitecolumns.append(line.split()[0])
            if '_atom_site_fract_z' in line:
                fractgrab=True
                print("Grabbing coordinates")
            if '_space_group_s' in line:
                symmopgrab=True
            if '_symmetry_equiv_pos_as_xyz' in line:
                symmopgrab_oldsyntax=True


    #Checking if "_atom_site_symmetry_multiplicity" or "_atom_site_Wyckoff_symbol" in atom_site lines
    #If so then we have more complicated symmettry, not coded yet.
    #TODO: We should finish this at some point
    if '_atom_site_symmetry_multiplicity' in atomsitecolumns or '_atom_site_site_symmetry_multiplicity' in atomsitecolumns:
        print("Warning: site_symmetry_multiplicity information in file. May not be handled correctly. Check results...")
    if '_atom_site_Wyckoff_symbol' in atomsitecolumns:
        print("Wyckoff_symbols found in CIF-file. We do not handle this correctly. Exiting.")
        print("Please use another format than CIF-file, e.g. XTL.")
        ashexit()


    #Removing any numbers from atomlabels in order to get element information
    for atomlabel in atomlabels:
        el = ''.join([i for i in atomlabel if not i.isdigit()])
        firstcolumn.append(el)

    #Checking if first or second column contains strings that are real periodic-table elements
    if ash.modules.module_coords.isElementList(firstcolumn):
        print("Found correct elements in 1st column")
        elems=firstcolumn
    else:
        if ash.modules.module_coords.isElementList(secondcolumns):
            print("Found correct elements in 2nd column")
            elems = secondcolumns
        else:
            print("Found no valid element list from CIF file in either 1st or 2nd column. Check CIF-file format")
            print("firstcolumn: ", firstcolumn)
            print("secondcolumns: ", secondcolumns)
            ashexit()

    print("Symmetry operations found in CIF:", symmops)
    if len(coords) == 0:
        print("Found zero coordinates in CIF-file: {}. Something wrong with file. Exiting...".format(file))
        ashexit()
    print("Cell parameters:", cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma)
    return [cell_a, cell_b, cell_c],[cell_alpha, cell_beta, cell_gamma],atomlabels,elems,coords,symmops,cellunits


#Shift fractional_coordinates by x,y,z amount
def shift_fractcoords(coords,shift):
    newcoords=[]
    for coord in coords:
        coord_x=coord[0]+shift[0]
        coord_y=coord[1]+shift[1]
        coord_z=coord[2]+shift[2]
        if coord_x < 0:
            coord_x=coord_x+1
        if coord_y < 0:
            coord_y=coord_y+1
        if coord_z < 0:
            coord_z=coord_z+1
        if coord_x > 1:
            coord_x=coord_x-1
        if coord_y > 1:
            coord_y=coord_y-1
        if coord_z > 1:
            coord_z=coord_z-1
        newcoords.append([coord_x,coord_y,coord_z])
    return newcoords
#From cell parameters, fractional coordinates of asymmetric unit and symmetry operations
#create fractional coordinates for atoms of whole cell
def fill_unitcell(cell_length,cell_angles,atomlabels,elems,coords,symmops):
    fullcell=[]
    for i in symmops:
        #Making symmop lowercase just in case
        symmop_i=i.lower()
        #print("symmop i:", symmop_i)
        operations_x=[];operations_y=[];operations_z=[]
        #Multoperations are unity by default. Sumoperations are 0 by default
        multoperation_x=1;sumoperation_x=0
        multoperation_y=1;sumoperation_y=0
        multoperation_z=1;sumoperation_z=0
        #Splitting by comma, removing comma and spaces
        op_x=symmop_i.split(',')[0].replace(",","").replace(" ","")
        op_y=symmop_i.split(',')[1].replace(",","").replace(" ","")
        op_z = symmop_i.split(',')[2].replace(",", "").replace(" ","")
        #print("op_x: {} op_y: {} op_z: {}".format(op_x,op_y,op_z))
        #op_z=i.split(',')[2].replace(",","").replace(" ","").replace("\n","")
        if len(op_x)==1 and len(op_y)==1 and len(op_z)==1:
            for c in coords:
                if c[0] < 0:
                    cnew_x=1+c[0]
                elif c[0] == 1.0:
                    cnew_x=0.0
                else:
                    cnew_x=c[0]
                if c[1] < 0:
                    cnew_y=1+c[1]
                elif c[1] == 1.0:
                    cnew_y=0.0
                else:
                    cnew_y=c[1]
                if c[2] < 0:
                    cnew_z = 1 + c[2]
                elif c[2] == 1.0:
                    cnew_z=0.0
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
                #exit()
                #Translating coordinates so always positive
                if c_new[0] < 0:
                    cnew_x=1+c_new[0]
                elif c_new[0] == 1.0 or c_new[0] == -1.0: #If coord is 1.0 then set to 0
                    cnew_x=0.0
                else:
                    cnew_x=c_new[0]
                #
                if c_new[1] < 0:
                    cnew_y=1+c_new[1]
                elif c_new[1] == 1.0 or c_new[1] == -1.0: #If coord is 1.0 then set to 0
                    cnew_y=0.0
                else:
                    cnew_y=c_new[1]
                #
                if c_new[2] < 0:
                    cnew_z = 1 + c_new[2]
                elif c_new[2] == 1.0 or c_new[2] == -1.0: #If coord is 1.0 then set to 0
                    cnew_z=0.0
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



#Filter coords array based on duplicate condition.
#Gives list of duplicate row indices if less than threshold
#Old slow version
def filter_duplicate_old(data):
    def condition(xs,prev):
        threshold=1e-5
        val=sum((x-yp)*(x-yp) for x,yp in zip(xs,prev))
        return val > threshold
    result = []; duplicate_indices=[]
    for rowindex,element in enumerate(data):
        if all(condition(element,previous) for previous in result):
            result.append(element)
        else:
            duplicate_indices.append(rowindex)
    return duplicate_indices

#New fast version
def filter_duplicate(data):
    threshold = 1e-5
    squared_diff = np.sum((data[:, np.newaxis] - data[np.newaxis, :])**2, axis=2)
    np.fill_diagonal(squared_diff, np.inf)  # Set diagonal elements to infinity to exclude self-comparison
    duplicate_indices = np.where(np.any(squared_diff < threshold, axis=1))[0]
    return duplicate_indices


def create_MMcluster(orthogcoords,elems,cell_vectors,sphereradius):
    print("Creating MM cluster-sphere with radius {} Å".format(sphereradius))
    print("Extending MM unit cell")
    largest_cell_length=np.amax(cell_vectors)
    print("Largest_cell_length: {} Å".format(largest_cell_length))
    #Simple equation to find out roughly how large the extended cell has to be to accommodate cluster-radius
    #Rounds up.
    #Added extra cell. Nov 2020: Added another one because of XUQVAI.xtl
    #More  (+5) in 5 APril 2021. NaFeCl4 crystal
    cell_expansion=math.ceil(sphereradius/largest_cell_length)+4
    print("Using cell expansion: [{},{},{}]".format(cell_expansion,cell_expansion,cell_expansion))
    extended_coords,extended_elems=cell_extend_frag(cell_vectors,orthogcoords,elems,[cell_expansion,cell_expansion,cell_expansion])
    #Write XYZ-file with orthogonal coordinates for cell
    ash.modules.module_coords.write_xyzfile(extended_elems,extended_coords,"cell_extended_coords")
    printdebug(len(extended_coords))
    printdebug(len(extended_elems))
    deletionlist=[]
    origin=np.array([0.0,0.0,0.0])
    comparecoords = np.tile(origin, (len(extended_coords), 1))
    print("Now cutting spherical cluster with radius {} Å from super-cell".format(sphereradius))
    # Einsum is slightly faster than bare_numpy_mat.
    # All atom-distances compared to origin in one go
    distances = ash.modules.module_coords.einsum_mat(extended_coords, comparecoords)
    for count in range(len(extended_coords)):
        #print("count:", count)
        if distances[count] > sphereradius:
            deletionlist.append(count)
    #Deleting atoms in deletion list in reverse
    extended_coords=np.delete(extended_coords, list(reversed(deletionlist)), 0)
    for d in reversed(deletionlist):
        del extended_elems[d]

    printdebug(len(extended_coords))
    printdebug(len(extended_elems))

    #Find duplicate coordinates (atoms on top of each other). Add index to deletion list. Happens if atoms have coordinates right on box boundary
    #List of Bools, duplicates are True
    #NOTE: Problem, way too slow
    print("Starting filter duplicate")
    timestampA=time.time()
    dupls=np.array(filter_duplicate(extended_coords))
    print_time_rel(timestampA, modulename='filter duplicate', moduleindex=4)
    #Deleting atoms in duplication list in reverse
    extended_coords=np.delete(extended_coords, list(reversed(dupls)), 0)
    for d in reversed(dupls):
        del extended_elems[d]
    #Write XYZ-file
    ash.modules.module_coords.write_xyzfile(extended_elems,extended_coords,"trimmedcell_extended_coords")
    return extended_coords,extended_elems

#Remove partial fragments of MM cluster
def remove_partial_fragments(coords,elems,sphereradius,fragmentobjects, scale=None, tol=None, codeversion='julia'):
    if scale is None:
        scale=ash.settings_ash.settings_dict["scale"]
    if tol is None:
        tol=ash.settings_ash.settings_dict["tol"]
    print("Removing partial fragments from MM cluster")
    #Finding surfaceatoms
    origin=np.array([0.0,0.0,0.0])
    comparecoords = np.tile(origin, (len(coords), 1))
    distances = ash.modules.module_coords.einsum_mat(coords, comparecoords)
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

    if codeversion=='julia':
        print("Will use julia for finding surface atoms")
        try:
            # Import Julia
            print("Loading Julia")
            Juliafunctions=load_julia_interface()
            print("Load successful")
            #Get list of fragments for all surfaceatoms
            print("Now calling Julia function")
            fraglist_temp = Juliafunctions.calc_fraglist_for_atoms_julia(surfaceatoms,coords, np.array(elems), 99, scale, tol,ash.modules.module_coords.eldict_covrad)
            #TODO: Necessary. Can we change return of Julia function instead??
            fraglist_temp = [list(i) for i in fraglist_temp]

            # Converting from numpy to list of lists
            for sublist in fraglist_temp:
                fraglist.append(list(sublist))
        except:
            print(BC.FAIL, "Problem importing Julia interface or loading function", BC.END)
            print("Make sure Julia is installed and Python-Julia interface has been installed")
            print("")
            print(BC.FAIL, "Using py version instead (slow for large systems)", BC.END)
            for surfaceatom in surfaceatoms:
                if surfaceatom not in found_atoms:
                    count += 1
                    members = ash.modules.module_coords.get_molecule_members_loop_np2(coords, elems, 99, scale, tol, atomindex=surfaceatom)
                    if members not in fraglist:
                        fraglist.append(members)
                        found_atoms += members
    elif codeversion == 'py':
        print("using py for finding surface atoms")
        for surfaceatom in surfaceatoms:
            if surfaceatom not in found_atoms:
                count+=1
                members=ash.modules.module_coords.get_molecule_members_loop_np2(coords, elems, 99, scale, tol,atomindex=surfaceatom)
                if members not in fraglist:
                    fraglist.append(members)
                    found_atoms+=members

    flat_fraglist = [item for sublist in fraglist for item in sublist]
    #Todo: remove?
    with open('foundatoms', 'w') as ffile:
        ffile.write('found_atoms: {}'.format(found_atoms))
    #print("len(found_atoms)", len(found_atoms))
    #print("len(flat_fraglist)", len(flat_fraglist))
    #print("final counted atoms:", count)
    #ashexit()
    #Going through found frags. If nuccharge of frag does not match known nuccharge it goes to deletionlist
    nuccharges=[fragmentobject.Nuccharge for fragmentobject in fragmentobjects]
    #18June 2020 update. Adding masses as another discriminator.
    masses=[fragmentobject.mass for fragmentobject in fragmentobjects]
    deletionlist=[]
    for frag in fraglist:
        el_list = [elems[i] for i in frag]
        ncharge = ash.modules.module_coords.nucchargelist(el_list)
        mass = ash.modules.module_coords.totmasslist(el_list)

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
    deletionlist=np.unique(deletionlist).tolist()
    print(f"Atoms to delete: {len(deletionlist)} atoms")
    #Deleting atoms in deletion list in reverse
    coords=np.delete(coords, list(reversed(deletionlist)), 0)
    for d in reversed(deletionlist):
        del elems[d]
    return coords,elems

#
def reordercluster(fragment,fragmenttype,code_version='py'):
    print("Reordering Cluster fraglists")
    #print("fragment:", fragment)
    #print("fragmenttype:", fragmenttype)
    fraglists=fragmenttype.clusterfraglist
    #fraglists=[[956, 964, 972, 980, 988, 1004, 7644]]



    print("Before reorder")
    #print(fraglists)
    #exit()
    if len(fraglists) == 0:
        print(BC.FAIL, "Fragment lists for fragment-type are empty. Makes no sense (too small cluster radius?!). Exiting...", BC.END)
        ashexit()

    timestampA=time.time()
    if code_version=='julia':
        print("Calling reorder_cluster_julia")
        ashexit()
        try:
            Juliafunctions=load_julia_interface()
        except:
            print("Problem loading Julia")
            ashexit()
        #print(fragmenttype.clusterfraglist[5])
        #Converting from 0-based to 1-based indexing before passing to Julia
        jul_fraglists=[[number+1 for number in group] for group in fraglists]
        new_jul_fraglists = Juliafunctions.reorder_cluster_julia(fragment.elems,fragment.coords,jul_fraglists)
        print("new_jul_fraglists:", new_jul_fraglists)
        #Converting back from 1-based indexing to 0-based indexing
        fragmenttype.clusterfraglist=[[number-1 for number in group] for group in new_jul_fraglists]
        #print("After. fragmenttype.clusterfraglist:", fragmenttype.clusterfraglist)
        #print(fragmenttype.clusterfraglist[236])
        ashexit()
        print_time_rel(timestampA, modulename='reorder_cluster julia', moduleindex=4)
    elif code_version=='py':
        print("Calling reorder_cluster py")
        print("Now trying to importing scipy")
        try:
            import scipy.spatial.distance
            import scipy.optimize
        except:
            print("Problem importing scipy library. This part of the code requires scipy")
            print("Exiting.")
            ashexit()
        #print(fragmenttype.clusterfraglist[5])
        elems_frag_ref = np.array([fragment.elems[i] for i in fraglists[0]])
        coords_frag_ref = np.array([fragment.coords[i] for i in fraglists[0]])
        for fragindex,frag in enumerate(fraglists):
            #print("frag:", frag)
            if fragindex > 0:
                elems_frag=np.array([fragment.elems[i] for i in frag])
                coords_frag = np.array([fragment.coords[i] for i in frag])

                order = ash.modules.module_coords.reorder(ash.modules.module_coords.reorder_hungarian_scipy, coords_frag_ref, coords_frag,
                                elems_frag_ref, elems_frag)

                #Using order list reshuffle frag:
                neworderfrag=[frag[i] for i in order]
                #print("neworderfrag:", neworderfrag)
                fragmenttype.clusterfraglist[fragindex]=neworderfrag

        #print("After. fragmenttype.clusterfraglist:", fragmenttype.clusterfraglist)
        #ashexit()
        #print(fragmenttype.clusterfraglist[236])
        print_time_rel(timestampA, modulename='reorder_cluster py', moduleindex=4)
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
#ORCA-specific function
def gasfragcalc_ORCA(fragmentobjects,Cluster,chargemodel,orcadir,orcasimpleinput,orcablocks,NUMPROC,
                     brokensym=None, HSmult=None, atomstoflip=None):
    blankline()
    origtime=time.time()
    currtime=time.time()
    print(BC.OKBLUE, BC.BOLD, "Now calculating atom charges for each fragment type in cluster", BC.END)
    #print(BC.OKBLUE, BC.BOLD, "Frag_Define: Defining fragments of unit cell", BC.END)
    for id, fragmentobject in enumerate(fragmentobjects):
        blankline()
        print("Fragmentobject:", BC.WARNING, BC.BOLD, fragmentobject.Name, BC.END)
        #Charge-model info to add to inputfile
        chargemodelline = ash.interfaces.interface_ORCA.chargemodel_select(chargemodel)

        #Call Clusterfragment and have print/write/something out coords and elems for atoms in list [0,1,2,3 etc.]
        atomlist=fragmentobject.clusterfraglist[0]
        fragcoords,fragelems=Cluster.get_coords_for_atoms(atomlist)
        ash.modules.module_coords.write_xyzfile(fragelems, fragcoords, "fragment")
        gasfrag=ash.Fragment(coords=fragcoords,elems=fragelems)

        print("Defined gasfrag:", gasfrag)



        #print(gasfrag.__dict__)
        #Creating ORCA theory object with fragment

        #print_time_rel_and_tot(currtime, origtime, modulename='gasfragcalc_ORCA prep stuff')
        currtime = time.time()
        #Assuming mainfrag is fragmentobject 0 and only mainfrag can be Broken-symmetry
        if id == 0:
            if brokensym==True:
                ORCASPcalculation = ash.interfaces.interface_ORCA.ORCATheory(orcadir=orcadir,
                                       orcasimpleinput=orcasimpleinput, numcores=NUMPROC,
                                       orcablocks=orcablocks, extraline=chargemodelline, brokensym=brokensym, HSmult=HSmult, atomstoflip=atomstoflip)
            else:
                ORCASPcalculation = ash.interfaces.interface_ORCA.ORCATheory(orcadir=orcadir,
                                       orcasimpleinput=orcasimpleinput, numcores=NUMPROC,
                                       orcablocks=orcablocks, extraline=chargemodelline)
        else:
            ORCASPcalculation = ash.interfaces.interface_ORCA.ORCATheory(orcadir=orcadir,
                                           orcasimpleinput=orcasimpleinput, numcores=NUMPROC,
                                           orcablocks=orcablocks, extraline=chargemodelline)
        print("ORCASPcalculation:", ORCASPcalculation)
        #print(ORCASPcalculation.__dict__)
        #Run ORCA calculation with charge-model info
        #ORCASPcalculation.run(numcores=NUMPROC, charge=fragmentobject.Charge, mult=fragmentobject.Mult)
        ash.Singlepoint(theory=ORCASPcalculation, fragment=gasfrag, charge=fragmentobject.charge,mult=fragmentobject.mult)
        #print_time_rel_and_tot(currtime, origtime, modulename='gasfragcalc_ORCA orca run', moduleindex=4)
        currtime = time.time()

        if chargemodel == 'DDEC3' or chargemodel == 'DDEC6':
            #Calling DDEC_calc (calls chargemol)
            atomcharges, molmoms, voldict = DDEC_calc(elems=gasfrag.elems, theory=ORCASPcalculation,
                                            numcores=NUMPROC, DDECmodel=chargemodel, molecule_charge=fragmentobject.charge,
                                            molecule_spinmult=fragmentobject.mult,
                                            calcdir="DDEC_fragment"+str(id), gbwfile=ORCASPcalculation.filename+'.gbw')

            print("atomcharges:", atomcharges)

            #Adding molmoms and voldict to fragmentobject
            fragmentobject.molmoms=molmoms
            fragmentobject.voldict=voldict
            #NOTE: We are not going to derive DDEC LJ parameters here but rather at end of SP loop.
        else:
            #Grab atomic charges for fragment.
            atomcharges=ash.interfaces.interface_ORCA.grabatomcharges_ORCA(chargemodel,ORCASPcalculation.filename+'.out')
            #print_time_rel_and_tot(currtime, origtime, modulename='gasfragcalc_ORCA grabatomcharges', moduleindex=4)
            currtime = time.time()

        print("Elements:", gasfrag.elems)
        print("Gasloop atomcharges:", atomcharges)
        assert len(atomcharges) != 0, "Atomcharges list is empty. Something went wrong with grabbing charges"

        #Updating charges inside mainfrag/counterfrag object
        fragmentobject.add_charges(atomcharges)


        #print_time_rel_and_tot(currtime, origtime, modulename='gasfragcalc_ORCA fragmentobject add charges', moduleindex=4)
        currtime = time.time()
        #Assign pointcharges to each atom of MM cluster.
        pointchargeupdate(Cluster,fragmentobject,atomcharges)
        #print_time_rel_and_tot(currtime, origtime, modulename='gasfragcalc_ORCA pointchargeupdate', moduleindex=4)
        currtime = time.time()
        #Keep backup of ORCA outputfile and GBW file
        shutil.copy(ORCASPcalculation.filename + '.out', fragmentobject.Name + '.out')
        shutil.copyfile(ORCASPcalculation.filename + '.out', './SPloop-files/'+fragmentobject.Name+'-Gascalc' + '.out')
        shutil.copyfile(ORCASPcalculation.filename + '.gbw', './SPloop-files/'+fragmentobject.Name+'-Gascalc' + '.gbw')
        if id ==0:
            shutil.copy(ORCASPcalculation.filename + '.gbw', 'lastorbitals.gbw')
        #Keeping copy of each fragment GBW file: fragment0.gbw, fragment1.gbw etc.
        shutil.copy(ORCASPcalculation.filename + '.gbw', 'fragment{}.gbw'.format(id))
        #print_time_rel_and_tot(currtime, origtime, modulename='gasfragcalc_ORCA shutil stuff', moduleindex=4)
        #currtime = time.time()
        #Clean up ORCA job.
        ORCASPcalculation.cleanup()
        #print_time_rel_and_tot(currtime, origtime, modulename='gasfragcalc_ORCA orca cleanup', moduleindex=4)
        #currtime = time.time()
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
        chargemodelline = ash.interfaces.interface_ORCA.chargemodel_select(chargemodel)

        #Call Clusterfragment and have print/write/something out coords and elems for atoms in list [0,1,2,3 etc.]
        atomlist=fragmentobject.clusterfraglist[0]
        fragcoords,fragelems=Cluster.get_coords_for_atoms(atomlist)
        ash.modules.module_coords.write_xyzfile(fragelems, fragcoords, "fragment")
        gasfrag=ash.Fragment(coords=fragcoords,elems=fragelems)

        #print("Defined gasfrag:", gasfrag)
        #print(gasfrag.__dict__)
        #Creating xTB theory object with fragment
        xTBSPcalculation = ash.xTBTheory(xtbdir=xtbdir,xtbmethod=xtbmethod)

        print("xTBSPcalculation:", xTBSPcalculation)
        print(xTBSPcalculation.__dict__)
        #Run xTB calculation with charge-model info
        #xTBSPcalculation.run(numcores=NUMPROC, charge=fragmentobject.Charge, mult=fragmentobject.mult)
        ash.Singlepoint(theory=xTBSPcalculation, fragment=gasfrag, charge=fragmentobject.charge, mult=fragmentobject.mult)

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



def choose_shortrangemodel(Cluster,shortrangemodel,fragmentobjects,QMtheory,mainfrag_gbwfile,numcores,LJHparameters):

    if shortrangemodel=='UFF':
        print("Using UFF forcefield for all elements")
        for fragmentobject in fragmentobjects:
            #fragmentobject.Elements
            for el in fragmentobject.Elements:
                print("UFF parameter for {} : {}".format(el, UFFdict[el]))

        #Using UFF_ prefix before element
        atomtypelist=['UFF_'+i for i in Cluster.elems]
        atomtypelist_uniq = np.unique(atomtypelist).tolist()
        #Adding atomtypes to Cluster object
        Cluster.atomtypes=atomtypelist

        #Adding atomtypes to fragmentobjects
        for fragmentobject in fragmentobjects:
            fragmentobject.atomtypelist = ["UFF_{}".format(el) for el in fragmentobject.Atoms]

        #Create ASH forcefield file by looking up UFF parameters
        with open('Cluster_forcefield.ff', 'w') as forcefile:
            forcefile.write('#UFF Lennard-Jones parameters (R0 in Angstrom and eps in kcal/mol) \n')
            for atomtype in atomtypelist_uniq:
                #Getting just element-par for UFFdict lookup
                atomtype_el=atomtype.replace('UFF_','')
                forcefile.write('LennardJones_i_R0 {}  {:12.6f}   {:12.6f}\n'.format(atomtype, UFFdict[atomtype_el][0],UFFdict[atomtype_el][1]))
    #Modified UFF forcefield with 0 parameter on H atom (avoids repulsion)
    elif shortrangemodel=='UFF_modH':
        print("Using UFF forcefield with modified H-parameter")
        print("H parameters :", LJHparameters)
        print("")
        UFFdict_Hzero=copy.deepcopy(UFFdict)
        UFFdict_Hzero['H'] = [LJHparameters[0], LJHparameters[1]]

        #print("UFF parameters:", UFFdict)
        for fragmentobject in fragmentobjects:
            #fragmentobject.Elements
            for el in fragmentobject.Elements:
                print("UFF parameter for {} : {}".format(el, UFFdict_Hzero[el]))

        #Using UFF_ prefix before element
        atomtypelist=['UFF_'+i for i in Cluster.elems]
        #Adding atomtypes to Cluster object
        Cluster.atomtypes=atomtypelist
        atomtypelist_uniq = np.unique(atomtypelist).tolist()

        #Adding atomtypes to fragmentobjects
        for fragmentobject in fragmentobjects:
            fragmentobject.atomtypelist = ["UFF_{}".format(el) for el in fragmentobject.Atoms]

        #Create ASH forcefield file by looking up UFF parameters
        with open('Cluster_forcefield.ff', 'w') as forcefile:
            forcefile.write('#UFF Lennard-Jones parameters (R0 in Angstrom and eps in kcal/mol) \n')
            for atomtype in atomtypelist_uniq:
                #Getting just element-par for UFFdict lookup
                atomtype_el=atomtype.replace('UFF_','')
                forcefile.write('LennardJones_i_R0 {}  {:12.6f}   {:12.6f}\n'.format(atomtype, UFFdict_Hzero[atomtype_el][0],UFFdict_Hzero[atomtype_el][1]))

    elif shortrangemodel=='DDEC3' or shortrangemodel=='DDEC6':
        print("Deriving DDEC Lennard-Jones parameters")
        print("DDEC model :", shortrangemodel)

        #Getting R0 and epsilon for mainfrag
        #fragmentobjects[0].r0list, fragmentobjects[0].epsilonlist = DDEC_to_LJparameters(elems, molmoms, voldict)

        #Getting R0 and epsilon for counterfrags
        for fragindex,fragmentobject in enumerate(fragmentobjects):
            print("Fragmentobject with fragindex: ", fragindex)
            print("fragmentobject Atoms:", fragmentobject.Atoms)
            print("fragmentobject molmoms:", fragmentobject.molmoms)
            print("fragmentobject voldict:", fragmentobject.voldict)

            #If molmoms and voldict not already calculated (could be if chargemodel is DDEC)
            if len(fragmentobject.molmoms) == 0:
                print("No molmoms available. Calculating.")

                #Using last mainfrag GBW-file (from SC-QM/MM)
                if fragindex==0:
                    #gbwfile=mainfrag_gbwfile
                    #Trying to use gas fragment instead. Non-polarized electron density
                    print("USING NONPOLARIZED DENSITY FOR MAINFRAG")
                    gbwfile="fragment{}.gbw".format(fragindex)
                else:
                    #Use GBWfile created by gasfragcalc_ORCA (non-polarized)
                    gbwfile="fragment{}.gbw".format(fragindex)

                print("Using GBW file: ", gbwfile)
                DDECcharges, fragmentobject.molmoms, fragmentobject.voldict = DDEC_calc(elems=fragmentobject.Atoms, theory=QMtheory,
                                                        numcores=numcores, DDECmodel=shortrangemodel,
                                                        molecule_spinmult=fragmentobject.mult, molecule_charge=fragmentobject.charge,
                                                        calcdir="DDEC_LJcalc_fragment_{}".format(fragmentobject.Name), gbwfile=gbwfile)
                print("DDECcharges:", DDECcharges)
            #Getting R0 and epsilon
            fragmentobject.r0list, fragmentobject.epsilonlist = DDEC_to_LJparameters(fragmentobject.Atoms, fragmentobject.molmoms, fragmentobject.voldict)
            #Creating list of atomtypes for fragmenttype
            fragmentobject.atomtypelist = ["DDEC_f{}_{}_{}".format(fragindex,el,m) for m,el in enumerate(fragmentobject.Atoms)]
            print("fragmentobject.atomtypelist:", fragmentobject.atomtypelist)

        print("Using {}-derived forcefield for all elements".format(shortrangemodel))
        #atomtypelist_uniq = np.unique(atomtypelist).tolist()
        #print("atomtypelist_uniq:", atomtypelist_uniq)
        #Create ASH forcefield file by looking up UFF parameters
        with open('Cluster_forcefield.ff', 'w') as forcefile:
            forcefile.write('#{} Lennard-Jones parameters (R0 in Angstrom and eps in kcal/mol) \n'.format(shortrangemodel))
            #for atomtype in atomtypelist_uniq:
            for fragmentobject in fragmentobjects:
                for atomtype,r0,eps in zip(fragmentobject.atomtypelist,fragmentobject.r0list,fragmentobject.epsilonlist):
                    forcefile.write('LennardJones_i_R0 {}  {:12.6f}   {:12.6f}\n'.format(atomtype, r0, eps))


    elif shortrangemodel=='manual':
        print("shortrangemodel option: manual")
        print("Using atomtypes for Cluster: MAN_X  where X is an element, e.g. MAN_O, MAN_C, MAN_H")
        print("Will assume presence of ASH forcefield file called: Cluster_forcefield.ff")
        print("Should contain Lennard-Jones entries for atomtypes MAN_X.")
        print("File needs to be copied to scratch for geometry optimization job.")
        #Using MAN prefix before element
        atomtypelist=['MAN_'+i for i in Cluster.elems]
        #Adding atomtypes to Cluster object
        Cluster.atomtypes=atomtypelist
    else:
        print("Undefined shortrangemodel")
        ashexit()

    #Create full atomtypelist to be added to Cluster object
    #atomtypelist = [item for frag in fragmentobjects for item in frag.atomtypelist]
    #print("atomtypelist:", atomtypelist)
    full_list=[None]*Cluster.numatoms
    #print("full_list:", full_list)
    for fragmentobject in fragmentobjects:
        #print("fragmentobject Name:", fragmentobject.Name)
        #print("fragmentobject.clusterfraglist:", fragmentobject.clusterfraglist)
        for fraglist in fragmentobject.clusterfraglist:
            #print("Fragmentobject is:", fragmentobject)
            #print("Fragmentobject is:", fragmentobject.Name)
            #print("Fragmentobject dict:", fragmentobject.__dict__)
            #print("fraglist:", fraglist)
            for atomid,attype in zip(fraglist,fragmentobject.atomtypelist):
                #print("atomid : {} and attype: {}".format(atomid,attype))
                full_list[atomid] = attype
            #print("full_list:", full_list)
            #print("-----------")
    #print("full_list:", full_list)
    if None in full_list:
        print("problem")
        print(full_list)
        ashexit()
    Cluster.atomtypes=full_list
    #print("Cluster.atomtypes:", Cluster.atomtypes)
