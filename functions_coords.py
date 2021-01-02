from functions_general import *
#import functions_molcrys
import numpy as np
#import settings_molcrys
import settings_ash
import constants
#from math import sqrt
#from math import pow
import copy
import math
import ash
sqrt = math.sqrt
pow = math.pow

#Elements and atom numbers
elematomnumbers = {'h':1, 'he': 2, 'li':3, 'be':4, 'b':5, 'c':6, 'n':7, 'o':8, 'f':9, 'ne':10, 'na':11, 'mg':12, 'al':13, 'si':14, 'p':15, 's':16, 'cl':17, 'ar':18, 'k':19, 'ca':20, 'sc':21, 'ti':22, 'v':23, 'cr':24, 'mn':25, 'fe':26, 'co':27, 'ni':28, 'cu':29, 'zn':30, 'ga':31, 'ge':32, 'as':33, 'se':34, 'br':35, 'kr':36, 'rb':37, 'sr':38, 'y':39, 'zr':40, 'nb':41, 'mo':42, 'tc':43, 'ru':44, 'rh':45, 'pd':46, 'ag':47, 'cd':48, 'in':49, 'sn':50, 'sb':51, 'te':52, 'i':53, 'xe':54, 'cs':55, 'ba':56, 'la':57, 'ce':58, 'pr':59, 'nd':60, 'pm':61, 'sm':62, 'eu':63, 'gd':64, 'tb':65, 'dy':66, 'ho':67, 'er':68, 'tm':69, 'yb':70, 'lu':71, 'hf':72, 'ta':73, 'w':74, 're':75, 'os':76, 'ir':77, 'pt':78, 'au':79, 'hg':80, 'tl':81, 'pb':82, 'bi':83, 'po':84, 'at':85, 'rn':86, 'fr':87, 'ra':88, 'ac':89, 'th':90, 'pa':91, 'u':92, 'np':93, 'pu':94, 'am':95, 'cm':96, 'bk':97, 'cf':98, 'es':99, 'fm':100, 'md':101, 'no':102, 'lr':103, 'rf':104, 'db':105, 'sg':106, 'bh':107, 'hs':108, 'mt':109, 'ds':110, 'rg':111, 'cn':112, 'nh':113, 'fl':114, 'mc':115, 'lv':116, 'ts':117, 'og':118}
#Atom masses
atommasses = [1.00794, 4.002602, 6.94, 9.0121831, 10.81, 12.01070, 14.00670, 15.99940, 18.99840316, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085, 30.973762, 32.065, 35.45, 39.948, 39.0983, 40.078, 44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.96, 97, 101.07, 102.9055, 106.42, 107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.905452, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145, 150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592, 204.38, 207.2, 208.9804, 209, 210, 222, 223, 226, 227, 232.0377, 231.03588, 238.02891, 237, 244, 243, 247, 247, 251, 252, 257, 258, 259, 262 ]

#Covalent radii for elements (Alvarez) in Angstrom.
#Used for connectivity
eldict_covrad={'H':0.31, 'He':0.28, 'Li':1.28, 'Be':0.96, 'B':0.84, 'C':0.76, 'N':0.71, 'O':0.66, 'F':0.57, 'Ne':0.58, 'Na':1.66, 'Mg':1.41, 'Al':1.21, 'Si':1.11, 'P':1.07, 'S':1.05, 'Cl':1.02, 'Ar':1.06, 'K':2.03, 'Ca':1.76, 'Sc':1.70, 'Ti':1.6, 'V':1.53, 'Cr':1.39, 'Mn':1.61, 'Fe':1.52, 'Co':1.50, 'Ni':1.24, 'Cu':1.32, 'Zn':1.22, 'Ga':1.22, 'Ge':1.20, 'As':1.19, 'Se':1.20, 'Br':1.20, 'Kr':1.16, 'Rb':2.2, 'Sr':1.95, 'Y':1.9, 'Zr':1.75, 'Nb':1.64, 'Mo':1.54, 'Tc':1.47, 'Ru':1.46, 'Rh':1.42, 'Pd':1.39, 'Ag':1.45, 'Cd':1.44, 'In':1.42, 'Sn':1.39, 'Sb':1.39, 'Te':1.38, 'I':1.39, 'Xe':1.40,}

#Modified radii for certain elements like Na, K
eldict_covrad['Na']=0.0001
eldict_covrad['K']=0.0001

#print(eldict_covrad)


def print_internal_coordinate_table(fragment):
    print('='*50)
    print("Optimized internal coordinates")
    print('='*50)
    #Looping over connected fragments
    bondpairs=[]
    bondpairsdict={}
    for conn_fragment in fragment.connectivity:
        #Looping over atom indices in fragment
        for atom in conn_fragment:
            #print("atom:", atom)
            connatoms = get_connected_atoms(fragment.coords, fragment.elems,settings_ash.scale,settings_ash.tol,atom)
            #print("connatoms:", connatoms)
            for conn_i in connatoms:
                dist=distance_between_atoms(fragment=fragment, atom1=atom, atom2=conn_i)
                #bondpairs.append([atom,conn_i,dist])
                bondpairsdict[frozenset((atom,conn_i))] = dist
    
    #Using frozenset: https://stackoverflow.com/questions/46633065/multiples-keys-dictionary-where-key-order-doesnt-matter
    #sort bondpairs list??
    
    #print bondpairs list
    print("Bond lengths (Å):")
    print('-'*38)
    for key,val in bondpairsdict.items():
        listkey=list(key)
        elA=fragment.elems[listkey[0]]
        elB=fragment.elems[listkey[1]]
        print("Bond: {:8}{:4} - {:4}{:4} {:>6.3f}".format(listkey[0],elA,listkey[1],elB, val ))
    print('='*50)
#Function to check if string corresponds to an element symbol or not.
#Compares in lowercase
def isElement(string):
    if string.lower() in elematomnumbers:
        return True
    else:
        return False

#Checks if list of string is list of elements or no
def isElementList(list):
    for l in list:
        if not isElement(l):
            return False
    return True


#From lists of coords,elems and atom indices, print coords with elem
def print_coords_for_atoms(coords,elems,members):
    for m in members:
        print("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}".format(elems[m],coords[m][0], coords[m][1], coords[m][2]))

#From lists of coords,elems and atom indices, write XYZ file coords with elem
#Todo: make part of Fragment class
def write_XYZ_for_atoms(coords,elems,members,name):
    subset_elems=[elems[i] for i in members]
    subset_coords=[coords[i] for i in members]
    with open(name+'.xyz', 'w') as ofile:
        ofile.write(str(len(subset_elems))+'\n')
        ofile.write("title"+'\n')
        for el,c in zip(subset_elems,subset_coords):
            line="{:4} {:>12.6f} {:>12.6f} {:>12.6f}".format(el,c[0], c[1], c[2])
            ofile.write(line+'\n')


#From lists of coords,elems and atom indices, print coords with elems
#If list of atom indices provided, print as leftmost column
#If list of labels provided, print as rightmost column
#If list of labels2 provided, print as rightmost column
def print_coords_all(coords,elems,indices=None, labels=None, labels2=None):
    if indices is None:
        if labels is None:
            for i in range(len(elems)):
                print("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}".format(elems[i],coords[i][0], coords[i][1], coords[i][2]))
        else:
            if labels2 is None:
                for i in range(len(elems)):
                    print("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6}".format(elems[i],coords[i][0], coords[i][1], coords[i][2], labels[i]))
            else:
                for i in range(len(elems)):
                    print("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6} :>6".format(elems[i],coords[i][0], coords[i][1], coords[i][2], labels[i], label2[i]))
    else:
        if labels is None:
            for i in range(len(elems)):
                print("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}".format(indices[i],elems[i],coords[i][0], coords[i][1], coords[i][2]))
        else:
            if labels2 is None:
                for i in range(len(elems)):
                    print("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6}".format(indices[i],elems[i],coords[i][0], coords[i][1], coords[i][2], labels[i]))
            else:
                for i in range(len(elems)):
                    print("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6} {:>6}".format(indices[i],elems[i],coords[i][0], coords[i][1], coords[i][2], labels[i], labels2[i]))

def distance(A,B):
    return sqrt(pow(A[0] - B[0],2) + pow(A[1] - B[1],2) + pow(A[2] - B[2],2)) #fastest
    #return sum((v_i - u_i) ** 2 for v_i, u_i in zip(A, B)) ** 0.5 #slow
    #return np.sqrt(np.sum((A - B) ** 2)) #very slow
    #return np.linalg.norm(A - B) #VERY slow
    #return sqrt(sum((px - qx) ** 2.0 for px, qx in zip(A, B))) #slow
    #return sqrt(sum([pow((a - b),2) for a, b in zip(A, B)])) #OK
    #return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2) #Very slow
    #return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2) #faster
    #return math.sqrt(math.pow(A[0] - B[0],2) + math.pow(A[1] - B[1],2) + math.pow(A[2] - B[2],2)) #faster
    #return sqrt(sum((A-B)**2)) #slow
    #return sqrt(sum(pow((A - B),2))) does not work
    #return np.sqrt(np.power((A-B),2).sum()) #very slow
    #return sqrt(np.power((A - B), 2).sum())
    #return np.sum((A - B) ** 2)**0.5 #very slow



def center_of_mass(coords,masses):
    print("to be finished")
    exit()

def get_centroid(coords):
    sum_x=0; sum_y=0; sum_z=0
    for c in coords:
        sum_x+=c[0]; sum_y+=c[1]; sum_z+=c[2]
    return [sum_x/len(coords),sum_y/len(coords),sum_z/len(coords)]

#Change origin to centroid of coords
def change_origin_to_centroid(coords):
    centroid = get_centroid(coords)
    new_coords=[]
    for c in coords:
        new_coords.append(c-centroid)
    return new_coords

#get_solvshell function based on single point of origin. Using geometric center of molecule
def get_solvshell_origin():
    print("to finish")
    #TODO: finish get_solvshell_origin
    exit()
#Determine threshold for whether atoms are connected or not based on covalent radii for pair of atoms
# R_ij < scale*(rad_i + rad_j) + tol
#Uses global scale and tol parameters that may be changed at input
def threshold_conn(elA,elB,scale,tol):
    #crad=list(map(eldict_covrad.get, [elA,elB]))
    #crad=[eldict_covrad.get(key) for key in [elA,elB]]
    return scale*(eldict_covrad[elA]+eldict_covrad[elB]) + tol
    #print(crad)
    #return scale*(crad[0]+crad[1]) + tol

#Connectivity function (called by Fragment object)
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

#Get connected atoms to chosen atom index based on threshold
#Uses slow for-loop structure with distance-function call
#Don't use unless system is small
def get_connected_atoms(coords, elems,scale,tol,atomindex):
    connatoms=[]
    coords_ref=coords[atomindex]
    elem_ref=elems[atomindex]
    for i,c in enumerate(coords):
        if distance(coords_ref,c) < threshold_conn(elems[i], elem_ref,scale,tol):
            if i != atomindex:
                connatoms.append(i)
    return connatoms

#Euclidean distance functions:
#https://semantive.com/pl/blog/high-performance-computation-in-python-numpy/
def einsum_mat(mat_v, mat_u):
    mat_z = mat_v - mat_u
    return np.sqrt(np.einsum('ij,ij->i', mat_z, mat_z))

def bare_numpy_mat(mat_v, mat_u):
   return np.sqrt(np.sum((mat_v - mat_u) ** 2, axis=1))

def l2_norm_mat(mat_v, mat_u):
   return np.linalg.norm(mat_v - mat_u, axis=1)

def dummy_mat(mat_v, mat_u):
   return [sum((v_i - u_i)**2 for v_i, u_i in zip(v, u))**0.5 for v, u in zip(mat_v, mat_u)]

#Get connected atoms to chosen atom index based on threshold
#Clever np version for calculating the euclidean distance without a for-loop and having to call distance function
#many time
#https://semantive.com/pl/blog/high-performance-computation-in-python-numpy/
#Avoiding for loops
def get_connected_atoms_np(coords, elems,scale,tol, atomindex):
    #print("inside get conn atoms np")
    #print("atomindex:", atomindex)
    connatoms = []
    #Creating np array of the coords to compare
    compcoords = np.tile(coords[atomindex], (len(coords), 1))
    #Einsum is slightly faster than bare_numpy_mat. All distances in one go
    distances=einsum_mat(coords,compcoords)
    #Getting all thresholds as list via list comprehension.
    el_covrad_ref=eldict_covrad[elems[atomindex]]
    #Cheaper way of getting thresholds list than calling threshold_conn
    #List comprehension of dict lookup and convert to numpy. Should be as fast as can be done
    #thresholds = np.empty(len(elems))
    #for i in range(len(thresholds)):
    #    thresholds[i]=eldict_covrad[elems[i]]
    # TODO: Slowest part but hard to make faster
    thresholds=np.array([eldict_covrad[elems[i]] for i in range(len(elems))])
    #Numpy addition and multiplication done on whole array
    thresholds=thresholds+el_covrad_ref
    thresholds=thresholds*scale
    thresholds=thresholds+tol
    #Old slow way
    #thresholds=np.array([threshold_conn(elems[i], elem_ref,scale,tol) for i in range(len(elems))])
    #Getting difference of distances and thresholds
    diff=distances-thresholds
    #Getting connatoms by finding indices of diff with negative values (i.e. where distance is smaller than threshold)
    connatoms=np.where(diff<0)[0].tolist()
    #print("connatoms ", connatoms)
    return connatoms



#Numpy clever loop test.
#Either atomindex or membs has to be defined
def get_molecule_members_loop_np(coords, elems, loopnumber,scale,tol, atomindex='', membs=None):
    if membs is None:
        membs = []
        membs.append(atomindex)
        membs = get_connected_atoms_np(coords, elems, scale,tol, atomindex)
    # How often to search for connected atoms as the members list grows:
    #TODO: Need to make this better
    for i in range(loopnumber):
        for j in membs:
            conn = get_connected_atoms_np(coords, elems, scale,tol,j)
            membs = membs + conn
        membs = np.unique(membs).tolist()
    # Remove duplicates and sort
    membs = np.unique(membs).tolist()
    return membs

#Numpy clever loop test.
#Version 2 never goes through same atom

#@timefn
def get_molecule_members_loop_np2(coords, elems, loopnumber, scale, tol, atomindex=None, membs=None):
    if membs is None:
        membs = []
        membs.append(atomindex)
        timestampA = time.time()
        membs = get_connected_atoms_np(coords, elems, scale,tol, atomindex)
        #print("membs:", membs)
        #ash.print_time_rel(timestampA, modulename='membs first py')

    finalmembs=membs
    for i in range(loopnumber):
        #Get list of lists of connatoms for each member
        newmembers=[get_connected_atoms_np(coords, elems, scale,tol, k) for k in membs]
        #print("newmembers:", newmembers)
        #exit()
        #Get a unique flat list
        trimmed_flat=np.unique([item for sublist in newmembers for item in sublist]).tolist()
        #print("trimmed_flat:", trimmed_flat)
        #print("finalmembs ", finalmembs)

        #Check if new atoms not previously found
        membs = listdiff(trimmed_flat, finalmembs)
        #print("membs:", membs)
        #exit()
        #Exit loop if nothing new found
        if len(membs) == 0:
            #print("exiting...")
            #exit()
            return finalmembs
        #print("type of membs:", type(membs))
        #print("type of finalmembs:", type(finalmembs))
        finalmembs+=membs
        #print("finalmembs ", finalmembs)
        finalmembs=np.unique(finalmembs).tolist()
        #print("finalmembs ", finalmembs)
        #exit()
        #print("finalmembs:", finalmembs)
        #print("----------")
        #ash.print_time_rel(timestampA, modulename='finalmembs  py')
        #exit()
    return finalmembs



#Get molecule members by running get_connected_atoms function on expanding member list
#Uses loopnumber for when to stop searching.
#Does extra work but not too bad
#Uses either single atomindex or members lists
def get_molecule_members_loop(coords, elems, loopnumber,scale,tol, atomindex='', members=None):
    if members is None:
        members = []
        members.append(atomindex)
        connatoms = get_connected_atoms(coords, elems,scale,tol,atomindex)
        members = members + connatoms
    # How often to search for connected atoms as the members list grows:
    for i in range(loopnumber):
        #conn = [get_connected_atoms(coords, elems, scale,tol,j) for j in members]
        for j in members:
            conn = get_connected_atoms(coords, elems, scale,tol,j)
            members = members + conn
            #members=np.concatenate((members, conn))
        members = np.unique(members).tolist()
        members=members+conn
    # Remove duplicates and sort
    members = np.unique(members).tolist()
    return members

#Get-molecule-members with fixed recursion-depth of 4
#Efficient but limited to 4
#Updated to 5
#Maybe not so efficient after all
def get_molecule_members_fixed(coords,elems, scale,tol, atomindex='', members=None):
    print("Disabled")
    print("not so efficient")
    exit()
    if members is None:
        members = []
        members.append(atomindex)
        connatoms = get_connected_atoms(coords, elems, scale, tol,atomindex)
        members=members+connatoms
    finalmembers=members
    #How often to search for connected atoms as the members list grows:
    for j in members:
        conn=get_connected_atoms(coords, elems, scale,tol,j)
        finalmembers=finalmembers+conn
        for k in conn:
            conn2 = get_connected_atoms(coords, elems, scale,tol,k)
            finalmembers = finalmembers + conn2
            #for l in conn2:
            #    conn3 = get_connected_atoms(coords, elems, scale,tol,l)
            #    finalmembers = finalmembers + conn3
                #for m in conn3:
                #    conn4 = get_connected_atoms(coords, elems, scale, tol,m)
                #    finalmembers = finalmembers + conn4
    #Remove duplicates and sort
    finalmembers=np.unique(finalmembers).tolist()
    return finalmembers

def create_coords_string(elems,coords):
    coordsstring=''
    for el, c in zip(elems,coords):
        coordsstring=coordsstring+el+'  '+str(c[0])+'  '+str(c[1])+'  '+str(c[2])+'\n'
    return coordsstring[:-1]

#Takes list of elements and gives formula
def elemlisttoformula(list):
    #This dict comprehension was slow for large systems. Using set to reduce iterations
    dict = {i: list.count(i) for i in set(list)}
    formula=""
    for item in dict.items():
        el=item[0];count=item[1]
        #string=el+str(count)
        formula=formula+el+str(count)
    return formula

#From molecular formula (string, e.g. "FeCl4") to list of atoms
def molformulatolist(formulastring):
    el=""
    diff=""
    els=[]
    atomunits=[]
    numels=[]
    #Read string by character backwards
    for count,char in enumerate(formulastring[::-1]):
        if isint(char):
            el=char+el
        if char.islower():
            el=char+el
            diff=char+diff
        if char.isupper():
            el=char+el
            diff=char+diff
            atomunits.append(el)
            els.append(diff)
            el=""
            diff=""
    for atm,element in zip(atomunits,els):
        if atm > element:
            number=atm[len(element):]
            numels.append(int(number))
        else:
            number=1
            numels.append(int(number))
    atoms = []
    for i, j in zip(els, numels):
        for k in range(j):
            atoms.append(i)
    #Final reverse
    els.reverse()
    numels.reverse()
    atoms.reverse()
    return atoms


#Read XYZ file
def read_xyzfile(filename):
    #Will accept atom-numbers as well as symbols
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
            'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    print("Reading coordinates from XYZfile {} ".format(filename))
    coords=[]
    elems=[]
    with open(filename) as f:
        for count,line in enumerate(f):
            if count == 0:
                numatoms=int(line.split()[0])
            if count > 1:
                if len(line.strip()) > 0:
                    if isint(line.split()[0]) is True:
                        elems.append(elements[int(line.split()[0])-1])
                    else:
                        elems.append(line.split()[0])
                    coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    assert len(coords) == numatoms, "Number of coordinates does not match header line"
    assert len(coords) == len(elems), "Number of coordinates does not match elements."
    return elems,coords


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

    fmt = "{:2s}" + (" {:15."+str(decimals)+"f}")*3

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
    V=np.array(V)
    print(set_coordinates(atoms, V, title=title))
    return

#Write XYZfile provided list of elements and list of list of coords and filename
def write_xyzfile(elems,coords,name,printlevel=2):
    with open(name+'.xyz', 'w') as ofile:
        ofile.write(str(len(elems))+'\n')
        ofile.write("title"+'\n')
        for el,c in zip(elems,coords):
            line="{:4} {:16.12f} {:16.12f} {:16.12f}".format(el,c[0], c[1], c[2])
            ofile.write(line+'\n')
    if printlevel >= 2:
        print("Wrote XYZ file:", name+'.xyz')


#Function that reads XYZ-file with multiple files, splits and return list of coordinates
#Created for splitting crest_conformers.xyz but may also be used for MD traj.
#Also grabs last word in title line. Typically an energy (has to be converted to float outside)
def split_multimolxyzfile(file, writexyz=False):
    all_coords=[]
    all_elems=[]
    all_titles=[]
    molcounter = 0
    coordgrab=False
    titlegrab=False
    coords = []
    elems = []
    with open(file) as f:
        for index, line in enumerate(f):
            if index == 0:
                numatoms = line.split()[0]
            if coordgrab == True:
                if len(line.split()) > 1:
                    elems.append(line.split()[0])
                    coords_x=float(line.split()[1]);coords_y=float(line.split()[2]);coords_z=float(line.split()[3])
                    coords.append([coords_x,coords_y,coords_z])
                if len(coords) == int(numatoms):
                    all_coords.append(coords)
                    all_elems.append(elems)
                    if writexyz is True:
                        #Alternative option: write each conformer/molecule to disk as XYZfile
                        write_xyzfile(elems, coords, "molecule"+str(molcounter))
                    coords = []
                    elems = []
            if titlegrab is True:
                all_titles.append(line.split()[-1])
                titlegrab=False
                coordgrab = True
            if line.split()[0] == str(numatoms):
                # print("coords is", len(coords))
                molcounter += 1
                titlegrab=True
                coordgrab=False

    return all_elems,all_coords, all_titles


#Read Tcl-Chemshell fragment file and grab elems and coords. Coordinates converted from Bohr to Angstrom
#Taken from functions_solv
def read_fragfile_xyz(fragfile):
    #removing extension from fragfile name if present and then adding back.
    pathtofragfile=fragfile.split('.')[0]+'.c'
    coords=[]
    elems=[]
    #TODO: Change elems and coords to numpy array instead
    grabcoords=False
    with open(pathtofragfile) as ffile:
        for line in ffile:
            if 'block = connectivity' in line:
                grabcoords=False
            if grabcoords==True:
                coords.append([float(i)*constants.bohr2ang for i in line.split()[1:]])
                elems.append(line.split()[0])
            if 'block = coordinates records ' in line:
                #numatoms=int(line.split()[-1])
                grabcoords=True
    return elems,coords



#Write PDBfile (dummy version) for PyFrame
def write_pdbfile_dummy(elems,coords,name, atomlabels,residlabels):
    with open(name+'.pdb', 'w') as pfile:
        resnames=atomlabels
        #resnames=['QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'HOH', 'HOH','HOH']
        #resids=[1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2]
        #Example:
        #pfile.write("ATOM      1  N   SER A   2      65.342  32.035  32.324  1.00  0.00           N\n")
        for count,(el,c,resname,resid) in enumerate(zip(elems,coords, resnames, residlabels)):
            #print(count, el,c,resname)
            #Dummy resid for everything
            #resid=1
            #Using string format from: https://cupnet.net/pdb-format/
            line="{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format(
                'ATOM', count+1, el, '', resname, '', resid, '',    c[0], c[1], c[2], 1.0, 0.00, el, '')
            pfile.write(line+'\n')
    print("Wrote PDB file:", name+'.pdb')

#set out [open "result.pdb" w ]
#foreach a $atomindexlist b $segmentlist c $residlist d $resnamelist e $atomnamelist f $typeslist cx $coords_x cy $coords_y cz $coords_z el $ellist {
# #ATOM      1  N   SER A   2      65.342  32.035  32.324  1.00  0.00           N
#             set fmt1 "ATOM%7d %4s%4s%-1s%5d%12.3f%8.3f%8.3f%6s%6s%10s%2s"
# puts $out [format $fmt1 $a $e $d " " $c $cx $cy $cz "1.00" "0.00" $b $el]
#
#}
#close $out


#Calculate nuclear charge from XYZ-file
def nucchargexyz(file):
    el=[]
    with open(file) as f:
        for count,line in enumerate(f):
            if count >1:
                el.append(line.split()[0])
    totnuccharge=0
    for e in el:
        atcharge=eldict[e]
        totnuccharge+=atcharge
    return totnuccharge

#Calculate total nuclear charge from list of elements
def nucchargelist(ellist):
    totnuccharge=0
    els=[]
    for e in ellist:
        atcharge=elematomnumbers[e.lower()]
        totnuccharge+=atcharge
    return totnuccharge

#get list of nuclear charges from list of elements
#Used by Psi4 and CM5calc
# aka atomic numbers, aka atom numbers
def elemstonuccharges(ellist):
    nuccharges=[]
    for e in ellist:
        atcharge=elematomnumbers[e.lower()]
        nuccharges.append(atcharge)
    return nuccharges

#Calculate molecular mass from list of atoms
def totmasslist(ellist):
    totmass=0
    for e in ellist:
        atcharge = int(elematomnumbers[e.lower()])
        atmass=atommasses[atcharge-1]
        totmass+=atmass
    return totmass

#Calculate list of masses from list of elements
def list_of_masses(ellist):
    masses=[]
    for e in ellist:
        atcharge = int(elematomnumbers[e.lower()])
        atmass=atommasses[atcharge-1]
        masses.append(atmass)
    return masses

##############################
#RMSD and align related functions
#Many more to be added.
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

#Old list version
def old_centroid(X):
    """
    Calculate the centroid from a vectorset X
    """
    C = sum(X)/len(X)
    return C

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
        rmsd += sum([(v[i]-w[i])**2.0 for i in range(D)])
    return np.sqrt(rmsd/N)

#Turbomol coord->xyz
def coord2xyz(inputfile):
    """convert TURBOMOLE coordfile to xyz"""
    with open(inputfile, 'r') as f:
        coord = f.readlines()
        x = []
        y = []
        z = []
        atom = []
        for line in coord[1:-1]:
            x.append(float(line.split()[0])*constants.bohr2ang)
            y.append(float(line.split()[1])*constants.bohr2ang)
            z.append(float(line.split()[2])*constants.bohr2ang)
            atom.append(str(line.split()[3]))
        for item in atom:
            if len(item) == 1:
                atom[atom.index(item)] = item.replace(item[0], item[0].upper())
            if len(item) >= 2:
                atom[atom.index(item)] = item.replace(item, item[0].upper()+item[1:].lower())
        #natoms = int(len(coord[1:-1]))
        return atom, np.array(x), np.array(y), np.array(z)



#Get partial list by deleting elements not present in provided list of indices.
def get_partial_list(allatoms,partialatoms,list):
    otheratoms=listdiff(allatoms,partialatoms)
    otheratoms.reverse()
    for at in otheratoms:
        del list[at]
    return list


#Old function that used scipy to do distances and Hungarian. 
def scipy_hungarian(A,B):
    import scipy
    #timestampA = time.time()
    distances = scipy.spatial.distance.cdist(A, B, 'euclidean')
    #print("distances:", distances)
    #ash.print_time_rel(timestampA, modulename='scipy distances_cdist')
    #timestampA = time.time()
    indices_a, assignment = scipy.optimize.linear_sum_assignment(distances)
    #print("indices_a:", indices_a)
    #print("assignment:", assignment)
    #ash.print_time_rel(timestampA, modulename='scipy linear sum assignment')
    return assignment

#Hungarian algorithm to reorder coordinates. Uses Julia to calculates distances between coordinate-arrays A and B and then Hungarian Julia package.
#PyJulia needs to have been imported before (ash.py)
def hungarian_julia(A, B):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    try:
        #Calculating distances via Julia
        #print("Here. Calling Julia distances")
        #timestampA = time.time()
        
        #This one is SLOW!!! For rad30 Bf3hcn example it takes 23 seconds compare to 3.8 sec for scipy. 0.8 sec for scipy for both dist and hungarian
        #distances =ash.Main.Juliafunctions.distance_array(A,B)
        distances = cdist(A, B, 'euclidean')
        #ash.print_time_rel(timestampA, modulename='julia distance array')
        #timestampA = time.time()
        # Julian Hungarian call. Requires Hungarian package
        assignment, cost = ash.Hungarian.hungarian(distances)
        
        #ash.print_time_rel(timestampA, modulename='julia hungarian')
        #timestampA = time.time()
        #Removing zeros and offsetting by 1 (Julia 1-indexing)
        final_assignment=assignment[assignment != 0]-1
        
        #final_assignment = scipy_hungarian(A,B)
        
    except:
        print("Problem running Julia Hungarian function. Trying scipy instead")
        
        exit()
        
        final_assignment = scipy_hungarian(A,B)
    
    return final_assignment

#Hungarian reorder algorithm
#From RMSD
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
        #print("A_coord:", A_coord)
        #print("B_coord:", B_coord)

        view = scipy_hungarian(A_coord, B_coord)
        view_reorder[p_atom_idx] = q_atom_idx[view]
    #print("view_reorder:", view_reorder)
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
    print("unique_atoms:", unique_atoms)
    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)
    view_reorder -= 1
    print("view_reorder:", view_reorder)
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
    swap_mask = [1,-1,-1,1,-1,1]
    reflection_mask = [1,-1,-1,-1,1,1,1,-1]

    for swap, i in zip(AXIS_SWAPS, swap_mask):
        for reflection, j in zip(AXIS_REFLECTIONS, reflection_mask):
            if keep_stereo and  i * j == -1: continue # skip enantiomers

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


def reorder(reorder_method, p_coord,q_coord,p_atoms,q_atoms):

    p_cent = centroid(p_coord)
    q_cent = centroid(q_coord)
    p_coord -= p_cent
    q_coord -= q_cent

    q_review = reorder_method(p_atoms, q_atoms, p_coord, q_coord)
    reorderlist = [q_review.tolist()][0]
    #q_coord = q_coord[q_review]
    #q_atoms = q_atoms[q_review]

    #print("q_coord:", q_coord)
    #print("q_atoms:", q_atoms)
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

#QM-region expand function. Finds whole fragments.
#Used by molcrys. Similar to get_solvshell function in functions_solv.py
def QMregionfragexpand(fragment=None,initial_atoms=None, radius=None):
    #If needed (connectivity ==0):
    scale=settings_ash.scale
    tol=settings_ash.tol
    if fragment is None or initial_atoms is None or radius == None:
        print("Provide fragment, initial_atoms and radius keyword arguments to QMregionfragexpand!")
        exit()
    subsetelems = [fragment.elems[i] for i in initial_atoms]
    subsetcoords=[fragment.coords[i]for i in initial_atoms ]
    print("subsetcoords:", subsetcoords)
    print("subsetelems:", subsetelems)
    if len(fragment.connectivity) == 0:
        print("No connectivity found. Using slow way of finding nearby fragments...")
    atomlist=[]
    for i,c in enumerate(subsetcoords):
        el=subsetelems[i]
        for index,allc in enumerate(fragment.coords):
            all_el=fragment.elems[index]
            if index >= len(subsetcoords):
                dist=distance(c,allc)
                if dist < radius:
                    #Get molecule members atoms for atom index.
                    #Using stored connectivity because takes forever otherwise
                    #If no connectivity
                    if len(fragment.connectivity) == 0:
                        #wholemol=get_molecule_members_loop(fragment.coords, fragment.elems, index, 1, scale, tol)
                        wholemol=get_molecule_members_loop_np2(fragment.coords, fragment.elems, 99, scale, tol, atomindex=index)
                    #If stored connectivity
                    else:
                        for q in fragment.connectivity:
                            if index in q:
                                wholemol=q
                                break
                    elematoms=[fragment.elems[i] for i in wholemol]
                    atomlist=atomlist+wholemol
    atomlist = np.unique(atomlist).tolist()
    return atomlist

def distance_between_atoms(fragment=None, atom1=None, atom2=None):
    atom1_coords=fragment.coords[atom1]
    atom2_coords=fragment.coords[atom2]
    dist=distance(atom1_coords,atom2_coords)
    return dist



def get_boundary_atoms(qmatoms, coords, elems, scale, tol):
    # For each QM atom, do a get_conn_atoms, for those atoms, check if atoms are in qmatoms,
    # if not, then we have found an MM-boundary atom
    
    #TODO: Note, there can can be problems here if either scale, tol is non-ideal value (should be set in inputfile)
    #TODO: Or if eldict_covrad needs to be modified, also needs to be set in inputfile then.
    
    qm_mm_boundary_dict = {}
    for qmatom in qmatoms:
        #print("qmatom:", qmatom)
        connatoms = get_connected_atoms(coords, elems, scale, tol, qmatom)
        #print("connatoms:", connatoms)
        # Find connected atoms that are not in QM-atoms
        boundaryatom = listdiff(connatoms, qmatoms)
        if len(boundaryatom) > 1:
            print("Boundaryatom : ", boundaryatom)
            print("Problem. Found more than 1 boundaryatom for QM-atom {} . This is not allowed".format(qmatoms))
            exit()
        elif len(boundaryatom) == 1:
            # Adding to dict
            qm_mm_boundary_dict[qmatom] = boundaryatom[0]
    return qm_mm_boundary_dict

#Get linkatom positions for a list of qmatoms and the current set of coordinates
# Using linkatom distance of 1.08999 Å for now as default. Makes sense for C-H link atoms. Check what Chemshell does
def get_linkatom_positions(qm_mm_boundary_dict,qmatoms, coords, elems, linkatom_distance=1.09):
    
    #Get boundary atoms
    #TODO: Should we always call get_boundary_atoms or we should use previously defined dict??
    #qm_mm_boundary_dict = get_boundary_atoms(qmatoms, coords, elems, scale, tol)
    #print("qm_mm_boundary_dict :", qm_mm_boundary_dict)
    
    # Get coordinates for QMX and MMX pair. Create new L coordinate that has a modified distance to QMX
    linkatoms_dict = {}
    for dict_item in qm_mm_boundary_dict.items():
        qmatom_coords = np.array(coords[dict_item[0]])
        mmatom_coords = np.array(coords[dict_item[1]])

        linkatom_coords = list(qmatom_coords + (mmatom_coords - qmatom_coords) * (linkatom_distance / distance(qmatom_coords, mmatom_coords)))
        linkatoms_dict[(dict_item[0], dict_item[1])] = linkatom_coords
    return linkatoms_dict




#Grabbing molecules from multi-XYZ trajectory file (can be MD-file, optimization traj, nebpath traj etc).
#Creating ASH fragments for each conformer
def get_molecules_from_trajectory(file):
    print("")
    print("Now finding molecules in multi-XYZ trajectory file and creating ASH fragments...")
    list_of_molecules=[]
    all_elems, all_coords, all_titles = split_multimolxyzfile(file,writexyz=True)
    print("Found {} molecules in file".format(len(all_elems)))
    
    for els,cs in zip(all_elems,all_coords):
        conf = ash.Fragment(elems=els, coords=cs)
        list_of_molecules.append(conf)

    return list_of_molecules