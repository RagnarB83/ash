from functions_general import *
import numpy as np
import settings_molcrys
import constants
#from math import sqrt
#from math import pow
import copy
import math
sqrt = math.sqrt
pow = math.pow
#Elements and atom numbers
elematomnumbers = {'h':1, 'he': 2, 'li':3, 'be':4, 'b':5, 'c':6, 'n':7, 'o':8, 'f':9, 'ne':10, 'na':11, 'mg':12, 'al':13, 'si':14, 'p':15, 's':16, 'cl':17, 'ar':18, 'k':19, 'ca':20, 'sc':21, 'ti':22, 'v':23, 'cr':24, 'mn':25, 'fe':26, 'co':27, 'ni':28, 'cu':29, 'zn':30, 'ga':31, 'ge':32, 'as':33, 'se':34, 'br':35, 'kr':36, 'rb':37, 'sr':38, 'y':39, 'zr':40, 'nb':41, 'mo':42, 'tc':43, 'ru':44, 'rh':45, 'pd':46, 'ag':47, 'cd':48, 'in':49, 'sn':50, 'sb':51, 'te':52, 'i':53, 'xe':54, 'cs':55, 'ba':56, 'la':57, 'ce':58, 'pr':59, 'nd':60, 'pm':61, 'sm':62, 'eu':63, 'gd':64, 'tb':65, 'dy':66, 'ho':67, 'er':68, 'tm':69, 'yb':70, 'lu':71, 'hf':72, 'ta':73, 'w':74, 're':75, 'os':76, 'ir':77, 'pt':78, 'au':79, 'hg':80, 'tl':81, 'pb':82, 'bi':83, 'po':84, 'at':85, 'rn':86, 'fr':87, 'ra':88, 'ac':89, 'th':90, 'pa':91, 'u':92, 'np':93, 'pu':94, 'am':95, 'cm':96, 'bk':97, 'cf':98, 'es':99, 'fm':100, 'md':101, 'no':102, 'lr':103, 'rf':104, 'db':105, 'sg':106, 'bh':107, 'hs':108, 'mt':109, 'ds':110, 'rg':111, 'cn':112, 'nh':113, 'fl':114, 'mc':115, 'lv':116, 'ts':117, 'og':118}
#Atom masses
atommasses = [1.00794, 4.002602, 6.94, 9.0121831, 10.81, 12.01070, 14.00670, 15.99940, 18.99840316, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085, 30.973762, 32.065, 35.45, 39.948, 39.0983, 40.078, 44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.96, 97, 101.07, 102.9055, 106.42, 107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.905452, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145, 150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592, 204.38, 207.2, 208.9804, 209, 210, 222, 223, 226, 227, 232.0377, 231.03588, 238.02891, 237, 244, 243, 247, 247, 251, 252, 257, 258, 259, 262 ]

#Covalent radii for elements (Alvarez) in Angstrom.
#Used for connectivity
eldict_covrad={'H':0.31, 'He':0.28, 'Li':1.28, 'Be':0.96, 'B':0.84, 'C':0.76, 'N':0.71, 'O':0.66, 'F':0.57, 'Ne':0.58, 'Na':1.66, 'Mg':1.41, 'Al':1.21, 'Si':1.11, 'P':1.07, 'S':1.05, 'Cl':1.02, 'Ar':1.06, 'K':2.03, 'Ca':1.76, 'Sc':1.70, 'Ti':1.6, 'V':1.53, 'Cr':1.39, 'Mn':1.61, 'Fe':1.52, 'Co':1.50, 'Ni':1.24, 'Cu':1.32, 'Zn':1.22, 'Ga':1.22, 'Ge':1.20, 'As':1.19, 'Se':1.20, 'Br':1.20, 'Kr':1.16, 'Rb':2.2, 'Sr':1.95, 'Y':1.9, 'Zr':1.75, 'Nb':1.64, 'Mo':1.54, 'Tc':1.47, 'Ru':1.46, 'Rh':1.42, 'Pd':1.39, 'Ag':1.45, 'Cd':1.44, 'In':1.42, 'Sn':1.39, 'Sb':1.39, 'Te':1.38, 'I':1.39, 'Xe':1.40,}

#From lists of coords,elems and atom indices, print coords with elem
def print_coords_for_atoms(coords,elems,members):
    for m in members:
        print("{:4} {:8.8f}  {:8.8f}  {:8.8f}".format(elems[m],coords[m][0], coords[m][1], coords[m][2]))

#From lists of coords,elems and atom indices, print coords with elems
#If list of atom indices provided, print as leftmost column
#If list of lables provided, print as rightmost column
def print_coords_all(coords,elems,indices=[], labels=[]):
    if indices == []:
        if labels == []:
            for i in range(len(elems)):
                print("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}".format(elems[i],coords[i][0], coords[i][1], coords[i][2]))
        else:
            for i in range(len(elems)):
                print("{:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6}".format(elems[i],coords[i][0], coords[i][1], coords[i][2], labels[i]))
    else:
        if labels == []:
            for i in range(len(elems)):
                print("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f}".format(indices[i],elems[i],coords[i][0], coords[i][1], coords[i][2]))
        else:
            for i in range(len(elems)):
                print("{:>1} {:>4} {:>12.8f}  {:>12.8f}  {:>12.8f} {:>6}".format(indices[i],elems[i],coords[i][0], coords[i][1], coords[i][2], labels[i]))


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

#Get connected atoms to chosen atom index based on threshold
#Uses slow for-loop structure with distance-function call
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
    return connatoms



#Numpy clever loop test.
#Either atomindex or membs has to be defined
def get_molecule_members_loop_np(coords, elems, loopnumber,scale,tol, atomindex='', membs=[]):
    if membs==[]:
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
def get_molecule_members_loop_np2(coords, elems, loopnumber, scale, tol, atomindex=None, membs=None):
    if membs is None:
        membs = []
        membs.append(atomindex)
        membs = get_connected_atoms_np(coords, elems, scale,tol, atomindex)
    finalmembs=membs
    for i in range(loopnumber):
        #Get list of lists of connatoms for each member
        newmembers=[get_connected_atoms_np(coords, elems, scale,tol, k) for k in membs]
        #Get a unique flat list
        trimmed_flat=np.unique([item for sublist in newmembers for item in sublist]).tolist()
        #Check if new atoms not previously found
        membs = listdiff(trimmed_flat, finalmembs)
        #Exit loop if nothing new found
        if len(membs) == 0:
            return finalmembs
        finalmembs+=membs
        finalmembs=np.unique(finalmembs).tolist()
    return finalmembs

#Get molecule members by running get_connected_atoms function on expanding member list
#Uses loopnumber for when to stop searching.
#Does extra work but not too bad
#Uses either single atomindex or members lists
def get_molecule_members_loop(coords, elems, loopnumber,scale,tol, atomindex='', members=[]):
    if members== []:
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
def get_molecule_members_fixed(coords,elems, scale,tol, atomindex='', members=[]):
    print("Disabled")
    print("not so efficient")
    exit()
    if members == []:
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
    print("Reading coordinates from XYZfile {} into fragment".format(filename))
    coords=[]
    elems=[]
    with open(filename) as f:
        for count,line in enumerate(f):
            if count == 0:
                numatoms=int(line.split()[0])
            if count > 1:
                elems.append(line.split()[0])
                coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
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
def write_xyzfile(elems,coords,name):
    with open(name+'.xyz', 'w') as ofile:
        ofile.write(str(len(elems))+'\n')
        ofile.write("title"+'\n')
        for el,c in zip(elems,coords):
            line="{:4} {:12.6f} {:12.6f} {:12.6f}".format(el,c[0], c[1], c[2])
            ofile.write(line+'\n')
    print("Wrote XYZ file:", name+'.xyz')

#Write PDBfile (dummy version) for PyFrame
def write_pdbfile_dummy(elems,coords,name, atomlabels):
    with open(name+'.pdb', 'w') as pfile:
        resnames=atomlabels
        #resnames=['QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'QM', 'HOH', 'HOH','HOH']
        resids=[1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2]
        print("ATOM      1  N   SER A   2      65.342  32.035  32.324  1.00  0.00           N")
        for count,(el,c,resname, resid) in enumerate(zip(elems,coords, resnames, resids)):
            print(count, el,c,resname, resid)
            line="{:4} {:4d} {:4} {:4} {:4d} {:8.3f} {:8.3f} {:8.3f} {:4} {:4} {:4}".format('ATOM', count+1, el,
                                                                                            resname, resid,
                                                                                            c[0], c[1], c[2],
                                                                                            '1.0', '0.00', 'X')
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

#Calculate nuclear charge from list of elements
def nucchargelist(ellist):
    totnuccharge=0
    els=[]
    for e in ellist:
        atcharge=elematomnumbers[e.lower()]
        totnuccharge+=atcharge
    return totnuccharge

#get list of nuclear charges from list of elements
#Used by Psi4
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

#HESSIAN-related functions below

#Taken from Hess-tool on 21st Dec 2019. Modified to read in Hessian array instead of ORCA-hessfile
def diagonalizeHessian(hessian, masses, elems):
    # Grab masses, elements and numatoms from Hessianfile
    #masses, elems, numatoms = masselemgrab(hessfile)
    numatoms=len(elems)
    atomlist = []
    for i, j in enumerate(elems):
        atomlist.append(str(j) + '-' + str(i))
    # Massweight Hessian
    mwhessian, massmatrix = massweight(hessian, masses, numatoms)

    # Diagonalize mass-weighted Hessian
    evalues, evectors = np.linalg.eigh(mwhessian)
    evectors = np.transpose(evectors)

    # Calculate frequencies from eigenvalues
    vfreqs = calcfreq(evalues)

    # Unweight eigenvectors to get normal modes
    nmodes = np.dot(evectors, massmatrix)
    return vfreqs,nmodes,numatoms,elems,evectors,atomlist,masses


#Massweight Hessian
def massweight(matrix,masses,numatoms):
    mass_mat = np.zeros( (3*numatoms,3*numatoms), dtype = float )
    molwt = [ masses[int(i)] for i in range(numatoms) for j in range(3) ]
    for i in range(len(molwt)):
        mass_mat[i,i] = molwt[i] ** -0.5
    mwhessian = np.dot((np.dot(mass_mat,matrix)),mass_mat)
    return mwhessian,mass_mat

#Calculate frequencies from eigenvalus
def calcfreq(evalues):
    hartree2j = constants.hartree2j
    bohr2m = constants.bohr2m
    amu2kg = constants.amu2kg
    c = constants.c
    pi = constants.pi
    evalues_si = [val*hartree2j/bohr2m/bohr2m/amu2kg for val in evalues]
    vfreq_hz = [1/(2*pi)*np.sqrt(np.complex_(val)) for val in evalues_si]
    vfreq = [val/c for val in vfreq_hz]
    return vfreq


# Function to print normal mode composition factors for all atoms, element-groups, specific atom groups or specific atoms
def printfreqs(vfreq,numatoms):
    if numatoms == 2:
        TRmodenum=5
    else:
        TRmodenum=6
    line = "{:>4}{:>14}".format("Mode", "Freq(cm**-1)")
    print(line)
    for mode in range(0,3*numatoms):
        if mode < TRmodenum:
            line = "{:>3d}   {:>9.4f}".format(mode,0.000)
            print(line)
        else:
            vib=clean_number(vfreq[mode])
            line = "{:>3d}   {:>9.4f}".format(mode, vib)
            print(line)

# Function to print normal mode composition factors for all atoms, element-groups, specific atom groups or specific atoms
def printnormalmodecompositions(option,TRmodenum,vfreq,numatoms,elems,evectors,atomlist):
    # Normalmodecomposition factors for mode j and atom a
    freqs=[]
    # If one set of normal atom compositions (1 atom or 1 group)
    comps=[]
    # If multiple (case: all or elements)
    allcomps=[]
    # Change TRmodenum to 5 if diatomic molecule since linear case
    if numatoms==2:
        TRmodenum=5

    if option=="all":
        # Case: All atoms
        line = "{:>4}{:>14}      {:}".format("Mode", "Freq(cm**-1)", '       '.join(atomlist))
        print(line)
        for mode in range(0,3*numatoms):
            normcomplist=[]
            if mode < TRmodenum:
                line = "{:>3d}   {:>9.4f}".format(mode,0.000)
                print(line)
            else:
                vib=clean_number(vfreq[mode])
                freqs.append(float(vib))
                for n in range(0, numatoms):
                    normcomp=normalmodecomp(evectors,mode,n)
                    normcomplist.append(normcomp)
                allcomps.append(normcomplist)
                normcomplist=['{:.6f}'.format(x) for x in normcomplist]
                line = "{:>3d}   {:>9.4f}        {}".format(mode, vib, '   '.join(normcomplist))
                print(line)
    elif option=="elements":
        # Case: By elements
        uniqelems=[]
        for i in elems:
            if i not in uniqelems:
                uniqelems.append(i)
        line = "{:>4}{:>14}      {:45}".format("Mode", "Freq(cm**-1)", '         '.join(uniqelems))
        print(line)
        for mode in range(0,3*numatoms):
            normcomplist=[]
            if mode < TRmodenum:
                line = "{:>3d}   {:>9.4f}".format(mode,0.000)
                print(line)
            else:
                vib=clean_number(vfreq[mode])
                freqs.append(float(vib))
                for n in range(0,numatoms):
                    normcomp=normalmodecomp(evectors,mode,n)
                    normcomplist.append(normcomp)
                elementnormcomplist=[]
                # Sum components together
                for u in uniqelems:
                    elcompsum=0.0
                    elindices=[i for i, j in enumerate(elems) if j == u]
                    for h in elindices:
                        elcompsum=float(elcompsum+float(normcomplist[h]))
                    elementnormcomplist.append(elcompsum)
                # print(elementnormcomplist)
                allcomps.append(elementnormcomplist)
                elementnormcomplist=['{:.6f}'.format(x) for x in elementnormcomplist]
                line = "{:>3d}   {:>9.4f}        {}".format(mode, vib, '   '.join(elementnormcomplist))
                print(line)
    elif isint(option)==True:
        # Case: Specific atom
        atom=int(option)
        if atom > numatoms-1:
            print(bcolors.FAIL, "Atom index does not exist. Note: Numbering starts from 0", bcolors.ENDC)
            exit()
        line = "{:>4}{:>14}      {:45}".format("Mode", "Freq(cm**-1)", atomlist[atom])
        print(line)
        for mode in range(0,3*numatoms):
            normcomplist=[]
            if mode < TRmodenum:
                line = "{:>3d}   {:>9.4f}".format(mode,0.000)
                print(line)
            else:
                vib=clean_number(vfreq[mode])
                freqs.append(float(vib))
                for n in range(0, numatoms):
                    normcomp=normalmodecomp(evectors,mode,n)
                    normcomplist.append(normcomp)
                comps.append(normcomplist[atom])
                normcomplist=['{:.6f}'.format(x) for x in normcomplist]
                line = "{:>3d}   {:>9.4f}        {}".format(mode, vib, normcomplist[atom])
                print(line)
    elif len(option.split(",")) > 1:
        # Case: Chemical group defined as list of atoms
        selatoms = option.split(",")
        selatoms=[int(i) for i in selatoms]
        grouplist=[]
        for at in selatoms:
            if at > numatoms-1:
                print(bcolors.FAIL,"Atom index does not exist. Note: Numbering starts from 0",bcolors.ENDC)
                exit()
            grouplist.append(atomlist[at])
        simpgrouplist='_'.join(grouplist)
        grouplist=', '.join(grouplist)
        line = "{}   {}    {}".format("Mode", "Freq(cm**-1)", "Group("+grouplist+")")
        print(line)
        for mode in range(0,3*numatoms):
            normcomplist=[]
            if mode < TRmodenum:
                line = "{:>3d}   {:>9.4f}".format(mode,0.000)
                print(line)
            else:
                vib=clean_number(vfreq[mode])
                freqs.append(float(vib))
                for n in range(0,numatoms):
                    normcomp=normalmodecomp(evectors,mode,n)
                    normcomplist.append(normcomp)
                # normcomplist=['{:.6f}'.format(x) for x in normcomplist]
                groupnormcomplist=[]
                for q in selatoms:
                    groupnormcomplist.append(normcomplist[q])
                comps.append(sum(groupnormcomplist))
                sumgroupnormcomplist='{:.6f}'.format(sum(groupnormcomplist))
                line = "{:>3d}   {:9.4f}        {}".format(mode, vib, sumgroupnormcomplist)
                print(line)
    else:
        print("Something went wrong")

    return allcomps,comps,freqs


#Give normal mode composition factors for mode j and atom a
def normalmodecomp(evectors,j,a):
    #square elements of mode j
    esq_j=[i ** 2 for i in evectors[j]]
    #Squared elements of atom a in mode j
    esq_ja=[]
    esq_ja.append(esq_j[a*3+0]);esq_ja.append(esq_j[a*3+1]);esq_ja.append(esq_j[a*3+2])
    return sum(esq_ja)

# Write normal mode as XYZ-trajectory.
# Read in normalmode vectors from diagonalized mass-weighted Hessian after unweighting.
# Print out XYZ-trajectory of mode
def write_normalmodeXYZ(nmodes,Rcoords,modenumber,elems):
    if modenumber > len(nmodes):
        print("Modenumber is larger than number of normal modes. Exiting. (Note: We count from 0.)")
        return
    else:
        # Modenumber: number mode (starting from 0)
        modechosen=nmodes[modenumber]

    # hessatoms: list of atoms involved in Hessian. Usually all atoms. Unnecessary unless QM/MM?
    # Going to disable hessatoms as an argument for now but keep it in the code like this:
    hessatoms=list(range(1,len(elems)))

    #Creating dictionary of displaced atoms and chosen mode coordinates
    modedict = {}
    # Convert ndarray to list for convenience
    modechosen=modechosen.tolist()
    for fo in range(0,len(hessatoms)):
        modedict[hessatoms[fo]] = [modechosen.pop(0),modechosen.pop(0),modechosen.pop(0)]
    f = open('Mode'+str(modenumber)+'.xyz','w')
    # Displacement array
    dx = np.array([0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,0.4,0.3,0.2,0.1,0.0])
    dim=len(modechosen)
    for k in range(0,len(dx)):
        f.write('%i\n\n' % len(hessatoms))
        for j,w in zip(range(0,numatoms),Rcoords):
            if j+1 in hessatoms:
                f.write('%s %12.8f %12.8f %12.8f  \n' % (elems[j], (dx[k]*modedict[j+1][0]+w[0]), (dx[k]*modedict[j+1][1]+w[1]), (dx[k]*modedict[j+1][2]+w[2])))
    f.close()
    print("All done. File Mode%s.xyz has been created!" % (modenumber))

# Compare the similarity of normal modes by cosine similarity (normalized dot product of normal mode vectors).
#Useful for isotope-substitutions. From Hess-tool.
def comparenormalmodes(hessianA,hessianB,massesA,massesB):
    numatoms=len(massesA)
    # Massweight Hessians
    mwhessianA, massmatrixA = massweight(hessianA, massesA, numatoms)
    mwhessianB, massmatrixB = massweight(hessianB, massesB, numatoms)

    # Diagonalize mass-weighted Hessian
    evaluesA, evectorsA = la.eigh(mwhessianA)
    evaluesB, evectorsB = la.eigh(mwhessianB)
    evectorsA = np.transpose(evectorsA)
    evectorsB = np.transpose(evectorsB)

    # Calculate frequencies from eigenvalues
    vfreqA = calcfreq(evaluesA)
    vfreqB = calcfreq(evaluesB)

    print("")
    # Unweight eigenvectors to get normal modes
    nmodesA = np.dot(evectorsA, massmatrixA)
    nmodesB = np.dot(evectorsB, massmatrixB)
    line = "{:>4}".format("Mode  Freq-A(cm**-1)  Freq-B(cm**-1)    Cosine-similarity")
    print(line)
    for mode in range(0, 3 * numatoms):
        if mode < TRmodenum:
            line = "{:>3d}   {:>9.4f}       {:>9.4f}".format(mode, 0.000, 0.000)
            print(line)
        else:
            vibA = clean_number(vfreqA[mode])
            vibB = clean_number(vfreqB[mode])
            cos_sim = np.dot(nmodesA[mode], nmodesB[mode]) / (
                        np.linalg.norm(nmodesA[mode]) * np.linalg.norm(nmodesB[mode]))
            if abs(cos_sim) < 0.9:
                line = "{:>3d}   {:>9.4f}       {:>9.4f}          {:.3f} {}".format(mode, vibA, vibB, cos_sim, "<------")
            else:
                line = "{:>3d}   {:>9.4f}       {:>9.4f}          {:.3f}".format(mode, vibA, vibB, cos_sim)
            print(line)



#Get partial matrix by deleting rows not present in list of indices.
#Deletes numpy rows
def get_partial_matrix(allatoms,hessatoms,matrix):
    nonhessatoms=listdiff(allatoms,hessatoms)
    nonhessatoms.reverse()
    for at in nonhessatoms:
        matrix=np.delete(matrix, at, 0)
    return matrix

#Get partial list by deleting elements not present in provided list of indices.
def get_partial_list(allatoms,partialatoms,list):
    otheratoms=listdiff(allatoms,partialatoms)
    otheratoms.reverse()
    for at in otheratoms:
        del list[at]
    return list

#list of frequencies and fragment object
#TODO: Make sure distinction between initial coords and optimized coords?
def thermochemcalc(vfreq,hessatoms,fragment, multiplicity, temp=298.18,pressure=1):
    if len(hessatoms) == 2:
        TRmodenum=5
    else:
        TRmodenum=6
    elems=fragment.elems
    masses=fragment.list_of_masses
    #Total atomlist from fragment object. Not hessatoms.
    #atomlist=fragment.atomlist

    #Some constants
    joule_to_hartree=2.293712317E+17
    c_cm_s=29979245800.00
    #Planck's constant in Js
    h_planck_Js=6.62607015000000E-34
    h_planck_hartreeseconds=1.5198298716361000E-16

    #0.5*h*c: 0.5 * h_planck_hartreeseconds*29979245800cm/s  hartree cm
    halfhcfactor=2.27816766479806E-06
    # R in hartree/K. Converted from 8.31446261815324000 J/Kmol
    R_gasconst=3.16681161675373E-06
    #Boltzmann's constant. Converted from 1.380649000E-23 J/K to hartree/K
    k_b=3.16681161675373E-06


    freqs=[]
    vibtemps=[]
    #Vibrational part
    print(vfreq)
    for mode in range(0, 3 * len(hessatoms)):
        print(mode)
        if mode < TRmodenum:
            continue
            #print("skipping TR mode with freq:", clean_number(vfreq[mode]) )
        else:
            vib = clean_number(vfreq[mode])
            freqs.append(float(vib))
            freq_Hz=vib*c_cm_s
            vibtemp=(h_planck_hartreeseconds * freq_Hz) / k_b
            vibtemps.append(vibtemp)


    #print(vibtemps)

    #Zero-point vibrational energy
    zpve=sum([i*halfhcfactor for i in freqs])

    #Thermal vibrational energy
    sumb=0
    for v in vibtemps:
        #print(v*(0.5+(1/(np.exp((v/temp) - 1)))))
        sumb=sumb+v*(0.5+(1/(np.exp((v/temp) - 1))))
    vibenergy=sumb*R_gasconst
    vibenergycorr=vibenergy-zpve

    #Moments of inertia and rotational part


    rotenergy=R_gasconst*temp

    #Translational part
    transenergy=1.5*R_gasconst*temp

    #Mass stuff
    totalmass=sum(masses)
    print("")

    ###############
    # ENTROPY TERMS:
    #
    # https://github.com/eljost/thermoanalysis/blob/89b28941520fdeee1c96315b1900e124f094df49/thermoanalysis/thermo.py#L74

    # Compare to this: https://github.com/eljost/thermoanalysis/blob/89b28941520fdeee1c96315b1900e124f094df49/thermoanalysis/thermo.py#L46
    #Electronic entropy
    #TODO: KBAU?
    #S_el = KBAU * np.log(multiplicity)




    print("Thermochemistry")
    print("--------------------")
    print("Temperature:", temp, "K")
    print("Pressure:", pressure, "atm")
    print("Total atomlist:", fragment.atomlist)
    print("Hessian atomlist:", hessatoms)
    print("Masses:", masses)
    print("Total mass:", totalmass)
    print("")
    #  stuff
    print("Moments of inertia:")
    print("Rotational constants:")

    print("")
    #Thermal corrections
    print("Energy corrections:")
    print("Zero-point vibrational energy:", zpve)
    print("{} {} {} {}".format("Translational energy (", temp, "K) :", transenergy))
    print("{} {} {} {}".format("Rotational energy (", temp, "K) :", rotenergy))
    print("{} {} {} {}".format("Total vibrational energy (", temp, "K) :", vibenergy))
    print("{} {} {} {}".format("Vibrational energy correction (", temp, "K) :", vibenergycorr))


def hungarian(A, B):
    """
    Hungarian reordering.

    Assume A and B are coordinates for atoms of SAME type only
    """

    # should be kabasch here i think
    #TODO: get rid of cdist and linear_sum_assignment
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    distances = cdist(A, B, 'euclidean')

    # Perform Hungarian analysis on distance matrix between atoms of 1st
    # structure and trial structure
    indices_a, indices_b = linear_sum_assignment(distances)

    return indices_b

#Hungarian reorder algorithm
def reorder_hungarian(p_atoms, q_atoms, p_coord, q_coord):
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

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)
    view_reorder -= 1

    for atom in unique_atoms:
        p_atom_idx, = np.where(p_atoms == atom)
        q_atom_idx, = np.where(q_atoms == atom)

        A_coord = p_coord[p_atom_idx]
        B_coord = q_coord[q_atom_idx]

        view = hungarian(A_coord, B_coord)
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder



def check_reflections(p_atoms, q_atoms, p_coord, q_coord,
                      reorder_method=reorder_hungarian,
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