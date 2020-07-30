from functions_general import *
import numpy as np
import settings_molcrys

#Elements and atom numbers
elematomnumbers = {'h':1, 'he': 2, 'li':3, 'be':4, 'b':5, 'c':6, 'n':7, 'o':8, 'f':9, 'ne':10, 'na':11, 'mg':12, 'al':13, 'si':14, 'p':15, 's':16, 'cl':17, 'ar':18, 'k':19, 'ca':20, 'sc':21, 'ti':22, 'v':23, 'cr':24, 'mn':25, 'fe':26, 'co':27, 'ni':28, 'cu':29, 'zn':30, 'ga':31, 'ge':32, 'as':33, 'se':34, 'br':35, 'kr':36, 'rb':37, 'sr':38, 'y':39, 'zr':40, 'nb':41, 'mo':42, 'tc':43, 'ru':44, 'rh':45, 'pd':46, 'ag':47, 'cd':48, 'in':49, 'sn':50, 'sb':51, 'te':52, 'i':53, 'xe':54, 'cs':55, 'ba':56, 'la':57, 'ce':58, 'pr':59, 'nd':60, 'pm':61, 'sm':62, 'eu':63, 'gd':64, 'tb':65, 'dy':66, 'ho':67, 'er':68, 'tm':69, 'yb':70, 'lu':71, 'hf':72, 'ta':73, 'w':74, 're':75, 'os':76, 'ir':77, 'pt':78, 'au':79, 'hg':80, 'tl':81, 'pb':82, 'bi':83, 'po':84, 'at':85, 'rn':86, 'fr':87, 'ra':88, 'ac':89, 'th':90, 'pa':91, 'u':92, 'np':93, 'pu':94, 'am':95, 'cm':96, 'bk':97, 'cf':98, 'es':99, 'fm':100, 'md':101, 'no':102, 'lr':103, 'rf':104, 'db':105, 'sg':106, 'bh':107, 'hs':108, 'mt':109, 'ds':110, 'rg':111, 'cn':112, 'nh':113, 'fl':114, 'mc':115, 'lv':116, 'ts':117, 'og':118}
#Atom masses
atommasses = [1.00794, 4.002602, 6.94, 9.0121831, 10.81, 12.01070, 14.00670, 15.99940, 18.99840316, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085, 30.973762, 32.065, 35.45, 39.948, 39.0983, 40.078, 44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.96, 97, 101.07, 102.9055, 106.42, 107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.905452, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145, 150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592, 204.38, 207.2, 208.9804, 209, 210, 222, 223, 226, 227, 232.0377, 231.03588, 238.02891, 237, 244, 243, 247, 247, 251, 252, 257, 258, 259, 262 ]

#Covalent radii for elements (Alvarez) in Angstrom.
#Used for connectivity
eldict_covrad={'H':0.31, 'He':0.28, 'Li':1.28, 'Be':0.96, 'B':0.84, 'C':0.76, 'N':0.71, 'O':0.66, 'F':0.57, 'Ne':0.58, 'Na':1.66, 'Mg':1.41, 'Al':1.21, 'Si':1.11, 'P':1.07, 'S':1.05, 'Cl':1.02, 'Ar':1.06, 'K':2.03, 'Ca':1.76, 'Sc':1.70, 'Ti':1.6, 'V':1.53, 'Cr':1.39, 'Mn':1.61, 'Fe':1.52, 'Co':1.50, 'Ni':1.24, 'Cu':1.32, 'Zn':1.22, 'Ga':1.22, 'Ge':1.20, 'As':1.19, 'Se':1.20, 'Br':1.20, 'Kr':1.16, 'Rb':2.2, 'Sr':1.95, 'Y':1.9, 'Zr':1.75, 'Nb':1.64, 'Mo':1.54, 'Tc':1.47, 'Ru':1.46, 'Rh':1.42, 'Pd':1.39, 'Ag':1.45, 'Cd':1.44}




#From lists of coords,elems and atom indices, print coords with elem
def print_coords_for_atoms(coords,elems,members):
    for m in members:
        print("{:4} {:8.8f}  {:8.8f}  {:8.8f}".format(elems[m],coords[m][0], coords[m][1], coords[m][2]))

#From lists of coords,elems and atom indices, print coords with elem
def print_coords_all(coords,elems):
    for i in range(len(elems)):
        print("{:4} {:8.8f}  {:8.8f}  {:8.8f}".format(elems[i],coords[i][0], coords[i][1], coords[i][2]))

# Distance between 2 atoms A, B. Using numpy arrays
def distance(A,B):
    return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)


#get_solvshell function based on single point of origin. Using geometric center of molecule
def get_solvshell_origin():
    print("to finish")
    #TODO: finish get_solvshell_origin
    exit()
#Determine threshold for whether atoms are connected or not based on covalent radii for pair of atoms
# R_ij < scale*(rad_i + rad_j) + tol
#Uses global scale and tol parameters that may be changed at input
def threshold_conn(elA,elB,scale,tol):
    covrad_A=eldict_covrad[elA]
    covrad_B=eldict_covrad[elB]
    threshold=scale*(covrad_A+covrad_B) + tol
    #if elA == 'S' and elB == 'C':
    #    print("ElA: {} ElB: {}  threshold:{}".format(elA, elB, threshold))
    return threshold

#Get connected atoms to chosen atom index based on threshold
def get_connected_atoms(coords, elems, atomindex,scale,tol):
    connatoms=[]
    coords_ref=coords[atomindex]
    elem_ref=elems[atomindex]

    for i,c in enumerate(coords):
        dist=distance(coords_ref,c)
        elA=elems[i]
        elB=elem_ref
        threshold = threshold_conn(elA, elB,scale,tol)
        #if elA == 'S' and elB == 'C':
        #    print("elA: {}  elB: {} . Dist: {}".format(elA,elB,dist))
        if dist < threshold:
            if i != atomindex:
                connatoms.append(i)
    return connatoms

#Get molecule members by running get_connected_atoms function on expanding member list
#Uses loopnumber for when to stop searching.
#Does extra work but not too bad
def get_molecule_members_loop(coords, elems, atomindex, loopnumber,scale,tol):
    members = []
    members.append(atomindex)
    connatoms = get_connected_atoms(coords, elems, atomindex,scale,tol)
    members = members + connatoms
    # How often to search for connected atoms as the members list grows:
    for i in range(loopnumber):
        for j in members:
            conn = get_connected_atoms(coords, elems, j,scale,tol)
            members = members + conn
        members = np.unique(members).tolist()
    # Remove duplicates and sort
    members = np.unique(members).tolist()
    return members

#Get-molecule-members with fixed recursion-depth of 4
#Efficient but limited to 4
def get_molecule_members_fixed(coords,elems, atomindex,scale,tol):
    members=[]
    members.append(atomindex)
    connatoms = get_connected_atoms(coords, elems, atomindex)
    members=members+connatoms
    finalmembers=members
    #How often to search for connected atoms as the members list grows:
    for j in members:
        conn=get_connected_atoms(coords, elems, j,scale,tol)
        finalmembers=finalmembers+conn
        for k in conn:
            conn2 = get_connected_atoms(coords, elems, k,scale,tol)
            finalmembers = finalmembers + conn2
            for l in conn2:
                conn3 = get_connected_atoms(coords, elems, l,scale,tol)
                finalmembers = finalmembers + conn3
    #Remove duplicates and sort
    finalmembers=np.unique(finalmembers).tolist()
    return finalmembers


#Convert cell parameters to cell vectors. Currently only works for orthorhombic, alpha=beta=gamme=90.0
def cellparamtovectors(cell_length,cell_angles):
    if cell_angles[0] == cell_angles[1] and cell_angles[2] == cell_angles[0] and cell_angles[0] == 90.0:
        cell_vectors=[[cell_length[0], 0.0, 0.0], [0.0, cell_length[1], 0.0],[0.0, 0.0, cell_length[2]]]
    else:
        print("Need to finish this")
        exit()
    return cell_vectors

#https://github.com/ghevcoul/coordinateTransform/blob/master/coordinateTransform.py
#Convert from fractional coordinates to orthogonal Cartesian coordinates in Angstrom
def fract_to_orthogonal(cellvectors, fraccoords):
    orthog = []
    for i in fraccoords:
        x = i[0]*cellvectors[0][0] + i[1]*cellvectors[1][0] + i[2]*cellvectors[2][0]
        y = i[0]*cellvectors[0][1] + i[1]*cellvectors[1][1] + i[2]*cellvectors[2][1]
        z = i[0]*cellvectors[0][2] + i[1]*cellvectors[1][2] + i[2]*cellvectors[2][2]
        orthog.append([x, y, z])
    return orthog

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

#Read CIF_file
#Grab coordinates, cell parameters and symmetry operations
def read_ciffile(file):
    cell_a=0;cell_b=0;cell_c=0;cell_alpha=0;cell_beta=0;cell_gamma=0
    atomlabels=[]
    elems=[]
    coords=[]
    symmops=[]
    newmol=False
    fractgrab=False;symmopgrab=False
    with open(file) as f:
        for line in f:
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
            if fractgrab == True:
                if '_atom_site' not in line and len(line) >5 and 'loop' not in line:
                    atomlabels.append(line.split()[0])
                    elems.append(line.split()[1])
                    coords.append([float(line.split()[2].split('(')[0]),float(line.split()[3].split('(')[0]),float(line.split()[4].split('(')[0])])
            if 'data_' in line:
                newmol = True
            if '_atom_site_fract_x' in line:
                fractgrab=True
            if '_space_group_s' in line:
                symmopgrab=True
            if '_symmetry_equiv_pos_as_xyz' in line:
                print("old syntax. Yet to do")
                exit()
    return [cell_a, cell_b, cell_c],[cell_alpha, cell_beta, cell_gamma],atomlabels,elems,coords,symmops

#From cell parameters, fractional coordinates of asymmetric unit and symmetry operations
#create fractional coordinates for atoms of whole cell
def fill_unitcell(cell_length,cell_angles,atomlabels,elems,coords,symmops):
    fullcell=[]
    for i in symmops:
        operations_x=[];operations_y=[];operations_z=[]
        #Multoperations are unity by default. Sumoperations are 0 by default
        multoperation_x=1;sumoperation_x=0
        multoperation_y=1;sumoperation_y=0
        multoperation_z=1;sumoperation_z=0
        op_x=i.split()[0].replace(",","")
        op_y=i.split()[1].replace(",","")
        op_z=i.split()[2].replace(",","")
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
                    elif zj == '+1/2':
                        sumoperation_z = 0.5
                    elif zj == '-1/2':
                        sumoperation_z = -0.5
            for c in coords:
                #print(c)
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

#Marius read_xyz
def read_xyz(xyz):
    x = []
    y = []
    z = []
    atom = []
    f = open(xyz, "r")
    f.next()
    f.next()
    for line in f:
        data = line.split()
        atom.append(data[0])
        x.append(float(data[1]))
        y.append(float(data[2]))
        z.append(float(data[3]))
    f.close()
    for item in atom:
        if len(item) == 1:
            atom[atom.index(item)] = item.replace(item[0], item[0].upper())
        if len(item) >= 2:
            atom[atom.index(item)] = item.replace(item, item[0].upper()+item[1:].lower())
    return atom, np.array(x), np.array(y), np.array(z)


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

#Calculate nuclear charge from list of atoms
def nucchargelist(ellist):
    totnuccharge=0
    els=[]
    for e in ellist:
        atcharge=elematomnumbers[e.lower()]
        totnuccharge+=atcharge
    return totnuccharge

#Calculate molecular mass from list of atoms
def totmasslist(ellist):
    totmass=0
    els=[]
    for e in ellist:
        atcharge = int(elematomnumbers[e.lower()])
        atmass=atommasses[atcharge-1]
        totmass+=atmass
    return totmass


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


def centroid(X):
    """
    Calculate the centroid from a vectorset X
    """
    C = sum(X)/len(X)
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