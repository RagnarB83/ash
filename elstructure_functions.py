import numpy as np
import functions_coords
import os
import glob
import ash
import subprocess as sp
import shutil

#CM5. from https://github.com/patrickmelix/CM5-calculator/blob/master/cm5calculator.py

#data from paper for element 1-118
_radii = np.array([0.32, 0.37, 1.30, 0.99, 0.84, 0.75,
          0.71, 0.64, 0.60, 0.62, 1.60, 1.40,
          1.24, 1.14, 1.09, 1.04, 1.00, 1.01,
          2.00, 1.74, 1.59, 1.48, 1.44, 1.30,
          1.29, 1.24, 1.18, 1.17, 1.22, 1.20,
          1.23, 1.20, 1.20, 1.18, 1.17, 1.16,
          2.15, 1.90, 1.76, 1.64, 1.56, 1.46,
          1.38, 1.36, 1.34, 1.30, 1.36, 1.40,
          1.42, 1.40, 1.40, 1.37, 1.36, 1.36,
          2.38, 2.06, 1.94, 1.84, 1.90, 1.88,
          1.86, 1.85, 1.83, 1.82, 1.81, 1.80,
          1.79, 1.77, 1.77, 1.78, 1.74, 1.64,
          1.58, 1.50, 1.41, 1.36, 1.32, 1.30,
          1.30, 1.32, 1.44, 1.45, 1.50, 1.42,
          1.48, 1.46, 2.42, 2.11, 2.01, 1.90,
          1.84, 1.83, 1.80, 1.80, 1.73, 1.68,
          1.68, 1.68, 1.65, 1.67, 1.73, 1.76,
          1.61, 1.57, 1.49, 1.43, 1.41, 1.34,
          1.29, 1.28, 1.21, 1.22, 1.36, 1.43,
          1.62, 1.75, 1.65, 1.57])


_Dz = np.array([0.0056, -0.1543, 0.0000, 0.0333, -0.1030, -0.0446,
      -0.1072, -0.0802, -0.0629, -0.1088, 0.0184, 0.0000,
      -0.0726, -0.0790, -0.0756, -0.0565, -0.0444, -0.0767,
       0.0130, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      -0.0512, -0.0557, -0.0533, -0.0399, -0.0313, -0.0541,
       0.0092, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      -0.0361, -0.0393, -0.0376, -0.0281, -0.0220, -0.0381,
       0.0065, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, -0.0255, -0.0277, -0.0265, -0.0198,
      -0.0155, -0.0269, 0.0046, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, -0.0179, -0.0195,
      -0.0187, -0.0140, -0.0110, -0.0189])

_alpha = 2.474
_C = 0.705
_DHC = 0.0502
_DHN = 0.1747
_DHO = 0.1671
_DCN = 0.0556
_DCO = 0.0234
_DNO = -0.0346


#Get list-of-lists of distances of coords
def distance_matrix_from_coords(coords):
    distmatrix=[]
    for i in coords:
        dist_row=[functions_coords.distance(i,j) for j in coords]
        distmatrix.append(dist_row)
    return distmatrix
            


def calc_cm5(atomicNumbers, coords, hirschfeldcharges):
    coords=np.array(coords)
    atomicNumbers=np.array(atomicNumbers)
    #all matrices have the naming scheme matrix[k,k'] according to the paper
    #distances = atoms.get_all_distances(mic=True)
    distances = np.array(distance_matrix_from_coords(coords))
    print("distances:", distances)
    #atomicNumbers = np.array(atoms.numbers)
    print("atomicNumbers", atomicNumbers)
    Rz = _radii[atomicNumbers-1]
    RzSum = np.tile(Rz,(len(Rz),1))
    RzSum = np.add(RzSum, np.transpose(RzSum))
    Bkk = np.exp(-_alpha * (np.subtract(distances,RzSum)), out=np.zeros_like(distances), where=distances!=0)
    assert (np.diagonal(Bkk) == 0).all()

    Dz = _Dz[atomicNumbers]
#    Tkk = np.tile(Dz,(len(Dz),1))
#    Tkk = np.subtract(Tkk, np.transpose(Tkk))
    Tkk = np.zeros(shape=Bkk.shape)
    shape = Tkk.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            numbers = [atomicNumbers[i], atomicNumbers[j]]
            if numbers[0] == numbers[1]:
                continue
            if set(numbers) == set([1,6]):
                Tkk[i,j] = _DHC
                if numbers == [6,1]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([1,7]):
                Tkk[i,j] = _DHN
                if numbers == [7,1]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([1,8]):
                Tkk[i,j] = _DHO
                if numbers == [8,1]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([6,7]):
                Tkk[i,j] = _DCN
                if numbers == [7,6]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([6,8]):
                Tkk[i,j] = _DCO
                if numbers == [8,6]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([7,8]):
                Tkk[i,j] = _DNO
                if numbers == [8,7]:
                    Tkk[i,j] *= -1.0
            else:
                Tkk[i,j] = _Dz[numbers[0]-1] - _Dz[numbers[1]-1]
    assert (np.diagonal(Tkk) == 0).all()
    product = np.multiply(Tkk, Bkk)
    assert (np.diagonal(product) == 0).all()
    result = np.sum(product,axis=1)
    print("hirschfeldcharges:", hirschfeldcharges)
    print("result:", result)
    print(type(result))
    return np.array(hirschfeldcharges) + result


#Read cubefile.
def read_cube (cubefile):
    bohrang = 0.52917721067
    LargePrint = True
    #Opening orbital cube file
    try:
        filename = cubefile
        a = open(filename,"r")
        print("Reading orbital file:", filename)
        filebase=os.path.splitext(filename)[0]
    except IndexError:
        print("error")
        quit()
    #Read cube file and get all data. Square values
    count = 0
    X = False
    d = []
    vals=[]
    elems=[]
    molcoords=[]
    molcoords_ang=[]
    numatoms=0
    for line in a:
        count += 1
        words = line.split()
        numwords=len(words)
        #Grabbing origin
        if count == 3:
            numatoms=abs(int(line.split()[0]))
            orgx=float(line.split()[1])
            orgy=float(line.split()[2])
            orgz=float(line.split()[3])
            rlowx=orgx;rlowy=orgy;rlowz=orgz
        if count == 4:
            nx=int(line.split()[0])
            dx=float(line.split()[1])
        if count == 5:
            ny=int(line.split()[0])
            dy=float(line.split()[2])
        if count == 6:
            nz=int(line.split()[0])
            dz=float(line.split()[3])
        #Grabbing molecular coordinates
        if count > 6 and count <= 6+numatoms:
            elems.append(int(line.split()[0]))
            molcoord=[float(line.split()[2]),float(line.split()[3]),float(line.split()[4])]
            molcoord_ang=[bohrang*float(line.split()[2]),bohrang*float(line.split()[3]),bohrang*float(line.split()[4])]
            molcoords.append(molcoord)
            molcoords_ang.append(molcoord_ang)
        # reading gridpoints
        if X == True:
            b = line.rstrip('\n').replace('  ', ' ').replace('  ', ' ').split(' ')
            b=list(filter(None, b))
            c =[float(i) for i in b]
            #print("c is", c)
            #exit()
            #dbq = [float('%.5e' % i) for i in c]
            if len(c) >0:
                vals.append(c)
        # when to begin reading gridpoints
        if (count >= 6+numatoms and X==False):
            X = True
    if LargePrint==True:
        print("Number of orb/density points:", len(vals))
    return rlowx,dx,nx,orgx,rlowy,dy,ny,orgy,rlowz,dz,nz,orgz,elems,molcoords,molcoords_ang,numatoms,filebase,vals

#Subtract one Cube-file from another
def write_cube_diff(numatoms,orgx,orgy,orgz, nx,dx,ny,dy,nz,dz,elems, molcoords, val1,val2,name):
    with open(name+".cube", 'w') as file:
        file.write("Cube file generated by ASH\n")
        file.write("Density difference\n")
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(numatoms,orgx,orgy,orgz))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(nx,dx,0.0,0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(ny,0.0,dy,0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(nz,0.0,0.0,dz))
        for el,c in zip(elems,molcoords):
            file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(el,el,c[0],c[1],c[2]))
        for v1,v2 in zip(val1,val2):
            diff = [i-j for i,j in zip(v1,v2)]

            if len(v1) == 6:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(diff[0],diff[1],diff[2],diff[3],diff[4],diff[5]))
            elif len(v1) == 5:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(diff[0],diff[1],diff[2],diff[3],diff[4]))
            elif len(v1) == 4:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(diff[0],diff[1],diff[2],diff[3]))
            elif len(v1) == 3:
                file.write("   {:.6e}   {:.6e}   {:.6e}\n".format(diff[0],diff[1],diff[2]))
            elif len(v1) == 2:
                file.write("   {:.6e}   {:.6e}\n".format(diff[0],diff[1]))
            elif len(v1) == 1:
                file.write("   {:.6e}\n".format(diff[0]))


#Read cubefile. Grabs coords. Calculates density if MO
def create_density_from_orb (cubefile):
    global bohrang
    #Opening orbital cube file
    try:
        filename = cubefile
        a = open(filename,"r")
        print("Reading orbital file:", filename)
        filebase=os.path.splitext(filename)[0]
    except IndexError:
        print("error")
        quit()
    if denswrite==True:
        #Write orbital density cube file
        output = open(filebase+'-dens.cube', "w")
    #Read cube file and get all data. Square values
    count = 0
    X = False
    d = []
    densvals = []
    orbvals=[]
    elems=[]
    molcoords=[]
    molcoords_ang=[]
    numatoms=0
    for line in a:
        count += 1
        words = line.split()
        numwords=len(words)
        #Grabbing origin
        if count < 3:
            if denswrite==True:
                output.write(line)
        if count == 3:
            numatoms=abs(int(line.split()[0]))
            orgx=float(line.split()[1])
            orgy=float(line.split()[2])
            orgz=float(line.split()[3])
            rlowx=orgx;rlowy=orgy;rlowz=orgz
            if denswrite==True:
                output.write(line)
        if count == 4:
            nx=int(line.split()[0])
            dx=float(line.split()[1])
            if denswrite==True:
                output.write(line)
        if count == 5:
            ny=int(line.split()[0])
            dy=float(line.split()[2])
            if denswrite==True:
                output.write(line)
        if count == 6:
            nz=int(line.split()[0])
            dz=float(line.split()[3])
            if denswrite==True:
                output.write(line)
        #Grabbing molecular coordinates
        if count > 6 and count <= 6+numatoms:
            elems.append(int(line.split()[0]))
            molcoord=[float(line.split()[2]),float(line.split()[3]),float(line.split()[4])]
            molcoord_ang=[bohrang*float(line.split()[2]),bohrang*float(line.split()[3]),bohrang*float(line.split()[4])]
            molcoords.append(molcoord)
            molcoords_ang.append(molcoord_ang)
            if denswrite==True:
                output.write(line)
        # reading gridpoints
        if X == True:
            b = line.rstrip('\n').replace('  ', ' ').replace('  ', ' ').split(' ')
            b=list(filter(None, b))
            c =[float(i) for i in b]
            #print("c is", c)
            #Squaring orbital values to get density
            csq = [q** 2 for q in c]
            dsq = [float('%.5e' % i) for i in csq]
            densvals.append(dsq)
            dbq = [float('%.5e' % i) for i in c]
            orbvals.append(dbq)
        # when to begin reading gridpoints
        if (count > 6 and numwords == 2 and X==False):
            X = True
            if denswrite==True:
                output.write(line)

    # Go through orb and dens list and print out density file
    alldensvalues=[]
    allorbvalues=[]
    for line in densvals:
        columns = ["%13s" % cell for cell in line]
        for val in columns:
            alldensvalues.append(float(val))
        if denswrite==True:
            linep=' '.join( columns)
            output.write(linep+'\n')

    for line in orbvals:
        dolumns = ["%13s" % cell for cell in line]
        for oval in dolumns:
            allorbvalues.append(float(oval))
    if denswrite==True:
        output.close()
        print("Wrote orbital density file as:", filebase+'-dens.cube')
        print("")
    sumdensvalues=sum(i for i in alldensvalues)
    if LargePrint==True:
        print("Sum of density values is:", sumdensvalues)
        print("Number of density values is", len(alldensvalues))
        print("Number of orb values is", len(allorbvalues))
    return rlowx,dx,nx,orgx,rlowy,dy,ny,orgy,rlowz,dz,nz,orgz,alldensvalues,elems,molcoords_ang,numatoms,filebase


def centroid_calc (rlowx,dx,nx,orgx,rlowy,dy,ny,orgy,rlowz,dz,nz,orgz,alldensvalues ):
    #########################################################
    # Calculate centroid. Based on Multiwfn
    ############################################################

    #Largest x,y,z coordinates
    rhighx=rlowx+(dx*(nx-1))
    rhighy=rlowy+(dy*(ny-1))
    rhighz=rlowz+(dz*(nz-1))
    #Lowest and highest density values
    rlowv = min(float(s) for s in alldensvalues)
    rhighv = max(float(s) for s in alldensvalues)

    sumuppos=0.0
    cenxpos=0.0
    cenypos=0.0
    cenzpos=0.0
    vcount=0

    #print ("dx, dy, dz is", dx, dy, dz)
    #print("range of x:", rlowx, rhighx)
    #print("range of y:", rlowy, rhighy)
    #print("range of z:", rlowz, rhighz)

    for i in range(1,nx+1):
        if (orgx+(i-1)*dx)<rlowx or (orgx+(i-1)*dx)>rhighx:
            print("If statement. Look into. x")
            exit()
            continue
        for j in range(1,ny+1):
            if (orgy+(j-1)*dy)<rlowy or (orgy+(j-1)*dy)>rhighy:
                print("If statement. Look into. y")
                exit()
                continue
            for k in range(1,nz+1):
                if (orgz+(k-1)*dz)<rlowz or (orgz+(k-1)*dz)>rhighz:
                    print("If statement. Look into. z")
                    exit()
                    continue
                #print("i,j,k is", i,j,k)
                valtmp=alldensvalues[vcount]
                if valtmp<rlowv or valtmp>rhighv:
                    print("If statement. Look into. v")
                    exit()
                    continue
                if valtmp>0:
                    sumuppos=sumuppos+valtmp
                    #print("sumuppos is", sumuppos)
                    cenxpos=cenxpos+(orgx+(i-1)*dx)*valtmp
                    cenypos=cenypos+(orgy+(j-1)*dy)*valtmp
                    cenzpos=cenzpos+(orgz+(k-1)*dz)*valtmp
                    #print("valtmp is", valtmp)
                    #print("-----------------------")
                vcount+=1

    #Final values
    cenxpos=cenxpos/sumuppos
    cenypos=cenypos/sumuppos
    cenzpos=cenzpos/sumuppos
    return cenxpos,cenypos,cenzpos

# MO-DOS PLOT. Multiply MO energies by -1 and sort.
def modosplot(occorbs_alpha,occorbs_beta,hftyp):
    #Defining sticks as -1 times MO energy (eV)
    stk_alpha=[]
    stk_beta=[]
    for j in occorbs_alpha:
        stk_alpha.append(-1*j)
    if hftyp == "UHF":
        for k in occorbs_beta:
            stk_beta.append(-1*k)
        stk_beta.sort()
    stk_alpha.sort()
    return stk_alpha,stk_beta

#Calculate HOMO number from nuclear charge from XYZ-file and total charge
def HOMOnumbercalc(file,charge,mult):
    el=[]
    with open(file) as f:
        for count,line in enumerate(f):
            if count >1:
                el.append(line.split()[0])
    totnuccharge=0
    for e in el:
        atcharge=eldict[e]
        totnuccharge+=atcharge
    numel=totnuccharge-charge
    HOMOnum_a="unset";HOMOnum_b="unset"
    orcaoffset=-1
    if mult == 1:
        #RHF case. HOMO is numel/2 -1
        HOMOnum_a=(numel/2)+orcaoffset
        HOMOnum_b=(numel/2)+orcaoffset
    elif mult > 1:
        #UHF case.
        numunpel=mult-1
        Doubocc=(numel-numunpel)/2
        HOMOnum_a=Doubocc+numunpel+orcaoffset
        HOMOnum_b=Doubocc+orcaoffset
    return int(HOMOnum_a),int(HOMOnum_b)

#Calculate DDEC charges and derive LJ parameters from ORCA.
#Uses Chargemol program
# Uses ORCA to calculate densities of molecule and its free atoms. Uses orca_2mkl to create Molden file and molden2aim to create WFX file from Molden.
# Wfx file is read into Chargemol program for DDEC analysis which radial moments used to compute C6 parameters and radii for Lennard-Jones equation.
def DDEC_calc(fragment=None, theory=None, ncores=1, DDECmodel='DDEC3'):
    print("Warning: DDEC_calc requires chargemol-binary dir to be present in environment PATH variable.")
    #molden2aim=None

    #Finding chargemoldir from PATH in os.path
    PATH=os.environ.get('PATH').split(':')
    print("PATH: ", PATH)
    print("Searching for chargemol in PATH")
    for p in PATH:
        if 'chargemol' in p:
            print("Found chargemol in path line (this dir should contain the executables):", p)
            chargemolbinarydir=p


    #Finding molden2aim in PATH:
   # Below works if dir is in PATH
    molden2aim="molden2aim.exe"

    #Defining Chargemoldir (main dir) as 3-up from binary dir
    var=os.path.split(chargemolbinarydir)[0]
    var=os.path.split(var)[0]
    chargemoldir=os.path.split(var)[0]
    print("Chargemoldir (base directory): ", chargemoldir)
    print("Chargemol binary dir:", chargemolbinarydir)

    os.mkdir('DDEC_calc')
    os.chdir('DDEC_calc')

    if fragment is None or theory is None :
        print("DDEC_calc requires fragment, theory, keyword arguments")
        exit(1)
    if theory.__class__.__name__ != "ORCATheory":
        print("Only ORCA is supported as theory in DDEC_calc currently")
        exit(1)

    # What DDEC charge model to use. Jorgensen paper uses DDEC3. DDEC6 is the newer recommended chargemodel
    # Set variable to 'DDEC3' or 'DDEC6'
    print("DDEC model:", DDECmodel)

    # What oxygen LJ parameters to use in pair-pair parameters.
    # Choices: TIP3P, Chargemol, Manual
    H2Omodel = 'TIP3P'

    print("DDEC calc")

    #bindir=glob.glob('*chargemol*')[0]
    #chargemolbinarydir=chargemoldir+bindir+'compiled_binaries'+'linux'


    # Serial or parallel version
    if ncores == 1:
        print("Using serial version of Chargemol")
        chargemol=glob.glob(chargemolbinarydir+'/*serial*')[0]
        #chargemol=chargemolbinarydir+glob.glob('*serial*')[0]
    else:
        print("Using parallel version of Chargemol using {} cores".format(ncores))
        #chargemol=chargemolbinarydir+glob.glob('*parallel*')[0]
        chargemol=glob.glob(chargemolbinarydir+'/*parallel*')[0]
        # Parallelization of Chargemol code. 8 should be good.
        os.environ['OMP_NUM_THREADS'] = str(ncores)
    print("Using Chargemoldir executable: ", chargemol)


    #Dictionary for spin multiplicities of atoms
    spindictionary = {'H':2, 'He': 1, 'Li':2, 'Be':1, 'B':2, 'C':3, 'N':4, 'O':3, 'F':2, 'Ne':1, 'Na':2, 'Mg':1, 'Al':2, 'Si':3, 'P':4, 'S':3, 'Cl':2, 'Ar':1, 'K':2, 'Ca':1, 'Sc':2, 'Ti':3, 'V':4, 'Cr':7, 'Mn':6, 'Fe':5, 'Co':4, 'Ni':3, 'Cu':2, 'Zn':1, 'Ga':2, 'Ge':3, 'As':4, 'Se':3, 'Br':2, 'Kr':1, 'Rb':2, 'Sr':1, 'Y':2, 'Zr':3, 'Nb':6, 'Mo':7, 'Tc':6, 'Ru':5, 'Rh':4, 'Pd':1, 'Ag':2, 'Cd':1, 'In':2, 'Sn':3, 'Sb':4, 'Te':3, 'I':2, 'Xe':1, 'Cs':2, 'Ba':1, 'La':2, 'Ce':1, 'Pr':4, 'Nd':5, 'Pm':6, 'Sm':7, 'Eu':8, 'Gd':9, 'Tb':6, 'Dy':5, 'Ho':4, 'Er':3, 'Tm':2, 'Yb':1, 'Lu':2, 'Hf':3, 'Ta':4, 'W':5, 'Re':6, 'Os':5, 'Ir':4, 'Pt':3, 'Au':2, 'Hg':1, 'Tl':2, 'Pb':3, 'Bi':4, 'Po':3, 'At':2, 'Rn':1, 'Fr':2, 'Ra':1, 'Ac':2, 'Th':3, 'Pa':4, 'U':5, 'Np':6, 'Pu':7, 'Am':8, 'Cm':9, 'Bk':6, 'Cf':5, 'Es':5, 'Fm':3, 'Md':2, 'No':1, 'Lr':2, 'Rf':3, 'Db':4, 'Sg':5, 'Bh':6, 'Hs':5, 'Mt':4, 'Ds':3, 'Rg':2, 'Cn':1, 'Nh':2, 'Fl':3, 'Mc':4, 'Lv':3, 'Ts':2, 'Og':1 }
    #Rfree fit parameters. Jorgensen 2016 J. Chem. Theory Comput. 2016, 12, 2312−2323. H,C,N,O,F,S,Cl
    rfreedict = {'H':1.64, 'C':2.08, 'N':1.72, 'O':1.6, 'F':1.58, 'S':2.0, 'Cl':1.88}
    #C6 dictionary H-Kr. See MEDFF-horton-parcreate-for-chemshell.py for full periodic table.
    C6dictionary = {'H':6.5, 'He': 1.42, 'Li':1392, 'Be':227, 'B':99.5, 'C':46.6, 'N':24.2, 'O':15.6, 'F':9.52, 'Ne':6.20, 'Na':1518, 'Mg':626, 'Al':528, 'Si':305, 'P':185, 'S':134, 'Cl':94.6, 'Ar':64.2, 'K':3923, 'Ca':2163, 'Sc':1383, 'Ti':1044, 'V':832, 'Cr':602, 'Mn':552, 'Fe':482, 'Co':408, 'Ni':373, 'Cu':253, 'Zn':284, 'Ga':498, 'Ge':354, 'As':246, 'Se':210, 'Br':162, 'Kr':130}


    #Dictionary to keep track of radial volumes
    voldict = {}

    uniqelems=set(fragment.elems)
    numatoms=fragment.numatoms

    print("Lennard-Jones parameter creation from ORCA densities")
    print("")
    print("First calculating densities of free atoms")
    print("Will skip calculation if wfx file already exists")
    print("")

    # Calculate elements
    print("------------------------------------------------------------------------")
    for el in uniqelems:
        print("Doing element:", el)

        #Skipping analysis if wfx file exists
        if os.path.isfile(el+'.molden.wfx'):
            print(el+'.molden.wfx', "exists already. Skipping calculation.")
            continue

        scfextrasettingsstring="""%scf
Maxiter 500
DIISMaxIt 0
ShiftErr 0.0000
DampFac 0.8500
DampMax 0.9800
DampErr 0.0300
cnvsoscf false
cnvkdiis false
end"""

        #Creating ORCA object for  element
        ORCASPcalculation = ash.ORCATheory(orcadir=theory.orcadir, charge=0,
                                           mult=spindictionary[el], orcasimpleinput=theory.orcasimpleinput,
                                           orcablocks=theory.orcablocks, extraline=scfextrasettingsstring)

        #Element coordinates
        Elfrag = ash.Fragment(elems=[el], coords=[[0.0,0.0,0.0]])
        print("Elfrag dict ", Elfrag.__dict__)
        ash.Singlepoint(theory=ORCASPcalculation,fragment=Elfrag)
        #Preserve outputfile and GBW file for each element
        shutil.copyfile('orca-input.out', './' + str(el) + '.out')
        shutil.copyfile('orca-input.gbw', './' + str(el) + '.gbw')

        #Create molden file from el.gbw
        sp.call([theory.orcadir+'/orca_2mkl', el, '-molden'])


        #Write configuration file for molden2aim
        with open("m2a.ini", 'w') as m2afile:
            string = """########################################################################
        #  In the following 8 parameters,
        #     >0:  always performs the operation without asking the user
        #     =0:  asks the user whether to perform the operation
        #     <0:  always neglect the operation without asking the user
        molden= 1           ! Generating a standard Molden file in Cart. function
        wfn= -1              ! Generating a WFN file
        wfncheck= -1         ! Checking normalization for WFN
        wfx= 1              ! Generating a WFX file (not implemented)
        wfxcheck= 1         ! Checking normalization for WFX (not implemented)
        nbo= -1              ! Generating a NBO .47 file
        nbocheck= -1         ! Checking normalization for NBO's .47
        wbo= -1              ! GWBO after the .47 file being generated

        ########################################################################
        #  Which quantum chemistry program is used to generate the MOLDEN file?
        #  1: ORCA, 2: CFOUR, 3: TURBOMOLE, 4: JAGUAR (not supported),
        #  5: ACES2, 6: MOLCAS, 7: PSI4, 8: MRCC, 9: NBO 6 (> ver. 2014),
        #  0: other programs, or read [Program] xxx from MOLDEN.
        #
        #  If non-zero value is given, [Program] xxx in MOLDEN will be ignored.
        #
        program=1

        ########################################################################
        #  For ECP: read core information from Molden file
        #<=0: if the total_occupation_number is smaller than the total_Za, ask
        #     the user whether to read core information
        # >0: always search and read core information
        rdcore=0

        ########################################################################
        #  Which orbirals will be printed in the WFN/WFX file?
        # =0: print only the orbitals with occ. number > 5.0d-8
        # <0: print only the orbitals with occ. number > 0.1 (debug only)
        # >0: print all the orbitals
        iallmo=0

        ########################################################################
        #  Used for WFX only
        # =0: print "UNKNOWN" for Energy and Virial Ratio
        # .ne. 0: print 0.0 for Energy and 2.0 for Virial Ratio
        unknown=1

        ########################################################################
        #  Print supporting information or not
        # =0: print; .ne. 0: do not print
        nosupp=0

        ########################################################################
        #  The following parameters are used only for debugging.
        clear=1            ! delete temporary files (1) or not (0)

        ########################################################################
        """
            m2afile.write(string)

        #Write settings file
        mol2aiminput=[' ',  el+'.molden.input', 'Y', 'Y', 'N', 'N', ' ', ' ']
        m2aimfile = open("mol2aim.inp", "w")
        for mline in mol2aiminput:
            m2aimfile.write(mline+'\n')
        m2aimfile.close()

        #Run molden2aim
        m2aimfile = open('mol2aim.inp')
        p = sp.Popen(molden2aim, stdin=m2aimfile, stderr=sp.STDOUT)
        p.wait()

        #Write job control file for Chargemol
        wfxfile=el+'.molden.wfx'
        jobcontfilewrite=[
        '<atomic densities directory complete path>',
        chargemoldir+'/atomic_densities/',
        '</atomic densities directory complete path>',
        '<input filename>',
        wfxfile,
        '<charge type>',
        DDECmodel,
        '</charge type>',
        '<compute BOs>',
        '.true.',
        '</compute BOs>',
        ]
        jobfile = open("job_control.txt", "w")
        for jline in jobcontfilewrite:
            jobfile.write(jline+'\n')

        jobfile.close()
        sp.call(chargemol)
        print("------------------------------------------------------------------------")


    #DONE WITH ELEMENT CALCS
    
    print("")
    print("=============================")
    #Getting volumes from output
    for el in uniqelems:
        with open(el+'.molden.output') as momfile:
            for line in momfile:
                if ' The computed Rcubed moments of the atoms' in line:
                    elmom=next(momfile).split()[0]
                    print("2 elmom is", elmom)
                    voldict[el] = float(elmom)

        print("Element", el, "is done.")

    print("")
    print("Calculated radial volumes of free atoms (Bohrs^3):", voldict)
    print("")

    #Now doing main molecule. Skipping ORCA calculation if Molden file exists
    if os.path.isfile(molecule+'.molden.input') == False:
        print("Doing molecule:", molecule)
        inputfile = open(molecule+".inp", "w"); inputname=molecule+".inp";
        inputfile.write(orcasimpleinputline);inputfile.write("\n")
        inputfile.write("%pal nprocs "+str(numcores)+" end"); inputfile.write("\n")
        for blockline in orcablocks:
            inputfile.write(blockline)

        inputfile.write("\n")
        xyzline="*xyz "+str(charge)+' '+str(spinmult); inputfile.write(xyzline);inputfile.write("\n")
        for c in coords:
            inputfile.write(c)
        inputfile.write("*")

        inputfile.close()
        outputfile = open(molecule+".out", "w")
        #orcaerrors = open("orcaerrors.out", "w")
        sp.call([orcapath, inputname], stdout=outputfile)
        outputfile.close()
        #orcaerrors.close()
        #Creat molden file
        sp.call(['orca_2mkl', molecule, '-molden'])

    #Write input for molden2aim
    mol2aiminput=[' ',  molecule+'.molden.input', 'Y', 'Y', 'N', 'N', ' ', ' ']
    m2aimfile = open("mol2aim.inp", "w")
    for mline in mol2aiminput:
        m2aimfile.write(mline+'\n')
    m2aimfile.close()

    #Run molden2aim
    m2aimfile = open('mol2aim.inp')
    p = Popen(molden2aim, stdin=m2aimfile, stderr=sp.STDOUT)
    p.wait()
    #Write job control file for Chargemol
    wfxfile=molecule+'.molden.wfx'
    jobcontfilewrite=[
    '<atomic densities directory complete path>',
    '/users/work/ragnarbj/chargemol_09_26_2017/atomic_densities/',
    '</atomic densities directory complete path>',
    '<input filename>',
    wfxfile,
    '<charge type>',
    'DDEC3',
    '</charge type>',
    '<compute BOs>',
    '.true.',
    '</compute BOs>',
    ]
    jobfile = open("job_control.txt", "w")
    for jline in jobcontfilewrite:
        jobfile.write(jline+'\n')

    jobfile.close()
    if os.path.isfile(molecule+'.molden.output') == False:
        sp.call(chargemol)
    else:
        print("Skipping Chargemol step. Output file exists")


    #Grabbing radial moments from output
    molmoms=[]
    grabmoms=False
    with open(molecule+'.molden.output') as momfile:
        for line in momfile:
            if ' The computed Rfourth moments of the atoms' in line:
                grabmoms=False
                continue
            if grabmoms==True:
                temp=line.split()
                [molmoms.append(float(i)) for i in temp]
            if ' The computed Rcubed moments of the atoms' in line:
                grabmoms=True

    #Grabbing DDEC charges from output
    if DDECmodel == 'DDEC3':
        chargefile='DDEC3_net_atomic_charges.xyz'
    elif DDECmodel == 'DDEC6':
        chargefile='DDEC3_net_atomic_charges.xyz'

    grabcharge=False
    ddeccharges=[]
    print("numatoms is", numatoms)
    with open(chargefile) as chfile:
        for line in chfile:
            if grabcharge==True:
                ddeccharges.append(float(line.split()[5]))
                if int(line.split()[0]) == numatoms:
                    grabcharge=False
            if "atom number, atomic symbol, x, y, z, net_charge," in line:
                grabcharge=True

    print("")
    print("elems is", elems)
    print("molmoms is", molmoms)
    print("voldict is", voldict)
    hartokcal=627.5096080305927
    bohrang=0.529177249


    #Calculating A_i, B_i, epsilon, sigma, r0 parameters
    Blist=[]
    Alist=[]
    sigmalist=[]
    epsilonlist=[]
    r0list=[]
    for count,el in enumerate(elems):
        volratio=molmoms[count]/voldict[el]
        C6inkcal=hartokcal*(C6dictionary[el]**(1/6)*bohrang)**6
        B_i=C6inkcal*(volratio**2)
        Raim_i=volratio**(1/3)*rfreedict[el]
        A_i=0.5*B_i*(2*Raim_i)**6
        sigma=(A_i/B_i)**(1/6)
        r0=sigma*(2**(1/6))
        epsilon=(A_i/(4*sigma**12))
        sigmalist.append(sigma)
        Blist.append(B_i)
        Alist.append(A_i)
        epsilonlist.append(epsilon)
        r0list.append(r0)

    print("Before corrections:")
    print("Alist is", Alist)
    print("Blist is", Blist)

    # Corrections to B according to paper

    #Manual phenol modifications
    indextofix=11
    hindex=12
    nH=1
    Blist[indextofix]=( (Blist[indextofix])**(1/2) + nH*(Blist[hindex])**(1/2) )**2

    # Set LJ parameter for polar H to zero
    Blist[hindex]=0.0
    Alist[hindex]=0.0
    print("")
    print("After corrections:")

    print("")
    print("Single atom parameters:")
    print("Alist is", Alist)
    print("Blist is", Blist)
    #print("sigmalist is", sigmalist)
    #print("epsilonlist is", epsilonlist)
    #print("r0list is", r0list)
    print("")
    print("DDEC charge output:", ddeccharges)
    print("DDEC model is", DDECmodel)
    print("")
    print("Pair parameters:")

    #Water atom parameters
    if H2Omodel=='TIP3P':
        water_eps=0.15207217973231357
        water_sigma=3.15066
        water_A=582000.0
        water_B=595.0
    elif H2Omodel=='Chargemol':
        #Chargemol for H2o
        water_eps=0.10009723889719248
        water_sigma=3.069109977984782
        water_A=310617.9900759511
        #water_B=352.6584929270707
        #B has here been corrected according to eq. 10 in paper
        water_B=562.5399865110148
    elif H2Omodel=='Manual':
        #Manual mod
        #water_eps=0.75
        print("add stuff here")
        exit()


    water_r0=r0=water_sigma*(2**(1/6))
    print("Using Water parameters:")
    print("Model is:", H2Omodel)
    print("")
    print("Ai = ", water_A)
    print("Bi = ", water_B)
    print("epsilon=", water_eps, "kcal/mol")
    print("sigma=", water_sigma, "Angstrom")
    print("water_r0=", water_r0, "Angstrom")


    #Creating A and B pairlist

    Apairlist=[]
    Bpairlist=[]
    for A,B,elm in zip(Alist,Blist,elems):
        Apair=(A*water_A)**(1/2)
        Bpair=(B*water_B)**(1/2)
        Apairlist.append(Apair)
        Bpairlist.append(Bpair)



    r0pairlist=[]
    epspairlist=[]
    for r,eps,elm in zip(r0list,epsilonlist,elems):
        r0pair=(r*water_r0)**(1/2)
        epspair=(eps*water_eps)**(1/2)
        r0pairlist.append(r0pair)
        epspairlist.append(epspair)


    #print("r0pairlist :", r0pairlist)
    #print("epspairlist :", epspairlist)
    print("")
    ##Final m_n list
    #Create numbered element list
    templist=[]
    elemnumlist=[]
    for i in elems:
        elemnumlist.append(i+str(templist.count(i)+1))
        templist.append(i)

    print("New soltypes list is: ")
    [print(i, end=' ') for i in elemnumlist]
    print("")
    print("")


    #for r0,e,elm in zip(r0pairlist,epspairlist,elemnumlist):
    #    print("m_n_vdw", elm, "OT", 12, 6, r0, e)

    print("")
    for b,a,elm in zip(Bpairlist,Apairlist,elemnumlist):
        print("vdw", elm, "OT", b, a)