import numpy as np
import math
import shutil
import os
import glob
import subprocess as sp

#import ash
import ash.constants
import ash.modules.module_coords
import ash.dictionaries_lists
from ash.functions.functions_general import ashexit, isodd
import ash.interfaces.interface_ORCA
from ash.modules.module_coords import nucchargelist

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
        dist_row=[ash.modules.module_coords.distance(i,j) for j in coords]
        distmatrix.append(dist_row)
    return distmatrix
            


def calc_cm5(atomicNumbers, coords, hirschfeldcharges):
    coords=np.array(coords)
    atomicNumbers=np.array(atomicNumbers)
    #all matrices have the naming scheme matrix[k,k'] according to the paper
    #distances = atoms.get_all_distances(mic=True)
    distances = np.array(distance_matrix_from_coords(coords))
    #print("distances:", distances)
    #atomicNumbers = np.array(atoms.numbers)
    #print("atomicNumbers", atomicNumbers)
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
    #print("hirschfeldcharges:", hirschfeldcharges)
    #print("result:", result)
    #print(type(result))
    return np.array(hirschfeldcharges) + result


#Read cubefile.
#TODO: Clean up!
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

            if len(c) >0:
                vals.append(c)
        # when to begin reading gridpoints
        if (count >= 6+numatoms and X==False):
            X = True
    if LargePrint==True:
        print("Number of orb/density points:", len(vals))
    finaldict={'rlowx':rlowx,'dx':dx,'nx':nx,'orgx':orgx,'rlowy':rlowy,'dy':dy,'ny':ny,'orgy':orgy,'rlowz':rlowz,'dz':dz,'nz':nz,'orgz':orgz,'elems':elems,
        'molcoords':molcoords,'molcoords_ang':molcoords_ang,'numatoms':numatoms,'filebase':filebase,'vals':vals}
    return  finaldict

#Subtract one Cube-file from another
def write_cube_diff(cubedict1,cubedict2, name="Default"):

    #TODO: Check for consistency of cubefile with respect to grid points, coordinates etc

    numatoms=cubedict1['numatoms']
    orgx=cubedict1['orgx']
    orgy=cubedict1['orgy']
    orgz=cubedict1['orgz']
    nx=cubedict1['nx']
    dx=cubedict1['dx']
    ny=cubedict1['ny']
    dy=cubedict1['dy']
    nz=cubedict1['nz']
    dz=cubedict1['dz']
    elems=cubedict1['elems']
    molcoords=cubedict1['molcoords']
    val1=cubedict1['vals']
    val2=cubedict2['vals']
    #name=cubedict['name']

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


def centroid_calc(rlowx,dx,nx,orgx,rlowy,dy,ny,orgy,rlowz,dz,nz,orgz,alldensvalues ):
    #########################################################
    # Calculate centroid.
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
            ashexit()
            continue
        for j in range(1,ny+1):
            if (orgy+(j-1)*dy)<rlowy or (orgy+(j-1)*dy)>rhighy:
                print("If statement. Look into. y")
                ashexit()
                continue
            for k in range(1,nz+1):
                if (orgz+(k-1)*dz)<rlowz or (orgz+(k-1)*dz)>rhighz:
                    print("If statement. Look into. z")
                    ashexit()
                    continue
                #print("i,j,k is", i,j,k)
                valtmp=alldensvalues[vcount]
                if valtmp<rlowv or valtmp>rhighv:
                    print("If statement. Look into. v")
                    ashexit()
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
def DDEC_calc(elems=None, theory=None, gbwfile=None, numcores=1, DDECmodel='DDEC3', calcdir='DDEC', molecule_charge=None, 
              molecule_spinmult=None, chargemolbinarydir=None):
    #Creating calcdir. Should not exist previously
    try:
        shutil.rmtree(calcdir)
    except:
        pass
    os.mkdir(calcdir)
    os.chdir(calcdir)
    #Copying GBW file to current dir and label as molecule.gbw
    shutil.copyfile('../'+gbwfile, './' + 'molecule.gbw')

    #Finding molden2aim in PATH. Present in ASH (May require compilation)
    ashpath=os.path.dirname(ash.__file__)
    molden2aim=ashpath+"/external/Molden2AIM/src/"+"molden2aim.exe"
    if os.path.isfile(molden2aim) is False:
        print("Did not find {}. Did you compile it ? ".format(molden2aim))
        print("Go into dir:", ashpath+"/external/Molden2AIM/src")
        print("Compile using gfortran or ifort:")
        print("gfortran -O3 edflib.f90 edflib-pbe0.f90 molden2aim.f90 -o molden2aim.exe")
        print("ifort -O3 edflib.f90 edflib-pbe0.f90 molden2aim.f90 -o molden2aim.exe")
        ashexit()
    else:
        print("Found molden2aim.exe: ", molden2aim)
        
    print("Warning: DDEC_calc requires chargemol-binary dir to be present in environment PATH variable.")

    #Finding chargemoldir from PATH in os.path
    PATH=os.environ.get('PATH').split(':')
    print("PATH: ", PATH)
    print("Searching for molden2aim and chargemol in PATH")
    for p in PATH:
        if 'chargemol' in p:
            print("Found chargemol in path line (this dir should contain the executables):", p)
            chargemolbinarydir=p
    
    #Checking if we can proceed
    if chargemolbinarydir is None:
        print("chargemolbinarydir is not defined.")
        print("Please provide path as argument to DDEC_calc or put the location inside the $PATH variable on your Unix/Linux OS.")
        ashexit()

    #Defining Chargemoldir (main dir) as 3-up from binary dir
    var=os.path.split(chargemolbinarydir)[0]
    var=os.path.split(var)[0]
    chargemoldir=os.path.split(var)[0]
    print("Chargemoldir (base directory): ", chargemoldir)
    print("Chargemol binary dir:", chargemolbinarydir)

    if theory is None :
        print("DDEC_calc requires theory, keyword argument")
        ashexit()
    if theory.__class__.__name__ != "ORCATheory":
        print("Only ORCA is supported as theory in DDEC_calc currently")
        ashexit()

    # What DDEC charge model to use. Jorgensen paper uses DDEC3. DDEC6 is the newer recommended chargemodel
    print("DDEC model:", DDECmodel)

    # Serial or parallel version
    if numcores == 1:
        print("Using serial version of Chargemol")
        chargemol=glob.glob(chargemolbinarydir+'/*serial*')[0]
        #chargemol=chargemolbinarydir+glob.glob('*serial*')[0]
    else:
        print("Using parallel version of Chargemol using {} cores".format(numcores))
        #chargemol=chargemolbinarydir+glob.glob('*parallel*')[0]
        chargemol=glob.glob(chargemolbinarydir+'/*parallel*')[0]
        # Parallelization of Chargemol code. 8 should be good.
        os.environ['OMP_NUM_THREADS'] = str(numcores)
    print("Using Chargemoldir executable: ", chargemol)

    #Dictionary for spin multiplicities of atoms
    spindictionary = {'H':2, 'He': 1, 'Li':2, 'Be':1, 'B':2, 'C':3, 'N':4, 'O':3, 'F':2, 'Ne':1, 'Na':2, 'Mg':1, 'Al':2, 'Si':3, 'P':4, 'S':3, 'Cl':2, 'Ar':1, 'K':2, 'Ca':1, 'Sc':2, 'Ti':3, 'V':4, 'Cr':7, 'Mn':6, 'Fe':5, 'Co':4, 'Ni':3, 'Cu':2, 'Zn':1, 'Ga':2, 'Ge':3, 'As':4, 'Se':3, 'Br':2, 'Kr':1, 'Rb':2, 'Sr':1, 'Y':2, 'Zr':3, 'Nb':6, 'Mo':7, 'Tc':6, 'Ru':5, 'Rh':4, 'Pd':1, 'Ag':2, 'Cd':1, 'In':2, 'Sn':3, 'Sb':4, 'Te':3, 'I':2, 'Xe':1, 'Cs':2, 'Ba':1, 'La':2, 'Ce':1, 'Pr':4, 'Nd':5, 'Pm':6, 'Sm':7, 'Eu':8, 'Gd':9, 'Tb':6, 'Dy':5, 'Ho':4, 'Er':3, 'Tm':2, 'Yb':1, 'Lu':2, 'Hf':3, 'Ta':4, 'W':5, 'Re':6, 'Os':5, 'Ir':4, 'Pt':3, 'Au':2, 'Hg':1, 'Tl':2, 'Pb':3, 'Bi':4, 'Po':3, 'At':2, 'Rn':1, 'Fr':2, 'Ra':1, 'Ac':2, 'Th':3, 'Pa':4, 'U':5, 'Np':6, 'Pu':7, 'Am':8, 'Cm':9, 'Bk':6, 'Cf':5, 'Es':5, 'Fm':3, 'Md':2, 'No':1, 'Lr':2, 'Rf':3, 'Db':4, 'Sg':5, 'Bh':6, 'Hs':5, 'Mt':4, 'Ds':3, 'Rg':2, 'Cn':1, 'Nh':2, 'Fl':3, 'Mc':4, 'Lv':3, 'Ts':2, 'Og':1 }

    #Dictionary to keep track of radial volumes
    voldict = {}

    uniqelems=set(elems)
    numatoms=len(elems)

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
        #TODO: Revisit with ORCA5 and TRAH?
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
        ORCASPcalculation = ash.interfaces.interface_ORCA.ORCATheory(orcadir=theory.orcadir, orcasimpleinput=theory.orcasimpleinput,
                                           orcablocks=theory.orcablocks, extraline=scfextrasettingsstring)

        #Element coordinates
        Elfrag = ash.Fragment(elems=[el], coords=[[0.0,0.0,0.0]])
        print("Elfrag dict ", Elfrag.__dict__)
        ash.Singlepoint(theory=ORCASPcalculation,fragment=Elfrag, charge=0, mult=spindictionary[el])
        #Preserve outputfile and GBW file for each element
        shutil.copyfile(ORCASPcalculation.filename+'.out', './' + str(el) + '.out')
        shutil.copyfile(ORCASPcalculation.filename+'.gbw', './' + str(el) + '.gbw')

        #Create molden file from el.gbw
        sp.call([theory.orcadir+'/orca_2mkl', el, '-molden'])


        #Cleanup ORCA calc for each element
        ORCASPcalculation.cleanup()

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
        #CALLING chargemol
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
                    voldict[el] = float(elmom)

        print("Element", el, "is done.")

    print("")
    print("Calculated radial volumes of free atoms (Bohrs^3):", voldict)
    print("")

    #Now doing main molecule. Skipping ORCA calculation since we have copied over GBW file
    # Create molden file
    sp.call(['orca_2mkl', "molecule", '-molden'])

    #Write input for molden2aim
    
    if molecule_charge==0:
        mol2aiminput=[' ',  "molecule"+'.molden.input', str(molecule_spinmult), ' ', ' ', ' ']
    else:
        #Charged system, will ask for charge
        #str(molecule_charge)
        mol2aiminput=[' ',  "molecule"+'.molden.input', 'N', '2', ' ', str(molecule_spinmult), ' ', ' ', ' ']        
        
    m2aimfile = open("mol2aim.inp", "w")
    for mline in mol2aiminput:
        m2aimfile.write(mline+'\n')
    m2aimfile.close()

    #Run molden2aim
    print("Running Molden2Aim for molecule")
    m2aimfile = open('mol2aim.inp')
    p = sp.Popen(molden2aim, stdin=m2aimfile, stderr=sp.STDOUT)
    p.wait()

    # Write job control file for Chargemol
    wfxfile = "molecule" + '.molden.wfx'
    jobcontfilewrite = [
        '<net charge>',
        '{}'.format(str(float(molecule_charge))),
        '</net charge>',
        '<atomic densities directory complete path>',
        chargemoldir + '/atomic_densities/',
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
    if os.path.isfile("molecule"+'.molden.output') == False:
        sp.call(chargemol)
    else:
        print("Skipping Chargemol step. Output file exists")


    #Grabbing radial moments from output
    molmoms=[]
    grabmoms=False
    with open("molecule"+'.molden.output') as momfile:
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
        chargefile='DDEC6_even_tempered_net_atomic_charges.xyz'

    grabcharge=False
    ddeccharges=[]
    with open(chargefile) as chfile:
        for line in chfile:
            if grabcharge==True:
                ddeccharges.append(float(line.split()[5]))
                if int(line.split()[0]) == numatoms:
                    grabcharge=False
            if "atom number, atomic symbol, x, y, z, net_charge," in line:
                grabcharge=True

    print("")
    print("molmoms is", molmoms)
    print("voldict is", voldict)
    print("ddeccharges: ", ddeccharges)
    print("elems: ", elems)
    os.chdir('..')
    return ddeccharges, molmoms, voldict




#Tkatchenko
#alpha_q_m = Phi*Rvdw^7
#https://arxiv.org/pdf/2007.02992.pdf
def Rvdwfree(polz):
    #Fine-structure constant (2018 CODATA recommended value)
    FSC=0.0072973525693
    Phi=FSC**(4/3)
    RvdW=(polz/Phi)**(1/7)
    return RvdW
    

def DDEC_to_LJparameters(elems, molmoms, voldict, scale_polarH=False):
    
    #voldict: Vfree. Computed using MP4SDQ/augQZ and chargemol in Jorgensen paper
    # Testing: Use free atom volumes calculated at same level of theory as molecule
    
    #Rfree fit parameters. Jorgensen 2016 J. Chem. Theory Comput. 2016, 12, 2312−2323. H,C,N,O,F,S,Cl
    #Thes are free atomic vdW radii
    # In Jorgensen and Cole papers these are fit parameters : rfreedict = {'H':1.64, 'C':2.08, 'N':1.72, 'O':1.6, 'F':1.58, 'S':2.0, 'Cl':1.88}
    # We are instead using atomic Rvdw derived directly from atomic polarizabilities
    
    print("Elems:", elems)
    print("Molmoms:", molmoms)
    print("voldict:", voldict)

    #Calculating A_i, B_i, epsilon, sigma, r0 parameters
    Blist=[]
    Alist=[]
    sigmalist=[]
    epsilonlist=[]
    r0list=[]
    Radii_vdw_free=[]
    for count,el in enumerate(elems):
        print("el :", el, "count:", count)
        atmnumber=ash.modules.module_coords.elematomnumbers[el.lower()]
        print("atmnumber:", atmnumber)
        Radii_vdw_free.append(ash.dictionaries_lists.elems_C6_polz[atmnumber].Rvdw_ang)
        print("Radii_vdw_free:", Radii_vdw_free)
        volratio=molmoms[count]/voldict[el]
        print("volratio:", volratio)
        C6inkcal=ash.constants.harkcal*(ash.dictionaries_lists.elems_C6_polz[atmnumber].C6**(1/6)* ash.constants.bohr2ang)**6
        print("C6inkcal:", C6inkcal)
        B_i=C6inkcal*(volratio**2)
        print("B_i:", B_i)
        Raim_i=volratio**(1/3)*ash.dictionaries_lists.elems_C6_polz[atmnumber].Rvdw_ang
        print("Raim_i:", Raim_i)
        A_i=0.5*B_i*(2*Raim_i)**6
        print("A_i:", A_i)
        sigma=(A_i/B_i)**(1/6)
        print("sigma :", sigma)
        r0=sigma*(2**(1/6))
        print("r0:", r0)
        epsilon=(A_i/(4*sigma**12))
        print("epsilon:", epsilon)
        
        sigmalist.append(sigma)
        Blist.append(B_i)
        Alist.append(A_i)
        epsilonlist.append(epsilon)
        r0list.append(r0)

    print("Before corrections:")
    print("elems:", elems)
    print("Radii_vdw_free:", Radii_vdw_free)
    print("Alist is", Alist)
    print("Blist is", Blist)
    print("sigmalist is", sigmalist)
    print("epsilonlist is", epsilonlist)
    print("r0list is", r0list)
    
    #Accounting for polar H. This could be set to zero as in Jorgensen paper
    if scale_polarH is True:
        print("Scaling og polar H not implemented yet")
        ashexit()
        for count,el in enumerate(elems):
            if el == 'H':
                bla=""
                #Check if H connected to polar atom (O, N, S ?)
                #if 'H' connected to polar:
                    #1. Set eps,r0/sigma to 0 if so
                    #2. Add to heavy atom if so
                    #nH = 1
                    #indextofix = 11
                    #hindex = 12
                    #Blist[indextofix] = ((Blist[indextofix]) ** (1 / 2) + nH * (Blist[hindex]) ** (1 / 2)) ** 2

    return r0list, epsilonlist


#Get number of core electrons for list of elements
def num_core_electrons(elems):
    sum=0
    #formula_list = ash.modules.module_coords.molformulatolist(fragment.formula)
    for i in elems:
        cels = ash.dictionaries_lists.atom_core_electrons[i]
        sum+=cels
    return sum


#Check if electrons pairs in element list are less than numcores. Reduce numcores if so.
#Using even number of electrons
def check_cores_vs_electrons(elems,numcores,charge):
    print("numcores:", numcores)
    print("charge:", charge)
    numelectrons = int(nucchargelist(elems) - charge)
    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(elems)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of total electrons :", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        if isodd(electronpairs):
            if electronpairs > 1:
                #Changed from subtracting 1 to 3 after DLPNO-CC of NaH calculation failed (MB16-43)
                numcores=electronpairs-3
            else:
                numcores=electronpairs
        else:
            numcores=electronpairs
    if numcores == 0:
        numcores=1
    print("Setting numcores to:", numcores)
    return numcores



#Approximate J-coupling spin projection functions
def Jcoupling_Yamaguchi(HSenergy,BSenergy,HS_S2,BS_S2):
    print("Yamaguchi spin projection")
    J=-1*(HSenergy-BSenergy)/(HS_S2-BS_S2)
    J_kcal=J*ash.constants.harkcal
    J_cm=J*ash.constants.hartocm
    print("J coupling constant: {} Eh".format(J))
    print("J coupling constant: {} kcal/Mol".format(J_kcal))
    print("J coupling constant: {} cm**-1".format(J_cm))            
    return J
#Strong-interaction limit (bond-formation)
def Jcoupling_Bencini(HSenergy,BSenergy,smax):
    print("Bencini spin projection")
    J=-1*(HSenergy-BSenergy)/(smax*(smax+1))
    J_kcal=J*ash.constants.harkcal
    J_cm=J*ash.constants.hartocm
    print("Smax : ", smax)
    print("J coupling constant: {} Eh".format(J))
    print("J coupling constant: {} kcal/Mol".format(J_kcal))
    print("J coupling constant: {} cm**-1".format(J_cm))
    return J
#Weak-interaction limit
def Jcoupling_Noodleman(HSenergy,BSenergy,smax):
    print("Noodleman spin projection")
    J=-1*(HSenergy-BSenergy)/(smax)**2
    J_kcal=J*ash.constants.harkcal
    J_cm=J*ash.constants.hartocm
    print("Smax : ", smax)
    print("J coupling constant: {} Eh".format(J))
    print("J coupling constant: {} kcal/Mol".format(J_kcal))
    print("J coupling constant: {} cm**-1".format(J_cm))
    return J

#Select an active space from list of occupations and thresholds
def select_space_from_occupations(occlist, selection_thresholds=[1.98,0.02]):
    upper_threshold=selection_thresholds[0]
    lower_threshold=selection_thresholds[1]
    welloccorbs=[i for i in occlist if i < upper_threshold and i > lower_threshold]
    numelectrons=round(sum(welloccorbs))
    numorbitals=len(welloccorbs)
    return [numelectrons,numorbitals]

# Interface to XDM postg program
#https://github.com/aoterodelaroza/postg
def xdm_run(wfxfile=None, postgdir=None,a1=None, a2=None,functional=None):

    if postgdir == None:
        # Trying to find postgdir in path
        print("postgdir keyword argument not provided to xdm_run. Trying to find postg in PATH")
        try:
            postgdir = os.path.dirname(shutil.which('postg'))
            print("Found postg in path. Setting postgdir.")
        except:
            print("Found no postg executable in path. Exiting... ")
            ashexit()

    parameterdict= {'pw86pbe' : [0.7564,1.4545], 'b3lyp' : [0.6356, 1.5119],
    'b3pw91' : [0.6002,1.4043], 'b3p86' : [1.0400, 0.3741], 'pbe0' : [0.4186,2.6791],
    'camb3lyp' : [0.3248,2.8607], 'b97-1' : [0.1998,3.5367], 'bhandhlyp' : [0.5610, 1.9894],
    'blyp' : [0.7647,0.8457],'pbe' : [0.4492,2.5517],'lcwpbe' : [1.0149, 0.6755],
    'tpss' : [0.6612, 1.5111], 'b86bpbe' : [0.7443, 1.4072]}

    if a1 == None or a2 == None:
        print("a1/a2 parameters not given. Looking up functional in table")
        print("Parameter table:", parameterdict)
        a1, a2 = parameterdict[functional.lower()]
        print(f"XDM a1: {a1}, a2: {a2}")
    with open('xdm-postg.out', 'w') as ofile:
        process = sp.run([postgdir+'/postg', str(a1), str(a2), str(wfxfile), str(functional) ], check=True, 
            stdout=ofile, stderr=ofile, universal_newlines=True)

    dispgrab=False
    dispgradient=[]
    with open('xdm-postg.out', 'r') as xdmfile:
        for line in xdmfile:
            #TODO: Grab Hirshfeld charges
            #TODO: C6,C8, C10 coefficients, moments and volumes
            if 'dispersion energy' in line:
                dispenergy = float(line.split()[-1])
            if 'dispersion force constant matrix' in line:
                dispgrab=False
            if dispgrab == True:
                if '#' not in line:
                    grad_x=-1*float(line.split()[1])
                    grad_y=-1*float(line.split()[2])
                    grad_z=-1*float(line.split()[3])
                    dispgradient.append([grad_x,grad_y,grad_z])
            if 'dispersion forces' in line:
                dispgrab=True

    dispgradient=np.array(dispgradient)
    print("dispenergy:", dispenergy)
    print("dispgradient:", dispgradient)
    return dispenergy, dispgradient