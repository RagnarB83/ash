import numpy as np
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
        dist_row=[distance(i,j) for j in coords]
        distmatrix.append(dist_row)
    return distmatrix
            


def calc_cm5(atomicNumbers, coords, hirschfeldcharges):
    #all matrices have the naming scheme matrix[k,k'] according to the paper
    #distances = atoms.get_all_distances(mic=True)
    distances = distance_matrix_from_coords(coords)
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
    return np.array(hirschfeldcharges) + result




#Read cubefile. Grabs coords. Calculates density if MO
def read_cube (cubefile):
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