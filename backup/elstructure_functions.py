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