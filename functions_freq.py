from functions_general import listdiff, clean_number,blankline
import numpy as np
import constants

#HESSIAN-related functions below


#Get partial matrix by deleting rows not present in list of indices.
#Deletes numpy rows
def get_partial_matrix(allatoms,hessatoms,matrix):
    nonhessatoms=listdiff(allatoms,hessatoms)
    nonhessatoms.reverse()
    for at in nonhessatoms:
        matrix=np.delete(matrix, at, 0)
    return matrix


# Open-source project in Fortran:
#https://github.com/zorkzou/UniMoVib
#Calculates Hessian etc. Has IR and Raman intensity
#Todo: Add IR/Raman intensity support


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



#list of frequencies and fragment object
#TODO: Make sure distinction between initial coords and optimized coords?
def thermochemcalc(vfreq,hessatoms,fragment, multiplicity, temp=298.18,pressure=1):
    blankline()
    print("Printing thermochemistry")
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
    #print(vfreq)
    for mode in range(0, 3 * len(hessatoms)):
        #print(mode)
        if mode < TRmodenum:
            continue
            #print("skipping TR mode with freq:", clean_number(vfreq[mode]) )
        else:
            vib = clean_number(vfreq[mode])
            if np.iscomplex(vib):
                print("Mode {} with frequency {} is imaginary. Skipping in thermochemistry".format(mode,vib))
            else:
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

#From Hess-tool.py: Copied 13 May 2020
#Print dummy ORCA outputfile using coordinates and normal modes. Used for visualization of modes in Chemcraft
def printdummyORCAfile(elems,coords,vfreq,evectors,nmodes,hessfile):
    orca_header = """                                 *****************
                                 * O   R   C   A *
                                 *****************

           --- An Ab Initio, DFT and Semiempirical electronic structure package ---

                  #######################################################
                  #                        -***-                        #
                  #  Department of molecular theory and spectroscopy    #
                  #              Directorship: Frank Neese              #
                  # Max Planck Institute for Chemical Energy Conversion #
                  #                  D-45470 Muelheim/Ruhr              #
                  #                       Germany                       #
                  #                                                     #
                  #                  All rights reserved                #
                  #                        -***-                        #
                  #######################################################


                         Program Version 3.0.3 - RELEASE   -




                       *****************************
                       * Geometry Optimization Run *
                       *****************************

         *************************************************************
         *                GEOMETRY OPTIMIZATION CYCLE   1            *
         *************************************************************
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------"""
    #Very simple check for 2-atom linear molecule
    #Todo: Need to add support for n-atom linear molecule (HCN e.g.)
    if len(elems) == 2:
        TRmodenum=5
    else:
        TRmodenum=6

    outfile = open(hessfile+'_dummy.out', 'w')
    outfile.write(orca_header+'\n')
    for el,coord in zip(elems,coords):
        x=coord[0];y=coord[1];z=coord[2]
        line = "  {0:2s} {1:11.6f} {2:12.6f} {3:13.6f}".format(el, x, y, z)
        #print(line)
        #print('  S     51.226907   65.512868  106.021030')
        #exit()
        outfile.write(line+'\n')
    outfile.write('\n')
    outfile.write('-----------------------\n')
    outfile.write('VIBRATIONAL FREQUENCIES\n')
    outfile.write('-----------------------\n')
    outfile.write('\n')
    outfile.write('Scaling factor for frequencies =  1.000000000 (Found in file - NOT applied to frequencies read from HESS file)\n')
    outfile.write('\n')
    numatoms=(len(elems))
    for mode in range(3*numatoms):
        smode = str(mode) + ':'
        if mode < TRmodenum:
            freq=0.00
        else:
            freq=clean_number(vfreq[mode])
            print("freq:", freq)
            print("type freq:", type(freq))
            print("np.real freq", np.real(freq))
        line= "  {0:>3s}{1:13.2f} cm**-1".format(smode, freq)
        outfile.write(line+'\n')
    normalmodeheader="""------------
NORMAL MODES
------------

These modes are the cartesian displacements weighted by the diagonal matrix
M(i,i)=1/sqrt(m[i]) where m[i] is the mass of the displaced atom
Thus, these vectors are normalized but *not* orthogonal"""

#TODO: Finish write normal mode output in ORCA format from nmodes so that Chemcraft can read it

    outfile.write('\n')
    outfile.write('\n')
    outfile.write(normalmodeheader)
    outfile.write('\n')
    outfile.write('\n')

    orcahesscoldim = 6
    hessdim=3*numatoms
    hessrow = []
    index = 0
    line = ""
    chunkheader = ""

    chunks = hessdim // orcahesscoldim
    left = hessdim % orcahesscoldim
    #print("Chunks:", chunks)
    #print("left:", left)
    if left > 0:
        chunks = chunks + 1
    #print("Chunks:", chunks)

    #print("evectors", evectors)
    #print("")
    #print("nmodes", nmodes)
    #print("len(nmodes)", len(nmodes))
    #TODO: Should we be using the eigenvectors instead, i.e. the mass-weighted normalmodes.
    #Seems to be what ORCA is using?

    #Transpose of nmodes for convenience
    #nmodes_tp=np.transpose(nmodes)
    #print(nmodes_tp)
    #print("")

    #print("Beginning for loop")
    for chunk in range(chunks):
        #print("chunk is", chunk)
        if chunk == chunks - 1:
            #print("a last chunk is", chunk)
            # If last chunk and cleft is exactly 0 then all 5 columns should be done
            if left == 0:
                left = 6
            # print("index is", index)
            # print("left is", left)
            for temp in range(index, index + left):
                chunkheader = chunkheader + "          " + str(temp)
            #print(chunkheader)
        else:
            for temp in range(index, index + orcahesscoldim):
                chunkheader = chunkheader + "          " + str(temp)
            #print(chunkheader)
        outfile.write("        "+str(chunkheader) + "    \n")
        for i in range(0, hessdim):
            firstcolumnindex=6*chunk
            j=firstcolumnindex
            #print("firstcolumnindex j is:", firstcolumnindex)
            #print("i is", i)
            #print("nmodes[i]", nmodes[i])
            #print("hessdim:", hessdim)
            #If chunk = 0 then we are dealing with TR modes in first 6 columns
            if chunk == 0:
                val1 = 0.0; val2 = 0.0;val3 = 0.0; val4 = 0.0; val5 = 0.0;val6 = 0.0
            else :
                #TODO: Here defning values to print based on values in nmodes matrix. TO be confiremd that this is correct. TODO.
                if hessdim - j == 1:
                    val1 = nmodes[j][i]
                elif hessdim - j == 2:
                    val1 = nmodes[j][j]; val2 = nmodes[j+1][i]
                elif hessdim - j == 3:
                    val1 = nmodes[j][i]; val2 = nmodes[j+1][i];val3 = nmodes[j+2][i]
                elif hessdim - j == 4:
                    val1 = nmodes[j][i]; val2 = nmodes[j+1][i];val3 = nmodes[j+2][i];val4 = nmodes[j+3][i]
                elif hessdim - j == 5:
                    val1 = nmodes[j][i]; val2 = nmodes[j+1][i];val3 = nmodes[j+2][i];val4 = nmodes[j+3][i];val5 = nmodes[j+4][i]
                elif hessdim - j >= 6:
                    val1 = nmodes[j][i]; val2 = nmodes[j+1][i];val3 = nmodes[j+2][i];val4 = nmodes[j+3][i];val5 = nmodes[j+4][i];val6 = nmodes[j+5][i]
                else:
                    print("problem")
                    print("hessdim - j : ", hessdim - j)
                    exit()

            if chunk == chunks - 1:
                #print("b last chunk is", chunk)
                # print("index is", index)
                # print("index+orcahesscoldim-left is", index+orcahesscoldim-left)
                for k in range(index, index + left):
                    #print("i is", i, "and k is", k)
                    #print("left:", left)
                    if left == 6:
                        line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3, val4, val5, val6)
                    elif left == 5:
                        line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3, val4, val5)
                    elif left == 5:
                        line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3, val4)
                    elif left == 3:
                        line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3)
                    elif left ==2:
                        line = "{:>6d}} {:>14.6f} {:>10.6f}".format(i, val1, val2)
                    elif left == 1:
                        line = "{:>6d} {:>14.6f}".format(i, val1)
            else:
                for k in range(index, index + orcahesscoldim):
                    #print("i is", i, "and k is", k)
                    line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3, val4, val5, val6)
            #print(line)
            #outfile.write("    " + str(i) + "   " + str(line) + "\n")
            outfile.write(" " + str(line) + "\n")
            line = "";
            chunkheader = ""
        index += 6

    endstring="""

-----------
IR SPECTRUM
-----------

 Mode    freq (cm**-1)   T**2         TX         TY         TZ"""
    outfile.write(endstring)



    outfile.close()
    print("Created dummy ORCA outputfile: ", hessfile+'_dummy.out')