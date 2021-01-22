from functions_general import listdiff, clean_number,blankline
from functions_coords import elematomnumbers, atommasses
import numpy as np
import math
import constants
import ash

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


# Massweight Hessian
def massweight(matrix,masses,numatoms):
    mass_mat = np.zeros( (3*numatoms,3*numatoms), dtype = float )
    molwt = [ masses[int(i)] for i in range(numatoms) for j in range(3) ]
    for i in range(len(molwt)):
        mass_mat[i,i] = molwt[i] ** -0.5
    mwhessian = np.dot((np.dot(mass_mat,matrix)),mass_mat)
    return mwhessian,mass_mat

# Calculate frequencies from eigenvalus
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



#
def thermochemcalc(vfreq,atoms,fragment, multiplicity, temp=298.15,pressure=1.0):
    """[summary]

    Args:
        vfreq ([list]): list of vibrational frequencies in cm**-1
        atoms ([type]): number of active atoms (contributing to Hessian) 
        fragment ([type]): ASH fragment object
        multiplicity ([type]): spin multiplicity
        temp (float, optional): [description]. Defaults to 298.15.
        pressure (float, optional): [description]. Defaults to 1.0.

    Returns:
        dictionary with thermochemistry properties
    """
    blankline()
    print("Thermochemistry via rigid-rotor harmonic oscillator approximation")
    if len(atoms) == 1:
        print("System is an atom.")
        moltype="atom"
    elif len(atoms) == 2:
        print("System is 2-atomic and thus linear")
        moltype="linear"
        TRmodenum=5
    else:
        #TODO: Need to detect linearity properly
        print("System size N > 2, assumed to be nonlinear")
        moltype="nonlinear"
        TRmodenum=6
    
    coords=fragment.coords
    elems=fragment.elems
    masses=fragment.list_of_masses
    totalmass=sum(masses)
    
    ###################
    #VIBRATIONAL PART
    ###################
    if moltype != "atom":
        freqs=[]
        vibtemps=[]
        for mode in range(0, 3 * len(atoms)):
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
                    freq_Hz=vib*constants.c
                    vibtemp=(constants.h_planck_hartreeseconds * freq_Hz) / constants.R_gasconst
                    vibtemps.append(vibtemp)

        #Zero-point vibrational energy
        zpve=sum([i*constants.halfhcfactor for i in freqs])

        #Thermal vibrational energy
        sumb=0.0
        for v in vibtemps:
            #print(v*(0.5+(1/(np.exp((v/temp) - 1)))))
            sumb=sumb+v*(0.5+(1/(np.exp((v/temp) - 1))))
        E_vib=sumb*constants.R_gasconst
        vibenergycorr=E_vib-zpve

        #Vibrational entropy via RRHO.
        #TODO: Add Grimme QRRHO and otherss
        S_vib=0.0
        for vibtemp in vibtemps:
            S_vib+=constants.R_gasconst*(vibtemp/temp)/(math.exp(vibtemp/temp) - 1) - constants.R_gasconst*math.log(1-math.exp(-1*vibtemp/temp))
        TS_vib=S_vib*temp
        
    else:
        zpve=0.0
        E_vib=0.0
        freqs=[]
        vibenergycorr=0.0
        TS_vib=0.0

    ###################
    #ROTATIONAL PART
    ###################
    if moltype != "atom":
        # Moments of inertia (amu A^2 ), eigenvalues
        center = get_center(elems,coords)
        rinertia = list(inertia(elems,coords,center))
        print("Moments of inertia (amu Å^2):", rinertia)
        #Changing units to m and kg
        I=np.array(rinertia)*constants.amu2kg*constants.ang2m**2

        #Rotational temperatures
        #k_b_JK or R_JK
        rot_temps_x=constants.h_planck**2 / (8*math.pi**2 * constants.k_b_JK * I[0])
        rot_temps_y=constants.h_planck**2 / (8*math.pi**2 * constants.k_b_JK * I[1])
        rot_temps_z=constants.h_planck**2 / (8*math.pi**2 * constants.k_b_JK * I[2])
        print("Rotational temperatures: {}, {}, {} K".format(rot_temps_x,rot_temps_y,rot_temps_z))
        #Rotational constants
        rotconstants = calc_rotational_constants(fragment, printlevel=1)
        
        #Rotational energy and entropy
        if moltype == "atom":
            q_r=1.0
            S_rot=0.0
            E_rot=0.0
        elif moltype == "linear":
            #Symmetry number
            sigma_r=1.0
            q_r=(1/sigma_r)*(temp/(rot_temps_x))
            S_rot=constants.R_gasconst*(math.log(q_r)+1.0)
            E_rot=constants.R_gasconst*temp
        else:
            #Nonlinear case
            #Symmetry number hardcoded. TODO: properly
            sigma_r=2.0
            q_r=(math.pi**(1/2) / sigma_r ) * (temp**(3/2)) / ((rot_temps_x*rot_temps_y*rot_temps_z)**(1/2))
            S_rot=constants.R_gasconst*(math.log(q_r)+1.5)
            E_rot=1.5*constants.R_gasconst*temp
        TS_rot=temp*S_rot
    else:
        E_rot=0.0
        TS_rot=0.0

    ###################
    #TRANSLATIONAL PART
    ###################
    E_trans=1.5*constants.R_gasconst*temp
    
    #R gas constant in kcal/molK
    R_kcalpermolK=1.987E-3
    #Conversion factor for formula.
    #TODO: cleanup
    factor=0.025607868
    #Translation partition function and T*S_trans. Using kcal/mol
    qtrans=(factor*temp**2.5*totalmass**1.5)/pressure
    S_trans=R_kcalpermolK*(math.log(qtrans)+2.5)
    
    TS_trans=temp*S_trans/constants.harkcal #Energy term converted to Eh

    #######################
    #Electronic entropy
    #######################
    q_el=multiplicity
    S_el=constants.R_gasconst*math.log(q_el)
    TS_el=temp*S_el

    #######################
    # Thermodynamic corrections
    #######################
    E_tot = E_vib + E_trans + E_rot
    Hcorr = E_vib + E_trans + E_rot + constants.R_gasconst*temp
    TS_tot = TS_el + TS_trans + TS_rot + TS_vib
    Gcorr = Hcorr - TS_tot


    #######################
    #PRINTING
    #######################
    print("")
    print("Thermochemistry")
    print("--------------------")
    print("Temperature:", temp, "K")
    print("Pressure:", pressure, "atm")
    print("Total atomlist:", fragment.atomlist)
    print("Hessian atomlist:", atoms)
    print("Masses:", masses)
    print("Total mass:", totalmass)
    print("")

    if moltype != "atom":
        print("Moments of inertia:", rinertia)
        print("Rotational constants (cm-1):", rotconstants)

    print("")
    #Thermal corrections
    print("Energy corrections:")
    print("Zero-point vibrational energy:", zpve)
    print("{} {} {} {} {}".format("Translational energy (", temp, "K) :", E_trans, "Eh"))
    print("{} {} {} {} {}".format("Rotational energy (", temp, "K) :", E_rot, "Eh"))
    print("{} {} {} {} {}".format("Total vibrational energy (", temp, "K) :", E_vib, "Eh"))
    print("{} {} {} {} {}".format("Vibrational energy correction (", temp, "K) :", vibenergycorr, "Eh"))
    print("")
    print("Entropy terms (TS):")
    print("{} {} {} {} {}".format("Translational entropy (TS_trans) (", temp, "K) :", TS_trans, "Eh"))
    print("{} {} {} {} {}".format("Rotational entropy (TS_rot) (", temp, "K) :", TS_rot, "Eh"))
    print("{} {} {} {} {}".format("Vibrational entropy (TS_vib) (", temp, "K) :", TS_vib, "Eh"))
    print("{} {} {} {} {}".format("Electronic entropy (TS_el) (", temp, "K) :", TS_el, "Eh"))
    print("")
    if moltype != "atom":
        print("Note: symmetry number : {} used for rotational entropy".format(sigma_r))
        print("")
    print("Thermodynamic terms:")
    print("{} {} {} {} {}".format("Enthalpy correction (Hcorr) (", temp, "K) :", Hcorr, "Eh"))
    print("{} {} {} {} {}".format("Entropy correction (TS_tot) (", temp, "K) :", TS_tot, "Eh"))
    print("{} {} {} {} {}".format("Gibbs free energy correction (Gcorr) (", temp, "K) :", Gcorr, "Eh"))
    print("")
    
    #Dict with properties
    thermochemcalc = {}
    thermochemcalc['frequencies'] = freqs
    thermochemcalc['ZPVE'] = zpve
    thermochemcalc['E_trans'] = E_trans
    thermochemcalc['E_rot'] = E_rot
    thermochemcalc['E_vib'] = E_vib
    thermochemcalc['E_tot'] = E_tot
    thermochemcalc['TS_trans'] = TS_trans
    thermochemcalc['TS_rot'] = TS_rot
    thermochemcalc['TS_vib'] = TS_vib
    thermochemcalc['TS_el'] = TS_el
    thermochemcalc['vibenergycorr'] = vibenergycorr
    thermochemcalc['Hcorr'] = Hcorr
    thermochemcalc['Gcorr'] = Gcorr
    thermochemcalc['TS_tot'] = TS_tot
    
    return thermochemcalc

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
    complexflag=False
    for mode in range(3*numatoms):
        smode = str(mode) + ':'
        if mode < TRmodenum:
            freq=0.00
        else:
            freq=clean_number(vfreq[mode])
            if np.iscomplex(freq):
                imagfreq=-1*abs(freq)
                complexflag=True
            else:
                complexflag=False
        if complexflag==True:
            line= "  {0:>3s}{1:13.2f} cm**-1 ***imaginary mode***".format(smode, imagfreq)
        else:
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
    
    

#Center of mass, adapted from https://code.google.com/p/pistol/source/browse/trunk/Pistol/Thermo.py?r=4
def get_center(elems,coords):
    "compute center of mass"
    xcom,ycom,zcom = 0,0,0
    totmass = 0
    for el,coord in zip(elems,coords):
        mass = atommasses[elematomnumbers[el.lower()]-1]
        xcom += float(mass)*float(coord[0])
        ycom += float(mass)*float(coord[1])
        zcom += float(mass)*float(coord[2])
        totmass += float(mass)
    xcom = xcom/totmass
    ycom = ycom/totmass
    zcom = zcom/totmass
    return xcom,ycom,zcom

def inertia(elems,coords,center):
    xcom=center[0]
    ycom=center[1]
    zcom=center[2]
    Ixx = 0.
    Iyy = 0.
    Izz = 0.
    Ixy = 0.
    Ixz = 0.
    Iyz = 0.

    for index,(el,coord) in enumerate(zip(elems,coords)):
        mass = atommasses[elematomnumbers[el.lower()]-1]
        x = coord[0] - xcom
        y = coord[1] - ycom
        z = coord[2] - zcom

        Ixx += mass * (y**2. + z**2.)
        Iyy += mass * (x**2. + z**2.)
        Izz += mass * (x**2. + y**2.)
        Ixy += mass * x * y
        Ixz += mass * x * z
        Iyz += mass * y * z

    I_ = np.matrix([[ Ixx, -Ixy, -Ixz], [-Ixy,  Iyy, -Iyz], [-Ixz, -Iyz,  Izz]])
    I = np.linalg.eigvals(I_)
    return I
    
def calc_rotational_constants(frag, printlevel=2):
    coords=frag.coords
    elems=frag.elems
    center = get_center(elems,coords)
    rinertia = list(inertia(elems,coords,center))

    #Converting from moments of inertia in amu A^2 to rotational constants in Ghz.
    #COnversion factor from http://openmopac.net/manual/thermochemistry.html
    rot_constants=[]
    for inertval in rinertia:
        #Only calculating constant if moment of inertia value not zero
        if inertval != 0.0:
            rot_ghz=5.053791E5/(inertval*1000)
            rot_constants.append(rot_ghz)
    
    rot_constants_cm = [i*constants.GHztocm for i in rot_constants]
    if printlevel >= 2:
        print("Moments of inertia (amu A^2 ):", rinertia)
        print("Rotational constants (GHz):", rot_constants)
        print("Rotational constants (cm-1):", rot_constants_cm)
        print("Note: If moment of inertia is zero then rotational constant is infinite and not printed ")

    return rot_constants_cm


def calc_model_Hessian_ORCA(fragment,model='Almloef'):
    #Run ORCA dummy job to get Almloef/Lindh/Schlegel Hessian
    orcasimple="! hf noiter opt"
    orcablocks="""
    %geom
    maxiter 1
    inhess {}
    end
""".format(model)
    orcadummycalc=ash.ORCATheory(orcasimpleinput=orcasimple,orcablocks=orcablocks,charge=0,mult=1)
    ash.Singlepoint(theory=orcadummycalc, fragment=fragment)
    #Read orca-input.opt containing Hessian under hessian_approx
    hesstake=False
    j=0
    #Different from orca.hess apparently
    orcacoldim=6
    shiftpar=0
    lastchunk=False
    grabsize=False
    with open("orca-input.opt") as optfile:
        for line in optfile:
            if '$bmatrix' in line:
                hesstake=False
                continue
            if hesstake==True and len(line.split()) == 2 and grabsize==True:
                grabsize=False
                hessdim=int(line.split()[0])

                hessarray2d=np.zeros((hessdim, hessdim))
            if hesstake==True and len(line.split()) == 6:
                continue
                #Headerline
            if hesstake==True and lastchunk==True:
                if len(line.split()) == hessdim - shiftpar +1:
                    for i in range(0,hessdim - shiftpar):
                        hessarray2d[j,i+shiftpar]=line.split()[i+1]
                    j+=1
            if hesstake==True and len(line.split()) == 7:
                # Hessianline
                for i in range(0, orcacoldim):
                    hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                j += 1
                if j == hessdim:
                    shiftpar += orcacoldim
                    j = 0
                    if hessdim - shiftpar < orcacoldim:
                        lastchunk = True
            if '$hessian_approx' in line:
                hesstake = True
                grabsize = True
    fragment.hessian=hessarray2d



#Function to approximate large Hessian from smaller subsystem Hessian
def approximate_full_Hessian_from_smaller(fragment_small,fragment_large,hessian_small,capping_atoms,restHessian='Almloef'):
    #Capping atom Hessian indices are skipped
    capping_atom_hessian_indices=[3*i+j for i in capping_atoms for j in [0,1,2]]
    fragment_large.hessian=np.zeros((fragment_large.numatoms*3,fragment_large.numatoms*3))

    #Fill up hessian_large with Almlöf approximation from ORCA  here?
    calc_model_Hessian_ORCA(fragment_large,model=restHessian)

    for i in range(len(hessian_small)):
        for j in range(len(hessian_small)):
            #Only modifying full-Hessian if not capping atom
            if i not in capping_atom_hessian_indices or j not in capping_atom_hessian_indices:
                fragment_large.hessian[i,j]=hessian_small[i,j]
    return fragment_large.hessian
