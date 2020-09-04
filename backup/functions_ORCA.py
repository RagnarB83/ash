import subprocess as sp
from functions_solv import *
from functions_coords import *
from functions_general import *
import settings_solvation
import constants

#TODO: Remove solvsphere calls
#TODO: Also settings_solvation calls. orcadir and bulksphere

#Grab 2 energies from ORCA output, e.g. VIEs.
#TODO: move to functions_solv or make more general
def grab_energies_output(inpfiles):
    # Dictionaries to  hold VIEs. Currently not keeping track of total energies
    AsnapsABenergy = {}
    BsnapsABenergy = {}
    AllsnapsABenergy = {}
    for snap in inpfiles:
        snapbase=snap.split('_')[0]
        outfile=snap.replace('.inp','.out')
        done=checkORCAfinished(outfile)
        if done==True:
            energies=finalenergiesgrab(outfile)
            delta_AB=(energies[1]-energies[0])*constants.hartoeV
            if 'snapA' in snapbase:
                AsnapsABenergy[snapbase]=delta_AB
            elif 'snapB' in snapbase:
                BsnapsABenergy[snapbase]=delta_AB
            AllsnapsABenergy[snapbase]=delta_AB
    return AllsnapsABenergy, AsnapsABenergy, BsnapsABenergy


def run_inputfiles_in_parallel_AB(inpfiles):
    import multiprocessing as mp
    import shutil

    # Once inputfiles are ready, organize them. We want open-shell calculation (e.g. oxidized) to reuse closed-shell GBW file
    #https://www.machinelearningplus.com/python/parallel-processing-python/
    #Good subprocess documentation: http://queirozf.com/entries/python-3-subprocess-examples
    #https://shuzhanfan.github.io/2017/12/parallel-processing-python-subprocess/
    #https://data-flair.training/blogs/python-multiprocessing/
    #https://rsmith.home.xs4all.nl/programming/parallel-execution-with-python.html
    blankline()
    print("Number of CPU cores: ", settings_solvation.NumCores)
    print("Number of inputfiles:", len(inpfiles))
    print("Running snapshots in parallel (both states in each calculation).")
    pool = mp.Pool(settings_solvation.NumCores)
    results = pool.map(run_orca_SP, [file for file in inpfiles])
    pool.close()
    print("Calculations are done")

# Run ORCA single-point job.
# Unnecessary. To be deleted
def run_orca_SP(inpfile):
    basename = inpfile.split('.')[0]
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([settings_solvation.orcadir + '/orca', basename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

# Run ORCA single-point job using ORCA parallelization. Takes possible Grad boolean argument.
def run_orca_SP_ORCApar(orcadir, inpfile,nprocs=1, Grad=False):
    if Grad==True:
        with open(inpfile) as ifile:
            insert_line_into_file(inpfile, '!', '! Engrad')
    #Add pal block to inputfile before running. Adding after '!' line. Should work for regular, new_job and compound job.
    if nprocs>1:
        palstring='% pal nprocs {} end'.format(nprocs)
        with open(inpfile) as ifile:
            insert_line_into_file(inpfile, '!', palstring )
    basename = inpfile.split('.')[0]
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([orcadir + '/orca', basename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)


#Check if ORCA finished.
#Todo: Use reverse-read instead to speed up?
def checkORCAfinished(file):
    with open(file) as f:
        for line in f:
            if 'TOTAL RUN TIME:' in line:
                return True

#Grab Final single point energy
def finalenergygrab(file):
    with open(file) as f:
        for line in f:
            if 'FINAL SINGLE POINT ENERGY' in line:
                Energy=float(line.split()[-1])
    return Energy

#Grab multiple Final single point energies in output. e.g. new_job calculation
def finalenergiesgrab(file):
    energies=[]
    with open(file) as f:
        for line in f:
            if 'FINAL SINGLE POINT ENERGY' in line:
                energies.append(float(line.split()[-1]))
    return energies

#Grab SCF energy (non-dispersion corrected)
def scfenergygrab(file):
    with open(file) as f:
        for line in f:
            if 'Total Energy       :' in line:
                Energy=float(line.split()[-4])
    return Energy

#Grab TDDFT states from ORCA output
def tddftgrab(file):
    tddft=True
    tddftgrab=False
    if tddft==True:
        with open(file) as f:
            for line in file:
                if tddftgrab==True:
                    if 'STATE' in line:
                        tddftstates.append(float(line.split()[5]))
                        tddftgrab=True
                if 'the weight of the individual excitations' in line:
                    tddftgrab=True
    return tddftstates

#Grab energies from unrelaxed scan in ORCA (paras block type)
def grabtrajenergies(filename):
    fullpes="unset"
    trajsteps=[]
    stepvals=[]
    stepval=0
    energies=[]
    with open(filename, errors='ignore') as file:
        for line in file:
            if 'Parameter Scan Calculation' in line:
                fullpes="yes"
            if fullpes=="yes":
                if 'TRAJECTORY STEP' in line:
                    trajstep=int(line.split()[2])
                    trajsteps.append(trajstep)
                    temp=next(file)
                    stepval=float(temp.split()[2])
                    stepvals.append(stepval)
            if 'FINAL SINGLE' in line:
                energies.append(float(line.split()[-1]))
    #if 'TOTAL RUN' in line:
    #    return energies
    return energies,stepvals




#Grab alpha and beta orbital energies from ORCA SCF job
def orbitalgrab(file):
    occorbsgrab=False
    virtorbsgrab=False
    endocc="unset"
    tddftgrab="unset"
    tddft="unset"
    bands_alpha=[]
    bands_beta=[]
    virtbands_a=[]
    virtbands_b=[]
    f=[]
    virtf=[]
    spinflag="unset"
    hftyp="unset"

    with open(file) as f:
        for line in f:
            if '%tddft' in line:
                tddft="yes"
            if 'Hartree-Fock type      HFTyp' in line:
                hftyp=line.split()[4]
                #if hftyp=="UHF":
            if hftyp == "RHF":
                spinflag="alpha"
            if 'SPIN UP ORBITALS' in line:
                spinflag="alpha"
            if 'SPIN DOWN ORBITALS' in line:
                spinflag="beta"
            if occorbsgrab==True:
                endocc=line.split()[1]
                if endocc == "0.0000" :
                    occorbsgrab=False
                    virtorbsgrab=True
                else:
                    if spinflag=="alpha":
                        bands_alpha.append(float(line.split()[3]))
                    if spinflag=="beta":
                        bands_beta.append(float(line.split()[3]))
            if virtorbsgrab==True:
                if '------------------' in line:
                    break
                if line == '\n':
                    virtorbsgrab=False
                    spinflag="unset"
                    continue
                if spinflag=="alpha":
                    virtbands_a.append(float(line.split()[3]))
                if spinflag=="beta":
                    virtbands_b.append(float(line.split()[3]))
                endvirt=line.split()[1]
            if 'NO   OCC          E(Eh)            E(eV)' in line:
                occorbsgrab=True
    return bands_alpha, bands_beta, hftyp

#Function to grab masses and elements from ORCA Hessian file
def masselemgrab(hessfile):
    grab=False
    elems=[]; masses=[]
    with open(hessfile) as hfile:
        for line in hfile:
            if '$actual_temperature' in line:
                grab=False
            if grab==True and len(line.split()) == 1:
                numatoms=int(line.split()[0])
            if grab==True and len(line.split()) == 5 :
                elems.append(line.split()[0])
                masses.append(float(line.split()[1]))
            if '$atoms' in line:
                grab=True
    return masses, elems,numatoms

def grabcoordsfromhessfile(hessfile):
    #Grab coordinates from hessfile
    numatomgrab=False
    cartgrab=False
    elements=[]
    coords=[]
    count=0
    bohrang=0.529177249
    with open(hessfile) as hfile:
        for line in hfile:
            if cartgrab==True:
                count=count+1
                elem=line.split()[0]; x_c=bohrang*float(line.split()[2]);y_c=bohrang*float(line.split()[3]);z_c=bohrang*float(line.split()[4])
                elements.append(elem)
                coords.append([x_c,y_c,z_c])
                if count == numatoms:
                    break
            if numatomgrab==True:
                numatoms=int(line.split()[0])
                numatomgrab=False
                cartgrab=True
            if "$atoms" in line:
                numatomgrab=True
    return elements,coords

#Function to grab Hessian from ORCA-Hessian file
def Hessgrab(hessfile):
    hesstake=False
    j=0
    orcacoldim=5
    shiftpar=0
    lastchunk=False
    grabsize=False
    with open(hessfile) as hfile:
        for line in hfile:
            if '$vibrational_frequencies' in line:
                hesstake=False
                continue
            if hesstake==True and len(line.split()) == 1 and grabsize==True:
                grabsize=False
                hessdim=int(line.split()[0])

                hessarray2d=np.zeros((hessdim, hessdim))
            if hesstake==True and len(line.split()) == 5:
                continue
                #Headerline
            if hesstake==True and lastchunk==True:
                if len(line.split()) == hessdim - shiftpar +1:
                    for i in range(0,hessdim - shiftpar):
                        hessarray2d[j,i+shiftpar]=line.split()[i+1]
                    j+=1
            if hesstake==True and len(line.split()) == 6:
                # Hessianline
                for i in range(0, orcacoldim):
                    hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                j += 1
                if j == hessdim:
                    shiftpar += orcacoldim
                    j = 0
                    if hessdim - shiftpar < orcacoldim:
                        lastchunk = True
            if '$hessian' in line:
                hesstake = True
                grabsize = True
            return hessarray2d

#Massweight Hessian
def massweight(matrix):
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


#Create PC-embedded ORCA inputfile from elems,coords, input, charge, mult,pointcharges
# Compound method version. Doing both redox states in same job.
def create_orca_inputVIEcomp_pc(name,name2, elems,coords,orcasimpleinput,orcablockinput,chargeA,multA,chargeB,multB, solvsphere, solvbasis):
    pcfile=name+'.pc'
    solvbasisline="newgto \"{}\" end".format(solvbasis)
    with open(name2+'.inp', 'w') as orcafile:
        #Geometry block first in compounds job
        #Adding xyzfile to orcasimpleinput
        orcafile.write('*xyz {} {}\n'.format(chargeA,multA))
        count=0
        for el,c in zip(elems,coords):
            if len(solvbasis) > 2 and count >= len(solvsphere.soluteatomsA):
                    orcafile.write('{} {} {} {} {} \n'.format(el, c[0], c[1], c[2], solvbasisline))
            else:
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
            count += 1
        orcafile.write('*\n')
        orcafile.write('\n')
        orcafile.write('%Compound\n')
        orcafile.write('New_Step\n')
        orcafile.write('\n')
        orcafile.write(orcasimpleinput+'\n')
        orcafile.write('%pointcharges "{}"\n'.format(pcfile))
        orcafile.write(orcablockinput + '\n')
        orcafile.write('\n')
        orcafile.write('*xyz {} {}\n'.format(chargeA,multA))
        count=0
        for el,c in zip(elems,coords):
            if len(solvbasis) > 2 and count >= len(solvsphere.soluteatomsA):
                    orcafile.write('{} {} {} {} {} \n'.format(el, c[0], c[1], c[2], solvbasisline))
            else:
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
            count += 1
        orcafile.write('*\n')
        orcafile.write('STEP_END\n')
        orcafile.write('\n')
        orcafile.write('New_Step\n')
        orcafile.write('\n')
        orcafile.write(orcasimpleinput+' MOREAD \n')
        #GBW filename of compound-job no. 1
        moinpfile=name2+'_Compound_1.gbw'
        orcafile.write('%moinp "{}"\n'.format(moinpfile))
        orcafile.write(orcablockinput + '\n')
        orcafile.write('%pointcharges "{}"\n'.format(pcfile))
        orcafile.write('\n')
        #Geometry block first in compounds job
        orcafile.write('*xyz {} {}\n'.format(chargeB,multB))
        count=0
        for el,c in zip(elems,coords):
            if len(solvbasis) > 2 and count >= len(solvsphere.soluteatomsA):
                    orcafile.write('{} {} {} {} {} \n'.format(el, c[0], c[1], c[2], solvbasisline))
            else:
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
            count += 1
        orcafile.write('*\n')
        orcafile.write('\n')
        orcafile.write('STEP_END\n')
        orcafile.write('end\n')





#Create PC-embedded ORCA inputfile from elems,coords, input, charge, mult,pointcharges
# new_job feature. Doing both redox states in same job.
#Works buts discouraged.
def create_orca_inputVIE_pc(name,name2, elems,coords,orcasimpleinput,orcablockinput,chargeA,multA,chargeB,multB):
    pcfile=name+'.pc'
    with open(name2+'.inp', 'w') as orcafile:
        #Adding xyzfile to orcasimpleinput
        orcasimpleinput=orcasimpleinput+' xyzfile'
        orcafile.write(orcasimpleinput+'\n')
        orcafile.write('%pointcharges "{}"\n'.format(pcfile))
        orcafile.write(orcablockinput + '\n')
        orcafile.write('\n')
        orcafile.write('*xyz {} {}\n'.format(chargeA,multA))
        for el,c in zip(elems,coords):
            orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
        orcafile.write('*\n')
        orcafile.write('\n')
        orcafile.write('$new_job\n')
        orcafile.write(orcasimpleinput+'\n')
        orcafile.write('%pointcharges "{}"\n'.format(pcfile))
        orcafile.write(orcablockinput + '\n')
        orcafile.write('\n')
        orcafile.write('*xyzfile {} {}\n'.format(chargeB, multB))

#Create gas ORCA inputfile from elems,coords, input, charge, mult. No pointcharges.
#new_job version. Works but discouraged.
def create_orca_inputVIEnewjob_gas(name,name2, elems,coords,orcasimpleinput,orcablockinput,chargeA,multA,chargeB,multB):
    with open(name2+'.inp', 'w') as orcafile:
        #Adding xyzfile to orcasimpleinput
        orcasimpleinput=orcasimpleinput+' xyzfile'
        orcafile.write(orcasimpleinput+'\n')
        orcafile.write(orcablockinput + '\n')
        orcafile.write('\n')
        orcafile.write('*xyz {} {}\n'.format(chargeA,multA))
        for el,c in zip(elems,coords):
            orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
        orcafile.write('*\n')
        orcafile.write('\n')
        orcafile.write('$new_job\n')
        orcafile.write(orcasimpleinput+'\n')
        orcafile.write(orcablockinput + '\n')
        orcafile.write('\n')
        orcafile.write('*xyzfile {} {}\n'.format(chargeB, multB))

# Create gas ORCA inputfile from elems,coords, input, charge, mult. No pointcharges.
# compoundmethod version.
def create_orca_inputVIEcomp_gas(name, name2, elems, coords, orcasimpleinput, orcablockinput, chargeA, multA, chargeB,
                                 multB):
    with open(name2+'.inp', 'w') as orcafile:
        #Geometry block first in compounds job
        #Adding xyzfile to orcasimpleinput
        orcafile.write('*xyz {} {}\n'.format(chargeA,multA))
        for el,c in zip(elems,coords):
            orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
        orcafile.write('*\n')
        orcafile.write('\n')
        orcafile.write('%Compound\n')
        orcafile.write('New_Step\n')
        orcafile.write('\n')
        orcafile.write(orcasimpleinput+' xyzfile \n')
        orcafile.write(orcablockinput + '\n')
        orcafile.write('\n')
        orcafile.write('STEP_END\n')
        orcafile.write('\n')
        orcafile.write('New_Step\n')
        orcafile.write('\n')
        orcafile.write(orcasimpleinput+' MOREAD \n')
        #GBW filename of compound-job no. 1
        moinpfile=name2+'_Compound_1.gbw'
        orcafile.write('%moinp "{}"\n'.format(moinpfile))
        orcafile.write(orcablockinput + '\n')
        orcafile.write('\n')
        orcafile.write('*xyzfile {} {}\n'.format(chargeB, multB))
        orcafile.write('\n')
        orcafile.write('STEP_END\n')
        orcafile.write('\n')
        orcafile.write('end\n')


#Create PC-embedded ORCA inputfile from elems,coords, input, charge, mult,pointcharges
#Used by Yggdrasill
def create_orca_input_pc(name,elems,coords,orcasimpleinput,orcablockinput,charge,mult):
    pcfile=name+'.pc'
    with open(name+'.inp', 'w') as orcafile:
        orcafile.write(orcasimpleinput+'\n')
        orcafile.write('%pointcharges "{}"\n'.format(pcfile))
        orcafile.write(orcablockinput + '\n')
        orcafile.write('\n')
        orcafile.write('*xyz {} {}\n'.format(charge,mult))
        for el,c in zip(elems,coords):
            orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
        orcafile.write('*\n')

#Create simple ORCA inputfile from elems,coords, input, charge, mult,pointcharges
#Used by Yggdrasill
def create_orca_input_plain(name,elems,coords,orcasimpleinput,orcablockinput,charge,mult):
    with open(name+'.inp', 'w') as orcafile:
        orcafile.write(orcasimpleinput+'\n')
        orcafile.write(orcablockinput + '\n')
        orcafile.write('\n')
        orcafile.write('*xyz {} {}\n'.format(charge,mult))
        for el,c in zip(elems,coords):
            orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
        orcafile.write('*\n')

#Create ORCA pointcharge file based on provided list of elems and coords (MM region elems and coords)
# and list of point charges of MM atoms
def create_orca_pcfile(name,elems,coords,listofcharges):
    with open(name+'.pc', 'w') as pcfile:
        pcfile.write(str(len(elems))+'\n')
        for p,c in zip(listofcharges,coords):
            line = "{} {} {} {}".format(p, c[0], c[1], c[2])
            pcfile.write(line+'\n')

#Create ORCA pointcharge file based on provided list of elems and coords (MM region elems and coords) and charges for solvent unit.
#Assuming elems and coords list are in regular order, e.g. for TIP3P waters: O H H O H H etc.
#Used in Solvshell program. Assumes TIP3P waters.
def create_orca_pcfile_solv(name,elems,coords,solventunitcharges,bulkcorr=False):
    #Creating list of pointcharges based on solventunitcharges and number of elements provided
    pchargelist=solventunitcharges*int(len(elems)/len(solventunitcharges))
    if bulkcorr==True:
        #print("Bulk Correction is On. Modifying Pointcharge file.")
        with open(name+'.pc', 'w') as pcfile:
            pcfile.write(str(len(elems)+settings_solvation.bulksphere.numatoms)+'\n')
            for p,c in zip(pchargelist,coords):
                line = "{} {} {} {}".format(p, c[0], c[1], c[2])
                pcfile.write(line+'\n')
            #Adding pointcharges from hollow bulk sphere
            with open(settings_solvation.bulksphere.pathtofile) as bfile:
                for line in bfile:
                    pcfile.write(line)
    else:
        with open(name+'.pc', 'w') as pcfile:
            pcfile.write(str(len(elems))+'\n')
            for p,c in zip(pchargelist,coords):
                line = "{} {} {} {}".format(p, c[0], c[1], c[2])
                pcfile.write(line+'\n')