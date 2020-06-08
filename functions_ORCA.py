import subprocess as sp
from functions_solv import *
from functions_coords import *
from functions_general import *
from elstructure_functions import *
import settings_solvation
import constants
import multiprocessing as mp


# Once inputfiles are ready, organize them. We want open-shell calculation (e.g. oxidized) to reuse closed-shell GBW file
# https://www.machinelearningplus.com/python/parallel-processing-python/
# Good subprocess documentation: http://queirozf.com/entries/python-3-subprocess-examples
# https://shuzhanfan.github.io/2017/12/parallel-processing-python-subprocess/
# https://data-flair.training/blogs/python-multiprocessing/
# https://rsmith.home.xs4all.nl/programming/parallel-execution-with-python.html
def run_inputfiles_in_parallel(orcadir, inpfiles, numcores):
    """
    Run inputfiles in parallel using multiprocessing
    :param orcadir: path to ORCA directory
    :param inpfiles: list of inputfiles
    :param numcores: number of cores to use (integer)
    ;return: returns nothing. Outputfiles on disk parsed separately
    """
    blankline()
    print("Number of CPU cores: ", numcores)
    print("Number of inputfiles:", len(inpfiles))
    print("Running snapshots in parallel")
    pool = mp.Pool(numcores)
    results = pool.map(run_orca_SP, [[orcadir,file] for file in inpfiles])
    pool.close()
    print("Calculations are done")

#Run single-point ORCA calculation (Energy or Engrad). Assumes no ORCA parallelization.
#Function can be called by multiprocessing.
def run_orca_SP(list, Grad=False):
    orcadir=list[0]
    inpfile=list[1]
    print("Running inpfile", inpfile)
    if Grad==True:
        with open(inpfile) as ifile:
            insert_line_into_file(inpfile, '!', '! Engrad')
    basename = inpfile.split('.')[0]
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([orcadir + '/orca', basename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

# Run ORCA single-point job using ORCA parallelization. Will add pal-block if nprocs >1.
# Takes possible Grad boolean argument.
def run_orca_SP_ORCApar(orcadir, inpfile, nprocs=1, Grad=False):
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
def ORCAfinalenergygrab(file):
    with open(file) as f:
        for line in f:
            if 'FINAL SINGLE POINT ENERGY' in line:
                Energy=float(line.split()[-1])
    return Energy

#Grab gradient from ORCA engrad file
def ORCAgradientgrab(engradfile):
    grab=False
    numatomsgrab=False
    row=0
    col=0
    with open(engradfile) as gradfile:
        for line in gradfile:
            if numatomsgrab==True:
                if '#' not in line:
                    numatoms=int(line.split()[0])
                    #Initializing array
                    gradient = np.zeros((numatoms, 3))
                    numatomsgrab=False
            if '# Number of atoms' in line:
                numatomsgrab=True
            if grab == True:
                if '#' not in line:
                    val=float(line.split()[0])
                    gradient[row, col] = val
                    if col == 2:
                        row+=1
                        col=0
                    else:
                        col+=1
            if '# The current gradient in Eh/bohr' in line:
                grab=True
            if '# The atomic numbers and ' in line:
                grab=False
    return gradient

#Grab pointcharge gradient from ORCA pcgrad file
def ORCApcgradientgrab(pcgradfile):
    with open(pcgradfile) as pgradfile:
        for count,line in enumerate(pgradfile):
            if count==0:
                numatoms=int(line.split()[0])
                #Initializing array
                gradient = np.zeros((numatoms, 3))
            elif count > 0:
                val_x=float(line.split()[0])
                val_y = float(line.split()[1])
                val_z = float(line.split()[2])
                gradient[count-1] = [val_x,val_y,val_z]
    return gradient



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
    with open(hessfile) as hfile:
        for line in hfile:
            if cartgrab==True:
                count=count+1
                elem=line.split()[0]; x_c=constants.bohr2ang*float(line.split()[2]);y_c=constants.bohr2ang*float(line.split()[3]);z_c=constants.bohr2ang*float(line.split()[4])
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

#Function to write ORCA-style Hessian file

def write_ORCA_Hessfile(hessian, coords, elems, masses, hessatoms,outputname):
    hessdim=hessian.shape[0]
    orcahessfile = open(outputname,'w')
    orcahessfile.write("$orca_hessian_file\n")
    orcahessfile.write("\n")
    orcahessfile.write("$hessian\n")
    orcahessfile.write(str(hessdim)+"\n")
    orcahesscoldim=5
    index=0
    tempvar=""
    temp2var=""
    chunks=hessdim//orcahesscoldim
    left=hessdim%orcahesscoldim
    if left > 0:
        chunks=chunks+1
    for chunk in range(chunks):
        if chunk == chunks-1:
            #If last chunk and cleft is exactly 0 then all 5 columns should be done
            if left == 0:
                left=5
            for temp in range(index,index+left):
                temp2var=temp2var+"         "+str(temp)
        else:
            for temp in range(index,index+orcahesscoldim):
                temp2var=temp2var+"         "+str(temp)
        orcahessfile.write(str(temp2var)+"\n")
        for i in range(0,hessdim):

            if chunk == chunks-1:
                for k in range(index,index+left):
                    tempvar=tempvar+"         "+str(hessian[i,k])
            else:
                for k in range(index,index+orcahesscoldim):
                    tempvar=tempvar+"         "+str(hessian[i,k])
            orcahessfile.write("    "+str(i)+"   "+str(tempvar)+"\n")
            tempvar="";temp2var=""
        index+=5
    orcahessfile.write("\n")
    orcahessfile.write("# The atoms: label  mass x y z (in bohrs)\n")
    orcahessfile.write("$atoms\n")
    orcahessfile.write(str(len(elems))+"\n")

    #Write coordinates and masses to Orca Hessian file
    print("hessatoms", hessatoms)
    print("masses ", masses)
    print("elems ", elems)
    print("coords", coords)
    print(len(elems))
    print(len(coords))
    print(len(hessatoms))
    print(len(masses))
    #TODO. Note. Changed things. We now don't go through hessatoms and analyze atom indices for full system
    #Either full system lists were passed or partial-system lists
    #for atom, mass in zip(hessatoms, masses):
    for el,mass,coord in zip(elems,masses,coords):
        #mass=atommass[elements.index(elems[atom-1].lower())]
        #print("atom:", atom)
        #print("mass:", mass)
        #print(str(elems[atom]))
        #print(str(mass))
        #print(str(coords[atom][0]/constants.bohr2ang))
        #print(str(coords[atom][1]/constants.bohr2ang))
        #print(str(coords[atom][2]/constants.bohr2ang))
        #orcahessfile.write(" "+str(elems[atom])+'    '+str(mass)+"  "+str(coords[atom][0]/constants.bohr2ang)+
        #                   " "+str(coords[atom][1]/constants.bohr2ang)+" "+str(coords[atom][2]/constants.bohr2ang)+"\n")
        orcahessfile.write(" "+el+'    '+str(mass)+"  "+str(coord[0]/constants.bohr2ang)+
                           " "+str(coord[1]/constants.bohr2ang)+" "+str(coord[2]/constants.bohr2ang)+"\n")
    orcahessfile.write("\n")
    orcahessfile.write("\n")
    orcahessfile.close()
    print("")
    print("ORCA-style Hessian written to:", outputname )

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



#Create PC-embedded ORCA inputfile from elems,coords, input, charge, mult,pointcharges
# Compound method version. Doing both redox states in same job.
#Adds specific basis set on atoms not defined as solute-atoms.
def create_orca_inputVIEcomp_pc(name,name2, elems,coords,orcasimpleinput,orcablockinput,chargeA,multA,chargeB,multB, soluteatoms, basisname):
    pcfile=name+'.pc'
    basisnameline="newgto \"{}\" end".format(basisname)
    with open(name2+'.inp', 'w') as orcafile:
        #Geometry block first in compounds job
        #Adding xyzfile to orcasimpleinput
        orcafile.write('*xyz {} {}\n'.format(chargeA,multA))
        count=0
        for el,c in zip(elems,coords):
            if len(basisname) > 2 and count >= len(soluteatoms):
                    orcafile.write('{} {} {} {} {} \n'.format(el, c[0], c[1], c[2], basisnameline))
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
            if len(basisname) > 2 and count >= len(soluteatoms):
                    orcafile.write('{} {} {} {} {} \n'.format(el, c[0], c[1], c[2], basisnameline))
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
            if len(basisname) > 2 and count >= len(soluteatoms):
                    orcafile.write('{} {} {} {} {} \n'.format(el, c[0], c[1], c[2], basisnameline))
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
#Allows for extraline that could be another '!' line or block-inputline.
#Used by Yggdrasill
def create_orca_input_pc(name,elems,coords,orcasimpleinput,orcablockinput,charge,mult, Grad=False, extraline='',
                         HSmult=None, atomstoflip=None):
    pcfile=name+'.pc'
    with open(name+'.inp', 'w') as orcafile:
        orcafile.write(orcasimpleinput+'\n')
        if extraline != '':
            orcafile.write(extraline + '\n')
        if Grad == True:
            orcafile.write('! Engrad' + '\n')
        orcafile.write('%pointcharges "{}"\n'.format(pcfile))
        orcafile.write(orcablockinput + '\n')
        if atomstoflip is not None:
            atomstoflipstring= ','.join(map(str, atomstoflip))
            orcafile.write('%scf\n')
            orcafile.write('Flipspin {}'.format(atomstoflipstring)+ '\n')
            orcafile.write('FinalMs {}'.format((mult-1)/2)+ '\n')
            orcafile.write('end  \n')
        orcafile.write('\n')
        if atomstoflip is not None:
            orcafile.write('*xyz {} {}\n'.format(charge,HSmult))
        else:
            orcafile.write('*xyz {} {}\n'.format(charge,mult))
        for el,c in zip(elems,coords):
            orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
        orcafile.write('*\n')

#Create simple ORCA inputfile from elems,coords, input, charge, mult,pointcharges
#Allows for extraline that could be another '!' line or block-inputline.
#Used by ASH
def create_orca_input_plain(name,elems,coords,orcasimpleinput,orcablockinput,charge,mult, Grad=False, extraline='',
                            HSmult=None, atomstoflip=None):
    with open(name+'.inp', 'w') as orcafile:
        orcafile.write(orcasimpleinput+'\n')
        if extraline != '':
            orcafile.write(extraline + '\n')
        if Grad == True:
            orcafile.write('! Engrad' + '\n')
        orcafile.write(orcablockinput + '\n')
        if atomstoflip is not None:
            if type(atomstoflip) == int:
                atomstoflipstring=str(atomstoflip)
            else:
                atomstoflipstring= ','.join(map(str, atomstoflip))
            orcafile.write('%scf\n')
            orcafile.write('Flipspin {}'.format(atomstoflipstring)+ '\n')
            orcafile.write('FinalMs {}'.format((mult-1)/2)+ '\n')
            orcafile.write('end  \n')
        orcafile.write('\n')
        if atomstoflip is not None:
            orcafile.write('*xyz {} {}\n'.format(charge,HSmult))
        else:
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

# Chargemodel select. Creates ORCA-inputline with appropriate keywords
# To be added to ORCA input.
def chargemodel_select(chargemodel):
    extraline=""
    if chargemodel=='NPA':
        extraline='! NPA'
    elif chargemodel=='CHELPG':
        extraline='! CHELPG'
    elif chargemodel=='Hirshfeld':
        extraline='! Hirshfeld'
    elif chargemodel=='CM5':
        extraline='! Hirshfeld'
    elif chargemodel=='Mulliken':
        pass
    elif chargemodel=='Loewdin':
        pass
    elif chargemodel=="IAO":
        extraline = '\n%loc LocMet IAOIBO \n T_CORE -99999999 end'

    return extraline

def grabatomcharges_ORCA(chargemodel,outputfile):
    grab=False
    coordgrab=False
    charges=[]

    if chargemodel=="NPA" or chargemodel=="NBO":
        print("Warning: NPA/NBO charge-option in ORCA requires setting environment variable NBOEXE:")
        print("e.g. export NBOEXE=/path/to/nbo7.exe")
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if '=======' in line:
                        grab=False
                    elif '------' not in line:
                        charges.append(float(line.split()[2]))
                if 'Atom No    Charge        Core      Valence    Rydberg      Total' in line:
                    grab=True
    elif chargemodel=="CHELPG":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if 'Total charge: ' in line:
                        grab=False
                    if len(line.split()) == 4:
                        charges.append(float(line.split()[-1]))
                if 'CHELPG Charges' in line:
                    grab=True
    elif chargemodel=="Hirshfeld":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if len(line) < 3:
                        grab=False
                    if len(line.split()) == 4:
                        charges.append(float(line.split()[-2]))
                if '  ATOM     CHARGE      SPIN' in line:
                    grab=True
    elif chargemodel=="CM5":
        elems = []
        coords = []
        with open(outputfile) as ofile:
            for line in ofile:
                #Getting coordinates as used in CM5 definition
                if coordgrab is True:
                    if '----------------------' not in line:
                        if len(line.split()) <2:
                            coordgrab=False
                        else:
                            elems.append(line.split()[0])
                            coords_x=float(line.split()[1]); coords_y=float(line.split()[2]); coords_z=float(line.split()[3])
                            coords.append([coords_x,coords_y,coords_z])
                if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                    coordgrab=True
                if grab==True:
                    if len(line) < 3:
                        grab=False
                    if len(line.split()) == 4:
                        charges.append(float(line.split()[-2]))
                if '  ATOM     CHARGE      SPIN' in line:
                    grab=True
        print("Hirshfeld charges :", charges)
        atomicnumbers=elemstonuccharges(elems)
        print("atomicnumbers:", atomicnumbers)
        charges = calc_cm5(atomicnumbers, coords, charges)
        print("CM5 charges :", charges)
    elif chargemodel == "Mulliken":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if 'Sum of atomic' in line:
                        grab=False
                    elif '------' not in line:
                        charges.append(float(line.split()[3]))
                if 'MULLIKEN ATOMIC CHARGES' in line:
                    grab=True
    elif chargemodel == "Loewdin":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if 'Sum of atomic' in line:
                        grab=False
                    elif len(line.replace(' ','')) < 2:
                        grab=False
                    elif '------' not in line:
                        charges.append(float(line.split()[3]))
                if 'LOEWDIN ATOMIC CHARGES' in line:
                    grab=True
    elif chargemodel == "IAO":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if 'Sum of atomic' in line:
                        grab=False
                    elif '------' not in line:
                        if 'Warning!!!' not in line:
                            charges.append(float(line.split()[3]))
                if 'IAO PARTIAL CHARGES' in line:
                    grab=True
    else:
        print("Unknown chargemodel. Exiting...")
        exit()
    return charges

