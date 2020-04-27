import constants
import subprocess as sp
import settings_solvation
from functions_general import *
import os
import sys
import shutil

#xTB functions: primarily for inputfile-based interface. Library-interfaces is in interface_xtb.py

#TODO: THis should be a general interface so remove settings_solvation calls.
#TODO: xtb. Need to combine OMP-parallelization of xtb and multiprocessing if possible
#TODO: Currently doing multiprocessing over all 8*2=16 snapshots. First A, then B.
#TODO: Might not be a need to do first A then B since a ROHF-type Hamiltonian
#TODO. Could parallelize over all 32 calculations. However, we are currently using 24 cores so...

#Grab Final single point energy
def xtbfinalenergygrab(file):
    with open(file) as f:
        for line in f:
            if 'TOTAL ENERGY' in line:
                Energy=float(line.split()[-3])
    return Energy

#Grab gradient and energy from gradient file
def xtbgradientgrab(numatoms):
    grab=False
    gradient = np.zeros((numatoms, 3))
    count=0
    #Converting Fortran D exponent to E
    t = 'string'.maketrans('D', 'E')
    #Reading file backwards so adding to gradient backwards too
    row=numatoms-1
    #Read file in reverse
    with open('gradient') as f:
        for line in reverse_lines(f):
            if '  cycle =' in line:
                energy=float(line.split()[6])
                return energy, gradient
            if count==numatoms:
                grab=False
            if grab==True:
                gradient[row] = [float( line.split()[0].translate(t)), float(line.split()[1].translate(t)),
                                 float(line.split()[2].translate(t))]
                count+=1
                row-=1
            if '$end' in line:
                grab=True


def xtbVIPgrab(file):
    with open(file) as f:
        for line in f:
            if 'delta SCC IP (eV):' in line:
                VIP=float(line.split()[-1])
    return VIP

def xtbVEAgrab(file):
    with open(file) as f:
        for line in f:
            if 'delta SCC EA' in line:
                VEA=float(line.split()[-1])
    return VIP

# Run xTB single-point job
def run_xtb_SP_serial(xtbdir, xtbmethod, xyzfile, charge, mult, Grad=False):
    basename = xyzfile.split('.')[0]
    uhf=mult-1
    print("uhf: ", uhf)
    print("charge: ", charge)
    print("mult: ", mult)
    print("xtbdir: ", xtbdir)
    print("xtbmethod: ", xtbmethod)
    #Writing xtbinputfile to disk so that we use ORCA-style PCfile and embedding
    with open('xtbinput', 'w') as xfile:
        xfile.write('$embedding\n')
        xfile.write('interface=orca\n')
        xfile.write('end\n')

    if 'GFN2' in xtbmethod.upper():
        xtbflag = 2
    elif 'GFN1' in xtbmethod.upper():
        xtbflag = 1
    elif 'GFN0' in xtbmethod.upper():
        xtbflag = 0
    else:
        print("Unknown xtbmethod chosen. Exiting...")
        exit()
    with open(basename+'.out', 'w') as ofile:
        if Grad==True:
            process = sp.run([xtbdir + '/xtb', basename+'.xyz', '--gfn', str(xtbflag), '--grad', '--chrg', str(charge), '--uhf',
                              str(uhf), '--input', 'xtbinput' ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        else:
            process = sp.run(
                [xtbdir + '/xtb', basename + '.xyz', '--gfn', str(xtbflag), '--chrg', str(charge), '--uhf', str(uhf),
                 '--input', 'xtbinput'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
# Run GFN-xTB single-point job (for multiprocessing execution) for both state A and B (e.g. VIE calc)
#Takes 1 argument: line with xyzfilename and the xtb options.
#Runs inside separate dir
def run_gfnxtb_SPVIE_multiproc(line):
    basename=line.split()[0].split('.')[0]
    xyzfile=line.split()[0]
    #Create dir for snapshot
    os.mkdir(basename)
    os.chdir(basename)
    #Copy xyzfile into it
    shutil.copyfile('../'+xyzfile, './'+xyzfile)
    #Copy pointcharge file into dir as pcharge
    shutil.copyfile('../'+basename+'.pc', './pcharge')
    os.listdir()
    #Silly way of getting arguments from line-string again.
    gfnoption=line.split()[1]
    chargeA=line.split()[2]
    uhfA=line.split()[3]
    chargeB=line.split()[4]
    uhfB=line.split()[5]
    with open(basename+'_StateA.out', 'w') as ofile:
        process = sp.run([settings_solvation.xtbdir + '/xtb', xyzfile, '--gfn', gfnoption, '--chrg', chargeA, '--uhf', uhfA ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    with open(basename+'_StateB.out', 'w') as ofile:
        process = sp.run([settings_solvation.xtbdir + '/xtb', xyzfile, '--gfn', gfnoption, '--chrg', chargeB, '--uhf', uhfB ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    os.chdir('..')

# Run xTB VIP single-point job (for multiprocessing execution)
#Takes 1 argument: line with xyzfilename and the xtb options
#PROBLEM: IPEA option has convergence issues for occasional snapshots.
#DISCOURAGED
def run_xtb_VIP_multiproc(line):
    basename=line.split()[0].split('.')[0]
    xyzfile=line.split()[0]
    #Create dir for snapshot
    os.mkdir(basename)
    os.chdir(basename)
    shutil.copyfile('../'+xyzfile, './'+xyzfile)
    chargeseg1=line.split()[1]
    chargeseg2=line.split()[2]
    uhfseg1=line.split()[3]
    uhfseg2=line.split()[4]
    ipseg=line.split()[5]
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([settings_solvation.xtbdir + '/xtb', xyzfile, chargeseg1, chargeseg2, uhfseg1, uhfseg2, ipseg], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
    os.chdir('..')

#Using IPEA-xtB method for IP calculations
def run_xtb_VIP(xyzfile, charge, mult):
    basename = xyzfile.split('.')[0]
    uhf=mult-1
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([settings_solvation.xtbdir + '/xtb', basename+'.xyz', '--vip', '--chrg', str(charge), '--uhf', str(uhf) ], check=True, stdout=ofile, universal_newlines=True)


#def run_inputfile_xtb(xyzfile, xtbmethod, chargeA, multA, chargeB, multB):
#    blankline()
#    print("Launching xTB job in serial")
#    print("Number of CPU cores: ", mp.cpu_count())
#    print("XYZ file:", xyzfiles)
#    run_xTB_SP
#    print("Calculations is done")

#TODO: Deal with pcharge pointcharge file.
def run_inputfiles_in_parallel_xtb(xyzfiles, xtbmethod, chargeA, multA, chargeB, multB):
    import multiprocessing as mp
    blankline()
    NumCoresToUse=settings_solvation.NumCores
    print("Launching xTB jobs in parallel")
    print("OMP_NUM_THREADS:", os.environ['OMP_NUM_THREADS'])
    xTBCoresRestriction = False
    if xTBCoresRestriction==True:
        NumCoresToUse=8
        print("xTBCoresRestriction Active!")
        print("Restricting multiprocessing cores to:", NumCoresToUse)
    print("Number of CPU cores: ", NumCoresToUse)
    blankline()
    print("Number of XYZ files:", len(xyzfiles))
    print("Running snapshots in parallel")
    #Create lines to serve as arguments to run_xtb_SP_multiproc
    inputlines=[]
    uhfA=multA-1
    uhfB=multB-1
    pool = mp.Pool(NumCoresToUse)
    if 'GFN' in xtbmethod.upper():
        print("GFN xTB flag")
        print("Will do 2 calculations for State A and State B")
        print("StateA: Charge: {} Mult: {}".format(chargeA, multA))
        print("StateB: Charge: {} Mult: {}".format(chargeB, multB))
        if 'GFN2' in xtbmethod.upper():
            xtbflag=2
        elif 'GFN1' in xtbmethod.upper():
            xtbflag=1
        elif 'GFN0' in xtbmethod.upper():
            xtbflag=0
        for xyzfile in xyzfiles:
            #Passing line with all info to run_gfnxtb_SPVIE_multiproc. Charge/Mult separated in function
            line="{} {} {} {} {} {}".format(xyzfile, xtbflag, chargeA, uhfA, chargeB, uhfB)
            inputlines.append(line)
        results = pool.map(run_gfnxtb_SPVIE_multiproc, [l for l in inputlines])
    elif 'VIP' or 'VEA' or 'VIPEA' in xtbmethod.upper():
        print("IP/EA option. Will do VIP/VEA calculation")
        if 'VIP' in xtbmethod.upper():
            print("VIP xtB flag!")
            xtbflag='--vip'
        elif 'VEA' in xtbmethod.upper():
            print("VEA xtB flag!")
            xtbflag = '--vea'
        elif 'VIPEA' in xtbmethod.upper():
            print("VIPEA xtB flag!")
            xtbflag = '--vipea'
        for xyzfile in xyzfiles:
            line = "{} --chrg {} --uhf {} {}".format(xyzfile, chargeA, uhfA, xtbflag )
            inputlines.append(line)

        results = pool.map(run_xtb_VIP_multiproc, [l for l in inputlines])

    pool.close()
    print("xTB Calculations are done")


#Create xTB pointcharge file based on provided list of elems and coords (MM region elems and coords) and charges for solvent unit.
#Assuming elems and coords list are in regular order, e.g. for TIP3P waters: O H H O H H etc.
#Using Bohrs for xTB. Will be renamed to pcharge when copied to dir.
#Hardness parameter removes the damping used by xTB.
def create_xtb_pcfile_solvent(name,elems,coords,solventunitcharges,bulkcorr=False):
    #Creating list of pointcharges based on solventunitcharges and number of elements provided
    #Modifying
    pchargelist=solventunitcharges*int(len(elems)/len(solventunitcharges))
    bohr2ang=constants.bohr2ang
    hardness=200
    #https://xtb-docs.readthedocs.io/en/latest/pcem.html
    with open(name+'.pc', 'w') as pcfile:
        pcfile.write(str(len(elems))+'\n')
        for p,c in zip(pchargelist,coords):
            line = "{} {} {} {} {}".format(p, c[0]/bohr2ang, c[1]/bohr2ang, c[2]/bohr2ang, hardness)
            pcfile.write(line+'\n')

#General xtb pointchargefile creation
#Using ORCA-style format: pc-coords in Å
def create_xtb_pcfile_general(coords,pchargelist):
    #Creating list of pointcharges based on solventunitcharges and number of elements provided
    bohr2ang=constants.bohr2ang
    hardness=1000
    #https://xtb-docs.readthedocs.io/en/latest/pcem.html
    with open('pcharge', 'w') as pcfile:
        pcfile.write(str(len(pchargelist))+'\n')
        for p,c in zip(pchargelist,coords):
            line = "{} {} {} {} {}".format(p, c[0], c[1], c[2], hardness)
            pcfile.write(line+'\n')


#Grab pointcharge gradient (Eh/Bohr) from xtb pcgrad file
def xtbpcgradientgrab(numatoms):
    gradient = np.zeros((numatoms, 3))
    with open('pcgrad') as pgradfile:
        for count,line in enumerate(pgradfile):
            val_x=float(line.split()[0])
            val_y = float(line.split()[1])
            val_z = float(line.split()[2])
            gradient[count-1] = [val_x,val_y,val_z]
    return gradient