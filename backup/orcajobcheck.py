#!/bin/env python3
debug="no"

#Updates:

#March 2019. Small things
# Fixed issue with copy files statements at bottom of outpufiles on MPI clusters 

#Jan 16 2019
#Small things


#26 nov
#Fixing some small things
# Added printing of UHF if present and S**2 value for UHF SP jobs. 
# Added printing of HOMO-LUMO gap. Currently only works for UHF jobs and only prints HOMO-LUMO for alpha-alpha.
#Need to do RHF soon. Then HOMO-LUMO for beta-beta and also alpha-beta

#4 nov
# More support for CASSCF failures. Also now allows CASSCF gradient printing (using grep)
# Need to think about better error message printing at some point. This last lines printing is stupid

#2 nov
# Fixed some small things. Multiple imaginary modes warning. Fixed by if/elif modification.
# Added recognition of SCF not converging message in ORCA 4 and fixed extrapolation message.
#Added new errormessage variable. Maybe used more in printing ???

# 9 oct
#Addex extrapolation recognition

# 7 oct 2016
# Added plotting feature via matplotlib 

# 6 oct 2016
# Did some changes for scan. Was not working for Angles and Dihedrals. Also fixed for scanjob in 1st scan step

#3sept 2016
#Fixed so that it works on files with full or relative paths.
#Also directories with full path

#Also fixed so that for an opt job that crashed one can still print last geometry

#Also fixed so that freq keyword is found in caseinputline if last keyword

#Now printing yellow numnoise warning if lowest mode is negative (but not flagged as imaginary by ORCA).

#Also added scan functionality. Not quite as flexible as zsh version but already useful.

#Also added bs-flipspin. skipped scfcycles for high-spin. only tested for converged calculation.

#10 may
#fixing "Cannot open input file:" stuff
#Also fixing various HF freq crashes behaviour
#freqsp behaviour looks complicated now. Both in sp section and freqsection

#14 apr
# Fixed wrong optenergy. Added finaloptenergy to make sure the final opt energy is the correct one.
#Fixed unset problem for integrated num el.

#12 apr.
#Fixed unset optenergy thing in running opt jobs. Fixed wrong optcycle printed

#10 Apr
#Added if statement for early crash (when ORCA complains about COORDS stuff )
# Also fixed CASSCF convergence issue

#13 feb
#encoding problem when bad characters like diamond-question mark character
# Fixed by errors=ignore in with open lines

# 26 jan
# Changed buffer size in reverse-read. Better

#14 jan
#Added -short printing
#Modification to pausecount. Changed 200 to 20. Safer. Might slow things down for large printout. Need to look more into formula

#12 jan
# Switched reverse code. Is much more efficient now.

# 10th or 11th:
# Made freq code more efficient by skipping lines to read cleverly

#31dec-1jan
#
#Replaced almost all re.search with 'if string in line'
#Much faster as shown in: /data/users/rbjorns/test-orcajobscript/jess-file
#Not as fast as grep but close


#PROBLEMS:

# Allow noiter in short mode. See pCCSD/2a calcs in EA

# Looks like first forward read on Ni_0 jess file is surprisingly slow. 0.2 - 0.4 sec for 1718 lines.

#Figured out that reversed_read is surprisingly slow
# Switched code. Better now but reverse read is still 2x times slower than regular for line read
#Teseting forward and reverse read on femoco file and last 15k lines of femoco file.
#Here: /data/users/rbjorns/test-orcajobscript/femoco-porca-tests
#Problem with original reverse-comparison.py. Was not working properly
#Need to redo tests

#This is reason for slow freq read. Probably have optimized freq section now with if statem.

#Ni_0 is main bad guy
# Tried to speed up opt. Not a lot of improvement
# Main culprit though seems to be first-read complete (maybe 0.2 too costly) and FREQ. Taking 0.4 which is way too much.
#
# van-opt-running.mpi6.out is tricky

# Big file: ~femoco-E4-dancestructure-mult2.mpi8.out
# jess file /data/users/rbjorns/test-orcajobscript/jess-file
# PYthon script is slower than orcajobcheck.sh

#Need to make big freq section better. Like in nickel jess file
#Nickel jess file is maybe only file that is quicker with orcajobecheks.h zsh script

scriptversion=3.0

# All dependencies (also matplotlib and subprocess for special cases)
import numpy
import math
import sys
import os
import re
import time
start_time = time.time()

# Reverse read function.
# Default buffersize was 4096. 20480 works better
def reverse_lines(filename, BUFSIZE=20480):
    #f = open(filename, "r")
    filename.seek(0, 2)
    p = filename.tell()
    remainder = ""
    while True:
        sz = min(BUFSIZE, p)
        p -= sz
        filename.seek(p)
        buf = filename.read(sz) + remainder
        if '\n' not in buf:
            remainder = buf
        else:
            i = buf.index('\n')
            for L in buf[i+1:].split("\n")[::-1]:
                yield L
            remainder = buf[:i]
        if p == 0:
            break
    yield remainder


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Conversion factors
#hartree to kcal/mol
harkcal = 627.50946900


##########################################
# Getting user arguments first
#########################################
# Read in filename or dir as argument
filelist=[]
try:
    if sys.argv[1]==".":
        dirmode="on"
        for file in sorted(os.listdir(sys.argv[1])):
            if file.endswith(".out"):
                filelist.append(file)
    #If using full or relative path for file or dir
    elif "/" in sys.argv[1]:
        #Checking if a single file with path
        if ".out" in sys.argv[1]:
            dirmode="off"
            filename=sys.argv[1]
            filelist.append(filename)
            print("filename is", filename)
        #Or a directory
        else:
            dirmode="on"
            for file in os.listdir(sys.argv[1]):
                if file.endswith(".out"):
                    filelist.append(sys.argv[1]+"/"+file)
    #If parent folder
    elif sys.argv[1]=="..":
        dirmode="on"
        for file in sorted(os.listdir(sys.argv[1])):
            if file.endswith(".out"):
                filelist.append(sys.argv[1]+"/"+file)
    else:
        dirmode="off"
        filename=sys.argv[1]
        if '.out' in filename == None :
            print("Not an ORCA outputfile?")
            exit()
        filelist.append(filename)
except IndexError:
    print(bcolors.OKBLUE +"ORCA JobCheck Utility version", scriptversion, "(Python version)", bcolors.ENDC)
    print("---------------------------------")
    print("Script usage:")
    print("On single file: porcajobcheck.sh orcafile.out")
    print("On directory: porcajobcheck.sh .")
    print("Short printing mode: porcajobcheck.sh . -short") 
    quit()

shortmode="unset"
try:
    if sys.argv[2]=="-short" :
        shortmode="yes"
except IndexError:
    pass    

####################################################################
#Here begins jobspecific section
####################################################################
if debug=="yes":
    if dirmode=="on":
        print("filelist is", filelist)
        print("Exiting dirmode")

#print('Here1. Script took %s' % (time.time() - start_time))
for filename in filelist:
    if debug=="yes":
        print("filename is", filename)
    earlycrash="unset"
    xyzfileerror="unset"
    runcomplete="unset"
    orcacrash="unset"
    conbf="unset"
    cpscferror="unset"
    scferrorgeneral="unset"
    scfstillrunning="unset"
    scfalmostconv="unset"
    parproc="unset"
    jobtype="unset"
    scfmethod="unset"
    dft="unset"
    optrunconverged="unset"
    linearcheck="unset"
    optenergy="unset"
    finaltopenergy="unset"
    opterror="unset"
    optnotconv="unset"
    optcycle="unset"
    intelectrons="unset"
    freqjob="unset"
    freqsection="unset"
    freqsearch="unset"
    hessfail="unset"
    optjob="unset"
    scfconv="unset"
    noiter="unset"
    moread="unset"
    autostart="unset"
    postHFmethod="unset"
    postHF="unset"
    frozel="unset"
    correl="unset"
    refenergy="unset"
    correnergy="unset"
    caseinputline="unset"
    inputline="unset"
    nofrozencore="unset"
    casscf="unset"
    lastmacroiter="unset"
    casscfconv="unset"
    nevpt2correnergy="unset"
    version="unset"
    semiempirical="unset"
    functional="unknown"
    engrad="unset"
    freqinrun="unset"
    charge="unset"
    endofinput="unset"
    numatoms="unset"
    spin="unset"
    actualelec="unset"
    temprcount="unset"
    diagerror="unset"
    brokensym="unset"
    flipspin="unset"
    errormult="unset"
    scfcycleslist=[]
    #errormessage="unset"
    errormessage=[]
    s2value="unset"
    ideals2value="unset"
    scftype="unset"
    occorbsgrab="unset"
    virtorbsgrab="unset"
    orbs=[]
    lastvirt_a="unset"
    lastocc_a="unset"
    gap_a="unset"
#spsection used to stop when SP output is done from bottom
    spsection="unset"
#optsection used to stop when output found
    optsection="unset"
    finalgeo="unset"
    coord="unset"
    geomconvtable="unset"
    geomconvgrab="unset"
    lastgeomark="unset"
    numatomcountstart="unset"
    scantest="unset"
    findenergy="unset"
    scanenergies=[]
    scanoptcycle=[]
    lastscanoptcycle="unset"
    lastenergy="unset"
    allscanstepnums=[]
    bsenergies=[]
    rctype="unset"
    scanatomA="unset"
    scanatomB="unset"
    scanatomC="unset"
    scanatomD="unset"
    extrapolate="unset"
    extrapscfenergy="unset"
    extrapcorrenergy="unset"
    basissets=[]
    newjob="unset"
    finalsingleline="unset"
#
    imaginmodes=[]
    atomcoord=[]
    optgeo=[]
    lastgeo=[]
    inputgeo=[]
    geomconv=[]
    count=0
    rmsgradlist=[]

#funcional list
    funclist=["b3lyp", "bp", "pbe", "tpss", "tpssh", "pbe0", "bp86", "blyp", "LDA", "BHLYP", "B2PLYP", "cam-b3lyp", "m06-2x", "pw6b95"]

    grabsimpleinput=False
    case=''
#Reading first lines of file normally until "Nuclear Repulsion      ENuc"
#utf-8 has been used. errors=ignore seems to work for encoding issues
    #with open(filename, encoding='utf-8') as bfile:
    with open(filename, errors='ignore') as bfile:
        for line in bfile:
            count=count+1
            if version =="unset":
                if 'Program Version' in line:
                    version=line.split()[2]
            if caseinputline=="unset" and version !="unset":
                if 'WARNING: The NDO methods cannot have' in line:
                    semiempirical="yes"
            #Grabbing simpleinputline
            if caseinputline=="unset" and version !="unset":
                if grabsimpleinput == True:
                    if '!' in line and not '#' in line:
                        var1=line.lower().split()[2:]
                        var2= ' '.join(var1)
                        var2=var2.replace("!","")
                        case=case+var2
                else:
                    if grabsimpleinput==False and 'INPUT FILE' in line:
                        grabsimpleinput=True

#Here checking for stuff in listed inputfile until 'END OF INPUT'
            if 'INPUT LINE' != "unset" and endofinput=="unset":
                if 'END OF INPUT' in line:
                    endofinput="yes"
                    grabsimpleinput=False
                    caseinputline=case
                    #Now going through simple-input keywords here to determine job-type
                    if caseinputline!="unset":
                        #Checking for functional
                        for i in funclist:
                            if i in caseinputline:
                                functional=i
                            #Checking for noiter
                            if "noiter" in caseinputline:
                                noiter="yes"
                            #Checking for moread
                            if "moread" in caseinputline:
                                moread="yes"
                            #Checking for nofrozencore
                            if "nofrozencore" in caseinputline:
                                nofrozencore="yes"
                            #Checking for post-HF keywords
                            #print("caseinputline is", caseinputline)
                            if "ccsd" in caseinputline or "mp2" in caseinputline or "qcisd" in caseinputline:
                                postHF="yes"
                                if "ccsd" in caseinputline:
                                    postHFmethod="CC"
                                if "qcisd" in caseinputline:
                                    postHFmethod="QCI"
                                if "mp2" in caseinputline:
                                    postHFmethod="MP2"
                            if "extrapolate" in caseinputline:
                                extrapolate="yes"
                            # Checking for opt or freq here
                            if " optts " in caseinputline or " optts\n" in caseinputline:
                                jobtype="optts"
                                if " freq " in caseinputline or " numfreq " in caseinputline or " freq" in caseinputline or " numfreq" in caseinputline:
                                    jobtype="opttsfreq"; freqsection="notyetdone"
                            elif " opt" in caseinputline or " tightopt" in caseinputline or "copt" in caseinputline:
                                jobtype="opt"
                                #print("jobtype is", jobtype)
                                if " freq" in caseinputline or " numfreq" in caseinputline:
                                    jobtype="optfreq"; freqsection="notyetdone"
                            elif " md " in caseinputline:
                                jobtype="md"
                            elif " freq " in caseinputline:
                                jobtype="freqsp"; freqsection="notyetdone"
                            elif " engrad " in caseinputline:
                                jobtype="sp"
                                engrad="yes"
                            else:
                                jobtype="sp"

                #Going through block input
                if casscf=="unset":
                    if '%casscf' in line:
                        casscf="yes"
                if "$new_job" in line and newjob == "unset":
                    newjob="yes"
                    print(bcolors.WARNING +"New_job feature detected! Script will probably not deal with multi-job outputfiles correctly! Complain to RB.",bcolors.ENDC)
                #Checking if brokensym job
                if 'flipspin' in line.lower():
                    flipspin="yes"
                if 'finalms' in line.lower():
                    bsms=float(line.split()[-1])
                if 'brokensym' in line.lower():
                    brokensym="yes"
            if endofinput=="yes":
                if scantest=="unset":
                    temp=next(bfile);temp=next(bfile);temp=next(bfile);temp=next(bfile);
                    if "Relaxed" in temp:
                        scantest="over"
                        jobtype="scan"
                        temp=next(bfile);temp=next(bfile);temp=next(bfile)
                        rctype=temp.split()[0]
                        scansteps=temp.split()[-1]
                        if rctype=="Bond":
                            scanatomA=temp.split()[2][:-1]
                            scanatomB=temp.split()[3][:-2]
                        elif rctype=="Angle":
                            scanatomA=temp.split()[2][:-1]
                            scanatomB=temp.split()[3][:-1]
                            scanatomC=temp.split()[4][:-2]
                        elif rctype=="Dihedral":
                            scanatomA=temp.split()[2][:-1]
                            scanatomB=temp.split()[3][:-1]
                            scanatomC=temp.split()[4][:-1]
                            scanatomD=temp.split()[5][:-2]
                        scanvalA=temp.split()[-6]
                        scanvalB=temp.split()[-4]
                        scanchange=(float(scanvalB) - float(scanvalA)) / (float(scansteps) - 1)
                    else:
                        scantest="over"
                if autostart=="unset" and conbf=="unset":
                    if 'Checking for AutoStart:' in line:
                        autostart="yes"
                if conbf=="unset" and semiempirical=="unset":
                    if '# of contracted basis functions' in line:
                        conbf=line.split()[-1]
                    if 'Number of basis functions' in line:
                        conbf=line.split()[5]
                    if casscf=="yes":
                        if 'Number of basis functions' in line:
                            conbf=line.split()[5]
                #    #Counting atoms and getting inputgeo
                if numatomcountstart=="unset" and scantest=="over":
                    if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                        numatomcountstart="on"
                        numatomcount=0
                if numatomcountstart=="on":
                    numatomcount += 1
                    inputgeo.append(line.strip())
                if 'CARTESIAN COORDINATES (A.U.)' in line and numatomcountstart=="on" and numatoms=="unset":
                    numatomcountstart="off"
                    inputgeo.pop(0);inputgeo.pop(0);inputgeo.pop();inputgeo.pop();inputgeo.pop()
                    numatoms=numatomcount-5
                    if numatoms > 3:
                        pausecount=int((numatoms*3)-6+10+((numatoms*3)/6)*(numatoms*3)+20+numatoms*3-20)
                        if debug=="yes":
                            print("pausecount is", pausecount)
                    else:
                        pausecount=33
            #Finding parallel and SCF output after basis output
            if conbf !="unset" or semiempirical=="yes":
                if parproc =="unset":
                    if 'parallel MPI-processes' in line:
                        parproc=line.split()[4]
            # SCF method #NOTE. MULTIPLE OCCURENCE
                if 'SCF SETTINGS' in line:
                        temp=next(bfile);temp=next(bfile);temp=next(bfile)
                        scftemp=temp.split()
                        if scftemp[0]=="ZDO-Hamiltonian":
                            semiempirical="yes"
                            scfmethod=scftemp[3]
                        else:
                            scfmethod=scftemp[4]
                        if 'DFT' in scfmethod:
                            dft="yes"
                 #Charge and spin
                if scfmethod!="unset":
                    if ' Hartree-Fock type      HFTyp' in line:
                        scftype=line.split()[4]
                    if ' Total Charge           Charge' in line:
                        charge=line.split()[4]
                    if charge !="unset":
                        if ' Multiplicity           Mult            ....' in line:
                            mult=int(line.split()[3])
                            spin=(mult-1)/2.0
                        if 'Number of Electrons    NEL             ....' in line:
                            actualelec=line.split()[5]
                        if 'Nuclear Repulsion      ENuc' in line:
                            nucrepuls=line.split()[5]
                            break


#Now we have read through the first few hundred lines (until nuc repulsion) of file that determines the jobtype and basic molecule info.
#If the ORCA job died prematurely (before nuc repulsion was printed) then that is fine (reversed read below will find the error)
    if debug=="yes":
        print('First read complete. Read', count, 'lines. Script took %s' % (time.time() - start_time))
        print("Last line was", line)
#Now Reading lines of file backwards. This should detect errors immediately etc. or find where job died.
    rcount=0
    with open(filename, errors='ignore') as file:
        for line in reverse_lines(file):
            rcount=rcount+1
            #Error messages. Only last 50 lines checked
            if rcount < 60:
                #If copy lines from suborca present at bottom of outputfile. Do not count
                if '->' in line:
                    rcount=rcount-1
                if 'ERROR: Unknown identifier' in line:
                    earlycrash="yes"; orcacrash="yes"; errormessage.append(line)
                if 'Error (ORCA/TRAFO/RI-GIAO):' in line:
                    orcacrash="yes"; errormessage.append(line)
                if 'Zero distance between atoms' in line:
                    orcacrash="yes";earlycrash="yes"
                if 'Cannot open input file:' in line:
                    orcacrash="yes";earlycrash="yes"
                if 'You must have a' in line:
                    orcacrash="yes";earlycrash="yes"
                if 'INPUT ERROR' in line:
                    inputerror="yes";orcacrash="yes";earlycrash="yes"
                if 'ERROR CODE RETURNED FROM CP-SCF PROGRAM' in line:
                    cpscferror="yes";orcacrash="yes"
                if 'ABORTING THE RUN' in line:
                    abortcode="yes";orcacrash="yes";earlycrash="yes"
                if 'Invalid assignment in' in line:
                    abortcode="yes";orcacrash="yes";earlycrash="yes"
                if 'Aborting the run' in line:
                    abortcode2="yes";orcacrash="yes"
                if 'Skipping actual calculation' in line:
                    abortcode3="yes";orcacrash="yes";earlycrash="yes"
                if 'Error : multiplicity' in line:
                    errormult="yes";orcacrash="yes";earlycrash="yes"
                if 'Unrecognized symbol in' in line:
                    orcacrash="yes";earlycrash="yes"
                    errormessage.append(line)
                if 'Basis not recognized' in line:
                    orcacrash="yes";earlycrash="yes"
                    errormessage.append(line)
                if 'Requested ECP not available' in line:
                    orcacrash="yes";earlycrash="yes"
                    errormessage.append(line)
                if 'Element name/number, dummy atom or point charge expected in COORDS' in line:
                    coorderror="yes";orcacrash="yes";earlycrash="yes"
                if 'FATAL ERROR ENCOUNTERED' in line:
                    fatalerrorcode="yes";orcacrash="yes";earlycrash="yes"
                if 'There is no basis function on atom' in line:
                    basiserror="yes";orcacrash="yes";earlycrash="yes"
                if 'ORCA finished by error termination' in line:
                    errortermin="yes";orcacrash="yes";errormessage.append(line)
                if 'An error has occured in the SCF module' in line:
                    scferrorgeneral="yes";orcacrash="yes"
                if 'An error has occured in the CASSCF module' in line:
                    casscferrorgeneral="yes";orcacrash="yes"
                    errormessage.append("CASSCF module failed")
                if 'ORCA finished by error termination in CASSCF' in line:
                    casscferrorgeneral="yes";orcacrash="yes"
                    errormessage.append("CASSCF module failed")
                if 'mpirun has exited due to process' in line:
                    mpiruncode="yes";orcacrash="yes"
                if 'mpirun noticed that process rank 0' in line:
                    mpiruncode2="yes";orcacrash="yes"
                if 'Job terminated from outer' in line:
                    jobtermin="yes";orcacrash="yes"
                if 'CANNOT OPEN FILE' in line:
                    cannotopenfile="yes";orcacrash="yes";earlycrash="yes"
                if 'Error: XYZ File reading requested' in line:
                    xyzfileerror="yes";orcacrash="yes";earlycrash="yes"
                    errormessage.append("XYZ file error problem")
                if '!!!               Filename:' in line:
                    xyzfileerror="yes";orcacrash="yes";earlycrash="yes"
                    errormessage.append("XYZ file error problem")
                if 'Unknown identifier in' in line:
                    unknownidentifier="yes";orcacrash="yes";earlycrash="yes"
                if 'ERROR: expect a' in line:
                    commanderror="yes";orcacrash="yes";earlycrash="yes"
                if 'Error: Number of NGauss expected' in line:
                    orcacrash="yes";earlycrash="yes";errormessage.append(line)
                if 'ERROR: found a coordinate defintion' in line:
                    coordinateerror="yes";orcacrash="yes";earlycrash="yes"
                if 'Diagonalization failure because of NANs in input matrix' in line:
                    diagerror="yes"; orcacrash="yes"
                if 'ERROR       : GSTEP Program returns an error' in line:
                    gsteperror="yes";orcacrash="yes"
                if 'ORCA TERMINATED NORMALLY' in line:
                    runcomplete="yes"
                if 'TOTAL RUN TIME:' in line:
                    runtime=line.split()[3:]
                if 'This wavefunction IS NOT CONVERGED!' in line:
                    scfconv="no";orcacrash="yes"
                    errormessage.append("SCF did not converge")
                if 'The optimization did not converge but reach' in line:
                    optnotconv="yes"
                if 'Error (ORCA_SCFGRAD): cannot find the xc-energy file:' in line:
                    scfconv="no";orcacrash="yes"
                    errormessage.append("SCF did not converge")
                if 'Error: XYZ File reading requested but the structur' in line:
                    xyzfileerror="yes";orcacrash="yes";earlycrash="yes"
                    errormessage.append("XYZ file error problem")
# Here need to add some kind of break so that we stop if errors above are encountered
            #if orcacrash="yes":
            #    break
            if rcount==50:
                if debug=="yes":
                    print('Here,rcount50. Script took %s' % (time.time() - start_time))
#Relaxed surface scan section
            if jobtype=="scan":
                if 'RELAXED SURFACE SCAN STEP' in line:
                    scanstep=line.split()[5]
                    allscanstepnums.append(scanstep)
                if '*                GEOMETRY OPTIMIZATION CYCLE' in line:
                    scanoptcycle.append(line.split()[4])
                if lastgeomark=="unset":
                    if 'CARTESIAN COORDINATES (A.U.)' in line and lastgeomark=="unset" :
                        coord="active"
                        if debug=="yes":
                            print('Here. Starting coord grab  Read', rcount, 'lines. %s' % (time.time() - start_time))
                    if coord=="active":
                        lastgeo.append(line.strip())
                        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                            coord="inactive"
                            lastgeomark="done"
                            lastgeo.pop(0);lastgeo.pop(0);lastgeo.pop(0);lastgeo.pop();lastgeo.pop()
                if lastenergy =="unset":
                    if 'FINAL SINGLE POINT ENERGY' in line:
                        lastenergy=line.split()[4]
                if 'OPTIMIZATION RUN DONE' in line:
                    findenergy="yes"
                if findenergy=="yes":
                    if 'FINAL SINGLE POINT ENERGY' in line:
                        scanenergies.append(line.split()[4])
                        findenergy="no"
#Single-point section
            if jobtype == "sp" or jobtype == "freqsp":
                #If brokensymm job
                if brokensym=="yes" or flipspin=="yes":
                    sdfds="sdf"
                    if 'E(BrokenSym)' in line:
                        bsenergy=line.split()[2]
                    if 'E(High-Spin)      =' in line:
                        hsenergy=line.split()[2]
                if 'DFT' in scfmethod and intelectrons=="unset":
                    if 'N(Total)' in line:
                        intelectrons=line.split()[2]
                if runcomplete=="yes":
                    sdf="ds"
                    #Grabbing HOMO-LUMO gap
                    #if 'NO   OCC          E(Eh)            E(eV)' in line:
                    if scftype=="UHF":
                        if 'SPIN DOWN ORBITALS' in line:
                            occorbsgrab="yes"
                    if scftype=="RHF":
                        if '                    * MULLIKEN POPULATION ANALYSIS *' in line:
                            occorbsgrab="yes"
                    if occorbsgrab=="yes":
                        #print("line is", line); print("line.split is", len(line.split()))
                        if len(line) > 1 and len(line.split())==4 and '*' not in line:
                            endocc=line.split()[1]
                            if endocc == "0.0000": 
                                #print(line)
                                lastvirt_a=float(line.split()[-1])
                                #print("last virt energy is", lastvirt)
                                #occorbsgrab="no"
                                #virtorbsgrab="yes"
                            elif endocc == "1.0000":
                                lastocc_a=float(line.split()[-1])
                                gap_a=lastvirt_a-lastocc_a
                                occorbsgrab="no"
                                #orbs.append(float(line.split()[-1]))
                            elif endocc == "2.0000":
                                lastocc_a=float(line.split()[-1])
                                gap_a=lastvirt_a-lastocc_a
                                occorbsgrab="no"
                                #orbs.append(float(line.split()[-1]))
                        if 'SPIN UP ORBITALS' in line:
                            occorbsgrab=="no"
                        if '  NO   OCC          E(Eh)            E(eV)' in line:
                            occorbsgrab=="no"
                        
                    #if virtorbsgrab=="yes":
                    #    if line == '\n':
                    #        virtorbsgrab="no"
                    #    virtbands.append(float(line.split()[3]))
                    #    endvirt=line.split()[1]
                    #print("orbs is", orbs)
                    #Grabbing S**2 value
                    if 'Expectation value of' in line:
                        s2value=line.split()[-1]
                    if 'Ideal value' in line:
                        ideals2value=line.split()[-1]
                    if postHF=="yes":
                        if extrapolate=="yes":
                            if "Extrapolated CBS SCF energy" in line:
                                extrapscfenergy=line.split()[-2]
                            if "Extrapolated CBS correlation energy" in line:
                                extrapcorrenergy=line.split()[-2]
                            if "Cardinal #:" in line:
                                basissets.append(line.split()[-1])
                            elif 'SCF energy with basis' in line:
                                basissets.append(line.split()[4][:-1])
                        if postHFmethod=="CC":
                            if 'Number of correlated electrons' in line:
                                correl=line.split()[5]
                                frozel=int(actualelec)-int(correl)
                                if noiter=="yes":
                                    break
                            if 'Reference energy' in line:
                                refenergy=line.split()[3]
                            if 'Final correlation energy' in line:
                                correnergy=line.split()[4]
                            if 'E(CORR)(total)' in line:
                                correnergy=line.split()[2]
                            if 'E(CORR)(corrected)' in line:
                                correnergy=line.split()[2]
                            if 'E(CORR)' in line:
                                correnergy=line.split()[2]
                        if postHFmethod=="QCI":
                            if 'Number of correlated electrons' in line:
                                correl=line.split()[5]
                                frozel=int(actualelec)-int(correl)
                                if noiter=="yes":
                                    break
                            if 'Reference energy' in line:
                                refenergy=line.split()[3]
                            if 'E(CORR)(corrected)' in line:
                                correnergy=line.split()[2]
                            if 'E(CORR)' in line:
                                correnergy=line.split()[2]
                        if postHFmethod=="MP2":
                            if 'CORRELATION ENERGY' in line:
                                correnergy=line.split()[3]
                            if 'chemical core electrons' in line:
                                frozel=line.split()[1].lstrip('NCore=')
                            if nofrozencore=="yes":
                                frozel="0"
                    if 'FINAL SINGLE POINT ENERGY' in line:
                        scfenergy=line.split()[4]
                        finalsingleline=True
                        if postHF=="unset" and noiter=="yes":
                            break
                    if 'SCF CONVERGED AFTER' in line:
                        scfconv="yes"
                        scfcycles=line.split()[4]
                        #if brokensym=="yes" or flipspin=="yes":
                        #    scfcycleslist.append(scfcycles)
                        #    print("scfcycleslist is", scfcycleslist)
                        spsection="done"
                    if 'The wavefunction IS NOT YET CONVERGED! It shows however signs of' in line:
                        scfalmostconv="yes"
                    if 'SCF NOT CONVERGED AFTER' in line:
                        scfconv="no"
                        unfinscfcycles=line.split()[5]
                        spsection="done"
                #If runcomplete not true. Like if SCF is still running or if FREQ-sp job and something happened during freq run.
                else:
                    if 'FINAL SINGLE POINT ENERGY' in line:
                        scfenergy=line.split()[4]
                        if postHF=="unset" and noiter=="yes":
                            break
                    if 'SCF CONVERGED AFTER' in line:
                        scfconv="yes"
                        scfcycles=line.split()[4]
                        spsection="done"
                    if 'Total Energy       :' in line:
                        purescfenergy=line.split()[3]
                    if 'The wavefunction IS NOT YET CONVERGED! It shows however signs of' in line:
                        scfalmostconv="yes"
                    if 'SCF NOT CONVERGED AFTER' in line:
                        scfconv="no"
                        unfinscfcycles=line.split()[5]
                        spsection="done"
                    if 'ERROR (ORCA_CASSCF): Convergence Failure.' in line:
                        scfconv="no"; spsection="done";casscfconv="no"
                    if 'Warning: Active Space composition changed by more than' in line:
                        scfconv="no"; spsection="done";casscfconv="no"

                    if 'SCF ITERATIONS' in line:
                        if scfconv=="unset":
                            scfstillrunning="yes"
                        spsection="done"
                ########
                if casscf=="yes":
                    if 'Total Energy Correction :' in line:
                        nevpt2correnergy=line.split()[6]
                    if lastmacroiter=="unset" and 'MACRO-ITERATION' in line:
                        lastmacroiter=line.split()[1].rstrip(':')
                    if 'THE CAS-SCF GRADIENT HAS CONVERGED' in line:
                        casscfconv="yes"
                    if 'THE CAS-SCF ENERGY   HAS CONVERGED' in line:
                        casscfconv="yes"
                    if 'CAS-SCF ITERATIONS' in line:
                        break
                #if 'SCF ITERATIONS' in line:
                #    scfstillrunning="yes"
                #    spsection="done"
                if spsection == "done":
                    break
#FREQSECTION
            #print("freqsection is", freqsection)
        #Will analyze the last freq output encountered in outputfile. But will only print it if optjob converged
            if freqsection=="notyetdone":
                if runcomplete=="yes":
                    freqjob="yes"
                    #print("freqjob is", freqjob)
                    #Once Temperature has been found, try to skip all Normal mode output without re.search lines.
                    #Until NORMAL MODES
                    if freqsearch!="on":
                        if 'Temperature         ...' in line:
                            temperature=line.split()[2]
                            temprcount=rcount
                            freqsearch="on"
                            if debug=="yes":
                                print('Here. Temperature line. rcount is:', rcount, 'Script took %s' % (time.time() - start_time))
                    if temprcount!="unset" and rcount-temprcount > pausecount :
                        #print("rcount is", rcount); print("line is", line)
                        if 'imaginary' in line:
                            freqimagtest="yes"
                            imaginmodes.append(line.split()[1])
                        if linearcheck=="yes":
                            if '5:' in line:
                                lowestvib=line.split()[1]
                        else:
                            if '6:' in line:
                                lowestvib=line.split()[1]
                            #Here we have reached the beginning of FREQ section, from bottom, hence freqsection is done
                        if 'VIBRATIONAL FREQUENCIES' in line:
                            freqsearch="off"
                            if debug=="yes":
                                print('Here. FREQdone. rcount is:', rcount, 'Script took %s' % (time.time() - start_time))
                            freqsection="done"
            #Therm stuff
                    if freqsearch!="on":
                        #print('bla Script took %s' % (time.time() - start_time))
                        if 'The molecule is recognized as being linear' in line:
                            linearcheck="yes"
                        if 'G-E(el)' in line:
                            if "nan" in line.split()[2]:
                                hessfail=True
                                freqsection="done"
                            else:
                                gthermcorr=float(line.split()[2])
                        if 'Zero point' in line:
                            zeropointcorr=float(line.split()[4])
                            enthalpycorr=totthermcorr+zeropointcorr+enthalpyterm
                        if 'Final entropy term' in line:
                            entropycorr=float(line.split()[4])
                        if 'Total thermal correction' in line:
                            totthermcorr=float(line.split()[3])
                        if 'Thermal Enthalpy correction' in line:
                            enthalpyterm=float(line.split()[4])
                #If freqjob but runcomplete is unset.
                elif rcount > 50 and runcomplete=="unset":
                    #print("Here. freqsection is:", freqsection)
                    freqsection="notpresent"
                    #print("fresection not present")
        #OPT job settings
            if freqsection=="done" or freqsection=="unset" or freqsection=="notpresent":
                if jobtype == "opt" or jobtype == "optts" or jobtype == "optfreq" or jobtype == "opttsfreq":
                    optjob="yes"
                    #if 'Analytical frequency calculation' in line:
                    #freqinrun="yes"
                    #numatoms
                    #print("optrunconverged is", optrunconverged)
                    #print("optnotconv is", optnotconv)
                    if optrunconverged=="unset" or optnotconv=="unset":
                        if 'OPTIMIZATION RUN DONE' in line:
                            optrunconverged="yes"
                        if 'The optimization did not converge but' in line:
                            optnotconv="yes"
                    if runcomplete=="yes":
                        if optrunconverged=="unset":
                            if 'OPTIMIZATION RUN DONE' in line:
                                if debug=="yes":
                                    print("we are here Script took %s" % (time.time() - start_time))
                                optrunconverged="yes"
                    if optenergy=="unset":
                        if 'FINAL SINGLE POINT ENERGY' in line:
                            #print("line is", line)
                            optenergy=float(line.split()[4])
                            if optrunconverged=="yes":
                                finaloptenergy=optenergy
                            #print("this optenergy is", optenergy)
                            if debug=="yes":
                                print(line)
                                print('In optsection. FINAL S P E line. Read', rcount, 'lines. Script took %s' % (time.time() - start_time))
                    if lastgeomark=="done":
                            #print('lastgeomark is done. Script took %s' % (time.time() - start_time))
                            if optrunconverged=="unset":
                                if 'FINAL ENERGY EVALUATION AT THE STATIONARY POINT' in line:
                                    optrunconverged="yes"
                                    if debug=="yes":
                                        print('Here. OPT HAS CONVERGED. ', rcount, 'lines. Script took %s' % (time.time() - start_time))

# Only Slow line maybe. Necessary??? Maybe
#Using rcount >15015 time reduces barely. Maybe this is not the bottleneck?
                #if rcount >15015:
                #        #print("line is", line)
                    if 'CARTESIAN COORDINATES (A.U.)' in line and lastgeomark=="unset" :
                        coord="active"
                        if debug=="yes":
                            print('Here. Starting coord grab  Read', rcount, 'lines. %s' % (time.time() - start_time))
                    if coord=="active":
                        lastgeo.append(line.strip())
                        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                            coord="inactive"
                            lastgeomark="done"
                            lastgeo.pop(0);lastgeo.pop(0);lastgeo.pop(0);lastgeo.pop();lastgeo.pop()

            #Finds optcycle number if optimization converged
                    #if re.search('***               (AFTER', line):
                    if runcomplete=="yes" and optrunconverged=="yes" and lastgeomark=="done":
                        if '***               (AFTER' in line:
                            optcycle=int(line.split()[2])
            # Signals that optsection is over for a converged done. Breaks later if optsection is set
                    if optcycle!="unset":
                        if 'FINAL ENERGY EVALUATION AT THE STATIONARY POINT' in line:
                            #print("Optsection is now done")
                            optsection="done"
                    #ONly search for opt not conv in last 100 lines
                    if rcount < 100:
                        if 'The optimization did not converge but reached the maximum number of' in line:
                            optnotconv="yes"
            #This finds optcycle number if optimization did not converge. Should break at some point though
#Previously slow LINE. Hopefully fixed now.
                    if lastgeomark=="done":
                        if 'GEOMETRY OPTIMIZATION CYCLE' in line and optcycle=="unset":
                            optcycle=int(line.split()[4])
                            prevoptcycle=optcycle-1
                            #3jan. Disabling optsection here and putting in geomconvtable section instead
                            #optsection="done"
                            #print("runcomplete is", runcomplete)
                            #print("optnotconv is", optnotconv)
                    if optnotconv=="yes" or runcomplete=="unset":
                        if optcycle!="unset":
                            if 'The optimization has not yet converged - more geometry cycles are needed' in line:
                                geomconvtable="active"
                                #print("geomconvtable ACTIVE")
                    if geomconvtable=="active":
                        #if 'RMS gradient' in line:
                        #    print("RMS line is", line)
                        #    rmsgradlist.append(line.strip()[2])
                        geomconv.append(line.strip())
                        if '|Geometry convergence|' in line:
                        #print("line is", line)
                            geomconv.pop(0);geomconv.pop(0)
                            geomconvtable="inactive"
                            geomconvgrab="done"
                            #optsection="done"
                    if geomconvgrab=="done":
                        if 'FINAL SINGLE POINT ENERGY' in line:
                            optenergy=float(line.split()[4])
                            optsection="done"

                if 'DFT' in scfmethod and optrunconverged=="yes" and intelectrons=="unset":
                    if 'N(Total)' in line:
                        intelectrons=line.split()[2]

                if optsection=="done":
                    if debug=="yes":    
                        print('Optsection done, breaking. Script took %s' % (time.time() - start_time))
                    break
        #Integration of electrons. Only necessary for DFT.
        #NOTE. MULTIPLE OCCURENCE
        #Jess example: Can save 0.062 seconds here
    if debug=="yes":
        print('Here. End of reverse loop. Script took %s' % (time.time() - start_time))
    #print("Last line is", line)
# Here begins printing
    #print("runcomplete is", runcomplete)
    #print("optnotconv is", optnotconv)

########################
# Now all read through file done. Printing below.
########################

#################
#SHORT PRINTING MODE
    if shortmode=="yes":
        #print("--------------")
        #print("filename is", filename)
        #print("jobtype is", jobtype)
        if runcomplete=="yes":
            #print("jobtype is", jobtype)
            if jobtype=="sp" or jobtype=="freqsp" and postHF=="unset":
                if jobtype=="freqsp":
                    if len(imaginmodes)==0:
                        print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.OKGREEN + scfenergy), bcolors.ENDC)
                    else:
                        print('{0:40}   {1:10}   {2:40}'.format(bcolors.HEADER + filename+":", bcolors.OKGREEN + scfenergy, bcolors.FAIL + "Imaginary modes"), bcolors.ENDC)
                if jobtype=="sp":
                    if scfconv=="yes" or casscfconv=="yes":
                        print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.OKGREEN + scfenergy), bcolors.ENDC)
                    else:
                        print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.FAIL + "Not converged!"), bcolors.ENDC)
            elif jobtype=="sp" and postHF!="unset":
                print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.OKGREEN + scfenergy), bcolors.ENDC)
            elif optjob=="yes" or jobtype=="optfreq" or jobtype=="opttsfreq":
                if optrunconverged=="yes":
                    #print("optenergy is", optenergy)
                    if hessfail==True:
                       print('{0:40}   {1:10}   {2:40}'.format(bcolors.HEADER + filename+":", bcolors.OKGREEN + str(optenergy), bcolors.FAIL + "Hessian incomplete"), bcolors.ENDC)
                       continue
                    if len(imaginmodes)==0:
                        print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.OKGREEN + str(optenergy)), bcolors.ENDC)
                    else:
                        print('{0:40}   {1:10}   {2:40}'.format(bcolors.HEADER + filename+":", bcolors.OKGREEN + str(optenergy), bcolors.FAIL + "Imaginary modes"), bcolors.ENDC)
                else:
                    print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.FAIL + "Optimization failed!"), bcolors.ENDC)
            elif jobtype=="scan":
                print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.OKGREEN + 'Scan'), bcolors.ENDC)
        elif optnotconv=="yes":
            print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.FAIL + "Optimization failed!"), bcolors.ENDC)
        elif orcacrash=="yes":
            print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.FAIL + "ORCA Crash!"), bcolors.ENDC)
        else:
            print('{0:40}   {1:40}'.format(bcolors.HEADER + filename+":", bcolors.WARNING + "Running?"), bcolors.ENDC)
# LONG PRINTING MODE
    else:
        print("")
        print(bcolors.OKBLUE +"ORCA JobCheck Utility version", scriptversion, "(Python3 version)", bcolors.ENDC)
        print("-----------------------------------------------------------------------")
        print(bcolors.HEADER +"File:", filename, bcolors.ENDC)
        if parproc!="unset":
            print("ORCA version", version, "ran", parproc,"MPI-process job.")
        else:
            print("ORCA version", version, "ran serial job")
        if runcomplete=="yes":
            print(bcolors.OKGREEN +"ORCA terminated normally (",' '.join(runtime),")", bcolors.ENDC)
        elif optnotconv=="yes":
            print(bcolors.FAIL +"ORCA Optimization failed to converge!", bcolors.ENDC)
        elif orcacrash=="yes":
            if optrunconverged=="yes":
                print(bcolors.OKGREEN +"Optimization converged! in (", optcycle, "iterations). YAY!", bcolors.ENDC)
                print(bcolors.OKBLUE + "FINAL OPTIMIZED ENERGY:", finaloptenergy, bcolors.ENDC)
            print(bcolors.FAIL +"ORCA JOB Crashed!", bcolors.ENDC)
            print("Error message:")
            for emes in errormessage:
                print(bcolors.FAIL+emes,bcolors.ENDC)
            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            #with open(filename, errors='ignore') as cfile:
            #    scount=0
            #    bla=[]
            #    nlines=5
            #    for dline in reverse_lines(cfile):
            #        scount += 1
            #        bla.append(dline.strip('\n'))
            #        if scount==nlines:
            #            break
            #        for bline in reversed(bla):
            #            print(bcolors.FAIL +bline, bcolors.ENDC),
            #            break
            #Will now allow last geometry printout as well
            try:
                if optjob=="yes":
                    if optrunconverged=="unset" and sys.argv[2]=="-p":
                        print("Cycle", optcycle, "Cartesian coordinates (", numatoms, "atoms) in Angstrom:")
                        for atom in reversed(lastgeo):
                            print(*atom, sep='')
                    if sys.argv[2]=="-plotgrad":
                        print(bcolors.OKBLUE +"Plotting Gradient in Matplotlib...", bcolors.ENDC)
                        allcycles=list(range(1, optcycle+1))
                        import subprocess
                        import matplotlib.pyplot as plt
                        proc = subprocess.Popen(['grep', " RMS grad", filename],stdout=subprocess.PIPE)
                        proc2 = subprocess.Popen(['grep', " MAX grad", filename],stdout=subprocess.PIPE)
                        all_rmsgrad=[]
                        for line in proc.stdout.readlines():
                            string=line.decode("utf-8").strip().split()
                            val_rmsg=float(string[2])
                            target_rmsg=float(string[3])
                            all_rmsgrad.append(val_rmsg)
                        all_maxgrad=[]
                        for line in proc2.stdout.readlines():
                            string=line.decode("utf-8").strip().split()
                            val_maxg=float(string[2])
                            target_maxg=float(string[3])
                            all_maxgrad.append(val_maxg)
                        plt.plot(allcycles, all_rmsgrad, linestyle='-', color='red', linewidth=2, label="RMS grad")
                        plt.plot(cycles, [target_rmsg] * len(cycles), linestyle='-', color='red', linewidth=1)
                        plt.plot(allcycles, all_maxgrad, linestyle='-', color='blue', linewidth=2, label="Max grad")
                        plt.plot(cycles, [target_maxg] * len(cycles), linestyle='-', color='blue', linewidth=1)
                        plt.xlabel('Optimization cycle')
                        plt.ylabel('Gradient (au/Bohr)')
                        plt.legend(shadow=True, fontsize='small')
                        plt.show()
                    if sys.argv[2]=="-plotstep":
                        print(bcolors.OKBLUE +"Plotting Step in Matplotlib...", bcolors.ENDC)
                        allcycles=list(range(1, optcycle+1))
                        import subprocess
                        import matplotlib.pyplot as plt
                        proc = subprocess.Popen(['grep', " RMS step", filename],stdout=subprocess.PIPE)
                        proc2 = subprocess.Popen(['grep', " MAX step", filename],stdout=subprocess.PIPE)
                        all_rmsstep=[]
                        for line in proc.stdout.readlines():
                            string=line.decode("utf-8").strip().split()
                            val_rmsstep=float(string[2])
                            target_rmsstep=float(string[3])
                            all_rmsstep.append(val_rmsstep)
                        all_maxstep=[]
                        for line in proc2.stdout.readlines():
                            string=line.decode("utf-8").strip().split()
                            val_maxstep=float(string[2])
                            target_maxstep=float(string[3])
                            all_maxstep.append(val_maxstep)
                        plt.plot(allcycles, all_rmsstep, linestyle='-', color='red', linewidth=2, label="RMS step")
                        plt.plot(cycles, [target_rmsstep] * len(cycles), linestyle='-', color='red', linewidth=1)
                        plt.plot(allcycles, all_maxstep, linestyle='-', color='blue', linewidth=2, label="Max step")
                        plt.plot(cycles, [target_maxstep] * len(cycles), linestyle='-', color='blue', linewidth=1)
                        plt.xlabel('Optimization cycle')
                        plt.ylabel('Step (Bohr)')
                        plt.legend(shadow=True, fontsize='small')
                        plt.show()
                    if sys.argv[2]=="-plotenergy":
                        print(bcolors.OKBLUE +"Plotting Energy in Matplotlib...", bcolors.ENDC)
                        allcycles=list(range(1, optcycle+1))
                        import subprocess
                        import matplotlib.pyplot as plt
                        proc = subprocess.Popen(['grep', "FINAL SINGLE", filename],stdout=subprocess.PIPE)
                        all_energies=[]
                        for line in proc.stdout.readlines():
                            string=float(line.decode("utf-8").strip().split()[4])
                            all_energies.append(string)
                        rel_energies=[]
                        for numb in all_energies:
                            rel_energies.append((numb-all_energies[0])*harkcal)
                        plt.plot(allcycles, rel_energies, linestyle='-', color='red', linewidth=2, label="Energy (kcal/mol)")
                        plt.xlabel('Optimization cycle')
                        plt.ylabel('Rel. Energy (kcal/mol)')
                        plt.legend(shadow=True, fontsize='small')
                        plt.show()

            except IndexError:
                print("Do orcajobcheck output -p  to print last geometry (Cycle", optcycle, ")")
                print("Do orcajobcheck output -plotgrad/-plotstep/-plotenergy  to plot gradient/step/energy using Matplotlib")
            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            if xyzfileerror=="yes":
                print(bcolors.FAIL +"Fatal error: Job could not open xyz file", bcolors.ENDC)
            if cpscferror=="yes":
                print(bcolors.FAIL +"CPSCF error", bcolors.ENDC)
        else:
            print(bcolors.WARNING +"ORCA has not terminated with message and may still be running this job", bcolors.ENDC)

    # If earlycrash do not show more output. Else continue
        if earlycrash=="yes":
            pass
        else:
            print("")
            if semiempirical=="yes":
                print(numatoms, "atoms.", "Charge:", charge, " Spin:", spin)
            else:
                print(numatoms, "atoms.", "Charge:", charge, " Spin:", spin, " Contracted basis functions:", conbf)
            if moread=="yes":
                print("Initial orbitals via MOREAD")
            elif autostart=="yes":
                print("Initial orbitals via Autostart")
            else:
                print("Initial orbitals via Guess")
            #print("This is an", jobtype.upper(), "job.")
            if jobtype=="scan":
                scanatomelA=inputgeo[int(scanatomA)].split()[0]
                scanatomelB=inputgeo[int(scanatomB)].split()[0]
                if rctype == "Bond":
                    print("This is a Relaxed Surface Scan. Scanning Bond between atoms", scanatomA+scanatomelA, "and", scanatomB+scanatomelB, ". There will be", scansteps, "steps")
                elif rctype == "Angle":
                    scanatomelC=inputgeo[int(scanatomC)].split()[0]
                    print("This is a Relaxed Surface Scan. Scanning Angle between atoms", scanatomA+scanatomelA, ",", scanatomB+scanatomelB, "and ", scanatomC+scanatomelC, ". There will be", scansteps, "steps")
                if rctype == "Dihedral":
                    scanatomelC=inputgeo[int(scanatomC)].split()[0]
                    scanatomelD=inputgeo[int(scanatomD)].split()[0]
                    print("This is a Relaxed Surface Scan. Scanning Bond between atoms", scanatomA+scanatomelA, ",", scanatomB+scanatomelB, ",",  scanatomC+scanatomelC, "and ", scanatomD+scanatomelD, ". There will be", scansteps, "steps")
                    
                print("Scanning from", scanvalA, "to ", scanvalB, "(change is", round(scanchange,6), " )")
                print("         Scan step       Scan parameter      Energy (hartree)        Rel. energy (kcal/mol)")
                print("====================================================================================================")

                if len(scanenergies)==0:
                    print(bcolors.WARNING +"Still running first scan step ...", bcolors.ENDC)
                    #break
                else:
                    count=0
                    scanpar=float(scanvalA)
                    scanenergies.reverse()
                    refenergy=scanenergies[0]
                    for energy in scanenergies:
                        count += 1
                        energy=float(energy)
                        delta=(float(energy)-float(refenergy))*harkcal
                        #print(count, scanpar, energy, delta)
                        print('{0:10} {1:25f} {2:25f} {3:25f}'.format(count, scanpar, energy, delta))
                        scanpar=float(scanpar)+float(scanchange)
                
                    print("")
                    if runcomplete!="yes":
                        allscanstepnums.reverse()
                        lastscanoptcycle=scanoptcycle[0]
                        print("Currently running: Scan step", allscanstepnums[-1], ", Scan value:", scanpar, ", Optcycle", lastscanoptcycle) 
                        print("Energy is", lastenergy, "and Rel. Energy is", round((float(lastenergy)-float(refenergy))*harkcal, 6), "kcal/mol")
                        try:
                            if sys.argv[2]=="-p":
                                print("Last geometry (", numatoms, "atoms) in Angstrom:")
                                for atom in reversed(lastgeo):
                                    print(*atom, sep='')
                        except IndexError:
                            print("Do orcajobcheck output -p to print current/last geometry.")
                    else:
                        print("Scan completed!")


            if (jobtype=="sp" or jobtype=="freqsp") and postHF=="unset":
                if dft=="yes":
                    if engrad=="yes":
                        print("This is a single-point (Engrad) DFT calculation. Functional:", functional)
                    else:
                        if brokensym=="yes" or flipspin=="yes":
                            print("This is a single-point Broken-symmetry DFT calculation. Functional:", functional)
                            print("First doing single-point High-spin S=", (mult-1)/2, "calculation, then converging to BS MS=", bsms)
                            if runcomplete=="yes":
                                print("HIGH-SPIN ENERGY:", hsenergy)
                                print("BROKEN-SYMMETRY ENERGY:", bsenergy)
                            #else:
                            #    print("Broken-symmetry calculation still running?")
                        else:
                            print("This is a single-point DFT calculation. Functional:", functional)
                elif casscf=="yes":
                    print("This is a single-point CASSCF calculation.")
                elif jobtype=="freqsp":
                    print("This is a single-point HF Freq calculation.")
                else:
                    if engrad=="yes":
                        print("This is a single-point (Engrad) HF calculation.")
                    else:
                        print("This is a single-point HF calculation.")
                if runcomplete=="yes" and scfconv=="yes":
                    print(bcolors.OKGREEN +"SCF CONVERGED AFTER", scfcycles, "CYCLES", bcolors.ENDC)
                    print("HOMO-LUMO gap (alpha) is:", gap_a, "eV")
                    if scftype=="UHF":
                        print("SCF type is", scftype, " S**2:", s2value, " Ideal value:", ideals2value)
                    print(bcolors.OKBLUE +"FINAL SINGLE POINT ENERGY IS", scfenergy, bcolors.ENDC) 
                    if dft=="yes":
                        print("Integrated no. electrons:", intelectrons, "(should be", actualelec, ")")
                elif runcomplete=="yes" and scfconv=="no":
                    print(bcolors.FAIL +"SCF DID NOT CONVERGE in", unfinscfcycles, "cycles. Check your SCF settings.", bcolors.ENDC)
                    if scfalmostconv=="yes":
                        print(bcolors.WARNING +"SCF was close to convergence though (\"signs of convergence\").", bcolors.ENDC)
                        print(bcolors.WARNING +"Energy is", scfenergy, bcolors.ENDC)
                elif runcomplete=="yes" and noiter=="yes":
                    print(bcolors.OKGREEN +"SCF mode: No iterations", bcolors.ENDC)
                    print(bcolors.OKBLUE +"FINAL SINGLE POINT ENERGY IS", scfenergy, bcolors.ENDC)
                    if dft=="yes":
                        print("Integrated no. electrons:", intelectrons, "(should be", actualelec, ")")
                #Runcomplete not true but SCFconv is yet. Probably freqsp job that failed in freq step
                elif runcomplete=="unset" and scfconv=="yes":
                    print(bcolors.OKGREEN +"SCF CONVERGED AFTER", scfcycles, "CYCLES", bcolors.ENDC)
                    if finalsingleline==True:
                        print(bcolors.OKBLUE +"FINAL SINGLE POINT ENERGY IS", scfenergy, bcolors.ENDC)
                    if orcacrash=="yes":
                        print(bcolors.FAIL +"PostSCFjob failed", bcolors.ENDC)
                    else:
                        print(bcolors.WARNING +"Job still running", bcolors.ENDC)
                elif scferrorgeneral=="yes":
                    print(bcolors.FAIL +"SCF has crashed. Sad...", bcolors.ENDC)
                elif scfstillrunning=="yes":
                    print(bcolors.WARNING +"SCF is probably still be running", bcolors.ENDC)
                elif casscf=="yes" and casscfconv=="yes":
                    print(bcolors.OKGREEN +"CASSCF CONVERGED AFTER", lastmacroiter, "CYCLES", bcolors.ENDC)
                    if nevpt2correnergy!="unset":
                        print("NEVPT2 calculation performed")
                        print("NEVPT2 correlation energy is", nevpt2correnergy)
                    print(bcolors.OKBLUE +"FINAL SINGLE POINT ENERGY IS", scfenergy, bcolors.ENDC)
                elif casscf=="yes" and casscfconv=="no":
                    print(bcolors.FAIL +"CASSCF did not converge", bcolors.ENDC)
                else:
                    if casscf=="yes":
                        print(bcolors.WARNING +"CASSCF still running?", bcolors.ENDC)
                    else:
                        print(bcolors.WARNING +"SCF still running?", bcolors.ENDC)
            if jobtype=="sp" and postHF=="yes":
                if extrapolate=="yes":
                    print("This is a single-point", postHFmethod, "calculation. Using extrapolation.")
                else:
                    print("This is a single-point", postHFmethod, "calculation")
                if runcomplete=="yes" and scfconv=="yes" and extrapolate!="yes":
                    print(bcolors.OKGREEN +"SCF CONVERGED AFTER", scfcycles, "CYCLES", bcolors.ENDC)
                    if scftype=="UHF":
                        print("SCF type is", scftype, " S**2:", s2value, " Ideal value:", ideals2value)
                    print("Frozen core is", frozel, "electrons.", "Correlated electrons:", correl)
                    if (postHFmethod=="CC" or postHFmethod=="QCI") and extrapolate!="yes":
                        print("Reference energy is:", refenergy)
                    if extrapolate!="yes":
                        print("Correlation energy is:", correnergy)
                    print(bcolors.OKBLUE +"FINAL SINGLE POINT ENERGY IS", scfenergy, bcolors.ENDC)
                elif runcomplete=="yes" and scfconv=="no":
                    print(bcolors.FAIL +"SCF DID NOT CONVERGE in", unfinscfcycles, "cycles. Check your SCF settings.", bcolors.ENDC)
                    if scfalmostconv=="yes":
                        print(bcolors.WARNING +"SCF was close to convergence though (\"signs of convergence\").", bcolors.ENDC)
                        print(bcolors.WARNING +"Energy is", scfenergy, bcolors.ENDC)
                elif runcomplete=="yes" and noiter=="yes":
                    print(bcolors.OKGREEN +"SCF mode: No iterations", bcolors.ENDC)
                    print("Frozen core is", frozel, "electrons")
                    if postHFmethod=="CC" or postHFmethod=="QCI"  and extrapolation!="yes":
                        print("Reference energy is:", refenergy)
                    print("Correlation energy is:", correnergy)
                    print(bcolors.OKBLUE +"FINAL SINGLE POINT ENERGY IS", scfenergy, bcolors.ENDC)
                elif scferrorgeneral=="yes":
                    print(bcolors.FAIL +"SCF has crashed. Sad...", bcolors.ENDC)
                elif scfstillrunning=="yes":
                    print(bcolors.WARNING +"SCF is probably still be running", bcolors.ENDC)
                elif scfconv=="yes" and runcomplete!="yes" and extrapolate=="unset":
                    print(bcolors.OKGREEN +"SCF is done.", bcolors.ENDC)
                    print(bcolors.OKBLUE +"SCF energy is:", purescfenergy, bcolors.ENDC)
                    print(bcolors.WARNING +"Running post-HF step", bcolors.ENDC)
                elif extrapolate=="yes" and runcomplete!="yes":
                    if orcacrash !="yes":
                        print(bcolors.WARNING+"This is a running extrapolation job",bcolors.ENDC)
                    else:
                        print(bcolors.FAIL+"This was an extrapolation job that crashed",bcolors.ENDC)

                elif extrapolate=="yes" and runcomplete=="yes":
                    print("Frozen core is", frozel, "electrons")
                    print("")
                    print("Extrapolation uses basis sets:", basissets[-1], "and", basissets[-2])
                    print("Extrapolated SCF energy is", extrapscfenergy)
                    print("Extrapolated correlation energy is", extrapcorrenergy)
                    print(bcolors.OKBLUE +"Final extrapolated total energy:", scfenergy, bcolors.ENDC)
                    print("")
                else:
                    print(bcolors.WARNING +"SCF still running?", bcolors.ENDC)

            if jobtype=="optfreq":
                print("This is an OPT+FREQ job")
            if optjob=="yes":
                if optrunconverged=="yes":
                    print(bcolors.OKGREEN +"Optimization converged! in (", optcycle, "iterations). YAY!", bcolors.ENDC) 
                    print(bcolors.OKBLUE + "FINAL OPTIMIZED ENERGY:", finaloptenergy, bcolors.ENDC)
                    #print("jobtype is", jobtype)
                    if jobtype=="optfreq":
                        if freqsection!="done":
                            print(bcolors.WARNING +"Frequency job did not finish", bcolors.ENDC)
                    if dft=="yes":
                        print("Integrated no. electrons:", intelectrons, "(should be", actualelec, ")")
                elif optnotconv=="yes":
                    print(bcolors.FAIL +"Optimization did not converge in", optcycle, "optimization steps", bcolors.ENDC)
                    for gline in reversed(geomconv):
                        print("        ",gline, sep='')
                elif optnotconv=="unset" and orcacrash=="yes":
                    print(bcolors.FAIL +"Optimization crashed", bcolors.ENDC)
                elif optnotconv=="unset" :
                    print(bcolors.WARNING +"Optimization may still be running", bcolors.ENDC)
                    if optcycle==1:
                        print("Optimization Cycle", optcycle, "running.")
                    else:
                        print("Optimization Cycle", prevoptcycle, "energy:", optenergy)
            #print(geomconv)
                    for gline in reversed(geomconv):
                        print("        ",gline, sep='')
                    print("")
                    print("Do orcajobcheck output -grad to print RMS gradient for all cycles")
                    print("Optimization Cycle", optcycle, "in progress")
            if freqjob=="yes":
                if jobtype=="opttsfreq":
                    if optrunconverged!="yes":
                        ssdfs="sdfs"
                    else:
                        if freqsection=="done":
                            if len(imaginmodes)==1:
                                print(bcolors.OKGREEN +"We have 1 imaginary mode (",imaginmodes[0], "cm^-1) for saddlepoint. Good!", bcolors.ENDC)
                            if len(imaginmodes)==0:
                                print(bcolors.FAIL +"We have no imaginary modes for saddlepoint. Bad...", bcolors.ENDC)
                            if len(imaginmodes)>1:
                                print(bcolors.FAIL +"We have many imaginary modes for saddlepoint. Bad...", bcolors.ENDC)
                        else:
                            print(bcolors.WARNING +"Frequency job did not finish", bcolors.ENDC)
                if jobtype=="optfreq" or jobtype=="freqsp":
                    if optrunconverged!="yes":
                        if jobtype=="freqsp":
                            print("Frequencies were calculated")
                            if freqsection=="done":
                                if len(imaginmodes)==1:
                                    print(bcolors.FAIL +"We have 1 imaginary mode for minimum. Bad...:", imaginmodes[0], bcolors.ENDC)
                                elif len(imaginmodes)==0:
                                    print(bcolors.OKGREEN + "We have no imaginary modes for minimum. Good. Lowest mode is", lowestvib, bcolors.ENDC)
                                    if float(lowestvib) < 0:
                                        print(bcolors.WARNING + "Probably some numerical noise present, however.", bcolors.ENDC)
                                elif len(imaginmodes)>1:
                                    print(bcolors.FAIL +"Wes have several imaginary modes for minimum. Bad...", bcolors.ENDC)

                        elif jobtype=="optfreq":
                            print(bcolors.FAIL +"Optimization did not finish properly.",bcolors.ENDC)
                        else:
                            print(bcolors.WARNING +"Frequency job did not finish", bcolors.ENDC)
                    else:
                        if freqsection=="done":
                            if hessfail==True:
                                print(bcolors.FAIL + "Hessian is not complete. Inspect output", bcolors.ENDC)
                                exit()
                            if len(imaginmodes)==1:
                                print(bcolors.FAIL +"We have 1 imaginary mode for minimum. Bad...:", imaginmodes[0], bcolors.ENDC)
                            elif len(imaginmodes)==0:
                                print(bcolors.OKGREEN + "We have no imaginary modes for minimum. Good. Lowest mode is", lowestvib, bcolors.ENDC)
                                if float(lowestvib) < 0:
                                    print(bcolors.WARNING +"Probably some numerical noise present, however.", bcolors.ENDC )
                            elif len(imaginmodes)>1:
                                print(bcolors.FAIL +"We have several imaginary modes for minimum. Bad...", bcolors.ENDC)
                        else:
                            print(bcolors.WARNING +"Frequency job did not finish", bcolors.ENDC)
                try:
                    if sys.argv[2]=="-t" and freqsection=="done":
                        print("")
                        print("Thermochemistry corrections:")
                        print("Zero-point energy correction, ZPE:", zeropointcorr, "Eh")
                        print("Total Enthalpy correction, Hcorr:", enthalpycorr, "Eh")
                        print("Total Entropy correction, TS:", entropycorr,"Eh")
                        print("Total Free energy correction (Hcorr - TS), Gcorr:", gthermcorr, "Eh")
                #else:
                #    if optrunconverged=="yes":

                #        print("Do orcajobcheck output -t  to print thermochemical corrections")
                except IndexError:
                        if optrunconverged=="yes" and freqsection=="done":
                            print("Do orcajobcheck output -t  to print thermochemical corrections")
            
            try:
                if jobtype=="sp":
                    if sys.argv[2]=="-l":
                        nlines=int(sys.argv[3])
                        with open(filename, errors='ignore') as cfile:
                            scount=0
                            bla=[]
                            for dline in reverse_lines(cfile):
                                scount += 1
                                bla.append(dline.strip('\n'))
                                if scount==nlines:
                                    print(bcolors.UNDERLINE +"Last", nlines, "lines of output:",bcolors.ENDC)
                                    for bline in reversed(bla):
                                        print(bline),
                                    break
                    if sys.argv[2]=="-grad":
                        print("CASSCF gradient (||g) per macroiteration (using grep)")
                        import subprocess
                        subprocess.call(['grep', '||g|| =', filename])
            except IndexError:
                print("Do orcajobcheck output -l N  to print last N lines.")
                if casscf=="yes":
                    print("Do orcajobcheck output -grad  to print CASSCF gradient.")
            try:
                if optjob=="yes":
                    if optrunconverged=="yes" and sys.argv[2]=="-p":
                        print("Optimized Cartesian coordinates (", numatoms, "atoms) in Angstrom:")
                        for atom in reversed(lastgeo):
                            print(*atom, sep='')
                    if optrunconverged=="unset" and sys.argv[2]=="-p":
                        print("Cycle", optcycle, "Cartesian coordinates (", numatoms, "atoms) in Angstrom:")
                        for atom in reversed(lastgeo):
                            print(*atom, sep='')
                    if optrunconverged=="unset" and sys.argv[2]=="-grad":
                        print("RMS gradient per Cycle (using grep)")
                        import subprocess
                        subprocess.call(['grep', ' RMS gradient', filename])
                    if sys.argv[2]=="-plotgrad":
                        print(bcolors.OKBLUE +"Plotting Gradient in Matplotlib...", bcolors.ENDC)
                        allcycles=list(range(1, optcycle+1))
                        import subprocess
                        import matplotlib.pyplot as plt
                        proc = subprocess.Popen(['grep', " RMS grad", filename],stdout=subprocess.PIPE)
                        proc2 = subprocess.Popen(['grep', " MAX grad", filename],stdout=subprocess.PIPE)
                        all_rmsgrad=[]
                        
                        for line in proc.stdout.readlines():
                            string=line.decode("utf-8").strip().split()
                            val_rmsg=float(string[2])
                            target_rmsg=float(string[3])
                            all_rmsgrad.append(val_rmsg)
                        all_maxgrad=[]
                        for line in proc2.stdout.readlines():
                            string=line.decode("utf-8").strip().split()
                            val_maxg=float(string[2])
                            target_maxg=float(string[3])
                            all_maxgrad.append(val_maxg)
                        if optrunconverged=="yes":
                            pass
                        elif len(allcycles) > len(all_rmsgrad):
                            #Removing current cycle from allcycles because ORCA has probably not calculated energy for this yet.
                            allcycles.pop(-1)
                        plt.ylim([-0.005,0.05])
                        plt.plot(allcycles, all_rmsgrad, linestyle='-', color='red', linewidth=2, label="RMS grad")
                        plt.plot(allcycles, [target_rmsg] * len(allcycles), linestyle='-', color='red', linewidth=1)
                        plt.plot(allcycles, all_maxgrad, linestyle='-', color='blue', linewidth=2, label="Max grad")
                        plt.plot(allcycles, [target_maxg] * len(allcycles), linestyle='-', color='blue', linewidth=1)

                        plt.xlabel('Optimization cycle')
                        plt.ylabel('Gradient (au/Bohr)')
                        plt.legend(shadow=True, fontsize='small')
                        plt.show()
                    if sys.argv[2]=="-plotstep":
                        print(bcolors.OKBLUE +"Plotting Step in Matplotlib...", bcolors.ENDC)
                        allcycles=list(range(1, optcycle+1))
                        import subprocess
                        import matplotlib.pyplot as plt
                        proc = subprocess.Popen(['grep', " RMS step", filename],stdout=subprocess.PIPE)
                        proc2 = subprocess.Popen(['grep', " MAX step", filename],stdout=subprocess.PIPE)
                        all_rmsstep=[]
                        for line in proc.stdout.readlines():
                            string=line.decode("utf-8").strip().split()
                            val_rmsstep=float(string[2])
                            target_rmsstep=float(string[3])
                            all_rmsstep.append(val_rmsstep)
                        all_maxstep=[]
                        for line in proc2.stdout.readlines():
                            string=line.decode("utf-8").strip().split()
                            val_maxstep=float(string[2])
                            target_maxstep=float(string[3])
                            all_maxstep.append(val_maxstep)
                        if optrunconverged=="yes":
                            pass
                        elif len(allcycles) > len(all_rmsstep):
                            #Removing current cycle from allcycles because ORCA has probably not calculated energy for this yet.
                            allcycles.pop(-1)
                        plt.plot(allcycles, all_rmsstep, linestyle='-', color='red', linewidth=2, label="RMS step")
                        plt.plot(allcycles, [target_rmsstep] * len(allcycles), linestyle='-', color='red', linewidth=1)
                        plt.plot(allcycles, all_maxstep, linestyle='-', color='blue', linewidth=2, label="Max step")
                        plt.plot(allcycles, [target_maxstep] * len(allcycles), linestyle='-', color='blue', linewidth=1)
                        plt.xlabel('Optimization cycle')
                        plt.ylabel('Step (Bohr)')
                        plt.legend(shadow=True, fontsize='small')
                        plt.show()
                    if sys.argv[2]=="-plotenergy":
                        print(bcolors.OKBLUE +"Plotting Energy in Matplotlib...", bcolors.ENDC)
                        allcycles=list(range(1, optcycle+1))
                        import subprocess
                        import matplotlib.pyplot as plt
                        proc = subprocess.Popen(['grep', "FINAL SINGLE", filename],stdout=subprocess.PIPE)
                        all_energies=[]
                        for line in proc.stdout.readlines():
                            string=float(line.decode("utf-8").strip().split()[4])
                            all_energies.append(string)
                        rel_energies=[]
                        for numb in all_energies:
                            rel_energies.append((numb-all_energies[0])*harkcal)
                        if optrunconverged=="yes":
                            #Removing last energy here because ORCA did extra energy step. No. energies and cycles have to match
                            rel_energies.pop(-1)
                        elif len(allcycles) > len(rel_energies):
                            #Removing current cycle from allcycles because ORCA has probably not calculated energy for this yet.
                            allcycles.pop(-1)
                        plt.plot(allcycles, rel_energies, linestyle='-', color='red', linewidth=2, label="Energy (kcal/mol)")
                        plt.xlabel('Optimization cycle')
                        plt.ylabel('Rel. Energy (kcal/mol)')
                        plt.legend(shadow=True, fontsize='small')
                        plt.show()

                elif jobtype=="sp":
                    if sys.argv[2]=="-p":
                        print("Cartesian coordinates of input geometry (", numatoms, "atoms) in Angstrom:")
                        for atom in inputgeo:
                            print(*atom, sep='')
            except IndexError:
                if optrunconverged=="yes":
                    print("Do orcajobcheck output -p  to print optimized geometry")
                    print("Do orcajobcheck output -plotgrad/-plotstep/-plotenergy  to plot gradient/step/energy using Matplotlib")
                elif jobtype=="sp":
                    print("Do orcajobcheck output -p  to print input geometry")
                else:
                    print("Do orcajobcheck output -p  to print last geometry (Cycle", optcycle, ")")
                    print("Do orcajobcheck output -plotgrad/-plotstep/-plotenergy  to plot gradient/step/energy using Matplotlib")
            print("")

if debug=="yes":
    print('Script took %s' % (time.time() - start_time))
    print("Last line read in file is:", line)

