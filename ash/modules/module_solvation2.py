#####################
# SOLVSHELL MODULE (PART OF ASH) #
#####################
# For now only the snapshot-part. Will read snapshots from QM/MM MD Tcl-Chemshell run.
import numpy as np
import time
import statistics
import shutil
import os
import sys
import multiprocessing as mp
import glob

import ash.functions.functions_solv
from ash.functions.functions_general import blankline,BC,listdiff,print_time_rel_and_tot,print_line_with_mainheader,print_line_with_subheader1, ashexit
from ash.modules.module_coords import read_fragfile_xyz
from ash.interfaces.interface_ORCA import run_inputfiles_in_parallel,finalenergiesgrab,run_orca_SP_ORCApar
import ash.settings_solvation
import ash.constants

#import ash


beginTime = time.time()
CheckpointTime = time.time()
#NEW SOLVSHELL VERSION. PolEmbedding used from beginning
def solvshell_v2 ( orcadir='', NumCores=None, calctype='', orcasimpleinput_LL='',
        orcablockinput_LL='', orcasimpleinput_HL='', orcablockinput_HL='',
        orcasimpleinput_SRPOL='', orcablockinput_SRPOL='', EOM='', BulkCorrection='',
        GasCorrection='', ShortRangePolarization='', SRPolShell='',
        LongRangePolarization='', PrintFinalOutput='', Testmode='', repsnapmethod='',
        repsnapnumber='', solvbasis='', chargeA='', multA='', chargeB='', multB='', psi4memory=3000,
        psi4_functional='', psi4dict='', pot_option='', LRPolRegion=20, PolQMRegion=0, psi4runmode='psithon'):

    #While charge/mult info is read from md-variables.defs in case redox AB job, this info is not present
    # for both states in case of single trajectory. Plus one might want to do either VIE, VEA or SpinState change
    #Hence defining in original py inputfile makes sense

    #Yggdrasill dir (needed for init function and print_header below). Todo: remove
    programdir=os.path.dirname(ash.__file__)
    programversion=0.1
    blankline()
    print_solvshell_header(programversion,programdir)


    calcdir=os.getcwd()
    sys.path.append(calcdir)
    os.chdir(calcdir)

    print(BC.OKBLUE,"Input variables defined:", BC.END)
    print("-----------------------------------")
    print("Calculation directory:", calcdir)
    print("orcadir:", orcadir )
    print("NumCores:", NumCores )
    print("calctype:", calctype )
    print("EOM:", EOM)
    print("orcasimpleinput_LL:", orcasimpleinput_LL)
    print("orcablockinput_LL:", orcablockinput_LL)
    print("ShortRangePolarization:", ShortRangePolarization)
    print("orcasimpleinput_SRPOL:", orcasimpleinput_SRPOL)
    print("orcablockinput_SRPOL:", orcablockinput_SRPOL)
    print("solvbasis:", solvbasis)
    print("SRPolShell:", SRPolShell)
    print("LongRangePolarization:", LongRangePolarization)
    print("LRPolRegion:", LRPolRegion)
    print("PolQMRegion:", PolQMRegion)
    print("BulkCorrection:", BulkCorrection)
    print("GasCorrection:", GasCorrection)
    print("orcasimpleinput_HL:", orcasimpleinput_HL)
    print("orcablockinput_HL:", orcablockinput_HL)
    print("repsnapmethod:", repsnapmethod)
    print("repsnapnumber:", repsnapnumber)
    print("PrintFinalOutput:", PrintFinalOutput)
    print("Testmode:", Testmode)
    print("-----------------------------------")

    #Load some global settings and making orcadir global
    settings_solvation.init(programdir,orcadir,NumCores)

    blankline()
    print_line_with_mainheader("CALCULATION TYPE: {}".format(calctype.upper()))
    blankline()

    mdvarfile=calcdir+'/md-variables.defs'

    #Getting system information from MD-run variable file
    print("Reading MD-run variable file:", mdvarfile)
    #calcdir contains md-variables.defs and snaps dir with snapshots
    #Create system object with information about the system (charge,mult of states A, B, forcefield, snapshotlist etc.)
    #Attributes: name, chargeA, multA, chargeB, multB, solutetypesA, solutetypesB, solventtypes, snapslist, snapshotsA, snapshotsB
    if calctype=="redox":
        solvsphere=read_md_variables_fileAB(mdvarfile)
    elif calctype=="vie":
        solvsphere=read_md_variables_fileA(mdvarfile)
    else:
        print("unknown calctype for md-read")
        ashexit()
    print("Solvsphere Object defined.")
    print("Solvsphere atoms:", solvsphere.numatoms)

    #Updating charge/mult info in solvsphere object since not always present (in md file) for VIE/VEA/Spinstate jobs
    solvsphere.ChargeA=chargeA
    solvsphere.MultA=multA
    solvsphere.ChargeB=chargeB
    solvsphere.MultB=multB

    #Simple general connectivity stored in solvsphere object: solvsphere.connectivity
    solvsphere.calc_connectivity()
    print("Solvsphere connectivity stored in file: snaps/stored_connectivity")
    with open(calcdir+'/snaps/stored_connectivity', 'w') as connfile:
        for i in solvsphere.connectivity:
            connfile.write(str(i)+'\n')

    #Temporary redefinition of lists for easier faster test runs
    if Testmode == True:
        if calctype=='redox':
            solvsphere.snapslist, solvsphere.snapshotsA, solvsphere.snapshotsB, solvsphere.snapshots = TestModerunAB()
        elif calctype=='vie':
            solvsphere.snapslist, solvsphere.snapshotsA, solvsphere.snapshots = TestModerunA()

    #Get solvent pointcharges for solvent-unit. e.g. [-0.8, 0.4, 0.4] for TIP3P, assuming [O, H, H] order
    # Later use dictionary or object or something for this
    blankline()
    print("Solvent information:")
    if solvsphere.solvtype=="tip3p":
        print("Solvent: TIP3P")
        solventunitcharges=[-0.834, 0.417, 0.417]
        print("Pointcharges of solvent fragment:", solventunitcharges)
        print("Atom types of solvent unit:", solvsphere.solventtypes )
    else:
        print("Unknown solvent")
        solventunitcharges=[]
        exit_solvshell()

    blankline()

    ###################################################
    # All snapshots: POLARIZED Low-Level Theory  PSI4 #
    ###################################################
    print_line_with_mainheader("Low-level theory on All Snapshots")
    CheckpointTime = time.time()
    # cd to snaps dir, create separate dir for calculations and copy fragmentfiles to it.
    os.chdir('./snaps')
    os.mkdir('snaps-LL')
    os.chdir('./snaps-LL')
    for i in solvsphere.snapshots:
        shutil.copyfile('../'+i+'.c', './'+i+'.c')
    print("Current dir:", os.getcwd())
    # Temp: Convert snapshots in Tcl-Chemshell fragment file format to xyz coords in Angstrom.
    #Future change: Have code spit out XYZ files instead of Chemshell fragment files

    # RUNNING POL PSI4 jobs in parallel
    # Cores for Psi4. Only 1 since we are parallelizing over all snapshots
    NumCoresPsi4 = 1
    pool = mp.Pool(NumCores)
    results = pool.map(Polsnapshotcalc, [[snapshot, solvsphere, psi4dict, psi4_functional, pot_option,
                                            NumCoresPsi4, PolQMRegion, LRPolRegion, psi4memory, psi4runmode]
                                           for snapshot in solvsphere.snapshots])
    pool.close()
    pool.join()
    print("results:", results)
    # Combining and making dicts
    #AllsnapsABenergy, AsnapsABenergy, BsnapsABenergy=grab_energies_output_ORCA(snapshotinpfiles)
    AsnapsABenergy = {}
    BsnapsABenergy = {}
    AllsnapsABenergy = {}
    for r in results:
        if 'snapA' in r[0]:
            AsnapsABenergy[r[0]] = r[1]
            #AsnapsABenergy.append(r[1])
        elif 'snapB' in r[0]:
            BsnapsABenergy[r[0]] = r[1]
            #BsnapsABenergy.append(r[1])
        if calctype == "redox":
            AllsnapsABenergy[r[0]] = r[1]
            #AllsnapsABenergy.append(r[1])

    blankline()

    print("AllsnapsABenergy:", AllsnapsABenergy)
    print("AsnapsABenergy:", AsnapsABenergy)
    print("BsnapsABenergy:", BsnapsABenergy)
    blankline()

    #Average and stdevs of
    ave_trajA = statistics.mean(AsnapsABenergy.values())
    stdev_trajA = statistics.stdev(AsnapsABenergy.values())
    if calctype=="redox":
        # Averages and stdeviations over whole trajectory at LL theory
        ave_trajAB = statistics.mean(AllsnapsABenergy.values())
        # stdev_trajAB = statistics.stdev(list(AllsnapsABenergy.values()))
        stdev_trajAB = 0.0
        ave_trajB = statistics.mean(BsnapsABenergy.values())
        stdev_trajB = statistics.stdev(BsnapsABenergy.values())
        print("TrajA average: {:3.3f} eV. Stdev: {:3.3f} eV.".format(ave_trajA, stdev_trajA))
        print("TrajB average: {:3.3f} eV. Stdev: {:3.3f} eV.".format(ave_trajB, stdev_trajB))
        print("A+B average: {:3.3f} eV. Stdev: {:3.3f} eV.".format(ave_trajAB, stdev_trajAB))
    else:
        print("TrajA average: {:3.3f} eV. Stdev: {:3.3f} eV.".format(ave_trajA, stdev_trajA))

    blankline()

    #######################################################
    # REPRESENTATIVE SNAPSHOTS
    #######################################################
    print("Representative snapshot method:", repsnapmethod)
    print("Representative snapshot number:", repsnapnumber)
    #Creating dictionaries:
    repsnapsA=repsnaplist(repsnapmethod, repsnapnumber, AsnapsABenergy)
    if calctype=="redox":
        repsnapsB=repsnaplist(repsnapmethod, repsnapnumber, BsnapsABenergy)
    #Combined list of repsnaps
    print("Representative snapshots for each trajectory")
    blankline()
    print("Traj A:")
    for i in repsnapsA:
        print(i)
    blankline()
    if calctype=="redox":
        print("Traj B:")
        for i in repsnapsB:
            print(i)

    repsnaplistA = list(repsnapsA.keys())
    if calctype=="redox":
        repsnaplistB = list(repsnapsB.keys())
        repsnaplistAB=repsnaplistA+repsnaplistB
        totrepsnaps=repsnaplistAB
    else:
        totrepsnaps = repsnaplistA
    #New repsnapsvariable:  totrepsnaps  (contains either A or B snapshots)
    blankline()
    #Averages and stdeviations over repsnaps  at LL theory
    repsnap_ave_trajA = statistics.mean(list(repsnapsA.values()))
    repsnap_stdev_trajA = statistics.stdev(list(repsnapsA.values()))
    if calctype=="redox":
        repsnap_ave_trajB = statistics.mean(list(repsnapsB.values()))
        repsnap_stdev_trajB = statistics.stdev(list(repsnapsB.values()))
        repsnap_ave_trajAB= statistics.mean([repsnap_ave_trajA,repsnap_ave_trajB])
        repsnap_stdev_trajAB= "TBD"
        print("Repsnaps TrajA average: {:3.3f} ± {:3.3f} eV".format(repsnap_ave_trajA, repsnap_stdev_trajA))
        print("Repsnaps TrajB average: {:3.3f} ± {:3.3f} eV".format(repsnap_ave_trajB, repsnap_stdev_trajB))
        print("Repsnaps TrajAB average: {:3.3f} ± TBD eV".format(repsnap_ave_trajAB))
    else:
        print("Repsnaps TrajA average: {:3.3f} ± {:3.3f} eV".format(repsnap_ave_trajA, repsnap_stdev_trajA))

    blankline()
    print("Deviation between repsnaps mean and full mean for A: {:3.3f} eV.".format(ave_trajA-repsnap_ave_trajA))
    if calctype=="redox":
        print("Deviation between repsnaps mean and full mean for B: {:3.3f} eV.".format(ave_trajB-repsnap_ave_trajB))
    blankline()
    print_time_rel_and_tot(CheckpointTime, beginTime,'All snaps')
    CheckpointTime = time.time()
    #Going up to snaps dir again
    os.chdir('..')

    if BulkCorrection==True:
        CheckpointTime = time.time()
        #############################################
        # Representative snapshots: Bulk Correction. Done with ORCA ??TODO:  #
        #############################################
        print_line_with_mainheader("Bulk Correction on Representative Snapshots")
        os.mkdir('Bulk-LL')
        os.chdir('./Bulk-LL')
        for i in totrepsnaps:
            shutil.copyfile('../' + i + '.c', './' + i + '.c')
        print("Entering dir:", os.getcwd())

        print("Doing BulkCorrection on representative snapshots. Creating inputfiles...")
        print("Using hollow bulk sphere:", ash.settings_solvation.bulksphere.pathtofile)
        print("Number of added bulk point charges:", ash.settings_solvation.bulksphere.numatoms)
        bulkcorr=True
        identifiername='_Bulk_LL'
        print("repsnaplistA:", repsnaplistA)
        if calctype == "redox":
            print("repsnaplistB:", repsnaplistB)
        blankline()
        solute_atoms = solvsphere.soluteatomsA
        solvent_atoms = solvsphere.solventatoms
        bulkinpfiles = create_AB_inputfiles_ORCA(solute_atoms, solvent_atoms, solvsphere, totrepsnaps,
                                                         orcasimpleinput_LL, orcablockinput_LL, solventunitcharges, identifiername, None, bulkcorr)

        # RUN BULKCORRECTION INPUTFILES
        print_line_with_subheader1("Running Bulk Correction calculations at LL theory")
        print(BC.WARNING,"LL-theory:", orcasimpleinput_LL,BC.END)
        run_inputfiles_in_parallel(orcadir, bulkinpfiles, NumCores)

        #GRAB output
        Bulk_Allrepsnaps_ABenergy, Bulk_Arepsnaps_ABenergy, Bulk_Brepsnaps_ABenergy=grab_energies_output_ORCA(bulkinpfiles)
        blankline()
        #Get bulk correction per snapshot
        #print("Bulk_Allrepsnaps_ABenergy:", Bulk_Allrepsnaps_ABenergy)
        print("Bulk-IP values for traj A:", Bulk_Arepsnaps_ABenergy)
        if calctype == "redox":
            print("Bulk-IP values for traj B:", Bulk_Brepsnaps_ABenergy)
        blankline()
        print("Non-Bulk values for repsnapsA:", repsnapsA)
        if calctype == "redox":
            print("Non-Bulk values for repsnapsB:", repsnapsB)

        Bulk_ave_trajA=statistics.mean(Bulk_Arepsnaps_ABenergy.values())
        Bulk_stdev_trajA=statistics.stdev(Bulk_Arepsnaps_ABenergy.values())
        if calctype == "redox":
            Bulk_ave_trajB=statistics.mean(Bulk_Brepsnaps_ABenergy.values())
            Bulk_ave_trajAB=statistics.mean([Bulk_ave_trajA,Bulk_ave_trajB])
            Bulk_stdev_trajB=statistics.stdev(Bulk_Brepsnaps_ABenergy.values())
            Bulk_stdev_trajAB=0.0

        blankline()
        if calctype == "redox":
            print("Bulk calculation TrajA average: {:3.3f} ± {:3.3f}".format(Bulk_ave_trajA, Bulk_stdev_trajA))
            print("Bulk calculation TrajB average: {:3.3f} ± {:3.3f}".format(Bulk_ave_trajB, Bulk_stdev_trajB))
            print("Bulk calculation TrajAB average: {:3.3f} ± TBD".format(Bulk_ave_trajAB))
        else:
            print("Bulk calculation TrajA average: {:3.3f} ± {:3.3f}".format(Bulk_ave_trajA, Bulk_stdev_trajA))

        bulkcorrdict_A={}
        bulkcorrdict_B={}
        for b in Bulk_Arepsnaps_ABenergy:
            for c in repsnapsA:
                if b==c:
                    bulkcorrdict_A[b]=Bulk_Arepsnaps_ABenergy[b]-repsnapsA[b]
        if calctype == "redox":
            for b in Bulk_Brepsnaps_ABenergy:
                for c in repsnapsB:
                    if b==c:
                        bulkcorrdict_B[b]=Bulk_Brepsnaps_ABenergy[b]-repsnapsB[b]

        blankline()
        print("Bulk corrections per snapshot:")
        print("bulkcorrdict_A:", bulkcorrdict_A)
        if calctype == "redox":
            print("bulkcorrdict_B", bulkcorrdict_B)
            Bulkcorr_mean_A=statistics.mean(bulkcorrdict_A.values())
            Bulkcorr_mean_B=statistics.mean(bulkcorrdict_B.values())
            Bulkcorr_stdev_A=statistics.stdev(bulkcorrdict_A.values())
            Bulkcorr_stdev_B=statistics.stdev(bulkcorrdict_B.values())
            Bulkcorr_mean_AB=statistics.mean([Bulkcorr_mean_A,Bulkcorr_mean_B])
            blankline()
            print("Traj A: Bulkcorrection {:3.3f} ± {:3.3f} eV".format(Bulkcorr_mean_A, Bulkcorr_stdev_A))
            print("Traj B: Bulkcorrection {:3.3f} ± {:3.3f} eV".format(Bulkcorr_mean_B, Bulkcorr_stdev_B))
            print("Combined Bulkcorrection {:3.3f} eV".format(Bulkcorr_mean_AB))
        else:
            Bulkcorr_mean_A = statistics.mean(bulkcorrdict_A.values())
            Bulkcorr_stdev_A = statistics.stdev(bulkcorrdict_A.values())
            blankline()
            print("Traj A: Bulkcorrection {:3.3f} ± {:3.3f} eV".format(Bulkcorr_mean_A, Bulkcorr_stdev_A))

        blankline()
        print_time_rel_and_tot(CheckpointTime, beginTime,'Bulk')
        CheckpointTime = time.time()
        #Clean up GBW files and other
        gbwfiles = glob.glob('*.gbw')
        fragfiles = glob.glob('*.c')
        pcfiles = glob.glob('*.pc')
        for gbwfile in gbwfiles:
            os.remove(gbwfile)
        for fragfile in fragfiles:
            os.remove(fragfile)
        for pcfile in pcfiles:
            os.remove(pcfile)
        #Going up to snaps dir again
        os.chdir('..')
    else:
        Bulk_ave_trajAB=0; Bulk_stdev_trajAB=0; Bulkcorr_mean_AB=0
        Bulk_ave_trajB=0; Bulk_stdev_trajB=0; Bulkcorr_mean_B=0
        Bulk_ave_trajA=0; Bulk_stdev_trajA=0; Bulkcorr_mean_A=0

    if ShortRangePolarization==True:
        CheckpointTime = time.time()
        #############################################
        # Short-Range Polarization (QM-region expansion) calculations #
        #############################################
        print_line_with_mainheader("Short-Range Polarization calculations: QM-Region Expansion")
        os.mkdir('SRPol-LL')
        os.chdir('./SRPol-LL')
        for i in totrepsnaps:
            shutil.copyfile('../' + i + '.c', './' + i + '.c')

        print("Snapshots:", totrepsnaps)
        print("Current dir:", os.getcwd())

        # INCREASED QM-REGION CALCULATIONS
        print("Doing Short-Range Polarization Step. Creating inputfiles...")
        print("Using QM-region shell:", SRPolShell, "Å")
        blankline()
        # PART 1
        #Create inputfiles of repsnapshots with increased QM regions
        identifiername='_SR_LL'
        SRPolinpfiles = create_AB_inputfiles_ORCA(solute_atoms, solvent_atoms, solvsphere, totrepsnaps,
                                                         orcasimpleinput_SRPOL, orcablockinput_SRPOL, solventunitcharges,
                                                          identifiername, SRPolShell, False, solvbasis)

        # Run ShortRangePol INPUTFILES (increased QM-region)
        print_line_with_subheader1("Running SRPol calculations at LL theory")
        print(BC.WARNING,"LL-theory:", orcasimpleinput_SRPOL,BC.END)
        print(BC.WARNING,"LL-theory:", orcablockinput_SRPOL,BC.END)
        print("Solvbasis:", solvbasis)
        SRORCAPar=True
        if SRORCAPar==True:
            print("Using ORCA parallelization. Running each file 1 by 1. ORCA using {} cores".format(NumCores))
            for SRPolinpfile in SRPolinpfiles:
                print("Running file: ", SRPolinpfile)
                run_orca_SP_ORCApar(orcadir, SRPolinpfile, numcores=NumCores)
        else:
            print("Using multiproc parallelization. All calculations running in parallel but each ORCA calculation using 1 core.")
            run_inputfiles_in_parallel(orcadir, SRPolinpfiles, NumCores)

        #GRAB output
        SRPol_Allrepsnaps_ABenergy, SRPol_Arepsnaps_ABenergy, SRPol_Brepsnaps_ABenergy=grab_energies_output_ORCA(SRPolinpfiles)
        blankline()

        # PART 2.
        # Whether to calculate repsnapshots again at SRPOL level of theory
        if orcasimpleinput_SRPOL == orcasimpleinput_LL:
            print("orcasimpleinput_SRPOL is same as orcasimpleinput_LL")
            print("Using previously calculated values for Region1")
            SRPol_Arepsnaps_ABenergy_Region1=repsnapsA
            if calctype=="redox":
                SRPol_Brepsnaps_ABenergy_Region1=repsnapsB
        else:
            print("orcasimpleinput_SRPOL is different")
            print("Need to recalculate repsnapshots at SRPOL level of theory using regular QM-region")
            identifiername = '_SR_LL_Region1'
            SRPolinpfiles_Region1 = create_AB_inputfiles_ORCA(solute_atoms, solvent_atoms, solvsphere, totrepsnaps,
                                                         orcasimpleinput_SRPOL, orcablockinput_SRPOL, solventunitcharges,
                                                          identifiername, None, False, solvbasis)
            #Run the inputfiles
            run_inputfiles_in_parallel(orcadir, SRPolinpfiles_Region1, NumCores)
            #Grab the energies
            SRPol_Allrepsnaps_ABenergy_Region1, SRPol_Arepsnaps_ABenergy_Region1, SRPol_Brepsnaps_ABenergy_Region1 = grab_energies_output_ORCA(
                SRPolinpfiles_Region1)


        #Calculate SRPol correction per snapshot
        #print("SRPol_Allrepsnaps_ABenergy:", SRPol_Allrepsnaps_ABenergy)
        print("Large QM-region SRPol-IP for traj A:", SRPol_Arepsnaps_ABenergy)
        if calctype=="redox":
            print("Large QM-region SRPol-IP for traj B", SRPol_Brepsnaps_ABenergy)
        blankline()
        print("Regular QM-region values for repsnapsA:", SRPol_Arepsnaps_ABenergy_Region1)
        if calctype=="redox":
            print("Regular QM-region for repsnapsB:", SRPol_Brepsnaps_ABenergy_Region1)


        if calctype=="redox":
            SRPol_ave_trajA=statistics.mean(SRPol_Arepsnaps_ABenergy.values())
            SRPol_ave_trajB=statistics.mean(SRPol_Brepsnaps_ABenergy.values())
            SRPol_ave_trajAB=statistics.mean([SRPol_ave_trajA,SRPol_ave_trajB])
            SRPol_stdev_trajA=statistics.stdev(SRPol_Arepsnaps_ABenergy.values())
            SRPol_stdev_trajB=statistics.stdev(SRPol_Brepsnaps_ABenergy.values())
            SRPol_stdev_trajAB=0.0
            blankline()
            print("SRPol calculation TrajA average: {:3.3f} ± {:3.3f}".format(SRPol_ave_trajA, SRPol_stdev_trajA))
            print("SRPol calculation TrajB average: {:3.3f} ± {:3.3f}".format(SRPol_ave_trajB, SRPol_stdev_trajB))
            print("SRPol calculation TrajAB average: {:3.3f} ± TBD".format(SRPol_ave_trajAB))
        else:
            SRPol_ave_trajA=statistics.mean(SRPol_Arepsnaps_ABenergy.values())
            SRPol_stdev_trajA=statistics.stdev(SRPol_Arepsnaps_ABenergy.values())
            blankline()
            print("SRPol calculation TrajA average: {:3.3f} ± {:3.3f}".format(SRPol_ave_trajA, SRPol_stdev_trajA))

        SRPolcorrdict_A={}
        SRPolcorrdict_B={}

        #Calculating correction per snapshot
        for b in SRPol_Arepsnaps_ABenergy:
            for c in repsnapsA:
                if b==c:
                    SRPolcorrdict_A[b]=SRPol_Arepsnaps_ABenergy[b]-SRPol_Arepsnaps_ABenergy_Region1[b]
        if calctype == "redox":
            for b in SRPol_Brepsnaps_ABenergy:
                for c in repsnapsB:
                    if b==c:
                        SRPolcorrdict_B[b]=SRPol_Brepsnaps_ABenergy[b]-SRPol_Brepsnaps_ABenergy_Region1[b]

        blankline()
        print("Dictionaries of corrections per snapshots:")
        print("SRPolcorrdict_A:", SRPolcorrdict_A)
        print("SRPolcorrdict_B", SRPolcorrdict_B)

        if calctype == "redox":
            SRPolcorr_mean_A=statistics.mean(SRPolcorrdict_A.values())
            SRPolcorr_mean_B=statistics.mean(SRPolcorrdict_B.values())
            SRPolcorr_stdev_A=statistics.stdev(SRPolcorrdict_A.values())
            SRPolcorr_stdev_B=statistics.stdev(SRPolcorrdict_B.values())
            SRPolcorr_mean_AB=statistics.mean([SRPolcorr_mean_A,SRPolcorr_mean_B])
            blankline()
            print("Traj A: SRPolcorrection {:3.3f} ± {:3.3f} eV".format(SRPolcorr_mean_A, SRPolcorr_stdev_A))
            print("Traj B: SRPolcorrection {:3.3f} ± {:3.3f} eV".format(SRPolcorr_mean_B, SRPolcorr_stdev_B))
            print("Combined SRPolcorrection {:3.3f} eV".format(SRPolcorr_mean_AB))
        else:
            SRPolcorr_mean_A=statistics.mean(SRPolcorrdict_A.values())
            SRPolcorr_stdev_A=statistics.stdev(SRPolcorrdict_A.values())
            blankline()
            print("Traj A: SRPolcorrection {:3.3f} ± {:3.3f} eV".format(SRPolcorr_mean_A, SRPolcorr_stdev_A))
        blankline()
        print_time_rel_and_tot(CheckpointTime, beginTime,'SRPol')
        CheckpointTime = time.time()
        #Going up to snaps dir again
        os.chdir('..')
    else:
        SRPol_ave_trajAB=0; SRPol_stdev_trajAB=0; SRPolcorr_mean_AB=0
        SRPol_ave_trajB=0; SRPol_stdev_trajB=0; SRPolcorr_mean_B=0
        SRPol_ave_trajA=0; SRPol_stdev_trajA=0; SRPolcorr_mean_A=0

    if LongRangePolarization==True:
        CheckpointTime = time.time()
        ##############################################################
        # Long-Range Polarization (QM-region expansion) calculations #
        #Now using PolEmbed via psi4
        ##############################################################
        print_line_with_mainheader("Long-Range Polarization calculations: Psi4 Level")
        os.mkdir('LRPol-LL')
        os.chdir('./LRPol-LL')
        for i in totrepsnaps:
            shutil.copyfile('../' + i + '.c', './' + i + '.c')
        print("Current dir:", os.getcwd())

        print("Doing Long-Range Polarization Step. Creating inputfiles...")
        print("Snapshots:", totrepsnaps)
        print("LRPolRegion1", LRPolRegion1, "Å")
        print("LRPolRegion2:", LRPolRegion2, "Å")
        print("LRPolQMRegion:", LRPolQMRegion, "Å")
        blankline()
        #RUNNING LRPOL PSI4 jobs in parallel
        # EXAMPLE:
        print("totrepsnaps:", totrepsnaps)

        # Cores for Psi4 set depending on snapshots available (len(totrepsnaps))and total cores(NumCores)
        NumCoresPsi4 = int(NumCores/len(totrepsnaps))
        pool = mp.Pool(len(totrepsnaps))
        results = pool.map(LRPolsnapshotcalc, [[snapshot, solvsphere, psi4dict, psi4_functional, pot_option,
                                                LRPolRegion1, LRPolRegion2, NumCoresPsi4, LRPolQMRegion, psi4memory, psi4runmode]
                                               for snapshot in totrepsnaps])
        pool.close()
        pool.join()
        #Results contain list of lists where each list : [snapshotname, VIE_LR1, VIE_LR2]
        print("results:", results)
        #Combinining
        LRPol_Arepsnaps_ABenergy_Region1=[]
        LRPol_Arepsnaps_ABenergy_Region2=[]
        LRPol_Brepsnaps_ABenergy_Region1=[]
        LRPol_Brepsnaps_ABenergy_Region2=[]
        LRPol_Allrepsnaps_ABenergy_Region1=[]
        LRPol_Allrepsnaps_ABenergy_Region2=[]
        LRPolcorrdict_A = {}
        LRPolcorrdict_B = {}
        for r in results:
            if 'snapA' in r[0]:
                LRPol_Arepsnaps_ABenergy_Region1.append(r[1])
                LRPol_Arepsnaps_ABenergy_Region2.append(r[2])
                LRPolcorrdict_A[r[0]]=r[2]-r[1]
            elif 'snapB' in r[0]:
                LRPol_Brepsnaps_ABenergy_Region1.append(r[1])
                LRPol_Brepsnaps_ABenergy_Region2.append(r[2])
                LRPolcorrdict_B[r[0]] = r[2]-r[1]
            if calctype=="redox":
                LRPol_Allrepsnaps_ABenergy_Region1.append(r[1])
                LRPol_Allrepsnaps_ABenergy_Region2.append(r[2])

        # Gathering stuff for both regions
        if calctype == "redox":
            print("LRPol_Allrepsnaps_ABenergy_Region1:", LRPol_Allrepsnaps_ABenergy_Region1)
            print("LRPol_Arepsnaps_ABenergy_Region1:", LRPol_Arepsnaps_ABenergy_Region1)
            print("LRPol_Brepsnaps_ABenergy_Region1:", LRPol_Brepsnaps_ABenergy_Region1)
            print("LRPol_Allrepsnaps_ABenergy_Region2:", LRPol_Allrepsnaps_ABenergy_Region2)
            print("LRPol_Arepsnaps_ABenergy_Region2:", LRPol_Arepsnaps_ABenergy_Region2)
            print("LRPol_Brepsnaps_ABenergy_Region2:", LRPol_Brepsnaps_ABenergy_Region2)
            #Calculating averages and stdevs
            LRPol_ave_trajA_Region2 = statistics.mean(LRPol_Arepsnaps_ABenergy_Region2)
            LRPol_ave_trajB_Region2 = statistics.mean(LRPol_Brepsnaps_ABenergy_Region2)
            LRPol_ave_trajAB_Region2 = statistics.mean([LRPol_ave_trajA_Region2, LRPol_ave_trajB_Region2])
            LRPol_stdev_trajA_Region2 = statistics.stdev(LRPol_Arepsnaps_ABenergy_Region2)
            LRPol_stdev_trajB_Region2 = statistics.stdev(LRPol_Brepsnaps_ABenergy_Region2)
            LRPol_stdev_trajAB_Region2 = 0.0
            LRPol_ave_trajA_Region1 = statistics.mean(LRPol_Arepsnaps_ABenergy_Region1)
            LRPol_ave_trajB_Region1 = statistics.mean(LRPol_Brepsnaps_ABenergy_Region1)
            LRPol_ave_trajAB_Region1 = statistics.mean([LRPol_ave_trajA_Region1, LRPol_ave_trajB_Region1])
            LRPol_stdev_trajA_Region1 = statistics.stdev(LRPol_Arepsnaps_ABenergy_Region1)
            LRPol_stdev_trajB_Region1 = statistics.stdev(LRPol_Brepsnaps_ABenergy_Region1)
            LRPol_stdev_trajAB_Region1 = 0.0
            blankline()
            print("LRPol_Region2 calculation TrajA average: {:3.3f} ± {:3.3f}".format(LRPol_ave_trajA_Region2, LRPol_stdev_trajA_Region2))
            print("LRPol_Region2 calculation TrajB average: {:3.3f} ± {:3.3f}".format(LRPol_ave_trajB_Region2, LRPol_stdev_trajB_Region2))
            print("LRPol_Region2 calculation TrajAB average: {:3.3f} ± TBD".format(LRPol_ave_trajAB_Region2))
            print("LRPol_Region1 calculation TrajA average: {:3.3f} ± {:3.3f}".format(LRPol_ave_trajA_Region1, LRPol_stdev_trajA_Region1))
            print("LRPol_Region1 calculation TrajB average: {:3.3f} ± {:3.3f}".format(LRPol_ave_trajB_Region1, LRPol_stdev_trajB_Region1))
            print("LRPol_Region1 calculation TrajAB average: {:3.3f} ± TBD".format(LRPol_ave_trajAB_Region1))

            LRPolcorr_mean_A = statistics.mean(LRPolcorrdict_A.values())
            LRPolcorr_mean_B = statistics.mean(LRPolcorrdict_B.values())
            LRPolcorr_stdev_A = statistics.stdev(LRPolcorrdict_A.values())
            LRPolcorr_stdev_B = statistics.stdev(LRPolcorrdict_B.values())
            LRPolcorr_mean_AB = statistics.mean([LRPolcorr_mean_A, LRPolcorr_mean_B])
            blankline()
            print("Traj A: LRPolcorrection {:3.3f} ± {:3.3f} eV".format(LRPolcorr_mean_A, LRPolcorr_stdev_A))
            print("Traj B: LRPolcorrection {:3.3f} ± {:3.3f} eV".format(LRPolcorr_mean_B, LRPolcorr_stdev_B))
            print("Combined LRPolcorrection {:3.3f} eV".format(LRPolcorr_mean_AB))

        else:
            print("LRPol_Allrepsnaps_ABenergy_Region1:", LRPol_Allrepsnaps_ABenergy_Region1)
            print("LRPol_Arepsnaps_ABenergy_Region1:", LRPol_Arepsnaps_ABenergy_Region1)
            print("LRPol_Brepsnaps_ABenergy_Region1:", LRPol_Brepsnaps_ABenergy_Region1)
            print("LRPol_Allrepsnaps_ABenergy_Region2:", LRPol_Allrepsnaps_ABenergy_Region2)
            print("LRPol_Arepsnaps_ABenergy_Region2:", LRPol_Arepsnaps_ABenergy_Region2)
            print("LRPol_Brepsnaps_ABenergy_Region2:", LRPol_Brepsnaps_ABenergy_Region2)
            # Calculating averages and stdevs
            LRPol_ave_trajA_Region2 = statistics.mean(LRPol_Arepsnaps_ABenergy_Region2)
            LRPol_stdev_trajA_Region2 = statistics.stdev(LRPol_Arepsnaps_ABenergy_Region2)
            LRPol_ave_trajA_Region1 = statistics.mean(LRPol_Arepsnaps_ABenergy_Region1)
            LRPol_stdev_trajA_Region1 = statistics.stdev(LRPol_Arepsnaps_ABenergy_Region1)
            blankline()
            print("LRPol_Region2 calculation TrajA average: {:3.3f} ± {:3.3f}".format(LRPol_ave_trajA_Region2,
                                                                                      LRPol_stdev_trajA_Region2))
            print("LRPol_Region1 calculation TrajA average: {:3.3f} ± {:3.3f}".format(LRPol_ave_trajA_Region1,
                                                                                      LRPol_stdev_trajA_Region1))

            print("LRPolcorrections (A):", list(np.array(LRPol_Arepsnaps_ABenergy_Region2)-np.array(LRPol_Arepsnaps_ABenergy_Region1)))
            print("LRPolcorrections (B):", list(np.array(LRPol_Brepsnaps_ABenergy_Region2)-np.array(LRPol_Brepsnaps_ABenergy_Region1)))
            LRPolcorr_mean_A = statistics.mean(LRPolcorrdict_A.values())
            LRPolcorr_stdev_A = statistics.stdev(LRPolcorrdict_A.values())
            blankline()
            print("Traj A: LRPolcorrection {:3.3f} ± {:3.3f} eV".format(LRPolcorr_mean_A, LRPolcorr_stdev_A))

        blankline()
        print_time_rel_and_tot(CheckpointTime, beginTime,'LRPol')
        CheckpointTime = time.time()
        #Going up to snaps dir again
        os.chdir('..')
    else:
        LRPol_ave_trajAB=0; LRPol_stdev_trajAB=0; LRPolcorr_mean_AB=0
        LRPol_ave_trajB=0; LRPol_stdev_trajB=0; LRPolcorr_mean_B=0
        LRPol_ave_trajA=0; LRPol_stdev_trajA=0; LRPolcorr_mean_A=0
        LRPol_ave_trajA_Region1=0; LRPol_stdev_trajA_Region1=0;
        LRPol_ave_trajA_Region2=0; LRPol_stdev_trajA_Region2=0
        LRPol_ave_trajB_Region1=0; LRPol_stdev_trajB_Region1=0;
        LRPol_ave_trajB_Region2=0; LRPol_stdev_trajB_Region2=0
        LRPol_ave_trajAB_Region1=0; LRPol_stdev_trajAB_Region1=0;
        LRPol_ave_trajAB_Region2=0; LRPol_stdev_trajAB_Region2=0
    blankline()


    if GasCorrection:
        ####################
        # Gas calculations #
        ####################
        print_line_with_mainheader("Gas calculations: Low-Level and High-Level Theory")

        gaslistA = ['gas-molA.c']

        os.mkdir('Gas-calculations')
        os.chdir('./Gas-calculations')
        shutil.copyfile('../' + 'gas-molA.c', './' + 'gas-molA.c')
        if calctype=="redox":
            gaslistB = ['gas-molB.c']
            shutil.copyfile('../' + 'gas-molB.c', './' + 'gas-molB.c')
        print("Current dir:", os.getcwd())

        print("Doing Gas calculations. Creating inputfiles...")
        print("gaslistA:", gaslistA)
        if calctype=="redox":
            print("gaslistB:", gaslistB)
            gaslist=gaslistA+gaslistB
        else:
            gaslist=gaslistA
        identifiername='_Gas_LL'
        #create_AB_inputfiles_onelist
        gasinpfiles_LL = create_AB_inputfiles_ORCA(solute_atoms, [], solvsphere, gaslist,orcasimpleinput_LL,
                                                       orcablockinput_LL, solventunitcharges, identifiername)
        identifiername='_Gas_HL'
        gasinpfiles_HL = create_AB_inputfiles_ORCA(solute_atoms, [], solvsphere, gaslist,orcasimpleinput_HL,
                                                       orcablockinput_HL, solventunitcharges, identifiername)

        print("Created inputfiles:")
        print(gasinpfiles_LL)
        print(gasinpfiles_HL)
        # RUN GASCORRECTION INPUTFILES
        print_line_with_subheader1("Running Gas calculations at LL theory")
        print(BC.WARNING, "LL-theory:", orcasimpleinput_LL, BC.END)
        run_inputfiles_in_parallel(orcadir, gasinpfiles_LL, NumCores)
        # HL Gas phase would later be run using OpenMPI parallelization
        print_line_with_subheader1("Running Gas calculations at HL theory")
        print(BC.WARNING,"HL-theory:", orcasimpleinput_HL,BC.END)
        #run_inputfiles_in_parallel_AB(gasinpfiles_HL)
        run_orca_SP_ORCApar(orcadir, gasinpfiles_HL[0], numcores=NumCores)
        if calctype=="redox":
            run_orca_SP_ORCApar(orcadir, gasinpfiles_HL[1], numcores=NumCores)

        #GRAB output
        gasA_stateA_LL=finalenergiesgrab('gas-molA_StateAB_Gas_LL.out')[0]
        gasA_stateB_LL=finalenergiesgrab('gas-molA_StateAB_Gas_LL.out')[1]
        gasA_VIE_LL=(gasA_stateB_LL-gasA_stateA_LL)*ash.constants.hartoeV
        print("gasA_VIE_LL:", gasA_VIE_LL)

        if calctype == "redox":
            gasB_stateA_LL=finalenergiesgrab('gas-molB_StateAB_Gas_LL.out')[0]
            gasB_stateB_LL=finalenergiesgrab('gas-molB_StateAB_Gas_LL.out')[1]
            gasB_VIE_LL=(gasA_stateB_LL-gasB_stateB_LL)*ash.constants.hartoeV
            gasAB_AIE_LL=(gasB_stateB_LL-gasA_stateA_LL)*ash.constants.hartoeV

            print("gasB_VIE_LL:", gasB_VIE_LL)
            print("gasAB_AIE_LL:", gasAB_AIE_LL)
        blankline()


        if calctype == "redox":
            gasA_stateA_HL=finalenergiesgrab('gas-molA_StateAB_Gas_HL.out')[0]
            gasA_stateB_HL=finalenergiesgrab('gas-molA_StateAB_Gas_HL.out')[1]
            gasA_VIE_HL=(gasA_stateB_HL-gasA_stateA_HL)*ash.constants.hartoeV
            gasB_stateA_HL=finalenergiesgrab('gas-molB_StateAB_Gas_HL.out')[0]
            gasB_stateB_HL=finalenergiesgrab('gas-molB_StateAB_Gas_HL.out')[1]
            gasB_VIE_HL=(gasA_stateB_HL-gasB_stateB_HL)*ash.constants.hartoeV
            gasAB_AIE_HL=(gasB_stateB_HL-gasA_stateA_HL)*ash.constants.hartoeV
            print("gasA_VIE_HL:", gasA_VIE_HL)
            print("gasB_VIE_HL:", gasB_VIE_HL)
            print("gasAB_AIE_HL:", gasAB_AIE_HL)
        else:
            gasA_stateA_HL=finalenergiesgrab('gas-molA_StateAB_Gas_HL.out')[0]
            gasA_stateB_HL=finalenergiesgrab('gas-molA_StateAB_Gas_HL.out')[1]
            gasA_VIE_HL=(gasA_stateB_HL-gasA_stateA_HL)*ash.constants.hartoeV
            print("gasA_VIE_HL:", gasA_VIE_HL)


        blankline()
        print_time_rel_and_tot(CheckpointTime, beginTime,'Gas calculations')
        CheckpointTime = time.time()
        # Going up to snaps dir again
        os.chdir('..')
    else:
        gasAB_AIE_LL=0; gasAB_AIE_HL=0;
        gasB_VIE_LL=0; gasB_VIE_HL=0;
        gasA_VIE_LL=0; gasA_VIE_HL=0


    #PRINT FINAL OUTPUT
    if PrintFinalOutput==True:
        if calctype=="redox":
            print_line_with_mainheader("FINAL OUTPUT: REDOX")
            blankline()
            print_line_with_subheader1("Trajectory A")

            print_redox_output_state("A", solvsphere, orcasimpleinput_LL, orcasimpleinput_HL, solvsphere.snapshotsA, ave_trajA, stdev_trajA,
                                     repsnap_ave_trajA, repsnap_stdev_trajA, repsnaplistA, Bulk_ave_trajA, Bulk_stdev_trajA, Bulkcorr_mean_A,
                                     SRPol_ave_trajA, SRPol_stdev_trajA, SRPolcorr_mean_A, LRPol_ave_trajA_Region1, LRPol_ave_trajA_Region2,
                                     LRPol_stdev_trajA_Region1, LRPol_stdev_trajA_Region2, LRPolcorr_mean_A, gasA_VIE_LL, gasA_VIE_HL)
            print_line_with_subheader1("Trajectory B")
            print_redox_output_state("B", solvsphere, orcasimpleinput_LL, orcasimpleinput_HL, solvsphere.snapshotsB, ave_trajB, stdev_trajB,
                                     repsnap_ave_trajB, repsnap_stdev_trajB, repsnaplistB, Bulk_ave_trajB, Bulk_stdev_trajB, Bulkcorr_mean_B,
                                     SRPol_ave_trajB, SRPol_stdev_trajB, SRPolcorr_mean_B, LRPol_ave_trajB_Region1, LRPol_ave_trajB_Region2,
                                     LRPol_stdev_trajB_Region1, LRPol_stdev_trajB_Region2, LRPolcorr_mean_B, gasB_VIE_LL, gasB_VIE_HL)
            print_line_with_subheader1("Final Average")
            print_redox_output_state("AB", solvsphere, orcasimpleinput_LL, orcasimpleinput_HL, solvsphere.snapshots, ave_trajAB, stdev_trajAB,
                                     repsnap_ave_trajAB, repsnap_stdev_trajAB, repsnaplistAB, Bulk_ave_trajAB, Bulk_stdev_trajAB, Bulkcorr_mean_AB,
                                     SRPol_ave_trajAB, SRPol_stdev_trajAB, SRPolcorr_mean_AB, LRPol_ave_trajAB_Region1, LRPol_ave_trajAB_Region2,
                                     LRPol_stdev_trajAB_Region1, LRPol_stdev_trajAB_Region2, LRPolcorr_mean_AB, gasAB_AIE_LL, gasAB_AIE_HL)
        else:
            print_line_with_mainheader("FINAL OUTPUT: {}".format(calctype))
            blankline()
            print_line_with_subheader1("Trajectory A")

            print_redox_output_state("A", solvsphere, orcasimpleinput_LL, orcasimpleinput_HL, solvsphere.snapshotsA, ave_trajA, stdev_trajA,
                                     repsnap_ave_trajA, repsnap_stdev_trajA, repsnaplistA, Bulk_ave_trajA, Bulk_stdev_trajA, Bulkcorr_mean_A,
                                     SRPol_ave_trajA, SRPol_stdev_trajA, SRPolcorr_mean_A, LRPol_ave_trajA_Region1, LRPol_ave_trajA_Region2,
                                     LRPol_stdev_trajA_Region1, LRPol_stdev_trajA_Region2, LRPolcorr_mean_A, gasA_VIE_LL, gasA_VIE_HL)

            blankline()
    blankline()
    blankline()
    print_time_rel_and_tot(CheckpointTime, beginTime, 'final output')
    print_solvshell_footer()

    #Clean-up files
    #Snapshot frag files
    fragfiles = glob.glob('*.c')
    for fragfile in fragfiles:
        os.remove(fragfile)

    print("Solvshell done!")

# Function to do all calcs for 1 snapshot (used with multiprocessing)
def Polsnapshotcalc(args):
    print("Starting function: LRPolsnapshotcalc")
    print(mp.current_process())
    print("args:", args)
    snapshot=args[0]
    solvsphere=args[1]
    psi4dict=args[2]
    psi4_functional=args[3]
    pot_option=args[4]
    NumCoresPsi4=args[5]
    PolQMRegion=args[6]
    LRPolRegion=args[7]
    psi4memory=args[8]
    psi4runmode=args[9]

    # create dir for each snapshot and copy snapshot into it
    os.mkdir(snapshot+'_dir')
    os.chdir(snapshot+'_dir')
    shutil.copyfile('../' + snapshot + '.c', './' + snapshot + '.c')

    # Potential options: SEP (Standard Potential), TIP3P Todo: Other options: To be done!
    # PE Solvent-type label for PyFrame. For water, use: HOH, TIP3? WAT?
    PElabel_pyframe = 'HOH'

    # Get elems and coords from each Chemshell frament file
    # Todo: Change to XYZ-file read-in instead (if snapfiles have been converted)
    elems, coords = read_fragfile_xyz(snapshot)
    # create Yggdrasill fragment
    snap_frag = ash.Fragment(elems=elems, coords=coords)
    # QM and PE regions
    solute_elems = [elems[i] for i in solvsphere.soluteatomsA]
    #solute_coords = [coords[i] for i in solvsphere.soluteatomsA]
    solute_coords = np.take(coords,solvsphere.soluteatomsA,axis=0)
    # Defining QM and PE regions
    PEsolvshell = get_solvshell(solvsphere, snap_frag.elems, snap_frag.coords, LRPolRegion, solute_elems,
                                  solute_coords,
                                  ash.settings_solvation.scale, ash.settings_solvation.tol)
    qm_solvshell = get_solvshell(solvsphere, snap_frag.elems, snap_frag.coords, PolQMRegion, solute_elems,
                                  solute_coords,
                                  ash.settings_solvation.scale, ash.settings_solvation.tol)
    qmatoms = solvsphere.soluteatomsA + qm_solvshell #QMatoms. solute + possible QM solvshell
    peatoms = listdiff(PEsolvshell, qmatoms )  # Polarizable atoms, except QM shell
    mmatoms = listdiff(solvsphere.allatoms, qmatoms + peatoms)  # Nonpolarizable atoms

    print("qmatoms ({} atoms): {}".format(len(qmatoms), qmatoms))
    print("PEsolvshell num is", len(PEsolvshell))
    print("peatoms ({} atoms)".format(len(peatoms)))
    print("mmatoms ({} atoms)".format(len(mmatoms)))
    print("Sum of  QM+PE+MM atoms:", len(qmatoms)+len(peatoms)+len(mmatoms))
    blankline()
    blankline()
    print("Num All atoms:", len(solvsphere.allatoms))
    if len(qmatoms)+len(peatoms)+len(mmatoms) != len(solvsphere.allatoms):
        print("QM + MM + PE atoms ({})not equal to total numatoms({})".format(len(qmatoms)+len(peatoms)+len(mmatoms),
                                                                              len(solvsphere.allatoms)))
        ashexit()

    # Define Psi4 QMregion
    Psi4QMpart_A = ash.Psi4Theory(charge=solvsphere.ChargeA, mult=solvsphere.MultA, label=snapshot+'A',
                                         psi4settings=psi4dict, outputname=snapshot+'A.out', psi4memory=psi4memory,
                                         psi4functional=psi4_functional, runmode=psi4runmode, printsetting=False)
    Psi4QMpart_B = ash.Psi4Theory(charge=solvsphere.ChargeB, mult=solvsphere.MultB, label=snapshot+'B',
                                         psi4settings=psi4dict, outputname=snapshot+'B.out', psi4memory=psi4memory,
                                         psi4functional=psi4_functional, runmode=psi4runmode, printsetting=False)

    # Create PolEmbed theory object. fragment always defined with it
    PolEmbed_SP_A = ash.PolEmbedTheory(fragment=snap_frag, qm_theory=Psi4QMpart_A,
                                                  qmatoms=qmatoms, peatoms=peatoms, mmatoms=mmatoms,
                                                  pot_option=pot_option, potfilename=snapshot+'System',
                                                  pyframe=True, pot_create=True, PElabel_pyframe=PElabel_pyframe)

    # Note: pot_create=False for B since the embedding potential is the same
    PolEmbed_SP_B = ash.PolEmbedTheory(fragment=snap_frag, qm_theory=Psi4QMpart_B,
                                                  qmatoms=qmatoms, peatoms=peatoms, mmatoms=mmatoms,
                                                  pot_option=pot_option, potfilename=snapshot+'System',
                                                  pyframe=True, pot_create=False, PElabel_pyframe=PElabel_pyframe)


    # Simple Energy SP calc. potfile needed for B run.
    blankline()
    print(BC.OKGREEN,
          "Starting PolEmbed job for snapshot {} with PolRegion: {}. State A: Charge: {}  Mult: {}".format(
              snapshot, LRPolRegion, solvsphere.ChargeA, solvsphere.MultA), BC.END)
    PolEmbedEnergyA = PolEmbed_SP_A.run(potfile=snapshot+'System.pot', numcores=NumCoresPsi4, restart=False)

    # Doing chargeB (assumed open-shell) after closed-shell.
    print(BC.OKGREEN,
          "Starting PolEmbed job for snapshot {} with LRPolRegion1: {}. State B: Charge: {}  Mult: {}".format(
              snapshot, LRPolRegion, solvsphere.ChargeB, solvsphere.MultB), BC.END)
    PolEmbedEnergyB = PolEmbed_SP_B.run(potfile=snapshot+'System.pot', numcores=NumCoresPsi4, restart=True)

    PolEmbedEnergyAB = (PolEmbedEnergyB - PolEmbedEnergyA) * ash.constants.hartoeV

    #Going up one dir
    os.chdir('..')

    #Returning list of snapshotname and energies for both regions
    return [snapshot, PolEmbedEnergyAB]
