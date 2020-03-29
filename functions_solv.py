import statistics
from functions_ORCA import *
import time
import constants
from functions_xtb import *
import functions_general


def TestModerunAB():
    snapslist = ['60000', '60400', '60800', '61200', '61600', '62000']
    snapshotsA = ['snapA-60000', 'snapA-60400', 'snapA-60800', 'snapA-61200', 'snapA-61600', 'snapA-62000']
    snapshotsB = ['snapB-60000', 'snapB-60400', 'snapB-60800', 'snapB-61200', 'snapB-61600', 'snapB-62000']
    snapshots = ['snapA-60000', 'snapA-60400', 'snapA-60800', 'snapA-61200', 'snapA-61600', 'snapA-62000',
                 'snapB-60000', 'snapB-60400', 'snapB-60800', 'snapB-61200', 'snapB-61600', 'snapB-62000']
    print("Test mode True. Only running snapshots (for A and B respectively):", snapslist)
    return snapslist, snapshotsA, snapshotsB, snapshots

def TestModerunA():
    snapslist = ['60000', '60400', '60800', '61200', '61600', '62000']
    snapshotsA = ['snapA-60000', 'snapA-60400', 'snapA-60800', 'snapA-61200', 'snapA-61600', 'snapA-62000']
    snapshots = ['snapA-60000', 'snapA-60400', 'snapA-60800', 'snapA-61200', 'snapA-61600', 'snapA-62000']
    print("Test mode True. Only running snapshots (for A ):", snapslist)
    return snapslist, snapshotsA, snapshots

def print_time_rel_and_tot(timestampA,timestampB, modulename=''):
    secsA=time.time()-timestampA
    minsA=secsA/60
    hoursA=minsA/60
    secsB=time.time()-timestampB
    minsB=secsB/60
    hoursB=minsB/60
    print("-------------------------------------------------------------------")
    print("Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes, {:3.1f} hours".format(modulename, secsA, minsA, hoursA ))
    print("Total Walltime: {:3.1f} seconds, {:3.1f} minutes, {:3.1f} hours".format(secsB, minsB, hoursB ))
    print("-------------------------------------------------------------------")

def print_time(timestamp):
    secs=time.time()-timestamp
    mins=secs/60
    hours=mins/60
    print("-------------------------------------------------------------------")
    print("Total Walltime: {:3.1f} seconds, {:3.1f} minutes, {:3.1f} hours".format(secs, mins, hours ))
    print("-------------------------------------------------------------------")

def print_reltime(timestamp):
    secs=time.time()-timestamp
    mins=secs/60
    hours=mins/60
    print("-------------------------------------------------------------------")
    print("Time to calculate step: {:3.1f} seconds, {:3.1f} minutes, {:3.1f} hours".format(secs, mins, hours ))
    print("-------------------------------------------------------------------")

def exit_solvshell():
    print("Solvshell exited ")

# Define System Class. Useful for storing information about the system
class SystemA:
    def __init__(self, name, chargeA, multA, solutetypesA, solventtypes, snapslist, snapshotsA, solvtype):
        self.snapdir = 'snaps'
        self.Name = name
        self.ChargeA = chargeA
        self.MultA = multA
        self.solutetypesA = solutetypesA
        self.solventtypes = solventtypes
        # Creating list of atoms. Assuming solute atoms are at top.
        self.soluteatomsA = list(range(0,len(solutetypesA)))
        self.snapslist = snapslist
        self.snapshotsA = snapshotsA
        self.snapshots = snapshotsA
        self.solvtype = solvtype
        self.numatoms = self.get_numatoms_from_snaps(self.snapshotsA[0])
        self.allatoms = list(range(0,self.numatoms))
        #List of solvent atoms
        self.solventatoms = listdiff(self.allatoms, self.soluteatomsA)
        #No connectivity unless function is run
        self.connectivity=[]
    def change_name(self, new_name): # note that the first argument is self
        self.name = new_name # access the class attribute with the self keyword1
    #Get number of atoms from first snapshot in list. Assuming all snapshots the same. To be replaced by reading in from md-variables.defs
    def get_numatoms_from_snaps(self, snap):
        with open(self.snapdir+'/'+snap+'.c') as file:
            for line in file:
                if 'block = coordinates records = ' in line:
                    return int(line.split()[-1])
    #Calculate connectivity. Assuming all snapshots are identical w.r.t. connectivity. Assuming 3-atom solvent also.
    def calc_connectivity(self):
        self.connectivity=[]
        self.connectivity.append(self.soluteatomsA)
        for i in range(0,len(self.solventatoms),3):
            solv=self.solventatoms[i]
            self.connectivity.append([solv, solv+1, solv+2])

# Define System Class. Useful for storing information about the system
class SystemAB:
    def __init__(self, name, chargeA, multA, chargeB, multB, solutetypesA, solutetypesB, solventtypes, snapslist, snapshotsA, snapshotsB, solvtype):
        self.snapdir = 'snaps'
        self.Name = name
        self.ChargeA = chargeA
        self.MultA = multA
        self.ChargeB = chargeB
        self.MultB = multB
        self.solutetypesA = solutetypesA
        self.solutetypesB = solutetypesB
        self.solventtypes = solventtypes
        # Creating list of atoms. Assuming solute atoms are at top.
        self.soluteatomsA = list(range(0,len(solutetypesA)))
        self.soluteatomsB = list(range(0, len(solutetypesB)))
        self.snapslist = snapslist
        self.snapshotsA = snapshotsA
        self.snapshotsB = snapshotsB
        self.snapshots = snapshotsA+snapshotsB
        self.solvtype = solvtype
        self.numatoms = self.get_numatoms_from_snaps(self.snapshotsA[0])
        self.allatoms = list(range(0,self.numatoms))
        #List of solvent atoms
        self.solventatoms = listdiff(self.allatoms, self.soluteatomsA)
        #No connectivity unless function is run
        self.connectivity=[]
    def change_name(self, new_name): # note that the first argument is self
        self.name = new_name # access the class attribute with the self keyword1
    #Get number of atoms from first snapshot in list. Assuming all snapshots the same. To be replaced by reading in from md-variables.defs
    def get_numatoms_from_snaps(self, snap):
        with open(self.snapdir+'/'+snap+'.c') as file:
            for line in file:
                if 'block = coordinates records = ' in line:
                    return int(line.split()[-1])
    #Calculate connectivity. Assuming all snapshots are identical w.r.t. connectivity. Assuming 3-atom solvent also.
    def calc_connectivity(self):
        self.connectivity=[]
        self.connectivity.append(self.soluteatomsA)
        for i in range(0,len(self.solventatoms),3):
            solv=self.solventatoms[i]
            self.connectivity.append([solv, solv+1, solv+2])



#Function to determine solvshell or increased QM-region
#Include all whole solvent molecules within X Å from every atom in solute
def get_solvshell(solvsphere, allelems,allcoords,QMregion,subsetelems,subsetcoords,scale,tol):
    #print("inside get_solvshell")
    if len(solvsphere.connectivity) == 0:
        print("No connectivity found. Using slow way of finding nearby solvent molecules...")
    atomlist=[]
    for i,c in enumerate(subsetcoords):
        el=subsetelems[i]
        #print("---Solute atom:", el, c)
        for index,allc in enumerate(allcoords):
            all_el=allelems[index]
            if index >= len(subsetcoords):

                dist=distance(c,allc)
                #print("dist:", dist)
                if dist < QMregion:
                    #print("yes. index is:", index)
                    #print("")
                    #Get molecule members atoms for atom index.
                    #Using stored connectivity because takes forever otherwise
                    #If no connectivity
                    if len(solvsphere.connectivity) == 0:
                        wholemol=get_molecule_members_loop(allcoords, allelems, index, 1, scale, tol)
                    #If stored connectivity
                    else:
                        for q in solvsphere.connectivity:
                            #print("q:", q)
                            if index in q:
                                wholemol=q
                                #print("wholemol", wholemol)
                                break
                    elematoms=[allelems[i] for i in wholemol]
                    #print("wholemol:", wholemol)
                    #print("elematoms:", elematoms)
                    atomlist=atomlist+wholemol
                    #print(len(atomlist))
                    #print(atomlist)
    atomlist = np.unique(atomlist).tolist()
    return atomlist

#Grab 2 total energies from list of ORCA outputfiles (basenames), e.g. VIEs.
def grab_energies_output_ORCA(inpfiles):
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

def grab_energies_output_xtb(xtbmethod, inpfiles):
    # Dictionaries to  hold VIEs. Currently not keeping track of total energies
    #For xTB each calculation in separate dir so we go in and out of dir
    AsnapsABenergy = {}
    BsnapsABenergy = {}
    AllsnapsABenergy = {}
    for snap in inpfiles:
        dir=snap.split('.')[0]
        basename = snap.split('.')[0]
        os.chdir(dir)
        os.listdir('.')
        snapbase=snap.split('_')[0]
        if 'GFN' in xtbmethod.upper():
            #Names of outputfiles given in run_gfnxtb_SPVIE_multiproc
            outfileA = basename+'_StateA.out'
            outfileB = basename+'_StateB.out'
            energyA=xtbfinalenergygrab(outfileA)
            energyB = xtbfinalenergygrab(outfileB)
            VIP=(energyB-energyA)*constants.hartoeV
        elif 'VIP' in xtbmethod.upper():
            outfile = snap.replace('.xyz', '.out')
            VIP = xtbVIPgrab(outfile)
        else:
            print("Unknown xtboption")
            exit()
        if 'snapA' in snapbase:
            AsnapsABenergy[snapbase]=VIP
            AllsnapsABenergy[snapbase] = VIP
        elif 'snapB' in snapbase:
            BsnapsABenergy[snapbase]=VIP
            AllsnapsABenergy[snapbase]=VIP
        os.chdir('..')
    return AllsnapsABenergy, AsnapsABenergy, BsnapsABenergy

def print_solvshell_header(version, progdir):
    print(BC.WARNING,"--------------------------------------------------",BC.END)
    print(BC.WARNING,"--------------------------------------------------",BC.END)
    print(BC.WARNING,BC.BOLD,"SOLVSHELL version", version,BC.END)
    print("SOLVSHELL dir:", progdir)
    print(BC.WARNING,"--------------------------------------------------",BC.END)
    print(BC.WARNING,"--------------------------------------------------",BC.END)

def print_solvshell_footer():
    print(BC.WARNING,"--------------------------------------------------",BC.END)
    print(BC.OKGREEN,BC.BOLD,"SOLVSHELL END OF OUTPUT",BC.END)
    print(BC.WARNING,"--------------------------------------------------",BC.END)




def create_AB_inputfiles_xtb(solute_atoms, solvent_atoms, solvsphere, snapshots,
                                     solventunitcharges, identifier,shell=None):
    xtb_xyzfiles = []
    for fragfile in snapshots:
        name = fragfile.split('.')[0]
        elems, coords = read_fragfile_xyz(fragfile)
        #QM and MM region coords and elems lists modified based if solvshell == True
        allatoms=solvsphere.allatoms
        qmatoms=solute_atoms
        mmatoms=solvent_atoms
        #Getting a QM solvent shell
        if shell!=None:
            secondsA = time.time()
            solute_elems =[elems[i] for i in solute_atoms]
            solute_coords =[coords[i] for i in solute_atoms]
            #TODO: Look into get_solvshell more regarding speed.
            #TODO: Both lists vs. numpy array. And the recursion depth
            #Run this in parallel ???
            solvshell=get_solvshell(solvsphere, elems,coords,shell,solute_elems,solute_coords,settings_solvation.scale,settings_solvation.tol)
            qmatoms=qmatoms+solvshell
            mmatoms=listdiff(allatoms,qmatoms)
            secondsB = time.time()
            #print("Execution time (seconds):", secondsB-secondsA)

        qmregion_elems=[elems[i] for i in qmatoms]
        qmregion_coords=[coords[i] for i in qmatoms]
        mmregion_elems=[elems[i] for i in mmatoms]
        mmregion_coords=[coords[i] for i in mmatoms]
        if len(mmatoms)+len(qmatoms) != len(coords):
            print("MM atoms: {} and QM atoms: {} . Sum: {} . Differs from total number {}".format(len(mmatoms),len(qmatoms), len(mmatoms)+len(qmatoms), len(coords)))
            print("Exiting...")
            exit()
        # Create XYZ file containing solute coordinates and point to pointchargefile containing solvent coordinates and charges
        #Possible: Write charge and mult to XYZ file header
        create_xtb_pcfile(name+identifier, mmregion_elems, mmregion_coords, solventunitcharges)
        write_xyzfile(qmregion_elems, qmregion_coords, name+identifier)
        xtb_xyzfiles.append(name+identifier + '.xyz')
    return xtb_xyzfiles


#Create ORCA inputfiles where state A and B (e.g. VIE calc) are calculated in same job using $new_job
def create_AB_inputfiles_ORCA(solute_atoms, solvent_atoms, solvsphere, snapshots, orcasimpleinput,
                              orcablockinput, solventunitcharges, identifier,shell=None, bulkcorr=False, solvbasis=''):
    snaphotinpfiles = []
    for fragfile in snapshots:
        name = fragfile.split('.')[0]
        elems, coords = read_fragfile_xyz(fragfile)
        #QM and MM region coords and elems lists modified based if solvshell == True
        allatoms=solvsphere.allatoms
        qmatoms=solute_atoms
        mmatoms=solvent_atoms
        #Getting a QM solvent shell
        if shell!=None:
            secondsA = time.time()
            solute_elems =[elems[i] for i in solute_atoms]
            solute_coords =[coords[i] for i in solute_atoms]
            #TODO: Look into solvshell more regarding speed.
            #TODO: Both lists vs. numpy array. And the recursion depth
            solvshell=get_solvshell(solvsphere, elems,coords,shell,solute_elems,solute_coords,settings_solvation.scale,settings_solvation.tol)
            qmatoms=qmatoms+solvshell
            mmatoms=listdiff(allatoms,qmatoms)
            secondsB = time.time()
            print("Excution time (seconds):", secondsB-secondsA)

        qmregion_elems=[elems[i] for i in qmatoms]
        qmregion_coords=[coords[i] for i in qmatoms]
        mmregion_elems=[elems[i] for i in mmatoms]
        mmregion_coords=[coords[i] for i in mmatoms]
        if len(mmatoms)+len(qmatoms) != len(coords):
            print("MM atoms: {} and QM atoms: {} . Sum: {} . Differs from total number {}".format(len(mmatoms),len(qmatoms), len(mmatoms)+len(qmatoms), len(coords)))
            print("Exiting...")
            exit()
        # Create ORCA inputfile containing solute coordinates and point to pointchargefile containing solvent coordinates and charges
        inpname_AB = fragfile.split('.')[0] + '_StateAB'+identifier
        if bulkcorr==True:
            create_orca_pcfile_solv(name+identifier, mmregion_elems, mmregion_coords, solventunitcharges, bulkcorr)
            create_orca_inputVIEcomp_pc(name+identifier, inpname_AB, qmregion_elems, qmregion_coords, orcasimpleinput, orcablockinput,
                                    solvsphere.ChargeA, solvsphere.MultA, solvsphere.ChargeB, solvsphere.MultB, solvsphere.soluteatomsA, '')
        elif 'Gas' in identifier:
            create_orca_inputVIEcomp_gas(name+identifier, inpname_AB, qmregion_elems, qmregion_coords, orcasimpleinput, orcablockinput,
                                    solvsphere.ChargeA, solvsphere.MultA, solvsphere.ChargeB, solvsphere.MultB)
        else:
            create_orca_pcfile_solv(name+identifier, mmregion_elems, mmregion_coords, solventunitcharges)
            create_orca_inputVIEcomp_pc(name+identifier, inpname_AB, qmregion_elems, qmregion_coords, orcasimpleinput, orcablockinput,
                                    solvsphere.ChargeA, solvsphere.MultA, solvsphere.ChargeB, solvsphere.MultB, solvsphere.soluteatomsA, solvbasis)

        snaphotinpfiles.append(inpname_AB + '.inp')
    return snaphotinpfiles



#Create ORCA inputfiles where state A and B (e.g. VIE calc) are calculation in same job using $new_job
# OLD: TO BE DELETED
def create_AB_inputfiles_old(solvsphere, snapshotsA, snapshotsB, orcasimpleinput, orcablockinput, solventunitcharges, bulkcorr=False):
    snaphotinpfiles = []
    for X in ['A','B']:
        if X=='A':
            snapshots=snapshotsA
            solutetypes=solvsphere.solutetypesA
        elif X=='B':
            snapshots = snapshotsB
            solutetypes = solvsphere.solutetypesB
        for fragfile in snapshots:
            name = fragfile.split('.')[0]
            elems, coords = read_fragfile_xyz(fragfile)
            # Grabbing solute elems and coords
            solute_elems = elems[0:len(solutetypes)]
            solute_coords = coords[0:len(solutetypes)]
            # Grabbing solvent elems and coords
            solvent_elems = elems[len(solutetypes):]
            solvent_coords = coords[len(solutetypes):]
            # Create ORCA inputfile containing solute coordinates and point to pointchargefile containing solvent coordinates and charges
            if bulkcorr==True:
                inpname_AB = fragfile.split('.')[0] + '_StateAB_Bulk'
                create_orca_pcfile_solv(name, solvent_elems, solvent_coords, solventunitcharges,bulkcorr)
            else:
                inpname_AB = fragfile.split('.')[0] + '_StateAB_LL'
                create_orca_pcfile_solv(name, solvent_elems, solvent_coords, solventunitcharges)
            create_orca_inputVIE_pc(name, inpname_AB, solute_elems, solute_coords, orcasimpleinput, orcablockinput, solvsphere.ChargeA, solvsphere.MultA, solvsphere.ChargeB, solvsphere.MultB)
            snaphotinpfiles.append(inpname_AB + '.inp')
    return snaphotinpfiles


def read_md_variables_fileAB(varfile):
    with open(varfile) as vfile:
        for line in vfile:
            if 'snapslist' in line:
                snapslist = [i.replace('{', '').replace('}', '') for i in line.split()[2:]]
            if 'snapshotsA' in line:
                snapshotsA = [i.replace('{', '').replace('}', '') for i in line.split()[2:]]
                snapshotsA = [i.replace('.c','') for i in snapshotsA]
            if 'snapshotsB' in line:
                snapshotsB = [i.replace('{', '').replace('}', '') for i in line.split()[2:]]
                snapshotsB = [i.replace('.c', '') for i in snapshotsB]
            if 'chargeA' in line:
                chargeA=int(line.split()[-1])
            if 'chargeB' in line:
                chargeB=int(line.split()[-1])
            if 'multA' in line:
                multA=int(line.split()[-1])
            if 'multB' in line:
                multB=int(line.split()[-1])
            if 'solvwitht' in line:
                solventtypes=[i.replace('{','').replace('}','') for i in line.split()[2:]]
            if 'solutetypesA' in line:
                solutetypesA=[i.replace('{','').replace('}','') for i in line.split()[2:]]
            if 'solutetypesB' in line:
                solutetypesB=[i.replace('{','').replace('}','') for i in line.split()[2:]]
            if 'numatomsoluteA' in line:
                numatomsoluteA=int(line.split()[-1])
            if 'numatomsoluteB' in line:
                numatomsoluteB=int(line.split()[-1])
            if 'solvtype' in line:
                solvtype = line.split()[-1]
            #TODO: File should contain information about number of solvent atoms. To be added to Chemshell MD file

    system = SystemAB("Solvsphere", chargeA, multA, chargeB, multB, solutetypesA,
                      solutetypesB, solventtypes, snapslist, snapshotsA, snapshotsB, solvtype )
    return system

def read_md_variables_fileA(varfile):
    with open(varfile) as vfile:
        for line in vfile:
            if 'snapslist' in line:
                snapslist = [i.replace('{', '').replace('}', '') for i in line.split()[2:]]
            if 'snapshotsA' in line:
                snapshotsA = [i.replace('{', '').replace('}', '') for i in line.split()[2:]]
                snapshotsA = [i.replace('.c','') for i in snapshotsA]
            if 'chargeA' in line:
                chargeA=int(line.split()[-1])
            if 'multA' in line:
                multA=int(line.split()[-1])
            if 'solvwitht' in line:
                solventtypes=[i.replace('{','').replace('}','') for i in line.split()[2:]]
            if 'solutetypesA' in line:
                solutetypesA=[i.replace('{','').replace('}','') for i in line.split()[2:]]
            if 'numatomsoluteA' in line:
                numatomsoluteA=int(line.split()[-1])
            if 'solvtype' in line:
                solvtype = line.split()[-1]
            #TODO: File should contain information about number of solvent atoms. To be added to Chemshell MD file

    system = SystemA("Solvsphere", chargeA, multA, solutetypesA, solventtypes, snapslist, snapshotsA, solvtype )
    return system

#Read Tcl-Chemshell fragment file and grab elems and coords. Coordinates converted from Bohr to Angstrom
def read_fragfile_xyz(fragfile):
    #removing extension from fragfile name if present and then adding back.
    pathtofragfile=fragfile.split('.')[0]+'.c'
    coords=[]
    elems=[]
    #TODO: Change elems and coords to numpy array instead
    grabcoords=False
    with open(pathtofragfile) as ffile:
        for line in ffile:
            if 'block = connectivity' in line:
                grabcoords=False
            if grabcoords==True:
                coords.append([float(i)*constants.bohr2ang for i in line.split()[1:]])
                elems.append(line.split()[0])
            if 'block = coordinates records ' in line:
                #numatoms=int(line.split()[-1])
                grabcoords=True
    return elems,coords

# Find representative snapshots
# Options: 'num', 'numalt', 'cut'
# num: Chooses X shapshots with energies closest to average
# numalt: Chooses X closest snapshots but alternates greater/less than average
# cut: Chooses all snapshots within X kcal/mol of average
# "estimnum" should be value of X in above options
def repsnaplist (estimtype, estimnum, IPdict):
    ave = statistics.mean(list(IPdict.values()))
    stdev = statistics.stdev(list(IPdict.values()))
    #Return dictionary of repsnaps and values
    repsnaps_dict = {}
    temp_dict = {}
    if estimtype == "num":
        for i in IPdict.items():
            #Creating temp dict with absolute deviations of values from ave
            temp_dict[i[0]] = abs(i[1]-ave)
        #Sorting list from tempdict by ascending abs deviation
        sorted_tempdict_list = sorted(temp_dict.items(), key=lambda x: x[1])
        #Grabbing first X (estimnum) entries from sorted_tempdict_list
        tempentries=sorted_tempdict_list[0:estimnum]
        #grabbing those corresponding entries from original dict
        for j in IPdict.items():
            for k in tempentries:
                if j[0] == k[0]:
                    repsnaps_dict[j[0]] = j[1]
    elif estimtype == "cut":
        # Cut threshold for cut option in eV. If estimtype=cut then cut_threshold=estimnum
        cut_threshold = estimnum
        for i in list(IPdict.items()):
            if abs(i[1]-ave) < cut_threshold:
                repsnaps_dict[i[0]]=i[1]
    elif estimtype == "numalt":
        print("not coded yet")
        exit()
    return repsnaps_dict


def print_redox_output_state(state, solvsphere, orca_LL, orca_HL, snapshots, ave_traj, stdev_traj, repsnap_ave_traj,
                                 repsnap_stdev_traj, repsnaplist, Bulk_ave_traj, Bulk_stdev_traj, Bulkcorr_mean,
                                 SRPol_ave_traj, SRPol_stdev_traj, SRPolcorr_mean, LRPol_ave_traj_R1, LRPol_ave_traj_R2,
                                 LRPol_stdev_traj_R1, LRPol_stdev_traj_R2, LRPolcorr_mean, gas_VIE_LL, gas_VIE_HL):
    if state=="AB":
        Etype="AIE"
    else:
        Etype="VIE"
    print("Low-level theory:", orca_LL)
    print("LL-{} for {} snapshots: {} ± {}".format(Etype, len(snapshots), ave_traj, stdev_traj))
    print("Repsnaps Traj{} average: {} ± {}".format(state,repsnap_ave_traj, repsnap_stdev_traj))
    print("Repsnaps list:", repsnaplist)
    blankline()
    print("Bulk LL-{} for {} repsnapshots: {} ± {}".format(Etype, len(repsnaplist), Bulk_ave_traj, Bulk_stdev_traj))
    print("Bulk correction for {} repsnapshots: {}".format(len(repsnaplist), Bulkcorr_mean))
    print("SRPol LL-{} for {} repsnapshots: {} ± {}".format(Etype, len(repsnaplist), SRPol_ave_traj, SRPol_stdev_traj))
    print("SRPol correction for {} repsnapshots: {}".format(len(repsnaplist), SRPolcorr_mean))
    print("LRPol-Region1 LL-{} for {} repsnapshots: {} ± {}".format(Etype, len(repsnaplist), LRPol_ave_traj_R1, LRPol_stdev_traj_R1))
    print("LRPol-Region2 LL-{} for {} repsnapshots: {} ± {}".format(Etype, len(repsnaplist), LRPol_ave_traj_R2, LRPol_stdev_traj_R2))
    print("LRPol correction for {} repsnapshots: {}".format(len(repsnaplist), LRPolcorr_mean))
    sumofcorrections = Bulkcorr_mean + SRPolcorr_mean + LRPolcorr_mean
    final_LL_VIE = ave_traj + sumofcorrections
    print("Sum of corrections:", sumofcorrections)
    print("Final {} at LL-level: {} ± {}".format(Etype, final_LL_VIE, stdev_traj))
    blankline()
    print("High-level theory:", orca_HL)
    print("Gas LL-{}: {} eV".format(Etype, gas_VIE_LL))
    print("Gas HL-{}: {} eV".format(Etype, gas_VIE_HL))
    gascorrection=gas_VIE_HL-gas_VIE_LL
    print("Gas correction: {} eV".format(gascorrection))
    print("DeltaDeltaGsolv : {} eV".format(ave_traj-gas_VIE_LL+sumofcorrections))
    blankline()
    print("Final {} at HL-level: {}".format(Etype, final_LL_VIE+gascorrection))

    if solvsphere.ChargeA > solvsphere.ChargeB:
        print("A -> B  Reduction")
        HLreductionenergy = gascorrection + final_LL_VIE
    else:
        print("A -> B  Oxidation")
        HLreductionenergy = -1*(gascorrection+final_LL_VIE)
    print("High-level reduction energy is : {} eV.".format(HLreductionenergy))

    if state == "AB":
        print("SHE: 4.28 V")
        print("Final Redox Potential (vs. SHE) at HL-level: {}".format(-1*HLreductionenergy-constants.SHE))

    return


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