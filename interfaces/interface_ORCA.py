import subprocess as sp
import module_coords
from functions_general import blankline,insert_line_into_file,BC,print_time_rel
import functions_elstructure
import constants
import multiprocessing as mp
import numpy as np
import os
import settings_ash
import time

#ORCA Theory object. Fragment object is optional. Only used for single-points.
class ORCATheory:
    def __init__(self, orcadir=None, fragment=None, charge=None, mult=None, orcasimpleinput='', printlevel=2, extrabasisatoms=None, extrabasis=None,
                 orcablocks='', extraline='', brokensym=None, HSmult=None, atomstoflip=None, nprocs=1, label=None, moreadfile=None, autostart=True, propertyblock=None):

        if orcadir is None:
            print(BC.WARNING, "No orcadir argument passed to ORCATheory. Attempting to find orcadir variable inside settings_ash", BC.END)
            try:
                self.orcadir=settings_ash.settings_dict["orcadir"]
            except:
                print(BC.FAIL,"Found no orcadir variable in settings_ash module either. Exiting.",BC.END)
                exit()
        else:
            self.orcadir = orcadir

        #Label to distinguish different ORCA objects
        self.label=label

        #Create inputfile with generic name
        self.filename="orca"

        #MOREAD-file
        self.moreadfile=moreadfile
        #Autostart
        self.autostart=autostart

        #Using orcadir to set LD_LIBRARY_PATH
        old = os.environ.get("LD_LIBRARY_PATH")
        if old:
            os.environ["LD_LIBRARY_PATH"] = self.orcadir + ":" + old
        else:
            os.environ["LD_LIBRARY_PATH"] = self.orcadir
        #os.environ['LD_LIBRARY_PATH'] = orcadir + ':$LD_LIBRARY_PATH'

        #Printlevel
        self.printlevel=printlevel

        #Setting nprocs of object
        self.nprocs=nprocs

        #Property block. Added after coordinates unless None
        self.propertyblock=propertyblock

        if fragment != None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge!=None:
            self.charge=int(charge)
        else:
            self.charge=None
        if mult!=None:
            self.mult=int(mult)
        else:
            self.charge=None
        
        #Adding NoAutostart keyword to extraline if requested
        if self.autostart == False:
            self.extraline=extraline+"\n! Noautostart"
        else:
            self.extraline=extraline
        
        self.orcasimpleinput=orcasimpleinput
        self.orcablocks=orcablocks

        #BROKEN SYM OPTIONS
        self.brokensym=brokensym
        self.HSmult=HSmult
        if type(atomstoflip) is int:
            print(BC.FAIL,"Error: atomstoflip should be list of integers (e.g. [0] or [2,3,5]), not a single integer.", BC.END)
            exit(1)
        if atomstoflip != None:
            self.atomstoflip=atomstoflip
        else:
            self.atomstoflip=[]
        #Extrabasis
        if extrabasisatoms != None:
            self.extrabasisatoms=extrabasisatoms
            self.extrabasis=extrabasis
        else:
            self.extrabasisatoms=[]
            self.extrabasis=""
        
        
        # self.qmatoms need to be set for Flipspin to work for QM/MM job.
        #Overwritten by QMMMtheory, used in Flip-spin
        self.qmatoms=[]
            
        if self.printlevel >=2:
            print("")
            print("Creating ORCA object")
            print("ORCA dir:", self.orcadir)
            #if molcrys then there is not charge and mult available
            #print("Charge: {} Mult: {}".format(self.charge,self.mult))
            print(self.orcasimpleinput)
            print(self.orcablocks)
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old ORCA files")
        list_files=[]
        list_files.append(self.filename + '.gbw')
        list_files.append(self.filename + '.ges')
        list_files.append(self.filename + '.prop')
        list_files.append(self.filename + '.uco')
        list_files.append(self.filename + '_property.txt')
        list_files.append(self.filename + '.inp')
        list_files.append(self.filename + '.out')
        list_files.append(self.filename + '.engrad')
        for file in list_files:
            try:
                os.remove(file)
            except:
                pass
        # os.remove(self.filename + '.out')
        try:
            for tmpfile in glob.glob("self.filename*tmp"):
                os.remove(tmpfile)
        except:
            pass
    
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, nprocs=None, label=None ):
        module_init_time=time.time()
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING ORCA INTERFACE-------------", BC.END)
        #Coords provided to run or else taken from initialization.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems=self.elems
            else:
                qm_elems = elems

        #If QM/MM then extrabasisatoms and atomstoflip have to be updated
        if len(self.qmatoms) != 0:
            #extrabasisatomindices if QM/MM
            print("QM atoms :", self.qmatoms)
            qmatoms_extrabasis=[self.qmatoms.index(i) for i in self.extrabasisatoms]
            #new QM-region indices for atomstoflip if QM/MM
            qmatomstoflip=[self.qmatoms.index(i) for i in self.atomstoflip]
        else:
            qmatomstoflip=self.atomstoflip
            qmatoms_extrabasis=self.extrabasisatoms
        
        if nprocs==None:
            nprocs=self.nprocs
        print("Running ORCA object with {} cores available".format(nprocs))
        print("Job label:", label)


        print("Creating inputfile:", self.filename+'.inp')
        print("ORCA input:")
        print(self.orcasimpleinput)
        print(self.extraline)
        print(self.orcablocks)
        print("Charge: {}  Mult: {}".format(self.charge, self.mult))
        #Printing extra options chosen:
        if self.brokensym==True:
            print("Brokensymmetry SpinFlipping on! HSmult: {}.".format(self.HSmult))
            for flipatom,qmflipatom in zip(self.atomstoflip,qmatomstoflip):
                print("Flipping atom: {} QMregionindex: {} Element: {}".format(flipatom, qmflipatom, qm_elems[qmflipatom]))
        if self.extrabasis != "":
            print("Using extra basis ({}) on QM-region indices : {}".format(self.extrabasis,qmatoms_extrabasis))

        if PC==True:
            print("Pointcharge embedding is on!")
            create_orca_pcfile(self.filename, current_MM_coords, MMcharges)
            if self.brokensym == True:
                create_orca_input_pc(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        self.charge, self.mult, extraline=self.extraline, HSmult=self.HSmult, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                     atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock)
            else:
                create_orca_input_pc(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        self.charge, self.mult, extraline=self.extraline, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                        extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock)
        else:
            if self.brokensym == True:
                create_orca_input_plain(self.filename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        self.charge,self.mult, extraline=self.extraline, HSmult=self.HSmult, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                     atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock)
            else:
                create_orca_input_plain(self.filename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        self.charge,self.mult, extraline=self.extraline, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                        extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock)

        #Run inputfile using ORCA parallelization. Take nprocs argument.
        #print(BC.OKGREEN, "------------Running ORCA calculation-------------", BC.END)
        print(BC.OKGREEN, "ORCA Calculation started.", BC.END)
        # Doing gradient or not. Disabling, doing above instead.
        #if Grad == True:
        #    run_orca_SP_ORCApar(self.orcadir, self.filename + '.inp', nprocs=nprocs, Grad=True)
        #else:
        run_orca_SP_ORCApar(self.orcadir, self.filename + '.inp', nprocs=nprocs)
        print(BC.OKGREEN, "ORCA Calculation done.", BC.END)

        #Now that we have possibly run a BS-DFT calculation, turning Brokensym off for future calcs (opt, restart, etc.)
        # using this theory object
        #TODO: Possibly use different flag for this???
        print("ORCA Flipspin calculation done. Now turning off brokensym in ORCA object for possible future calculations")
        self.brokensym=False

        #Check if finished. Grab energy and gradient
        outfile=self.filename+'.out'
        engradfile=self.filename+'.engrad'
        pcgradfile=self.filename+'.pcgrad'
        if checkORCAfinished(outfile) == True:
            self.energy=ORCAfinalenergygrab(outfile)
            print("ORCA energy:", self.energy)

            if Grad == True:
                self.grad=ORCAgradientgrab(engradfile)
                if PC == True:
                    #Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                    self.pcgrad=ORCApcgradientgrab(pcgradfile)
                    print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                    print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2)
                    return self.energy, self.grad, self.pcgrad
                else:
                    print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                    print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2)
                    return self.energy, self.grad

            else:
                print("Single-point ORCA energy:", self.energy)
                print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2)
                return self.energy
        else:
            print(BC.FAIL,"Problem with ORCA run", BC.END)
            print(BC.OKBLUE,BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
            print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2)
            exit(1)










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
def run_orca_SP(list):
    orcadir=list[0]
    inpfile=list[1]
    print("Running inpfile", inpfile)
    #if Grad==True:
    #    with open(inpfile) as ifile:
    #        insert_line_into_file(inpfile, '!', '! Engrad')
    basename = inpfile.split('.')[0]
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([orcadir + '/orca', basename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

# Run ORCA single-point job using ORCA parallelization. Will add pal-block if nprocs >1.
# Takes possible Grad boolean argument.

def run_orca_SP_ORCApar(orcadir, inpfile, nprocs=1):
    #if Grad==True:
    #    with open(inpfile) as ifile:
    #        insert_line_into_file(inpfile, '!', '! Engrad')
    #Add pal block to inputfile before running. Adding after '!' line. Should work for regular, new_job and compound job.
    if nprocs>1:
        palstring='% pal nprocs {} end'.format(nprocs)
        with open(inpfile) as ifile:
            insert_line_into_file(inpfile, '!', palstring, Once=True )
    #basename = inpfile.split('.')[0]
    basename = inpfile.replace('.inp','')
    with open(basename+'.out', 'w') as ofile:
        #process = sp.run([orcadir + '/orca', basename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
        process = sp.run([orcadir + '/orca', inpfile], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#Check if ORCA finished.
#Todo: Use reverse-read instead to speed up?
def checkORCAfinished(file):
    with open(file) as f:
        for line in f:
            
            if 'SCF CONVERGED AFTER' in line:
                iter=line.split()[-3]
                print("ORCA converged in {} iterations".format(iter))
            if 'TOTAL RUN TIME:' in line:
                return True

#Grab Final single point energy. Ignoring possible encoding errors in file
def ORCAfinalenergygrab(file, errors='ignore'):
    with open(file) as f:
        for line in f:
            if 'FINAL SINGLE POINT ENERGY' in line:
                if "Wavefunction not fully converged!" in line:
                    print("ORCA WF not fully converged!")
                    print("Not using energy. Modify ORCA settings")
                    exit()
                else:
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

#Get reference energy and correlation energy from a single post-HF calculation
#Support regular CC, DLPNO-CC, CC-12, DLPNO-CC-F12
#Note: CC-12 untested
def grab_HF_and_corr_energies(file, DLPNO=False, F12=False):
    edict = {}
    with open(file) as f:
        for line in f:
            #Reference energy found in CC output. To be made more general. Works for CC and DLPNO-CC
            #if 'Reference energy                           ...' in line:
            if F12 is True:
                #F12 has a basis set correction for HF energy
                if 'Corrected 0th order energy                 ...' in line:
                    HF_energy=float(line.split()[-1])
                    edict['HF'] = HF_energy             
            else:    
                if 'E(0)                                       ...' in line:
                    HF_energy=float(line.split()[-1])
                    edict['HF'] = HF_energy
                    

            if DLPNO is True:
                if F12 is True:
                    if 'Final F12 correlation energy               ...' in line:
                        CCSDcorr_energy=float(line.split()[-1])
                        edict['CCSD_corr'] = CCSDcorr_energy
                        edict['full_corr'] = CCSDcorr_energy
                else:    
                    if 'E(CORR)(corrected)                         ...' in line:
                        CCSDcorr_energy=float(line.split()[-1])
                        edict['CCSD_corr'] = CCSDcorr_energy
                        edict['full_corr'] = CCSDcorr_energy
            else:
                if F12 is True:
                    if 'Final F12 correlation energy               ...' in line:
                        CCSDcorr_energy=float(line.split()[-1])
                        edict['CCSD_corr'] = CCSDcorr_energy
                        edict['full_corr'] = CCSDcorr_energy
                else:        
                    if 'E(CORR)                                    ...' in line:
                        CCSDcorr_energy=float(line.split()[-1])
                        edict['CCSD_corr'] = CCSDcorr_energy
                        edict['full_corr'] = CCSDcorr_energy
                        

            if DLPNO is True:
                if 'Triples Correction (T)                     ...' in line:
                    CCSDTcorr_energy=float(line.split()[-1])
                    edict['CCSD(T)_corr'] = CCSDTcorr_energy
                    edict['full_corr'] = CCSDcorr_energy+CCSDTcorr_energy
            else:
                if 'Scaled triples correction (T)              ...' in line:
                    CCSDTcorr_energy=float(line.split()[-1])
                    edict['CCSD(T)_corr'] = CCSDTcorr_energy
                    edict['full_corr'] = CCSDcorr_energy+CCSDTcorr_energy
            if 'T1 diagnostic                              ...' in line:
                T1diag = float(line.split()[-1])
                edict['T1diag'] = T1diag
    return edict


#Grab XES state energies and intensities from ORCA output
def xesgrab(file):
    xesenergies=[]
    #
    intensities=[]
    xesgrab=False
    
    with open(file) as f:
        for line in f:
            if xesgrab==True:
                if 'Getting memory' in line:
                    xesgrab=False
                if "->" in line:
                    xesenergies.append(float(line.split()[4]))
                    intensities.append(float(line.split()[8]))
            if "COMBINED ELECTRIC DIPOLE + MAGNETIC DIPOLE + ELECTRIC QUADRUPOLE X-RAY EMISSION SPECTRUM" in line:
                xesgrab=True
    return xesenergies,intensities

#Grab TDDFT states from ORCA output
def tddftgrab(file):
    tddftstates=[]
    tddft=True
    tddftgrab=False
    if tddft==True:
        with open(file) as f:
            for line in f:
                if tddftgrab==True:
                    if 'STATE' in line:
                        if 'eV' in line:
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


#TODO: Limited, kept for now for module_PES compatibility. Better version below
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



def MolecularOrbitalGrab(file):
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
    
    if hftyp != "RHF":
        Openshell=True
    else:
        Openshell=False

    #Final dict
    MOdict= {"occ_alpha":bands_alpha, "occ_beta":bands_alpha, "unocc_alpha":virtbands_a, "unocc_beta":virtbands_b, "Openshell":Openshell}
    return MOdict







#Grab <S**2> expectation values from outputfile
def grab_spin_expect_values_ORCA(file):
    S2value=None
    with open(file) as f:
        for line in f:
            #Note: if flip-spin job(line appears twice), then we take the latter
            if 'Expectation value of <S**2>' in line:
                S2value=float(line.split()[-1])
        return S2value


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
    #print("hessatoms", hessatoms)
    #print("masses ", masses)
    #print("elems ", elems)
    #print("coords", coords)
    #print(len(elems))
    #print(len(coords))
    #print(len(hessatoms))
    #print(len(masses))
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


def read_ORCA_Hessian(hessfile):
    hessian = Hessgrab(hessfile)
    elems,coords = grabcoordsfromhessfile(hessfile)
    masses, elems, numatoms = masselemgrab(hessfile)
    
    return hessian, elems, coords, masses


#Grab frequencies from ORCA-Hessian file
def ORCAfrequenciesgrab(hessfile):
    freqs=[]
    grab=False
    with open(hessfile) as hfile:
        for line in hfile:
            if grab is True:
                if len(line.split()) > 1:
                    freqs.append(float(line.split()[-1]))
            if '$vibrational_frequencies' in line:
                grab=True
            if '$normal_modes' in line:
                grab=False
    return freqs

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
def create_orca_input_pc(name,elems,coords,orcasimpleinput,orcablockinput,charge,mult, Grad=False, extraline='',
                         HSmult=None, atomstoflip=None, Hessian=False, extrabasisatoms=None, extrabasis=None, moreadfile=None, propertyblock=None):
    if extrabasisatoms is None:
        extrabasisatoms=[]
    pcfile=name+'.pc'
    with open(name+'.inp', 'w') as orcafile:
        orcafile.write(orcasimpleinput+'\n')
        if extraline != '':
            orcafile.write(extraline + '\n')
        if Grad == True:
            orcafile.write('! Engrad' + '\n')
        if Hessian == True:
            orcafile.write('! Freq' + '\n')
        if moreadfile is not None:
            print("MOREAD option active. Will read orbitals from file:", moreadfile)
            orcafile.write('! MOREAD' + '\n')
            orcafile.write('%moinp \"{}\"'.format(moreadfile) + '\n')
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
        #Writing coordinates. Adding extrabasis keyword for atom if option active
        for i,(el,c) in enumerate(zip(elems,coords)):
            if i in extrabasisatoms:
                orcafile.write('{} {} {} {} newgto \"{}\" end\n'.format(el,c[0], c[1], c[2], extrabasis))                
            else:
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
        orcafile.write('*\n')
        if propertyblock != None:
            orcafile.write(propertyblock)
#Create simple ORCA inputfile from elems,coords, input, charge, mult,pointcharges
#Allows for extraline that could be another '!' line or block-inputline.

def create_orca_input_plain(name,elems,coords,orcasimpleinput,orcablockinput,charge,mult, Grad=False, Hessian=False, extraline='',
                            HSmult=None, atomstoflip=None, extrabasis=None, extrabasisatoms=None, moreadfile=None, propertyblock=None):
    if extrabasisatoms is None:
        extrabasisatoms=[]
    
    with open(name+'.inp', 'w') as orcafile:
        orcafile.write(orcasimpleinput+'\n')
        if extraline != '':
            orcafile.write(extraline + '\n')
        if Grad == True:
            orcafile.write('! Engrad' + '\n')
        if Hessian == True:
            orcafile.write('! Freq' + '\n')
        if moreadfile is not None:
            print("MOREAD option active. Will read orbitals from file:", moreadfile)
            orcafile.write('! MOREAD' + '\n')
            orcafile.write('%moinp \"{}\"'.format(moreadfile) + '\n')
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

        for i,(el,c) in enumerate(zip(elems,coords)):
            if i in extrabasisatoms:
                orcafile.write('{} {} {} {} newgto \"{}\" end\n'.format(el,c[0], c[1], c[2], extrabasis))                
            else:
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
        orcafile.write('*\n')
        if propertyblock != None:
            orcafile.write(propertyblock)
# Create ORCA pointcharge file based on provided list of elems and coords (MM region elems and coords)
# and list of point charges of MM atoms
def create_orca_pcfile(name,coords,listofcharges):
    with open(name+'.pc', 'w') as pcfile:
        pcfile.write(str(len(listofcharges))+'\n')
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
    elif chargemodel=='DDEC6':
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
                    #Setting charges list to zero in case of multiple charge-tables. Means we grab second table
                    charges=[]
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
                    #Setting charges list to zero in case of multiple charge-tables. Means we grab second table
                    charges=[]
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
                    #Setting charges list to zero in case of multiple charge-tables. Means we grab second table
                    charges=[]
                    grab=True
        print("Hirshfeld charges :", charges)
        atomicnumbers=module_coords.elemstonuccharges(elems)
        charges = functions_elstructure.calc_cm5(atomicnumbers, coords, charges)
        print("CM5 charges :", list(charges))
    elif chargemodel == "Mulliken":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if 'Sum of atomic' in line:
                        grab=False
                    elif '------' not in line:
                        charges.append(float(line.split()[-1]))
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
                        charges.append(float(line.split()[-1]))
                if 'LOEWDIN ATOMIC CHARGES' in line:
                    grab=True
    elif chargemodel == "IAO":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if 'Sum of atomic' in line:
                        grab=False
                    elif '------' not in line:
                        if 'Warning' not in line:
                            print("line:", line)
                            charges.append(float(line.split()[-1]))
                if 'IAO PARTIAL CHARGES' in line:
                    grab=True
    else:
        print("Unknown chargemodel. Exiting...")
        exit()
    return charges


# Wrapper around interactive orca_plot
# Todo: add TDDFT difference density, natural orbitals, MDCI spin density?
def run_orca_plot(orcadir, filename, option, gridvalue=40,densityfilename=None, mo_operator=0, mo_number=None):
    # Always creating Cube file (5,7 option)
    #Always setting grid (4,gridvalue option)
    #Always choosing a plot (2,X) option:
    # Plot option in orca_plot
    if option=='density':
        plottype = 2
    elif option=='cisdensity':
        plottype = 2
    elif option=='spindensity':
        plottype = 3
    elif option=='cisspindensity':
        plottype = 3
    elif option=='mo':
        plottype = 1
    else:
        plottype = 1
    if option=='density' or option=='spindensity':
         p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                       input='5\n7\n4\n{}\n1\n{}\ny\n10\n11\n\n'.format(gridvalue, plottype), encoding='ascii')       
    elif option=='mo':
        p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                       input='5\n7\n4\n{}\n3\n{}\n2\n{}\n10\n11\n\n'.format(gridvalue,mo_operator,mo_number), encoding='ascii')
    #If plotting CIS/TDDFT density then we tell orca_plot explicity.
    elif option == 'cisdensity' or option == 'cisspindensity':
        p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                       input='5\n7\n4\n{}\n1\n{}\nn\n{}\n10\n11\n\n'.format(gridvalue, plottype,densityfilename), encoding='ascii')

    #print(p.returncode)
    
#Grab IPs from an EOM-IP calculation and also largest singles amplitudes. Approximation to Dyson norm.
def grabEOMIPs(file):
    IPs=[]
    final_singles_amplitudes=[]
    state_amplitudes=[]
    stateflag=False
    with open(file) as f:
        for line in f:
            if 'IROOT' in line:
                state_amplitudes=[]
                IP=float(line.split()[4])
                IPs.append(IP)
                stateflag=True
            if stateflag is True:
                if '-> x' in line:
                    if line.count("->") == 1:
                        amplitude=float(line.split()[0])
                        state_amplitudes.append(amplitude)
            if 'Percentage singles' in line:
                #Find dominant singles
                #print("state_amplitudes:", state_amplitudes)
                
                #if no singles amplitude found then more complicated transition. set to 0.0
                if len(state_amplitudes) >0:
                    largest=abs(max(state_amplitudes, key=abs))
                    final_singles_amplitudes.append(largest)
                else:
                    final_singles_amplitudes.append(0.0)
                state_amplitudes=[]
    assert len(IPs) == len(final_singles_amplitudes), "Something went wrong here"
    return IPs, final_singles_amplitudes

#Reading stability analysis from output. Returns true if stab-analysis good, otherwise falsee
#If no stability analysis present in output, then also return true
def check_stability_in_output(file):
    with open(file) as f:
        for line in f:
            if 'Stability Analysis indicates a stable HF/KS wave function.' in line:
                print("WF is stable")
                return True
            if 'Stability Analysis indicates an UNSTABLE HF/KS wave' in line:
                print("ORCA output:", line)
                print("ASH: WF is NOT stable. Check ORCA output for details.")
                return False
    return True


def MP2_natocc_grab(filename):
    natoccgrab=False
    natoccupations=[]
    with open(filename) as f:
        for line in f:
            if natoccgrab==True:
                if 'N' in line:
                    natoccupations.append(float(line.split()[-1]))
                if '***' in line:
                    natoccgrab=False
            if 'Natural Orbital Occupation Num' in line:
                natoccgrab=True
    return natoccupations




def SCF_FODocc_grab(filename):
    occgrab=False
    occupations=[]
    with open(filename) as f:
        for line in f:
            if occgrab==True:
                if '  NO   OCC' not in line:
                    if len(line) >5:
                        occupations.append(float(line.split()[1]))
                    if len(line) < 2 or ' SPIN DOWN' in line:
                        occgrab=False
                        return occupations
            if 'SPIN UP ORBITALS' in line:
                occgrab=True
    return natoccupations

def CASSCF_natocc_grab(filename):
    natoccgrab=False
    natoccupations=[]
    with open(filename) as f:
        for line in f:
            if natoccgrab==True:
                if len(line) >5:
                    natoccupations.append(float(line.split()[1]))
                if len(line) < 2 or '----' in line:
                    natoccgrab=False
                    return natoccupations
            if 'NO   OCC          E(Eh)            E(eV)' in line:
                natoccgrab=True
    return natoccupations

def QRO_occ_energies_grab(filename):
    occgrab=False
    occupations=[]
    qro_energies=[]
    with open(filename) as f:
        for line in f:
            if occgrab==True:
                if len(line) < 2 or '----' in line:
                    occgrab=False
                    return occupations,qro_energies
                if len(line) >5:
                    occ=line.split()[1][0]
                    occupations.append(float(occ))
                    qro_energies.append(float(line.split()[-4]))

            if 'Orbital Energies of Quasi-Restricted' in line:
                occgrab=True

def ICE_WF_size(filename):
    after_SD_numCFGs=0
    num_genCFGs=0
    with open(filename) as g:
        for line in g:
            if '# of configurations after S+D' in line:
                after_SD_numCFGs=int(line.split()[-1])
            if 'Selecting from the generated configurations  ...    # of configurations after Selection' in line:
                num_genCFGs=int(line.split()[-1])
            if 'Final CASSCF energy       :' in line:
                return num_genCFGs,after_SD_numCFGs


def grab_EFG_from_ORCA_output(filename):
    occgrab=False
    occupations=[]
    qro_energies=[]
    with open(filename) as f:
        for line in f:
            if ' V(Tot)' in line:
                efg_values=[float(line.split()[-3]),float(line.split()[-2]),float(line.split()[-1])]
                return efg_values