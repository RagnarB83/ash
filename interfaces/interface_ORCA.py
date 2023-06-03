import subprocess as sp
import os
import shutil
import time
import multiprocessing as mp
import numpy as np
import glob
import copy

import ash.modules.module_coords
from ash.functions.functions_general import ashexit,insert_line_into_file,BC,print_time_rel, print_line_with_mainheader, pygrep2, pygrep, search_list_of_lists_for_index,print_if_level
from ash.modules.module_singlepoint import Singlepoint
from ash.modules.module_coords import check_charge_mult
#from ash.functions.functions_elstructure import xdm_run, calc_cm5
import ash.functions.functions_elstructure
import ash.constants
import ash.settings_ash
import ash.functions.functions_parallel


#ORCA Theory object.
class ORCATheory:
    def __init__(self, orcadir=None, orcasimpleinput='', printlevel=2, basis_per_element=None, extrabasisatoms=None, extrabasis=None, TDDFT=False, TDDFTroots=5, FollowRoot=1,
                 orcablocks='', extraline='', first_iteration_input=None, brokensym=None, HSmult=None, atomstoflip=None, numcores=1, nprocs=None, label=None, 
                 moreadfile=None, moreadfile_always=False, bind_to_core_option=True, ignore_ORCA_error=False,
                 autostart=True, propertyblock=None, save_output_with_label=False, keep_each_run_output=False, print_population_analysis=False, filename="orca", check_for_errors=True, check_for_warnings=True,
                 fragment_indices=None, xdm=False, xdm_a1=None, xdm_a2=None, xdm_func=None):
        print_line_with_mainheader("ORCATheory initialization")

        #Indicate that this is a QMtheory
        self.theorytype="QM"

        #Making sure we have a working ORCA location
        print("Checking for ORCA location")
        self.orcadir = check_ORCA_location(orcadir)
        #Making sure ORCA binary works (and is not orca the screenreader)
        check_ORCAbinary(self.orcadir)
        #Checking OpenMPI
        if numcores != 1:
            print(f"ORCA parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version (for the ORCA version) is available in your environment")
            ash.functions.functions_parallel.check_OpenMPI()

        #Bind to core option when calling ORCA: i.e. execute: /path/to/orca file.inp "--bind-to none"
        #TODO: Default False; make True?
        self.bind_to_core_option=bind_to_core_option
        print("bind_to_core_option:", self.bind_to_core_option)

        #Checking if user added Opt, Freq keywords
        if ' OPT' in orcasimpleinput.upper() or ' FREQ' in orcasimpleinput.upper() :
            print(BC.FAIL,"Error. orcasimpleinput variable can not contain ORCA job-directives like: Opt, Freq, Numfreq", BC.END)
            print("String:", orcasimpleinput.upper())
            print("orcasimpleinput should only contain information on electronic-structure method (e.g. functional), basis set, grid, SCF convergence etc.")
            ashexit()

        #Whether to check ORCA outputfile for errors and warnings or not
        #Generally recommended. Could be disabled to speed up I/O a tiny bit
        self.check_for_errors=check_for_errors
        self.check_for_warnings=check_for_warnings

        #Counter for how often ORCATheory.run is called
        self.runcalls=0

        #Whether to keep the ORCA outputfile for each run as orca_runX.out
        self.keep_each_run_output=keep_each_run_output
        #Whether to save ORCA outputfile with given label
        if save_output_with_label is True and label is None:
            print("Error: save_output_with_label option requires a label keyword also")
            ashexit()
        else:
            self.save_output_with_label=save_output_with_label

        #Print population_analysis in each run
        self.print_population_analysis=print_population_analysis

        #Label to distinguish different ORCA objects
        self.label=label

        #Create inputfile with generic name
        self.filename=filename

        #Whether to exit ORCA if subprocess command faile
        self.ignore_ORCA_error=ignore_ORCA_error


        #MOREAD-file
        self.moreadfile=moreadfile
        self.moreadfile_always=moreadfile_always
        #Autostart
        self.autostart=autostart
        # Each ORCA calculation will save path to last GBW-file used in case we have switched directories
        #and we want to use last one
        self.path_to_last_gbwfile_used=None #default None

        #Printlevel
        self.printlevel=printlevel

        #TDDFT
        self.TDDFT=TDDFT
        self.TDDFTroots=TDDFTroots
        self.FollowRoot=FollowRoot

        #Setting numcores of object
        #NOTE: nprocs is deprecated but kept on for a bit
        if nprocs==None:
            self.numcores=numcores
        else:
            self.numcores=nprocs
        
        #Property block. Added after coordinates unless None
        self.propertyblock=propertyblock

        #Store optional properties of ORCA run job in a dict
        self.properties ={}

        
        #Adding NoAutostart keyword to extraline if requested
        if self.autostart == False:
            self.extraline=extraline+"\n! Noautostart"
        else:
            self.extraline=extraline
        
        #Inputfile definitions
        self.orcasimpleinput=orcasimpleinput
        self.orcablocks=orcablocks

        #Input-lines only for first run call
        if first_iteration_input != None:
            self.first_iteration_input = first_iteration_input
        else:
            self.first_iteration_input=""

        #BROKEN SYM OPTIONS
        self.brokensym=brokensym
        self.HSmult=HSmult
        if type(atomstoflip) is int:
            print(BC.FAIL,"Error: atomstoflip should be list of integers (e.g. [0] or [2,3,5]), not a single integer.", BC.END)
            ashexit()
        if atomstoflip != None:
            self.atomstoflip=atomstoflip
        else:
            self.atomstoflip=[]
        #Basis sets per element
        self.basis_per_element=basis_per_element
        if self.basis_per_element != None:
            print("Basis set dictionary for each element provided:", basis_per_element)
        #Extrabasis
        if extrabasisatoms != None:
            self.extrabasisatoms=extrabasisatoms
            self.extrabasis=extrabasis
        else:
            self.extrabasisatoms=[]
            self.extrabasis=""
        
        #Used in the case of counterpoise calculations
        self.ghostatoms = [] #Adds ":" in front of element in coordinate block. Have basis functions and grid points
        self.dummyatoms = [] #Adds DA instead of element. No real atom

        # For ORCA calculations that define fragments within molecule
        self.fragment_indices = fragment_indices

        # self.qmatoms need to be set for Flipspin to work for QM/MM job.
        #Overwritten by QMMMtheory, used in Flip-spin
        self.qmatoms=[]
        
        #Whether to keep a copy of last output (filename_last.out) or not
        self.keep_last_output=True

        #XDM: if True then we add !AIM to input
        self.xdm=False
        if xdm == True:
            self.xdm=True
            self.xdm_a1=xdm_a1
            self.xdm_a2=xdm_a2
            self.xdm_func=xdm_func
            self.orcasimpleinput = self.orcasimpleinput + ' AIM'

        if self.printlevel >=2:
            print("")
            print("Creating ORCA object")
            print("ORCA dir:", self.orcadir)
            print(self.orcasimpleinput)
            print(self.orcablocks)
        print("\nORCATheory object created!")
    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old ORCA files")
        list_files=[]
        list_files.append(self.filename + '.gbw')
        list_files.append(self.filename + '.densities')
        list_files.append(self.filename + '.ges')
        list_files.append(self.filename + '.prop')
        list_files.append(self.filename + '.uco')
        list_files.append(self.filename + '_property.txt')
        list_files.append(self.filename + '.inp')
        list_files.append(self.filename + '.out')
        list_files.append(self.filename + '.engrad')
        list_files.append(self.filename + '.cis')
        list_files.append(self.filename + '_last.out')
        list_files.append(self.filename + '.xyz')
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

    #Do an ORCA-optimization instead of ASH optimization. Useful for gas-phase chemistry when ORCA-optimizer is better than geomeTRIC
    def Opt(self, fragment=None, Grad=None, Hessian=None, numcores=None, charge=None, mult=None):

        module_init_time=time.time()
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING INTERNAL ORCA OPTIMIZATION-------------", BC.END)
        #Coords provided to run or else taken from initialization.
        #if len(current_coords) != 0:



        if fragment == None:
            print("No fragment provided to Opt.")
            ashexit()
        else:
            print("Fragment provided to Opt")

        
        current_coords=fragment.coords
        elems=fragment.elems
        #Check charge/mult
        charge,mult = check_charge_mult(charge, mult, self.theorytype, fragment, "ORCATheory.Opt", theory=self)

        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for ORCATheory.Opt method", BC.END)
            ashexit()



        if numcores==None:
            numcores=self.numcores

        self.extraline=self.extraline+"\n! OPT "

        print("Running ORCA with {} cores available".format(numcores))
        print("Object label:", self.label)

        print("Creating inputfile:", self.filename+'.inp')
        print("ORCA input:")
        print(self.orcasimpleinput)
        print(self.extraline)
        print(self.orcablocks)
        print("Charge: {}  Mult: {}".format(charge, mult))


        #TODO: Make more general
        create_orca_input_plain(self.filename, elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                charge, mult, extraline=self.extraline, HSmult=self.HSmult, moreadfile=self.moreadfile)
        print(BC.OKGREEN, "ORCA Calculation started.", BC.END)
        run_orca_SP_ORCApar(self.orcadir, self.filename + '.inp', numcores=numcores, bind_to_core_option=self.bind_to_core_option, 
                            ignore_ORCA_error=self.ignore_ORCA_error)
        print(BC.OKGREEN, "ORCA Calculation done.", BC.END)

        outfile=self.filename+'.out'
        ORCAfinished,iter = checkORCAfinished(outfile)
        if ORCAfinished == True:
            print("ORCA job finished")
            if checkORCAOptfinished(outfile) ==  True:
                print("ORCA geometry optimization finished")
                self.energy=ORCAfinalenergygrab(outfile)
                #Grab optimized coordinates from filename.xyz
                opt_elems,opt_coords = ash.modules.module_coords.read_xyzfile(self.filename+'.xyz')
                print(opt_coords)
                
                fragment.replace_coords(fragment.elems,opt_coords)
            else:
                print("ORCA optimization failed to converge. Check ORCA output")
                ashexit()
        else:
            print("Something happened with ORCA job. Check ORCA output")
            ashexit()

        print("ORCA optimized energy:", self.energy)
        print("ASH fragment updated:", fragment)
        fragment.print_coords()
        #Writing out fragment file and XYZ file
        fragment.print_system(filename='Fragment-optimized.ygg')
        fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')

        #Printing internal coordinate table
        ash.modules.module_coords.print_internal_coordinate_table(fragment)
        print_time_rel(module_init_time, modulename='ORCA Opt-run', moduleindex=2)
        return 

    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, charge=None, mult=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, numcores=None, label=None):
        module_init_time=time.time()
        self.runcalls+=1
        if self.printlevel >= 2:
            print(BC.OKBLUE,BC.BOLD, "------------RUNNING ORCA INTERFACE-------------", BC.END)
            print("Object-label:", self.label) #To distinguish multiple objects
            print("Run-label:", label) #Primarily used in multiprocessing
        #Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for ORCATheory.run method", BC.END)
            ashexit()

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        #If QM/MM then atomindices lists like extrabasisatoms, atomstoflip and fragment_indices have to be updated
        if len(self.qmatoms) != 0:

            #Fragment indices need to be updated if QM/MM
            if self.fragment_indices != None:
                fragment_indices=[]
                for f in self.fragment_indices:
                    temp = [self.qmatoms.index(i) for i in f]
                    fragment_indices.append(temp)
            else:
                fragment_indices=self.fragment_indices
            #extrabasisatomindices if QM/MM
            #print("QM atoms :", self.qmatoms)
            qmatoms_extrabasis=[self.qmatoms.index(i) for i in self.extrabasisatoms]
            #new QM-region indices for atomstoflip if QM/MM
            try:
                qmatomstoflip=[self.qmatoms.index(i) for i in self.atomstoflip]
            except ValueError:
                print("Atoms to flip:", self.atomstoflip)
                print("Error: Atoms to flip are not all in QM-region")
                ashexit()
        else:
            qmatomstoflip=self.atomstoflip
            qmatoms_extrabasis=self.extrabasisatoms
            fragment_indices=self.fragment_indices
        
        if numcores==None:
            numcores=self.numcores
        
        #Basis set definition per element from input dict
        if self.basis_per_element != None:
            basisstring=""
            for el,b in self.basis_per_element.items():
                basisstring += f"newgto {el} \"{b}\" end\n"
            basisblock=f"""
%basis
{basisstring} 
end"""
            self.orcablocks = self.orcablocks + basisblock

        if self.printlevel >= 2:
            print("Running ORCA with {} cores available".format(numcores))

        #MOREAD. Checking file provided exists and determining what to do if not
        if self.moreadfile != None:
            print_if_level(f"Moreadfile option active. File path: {self.moreadfile}", self.printlevel,2)
            if os.path.isfile(self.moreadfile) is True:
                print_if_level(f"File exists in current directory: {os.getcwd()}", self.printlevel,2)
            else:
                print_if_level(f"File does not exist in current directory: {os.getcwd()}", self.printlevel,2)
                if os.path.isabs(self.moreadfile) is True:
                    print("Error: Absolute path provided but file does not exists. Exiting")
                    ashexit()
                else:
                    print_if_level("Checking if file exists in parentdir instead:", self.printlevel,2)
                    if os.path.isfile(f"../{self.moreadfile}") is True:
                        print_if_level("Yes. Copying file to current dir", self.printlevel,2)
                        shutil.copy(f"../{self.moreadfile}", f"./{self.moreadfile}")
        else:
            print_if_level(f"Moreadfile option not active", self.printlevel,2)
            if os.path.isfile(f"{self.filename}.gbw") is False:
                print_if_level(f"No {self.filename}.gbw file is present in dir.", self.printlevel,2)
                if self.path_to_last_gbwfile_used != None:
                    print_if_level("Found a path to last GBW-file used by this Theory object. Will try to copy this file do current dir", self.printlevel,2)
                    try:
                        shutil.copy(self.path_to_last_gbwfile_used, f"./{self.filename}.gbw")
                    except FileNotFoundError:
                        print_if_level("File was not found. May have been deleted", self.printlevel,2)
                    if self.autostart is False:
                        print_if_level("Autostart option is False. ORCA will ignore this file", self.printlevel,2)
                    else:
                        print_if_level("Autostart feature is active. ORCA will read GBW-file present.", self.printlevel,2)
                else:
                    print_if_level(f"Checking if a file {self.filename}.gbw exists in parentdir:", self.printlevel,2)
                    if os.path.isfile(f"../{self.filename}.gbw") is True:
                        print_if_level("Yes. Copying file from parentdir to current dir", self.printlevel,2)
                        shutil.copy(f"../{self.filename}.gbw", f"./{self.filename}.gbw")
                    else:
                        print_if_level("Found no file. ORCA will guess new orbitals", self.printlevel,2)
            else:
                print_if_level(f"A GBW-file with same basename : {self.filename}.gbw is present", self.printlevel,2)
                if self.autostart is False:
                    print_if_level("Autostart is False. ORCA will ignore any file present", self.printlevel,2)
                else:
                    print_if_level("Autostart feature is active. ORCA will read GBW-file present.", self.printlevel,2)

        #TDDFT option
        #If gradient requested by Singlepoint(Grad=True) or Optimizer then TDDFT gradient is calculated instead
        if self.TDDFT == True:
            if '%tddft' not in self.orcablocks:
                self.orcablocks=self.orcablocks+"""
                %tddft
                nroots {}
                IRoot {}
                end
                """.format(self.TDDFTroots, self.FollowRoot)

        #If 1st runcall, add this to inputfile
        if self.runcalls == 1:
            #first_iteration_input
            extraline=self.extraline+"\n"+self.first_iteration_input
        else:
            extraline=self.extraline

        if self.printlevel >= 2:
            print("Creating inputfile:", self.filename+'.inp')
            print("ORCA input:")
            print(self.orcasimpleinput)
            print(extraline)
            print(self.orcablocks)
            print("Charge: {}  Mult: {}".format(charge, mult))
        #Printing extra options chosen:
        if self.brokensym==True:
            if self.printlevel >= 2:
                print("Brokensymmetry SpinFlipping on! HSmult: {}.".format(self.HSmult))
            if self.HSmult == None:
                print("Error:HSmult keyword in ORCATheory has not been set. This is required. Exiting.")
                ashexit()
            if len(qmatomstoflip) == 0:
                print("Error: atomstoflip keyword needs to be set. This is required. Exiting.")
                ashexit()

            for flipatom,qmflipatom in zip(self.atomstoflip,qmatomstoflip):
                if self.printlevel >= 2:
                    print("Flipping atom: {} QMregionindex: {} Element: {}".format(flipatom, qmflipatom, qm_elems[qmflipatom]))
        if self.extrabasis != "":
            if self.printlevel >= 2:
                print("Using extra basis ({}) on QM-region indices : {}".format(self.extrabasis,qmatoms_extrabasis))
        if self.dummyatoms:
            if self.printlevel >= 2:
                print("Dummy atoms defined:", self.dummyatoms)
        if self.ghostatoms:
            if self.printlevel >= 2:
                print("Ghost atoms defined:", self.ghostatoms)
        if self.fragment_indices:
            if self.printlevel >= 2:
                print("List of fragment indices defined:", fragment_indices)
        if PC==True:
            if self.printlevel >= 2:
                print("Pointcharge embedding is on!")
            create_orca_pcfile(self.filename, current_MM_coords, MMcharges)
            if self.brokensym == True:
                create_orca_input_pc(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        charge, mult, extraline=extraline, HSmult=self.HSmult, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                     atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock,
                                     fragment_indices=fragment_indices)
            else:
                create_orca_input_pc(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        charge, mult, extraline=extraline, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                        extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock,
                                        fragment_indices=fragment_indices)
        else:
            if self.brokensym == True:
                create_orca_input_plain(self.filename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        charge,mult, extraline=extraline, HSmult=self.HSmult, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                     atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock, 
                                     ghostatoms=self.ghostatoms, dummyatoms=self.dummyatoms,
                                     fragment_indices=fragment_indices)
            else:
                create_orca_input_plain(self.filename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        charge,mult, extraline=extraline, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                        extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock,
                                        ghostatoms=self.ghostatoms, dummyatoms=self.dummyatoms,
                                        fragment_indices=fragment_indices)

        #Run inputfile using ORCA parallelization. Take numcores argument.
        #print(BC.OKGREEN, "------------Running ORCA calculation-------------", BC.END)
        if self.printlevel >= 2:
            print(BC.OKGREEN, "ORCA Calculation starting.", BC.END)

        run_orca_SP_ORCApar(self.orcadir, self.filename + '.inp', numcores=numcores, bind_to_core_option=self.bind_to_core_option,
                                check_for_errors=self.check_for_errors, check_for_warnings=self.check_for_warnings, ignore_ORCA_error=self.ignore_ORCA_error)
        if self.printlevel >= 1:
            print(BC.OKGREEN, "ORCA Calculation done.", BC.END)

        #Now that we have possibly run a BS-DFT calculation, turning Brokensym off for future calcs (opt, restart, etc.)
        # using this theory object
        #TODO: Possibly use different flag for this???
        if self.brokensym==True:
            if self.printlevel >= 2:
                print("ORCA Flipspin calculation done. Now turning off brokensym in ORCA object for possible future calculations")
            self.brokensym=False

        #Now that we have possibly run a ORCA job with moreadfile we now turn the moreadfile option off as we probably want to use the 
        if self.moreadfile != None:
            print("First ORCATheory calculation finished.")
            #Now either keeping moreadfile or removing it. Default: removing
            if self.moreadfile_always == False:
                print("Now turning moreadfile option off.")
                self.moreadfile=None



        #Check if finished. Grab energy and gradient
        outfile=self.filename+'.out'
        engradfile=self.filename+'.engrad'
        pcgradfile=self.filename+'.pcgrad'

        #Optional save ORCA output with filename according to label
        if self.save_output_with_label is True:
            shutil.copy(self.filename+'.out', self.filename+f'_{self.label}_{charge}_{mult}.out')

        #Keep outputfile from each run if requested
        if self.keep_each_run_output is True:
            print("\nkeep_each_run_output is True")
            print("Copying {} to {}".format(self.filename+'.out', self.filename+'_run{}'.format(self.runcalls)+'.out'))
            shutil.copy(self.filename+'.out', self.filename+'_run{}'.format(self.runcalls)+'.out')

        #Always make copy of last output file
        if self.keep_last_output is True:
            shutil.copy(self.filename+'.out', self.filename+'_last.out')

        #Save path to last GBW-file (used if ASH changes directories, e.g. goes from NumFreq)
        self.path_to_last_gbwfile_used=f"{os.getcwd()}/{self.filename}.gbw"


        if self.ignore_ORCA_error is False:
            ORCAfinished,numiterations = checkORCAfinished(outfile)
            #Check if ORCA finished or not. Exiting if so
            if ORCAfinished is False:
                print(BC.FAIL,"Problem with ORCA run", BC.END)
                print(BC.OKBLUE,BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
                print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2)
                ashexit()

            if self.printlevel >= 1:
                print(f"ORCA converged in {numiterations} iterations")

        #Print population analysis in each run if requested
        if self.print_population_analysis is True:
            if self.printlevel >= 2:
                print("\nPrinting Mulliken Population analysis:")
                print("-"*30)
                charges = grabatomcharges_ORCA("Mulliken",self.filename+'.out')
                spinpops = grabspinpop_ORCA("Mulliken",self.filename+'.out')
                self.properties["Mulliken_charges"] = charges
                self.properties["Mulliken_spinpops"] = spinpops
                print("{:<2} {:<2}  {:>10} {:>10}".format(" ", " ", "Charge", "Spinpop"))
                for i,(el, ch, sp) in enumerate(zip(qm_elems,charges, spinpops)):
                    print("{:<2} {:<2}: {:>10.4f} {:>10.4f}".format(i,el,ch,sp))

        #Grab energy
        if self.ignore_ORCA_error is False:
            self.energy=ORCAfinalenergygrab(outfile)
            if self.printlevel >= 1:
                print("ORCA energy:", self.energy)

        #Grab possible properties
        #ICE-CI
        try:
            E_PT2_rest = float(pygrep("\'rest\' energy", self.filename+'.out')[-1])
            num_genCFGs,num_selected_CFGs,num_after_SD_CFGs = ICE_WF_CFG_CI_size(self.filename+'.out')
            self.properties["E_var"] = self.energy
            self.properties["E_PT2_rest"] = E_PT2_rest
            self.properties["num_genCFGs"] = num_genCFGs
            self.properties["num_selected_CFGs"] = num_selected_CFGs
            self.properties["num_after_SD_CFGs"] = num_after_SD_CFGs
        except:
            pass

        #Grab timings from ORCA output
        orca_timings = ORCAtimingsgrab(outfile)

        #Initializing zero gradient array
        self.grad = np.zeros((len(qm_elems), 3))

        #XDM option: WFX file should have been created.
        if self.xdm == True:
            dispE,dispgrad = ash.functions.functions_elstructure.xdm_run(wfxfile=self.filename+'.wfx', a1=self.xdm_a1, a2=self.xdm_a2,functional=self.xdm_func)
            if self.printlevel >= 2:
                print("XDM dispersion energy:", dispE)
            self.energy = self.energy + dispE
            if self.printlevel >= 2:
                print("DFT+XDM energy:", self.energy )
            #TODO: dispgrad not yet done
            self.grad = self.grad + dispgrad



        if Grad == True:
            grad =ORCAgradientgrab(engradfile)
            self.grad = self.grad + grad
            if PC == True:
                #Print time to calculate ORCA QM-PC gradient
                if "pc_gradient" in orca_timings:
                    if self.printlevel >= 2:
                        print("Time calculating QM-Pointcharge gradient: {} seconds".format(orca_timings["pc_gradient"]))
                #Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                self.pcgrad=ORCApcgradientgrab(pcgradfile)
                if self.printlevel >= 2:
                    print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
                return self.energy, self.grad, self.pcgrad
            else:
                if self.printlevel >= 2:
                    print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
                return self.energy, self.grad

        else:
            if self.printlevel >= 2:
                print("Single-point ORCA energy:", self.energy)
                print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
            print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.energy




###############################################
#CHECKS FOR ORCA program
###############################################

def check_ORCA_location(orcadir):
    if orcadir != None:
        finalorcadir = orcadir
        print(BC.OKGREEN,f"Using orcadir path provided: {finalorcadir}", BC.END)
    else:
        print(BC.WARNING, "No orcadir argument passed to ORCATheory. Attempting to find orcadir variable in ASH settings file (~/ash_user_settings.ini)", BC.END)
        try:
            finalorcadir=ash.settings_ash.settings_dict["orcadir"]
            print(BC.OKGREEN,"Using orcadir path provided from ASH settings file (~/ash_user_settings.ini): ", finalorcadir, BC.END)
        except KeyError:
            print(BC.WARNING,"Found no orcadir variable in ASH settings file either.",BC.END)
            print(BC.WARNING,"Checking for ORCA in PATH environment variable.",BC.END)
            try:
                finalorcadir = os.path.dirname(shutil.which('orca'))
                print(BC.OKGREEN,"Found orca binary in PATH. Using the following directory:", finalorcadir, BC.END)
            except TypeError:
                print(BC.FAIL,"Found no orca binary in PATH environment variable either. Giving up.", BC.END)
                ashexit()
    return finalorcadir

def check_ORCAbinary(orcadir):
    """Checks if this is a proper working ORCA quantum chemistry binary
    Args:
        orcadir ([type]): [description]
    """
    print("Checking if ORCA binary works...",end="")
    try:
        p = sp.Popen([orcadir+"/orca"], stdout = sp.PIPE)
        out, err = p.communicate()
        if 'This program requires the name of a parameterfile' in str(out):
            print(BC.OKGREEN,"yes", BC.END)
            return True
        else:
            print(BC.FAIL,"Problem: ORCA binary: {} does not work. Exiting!".format(orcadir+'/orca'), BC.END)
            ashexit()
    except FileNotFoundError:
        print("ORCA binary was not found")
        ashexit()



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
    print("\nNumber of CPU cores: ", numcores)
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
    basename = inpfile.split('.')[0]
    with open(basename+'.out', 'w') as ofile:
        process = sp.run([orcadir + '/orca', basename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

# Run ORCA single-point job using ORCA parallelization. Will add pal-block if numcores >1.
def run_orca_SP_ORCApar(orcadir, inpfile, numcores=1, check_for_warnings=True, check_for_errors=True, bind_to_core_option=True, ignore_ORCA_error=False):
    if numcores>1:
        palstring='%pal nprocs {} end'.format(numcores)
        with open(inpfile) as ifile:
            insert_line_into_file(inpfile, '!', palstring, Once=True )
    basename = inpfile.replace('.inp','')

    #LD_LIBRARY_PATH enforce: https://orcaforum.kofo.mpg.de/viewtopic.php?f=11&t=10118
    #"-x LD_LIBRARY_PATH -x PATH"

    with open(basename+'.out', 'w') as ofile:
        try:
            if bind_to_core_option is True:
                #f"\"-x {orcadir} --bind-to none\""
                process = sp.run([orcadir + '/orca', inpfile, f"--bind-to none"], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            else:
                process = sp.run([orcadir + '/orca', inpfile], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            if check_for_errors:
                grab_ORCA_errors(basename+'.out')
            if check_for_warnings:
                grab_ORCA_warnings(basename+'.out')
        except Exception as e:
            print("Subprocess error! Exception message:", e)


            #We get an exception if 
            print(BC.FAIL,"ASH encountered a problem when running ORCA. Something went wrong, most likely ORCA ran into an error.",BC.END)
            print(BC.FAIL,f"Please check the ORCA outputfile: {basename+'.out'} for error messages", BC.END)
            print()
            if check_for_errors:
                grab_ORCA_errors(basename+'.out')
            if check_for_warnings:
                grab_ORCA_warnings(basename+'.out')
            print("ignore_ORCA_error:", ignore_ORCA_error)
            if ignore_ORCA_error is True:
                print("ignore_ORCA_error here")
                return
            else:
                ashexit()

def grab_ORCA_warnings(filename):
    warning_lines=[]
    #Error-words to search for
    #TODO: Avoid searching though file multiple times.
    #TODO: Write pygrep version that supports list of search-strings
    warning_strings=['WARNING', 'warning', 'Warning']
    for warnstring in warning_strings:
        warn_l = pygrep2(warnstring, filename)
        warning_lines+=warn_l

    warnings=[]
    #Lines that are not useful warnings
    ignore_lines=['                       Please study these wa','                                        WARNINGS', 
        'Warning: in a DFT calculation', 'WARNING: Old DensityContainer', 'WARNING: your system is open-shell' ]
    for warn in warning_lines:
        false_positive = any(warn.startswith(ign) for ign in ignore_lines)
        if false_positive is False:
            warnings.append(warn)
    if len(warnings):
        print("Found warning messages in ORCA outputfile:")
        print(*warnings)

def grab_ORCA_errors(filename):
    error_lines=[]
    #Error-words to search for
    #TODO: Avoid searching though file multiple times.
    #TODO: Write pygrep version that supports list of search-strings
    error_strings=['error', 'Error', 'ERROR', 'aborting']
    for errstring in error_strings:
        error_l = pygrep2(errstring, filename)
        for e in error_l:
            if e not in error_lines:
                error_lines.append(e)

    errors=[]
    #Lines that are not errors
    ignore_lines=['   Iter.        energy            ||Error||_2',' WARNING: the maximum gradient error','           *** ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR','   Startup', '   DIIS-Error',' DIIS', 'sum of PNO error', '  Last DIIS Error', '    DIIS-Error', ' Sum of total truncation errors', 
        '  Sum of total UMP2 truncation', ]
    for err in error_lines:
        false_positive = any(err.startswith(ign) for ign in ignore_lines)
        if false_positive is False:
            errors.append(err)
    if len(errors):
        print("Found possible error messages in ORCA outputfile:")
        print(*errors)

#Check if ORCA finished.
#Todo: Use reverse-read instead to speed up?
def checkORCAfinished(file):
    iter=None
    with open(file) as f:
        for line in f:
            if 'SCF CONVERGED AFTER' in line:
                iter=line.split()[-3]
            if 'TOTAL RUN TIME:' in line:
                return True,iter
    return False,None
def checkORCAOptfinished(file):
    converged=False
    with open(file) as f:
        for line in f:
            if 'THE OPTIMIZATION HAS CONVERGED' in line:
                converged=True
            if converged==True:
                if '***               (AFTER' in line:
                    cycles=line.split()[2]
                    print("ORCA Optimization converged in {} cycles".format(cycles))
        return converged

#Grab Final single point energy. Ignoring possible encoding errors in file
def ORCAfinalenergygrab(file, errors='ignore'):
    Energy=None
    with open(file) as f:
        for line in f:
            if 'FINAL SINGLE POINT ENERGY' in line:
                if "Wavefunction not fully converged!" in line:
                    print("ORCA WF not fully converged!")
                    print("Not using energy. Modify ORCA settings")
                    ashexit()
                else:
                    #Changing: sometimes ORCA adds info to the right of energy
                    #Energy=float(line.split()[-1])
                    Energy=float(line.split()[4])
    if Energy == None:
        print(BC.FAIL,"ASH found no energy in file:", file, BC.END)
        print(BC.FAIL,"Something went wrong with ORCA run. Check ORCA outputfile:", file, BC.END)
        print(BC.OKBLUE,BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
        ashexit()
    return Energy


#Grab ORCA timings. Return dictionary
def ORCAtimingsgrab(file):
    timings={} #in seconds
    try:
        with open(file) as f:
            for line in f:
                if 'Calculating one electron integrals' in line:
                    one_elec_integrals=float(line.split()[-2].replace("(",""))
                    timings["one_elec_integrals"]= one_elec_integrals
                if 'SCF Gradient evaluation         ...' in line:
                    time_scfgrad=float(line.split()[4])
                    timings["time_scfgrad"]=time_scfgrad
                if 'SCF iterations                  ...' in line:
                    time_scfiterations=float(line.split()[3])
                    timings["time_scfiterations"]=time_scfiterations
                if 'GTO integral calculation        ...' in line:
                    time_gtointegrals=float(line.split()[4])
                    timings["time_gtointegrals"]=time_gtointegrals
                if 'SCF Gradient evaluation         ...' in line:
                    time_scfgrad=float(line.split()[4])
                    timings["time_scfgrad"]=time_scfgrad
                if 'Sum of individual times         ...:' in line:
                    total_time=float(line.split()[4])
                    timings["total_time"]=total_time
                if 'One electron gradient       ....' in line:
                    one_elec_gradient=float(line.split()[4])
                    timings["one_elec_gradient"]=one_elec_gradient
                if 'RI-J Coulomb gradient       ....' in line:
                    rij_coulomb_gradient=float(line.split()[4])
                    timings["rij_coulomb_gradient"]=rij_coulomb_gradient
                if 'XC gradient                 ....' in line:
                    xc_gradient=float(line.split()[3])
                    timings["xc_gradient"]=xc_gradient
                if 'Point charge gradient       ....' in line:
                    pc_gradient=float(line.split()[4])
                    timings["pc_gradient"]=pc_gradient
    except:
        pass
    return timings



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

#Grab TDDFT orbital pairs from ORCA output
def tddft_orbitalpairs_grab(file):
    tddftstates=[]
    tddft=True
    tddftgrab=False
    stategrab=False
    states_dict={}
    if tddft==True:
        with open(file) as f:
            for line in f:
                if tddftgrab==True:
                    if 'STATE' in line:
                        stategrab=True
                        state=int(line.split()[1].replace(":",""))
                        states_dict[state]=[]
                    if stategrab is True:
                        if '->' in line:
                            orb_occ=line.split()[0]
                            orb_unocc=line.split()[2]
                            weight=float(line.split()[4])
                            states_dict[state].append((orb_occ,orb_unocc,weight))
                if 'the weight of the individual excitations' in line:
                    tddftgrab=True
    return states_dict

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

    #Total number of orbitals
    totnumorbitals=len(bands_alpha)+len(virtbands_a)
    #Final dict
    MOdict= {"occ_alpha":bands_alpha, "occ_beta":bands_alpha, "unocc_alpha":virtbands_a, "unocc_beta":virtbands_b, "Openshell":Openshell,
            "totnumorbitals":totnumorbitals}
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
                elem=line.split()[0]; x_c=ash.constants.bohr2ang*float(line.split()[2]);y_c=ash.constants.bohr2ang*float(line.split()[3]);z_c=ash.constants.bohr2ang*float(line.split()[4])
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
        #print(str(coords[atom][0]/ash.constants.bohr2ang))
        #print(str(coords[atom][1]/ash.constants.bohr2ang))
        #print(str(coords[atom][2]/ash.constants.bohr2ang))
        #orcahessfile.write(" "+str(elems[atom])+'    '+str(mass)+"  "+str(coords[atom][0]/ash.constants.bohr2ang)+
        #                   " "+str(coords[atom][1]/ash.constants.bohr2ang)+" "+str(coords[atom][2]/ash.constants.bohr2ang)+"\n")
        orcahessfile.write(" "+el+'    '+str(mass)+"  "+str(coord[0]/ash.constants.bohr2ang)+
                           " "+str(coord[1]/ash.constants.bohr2ang)+" "+str(coord[2]/ash.constants.bohr2ang)+"\n")
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
                         HSmult=None, atomstoflip=None, Hessian=False, extrabasisatoms=None, extrabasis=None, 
                         moreadfile=None, propertyblock=None, fragment_indices=None):
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
            #Adding fragment specification
            elif fragment_indices != None:
                fragmentindex= search_list_of_lists_for_index(i,fragment_indices)
                #To prevent linkatoms:
                if fragmentindex != None:
                    orcafile.write('{} {} {} {} \n'.format(f"{el}({fragmentindex+1})", c[0], c[1], c[2]))                
            else:
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
        orcafile.write('*\n')
        if propertyblock != None:
            orcafile.write(propertyblock)
#Create simple ORCA inputfile from elems,coords, input, charge, mult,pointcharges
#Allows for extraline that could be another '!' line or block-inputline.
def create_orca_input_plain(name,elems,coords,orcasimpleinput,orcablockinput,charge,mult, Grad=False, Hessian=False, extraline='',
                            HSmult=None, atomstoflip=None, extrabasis=None, extrabasisatoms=None, moreadfile=None, propertyblock=None, 
                            ghostatoms=None, dummyatoms=None,fragment_indices=None):
    if extrabasisatoms == None:
        extrabasisatoms=[]
    if ghostatoms == None:
        ghostatoms=[]
    if dummyatoms == None:
        dummyatoms = []

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
            #Setting atom to be a ghost atom
            elif i in ghostatoms:
                orcafile.write('{}{} {} {} {} \n'.format(el,":", c[0], c[1], c[2]))
            elif i in dummyatoms:
                orcafile.write('{} {} {} {} \n'.format("DA", c[0], c[1], c[2]))
            #Adding fragment specification
            elif fragment_indices != None:
                fragmentindex= search_list_of_lists_for_index(i,fragment_indices)
                orcafile.write('{} {} {} {} \n'.format(f"{el}({fragmentindex+1})", c[0], c[1], c[2]))
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

#Grabbing spin populations
def grabspinpop_ORCA(chargemodel,outputfile):
    grab=False
    coordgrab=False
    spinpops=[]
    BS=False #if broken-symmetry job
    #if
    if len(pygrep2("WARNING: Broken symmetry calculations", outputfile)):
        BS=True

    if chargemodel == "Mulliken":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if 'Sum of atomic' in line:
                        grab=False
                    elif '------' not in line:
                        spinpops.append(float(line.split()[-1]))
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
                        spinpops.append(float(line.split()[-1]))
                if 'LOEWDIN ATOMIC CHARGES' in line:
                    grab=True
    else:
        print("Unknown chargemodel. Exiting...")
        ashexit()
    #If BS then we have grabbed charges for both high-spin and BS solution
    if BS is True:
        print("Broken-symmetry job detected. Only taking BS-state populations")
        spinpops=spinpops[int(len(spinpops)/2):]
    return spinpops

def grabatomcharges_ORCA(chargemodel,outputfile):
    grab=False
    coordgrab=False
    charges=[]
    BS=False #if broken-symmetry job
    #if
    if len(pygrep2("WARNING: Broken symmetry calculations", outputfile)):
        BS=True

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
        atomicnumbers=ash.modules.module_coords.elemstonuccharges(elems)
        charges = ash.functions.functions_elstructure.calc_cm5(atomicnumbers, coords, charges)
        print("CM5 charges :", list(charges))
    elif chargemodel == "Mulliken":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if 'Sum of atomic' in line:
                        grab=False
                    elif '------' not in line:
                        charges.append(float(line.split()[column]))
                if 'MULLIKEN ATOMIC CHARGES' in line:
                    grab=True
                    if 'SPIN POPULATIONS' in line:
                        column=-2
                    else:
                        column=-1

    elif chargemodel == "Loewdin":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab==True:
                    if 'Sum of atomic' in line:
                        grab=False
                    elif len(line.replace(' ','')) < 2:
                        grab=False
                    elif '------' not in line:
                        charges.append(float(line.split()[column]))
                if 'LOEWDIN ATOMIC CHARGES' in line:
                    grab=True
                    if 'SPIN POPULATIONS' in line:
                        column=-2
                    else:
                        column=-1
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
        ashexit()
    
    #If BS then we have grabbed charges for both high-spin and BS solution
    if BS is True:
        print("Broken-symmetry job detected. Only taking BS-state populations")
        charges=charges[int(len(charges)/2):]

    return charges


# Wrapper around interactive orca_plot
# Todo: add TDDFT difference density, natural orbitals, MDCI spin density?
def run_orca_plot(filename, option, orcadir=None, gridvalue=40, specify_density=False, 
    densityfilename=None, individual_file=False, mo_operator=0, mo_number=None,):
    print("Running run_orca_plot")
    orcadir = check_ORCA_location(orcadir)
    def check_if_file_exists():
        if os.path.isfile(densityfilename) is True:
            print("File exists")
        else:
            print("File does not exist! Skipping")
            return
    #If individual_file is True then we can check if file exists (case for MRCI)
    if individual_file is True:
        check_if_file_exists()
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
        if specify_density is True:
            print("specify_density: True. Picking density filename:", densityfilename)
            #Choosing e.g. MRCI density
            p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                input=f'5\n7\n4\n{gridvalue}\n1\n{plottype}\nn\n{densityfilename}\n10\n11\n\n', encoding='ascii')  
        else:
            p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                       input=f'5\n7\n4\n{gridvalue}\n1\n{plottype}\ny\n10\n11\n\n', encoding='ascii')       
    elif option=='mo':
        p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                       input=f'5\n7\n4\n{gridvalue}\n3\n{mo_operator}\n2\n{mo_number}\n10\n11\n\n', encoding='ascii')
    #If plotting CIS/TDDFT density then we tell orca_plot explicity.
    elif option == 'cisdensity' or option == 'cisspindensity':
        p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                       input=f'5\n7\n4\n{gridvalue}\n1\n{plottype}\nn\n{densityfilename}\n10\n11\n\n', encoding='ascii')

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
    return occupations

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

#Grab ICE-WF info from CASSCF job
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

#Grab ICE-WF CFG info from CI job
def ICE_WF_CFG_CI_size(filename):
    num_after_SD_CFGs=0
    num_genCFGs=0
    num_selected_CFGs=0
    with open(filename) as g:
        for line in g:
            if '# of configurations after S+D' in line:
                num_after_SD_CFGs=int(line.split()[-1])
            if '# of configurations after Selection' in line:
                num_selected_CFGs=int(line.split()[-1])
            if ' # of generator configurations' in line:
                num_genCFGs=int(line.split()[5])
    return num_genCFGs,num_selected_CFGs,num_after_SD_CFGs



def grab_EFG_from_ORCA_output(filename):
    occgrab=False
    occupations=[]
    qro_energies=[]
    with open(filename) as f:
        for line in f:
            if ' V(Tot)' in line:
                efg_values=[float(line.split()[-3]),float(line.split()[-2]),float(line.split()[-1])]
                return efg_values

#Charge/mult must be in fragments
def counterpoise_calculation_ORCA(fragments=None, theory=None, monomer1_indices=None, monomer2_indices=None, charge=None, mult=None):
    print_line_with_mainheader("COUNTERPOISE CORRECTION JOB")
    print("\n Boys-Bernardi counterpoise correction\n")
    
    if theory == None and fragments == None:
        print("theory and list of ASH fragments required")
        ashexit()
    if monomer1_indices==None or monomer2_indices == None:
        print("Error: monomer1_indices and monomer2_indices need to be set")
        print("These are lists of atom indices indicating monomer1 and monomer2 in dimer fragment")
        print("Example: monomer1_indices=[0,1,2] (H2O monomer) and monomer2_indices=[3,4,5,6] (MeOH monomer) in an H2O...MeOH dimer with coordinates:")
        print("""O   -0.525329794  -0.050971084  -0.314516861
H   -0.942006633   0.747901631   0.011252816
H    0.403696525   0.059785981  -0.073568368
O    2.316633291   0.045500849   0.071858389
H    2.684616115  -0.526576554   0.749386716
C    2.781638362  -0.426129067  -1.190300721
H    2.350821267   0.224964624  -1.943414753
H    3.867602049  -0.375336206  -1.264612649
H    2.453295744  -1.445998564  -1.389381355
        """)
        print("")
        ashexit()

    print("monomer1_indices:", monomer1_indices)
    print("monomer2_indices:", monomer2_indices)
    print("")
    #list of fragment indices
    fragments_indices=[i for i in range(0,len(fragments))]

    for frag in fragments:
        if frag.charge == None:
            print("Charge/mult information not present in all fragments")
            ashexit()

    #Determine what is dimer and monomers in list of fragments
    numatoms_all=[fragment.numatoms for fragment in fragments]
    dimer_index = numatoms_all.index(max(numatoms_all))
    dimer=fragments[dimer_index]

    if len(monomer1_indices+monomer2_indices) != dimer.numatoms:
        print("Error: Something wrong with monomer1_indices or monomer2_indices. Don't add up ({}) to number of atoms in dimer ({})".format(len(monomer1_indices+monomer2_indices),dimer.numatoms))
        ashexit()

    fragments_indices.remove(dimer_index)

    #Decide which is monomer1 and monomer 2 by comparing indices list to fragment.numatoms
    if fragments[fragments_indices[0]].numatoms == len(monomer1_indices):
        monomer1=fragments[fragments_indices[0]]
        monomer2=fragments[fragments_indices[1]]
    else:
        monomer1=fragments[fragments_indices[1]]
        monomer2=fragments[fragments_indices[0]]

    #Print before we begin
    print("Monomer 1:")
    print("-"*20)
    monomer1.print_coords()
    print("Monomer 1 indices in dimer:", monomer1_indices)
    print("\nMonomer 2:")
    print("-"*20)
    monomer2.print_coords()
    print("Monomer 2 indices in dimer:", monomer2_indices)
    print("\nDimer:")
    print("-"*20)
    for a,i in enumerate(range(0,dimer.numatoms)):
        if i in monomer1_indices:
            label="Monomer1"
        elif i in monomer2_indices:
            label="Monomer2"
        print("{}   {} {} {} {}   {}".format(a, dimer.elems[i], *dimer.coords[i], label))


    #Initial cleanup
    theory.cleanup()
    
    #Run dimer
    print("\nRunning dimer calculation")
    dimer_result=Singlepoint(theory=theory,fragment=dimer)
    dimer_energy = dimer_result.energy
    theory.cleanup()
    #Run monomers
    print("\nRunning monomer1 calculation")
    monomer1_result=Singlepoint(theory=theory,fragment=monomer1)
    monomer1_energy = monomer1_result.energy
    theory.cleanup()
    print("\nRunning monomer2 calculation")
    monomer2_result=Singlepoint(theory=theory,fragment=monomer2)
    monomer2_energy = monomer2_result.energy
    theory.cleanup()
    print("\nUncorrected binding energy: {} kcal/mol".format((dimer_energy - monomer1_energy-monomer2_energy)*ash.constants.hartokcal))
    
    #Monomer calcs at dimer geometry
    print("\nRunning monomers at dimer geometry via dummy atoms")
    theory.dummyatoms=monomer1_indices
    monomer1_in_dimergeo_result=Singlepoint(theory=theory,fragment=dimer)
    monomer1_in_dimergeo_energy = monomer1_in_dimergeo_result.energy
    theory.cleanup()
    theory.dummyatoms=monomer2_indices
    monomer2_in_dimergeo_result=Singlepoint(theory=theory,fragment=dimer)
    monomer2_in_dimergeo_energy = monomer2_in_dimergeo_result.energy
    theory.cleanup()

    #Removing dummyatoms
    theory.dummyatoms=[]

    #Monomers in dimer geometry with dimer basis sets (ghost atoms)
    print("-------------------------------")
    print("\nRunning monomers at dimer geometry with dimer basis set via ghostatoms")
    theory.ghostatoms=monomer1_indices

    monomer1_in_dimer_dimerbasis_result=Singlepoint(theory=theory,fragment=dimer)
    monomer1_in_dimer_dimerbasis_energy = monomer1_in_dimer_dimerbasis_result.energy
    theory.cleanup()
    theory.ghostatoms=monomer2_indices
    monomer2_in_dimer_dimerbasis_result=Singlepoint(theory=theory,fragment=dimer)
    monomer2_in_dimer_dimerbasis_energy = monomer2_in_dimer_dimerbasis_result.energy
    theory.cleanup()

    #Removing ghost atoms
    theory.ghostatoms=[]


    #RESULTS
    print("\n\nCOUNTERPOISE CORRECTION RESULTS")
    print("="*50)
    print("\nMonomer 1 energy: {} Eh".format(monomer1_energy))
    print("Monomer 2 energy: {} Eh".format(monomer2_energy))
    print("Sum of monomers energy: {} Eh".format(monomer1_energy+monomer2_energy))
    print("Dimer energy: {} Eh".format(dimer_energy))


    #Monomers at dimer geometry and dimer basis (DUMMYATOMS)
    print("\nMonomer 1 at dimer geometry: {} Eh".format(monomer1_in_dimergeo_energy))
    print("Monomer 2 at dimer geometry: {} Eh".format(monomer2_in_dimergeo_energy))
    print("Sum of monomers at dimer geometry energy: {} Eh".format(monomer1_in_dimergeo_energy+monomer2_in_dimergeo_energy))

    #Monomers at dimer geometry and dimer basis (GHOSTATOMS)
    print("\nMonomer 1 at dimer geometry with dimer basis: {} Eh".format(monomer1_in_dimer_dimerbasis_energy)) #E_A^AB (AB)
    print("Monomer 2 at dimer geometry with dimer basis: {} Eh".format(monomer2_in_dimer_dimerbasis_energy)) #E_B^AB (AB)
    print("Sum of monomers at dimer geometry with dimer basis: {} Eh".format(monomer1_in_dimer_dimerbasis_energy+monomer2_in_dimer_dimerbasis_energy))

    #
    deltaE_unc=(dimer_energy - monomer1_energy-monomer2_energy)*ash.constants.hartokcal
    counterpoise_corr=-1*(monomer1_in_dimer_dimerbasis_energy-monomer1_in_dimergeo_energy+monomer2_in_dimer_dimerbasis_energy-monomer2_in_dimergeo_energy)*ash.constants.hartokcal
    print("counterpoise_corr: {} kcal/mol".format(counterpoise_corr))
    deltaE_corrected=deltaE_unc+counterpoise_corr

    print("\nUncorrected interaction energy: {} kcal/mol".format(deltaE_unc))

    print("Corrected interaction energy: {} kcal/mol".format(deltaE_corrected))


def print_gradient_in_ORCAformat(energy,gradient,basename):
    numatoms=len(gradient)
    with open(basename+"_EXT"+".engrad", "w") as f:
        f.write("#\n")
        f.write("# Number of atoms\n")
        f.write("#\n")
        f.write("{}\n".format(numatoms))
        f.write("#\n")
        f.write("# The current total energy in E\n")
        f.write("#\n")
        f.write("     {}\n".format(energy))
        f.write("#\n")
        f.write("# The current gradient in Eh/Bohr\n")
        f.write("#\n")
        for g in gradient:
            for gg in g:
                f.write("{}\n".format(gg))

def create_ASH_otool(basename=None, theoryfile=None, scriptlocation=None, charge=None, mult=None):
    import stat
    with open(scriptlocation+"/otool_external", 'w') as otool:
        otool.write("#!/usr/bin/env python3\n")
        otool.write("from ash import *\n")
        otool.write("import pickle\n")
        otool.write("import numpy as np\n\n")
        otool.write("frag=Fragment(xyzfile=\"{}.xyz\")\n".format(basename))
        otool.write("\n")
        #TODO: FINISH
        #otool.write("energy=54.4554\n")
        #otool.write("gradient=np.random.random((frag.numatoms,3))\n")
        otool.write("#Unpickling theory object\n")
        otool.write("theory = pickle.load(open(\"{}\", \"rb\" ))\n".format(theoryfile))
        #otool.write("theory=ZeroTheory()\n")
        #otool.write("theory=ZeroTheory()\n")
        otool.write("result=Singlepoint(theory=theory,fragment=frag,Grad=True, charge={}, mult={})\n".format(charge,mult))
        otool.write("energy = result.energy")
        otool.write("gradient = result.gradient")
        otool.write("print(gradient)\n")
        otool.write("ash.interfaces.interface_ORCA.print_gradient_in_ORCAformat(energy,gradient,\"{}\")\n".format(basename))
    st = os.stat(scriptlocation+"/otool_external")
    os.chmod(scriptlocation+"/otool_external", st.st_mode | stat.S_IEXEC)

# Using ORCA as External Optimizer for ASH
#Will only work for theories that can be pickled: not OpenMMTheory, probably not QMMMTheory
def ORCA_External_Optimizer(fragment=None, theory=None, orcadir=None, charge=None, mult=None):
    print_line_with_mainheader("ORCA_External_Optimizer")
    if fragment == None or theory == None:
        print("ORCA_External_Optimizer requires fragment and theory keywords")
        ashexit()

    if charge == None or mult == None:
        print(BC.WARNING,"Warning: Charge/mult was not provided to ORCA_External_Optimizer",BC.END)
        if fragment.charge != None and fragment.mult != None:
            print(BC.WARNING,"Fragment contains charge/mult information: Charge: {} Mult: {} Using this instead".format(fragment.charge,fragment.mult), BC.END)
            print(BC.WARNING,"Make sure this is what you want!", BC.END)
            charge=fragment.charge; mult=fragment.mult
        else:
            print(BC.FAIL,"No charge/mult information present in fragment either. Exiting.",BC.END)
            ashexit()

    #Making sure we have a working ORCA location
    print("Checking for ORCA location")
    orcadir = check_ORCA_location(orcadir)
    #Making sure ORCA binary works (and is not orca the screenreader)
    check_ORCAbinary(orcadir)
    #Adding orcadir to PATH. Only required if ORCA not in PATH already
    if orcadir != None:
        os.environ["PATH"] += os.pathsep + orcadir

    #Pickle for serializing theory object
    import pickle

    #Serialize theory object for later use
    theoryfilename="theory.saved"
    pickle.dump(theory, open(theoryfilename, "wb" ))

    #Write otool_script once in location that ORCA will launch. This is an ASH E+Grad calculator
    #ORCA will call : otool_external test_EXT.extinp.tmp
    #ASH_otool creates basename_Ext.engrad that ORCA reads
    basename = "ORCAEXTERNAL"
    scriptlocation="."
    os.environ["PATH"] += os.pathsep + "."
    create_ASH_otool(basename=basename, theoryfile=theoryfilename, scriptlocation=scriptlocation, charge=charge, mult=mult)

    #Create XYZ-file for ORCA-Extopt
    xyzfile="ASH-xyzfile.xyz"
    fragment.write_xyzfile(xyzfile)

    #ORCA input file
    with open(basename+".inp", 'w') as o:
        o.write("! ExtOpt Opt\n")
        o.write("\n")
        o.write("*xyzfile {} {} {}\n".format(charge,mult,xyzfile))
    
    #Call ORCA to do geometry optimization
    with open(basename+'.out', 'w') as ofile:
        process = sp.run(['orca', basename+'.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

    #Check if ORCA finished
    ORCAfinished, iter = checkORCAfinished(basename+'.out')
    if ORCAfinished is not True:
        print("Something failed about external ORCA job")
        ashexit()
    #Check if optimization completed
    if checkORCAOptfinished(basename+'.out') is not True:
        print("ORCA external optimization failed. Check outputfile:", basename+'.out')
        ashexit()
    print("ORCA external optimization finished")

    #Grabbing final geometry to update fragment object
    elems,coords=ash.modules.module_coords.read_xyzfile(basename+".xyz")
    fragment.coords=coords

    #Grabbing final energy
    energy = ORCAfinalenergygrab(basename+".out")
    print("Final energy from external ORCA optimization:", energy)

    return energy

def make_molden_file_ORCA(GBWfile, orcadir=None):
    print_line_with_mainheader("make_molden_file_ORCA")

    #Check for ORCA dir
    orcadir = check_ORCA_location(orcadir)

    print("Inputfile:", GBWfile) 
    #GBWfile should be ORCA file. Can be SCF GBW (.gbw) or natural orbital WF file (.nat)

    #Renaming file if GBW extension as orca_mkl needs it
    if '.gbw' not in GBWfile:
        newfile=GBWfile+'.gbw'
        print("Making copy of file:", newfile)
        shutil.copy(GBWfile,newfile)
    else:
        newfile=GBWfile

    #Now removing suffix
    GBWfile_noext=newfile.split('.gbw')[0]
    print("GBWfile_noext:", GBWfile_noext)
    #Create molden file from el.gbw
    print("Calling orca_2mkl to create molden file:")
    sp.call([orcadir+'/orca_2mkl', GBWfile_noext, '-molden'])
    moldenfile=GBWfile_noext+'.molden.input'
    print("Created molden file:", moldenfile)

    return moldenfile



# Simple Wrapper around orca_mapspc
def run_orca_mapspc(filename, option, start=0.0, end=100, unit='eV', broadening=1.0, points=5000, orcadir=None):

    print("-"*30)
    print("run_orca_mapspc function")
    print("-"*30)
    print(f"option: {option}")
    print(f"start: {start}")
    print(f"end: {end}")
    print(f"unit: {unit}")
    print(f"broadening: {broadening}")
    print(f"points: {points}")
    print(f"orcadir: {orcadir}")

    orcadir = check_ORCA_location(orcadir)
    p = sp.run([orcadir + '/orca_mapspc', filename, option, f"-{unit}" f"-w{broadening}", f"-n{points}"], encoding='ascii')

#Simple function to get elems and coordinates from ORCA outputfile
#Should read both single-point and optimization jobs correctly
def grab_coordinates_from_ORCA_output(filename):
    opt=False
    opt_converged=False
    grab=False
    elems=[]
    coords=[]
    with open(filename) as f:
        for line in f:
            if 'Geometry Optimization Run' in line:
                opt=True
            if 'FINAL ENERGY EVALUATION AT THE STATIONARY POINT' in line:
                opt_converged=True
            if grab is True:
                if len(line) >35:
                    elems.append(line.split()[0])
                    c_x=float(line.split()[1]); c_y=float(line.split()[2]); c_z=float(line.split()[3])
                    coords.append([c_x, c_y, c_z])
                elif len(line) < 10:
       	       	    grab=False
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                if opt is True:
                    if opt_converged is True:
                        grab=True
                else:
                    grab=True
    npcoords=np.array(coords)
    return elems, npcoords



#Make an ORCA fragment guess
def orca_frag_guess(fragment=None, theory=None, A_indices=None, B_indices=None, A_charge=None, B_charge=None, A_mult=None, B_mult=None):
    print_line_with_mainheader("orca_frag_guess")

    elems_A= [fragment.elems[i] for i in A_indices]
    coords_A= np.take(fragment.coords, A_indices, axis=0)

    elems_B= [fragment.elems[i] for i in B_indices]
    coords_B= np.take(fragment.coords, B_indices, axis=0)

    fragment_A = ash.Fragment(elems=elems_A, coords=coords_A, charge=A_charge, mult=A_mult)
    fragment_B = ash.Fragment(elems=elems_B, coords=coords_B, charge=B_charge, mult=B_mult)

    #Early exits
    if fragment is None:
        print("You need to provide an ASH fragment")
        ashexit()
    if fragment.charge == None or A_mult == None or B_mult == None or A_charge == None or B_charge == None:
        print("You must provide charge/multiplicity information to all fragments")
        ashexit()
    if theory == None or theory.__class__.__name__ != "ORCATheory":
        print("You must provide an ORCATheory level")
        ashexit()
    
    #Creating copies of theory object provided
    calc_AB = copy.copy(theory); calc_AB.filename="calcAB"
    calc_A = copy.copy(theory); calc_A.filename="calcA"
    calc_B = copy.copy(theory); calc_B.filename="calcB"

    #-------------------------
    #Calculation on A
    #------------------------
    print("-"*120)
    print("Performing ORCA calculation on fragment A")
    print("-"*120)
    #Run A SP
    result_calcA=ash.Singlepoint(theory=calc_A, fragment=fragment_A)

    #-------------------------
    #Calculation on B
    #------------------------
    print()
    print("-"*120)
    print("Performing ORCA calculation on fragment B")
    print("-"*120)
    #Run B SP
    result_calcB=ash.Singlepoint(theory=calc_B, fragment=fragment_B)

    #-----------------------------------------
    # merge A + B to get promolecular density
    #-----------------------------------------
    print()
    print("-"*120)
    print("Using orca_mergefrag to combine GBW-files for A and B into AB :")
    print("-"*120)
    p = sp.run(['orca_mergefrag', "calcA.gbw", "calcB.gbw", "orca_frag_guess.gbw"], encoding='ascii')

    print("Created new GBW-file: orca_frag_guess.gbw")
    return "orca_frag_guess.gbw"


#Find localized orbitals in ORCA outputfile for a given element
#Return orbital indices (to be fed into run_orca_plot)
def orblocfind(outputfile, atomindex_strings=None, popthreshold=0.1):
    #Elements of interest
    # Threshold for including orbitals
    #popthreshold=0.1

    #Atelemindices: 0 => 0Fe
    #for a in atomindices:
    #    atomindices

    print("Finding localized orbitals for atomindex_strings:", atomindex_strings, "with threshold:", popthreshold)

    ##################
    dict_alpha={}
    dict_beta={}

    stronggrab=False
    bondgrab=False
    delocgrab=False
    with open(outputfile) as f:
        for line in f:
            if 'More delocalized orbitals:' in line:
                #print(line)
                #print("deloc switch")
                stronggrab=False
                bondgrab=False
                delocgrab=True
            if 'ORCA ORBITAL LOCALIZATION' in line:
                loc=True
            if 'Operator                                 ... 0' in line:
                operator='alpha'
            if 'Operator                                 ... 1' in line:
                operator='beta'
            if 'Rather strongly localized orbitals:' in line:
                stronggrab=True
            if stronggrab == True:
                for atindex in atomindex_strings:
                    if str(atindex) in line:
                        #print(line)
                        if operator == 'alpha':
                            atom=line.split()[2]
                            monumber=line.split()[1][:-1]
                            dict_alpha.setdefault(atom, []).append(int(monumber))
                        elif operator == 'beta':
                            atom=line.split()[2]
                            monumber=line.split()[1][:-1]
                            dict_beta.setdefault(atom, []).append(int(monumber))
                        else:
                            print("neither wtf")
                            exit()
            if bondgrab == True:
                for atx in atomindex_strings:
                    if atx in line:
                        #print(line)
                        for i in line.split():
                            if atx in i:
                                at=i
                                pos=line.split().index(at)
                                if float(line.split()[pos+2]) > popthreshold:
                                    if operator == 'alpha':
                                        atom=line.split()[pos]
                                        monumber=line.split()[1][:-1]
                                        dict_alpha.setdefault(atom, []).append(int(monumber))
                                    elif operator == 'beta':
                                        atom=line.split()[pos]
                                        monumber=line.split()[1][:-1]
                                        #print("atom is", atom, "and monumber is", monumber)
                                        dict_beta.setdefault(atom, []).append(int(monumber))
                                    else:
                                        print("neither wtf")
                                        exit()
            if delocgrab == True:
                if 'More delocalized orbita' not in line:
                    for atx in atomindex_strings:
                        if atx in line:
                            #print(line)
                            linechanged=line.replace("-","")
                            #print(linechanged)
                            for i in linechanged.split():
                                if atx in i:
                                    at=i
                                    #print("at is", at)
                                    pos=linechanged.split().index(at)
                                    #print("pos is", pos)
                                    #print("float(line.split()[pos+2]) is", float(line.split()[pos+2]))
                                    if float(linechanged.split()[pos+1]) > popthreshold:
                                        if operator == 'alpha':
                                            atom=linechanged.split()[pos]
                                            monumber=linechanged.split()[1][:-1]
                                            dict_alpha.setdefault(atom, []).append(int(monumber))
                                        elif operator == 'beta':
                                            atom=linechanged.split()[pos]
                                            monumber=linechanged.split()[1][:-1]
                                            dict_beta.setdefault(atom, []).append(int(monumber))
                                        else:
                                            print("neither wtf")
                                            exit()
            if 'Bond-like localized orbitals:' in line:
                stronggrab=False
                bondgrab=True
            if 'Localized MO\'s were stored in:' in line:
                stronggrab=False
                bondgrab=False
                delocgrab=False

    #print("Alpha orbitals")
    #print(dict_alpha)

    #print("Beta orbitals")
    #print(dict_beta)

    alphalist=[]
    betalist=[]
    for avals in dict_alpha.items():
        j=sorted(avals[1])
        #Deleting metal s and p orbitals
        #j.pop(0);j.pop(0);j.pop(0);j.pop(0)
        for i in j:
            alphalist.append(i)

    for bvals in dict_beta.items():
        j=sorted(bvals[1])
        #Deleting metal s and p orbitals
        #j.pop(0);j.pop(0);j.pop(0);j.pop(0)
        for i in j:
            betalist.append(i)

    alphalist=sorted(list(set(alphalist)))
    betalist=sorted(list(set(betalist)))
    print("")

    print("Alpha orbitals to be plotted with getorbitals:")
    print(*alphalist, sep=' ')
    print("Beta orbitals to be plotted with getorbitals:")
    print(*betalist, sep=' ')

    return alphalist, betalist