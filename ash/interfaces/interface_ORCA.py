import subprocess as sp
import os
import shutil
import time
import multiprocessing as mp
import numpy as np
import glob
import copy

import ash.modules.module_coords
from ash.functions.functions_general import ashexit,insert_line_into_file,BC,print_time_rel, print_line_with_mainheader, pygrep2, \
    pygrep, search_list_of_lists_for_index,print_if_level, writestringtofile, check_program_location
from ash.modules.module_singlepoint import Singlepoint
from ash.modules.module_coords import check_charge_mult
import ash.functions.functions_elstructure
import ash.constants
import ash.settings_ash
import ash.functions.functions_parallel


# ORCA Theory object.
class ORCATheory:
    def __init__(self, orcadir=None, orcasimpleinput='', printlevel=2, basis_per_element=None, extrabasisatoms=None, extrabasis=None, atom_specific_basis_dict=None, ecp_dict=None, TDDFT=False, TDDFTroots=5, FollowRoot=1,
                 orcablocks='', extraline='', first_iteration_input=None, brokensym=None, HSmult=None, atomstoflip=None, numcores=1, nprocs=None, label="ORCA",
                 moreadfile=None, moreadfile_always=False, bind_to_core_option=True, ignore_ORCA_error=False,
                 autostart=True, propertyblock=None, save_output_with_label=False, keep_each_run_output=False, print_population_analysis=False, filename="orca", check_for_errors=True, check_for_warnings=True,
                 fragment_indices=None, xdm=False, xdm_a1=None, xdm_a2=None, xdm_func=None, NMF=False, NMF_sigma=None,
                 cpcm_radii=None, ROHF_UHF_swap=False):
        print_line_with_mainheader("ORCATheory initialization")


        self.theorynamelabel="ORCA"
        self.theorytype="QM"
        self.analytic_hessian=True

        # Making sure we have a working ORCA location
        print("Checking for ORCA location")
        self.orcadir = check_ORCA_location(orcadir, modulename="ORCATheory")
        # Making sure ORCA binary works (and is not orca the screenreader)
        check_ORCAbinary(self.orcadir)
        # Checking OpenMPI
        if numcores != 1:
            print(f"ORCA parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version (for the ORCA version) is available in your environment")
            ash.functions.functions_parallel.check_OpenMPI()

        # Bind to core option when calling ORCA: i.e. execute: /path/to/orca file.inp "--bind-to none"
        # TODO: Default False; make True?
        self.bind_to_core_option=bind_to_core_option
        print("bind_to_core_option:", self.bind_to_core_option)

        # Checking if user added Opt, Freq keywords
        if ' OPT' in orcasimpleinput.upper() or ' FREQ' in orcasimpleinput.upper() :
            print(BC.FAIL,"Error. orcasimpleinput variable can not contain ORCA job-directives like: Opt, Freq, Numfreq", BC.END)
            print("String:", orcasimpleinput.upper())
            print("orcasimpleinput should only contain information on electronic-structure method (e.g. functional), basis set, grid, SCF convergence etc.")
            ashexit()

        # Whether to check ORCA outputfile for errors and warnings or not
        # Generally recommended. Could be disabled to speed up I/O a tiny bit
        self.check_for_errors=check_for_errors
        self.check_for_warnings=check_for_warnings

        # Counter for how often ORCATheory.run is called
        self.runcalls=0

        # Whether to keep the ORCA outputfile for each run as orca_runX.out
        self.keep_each_run_output=keep_each_run_output
        # Whether to save ORCA outputfile with given label
        if save_output_with_label is True and label is None:
            print("Error: save_output_with_label option requires a label keyword also")
            ashexit()
        else:
            self.save_output_with_label=save_output_with_label

        # Print population_analysis in each run
        self.print_population_analysis=print_population_analysis

        # Label to distinguish different ORCA objects
        self.label=label

        # Create inputfile with generic name
        self.filename=filename

        # Whether to exit ORCA if subprocess command faile
        self.ignore_ORCA_error=ignore_ORCA_error


        # MOREAD-file
        self.moreadfile=moreadfile
        self.moreadfile_always=moreadfile_always
        # Autostart
        self.autostart=autostart
        # Each ORCA calculation will save path to last GBW-file used in case we have switched directories
        # and we want to use last one
        self.path_to_last_gbwfile_used=None #default None

        # Printlevel
        self.printlevel=printlevel

        # TDDFT
        self.TDDFT=TDDFT
        self.TDDFTroots=TDDFTroots
        self.FollowRoot=FollowRoot

        # Setting numcores of object
        # NOTE: nprocs is deprecated but kept on for a bit
        if nprocs is None:
            self.numcores=numcores
        else:
            self.numcores=nprocs

        # Property block. Added after coordinates unless None
        self.propertyblock=propertyblock

        # Store optional properties of ORCA run job in a dict
        self.properties ={}

        # Adding NoAutostart keyword to extraline if requested
        if self.autostart is False:
            self.extraline=extraline+"\n! Noautostart"
        else:
            self.extraline=extraline

        # Inputfile definitions
        self.orcasimpleinput=orcasimpleinput
        self.orcablocks=orcablocks

        # Input-lines only for first run call
        if first_iteration_input is not None:
            self.first_iteration_input = first_iteration_input
        else:
            self.first_iteration_input=""

        # BROKEN SYM OPTIONS
        self.brokensym=brokensym
        self.HSmult=HSmult
        if isinstance(atomstoflip, int):
            print(BC.FAIL,"Error: atomstoflip should be list of integers (e.g. [0] or [2,3,5]), not a single integer.", BC.END)
            ashexit()
        if atomstoflip is not None:
            self.atomstoflip=atomstoflip
        else:
            self.atomstoflip=[]
        # Basis sets per element
        self.basis_per_element=basis_per_element
        if self.basis_per_element is not None:
            print("Basis set dictionary for each element provided:", basis_per_element)

        # Extrabasis: add specific basis set keyword to certain atoms
        if extrabasisatoms is not None:
            self.extrabasisatoms=extrabasisatoms
            self.extrabasis=extrabasis
        else:
            self.extrabasisatoms=[]
            self.extrabasis=""
        # Atom-specific basis set options
        # Within ORCA inputfile, define a basis set for each and every atom. Requires a dictionary with element as key and basis set as value
        self.atom_specific_basis_dict=atom_specific_basis_dict
        self.ecp_dict=ecp_dict #ECP dict that usually goes with atom_specific dict

        # Used in the case of counterpoise calculations
        self.ghostatoms = [] #Adds ":" in front of element in coordinate block. Have basis functions and grid points
        self.dummyatoms = [] #Adds DA instead of element. No real atom

        # For ORCA calculations that define fragments within molecule
        self.fragment_indices = fragment_indices

        # self.qmatoms need to be set for Flipspin to work for QM/MM job.
        # Overwritten by QMMMtheory, used in Flip-spin
        self.qmatoms=[]

        # Whether to keep a copy of last output (filename_last.out) or not
        self.keep_last_output=True

        # NMF
        self.NMF=NMF
        if self.NMF is True:
            if NMF_sigma is None:
                print("NMF option requires setting NMF_sigma")
                ashexit()
            self.NMF_sigma=NMF_sigma

            print("NMF option is active. Will activate Fermi-smearing in ORCA input!")
            NMF_smeartemp = self.NMF_sigma / ash.constants.R_gasconst
            print(f"NMF_smeartemp = {NMF_smeartemp} calculated from NMF_sigma: {self.NMF_sigma}:")
            self.orcablocks=self.orcablocks+f"""
%scf
fracocc true
smeartemp {NMF_smeartemp}
end
            """

        #TDDFT option
        #If gradient requested by Singlepoint(Grad=True) or Optimizer then TDDFT gradient is calculated instead
        if self.TDDFT is True:
            if '%tddft' not in self.orcablocks:
                self.orcablocks=self.orcablocks+f"""
%tddft
nroots {self.TDDFTroots}
IRoot {self.FollowRoot}
end
"""
        #ROHF-UHF swap
        self.ROHF_UHF_swap=ROHF_UHF_swap

        #Specific CPCM radii. e.g. to use DRACO radii
        if cpcm_radii is not None:
            print("CPCM radii provided:", cpcm_radii)
            #if len(cpcm_radii) != len(c:
            #    print("Error: Number of radii provided does not match number of elements in molecule")
            #    ashexit()
            cpcm_block="%cpcm\n"
            for i,radius in enumerate(cpcm_radii):
                cpcm_block= cpcm_block+ f"AtomRadii({i},  {radius})\n"
            cpcm_block=cpcm_block+"end\n"
            print("cpcm_block:", cpcm_block)
            self.orcablocks=self.orcablocks+cpcm_block

        #XDM: if True then we add !AIM to input
        self.xdm=False
        if xdm is True:
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
        #Keeping outputfiles
        #list_files.append(self.filename + '.out')
        list_files.append(self.filename + '.gbw')
        list_files.append(self.filename + '.densities')
        list_files.append(self.filename + '.ges')
        list_files.append(self.filename + '.prop')
        list_files.append(self.filename + '.uco')
        list_files.append(self.filename + '_property.txt')
        list_files.append(self.filename + '.inp')
        list_files.append(self.filename + '.engrad')
        list_files.append(self.filename + '.cis')
        list_files.append(self.filename + '_last.out')
        list_files.append(self.filename + '.xyz')
        for file in list_files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
        try:
            for tmpfile in glob.glob("self.filename*tmp"):
                os.remove(tmpfile)
        except FileNotFoundError:
            pass

    # Do an ORCA-optimization instead of ASH optimization. Useful for gas-phase chemistry when ORCA-optimizer is better than geomeTRIC
    def Opt(self, fragment=None, Grad=None, Hessian=None, numcores=None, charge=None, mult=None):

        module_init_time=time.time()
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING INTERNAL ORCA OPTIMIZATION-------------", BC.END)
        # Coords provided to run or else taken from initialization.
        # if len(current_coords) != 0:

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
        if self.propertyblock != None:
            print(self.propertyblock)
        print("Charge: {}  Mult: {}".format(charge, mult))

        # TODO: Make more general
        create_orca_input_plain(self.filename, elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                charge, mult, extraline=self.extraline, HSmult=self.HSmult, moreadfile=self.moreadfile)
        print(BC.OKGREEN, f"ORCA Calculation started using {numcores} CPU cores", BC.END)
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
    # Method to grab dipole moment from an ORCA outputfile (assumes run has been executed)
    def get_dipole_moment(self):
        dm = grab_dipole_moment(self.filename+'.out')
        print("Dipole moment:", dm)
        return dm
    def get_polarizability_tensor(self):
        polarizability,diag_pz = grab_polarizability_tensor(self.filename+'.out')
        return polarizability
    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, charge=None, mult=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
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
            print("Error:no current_coords")
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

        # Basis set definition per element from input dict
        if self.basis_per_element != None:
            basisstring=""
            for el,b in self.basis_per_element.items():
                basisstring += f"newgto {el} \"{b}\" end\n"
            basisblock=f"""
%basis
{basisstring}
end"""

            if basisblock not in self.orcablocks:
                self.orcablocks = self.orcablocks + basisblock

        # If ECP-dict provided (often goes with atom_specific_basis_dict)
        if self.ecp_dict != None:
            bstring=""
            for el,b in self.ecp_dict.items():
                for x in b:
                    bstring += f"{x}"
            ecpbasisblock=f"""
%basis
{bstring}
end"""
            if ecpbasisblock not in self.orcablocks:
                self.orcablocks = self.orcablocks + ecpbasisblock

        if self.printlevel >= 2:
            print("Running ORCA with {} cores available".format(numcores))

        # MOREAD. Checking file provided exists and determining what to do if not
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
                    print_if_level(f"Found a path ({self.path_to_last_gbwfile_used}) to last GBW-file used by this Theory object. Will try to copy this file do current dir", self.printlevel,2)
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

        if PC is True:
            if self.printlevel >= 2:
                print("Pointcharge embedding is on!")
            create_orca_pcfile(self.filename, current_MM_coords, MMcharges)
            if self.brokensym == True:
                create_orca_input_pc(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        charge, mult, extraline=extraline, HSmult=self.HSmult, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                     atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock,
                                     fragment_indices=fragment_indices, atom_specific_basis_dict=self.atom_specific_basis_dict, ROHF_UHF_swap=self.ROHF_UHF_swap)
            else:
                create_orca_input_pc(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        charge, mult, extraline=extraline, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                        extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock,
                                        fragment_indices=fragment_indices, atom_specific_basis_dict=self.atom_specific_basis_dict, ROHF_UHF_swap=self.ROHF_UHF_swap)
        else:
            if self.brokensym == True:
                create_orca_input_plain(self.filename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        charge,mult, extraline=extraline, HSmult=self.HSmult, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                     atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock,
                                     ghostatoms=self.ghostatoms, dummyatoms=self.dummyatoms, ROHF_UHF_swap=self.ROHF_UHF_swap,
                                     fragment_indices=fragment_indices, atom_specific_basis_dict=self.atom_specific_basis_dict)
            else:
                create_orca_input_plain(self.filename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        charge,mult, extraline=extraline, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                        extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis, propertyblock=self.propertyblock,
                                        ghostatoms=self.ghostatoms, dummyatoms=self.dummyatoms,ROHF_UHF_swap=self.ROHF_UHF_swap,
                                        fragment_indices=fragment_indices, atom_specific_basis_dict=self.atom_specific_basis_dict)

        # Run inputfile using ORCA parallelization. Take numcores argument.
        # print(BC.OKGREEN, "------------Running ORCA calculation-------------", BC.END)
        if self.printlevel >= 2:
            print(BC.OKGREEN, "ORCA Calculation starting.", BC.END)

        run_orca_SP_ORCApar(self.orcadir, self.filename + '.inp', numcores=numcores, bind_to_core_option=self.bind_to_core_option,
                                check_for_errors=self.check_for_errors, check_for_warnings=self.check_for_warnings, ignore_ORCA_error=self.ignore_ORCA_error)
        if self.printlevel >= 1:
            print(BC.OKGREEN, "ORCA Calculation done.", BC.END)

        outfile=self.filename+'.out'
        engradfile=self.filename+'.engrad'
        pcgradfile=self.filename+'.pcgrad'

        #Checking if finished.
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
        else:
            print("There was an ORCA error that was ignored by user-input")

        if self.ROHF_UHF_swap:
            print("\nROHF UHF swap feature active.")
            print("This means that a $new_job ORCA job was run with a ROHF-UHF noiter switch")
            print(f"Note that the relevant GBW file is then: {self.filename}_job2.gbw\n")
            print("Stored as self.gbwfile of this ORCATheory object")
            self.gbwfile=self.filename+'_job2.gbw'
        else:
            self.gbwfile=self.filename+'.gbw'

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
        else:
            self.energy=ORCAfinalenergygrab(outfile)

            if self.energy is None:
                print("No energy could be found in ORCA outputfile.")
                print("Setting energy to 0.0 and returning")
                return 0.0
        #NMF
        if self.NMF is True:
            print("NMF option is active.")
            E_NMF = self.energy
            occupations = np.array(SCF_FODocc_grab(outfile))
            print("Fractional ccupations (Fermi distribution):", occupations)
            print("Now also calculating correlation energy from the fractional occupation numbers")
            print("Assuming Fermi distribution")
            Ec = ash.functions.functions_elstructure.get_ec_entropy(occupations, self.NMF_sigma, method='fermi')
            print("Ec:", Ec)
            self.properties["NMF_occupations"] = occupations
            self.properties["E_NMF"] = E_NMF
            self.properties["NMF_Ec"] = Ec
            self.energy = self.energy + Ec

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

        #TDDFT results
        if self.TDDFT is True:
            transition_energies = tddftgrab(f"{self.filename}.out")
            transition_intensities = tddftintens_grab(f"{self.filename}.out")

            self.properties["TDDFT_transition_energies"] = transition_energies
            self.properties["TDDFT_transition_intensities"] = transition_intensities

        #Grab timings from ORCA output
        orca_timings = ORCAtimingsgrab(outfile)

        #Initializing zero gradient array
        self.grad = np.zeros((len(qm_elems), 3))
        self.dipole_moment = None

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

        #Grab Hessian if calculated
        if Hessian is True:
            print("Reading Hessian from file:", self.filename+".hess")
            self.hessian = Hessgrab(self.filename+".hess")

            self.ir_intensities = grab_IR_intensities(self.filename+'.hess')

        if Grad == True:
            grad =ORCAgradientgrab(engradfile)
            self.grad = self.grad + grad
            if self.printlevel >= 3:
                print("ORCA gradient:", self.grad)

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

def check_ORCA_location(orcadir, modulename="ORCATheory"):
    if orcadir != None:
        finalorcadir = orcadir
        print(BC.OKGREEN,f"Using orcadir path provided: {finalorcadir}", BC.END)
    else:
        print(BC.WARNING, f"No orcadir argument passed to {modulename}. Attempting to find orcadir variable in ASH settings file (~/ash_user_settings.ini)", BC.END)
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
        palstring='%pal \nnprocs {}\nend'.format(numcores)
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
        warn_l = pygrep2(warnstring, filename, errors="ignore")
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
        error_l = pygrep2(errstring, filename, errors="ignore")
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
    with open(file, errors="ignore") as f:
        for line in f:
            if 'SCF CONVERGED AFTER' in line:
                iter=line.split()[-3]
            if 'TOTAL RUN TIME:' in line:
                return True,iter
    return False,None
def checkORCAOptfinished(file):
    converged=False
    with open(file, errors="ignore") as f:
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
    with open(file, errors=errors) as f:
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
    if Energy is None:
        print(BC.FAIL,"ASH found no energy in file:", file, BC.END)
        print(BC.FAIL,"Something went wrong with ORCA run. Check ORCA outputfile:", file, BC.END)
        print(BC.OKBLUE,BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
        return None
    return Energy


#Grab ORCA timings. Return dictionary
def ORCAtimingsgrab(file):
    timings={} #in seconds
    try:
        with open(file, errors="ignore") as f:
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

def grab_dipole_moment(outfile):
    dipole_moment = []
    with open(outfile) as f:
        for line in f:
            if 'Total Dipole Moment    :' in line:
                dipole_moment.append(float(line.split()[-3]))
                dipole_moment.append(float(line.split()[-2]))
                dipole_moment.append(float(line.split()[-1]))
    return dipole_moment

def grab_polarizability_tensor(outfile):
    pz_tensor = np.zeros((3,3))
    diag_pz_tensor=[]
    count=0
    grab=False;grab2=False
    with open(outfile) as f:
        for line in f:
            if grab2 is True:
                if len(line.split()) == 0:
                    grab2=False
                else:
                    diag_pz_tensor.append(float(line.split()[0]))
                    diag_pz_tensor.append(float(line.split()[1]))
                    diag_pz_tensor.append(float(line.split()[2]))
            if grab is True:
                if 'diagonalized tensor:' in line:
                    grab=False
                    grab2=True
                if len(line.split()) == 3:
                    pz_tensor[count,0]=float(line.split()[0])
                    pz_tensor[count,1]=float(line.split()[1])
                    pz_tensor[count,2]=float(line.split()[2])
                    count+=1
            if 'THE POLARIZABILITY TENSOR' in line:
                grab=True
    return pz_tensor, diag_pz_tensor

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

#Grab TDDFT state energies from ORCA output
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

#Grab TDDFT state intensities from ORCA output
def tddftintens_grab(file):
    intensities=[]
    tddftgrab=False
    with open(file) as f:
        for line in f:
            if tddftgrab==True:
                if len(line.split()) == 8:
                        intensities.append(float(line.split()[3]))
                if len(line.split()) == 0:
                    tddftgrab=False
            if 'State   Energy    Wavelength  fosc         T2        TX        TY        TZ' in line:
                tddftgrab=True
    return intensities

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

def grab_IR_intensities(filename):
    grab=False
    intensities=[]
    with open(filename) as f:
        for line in f:
            print(line)
            if grab:
                if len(line.split()) == 6:
                    intens = float(line.split()[2])
                    intensities.append(intens)
            if '$ir_spectrum' in line:
                grab=True
    return intensities

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
    numatoms=None
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

            if hesstake==True and lastchunk==True:
                if len(line.split()) == hessdim - shiftpar +1:
                    for i in range(0,hessdim - shiftpar):
                        hessarray2d[j,i+shiftpar]=line.split()[i+1]
                    j+=1
            elif hesstake==True and len(line.split()) == 5:
                continue
                #Headerline
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
                         HSmult=None, atomstoflip=None, Hessian=False, extrabasisatoms=None, extrabasis=None, atom_specific_basis_dict=None, extraspecialbasisatoms=None, extraspecialbasis=None,
                         moreadfile=None, propertyblock=None, fragment_indices=None, ROHF_UHF_swap=False):
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
            #Atom-specific basis-dict option (new basis set definition for each atom)
            elif atom_specific_basis_dict is not None:
                print("Writing atom-specific basis for atom:", i)
                #Regular line
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
                for bline in atom_specific_basis_dict[(el,i)]:
                    orcafile.write(str(bline))
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
        # For ROHF job, add newjob and switch to UHF noiter
        if ROHF_UHF_swap:
            newjobline=f"""\n$new_job
{orcasimpleinput.replace("ROHF","UHF noiter ")}
{orcablockinput}
* xyz {charge} {mult}
"""
            orcafile.write(newjobline)
            for i,(el,c) in enumerate(zip(elems,coords)):
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
            orcafile.write('*\n')
        
        
#Create simple ORCA inputfile from elems,coords, input, charge, mult,pointcharges
#Allows for extraline that could be another '!' line or block-inputline.
def create_orca_input_plain(name,elems,coords,orcasimpleinput,orcablockinput,charge,mult, Grad=False, Hessian=False, extraline='',
                            HSmult=None, atomstoflip=None, extrabasis=None, extrabasisatoms=None, moreadfile=None, propertyblock=None,
                            ghostatoms=None, dummyatoms=None,fragment_indices=None, atom_specific_basis_dict=None, ROHF_UHF_swap=False):
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
            #Extra basis on each atom
            if i in extrabasisatoms:
                orcafile.write('{} {} {} {} newgto \"{}\" end\n'.format(el,c[0], c[1], c[2], extrabasis))
            #Atom-specific basis-dict option (new basis set definition for each atom)
            elif atom_specific_basis_dict is not None:
                print("Writing atom-specific basis for atom:", i)
                #Regular line
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
                for bline in atom_specific_basis_dict[(el,i)]:
                    orcafile.write(str(bline))
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
        # For ROHF job, add newjob and switch to UHF noiter
        if ROHF_UHF_swap:
            newjobline=f"""\n$new_job
{orcasimpleinput.replace("ROHF","UHF noiter ")}
{orcablockinput}
* xyz {charge} {mult}
"""
            orcafile.write(newjobline)
            for i,(el,c) in enumerate(zip(elems,coords)):
                orcafile.write('{} {} {} {} \n'.format(el,c[0], c[1], c[2]))
            orcafile.write('*\n')
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
    column=None
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
    print("Gridvalue:", gridvalue)
    orcadir = check_ORCA_location(orcadir, modulename="run_orca_plot")
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
        print("Plotting density")
        if specify_density is True:
            print("specify_density: True. Picking density filename:", densityfilename)
            #Choosing e.g. MRCI density
            p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                input=f'5\n7\n4\n{gridvalue}\n1\n{plottype}\nn\n{densityfilename}\n11\n12\n\n', encoding='ascii')
        else:
            p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                       input=f'5\n7\n4\n{gridvalue}\n1\n{plottype}\ny\n11\n12\n\n', encoding='ascii')
    elif option=='mo':
        p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                       input=f'5\n7\n4\n{gridvalue}\n3\n{mo_operator}\n2\n{mo_number}\n11\n12\n\n', encoding='ascii')
    #If plotting CIS/TDDFT density then we tell orca_plot explicity.
    elif option == 'cisdensity' or option == 'cisspindensity':
        p = sp.run([orcadir + '/orca_plot', filename, '-i'], stdout=sp.PIPE,
                       input=f'5\n7\n4\n{gridvalue}\n1\n{plottype}\nn\n{densityfilename}\n11\n12\n\n', encoding='ascii')

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



def SCF_FODocc_grab(filename):
    occgrab=False
    occupations=[]
    with open(filename) as f:
        for line in f:
            if occgrab is True:
                if '***********' in line:
                    return occupations
                if ' SPIN DOWN' in line:
                    occgrab=False
                    return occupations
                if len(line.split()) == 4:
                    if '  NO   OCC' not in line:
                        occupations.append(float(line.split()[1]))
            if 'SPIN UP ORBITALS' in line or 'ORBITAL ENERGIES' in line:
                occgrab=True
    return occupations

def UHF_natocc_grab(filename):
    natoccgrab=False
    natoccupations=[]
    with open(filename) as f:
        for line in f:
            if natoccgrab==True:
                if 'LOEWDIN' in line:
                    natoccgrab=False
                if 'N' in line:
                    for s in line.split(' '):
                        try:
                            occ = float(s)
                            natoccupations.append(occ)
                        except:
                            pass
            if 'UHF NATURAL ORBITALS' in line:
                natoccgrab=True
    return natoccupations


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

def CCSD_natocc_grab(filename):
    natoccgrab=False
    natoccupations=[]
    with open(filename) as f:
        for line in f:
            if natoccgrab==True:
                if 'N' in line:
                    natoccupations.append(float(line.split()[-1]))
                if ' .... done' in line:
                    natoccgrab=False
            if 'Natural Orbital Occupation Numbers:' in line:
                natoccgrab=True
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

def MRCI_natocc_grab(filename):
    natoccgrab=False
    natoccupations=[]
    with open(filename) as f:
        for line in f:
            if natoccgrab==True:
                if 'N[' in line:
                    natoccupations.append(float(line.split()[-1]))
                if '  -> stored natural orbitals' in line:
                    natoccgrab=False
                    return natoccupations
            if 'NATURAL ORBITAL GENERATION' in line:
                natoccgrab=True
    return natoccupations

def FIC_natocc_grab(filename):
    natoccgrab=False
    natoccupations=[]
    with open(filename) as f:
        for line in f:
            if natoccgrab:
                if 'N[' in line:
                    natoccupations.append(float(line.split()[-1]))
                if ' --- Storing natural' in line:
                    natoccgrab=False
                    return natoccupations
            if 'Natural Orbital Occupation Numbers:' in line:
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


def grab_Mossbauer_parameters(filename,printlevel=2):
    rho=None
    deltaEQ=None
    with open(filename) as f:
        for line in f:
            if ' Delta-EQ' in line:
                deltaEQ=float(line.split()[-2])
            if ' RHO(0)=' in line:
                rho=float(line.split()[1])
    if printlevel == 2:
        print("Mossbauer parameters:")
        print("DeltaEQ:", deltaEQ)
        print("RHO(0):", rho)
    return rho, deltaEQ

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
    orcadir = check_ORCA_location(orcadir, modulename="ORCA_External_Optimizer")
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

def make_molden_file_ORCA(GBWfile, orcadir=None, printlevel=2):
    print_line_with_mainheader("make_molden_file_ORCA")

    #Check for ORCA dir
    orcadir = check_ORCA_location(orcadir, modulename="make_molden_file_ORCA")

    print("Inputfile:", GBWfile)
    #GBWfile should be ORCA file. Can be SCF GBW (.gbw) or natural orbital WF file (.nat)
    renamefile=False
    #Renaming file if GBW extension as orca_mkl needs it
    if '.gbw' not in GBWfile:
        newfile=GBWfile+'.gbw'
        print("Making copy of file:", newfile)
        shutil.copy(GBWfile,newfile)
        renamefile=True
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
    os.rename(moldenfile,GBWfile_noext+'.molden')
    print("Renaming to:", GBWfile_noext+'.molden')

    if renamefile is True:
        print("Removing copy of file:", newfile)
        os.remove(newfile)

    return GBWfile_noext+'.molden'



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

    orcadir = check_ORCA_location(orcadir, modulename="run_orca_mapspc")
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

#Reverse JSON to GBW
def create_GBW_from_json_file(jsonfile, orcadir=None):

    orcafile_basename = jsonfile.split('.')[0]
    orcadir = check_ORCA_location(orcadir, modulename="create_GBW_from_json_file")
    print("Calling orca_2json to convert JSON-file to GBW-file")
    sp.call([orcadir+'/orca_2json', jsonfile, '-gbw'])

    return f"{orcafile_basename}_copy.gbw"

#Using orca_2json to create JSON file from ORCA GBW file
#Format options: json, bson, ubjson, msgpack
def create_ORCA_json_file(file, orcadir=None, format="json", basis_set=True, mo_coeffs=True, one_el_integrals=True,
                          two_el_integrals=False, two_el_integrals_type="ALL", dipole_integrals=False, full_int_transform=False):
    print("create_ORCA_json_file")
    orcadir = check_ORCA_location(orcadir, modulename="create_ORCA_json_file")
    #orcafile_basename = file.split('.')[0]
    orcafile_basename='.'.join(file.split(".")[0:-1])

    print(f"Creating {orcafile_basename}json.conf file")

    prop_1e_integrals_line=""
    two_el_integrals_line=""
    basis_set_line=""
    mo_coeff_line=""
    #NOTE: problems with FullTrafo (orca_2json crashes)
    if full_int_transform is True:
        full_transform_integrals_line="\"FullTrafo\": true,"
    else:
        #Needs to be empty string
        full_transform_integrals_line=""
    if basis_set is True:
        print("Requesting printout of basis set")
        basis_set_line="\"Basisset\": true,"
    if dipole_integrals is True:
        print("Requesting printout of dipole integrals")
        prop_1e_integrals_line="\"1elPropertyIntegrals\": [\"dipole\"],"
    if one_el_integrals is True:
        print("Requesting printout of 1-electron integrals")
        one_el_integrals_line="\"1elIntegrals\": [\"H\",\"S\", \"T\", \"V\"],"
    if two_el_integrals is True:
        print("Requesting printout of 2-electron integrals")
        if two_el_integrals_type == "ALL":
            print("Warning: two_el_integrals_type set to ALL. This means all 2-electron integrals (a lot!)")
            two_el_integrals_line=f"\"2elIntegrals\": [\"MO_PQRS\", \"MO_PRQS\"],"
        else:
            two_el_integrals_line=f"\"2elIntegrals\": [\"MO_{two_el_integrals_type}\"],"
    if mo_coeffs is True:
        print("Requesting printout of MO coefficients")
        mo_coeff_line="\"MOCoefficients\": true,"
    print("here")
    confstring=f"""{{

{mo_coeff_line}
{basis_set_line}
{one_el_integrals_line}
{prop_1e_integrals_line}
{two_el_integrals_line}
{full_transform_integrals_line}
"Densities": ["all"],
"JSONFormats": ["{format}"]
}}
"""

#"JSONFormats": ["json", "bson", "msgpack"]
    with open(f"{orcafile_basename}.json.conf", "w") as conffile:
        conffile.write(confstring)

    #Creating copy of conf-file so that orca_2json picks up abnormal name
    #shutil.copy(f"{orcafile_basename}.json.conf", )

    print("Calling orca_2json to get JSON/BSON file:")
    #Note: ORCA6 changed from basename to file
    print("file:", file)
    sp.call([orcadir+'/orca_2json', file, f'-{format}'])

    # This is better when filename contains multiple .
    jsonfile='.'.join(file.split(".")[0:-1])+f'.{format}'
    print(f"Created file:", jsonfile)
    mb_size = (os.path.getsize(jsonfile))/(1024*1024)
    print(f"Size: {mb_size:7.1f} MB")
    return jsonfile

#Parse ORCA json file
#Good for getting MO-coefficients, MO-energies, basis set, H,S,T matrices, densities etc.
def read_ORCA_json_file(file):
    print("read_ORCA_json_file")
    print("File:", file)
    # Parsing of files
    orjson_loaded=False
    try:
        print("Trying to import orjson")
        import orjson as jsonlib
        print("orjson loaded")
        orjson_loaded=True
    except ModuleNotFoundError:
        print("orjson library not found (recommended for fast reading)")
        print("Trying ujson instead")
        try:
            import ujson as jsonlib
            print("ujson loaded")
        except ModuleNotFoundError:
            print("ujson library not found either")
            print("can be installed like this: pip install ujson")
            print("Falling back to standard json library (slower)")
            import json as jsonlib

    orcafile_basename='.'.join(file.split(".")[0:-1])
    print("orcafile_basename:", orcafile_basename)
    print("Opening file")
    print()
    #Loading
    with open(f"{orcafile_basename}.json") as f:
        if orjson_loaded:
            data = jsonlib.loads(f.read())
        else:
            data = jsonlib.load(f)
    print("Looping over dictionary")
    print("")
    for i in data["Molecule"]:
        print(i)

    print("ORCA Header:", data["ORCA Header"])
    #print("Molecule:", data["Molecule"])
    #Molecule
    print("Molecule-Atoms:", data["Molecule"]["Atoms"])
    print("Molecule-BaseName:", data["Molecule"]["BaseName"])
    print("Molecule-Charge:", data["Molecule"]["Charge"])
    print("Molecule-Multiplicity:", data["Molecule"]["Multiplicity"])
    print("Molecule-CoordinateUnits:", data["Molecule"]["CoordinateUnits"])
    print("Molecule-HFTyp:", data["Molecule"]["HFTyp"])
    print()
    print("Densities found:", data["Molecule"]["Densities"].keys())
    print("Dictionary keys of data", data["Molecule"].keys())
    #Note: only returning sub-dict Molecule
    return data["Molecule"]

def write_ORCA_json_file(data,filename="ORCA_ASH.json", ORCA_version="6.0.0"):
    print("write_ORCA_json_file")
    print("Filename:", filename)
    import json as jsonlib

    #Add header if missing from datadict
    final_data={}
    if "ORCA Header" not in data:
        final_data["ORCA Header"] = {"Version":ORCA_version}
        final_data["Molecule"]=data
    else:
        final_data=data
    #print("data:", data)
    with open(filename, "w") as f:
        #f.write(jsonlib.dumps(data))
        jsonlib.dump(final_data, f,indent=2)

    return filename

def read_ORCA_msgpack_file(file):
    print("read_ORCA_msgpack_file function")
    print("Trying to import msgspec")
    msgspec_loaded=False
    msgpack_loaded=False

    try:
        import msgspec
        msgspec_loaded=True
        print("Imported msgspec successfully")
    except ModuleNotFoundError:
        print("Problem importing msgspec (pip install msgspec)")
        print("Trying msgpack library")
        try:
            import msgpack
            msgpack_loaded=True
            print("Imported msgpack successfully")
        except ModuleNotFoundError:
            print("msgpack not found.")
            print("Install like this: pip install msgpack")
            ashexit()
    
    # Read msgpack file
    with open(file, "rb") as data_file:
        if msgspec_loaded:
            data = msgspec.msgpack.decode(data_file.read())
        elif msgpack_loaded:
            byte_data = data_file.read()
            data = msgpack.unpackb(byte_data)

    return data["Molecule"]

# Read BSON files using independent BSON codec for Python (not MongoDB)
#Msgpack probably better
def read_ORCA_bson_file(bsonfile):
    try:
        print("Importing bson")
        import bson
    except ImportError:
        print("Error: bson module not found. Please install bson module")
        print("See: https://pypi.org/project/bson/ and https://github.com/py-bson/bson")
        print("pip install bson")
        ashexit()

    print("reading BSON file:", bsonfile)
    with open(bsonfile, 'rb') as f:
        content = f.read()
        base = 0
        while base < len(content):
            base, data = bson.decode_document(content, base)
    return data["Molecule"]

def get_densities_from_ORCA_json(data):
    DMs={}
    for d in data["Densities"]:
        print(d)
        DMs[d] = np.array(data["Densities"][d])
    print("Found the following densities: ", DMs.keys())
    return DMs

#Grab ORCA wfn from jsonfile or data-dictionary
def grab_ORCA_wfn(data=None, jsonfile=None, density=None):
    print_line_with_mainheader("grab_ORCA_wfn")

    #If neither data object or dictionary was provided
    if data == None and jsonfile == None:
        print("grab_ORCA_wfn requires either data dictionary or jsonfile as input")
        ashexit()
    elif data != None and jsonfile != None:
        print("grab_ORCA_wfn requires either data dictionary or jsonfile as input")
        ashexit()
    elif data != None:
        print("Data dictionary provided")
        pass
    elif jsonfile != None:
        print("JSON file provided. Reading")

        if '.json' in jsonfile:
            data = read_ORCA_json_file(jsonfile)
        elif '.bson' in jsonfile:
            data = read_ORCA_bson_file(jsonfile)
        elif '.msgpack' in jsonfile:
            data = read_ORCA_msgpack_file(jsonfile)
        else:
            print("Unknown file")
            ashexit()

    if density == None:
        print("Error: You must pick a density option")
        print("Available densities in data object:")
        for d in data["Densities"]:
            print(d)
        ashexit()
    #Grab chosen density
    DM_AO = np.array(data["Densities"][density])
    print("DM_AO:", DM_AO)

    #Get the AO-basis
    AO_basis = data["Atoms"]
    AO_order = data["MolecularOrbitals"]["OrbitalLabels"]

    #Get the overlap
    S = np.array(data["S-Matrix"])

    #Grabbing C (MO coefficients)
    mos = data["MolecularOrbitals"]["MOs"]
    C = np.array([m["MOCoefficients"] for m in mos])
    #Transposing C so that rows are AOs and columns are MOs
    C = np.transpose(C) #NOTE IMPORTANT
    print("Warning: matrix C was transposed so that rows are AOs and columns are MOs")

    #MO energies and occupations
    MO_energies = np.array([m["OrbitalEnergy"] for m in mos])
    MO_occs = np.array([m["Occupancy"] for m in mos])

    print("MO_energies:", MO_energies)
    print("MO_occs:", MO_occs)
    print("MO coeffs:", C)

    return DM_AO,C,S, MO_occs, MO_energies, AO_basis, AO_order


#Function to prepare ORCA orbitals for another ORCA calculation
#Mainly for getting natural orbitals
def ORCA_orbital_setup(orbitals_option=None, fragment=None, basis=None, basisblock="", extrablock="", extrainput="", label="frag",
        MP2_density=None, MDCI_density=None, AutoCI_density=None, memory=10000, numcores=1, charge=None, mult=None, moreadfile=None,
        gtol=2.50e-04, nmin=1.98, nmax=0.02, CAS_nel=None, CAS_norb=None,CASCI=False, natorb_iterations=None,
        FOBO_excitation_options=None, MRCI_natorbiterations=0, MRCI_tsel=1e-6,
        ROHF=False, ROHF_case=None, MP2_nat_step=False, MREOMtype="MR-EOM",
        NMF=False, NMF_sigma=None):

    print_line_with_mainheader("ORCA_orbital_setup")


    if fragment is None:
        print("Error: No fragment provided to ORCA_orbital_setup.")
        ashexit()

    if basis is None:
        print("Error: No basis keyword provided to ORCA_orbital_setup. This is necessary")
        print("Note: you can additionally used basisblock string to provide additional basis options or override keyword")
        ashexit()

    if orbitals_option is None:
        print("Error: No orbitals_option keyword provided to ORCA_orbital_setup. This is necessary")
        print("orbitals_option: MP2, RI-MP2, CCSD, CCSD(T), DLPNO-CCSD, QCISD, CEPA/1, NCPF/1, HF, MRCI, CEPA2")
        ashexit()

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, "QM", fragment, "ORCA_orbital_setup", theory=None)

    #ORBITALS_OPTIONS
    #If ROHF only is requested we activate the ROHF procedure
    if 'ROHF' in orbitals_option:
        print("ROHF orbitals_option requested")
        ROHF=True
    if MP2_nat_step is True and MP2_density is None:
            print("Error: MP2_density must be provided for MP2_nat_step")
            print("Options: unrelaxed or relaxed")
            ashexit()
    if 'MP2' in orbitals_option:
        print("MP2-type orbitals requested. This means that natural orbitals will be created from the chosen MP2 density")
        if MP2_density is None:
            print("Error: MP2_density must be provided")
            print("Options: unrelaxed or relaxed")
            ashexit()
        print("MP2_density option:", MP2_density)
    if orbitals_option == 'CCSD':
        MDCIkeyword="CCSD"
        print("CCSD-type orbitals requested. This means that natural orbitals will be created from the chosen MDCI_density ")
        if MDCI_density is None:
            print("Error: MDCI_density must be provided")
            print("Options: linearized, unrelaxed or orbopt")
            ashexit()
        print("MDCI_density option:", MDCI_density)
    if orbitals_option == "DLPNO-CCSD":
        MDCIkeyword="DLPNO-CCSD"
        print("DLPNO-CCSD-type orbitals requested. This means that natural orbitals will be created from the chosen MDCI_density ")
        if MDCI_density is None:
            print("Error: MDCI_density must be provided")
            print("Options: linearized, unrelaxed or orbopt")
            ashexit()
        print("MDCI_density option:", MDCI_density) 
    if orbitals_option == "CCSD(T)":
        AUTOCIkeyword="AUTOCI-CCSD(T)"
        print("CCSD(T)-type natural orbitals requested.")
        print("Since ORCA 6.0 this is available in the AUTOCI module only")
        if AutoCI_density is None:
            print("Error: AutoCI_density keyword must be provided (not MDCI_density)")
            print("Options: linearized, unrelaxed or orbopt")
            ashexit()
        print("AutoCI_density option:", AutoCI_density)
    if 'QCISD' in orbitals_option:
        MDCIkeyword="QCISD"
        print("QCISD-type orbitals requested. This means that natural orbitals will be created from the chosen MDCI_density")
        if MDCI_density is None:
            print("Error: MDCI_density must be provided")
            print("Options: linearized, unrelaxed or orbopt")
            ashexit()
        print("MDCI_density option:", MDCI_density)
    if 'CEPA/1' in orbitals_option:
        MDCIkeyword="CEPA/1"
        print("CEPA/1-type orbitals requested. This means that natural orbitals will be created from the chosen MDCI_density")
        if MDCI_density is None:
            print("Error: MDCI_density must be provided")
            print("Options: linearized, unrelaxed or orbopt")
            ashexit()
        print("MDCI_density option:", MDCI_density)
    if 'CPF/1' in orbitals_option:
        MDCIkeyword="CPF/1"
        print("CPF/1-type orbitals requested. This means that natural orbitals will be created from the chosen MDCI_density")
        if MDCI_density is None:
            print("Error: MDCI_density must be provided")
            print("Options: linearized, unrelaxed or orbopt")
            ashexit()
        print("MDCI_density option:", MDCI_density)
    #CASSCF
    if 'CASSCF' in orbitals_option:
        print("CASSCF-type orbitals requested. This means that natural orbitals will be created from a CASSCF WF")
        if CAS_nel is None or CAS_nel is None:
            print("CASSCF natural orbitals required CAS_nel and CAS_norb keywords for CAS active space calculation")
            ashexit()
        if CASCI is True:
            print("Warning: CAS-CI is True. No CASSCF orbital optimization will be carried out.")
            print("Warning: To get natural orbitals from CAS-CI calculation we modify gtol instead of using noiter")
            gtol=9999999
    #FOBO
    if 'FOBO' in orbitals_option:
        print("FOBO-type orbitals requested.")

        if CASCI is True:
            print("Warning: CAS-CI is True. No CASSCF orbital optimization will be carried out.")
            extrainput += " noiter "
        elif natorb_iterations is not None:
            print("FOBO and natorbitations active. Turning off CASSCF orbital optimization")
            extrainput += " noiter "

    if 'MR' in orbitals_option:
        print("orbitals_option:", orbitals_option)
        if 'SORCI' in orbitals_option:
            MRCIkeyword="SORCI"
        elif 'ACPF' in orbitals_option:
            MRCIkeyword="MRACPF"
        elif 'AQCC' in orbitals_option:
            MRCIkeyword="MRAQCC"
        elif 'DDCI1' in orbitals_option:
            MRCIkeyword="MRDDCI1"
        elif 'DDCI2'  in orbitals_option:
            MRCIkeyword="MRDDCI2"
        elif 'DDCI3'  in orbitals_option:
            MRCIkeyword="MRDDCI3"
        elif 'MRCI+Q'  in orbitals_option:
            MRCIkeyword="MRCI+Q"
        elif 'MREOM'  in orbitals_option:
            MRCIkeyword=MREOMtype
        else:
            MRCIkeyword="MRCI"
        print("MRCIkeyword:", MRCIkeyword)
        print("MR-type orbitals requested. This means that natural orbitals will be created from the chosen MRCI_density")
        if CAS_nel is None or CAS_nel is None:
            print("MRCI natural orbitals required CAS_nel and CAS_norb keywords for CAS active space calculation")
            ashexit()
        if CASCI is True:
            print("Warning: CAS-CI is True. No CASSCF orbital optimization will be carried out.")
            print("Warning: To get natural orbitals from CAS-CI calculation we modify gtol instead of using noiter")
            gtol=9999999
    if 'FIC' in orbitals_option:
        if 'DDCI1' in orbitals_option:
            MRCIkeyword="FIC-DDCI1"
        AUTOCIkeyword="FIC-DDCI3"
        print("AUTOCIkeyword:", AUTOCIkeyword)
        print("AUTOCI-type orbitals requested. This means that natural orbitals will be created from the chosen AUTOCI density")
        if CAS_nel is None or CAS_nel is None:
            print("AUTOCI natural orbitals required CAS_nel and CAS_norb keywords for CAS active space calculation")
            ashexit()
    if natorb_iterations is not None:
        print("Natural orbital iterations will be performed!")

    #Always starting from scratch, unless moreadfile is provided (will override)
    autostart_option=False

    ######################
    #ROHF option for SCF
    #######################
    if ROHF is True:
        print("ROHF is True. Will first run a ROHF calculation preparatory step")
        print("Warning: ORCA will attempt an ROHF calculation while guessing what type of open-shell case you are dealing with")
        print("If ORCA fails in this step you may have to provide ROHF-case manually via the ROHF_case keyword")
        print("ROHF_case options: 'HighSpin' (typically what you want), 'CAHF', 'SAHF'  (see ORCA manual for details)")
        #ROHF bug in ORCA, necessary for cc-basis sets in ORCA 5:
        if ROHF_case == None:
            rohfcase_line=""
        else:
            rohfcase_line=f"ROHF_case {ROHF_case}"
        rohfblocks=f"""
%shark
usegeneralcontraction false
partialgcflag 0
end
%scf
{rohfcase_line}
end
{extrablock}
"""
        rohf = ash.ORCATheory(orcasimpleinput=f"! ROHF {basis} tightscf notrah  {extrainput}", orcablocks=rohfblocks,
                              numcores=numcores, autostart=autostart_option,
                                 label='ROHF', filename="ROHF", save_output_with_label=True, moreadfile=moreadfile)
        Singlepoint(theory=rohf,fragment=fragment)
        #Now SCF-step is done. Now adding noiter to extrainput and moreadfile
        moreadfile=f"{rohf.filename}.gbw"
        extrainput="noiter"
        print("ROHF step is done. Now adding noiter to extrainput and moreadfile for next step")

    #############################################################
    #Possible initial MP2-nat step before final orbitals
    ##############################################################
    if MP2_nat_step is True:
        print("MP2_nat_step is True. Will run an MP2-natural orbital calculation preparatory step")
        print("MP2_density option:", MP2_density)
        print("moreadfile:", moreadfile)
        mp2blocks=f"""
%mp2
density {MP2_density}
natorbs true
end
"""
        mp2_prep = ash.ORCATheory(orcasimpleinput=f"! RI-MP2 {basis} {extrainput} autoaux tightscf", orcablocks=mp2blocks, numcores=numcores,
                                 label='MP2prep', filename="MP2prep", save_output_with_label=True, moreadfile=moreadfile, autostart=autostart_option)
        Singlepoint(theory=mp2_prep,fragment=fragment)
        #Now MP2-step is done. Now adding noiter to extrainput and moreadfile
        moreadfile=f"{mp2_prep.filename}.mp2nat"
        extrainput="noiter"
        print("MP2-prep step is done. Now adding noiter to extrainput and moreadfile for next step")


    #############################################################
    #FINAL ORBITALS
    ##############################################################
    alldone=False
    #NOTE: HF, UHF-UNO and QRO not yet ready
    if orbitals_option =="HF" or orbitals_option =="RHF" :
        print("Performing HF orbital calculation")
        exit()
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} HF {basis}  tightscf", numcores=numcores,
                                 label='RHF', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.gbw"
        #Dummy occupations
        natoccgrab=UHF_natocc_grab
    elif orbitals_option =="UHF-UNO" or orbitals_option =="UHF" :
        print("Performing UHF natural orbital calculation")
        exit()
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} UHF {basis} normalprint UNO tightscf", numcores=numcores,
                                 label='UHF', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.unso"
        natoccgrab=UHF_natocc_grab
    elif orbitals_option =="UHF-QRO":
        print("Performing UHF-QRO natural orbital calculation")
        exit()
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} UHF {basis}  UNO tightscf", numcores=numcores,
                                 label='UHF-QRO', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.qro"
        natoccgrab=UHF_natocc_grab
    elif orbitals_option =="NMF":
        print("Performing Non-AufBau Mean-Field calculation")
        if NMF_sigma is None:
            print("Error: For orbitals_option NMF, NMF_sigma must also be provided")
            ashexit()
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} {basis}  tightscf", NMF=True, NMF_sigma=NMF_sigma, numcores=numcores,
                                 label='NMF', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.gbw"
        natoccgrab=SCF_FODocc_grab
    elif orbitals_option =="ROHF":
        print("ROHF orbitals_option was chosen. ROHF calculation should already have been carried out.")
        print("Returning")
        alldone=True
        mofile=f"{rohf.filename}.gbw"
        natoccgrab=None
        nat_occupations=None
    elif orbitals_option =="MP2" :
        mp2blocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %mp2
        natorbs true
        density {MP2_density}
        end
        """
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} MP2 {basis} autoaux tightscf", orcablocks=mp2blocks, numcores=numcores,
                                 label='MP2', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.mp2nat"
        natoccgrab=MP2_natocc_grab
    elif orbitals_option =="RI-MP2" :
        mp2blocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %mp2
        natorbs true
        density {MP2_density}
        end
        """
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} RI-MP2 {basis} autoaux tightscf", orcablocks=mp2blocks, numcores=numcores,
                                 label='RIMP2', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.mp2nat"
        natoccgrab=MP2_natocc_grab
    elif orbitals_option =="RI-SCS-MP2" :
        mp2blocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %mp2
        natorbs true
        density {MP2_density}
        end
        """
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} RI-SCS-MP2 {basis} autoaux tightscf", orcablocks=mp2blocks, numcores=numcores,
                                 label='RI-SCS-MP2', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.mp2nat"
        natoccgrab=MP2_natocc_grab
    elif orbitals_option =="OO-RI-MP2" :
        mp2blocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %mp2
        natorbs true
        density relaxed
        end
        """
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} OO-RI-MP2 {basis} autoaux tightscf", orcablocks=mp2blocks, numcores=numcores,
                                 label='OO-RI-MP2', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.mp2nat"
        natoccgrab=MP2_natocc_grab
    elif orbitals_option =="CCSD" or orbitals_option =="QCISD" or orbitals_option =="CEPA/1" or orbitals_option =="CPF/1":
        ccsdblocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %mdci
        natorbs true
        density {MDCI_density}
        end
        """
        mdcilabel = MDCIkeyword.replace("/","") #To avoid / in CEPA/1 etc
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} {MDCIkeyword} {basis} autoaux tightscf", orcablocks=ccsdblocks, numcores=numcores,
                                 label=mdcilabel, save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        #mofile=f"{natorbs.filename}.mdci.nat"
        #natoccgrab=CCSD_natocc_grab
        #For open-shell systems we get unrestricted natorbs it seems
        #Now diagonalizing manually
        natoccgrab=None
    elif orbitals_option =="DLPNO-CCSD":
        #NOTE: Due to a bug in ORCA version 5 and 6.0.0
        # Requesting DLPNO-CCSD natural orbitals from any density results in a wrong coupled cluster problem
        # Hence we have to request density calculation alone and then diagonalize to get the natural orbitals manually
        ccsdblocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %mdci
        density {MDCI_density}
        end
        """
        mdcilabel = MDCIkeyword.replace("/","") #To avoid / in CEPA/1 etc
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} {MDCIkeyword} {basis} autoaux tightscf", orcablocks=ccsdblocks, numcores=numcores,
                                 label=mdcilabel, save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        natoccgrab=None
    elif 'CASSCF' in orbitals_option:
        casscfblocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %casscf
        gtol {gtol}
        nel {CAS_nel}
        norb {CAS_norb}
        end
        """
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} CASSCF {basis} tightscf", orcablocks=casscfblocks, numcores=numcores,
                                 label='CASSCF', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.gbw"
        natoccgrab=CASSCF_natocc_grab
    elif 'MR' in orbitals_option:
        mrciblocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %casscf
        gtol {gtol}
        nel {CAS_nel}
        norb {CAS_norb}
        end
        %mrci
        natorbs 2
        tsel {MRCI_tsel}
        end
        """
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} {MRCIkeyword} {basis} autoaux tightscf", orcablocks=mrciblocks, numcores=numcores,
                                 label=MRCIkeyword, save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.b0_s0.nat"

        def dummy(f): return f
        natoccgrab=dummy
        print("Warning: can not get full natural occupations from MRCI+Q calculation")
    elif 'CCSD(T)' in orbitals_option:
        autociblocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %autoci
        density {AutoCI_density}
        natorbs true
        end
        """
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} {AUTOCIkeyword} {basis} autoaux tightscf", orcablocks=autociblocks, numcores=numcores,
                                 label=AUTOCIkeyword, save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        natoccgrab=None
    elif 'FIC' in orbitals_option:
        autociblocks=f"""
        %maxcore {memory}
        {basisblock}
        {extrablock}
        %casscf
        gtol {gtol}
        nel {CAS_nel}
        norb {CAS_norb}
        end
        %autoci
        density {AutoCI_density}
        natorbs true
        end
        """
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} {AUTOCIkeyword} {basis} autoaux tightscf", orcablocks=autociblocks, numcores=numcores,
                                 label=AUTOCIkeyword, save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"ficddci3.mult.{mult}.root.0.FIC-DDCI3.nat"
        natoccgrab=FIC_natocc_grab

    elif 'FOBO' in orbitals_option:
        #Defining a ROHF-type CASSCF
        if CAS_nel is None or CAS_norb is None:
            print("CAS_nel and CAS_norb not given. Guessing ROHF-type CASSCF based on multiplicity")
            CAS_nel=mult-1
            CAS_norb=mult-1
            print(f"CAS_nel: {CAS_nel} CAS_norb:{CAS_norb}")

        if CAS_nel == 0:
            print("Closed-shell system. Adding AllowRHF keyword")
            extrainput+="AllowRHF"
        #FOBO_options
        if FOBO_excitation_options is None:
            FOBO_excitation_options = {'1h':1,'1p':1,'1h1p':1,'2h':0,'2h1p':1}
            print("Using default FOBO_excitation_options:", FOBO_excitation_options)
        #exit()
        mrciblocks=f"""
%maxcore {memory}
{basisblock}
{extrablock}
%casscf
  nel {CAS_nel}
  norb {CAS_norb}
end

%mrci
newblock {mult} *
   excitations none
   refs cas({CAS_nel},{CAS_norb}) end
   Flags[is]   {FOBO_excitation_options['1h']}
   Flags[sa]   {FOBO_excitation_options['1p']}
   Flags[ia]   {FOBO_excitation_options['1h1p']}
   Flags[ijss] {FOBO_excitation_options['2h']}
   Flags[ijsa] {FOBO_excitation_options['2h1p']}
   nroots      1
end
AllSingles true
PrintWF det
natorbiters {MRCI_natorbiterations}
natorbs 2
tsel {MRCI_tsel}
end
"""
        #NOTE: Added noiter here to prevent CASSCF module from messing with orbitals
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput}  {basis} autoaux tightscf", orcablocks=mrciblocks, numcores=numcores,
                                 label='FOBO', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.b0_s0.nat"
        def dummy(f): return f
        natoccgrab=dummy
        print("Warning: can not get full natural occupations from FOBO calculation")
    else:
        print("Error: orbitals_option not recognized")
        ashexit()

    #Run natorb calculation unless everything is done
    if alldone is False:

        if natorb_iterations:
            print("Natural-orbital iterations option is ON!")
            print(f"Will run {natorb_iterations} natural-orbital iterations")
            for n_i in range(0,natorb_iterations):

                print(f"Now running natorb iteration {n_i}")
                ash.Singlepoint(theory=natorbs, fragment=fragment, charge=charge, mult=mult)

                if n_i == 0:
                    print("Iteration 0 done. Now setting moread option for next iteration")
                    natorbs.moreadfile=mofile
                    natorbs.moreadfile_always=True

                nat_occupations=natoccgrab(natorbs.filename+'.out')
                if orbitals_option not in ['MRCI','FOBO']:
                    natocc_print(nat_occupations,orbitals_option,nmin,nmax)
                os.rename(f"{natorbs.filename}.out", f"orca_natorbiter_{n_i}.out")
        else:
            print("Now running natorb calculation")
            ash.Singlepoint(theory=natorbs, fragment=fragment, charge=charge, mult=mult)
            if natoccgrab is not None:
                nat_occupations=natoccgrab(natorbs.filename+'.out')
                if orbitals_option not in ['MRCI','FOBO']:
                    natocc_print(nat_occupations,orbitals_option,nmin,nmax)
            #Special issues
            elif orbitals_option == "DLPNO-CCSD":
                print("Warning: DLPNO-CCSD natural orbitals requested (things can go wrong).")
                print("Due to ORCA bug, DLPNO-CCSD natural orbitals come from ASH diagonalization of density")
                mofile,nat_occupations = new_ORCA_natorbsfile_from_density(natorbs.gbwfile,densityname="mdcip",
                    result_file="ORCA_DLPNOCCSD_nat_ASH", ORCA_version="6.0.0", change_from_UHF_to_ROHF=True)
                natocc_print(nat_occupations,orbitals_option,nmin,nmax)
            elif orbitals_option =="CCSD" or orbitals_option =="QCISD" or orbitals_option =="CEPA/1" or orbitals_option =="CPF/1":
                print("Warning: open-shell MDCI natural orbitals give 2 sets of natural orbitals")
                print("Undesirable so we do diagonalization manually")
                mofile,nat_occupations = new_ORCA_natorbsfile_from_density(natorbs.gbwfile,densityname="mdcip",
                    result_file=f"ORCA_MDCI{orbitals_option}_nat_ASH", ORCA_version="6.0.0", change_from_UHF_to_ROHF=True)
                natocc_print(nat_occupations,orbitals_option,nmin,nmax)                
            elif orbitals_option == "CCSD(T)":
                print("Warning: CCSD(T) natural orbitals requested (things can go wrong).")
                print("Natural orbitals not directly available and have to be manually diagonalized from density")
                mofile, nat_occupations = new_ORCA_natorbsfile_from_density(natorbs.gbwfile,densityname="autocipur",
                    result_file="ORCA_CCSD_T_nat_ASH", ORCA_version="6.0.0", change_from_UHF_to_ROHF=True)
                natocc_print(nat_occupations,orbitals_option,nmin,nmax)
            else:
                nat_occupations=[]

    #Renaming mofile (for purposes of having unique mofiles if we run this function multiple times)
    newmofile = label + '_'+mofile
    os.rename(mofile, newmofile)
    print("\nReturning name of orbital file that can be used in next ORCATheory calculation (moreadfile option):", newmofile)
    print("Also returning natural occupations list:", nat_occupations)

    return newmofile, nat_occupations

def natocc_print(nat_occupations,orbitals_option,nmin,nmax):
    print(f"{orbitals_option} Natorb. ccupations:", nat_occupations)
    print("\nTable of natural occupation numbers")
    print(f"Dashed lines indicated occupations within nmin={nmin} and nmax={nmax} window")
    print("")
    print("{:<9} {:6} ".format("Orbital", f"{orbitals_option}-nat-occ"))
    print("----------------------------------------")
    init_flag=False
    final_flag=False
    for index,(nocc) in enumerate(nat_occupations):
        if init_flag is False and nocc<nmin:
            print("-"*40)
            init_flag=True
        if final_flag is False and nocc<nmax:
            print("-"*40)
            final_flag=True
        print(f"{index:<9} {nocc:9.4f}")


#TODO: fix once ORCA6 bugfix is done
# https://orcaforum.kofo.mpg.de/viewtopic.php?f=11&t=11657&p=47529&hilit=vpot#p47529
# Either use input-file option (vpot.inp) or other
def orca_vpot_run(gbwfile, densityfile, orcadir=None, numcores=1, input_points_string=None):
    
    vpotinp=f"""{numcores}                 # Number of parallel processes
       {gbwfile}        # GBW File
       {densityfile}      # Density
       input_points.xyz        # Coordinates
       vpot.out        # Output File
"""
    #TODO: make more flexible
    if input_points_string is None:
        input_points="""
    6
    5.0 0.0 0.0
    -5.0 0.0  0.0
    0.0 5.0  0.0
    0.0-5.0  0.0
    0.0 0.0  5.0
    0.0 0.0 -5.0
    """
    else:
        input_points=input_points_string
    writestringtofile(vpotinp, "vpot.inp")
    writestringtofile("input_points.xyz", input_points)
    orcadir = check_program_location(orcadir,"orcadir", "orca_vpot")

    p = sp.run([orcadir + '/orca_vpot', "vpot.inp", ], stdout=sp.PIPE)

    #vpot.out

    #TODO: Move to module_plotting
    def plot_electrostatic_potential(vpotfile="vpot.out"):
        pass




# Function to create FCIDUMP file 
# Change header_format from FCIDUMP to MRCC to get MRCC fort.55 file
# TODO: SCF-type beyond RHF
def create_ORCA_FCIDUMP(gbwfile, header_format="FCIDUMP", filename="FCIDUMP_ORCA", orca_json_format="msgpack",
                        int_threshold=1e-16,  mult=1, full_int_transform=False,
                        convert_UHF_to_ROHF=True):
    module_init_time=time.time()
    orca_basename=gbwfile.split('.')[0]

    #Create JSON-file
    print("Now creating JSON-file from GBW-file:", gbwfile)
    jsonfile = create_ORCA_json_file(gbwfile, two_el_integrals=True, format=orca_json_format,
                                     full_int_transform=full_int_transform)
    print("jsonfile:", jsonfile)
    print_time_rel(module_init_time, modulename='create_ORCA_FCIDUMP: jsoncreate done', moduleindex=3)
    #Get data from JSON-file as dict
    print("Now reading JSON-file")
    if orca_json_format == "json":
        datadict = read_ORCA_json_file(jsonfile)
    elif orca_json_format == "bson":
        datadict = read_ORCA_bson_file(jsonfile)
    elif orca_json_format == "msgpack":
        datadict = read_ORCA_msgpack_file(jsonfile)
    print_time_rel(module_init_time, modulename='create_ORCA_FCIDUMP: jsonread done', moduleindex=3)
    #Get coordinates from JSON (in Angstrom) and calculate repulsion
    coords = np.array([i["Coords"] for i in datadict["Atoms"]])
    nuc_charges = np.array([i["ElementNumber"] for i in datadict["Atoms"]])
    from ash.modules.module_coords import nuc_nuc_repulsion
    nuc_repulsion = nuc_nuc_repulsion(coords, nuc_charges)

    #MO coefficients (used for 1-elec integrals)
    mos = datadict["MolecularOrbitals"]["MOs"]
    C = np.array([m["MOCoefficients"] for m in mos])

    #Electrons
    occupations = np.array([m["Occupancy"] for m in datadict["MolecularOrbitals"]["MOs"]])
    print("Occupations:", occupations)

    num_tot_orbs = len(occupations)
    print("Total num orbitals:", num_tot_orbs)
    num_occ_orbs = len(np.nonzero(occupations)[0])
    print("Total num ccupied orbitals:", num_occ_orbs)
    num_act_el= int(round(sum(occupations))) #Rounding up to deal with possible non-integer occupations
    print("Number of (active) electrons:", num_act_el)

    WF_assignment = ash.functions.functions_elstructure.check_occupations(occupations)
    print("WF_assignment:", WF_assignment)
    conversion=False
    if WF_assignment == "RHF":
        print("Occupation assignment is RHF")
        print("This is straightforward")
    elif WF_assignment == "ROHF":
        print("Occupation assignment is ROHF")
        print("This should be straightforward")
    elif WF_assignment == "UHF":
        print("Occupation assignment is UHF")
        print("We currently can not handle UHF")
        if convert_UHF_to_ROHF:
            print("convert_UHF_to_ROHF is True")
            print("Will hack UHF WF into ROHF")
            print("Warning: not guaranteed to work")
            num_act_el= int(round(sum(occupations)))
            rohf_num_orbs= int(len(occupations)/2)
            alpha_occupations = occupations[0:rohf_num_orbs]
            beta_occupations = occupations[rohf_num_orbs:]
            excess_alpha = int(sum(alpha_occupations)-sum(beta_occupations))
            #Hacking occupations
            new_occupations = []
            for i in range(0,rohf_num_orbs):
                if beta_occupations[i] == 1.0:
                    new_occupations.append(2.0)
                elif alpha_occupations[i] == 1.0:
                    new_occupations.append(1.0)
                else:
                    new_occupations.append(0.0)
            print("New dummy ROHF occupations:", new_occupations)
            occupations=new_occupations
            WF_assignment="ROHF"
            conversion=True
            #Now proceeding as if were ROHF

            #Half of MO coefficients
            #C = C[0:rohf_num_orbs]
            C = C[:rohf_num_orbs,:rohf_num_orbs]
    elif WF_assignment == "FRACT":
        print("Occupation assignment is FRACT")
        print("This could be problematic")
        print("We will continue, however")
        #MO coefficients


    #Transpose MO coefficients
    MO_coeffs = np.transpose(C)

    print_time_rel(module_init_time, modulename='create_ORCA_FCIDUMP: before int', moduleindex=3)
    #1-electron integrals
    H = np.array(datadict["H-Matrix"])
    #1-elec
    from functools import reduce
    one_el = reduce(np.dot, (MO_coeffs.T, H, MO_coeffs))

    # 2-electron integrals
    twoint = datadict["2elIntegrals"]
    #print("twoint:", twoint)
    #exit()
    mo_COUL_aa = np.array(datadict["2elIntegrals"][f"MO_PQRS"]["alpha/alpha"])
    mo_EXCH_aa = np.array(datadict["2elIntegrals"][f"MO_PRQS"]["alpha/alpha"])

    print_time_rel(module_init_time, modulename='create_ORCA_FCIDUMP: after int', moduleindex=3)
    # Creating integral tensor
    orbind=num_tot_orbs
    two_el_tensor=np.zeros((orbind,orbind,orbind,orbind))

    #Processing Coulomb
    for i in mo_COUL_aa:
        two_el_tensor[int(i[0]), int(i[1]), int(i[2]), int(i[3])] = i[4]
    #Processing Exchange,  NOTE: index swap because Exchange
    for j in mo_EXCH_aa:
        two_el_tensor[int(j[0]), int(j[2]), int(j[1]), int(j[3])] = j[4]
    
    print_time_rel(module_init_time, modulename='create_ORCA_FCIDUMP: intproc done', moduleindex=3)
    #Write file
    from ash.functions.functions_elstructure import ASH_write_integralfile
    ASH_write_integralfile(two_el_integrals=two_el_tensor, one_el_integrals=one_el,
            nuc_repulsion_energy=nuc_repulsion, header_format=header_format,
                                num_corr_el=num_act_el, filename=filename,
                            int_threshold=int_threshold, scf_type=WF_assignment, mult=mult)
    
    print_time_rel(module_init_time, modulename='create_ORCA_FCIDUMP', moduleindex=2)
    return filename



# calculate_natorbs_from_density
# Convenient function to get natural orbitals from any density even if ORCA did create the natural orbitals

def calculate_ORCA_natorbs_from_density(gbwfile,densityname="mdcip"):
    from ash.functions.functions_elstructure import diagonalize_DM_AO
    #JSON file from GBW-file (NOTE: can be regular GBW-file even if we want the MDCI)
    jsonfile = create_ORCA_json_file(gbwfile, format="json", basis_set=True, mo_coeffs=True)
    DM_AO,C,S, MO_occs, MO_energies, AO_basis, AO_order = grab_ORCA_wfn(jsonfile=jsonfile, density=densityname)
    natorb, natocc = diagonalize_DM_AO(DM_AO, S)

    return natorb, natocc

# Get natural orbitals of any calculated density of an ORCA calculation
# Convenient when ORCA natural orbital printing is buggy
# NOTE: Not fully tested
def new_ORCA_natorbsfile_from_density(gbwfile, densityname="mdcip", result_file="ORCA_ASH", ORCA_version="6.0.0",
                                      change_from_UHF_to_ROHF=True):
    from ash.functions.functions_elstructure import diagonalize_DM_AO
    #JSON file from GBW-file (NOTE: can be regular GBW-file even if we want the MDCI)
    jsonfile = create_ORCA_json_file(gbwfile, format="json", basis_set=True, mo_coeffs=True)
    #Read all molecular data from GBW
    mol_data = read_ORCA_json_file(jsonfile)
    #Get Wfn data only
    DM_AO,C,S, MO_occs, MO_energies, AO_basis, AO_order = grab_ORCA_wfn(jsonfile=jsonfile, density=densityname)
    #Diagonalize to get natural orbitals
    natorb, natocc = diagonalize_DM_AO(DM_AO, S)

    print(f"Density {densityname} diagonalized by ASH")
    print("Natural orbital occupations:", natocc)
    natorb_transposed=natorb.T
    #Loop over MOs and replace canonical MOs with NOs
    print("len(natorb_transposed):", len(natorb_transposed))
    new_mos_sublist=[]
    for i in range(0,len(natocc)):
        #Grabbing old MO (may be UHF)
        oldmo = mol_data["MolecularOrbitals"]["MOs"][i]
        #Creating new MO using natorb MO coeffs and occupations. Setting orb energy to 0.0
        #Note: This will delete any beta information
        newmo = {"MOCoefficients":list(natorb_transposed[i]), "Occupancy":natocc[i], "OrbitalEnergy":0.0,
                    "OrbitalSymLabel":oldmo["OrbitalSymLabel"], "OrbitalSymmetry":oldmo["OrbitalSymmetry"]}   
        new_mos_sublist.append(newmo)

    mol_data["MolecularOrbitals"]["MOs"] = new_mos_sublist
    if change_from_UHF_to_ROHF is True:
        print("Changing UHF to ROHF")
        print("Skipping beta")
        mol_data["HFTyp"] = "ROHF"
        #Changing BF list from UHF to ROHF
        mol_data["MolecularOrbitals"]["OrbitalLabels"] = mol_data["MolecularOrbitals"]["OrbitalLabels"][0:int(len(mol_data["MolecularOrbitals"]["OrbitalLabels"])/2)]
        mol_data["Densities"] = "" #Removing densities

    jsonfile = write_ORCA_json_file(mol_data,filename=f"{result_file}_mod.json", ORCA_version=ORCA_version)
    print("New JSON-file created:", jsonfile)
    newgbwfile = create_GBW_from_json_file(jsonfile)

    #orca_basename=jsonfile.split('.')[0:-1]
    #print("orca_basename:", orca_basename)

    return newgbwfile, natocc