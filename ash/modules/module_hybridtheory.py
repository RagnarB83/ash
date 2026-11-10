"""
HybridTheory module:


"""
import time
from ash.interfaces.interface_pyscf import PySCFTheory
from ash.interfaces.interface_ORCA import ORCATheory
from ash.interfaces.interface_OpenMM import OpenMMTheory
import numpy as np
import os
import copy
from collections import defaultdict

from ash.functions.functions_general import BC, ashexit, print_time_rel,print_line_with_mainheader,listdiff, printdebug
from ash.modules.module_theory import Theory
from ash.modules.module_coords_PBC import cell_params_to_vectors, cell_vectors_to_params
from ash.interfaces.interface_ORCA import grabatomcharges_ORCA
from ash.interfaces.interface_xtb import grabatomcharges_xTB,grabatomcharges_xTB_output
from ash.modules.module_QMMM import linkatom_force_adv, linkatom_force_lever
from ash.modules.module_coords import print_coords_for_atoms,get_boundary_atoms,get_linkatom_positions,distance,get_connected_atoms, write_coords_all

class Correction_handler:
    def __init__(self):
        #Correction RMS/Max info
        self.correction_RMSgrad=[]
        self.correction_Maxgrad=[]
        self.correction_E=[]
        #actual theory1 and theory2 RMS/MaxE gradients
        self.theory1_RMSgrad=[]
        self.theory1_Maxgrad=[]
        self.theory2_RMSgrad=[]
        self.theory2_Maxgrad=[]
        #Combined grad info
        self.combined_RMSgrad=[]
        self.combined_Maxgrad=[]

#NOTE: DualTheory and multiprocessing parallelization is not compatible at the moment. Unclear how to make work
#Maybe after each Dualtheory-run by each worker we write a file to disk and read from disk so we know when to update ?
class DualTheory:
    """ASH DualTheory theory.
    Combines two theory levels to give a modified energy and modified gradient
    """
    def __init__(self, theory1=None, theory2=None, printlevel=1, label=None, correctiontype="Difference",
            update_schedule='frequency', update_freq=5, max_updates=100, numcores=1, Maxthreshold=3e-5, minimum_steps=7):
        print("Creating DualTheory object. Correctiontype: ", correctiontype)
        self.theorytype="QM"
        self.theory1=theory1
        self.theory2=theory2
        self.printlevel=printlevel
        self.label=label

        #Setting printlevel for both theories
        self.theory1.printlevel=printlevel
        self.theory2.printlevel=printlevel

        self.numcores=numcores #Needed for compatibility. Not sure if it will be used
        #This is an inputfilename that may be set externally (Singlepoint_par)
        self.filename=""
        self.correctiontype=correctiontype
        self.theory1.filename=self.filename+"theory1"
        self.theory2.filename=self.filename+"theory2"

        #Type of update schedule (default='frequency). Options: 'Robust_start',
        self.update_schedule=update_schedule
        #Minimum number of steps for
        self.minimum_steps=minimum_steps
        #self.RMSthreshold=RMSthreshold
        self.Maxthreshold=Maxthreshold
        #At which iteration do we update correction
        self.update_freq=update_freq

        #Max number of correction updates
        self.max_updates=max_updates

        #Set inital mode of object
        self.set_to_initial_mode()
    #Method that sets the initial mode. Can be called to reset object
    def set_to_initial_mode(self):
        #Booleans to switch modes.
        self.theory1_active=False #Unused for now
        self.theory2_active=False #Used for switching to performing only theory 2
        #Flag for correction. Only used by Robust_start for now
        self.correction_off=False

        #Keep track of total number of run calls
        self.totalruncalls = 0
        #Dictionary that keeps track of how often each calculation label has been run
        self.update_freq_dict={}
        self.correction_dict=defaultdict(Correction_handler)

        #Gradient and energy correction
        self.gradient_correction={}
        self.energy_correction={}

    #First storing RMS/Max Grad info for both theory levels
    def store_gradient_info(self,G, label,type=None):
        max_grad = np.amax(G)
        rms_grad = np.sqrt(np.mean(np.square(G)))
        if type == "T1":
            #Adding RMS/grad to dict
            self.correction_dict[label].theory1_Maxgrad.append(max_grad)
            self.correction_dict[label].theory1_RMSgrad.append(rms_grad)
        elif type == "T2":
            #Adding RMS/grad to dict
            self.correction_dict[label].theory2_Maxgrad.append(max_grad)
            self.correction_dict[label].theory2_RMSgrad.append(rms_grad)
        elif type == "combined":
            #Adding RMS/grad to dict
            self.correction_dict[label].combined_Maxgrad.append(max_grad)
            self.correction_dict[label].combined_RMSgrad.append(rms_grad)
        elif type == "correction":
            self.correction_dict[label].correction_RMSgrad.append(rms_grad)
            self.correction_dict[label].correction_Maxgrad.append(max_grad)

    #Determine correction. Also storing RMS/Max info on combined gradient
    def correction(self,Grad,label,T2_E=None,T1_E=None,T2_G=None,T1_G=None):
        if self.correctiontype=="Difference":
            #Energy correction
            self.energy_correction[label] = T2_E - T1_E
            self.correction_dict[label].correction_E.append(self.energy_correction[label])
            if Grad == True:
                #Gradient correction
                self.gradient_correction[label] = T2_G - T1_G
                #Storing T1, T2 and combined gradient RMS/Max info
                self.store_gradient_info(T1_G, label,type="T1")
                self.store_gradient_info(T2_G, label,type="T2")
                self.store_gradient_info(self.gradient_correction[label], label,type="correction")
        else:
            print("Correctiontype not available")
            ashexit()
    def switch_to_theory(self,num):
        print("Dualtheory object: Switching to theory:", num)
        if num == 2:
            self.theory2_active = True
        elif num == 1:
            self.theory1_active = True
    #def correction_stats(self,label):
    #    print("DualTheory Correction stats:")
    #    max_correction = np.amax(self.gradient_correction[label])
    #    rms_correction = np.sqrt(np.mean(np.square(self.gradient_correction[label])))
    #    if self.printlevel > 1:
    #        print(f"Gradient correction (label:{label}): {self.gradient_correction[label]}")
        #if self.printlevel > 0:
        #    print(f"Energy correction(label:{label}): {self.energy_correction[label]}")
        #    print(f"Max Gradient correction (label:{label}): {max_correction}")
        #    print(f"RMS Gradient correction (label:{label}): {rms_correction}")
        #Adding indfo to dict
    #    self.correction_dict[label].RMScorr.append(rms_correction)
    #    self.correction_dict[label].Maxcorr.append(max_correction)
    #    self.correction_dict[label].Ecorr.append(self.energy_correction[label])

    def set_numcores(self,numcores):
        print(f"Setting new numcores {numcores} for theory 1 and theory2")
        self.theory1.set_numcores(numcores)
        self.theory2.set_numcores(numcores)
    # Cleanup after run.
    def cleanup(self):
        self.theory1.cleanup()
        self.theory2.cleanup()
    def print_status(self,label):
        print("Theory1 Max grad", self.correction_dict[label].theory1_Maxgrad)
        print("Theory1 RMS grad", self.correction_dict[label].theory1_RMSgrad)
        print("Theory2 Max grad", self.correction_dict[label].theory2_Maxgrad)
        print("Theory2 RMS grad", self.correction_dict[label].theory2_RMSgrad)
        print("Combined Max grad", self.correction_dict[label].combined_Maxgrad)
        print("Combined RMS grad", self.correction_dict[label].combined_RMSgrad)
        print("Correction stats:")
        print("self.correction_dict[label].Ecorr:", self.correction_dict[label].correction_E)
        print("self.correction_dict[label].correction_RMSgrad:", self.correction_dict[label].correction_RMSgrad)
        print("self.correction_dict[label].correction_Maxgrad:", self.correction_dict[label].correction_Maxgrad)

    #Write info to disk
    def write_updatefreqdict_to_disk(self, filename="dualtheory_log"):
        print("Inside: write_updatefreqdict_to_disk")
        import json
        try:
            os.remove(filename)
        except:
            pass
        print("self.update_freq_dict:", self.update_freq_dict)
        #d = {"one":1, "two":2}
        #print("d:", d)
        json.dump(self.update_freq_dict, open("dualtheory_log",'w'))
        print("he")
        #exit()
        #json.dump(self.update_freq_dict, open(filename,'w'))
        #with open(filename,'w') as f:
        #    json.dump(self.update_freq_dict, f)
        print("xxx")
        #with open(filename, 'w') as f:
        #    f.write("[Data]\n")
        #    f.write(f"label : {label}\n")
        #    f.write(f"update_freq_dict : {update_freq_dict}\n")
    #Read info from disk
    def read_updatefreqdict_from_disk(self, update_freq_dict, filename="dualtheory_log"):
        print("Inside: read_updatefreqdict_from_disk")
        print("reading file:", filename)
        import json
        if os.path.isfile(filename) is False:
            print(f"File {filename} does not exist")
            print("Continuing")
            #Returning original
            return
        print("here")
        self.update_freq_dict = json.load(open(filename))
        return

    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, elems=None, Grad=False, numcores=None, label='default', charge=None, mult=None, run_both_theories=False,
            current_MM_coords=None, MMcharges=None, qm_elems=None, PC=None ):
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING DUALTHEORY INTERFACE-------------", BC.END)
        self.totalruncalls += 1
        print("running")
        print("label:", label)
        #Reading info from disk (this makes things compatible with multiprocessing)
        self.read_updatefreqdict_from_disk(self.update_freq_dict)

        #If theory2 switch is active then we only do theory2
        if self.theory2_active == True:
            print("Theory2 only mode active")
            if Grad == True:
                T2_energy, T2_grad = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                return T2_energy,T2_grad
            else:
                T2_energy = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                return T2_energy
        #If theory1 switch is active then we only do theory1
        elif self.theory1_active == True:
            print("Theory 1 only mode active")
            if Grad == True:
                T1_energy, T1_grad = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                return T1_energy,T1_grad
            else:
                T1_energy = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                return T1_energy
        #Else we do theory1 with regular theory2 correction
        else:
            print("Theory1 + Theory2-correction mode active")

            ##################
            if self.update_schedule=='Robust_start':
                print("Using update_schedule: Robust_start")
                if self.correction_off==True:
                    print("Correction is currently off.")
                    #TODO: Option to turn correction back on here.
                    print("Theory1 Max grad", self.correction_dict[label].theory1_Maxgrad)
                    print("Theory1 RMS grad", self.correction_dict[label].theory1_RMSgrad)
                    print("Theory2 Max grad", self.correction_dict[label].theory2_Maxgrad)
                    print("Theory2 RMS grad", self.correction_dict[label].theory2_RMSgrad)
                    print("Combined Max grad", self.correction_dict[label].combined_Maxgrad)
                    print("Combined RMS grad", self.correction_dict[label].combined_RMSgrad)
                    print("------------------")
                    #End-gradient threshold.
                    #Using default geometric gradient convergence criteria times factor
                    endgradient_threshold_RMS=1e-4
                    endgradient_threshold_Max=3e-4
                    if self.correction_dict[label].combined_RMSgrad[-1] < endgradient_threshold_RMS:
                        print("Below threshold")
                        print("Switching to Theory2 active")
                        self.theory2_active = True
                    else:
                        print("Keeping correction off until RMSgrad falls below threshold")
                        print("RMSgrad combined:", self.correction_dict[label].combined_RMSgrad[-1])
                        print("RMSThreshold:", endgradient_threshold_RMS)
                        print("Maxgrad combined:", self.correction_dict[label].combined_Maxgrad[-1])
                        print("MaxThreshold:", endgradient_threshold_Max)
                else:
                    print("Correction is currently on.")
                    print("Last set of correction stats:")
                    print("self.correction_dict[label].Ecorr:", self.correction_dict[label].correction_E)
                    print("self.correction_dict[label].correction_RMSgrad:", self.correction_dict[label].correction_RMSgrad)
                    print("self.correction_dict[label].correction_Maxgrad:", self.correction_dict[label].correction_Maxgrad)
                    try:
                        delta_Max=self.correction_dict[label].correction_Maxgrad[-2] - self.correction_dict[label].correction_Maxgrad[-1]
                        delta_RMS=self.correction_dict[label].correction_RMSgrad[-2] - self.correction_dict[label].correction_RMSgrad[-1]
                        print("Delta(Max):", delta_Max)
                        print("Delta(RMS):", delta_RMS)
                    except:
                        pass
                    print("self.Maxthreshold:", self.Maxthreshold)
                    if len(self.correction_dict[label].correction_Maxgrad) == 0:
                        print("First DualTheory call. Running both theories")
                        self.update_freq_dict[label] = [0,0,0] #Runcalls, theory1-runcalls, theory2-runcalls
                        run_both_theories=True
                    elif len(self.correction_dict[label].correction_Maxgrad) < self.minimum_steps:
                        print("Not enough data (< minimum_steps). Running both theories")
                        run_both_theories=True
                    elif abs(delta_Max) < self.Maxthreshold:
                        print(f"delta_Max:{delta_Max} below threshold:{self.Maxthreshold}.")
                        print("Switching Theory2 correction calculation off.")
                        self.correction_off=True
                    else:
                        print("Correction not stabilized. Doing Theory2 correction")
                        run_both_theories=True
            elif self.update_schedule=='frequency':
                print("here")
                print("label:", label)
                print("self.update_freq_dict:", self.update_freq_dict)
                if label not in self.update_freq_dict:
                    print("First DualTheory call. Running both theories")
                    self.update_freq_dict[label] = [0,0,0] #Runcalls, theory1-runcalls, theory2-runcalls
                    run_both_theories=True
                elif label in self.update_freq_dict:
                    print("XXXX")
                    print("self.update_freq_dict:",self.update_freq_dict)

                    #Checking if we have reached max_updates
                    print(f"Number of corrections {self.update_freq_dict[label][2]} for label: {label}")
                    if  self.update_freq_dict[label][2] >= self.max_updates:
                        print("Max number of corrections reached. Skipping correction (theory2 calc) in this step")
                    else:
                        #If runcalls for label matches update_freq
                        if self.update_freq_dict[label][0] % self.update_freq == 0:
                            run_both_theories=True
            else:
                print("Unknown update_schedule")
                ashexit()
            print("YYYY")
            print("self.update_freq_dict:", self.update_freq_dict)
            ################
            self.update_freq_dict[label][0] +=1
            if Grad == True:
                print("Grad True")
                self.update_freq_dict[label][1] +=1
                T1_energy, T1_grad = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                if run_both_theories == True:
                    print("runboth true")
                    self.update_freq_dict[label][2] +=1
                    self.write_updatefreqdict_to_disk(self.update_freq_dict)
                    T2_energy, T2_grad = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                    #Update gradient correction if both theories were calculated
                    self.correction(Grad,label,T2_E=T2_energy,T1_E=T1_energy,T2_G=T2_grad,T1_G=T1_grad)
                    #self.correction_stats(label)
            else:
                self.update_freq_dict[label][1] +=1
                T1_energy = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                if run_both_theories == True:
                    print("Here zz")
                    self.update_freq_dict[label][2] +=1
                    self.write_updatefreqdict_to_disk(self.update_freq_dict)
                    T2_energy = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                    #Update gradient correction if both theories were calculated
                    self.correction(Grad,label,T2_E=T2_energy,T1_E=T1_energy,T2_G=T2_grad,T1_G=T1_grad)
                    #self.correction_stats(label)

        #Current energy
        print("curr energy")
        print("label:", label)
        print("self.energy_correction:", self.energy_correction)
        energy = T1_energy + self.energy_correction[label]
        print("Dualtheory energy:", energy)
        if Grad == True:
            #Combine into current gradient
            gradient = T1_grad + self.gradient_correction[label]

            self.store_gradient_info(gradient, label,type="combined")

            #Print stuff
            if self.printlevel > 2:
                self.print_status(label)

            return energy,gradient
        else:
            return energy



#########################
# WrapTheory class
#########################

class WrapTheory(Theory):
    """ASH WrapTheory theory.
    Combines multiple theories to give a modified energy and modified gradient
    """
    def __init__(self, theory1=None, theory2=None, theories=None, printlevel=2, label=None,
                 theory1_atoms=None, theory2_atoms=None, theory3_atoms=None, theory4_atoms=None,
                 theory5_atoms=None, theories_chargemult=None,
                 theory_operators=None, skip_openmmtheory_run=False,
                 periodic=False, periodic_cell_vectors=None, periodic_cell_dimensions=None,
                 use_linkatoms=True,
                 linkatom_method='ratio', linkatom_simple_distance=None,
                 linkatom_ratio=0.723, linkatom_type='H', linkatom_conn_scale=1.0, linkatom_conn_tolerance=0.4):
        super().__init__()
        self.theorytype="QM"
        self.theory1=theory1
        self.theory2=theory2
        self.printlevel=printlevel
        self.label=label
        self.filename=""
        self.theorynamelabel="WrapTheory"
        self.theories=theories

        print_line_with_mainheader(f"{self.theorynamelabel} initialization")
        print("Creating WrapTheory object")


        if theories is not None :
            print("Theories defined by keyword:", theories)
            if theory1 is not None or theory2 is not None:
                print("Error: Both theory1/theory2 keywords and theories keyword cannot be defined")
                print("Choose either to define: theory1 and theory2       or provide a list of all theories using the theories keyword")
                ashexit()
        else:
            if theory1 is None or theory2 is None:
                print("Error: Either theories keyword or theory1 and theory2 have to be provided to WrapTheory")
                ashexit()
            self.theories=[theory1,theory2]

        # Charge/mult for each theory. e.g. [[0,1],[1,2]]
        self.theories_chargemult=theories_chargemult

        print("Checking if theories_chargemult information has been provided and has correct length (should be a list of lists, e.g. [[0,1],[1,2]]) . ")
        if self.theories_chargemult is not None:
            print(f"theories_chargemult information provided: {self.theories_chargemult}")
            print("Checking if complete")
            if len(self.theories_chargemult) != len(self.theories):
                print(f"Error: Number of theories_chargemult {len(self.theories_chargemult)} is not equal to number of theories specified {len(self.theories)}")
                ashexit()
        else:
            print("Warning: theories_chargemult information not provided.")
            print("This means that each theory will assume the same charge/mult information for all theories. Charge/mult information needs to be provided via either Fragment or Job-function in this case")

        # Option to have theory only calculate certain atoms
        self.theory1_atoms=theory1_atoms
        self.theory2_atoms=theory2_atoms
        self.theory3_atoms=theory3_atoms
        self.theory4_atoms=theory4_atoms
        self.theory5_atoms=theory5_atoms


        # Whether to skip the OpenMM theory run or not (applies to MD)
        self.skip_openmmtheory_run=skip_openmmtheory_run

        # Linkatoms False by default. Later checked.
        self.linkatoms = use_linkatoms
        self.conn_scale = linkatom_conn_scale
        self.conn_tolerance = linkatom_conn_tolerance
        self._boundary_cache = {}
        # Linkatom method strategy to determine linkatom position or QM-L distance
        self.linkatom_type=linkatom_type # Usually 'H'
        self.linkatom_method = linkatom_method # Options: 'simple' or 'ratio'
        self.linkatom_simple_distance = linkatom_simple_distance # For method simple, Default 1.09 Angstrom
        # For method ratio. see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9314059/
        self.linkatom_ratio=linkatom_ratio
        if self.linkatom_method == "simple":
            self.linkatom_forceproj_method="adv"
        elif self.linkatom_method == "ratio":
            self.linkatom_forceproj_method="lever"

        # PBC
        self.periodic=periodic
        if self.periodic:
            if periodic_cell_dimensions is not None:
                print("periodic_cell_dimensions:", periodic_cell_dimensions)
                self.periodic_cell_dimensions = periodic_cell_dimensions
                # Convert to cell vectors
                self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)
            elif periodic_cell_vectors is not None:
                self.periodic_cell_vectors = periodic_cell_vectors
                self.periodic_cell_dimensions = cell_vectors_to_params(periodic_cell_vectors)


        # Operators: '+' or '-' for each theory
        # By default we sum
        self.theory_operators=theory_operators
        if self.theory_operators is not None:
            print("self.theory_operators option active!")
            if self.theory_operators == "average" or self.theory_operators == "ensemble":
                print("Using average/ensemble operator. This means that the energies and gradients of all theories will be averaged.")
            else:
                print("self.theory_operators:", self.theory_operators)
                if len(self.theory_operators) != len(self.theories):
                    print(f"Error: Number of theory-operators {len(self.theory_operators)} is not equal to number of theories {len(self.theories)}")
                    ashexit()

    # Return cell gradient if periodic is True
    # Find first theory that has a cell gradient and return it
    def get_cell_gradient(self):
        self.cell_gradient = None
        # Loop over theories and find the first theory that has a cell gradient
        for theory in self.theories:
            if hasattr(theory, 'cell_gradient') and theory.cell_gradient is not None:
                self.cell_gradient = theory.cell_gradient
        return self.cell_gradient

    # Update cell using either periodic_cell_vectors or periodic_cell_dimensions
    def update_cell(self,periodic_cell_vectors=None, periodic_cell_dimensions=None):
        print("Updating cell vectors")
        if periodic_cell_vectors is not None:
            self.periodic_cell_vectors = periodic_cell_vectors

            self.periodic_cell_dimensions = cell_vectors_to_params(periodic_cell_vectors)
        elif periodic_cell_dimensions is not None:
            self.periodic_cell_dimensions=periodic_cell_dimensions

            self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)

    def _get_theory_atoms(self, i):
        atoms_list = [self.theory1_atoms, self.theory2_atoms, self.theory3_atoms,
                    self.theory4_atoms, self.theory5_atoms]
        return atoms_list[i] if i < len(atoms_list) else None

    def _get_boundaryatoms(self, i, region_atoms, current_coords, elems):
        """Boundary atoms (QM1:MM1 dict) for theory i, determined once and cached"""
        if i not in self._boundary_cache:
            self._boundary_cache[i] = get_boundary_atoms(
                region_atoms, current_coords, elems, self.conn_scale, self.conn_tolerance,
                excludeboundaryatomlist=None, unusualboundary=None)
            print(f"Theory {i+1}: boundary atoms (region:outside):", self._boundary_cache[i])
        return self._boundary_cache[i]

    def create_linkatoms(self, boundaryatoms, current_coords, region_atoms, elems):
        checkpoint = time.time()
        linkatoms_dict = get_linkatom_positions(
            boundaryatoms, region_atoms, current_coords, elems,
            linkatom_type=self.linkatom_type, linkatom_method=self.linkatom_method,
            linkatom_simple_distance=self.linkatom_simple_distance,
            linkatom_ratio=self.linkatom_ratio)
        pairs = sorted(linkatoms_dict.keys())   # defines linkatom ordering (appended after region atoms)
        coords = np.array([linkatoms_dict[p] for p in pairs])
        print_time_rel(checkpoint, modulename='create_linkatoms', moduleindex=3,
                    currprintlevel=self.printlevel, currthreshold=2)
        return linkatoms_dict, pairs, coords

    def linkatom_force_projection(self, pairs, linkatoms_dict, region_atoms, regiongrad, 
                                  fullgrad, current_coords):
        """Project linkatom gradients onto the QM1 (in region) and MM1 (outside region) atoms.
    regiongrad: gradient from theory, shape (n_region + n_linkatoms, 3)
    fullgrad:   (N_full,3) array already holding regiongrad[:n_region] at region_atoms
        """

        nreg = len(region_atoms)
        for i, pair in enumerate(pairs):
            printdebug(f"i: {i}, pair: {pair}")
            Lgrad = regiongrad[nreg + i]
            Lcoord = linkatoms_dict[pair]
            Qcoord = current_coords[pair[0]]
            Mcoord = current_coords[pair[1]]
            method = self.linkatom_forceproj_method.lower()
            if method == "adv":
                print("adv")
                QM1grad, MM1grad = linkatom_force_adv(Qcoord, Mcoord, Lcoord, Lgrad)
            elif method == "lever":
                QM1grad, MM1grad = linkatom_force_lever(Qcoord, Mcoord, Lcoord, Lgrad)
            else:
                print("Unknown linkatom_forceproj_method:", self.linkatom_forceproj_method)
                ashexit()
            if self.printlevel > 2:
                print(f"Linkatom {pair}: QM1 contrib {QM1grad}, MM1 contrib {MM1grad}")
            fullgrad[pair[0]] += QM1grad
            fullgrad[pair[1]] += MM1grad
        return fullgrad


    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):

        module_init_time = time.time()
        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        full_dimension=current_coords.shape[0]
        energies=[]
        gradients=[]
        pc_gradient=None
        current_coords = np.asarray(current_coords)
        #chosen_coords=current_coords
        #chosen_elems=qm_elems
        for i,theory in enumerate(self.theories):
            print(f"Now running Theory: {theory.theorynamelabel}")

            if isinstance(theory, OpenMMTheory) and self.skip_openmmtheory_run is True:
                print("skip_openmmtheory_run is True. Skipping OpenMMTheory run. Setting to zero energy and zero gradient.")
                energy = 0.0
                grad = np.zeros((full_dimension,3))
                energies.append(energy)
                gradients.append(grad)
                continue

            # Reset every iteration (fixes leaking between theories)
            atoms = self._get_theory_atoms(i)
            chosen_coords, chosen_elems = current_coords, qm_elems
            linkinfo = None

            if atoms is not None:
                print(f"theory{i+1}_atoms has been set:", atoms)
                chosen_coords = np.take(current_coords, atoms, axis=0)
                chosen_elems = [qm_elems[j] for j in atoms]
                if self.linkatoms:
                    boundaryatoms = self._get_boundaryatoms(i, atoms, current_coords, qm_elems)
                    if len(boundaryatoms) > 0:
                        ldict, pairs, lcoords = self.create_linkatoms(boundaryatoms, current_coords,
                                                                    atoms, qm_elems)
                        linkinfo = (ldict, pairs)
                        chosen_coords = np.append(chosen_coords, lcoords, axis=0)
                        chosen_elems = chosen_elems + [self.linkatom_type]*len(pairs)
                        print(f"Theory {i+1}: added {len(pairs)} linkatoms")
            # Charge/mult information for each theory. If not provided, use the charge/mult of the system
            if self.theories_chargemult is not None:
                theory_charge, theory_mult = self.theories_chargemult[i]

                if theory_charge is None or theory_mult is None:
                    print("Error: Charge or multiplicity for theory", i+1, "has not been correctly specified in theories_chargemult. Please provide charge/mult for each theory.")
                    ashexit()
                print(f"Using charge={theory_charge} and mult={theory_mult} for this theory")
            else:
                print("Warning: theories_chargemult information was not provided. Checking if charge/mult was provided to WrapTheory.run()")
                if charge is None or mult is None:
                    print("Error: Charge or multiplicity has not been specified for WrapTheory.run(). Please provide charge/mult for each theory using theories_chargemult keyword.")
                    ashexit()
                print(f"Warning: Using charge={charge} and mult={mult} for this theory")
                theory_charge, theory_mult = charge, mult

            # Run theory calculation
            eg_tuple = theory.run(current_coords=chosen_coords, current_MM_coords=current_MM_coords,
                                MMcharges=MMcharges, qm_elems=chosen_elems, elems=chosen_elems,
                                Grad=Grad, PC=PC, numcores=numcores, label=label,
                                charge=theory_charge, mult=theory_mult)

            if Grad:
                #print(f"Theory: {theory.theorynamelabel}  gradient shape", eg_tuple[1].shape)
                energy = eg_tuple[0]
                tempgrad = eg_tuple[1]
                energy, tempgrad = eg_tuple[0], eg_tuple[1]
                if PC and len(eg_tuple) >= 3 and eg_tuple[2] is not None:
                    component_pc_gradient = np.asarray(eg_tuple[2], dtype=np.float64)

                    if pc_gradient is None:
                        pc_gradient = np.zeros_like(component_pc_gradient)
                    elif pc_gradient.shape != component_pc_gradient.shape:
                        raise ValueError(
                            f"Incompatible PC-gradient shapes: "
                            f"{pc_gradient.shape} and {component_pc_gradient.shape}"
                        )

                    sign = 1.0
                    # Check to make sure theory_operators is a list of strings with '+' or '-' for each theory
                    if isinstance(self.theory_operators, list):
                        sign = 1.0 if self.theory_operators[i] == "+" else -1.0

                    pc_gradient += sign * component_pc_gradient
                # If region
                if atoms is not None:
                    nreg = len(atoms)
                    grad = np.zeros((full_dimension, 3))
                    grad[atoms] = tempgrad[:nreg]
                    if linkinfo is not None:
                        grad = self.linkatom_force_projection(linkinfo[1], linkinfo[0], atoms,
                                        tempgrad, grad, current_coords)
                else:
                    grad = tempgrad
                energies.append(energy)
                gradients.append(grad)
            else:
                energies.append(eg_tuple)
        print("\nAll WrapTheory calculations are done!\n")

        # Combine energy and gradient
        # Regular summation
        if self.theory_operators is None:
            self.energy = sum(energies)
            if self.printlevel > 1:
                for count,e in enumerate(energies):
                    print(f"Energy ({self.theories[count].theorynamelabel}):", e)
                print("Energy (Combined):", self.energy)
            if Grad:
                self.gradient = sum(gradients)
                if self.printlevel > 2:
                    for count,g in enumerate(gradients):
                        print(f"Gradient ({self.theories[count].theorynamelabel}):", g)
                    print("Gradient (Combined):", self.gradient)
        # AVERAGE/ENSEMBLE OPTION
        elif self.theory_operators == "average" or self.theory_operators == "ensemble":
            print("theory_operators option is set to 'average'.")
            self.energy = sum(energies) / len(energies)

            if self.printlevel > 1:
                for count,e in enumerate(energies):
                    print(f"Energy ({self.theories[count].theorynamelabel}):", e)
                print("Energy (Combined):", self.energy)
                # Print standard deviation of energies
                std_energy = np.std(energies)
                print("Energy (Standard Deviation):", std_energy)
            if Grad:
                self.gradient = sum(gradients) / len(gradients)
                if self.printlevel > 2:
                    for count,g in enumerate(gradients):
                        print(f"Gradient ({self.theories[count].theorynamelabel}):", g)
                    print("Gradient (Combined):", self.gradient)
        # User-defined operations
        else:
            print("theory_operators option is active.")
            print("theory_operators:", self.theory_operators)
            self.energy=0.0
            for e,op in zip(energies,self.theory_operators):
                if op == '+':
                    self.energy+=e
                elif op == '-':
                    self.energy-=e
                else:
                    print("Error: unknown operator:", op)
                    ashexit()
            if self.printlevel > 1:
                for count,e in enumerate(energies):
                    print(f"Energy ({self.theories[count].theorynamelabel}):", e)
                print("Energy (Combined):", self.energy)
            if Grad:
                self.gradient=np.zeros((full_dimension,3))
                for g,op in zip(gradients,self.theory_operators):
                    if op == '+':
                        self.gradient+=g
                    elif op == '-':
                        self.gradient-=g
                    else:
                        print("Error: unknown operator:", op)
                        ashexit()

                if self.printlevel > 2:
                    for count,g in enumerate(gradients):
                        print(f"Gradient ({self.theories[count].theorynamelabel}):", g)
                    print("Gradient (Combined):", self.gradient)

        if Grad and self.printlevel > 2:
            print("Writing out gradient")
            from ash.modules.module_coords import write_coords_all
            write_coords_all(self.gradient, qm_elems, indices=[i for i in range(len(qm_elems))], 
                                        file=f"Wrapgradient", 
                                        description=f"WrapTheory gradient {label} (au/Bohr):")


        print_time_rel(module_init_time, modulename='WrapTheory run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        if Grad:
            if PC and pc_gradient is not None:
                return self.energy, self.gradient, pc_gradient
            return self.energy, self.gradient
        else:
            return self.energy


#########################
# FractTheory class
#########################

class FractTheory:
    def __init__(self,theory=None, eta=None, chargeA=None, chargeB=None, 
                 multA=None, multB=None, DMinterpol=False, 
                 resultfile="fract_theory_gaps.txt"):

        self.eta=eta
        self.chargeA=chargeA
        self.chargeB=chargeB
        self.multA=multA
        self.multB=multB
        self.theory=theory

        self.theorynamelabel="Fract"
        self.theorytype="QM"
        self.analytic_hessian=False
        self.printlevel=2
        self.DMinterpol=DMinterpol # density matrix interpolation or not
        self.resultfile=resultfile

        # Check for compatibility of theory

        # pyscf
        if isinstance(self.theory,PySCFTheory):
            if self.DMinterpol:
                # Defining theory that has density matrix
                self.dmtheory=self.theory
        # QM/MM
        elif self.theory.theorytype == "QM/MM":
            if self.DMinterpol:
                print("Theory type is QM/MM and DMinterpol is True. Checking if QM theory is compatible")
                if not isinstance(self.theory.qm_theory, PySCFTheory):
                    print("Error: QM-theory inside QMMMTheory is not yet compatible with FractTheory.")
                    ashexit()
                self.dmtheory=self.qm_theory
        # Wraptheory
        elif self.theory.theorynamelabel == "WrapTheory":
            if self.DMinterpol:
                print("Theory is WrapTheory and DMinterpol is True. Checking if it contains a compatible QM-theory")
                # Checking instance of all theories via list comprehension
                ll = [isinstance(t, PySCFTheory) for t in self.theory.theories]
                # Check if any of the theories are PySCFTheory
                if not any(ll):
                    print("Error: No compatible QM-theory found inside WrapTheory.")
                ashexit()
                # Defining dm theory the first theory that is compatible
                self.dmtheory=self.theory.theories[ll.index(True)]

        # Results (gaps) writing to file, here header.
        with open(self.resultfile, 'w') as f:
            f.write(f"# eta deltaE (Eh)\n")


    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None, Hessian=False):

        print("Running FractTheory")

        # Init
        print("Now running inital state SCF")
        res_init = self.theory.run(current_coords=current_coords, elems=elems, 
                                   charge=self.chargeA, mult=self.multA, Grad=Grad)
        if Grad:
            E_init = res_init[0]
            G_init = res_init[1]
        else:
            E_init = res_init
        if self.DMinterpol:
            dm_init = self.dmtheory.get_density_matrix()

        # Ion
        print("Now running final-state SCF")
        res_final = self.theory.run(current_coords=current_coords, elems=elems, charge=self.chargeB, mult=self.multB, Grad=Grad)
        if Grad:
            E_fin = res_final[0]
            G_fin = res_final[1]
        else:
            E_fin = res_final
        if self.DMinterpol:
            dm_Nm1 = self.dmtheory.get_density_matrix()

        # Printing E(init) and E(final) and gap
        print("E(init)", E_init)
        print("E(final)", E_fin)
        deltaE = E_fin - E_init
        print("deltaE:", deltaE)

        # Interpolated Energy
        E_eta = (1 - self.eta) * E_init + self.eta * E_fin
        print("E_eta:", E_eta)

        self.energy=E_eta

        # Interpolated Gradient
        if Grad:
            G_eta = (1 - self.eta) * G_init + self.eta * G_fin
            print("G_eta:", G_eta)
            self.gradient=G_eta

        # Write gap to file
        with open(self.resultfile, 'a') as f:
            f.write(f"{self.eta} {deltaE}\n")

        # Derive Fractional energy via interpolated DM
        if self.DMinterpol is True:
            # Derive interpolated density matrix
            print("Deriving fractional density matrix using eta:", self.eta)
            # Fractional electron ensemble density
            dm_fne = (1 - self.eta) * dm_init + self.eta * dm_Nm1
            self.dm=dm_fne
            print("Deriving new energy from energy of interpolated DM")
            if isinstance(self.dmtheory,PySCFTheory):
                interpol_DM_E = float(self.dmtheory.mf.energy_tot(dm_fne))
                print("Interpol-DM energy:", interpol_DM_E)
            elif isinstance(self.dmtheory,ORCATheory):
                ashexit()
                # Derive new energy from FNE DM
                # Todo: need Fock matrices, density matrices and 1-el integrals
            else:
                print("Error: Incompatible input theory for FractTheory. Currently supporting: PySCFTheory")
                ashexit()
            self.energy=interpol_DM_E

        print("FractTheory energy:", self.energy)

        if Grad is True:
            return self.energy,self.gradient
        else:
            return self.energy