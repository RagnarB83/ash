"""
HybridTheory module:


"""
import time
import numpy as np
import os
from collections import defaultdict

from ash.functions.functions_general import BC, ashexit, print_time_rel,print_line_with_mainheader,listdiff
from ash.modules.module_theory import Theory
from ash.interfaces.interface_ORCA import grabatomcharges_ORCA
from ash.interfaces.interface_xtb import grabatomcharges_xTB,grabatomcharges_xTB_output
import ash.constants
import ash.functions.functions_elstructure

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
    def run(self, current_coords=None, elems=None, Grad=False, numcores=None, label='default', charge=None, mult=None, run_both_theories=False ):
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
# Similar in a way to DualTheory but we simply want to combine a regular Theory with a basic correction
# Intended to be used for simple corrections like DFTD4.

class WrapTheory:
    """ASH WrapTheory theory.
    Combines 2 theories to give a modified energy and modified gradient
    """
    def __init__(self, theory1=None, theory2=None, printlevel=1, label=None):

        self.theorytype="QM"
        self.theory1=theory1
        self.theory2=theory2
        self.printlevel=printlevel
        self.label=label
        self.filename=""
        self.theorynamelabel="WrapTheory"

        print_line_with_mainheader(f"{self.theorynamelabel} initialization")

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):

        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        print("Running Theory 1:", self.theory1.theorynamelabel)
        # Calculate Theory 1
        if Grad:
            e_theory1, g_theory1 = self.theory1.run(current_coords=current_coords,
                                                    current_MM_coords=current_MM_coords,
                                                    MMcharges=MMcharges, qm_elems=qm_elems,
                                                    elems=elems, Grad=Grad, PC=PC, numcores=numcores,
                                                    label=label, charge=charge, mult=mult)
        else:
            e_theory1 = self.theory1.run(current_coords=current_coords,
                                                    current_MM_coords=current_MM_coords,
                                                    MMcharges=MMcharges, qm_elems=qm_elems,
                                                    elems=elems, Grad=Grad, PC=PC, numcores=numcores,
                                                    label=label, charge=charge, mult=mult)
        # Calculate Theory 2
        print("Running Theory 2:", self.theory2.theorynamelabel)
        if Grad:
            e_theory2, g_theory2 = self.theory2.run(current_coords=current_coords,
                                                    current_MM_coords=current_MM_coords,
                                                    MMcharges=MMcharges, qm_elems=qm_elems,
                                                    elems=elems, Grad=Grad, PC=PC, numcores=numcores,
                                                    label=label, charge=charge, mult=mult)
        else:
            e_theory2 = self.theory2.run(current_coords=current_coords,
                                                    current_MM_coords=current_MM_coords,
                                                    MMcharges=MMcharges, qm_elems=qm_elems,
                                                    elems=elems, Grad=Grad, PC=PC, numcores=numcores,
                                                    label=label, charge=charge, mult=mult)

        # Combine energy and gradient
        energy = e_theory1 + e_theory2
        if self.printlevel == 3:
            print("Energy (Theory1):", e_theory1)
            print("Energy (Theory2):", e_theory2)
            print("Energy (Combined):", energy)
        if Grad:
            gradient = g_theory1 + g_theory2
            if self.printlevel == 3:
                print("Gradient (Theory1):", g_theory1)
                print("Gradient (Theory2):", g_theory2)
                print("Gradient (Combined):", gradient)

        if Grad:
            return energy, gradient
        else:
            return energy



class ONIOMTheory(Theory):
    def __init__(self, theory1=None, theory2=None, theories_N=None, regions_N=None, regions_chargemult=None,
                 embedding=None, full_pointcharges=None, chargemodel="CM5",
                 fullregion_charge=None, fullregion_mult=None, fragment=None, label=None,
                 printlevel=2, numcores=1,):
        super().__init__()
        self.theorytype="ONIOM"
        self.theory1=theory1
        self.theory2=theory2
        self.printlevel=printlevel
        self.label=label
        self.filename=""
        self.theorynamelabel="ONIOMTheory"
        print_line_with_mainheader("ONIOM Theory")
        print("A N-layer ONIOM module")

        # Early exits
        # If fragment object has not been defined
        if fragment is None:
            print("Error: fragment= keyword has not been defined for QM/MM. Exiting")
            ashexit()
        if fullregion_charge is None or fullregion_mult is None:
            print("Error: Full-region charge and multiplicity must be provided (fullregion_charge, fullregion_mult keywords)")
            ashexit()

        if type(theories_N) != list:
            print("Error: theories_N should be a list")
            ashexit()
        print(f"{len(theories_N)} theories provided. This is a {len(theories_N)}-layer ONIOM.")
        if regions_N is None:
            print("Error: regions_N must be provided for N-layer ONIOM")
            ashexit()
        if regions_chargemult is None:
            print("Error: regions_chargemult must be provided for N-layer ONIOM (list of lists of charge,mult for each region)")
            ashexit()
        if len(theories_N) != len(regions_N):
            print("Error: Number of theories and regions must match")
            ashexit()
        if len(theories_N) != len(regions_chargemult):
            print("Error: Number of theories and regions_chargemult must match")
            ashexit()
        # Full system
        self.allatoms = fragment.allatoms
        self.theories_N=theories_N
        self.regions_N=regions_N
        self.regions_chargemult=regions_chargemult # List of list of charge,mult combos

        # Embedding
        #Note: by default no embedding, meaning LL theory for everything
        self.embedding=embedding
        self.chargemodel=chargemodel
        # Defining pointcharges for full system
        self.full_pointcharges=full_pointcharges

        # N-layer ONIOM
        self.fullregion_charge=fullregion_charge
        self.fullregion_mult=fullregion_mult

        # Defining charge/mult here as well (ASH jobtypes stop otherwise)
        self.charge = self.fullregion_charge
        self.mult = self.fullregion_mult



        #
        print("Theories:")
        for i,t in enumerate(self.theories_N):
            print(f"Theory {i+1}:", t.theorynamelabel)
        print("\nRegions provided:")
        #
        for i,r in enumerate(self.regions_N):
            print(f"Region {i+1} ({len(r)} atoms):", r)
        print("Allatoms:", self.allatoms)
        print("\nRegion-chargemult info provided:")
        #
        for i,r in enumerate(self.regions_chargemult):
            print(f"Region {i+1} Charge:{r[0]} Mult:{r[1]}")

        if len(self.theories_N) > 3:
            print("Error: N>3 layer ONIOM is not yet supported")
            ashexit()

    def run(self, current_coords=None, Grad=False, elems=None, charge=None, mult=None, label=None, numcores=None):

        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Charge/mult. Note: ignoring charge/mult from run keyword.
        # Charge/mult definitions for full system and regions must have been provided on object creation

        if numcores is None:
            numcores = self.numcores

        # Full coordinates
        full_coords=current_coords
        full_elems=elems


        # Dicts to keep energy and gradient for each theory-region combo
        E_dict={} # (theory,region) -> energy
        G_dict={} # (theory,region) -> gradient
        num_theories = len(self.theories_N)

        # First doing LowLevel (LL) theory on Full region
        ll_theory = self.theories_N[-1]
        print(f"Running Theory LL ({ll_theory.theorynamelabel}) on Full-region ({len(full_elems)} atoms)")

        # Derive pointcharges unless full_pointcharges were already provided
        if self.full_pointcharges is None and self.embedding == "Elstat":
            print("Full-system pointcharges were not made available initially")
            print("This means that we must derive pointcharges for full system")
            # TODO: How to do this in general
            # Check if the low-level theory is compatible with some charge model
            # Should probably this in init instad though
            if isinstance(ll_theory, ash.ORCATheory):
                print(f"Theory is ORCATheory. Using {self.chargemodel} charge model")
                theory.extrakeyword+="\n! hirshfeld "
            elif isinstance(ll_theory, ash.xTBTheory):
                 print(f"Theory is xTBTheory. Using default xtb charge model")
            else:
                print("Problem: Theory-level not compatible with pointcharge-creation")
                ashexit()


        if Grad:
            e_LL_full, g_LL_full = ll_theory.run(current_coords=full_coords,
                                                    elems=full_elems, Grad=Grad, numcores=numcores,
                                                    label=label, charge=self.fullregion_charge, mult=self.fullregion_mult)
        else:
            e_LL_full = ll_theory.run(current_coords=full_coords,
                                    elems=full_elems, Grad=Grad, numcores=numcores,
                                    label=label, charge=self.fullregion_charge, mult=self.fullregion_mult)

        # Grabbing atom charges from ORCA output
        # TODO: Remove theory-specific code
        if self.full_pointcharges is None and self.embedding == "Elstat":
            print("Grabbing atom charges for whole system")
            if isinstance(ll_theory, ash.ORCATheory):
                self.full_pointcharges = grabatomcharges_ORCA(self.chargemodel,"orca.out")
            elif isinstance(ll_theory, ash.xTBTheory):
                print(f"{ll_theory.filename}.out")
                self.full_pointcharges = grabatomcharges_xTB_output(ll_theory.filename+'.out', chargemodel=self.chargemodel)
            print("self.full_pointcharges:", self.full_pointcharges)

        E_dict[(num_theories-1,-1)] = e_LL_full
        if Grad:
            G_dict[(num_theories-1,-1)] = g_LL_full

        # Other theory-region combos
        for j,region in enumerate(self.regions_N):
            # Skipping last region
            if j == len(self.regions_N)-1:
                print("Last region always skipped")
                continue
            for i,theory in enumerate(self.theories_N):
                # Skipping HL on everything but first region
                if i == 0 and j>0:
                    print("HL theory on non-first region. Skipping")
                    continue
                # 3-layer ONIOM case (LL theory on first region). Skipping
                if i == 2 and j == 0:
                    print("Case 3-layer ONIOM: LL theory on first region. Skipping")
                    continue

                # Taking coordinates for region
                r_coords = np.take(current_coords,region,axis=0)
                r_elems = [elems[x] for x in region]

                # Activate embedding for HL region 1
                #TODO: also embedding for LL theory region1
                if i == 0 and self.embedding == "Elstat":
                    print("Embedding activated for HL theory")
                    PC=True
                    # Which region?
                    PCregion=listdiff(self.allatoms,region)
                    print("PCregion:", PCregion)
                    pointchargecoords = np.take(current_coords,PCregion,axis=0)
                    # Pointcharges
                    if self.full_pointcharges is None:
                        print("Warning: Pointcharges for full system not available")
                        ashexit()
                    pointcharges=[self.full_pointcharges[x] for x in PCregion]
                else:
                    PC=False
                    pointchargecoords=None
                    pointcharges=None

                # Running
                print(f"Running Theory {i+1} ({theory.theorynamelabel}) on Region {j+1} ({len(r_elems)} atoms)")
                res = theory.run(current_coords=r_coords, elems=r_elems, Grad=Grad, numcores=numcores,
                                                PC=PC, current_MM_coords=pointchargecoords, MMcharges=pointcharges,
                                                label=label, charge=self.regions_chargemult[j][0], mult=self.regions_chargemult[j][1])
                if PC and Grad:
                    e,g,pg = res
                elif not PC and Grad:
                    e,g = res
                elif not PC and not Grad:
                    e = res

                E_dict[(i,j)] = e
                if Grad:
                    G_dict[(i,j)] = g

        # 2-layer ONIOM Energy and Gradient expression
        if len(self.theories_N) == 2:
            # E_dict: {(1, -1): -14.73707412056, (0, 0): -115.486915570077, (1, 0): -8.9611505694}
            self.energy = E_dict[(1,-1)] + E_dict[(0,0)] - E_dict[(1,0)]
            if Grad:
                # Gradient assembled
                self.gradient = G_dict[(1,-1)]
                for at, g in zip(self.regions_N[0], G_dict[(0,0)]):
                    self.gradient[at] += g
                for at, g in zip(self.regions_N[0], G_dict[(1,0)]):
                    self.gradient[at] -= g
                #Pointcharge gradient contribution
                #TODO
                if self.embedding == "Elstat":
                    print("TODO pcgrad")
                    ashexit()




        # 3-layer ONIOM Energy and Gradient expression
        elif len(self.theories_N) == 3:
            self.energy = E_dict[(2,-1)] + E_dict[(0,0)] - E_dict[(1,0)] + E_dict[(1,1)] - E_dict[(2,1)]
            # E(High,Real) = E(Low,Real) + [E(High,Model) - E(Medium,Model)]
            #                +  [E(Medium,Inter) - E(Low,Inter)].
            if Grad:
                # Gradient assembled
                self.gradient = G_dict[(2,-1)]
                for at, g in zip(self.regions_N[0], G_dict[(0,0)]):
                    self.gradient[at] += g
                for at, g in zip(self.regions_N[0], G_dict[(1,0)]):
                    self.gradient[at] -= g
                for at, g in zip(self.regions_N[1], G_dict[(1,1)]):
                    self.gradient[at] += g
                for at, g in zip(self.regions_N[1], G_dict[(2,1)]):
                    self.gradient[at] -= g
        # 4-layer ONIOM
        elif len(self.theories_N) == 4:
            print("4-layer ONIOM not ready")
            ashexit()

        print("ONIOM energy:", self.energy)

        if Grad:
            return self.energy,self.gradient
        else:
            return self.energy
