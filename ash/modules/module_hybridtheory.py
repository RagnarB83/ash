"""
HybridTheory module:


"""
import time
import numpy as np
import os
import copy
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
    Combines 2 theories to give a modified energy and modified gradient
    """
    def __init__(self, theory1=None, theory2=None, theories=None, printlevel=2, label=None,
                 theory1_atoms=None, theory2_atoms=None, theory3_atoms=None, theory4_atoms=None,
                 theory5_atoms=None,
                 theory_operators=None):
        super().__init__()

        self.theorytype="QM"
        self.theory1=theory1
        self.theory2=theory2
        self.printlevel=printlevel
        self.label=label
        self.filename=""
        self.theorynamelabel="WrapTheory"
        self.theories=theories
        # Option to have theory only calculate certain atoms
        self.theory1_atoms=theory1_atoms
        self.theory2_atoms=theory2_atoms
        self.theory3_atoms=theory3_atoms
        self.theory4_atoms=theory4_atoms
        self.theory5_atoms=theory5_atoms


        print_line_with_mainheader(f"{self.theorynamelabel} initialization")
        print("Creating WrapTheory object")

        if theories is not None :
            print("Theories defined by keyword")
            if theory1 is not None or theory2 is not None:
                print("Error: Both theory1/theory2 keywords and theories keyword cannot be defined")
                print("Choose either to define: theory1 and theory2       or provide a list of all theories using the theories keyword")
                exit()
        else:
            if theory1 is None or theory2 is None:
                print("Error: Either theories keyword or theory1 and theory2 have to be provided to WrapTheory")
                ashexit()
            self.theories=[theory1,theory2]

        # Operators: '+' or '-' for each theory
        # By default we sum
        self.theory_operators=theory_operators
        if self.theory_operators is not None:
            print("self.theory_operators option active!")
            if len(self.theory_operators) != len(self.theories):
                print(f"Error: Number of theory-operators {len(self.theory_operators)} is not equal to number of theories {len(self.theories)}")
                ashexit()


    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):

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
        chosen_coords=current_coords
        chosen_elems=qm_elems
        for i,theory in enumerate(self.theories):
            print(f"Now running Theory: {theory.theorynamelabel}")

            # If theory_atoms have been set then we only pass part of coordinates over
            if i+1 == 1 and self.theory1_atoms is not None:
                print("theory1_atoms has been set:", self.theory1_atoms)
                chosen_coords = np.take(current_coords, self.theory1_atoms, axis=0)
                chosen_elems = [qm_elems[i] for i in self.theory1_atoms]
            elif i+1 == 2 and self.theory2_atoms is not None:
                print("theory2_atoms has been set:", self.theory2_atoms)
                chosen_coords = np.take(current_coords, self.theory2_atoms, axis=0)
                chosen_elems = [qm_elems[i] for i in self.theory2_atoms]
            elif i+1 == 3 and self.theory3_atoms is not None:
                print("theory3_atoms has been set:", self.theory3_atoms)
                chosen_coords = np.take(current_coords, self.theory3_atoms, axis=0)
                chosen_elems = [qm_elems[i] for i in self.theory3_atoms]
            elif i+1 == 4 and self.theory4_atoms is not None:
                print("theory4_atoms has been set:", self.theory4_atoms)
                chosen_coords = np.take(current_coords, self.theory4_atoms, axis=0)
                chosen_elems = [qm_elems[i] for i in self.theory4_atoms]
            elif i+1 == 5 and self.theory5_atoms is not None:
                print("theory5_atoms has been set:", self.theory5_atoms)
                chosen_coords = np.take(current_coords, self.theory5_atoms, axis=0)
                chosen_elems = [qm_elems[i] for i in self.theory5_atoms]
            eg_tuple = theory.run(current_coords=chosen_coords,
                                                current_MM_coords=current_MM_coords,
                                                MMcharges=MMcharges, qm_elems=chosen_elems,
                                                elems=elems, Grad=Grad, PC=PC, numcores=numcores,
                                                label=label, charge=charge, mult=mult)
            if Grad:
                #print(f"Theory: {theory.theorynamelabel}  gradient shape", eg_tuple[1].shape)
                energy = eg_tuple[0]
                tempgrad = eg_tuple[1]
                # Assemble gradient of correct dimension
                if i+1 == 1 and self.theory1_atoms is not None:
                    fullgrad=np.zeros((full_dimension,3))
                    fullgrad[self.theory1_atoms] = tempgrad
                    grad=fullgrad
                elif i+1 == 2 and self.theory2_atoms is not None:
                    fullgrad=np.zeros((full_dimension,3))
                    fullgrad[self.theory2_atoms] = tempgrad
                    grad=fullgrad
                elif i+1 == 3 and self.theory3_atoms is not None:
                    fullgrad=np.zeros((full_dimension,3))
                    fullgrad[self.theory3_atoms] = tempgrad
                    grad=fullgrad
                else:
                    grad=tempgrad
                energies.append(energy)
                gradients.append(grad)
            else:
                energy = eg_tuple
                energies.append(energy)  
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

        if Grad:
            return self.energy, self.gradient
        else:
            return self.energy
