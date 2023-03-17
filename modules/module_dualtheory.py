"""
DualTheory module:

class DualTheory

"""
from ash.functions.functions_general import BC, ashexit, print_time_rel
import ash.constants
import ash.functions.functions_elstructure
import time
import numpy as np
from collections import defaultdict

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
    #Cleanup after run.
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

    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, elems=None, Grad=False, numcores=None, label=None, charge=None, mult=None, run_both_theories=False ):
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING DUALTHEORY INTERFACE-------------", BC.END)
        self.totalruncalls += 1

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
                if label not in self.update_freq_dict:
                    print("First DualTheory call. Running both theories")
                    self.update_freq_dict[label] = [0,0,0] #Runcalls, theory1-runcalls, theory2-runcalls
                    run_both_theories=True
                elif label in self.update_freq_dict:
                    print("")

                    #Checking if we have reached max_updates
                    print(f"Number of corrections {self.update_freq_dict[label][2]} for label: {label}")
                    if  self.update_freq_dict[label][2] >= self.max_updates:
                        print("Max number of corrections reached. Skipping correction (theory2 calc) in this step")
                    else:
                        #If runcalls for label mataches update_freq
                        if self.update_freq_dict[label][0] % self.update_freq == 0:
                            run_both_theories=True
            else:
                print("Unknown update_schedule")
                ashexit()
            ################
            self.update_freq_dict[label][0] +=1 
            if Grad == True:
                self.update_freq_dict[label][1] +=1 
                T1_energy, T1_grad = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                if run_both_theories == True:
                    self.update_freq_dict[label][2] +=1 
                    T2_energy, T2_grad = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                    #Update gradient correction if both theories were calculated
                    self.correction(Grad,label,T2_E=T2_energy,T1_E=T1_energy,T2_G=T2_grad,T1_G=T1_grad)
                    #self.correction_stats(label)
            else:
                self.update_freq_dict[label][1] +=1 
                T1_energy = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                if run_both_theories == True:
                    self.update_freq_dict[label][2] +=1 
                    T2_energy = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                    #Update gradient correction if both theories were calculated
                    self.correction(Grad,label,T2_E=T2_energy,T1_E=T1_energy,T2_G=T2_grad,T1_G=T1_grad)
                    #self.correction_stats(label)

        #Current energy
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
    

