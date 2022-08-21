"""
DualTheory module:

class DualTheory

"""
from ash.functions.functions_general import BC, ashexit, print_time_rel
import ash.interfaces.interface_ORCA
import ash.constants
import ash.functions.functions_elstructure
import time

class DualTheory:
    """ASH DualTheory theory.
    Combines two theory levels to give a modified energy and modified gradient
    """
    def __init__(self, theory1=None, theory2=None, printlevel=2, label=None, correctiontype="Difference", update_freq=5, numcores=1):
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
        #Gradient and energy correction ad sicts
        self.gradient_correction={}
        self.energy_correction={}
        #Keep track of total number of run calls
        self.totalruncalls = 0
        #At which iteration do we update correction
        self.update_freq=update_freq
        #Dictionary that keeps track of how often each calculation label has been run
        self.update_freq_dict={}

        #Booleans to switch modes.
        self.theory1_active=False #Unused for now
        self.theory2_active=False #Used for switching to performing only theory 2
    
    def correction(self,Grad,label,T2_E=None,T1_E=None,T2_G=None,T1_G=None):

        if self.correctiontype=="Difference":
            self.energy_correction[label] = T2_E - T1_E
            if Grad == True:
                self.gradient_correction[label] = T2_G - T1_G
        else:
            print("Correctiontype not available")
            ashexit()
    def switch_to_theory(self,num):
        print("Dualtheory object: Switching to theory:", num)
        if num == 2:
            self.theory2_active = True
        elif num == 1:
            self.theory1_active = True
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, elems=None, Grad=False, numcores=None, label=None, charge=None, mult=None, run_both_theories=False ):
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING DUALTHEORY INTERFACE-------------", BC.END)
        self.totalruncalls += 1

        #If theory2 switch is active then we only do theory2
        if self.theory2_active == True:
            print("Theory 2 active")
            if Grad == True:
                T2_energy, T2_grad = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                return T2_energy,T2_grad
            else:
                T2_energy = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                return T2_energy
        #If theory1 switch is active then we only do theory1
        elif self.theory1_active == True:
            print("Theory 1 only active")
            if Grad == True:
                T1_energy, T1_grad = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                return T1_energy,T1_grad
            else:
                T1_energy = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                return T1_energy
        #Else we do theory1 with regular theory2 correction
        else:
            print("Theory 1+ Theory2-correction active")
            if label not in self.update_freq_dict:
                self.update_freq_dict[label] = [1,1,1] #Runcalls, theory1-runcalls, theory2-runcalls
                run_both_theories=True
            elif label in self.update_freq_dict:
                self.update_freq_dict[label][0] +=1
                self.update_freq_dict[label][1] +=1 
                self.update_freq_dict[label]

                #If runcalls for label mataches update_freq
                if self.update_freq_dict[label][0] % self.update_freq == 0:
                    run_both_theories=True
                    self.update_freq_dict[label][2] +=1 

            if Grad == True:
                T1_energy, T1_grad = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                if run_both_theories == True:
                    T2_energy, T2_grad = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=True, charge=charge, mult=mult)
                    #Update gradient correction if both theories were calculated
                    self.correction(Grad,label,T2_E=T2_energy,T1_E=T1_energy,T2_G=T2_grad,T1_G=T1_grad)

            else:
                T1_energy = self.theory1.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                if run_both_theories == True:
                    T2_energy = self.theory2.run(current_coords=current_coords, elems=elems, numcores=numcores, Grad=False, charge=charge, mult=mult)
                    #Update gradient correction if both theories were calculated
                    self.correction(Grad,label,T2_E=T2_energy,T1_E=T1_energy,T2_G=T2_grad,T1_G=T1_grad)

        #Current energy
        energy = T1_energy + self.energy_correction[label]
        print("Dualtheory energy:", energy)
        if Grad == True:
            #Combine into current gradient
            gradient = T1_grad + self.gradient_correction[label]
            return energy,gradient
        else:
            return energy
    

