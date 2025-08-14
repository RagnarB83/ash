from ash.modules.module_coords import print_coords_for_atoms, print_coords_all
from ash.functions.functions_general import ashexit, BC
import numpy as np
import ash
import os
# Basic Theory class

class Theory:
    def __init__(self):
        self.theorytype = None
        self.theorynamelabel = None
        self.label = None
        self.analytic_hessian = False
        self.numcores = 1
        self.filename = None
        self.printlevel = None
        self.properties = {}
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleanup method called but not yet implemented for this theory")
    def run(self):
        print("Run was called but not yet implemented for this theory")

class QMTheory(Theory):
    def __init__(self):
        super().__init__()
        self.theorytype = "QM"
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleanup method called but not yet implemented for this theory")
    def run(self):
        print("Run was called but not yet implemented for this theory")


# Numerical gradient class
class NumGradclass:
    def __init__(self, theory, npoint=2, displacement=0.00264589,  runmode="serial", numcores=1, printlevel=2):
        print("Creating NumGrad wrapper object")
        self.theory=theory
        self.theorytype="QM"
        self.theorynamelabel="NumGrad"
        self.displacement=displacement
        self.npoint=npoint
        self.runmode=runmode
        self.printlevel=printlevel
        self.numcores=numcores

    def set_numcores(self,numcores):
        self.numcores=numcores

    def cleanup(self):
        print("Cleanup method called but not yet implemented for Numgrad")

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):

        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} WRAPPER -------------", BC.END)

        numatoms= len(current_coords)
        displacement_bohr = self.displacement*1.88972612546

        list_of_displaced_geos, list_of_displacements, all_disp_fragments = creating_displaced_geos(current_coords, elems, self.displacement,self.npoint, charge, mult, printlevel=2)
        if self.runmode == "serial":
            print("Numgrad: runmode is serial")
            print("Running original geometry first")
            # Energy for original geometry.
            orig_energy = self.theory.run(current_coords=current_coords, elems=elems, Grad=False, label=label,
                                charge=charge, mult=mult)
            #
            dispdict={}
            print(f"Will now loop over {len(list_of_displacements)} displacements")

            # Looping over displacements
            for i,dispgeo in enumerate(list_of_displaced_geos):
                disp = list_of_displacements[i]
                print(f"Running displacement {i+1} / {len(list_of_displaced_geos)}. Displacing Atom:{disp[0]} Coord:{disp[1]} Direction:{disp[2]}")
                energy = self.theory.run(current_coords=dispgeo, elems=elems, Grad=False, label=label, charge=charge, mult=mult)
                dispdict[(disp)] = energy

        elif self.runmode == "parallel":
            print("Numgrad: runmode is parallel")
            origfrag=ash.Fragment(coords=current_coords, elems=elems,label="orig", printlevel=0, charge=charge, mult=mult)
            all_disp_fragments = [origfrag] + all_disp_fragments
            result = ash.functions.functions_parallel.Job_parallel(fragments=all_disp_fragments, theories=[self.theory], numcores=self.numcores,
                allow_theory_parallelization=True, Grad=False, printlevel=self.printlevel, copytheory=True)
            print("result:", result)
            dispdict = result.energies_dict
            orig_energy = dispdict["orig"]

        # Assemble gradient
        gradient = np.zeros((numatoms,3))
        # 2-point
        if self.npoint == 2:
            for atindex in range(0,numatoms):
                # Looping over x,yz
                for u in [0,1,2]:
                    # Pos and neg directions
                    if self.runmode == "parallel":
                        #'0_0_+'
                        posval = dispdict[f"{atindex}_{u}_+"]
                        negval = dispdict[f"{atindex}_{u}_-"]
                    else:
                        posval = dispdict[(atindex,u,'+')]
                        negval = dispdict[(atindex,u,'-')]
                    grad_component=(posval - negval)/(2*displacement_bohr)
                    gradient[atindex,u] = grad_component
        # 1-point 
        elif self.npoint == 1:
            for atindex in range(0,numatoms):
                # Looping over x,yz
                for u in [0,1,2]:
                    # Pos direction only
                    if self.runmode == "parallel":
                        #'0_0_+'
                        posval = dispdict[f"{atindex}_{u}_+"]
                    else:
                        posval = dispdict[(atindex,u,'+')]
                    grad_component=(posval - orig_energy)/displacement_bohr
                    gradient[atindex,u] = grad_component

        self.energy = orig_energy
        self.gradient = gradient 

        return self.energy, self.gradient


# MEPC-gradient class
class MECPGradclass:
    def __init__(self, theory_1=None,theory_2=None, charge_1=None, charge_2=None, 
                 mult_1=None, mult_2=None, runmode="serial", numcores=1, printlevel=2):
        print("Creating MECPGrad wrapper object")

        if charge_1 is None or charge_2 is None:
            print("Error: Please set both charge_1 and charge_2!")
            ashexit()
        if mult_1 is None or mult_2 is None:
            print("Error: Please set both mult_1 and mult_2!")
            ashexit()
        self.theory_1=theory_1
        self.theory_2=theory_2
        self.charge_1=charge_1
        self.charge_2=charge_2
        self.mult_1=mult_1
        self.mult_2=mult_2
        self.theorytype="QM"
        self.theorynamelabel="MECPGrad"
        self.runmode=runmode
        self.printlevel=printlevel
        self.numcores=numcores


    def set_numcores(self,numcores):
        self.numcores=numcores

    def cleanup(self):
        print("Cleanup method called but not yet implemented for MECPGrad")

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):

        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} WRAPPER -------------", BC.END)

        numatoms = len(current_coords)
        # Theory 1
        #Creating dir
        try:
            os.mkdir("theory1")
        except FileExistsError:
            pass
        os.chdir("theory1")
        print(f"Running Theory 1 with charge {self.charge_1} and mult {self.mult_1}")
        energy_1,gradient_1 = self.theory_1.run(current_coords=current_coords, elems=elems, Grad=True, 
                                   label="theory1", charge=self.charge_1, mult=self.mult_1)
        os.chdir("..")

        # Theory 2
        # Creating dir
        try:
            os.mkdir("theory2")
        except FileExistsError:
            pass
        os.chdir("theory2")
        print(f"Running Theory 2 with charge {self.charge_2} and mult {self.mult_2}")
        energy_2,gradient_2 = self.theory_2.run(current_coords=current_coords, elems=elems, Grad=True, 
                                   label="theory2", charge=self.charge_2, mult=self.mult_2)
        os.chdir("..")
        x1 = (gradient_1-gradient_2)
        x1norm= x1/np.linalg.norm(x1)
        f = 2*(energy_1-energy_2)*x1norm
        g = gradient_1 - np.dot(x1norm.flatten(),gradient_1.flatten())*x1norm
        # Surface crossing gradient
        g_sc = g + f
        self.gradient=g_sc
        self.energy=energy_1-energy_2
        print("E1:", energy_1)
        print("E2:", energy_2)
        return self.energy, self.gradient


def creating_displaced_geos(current_coords,elems,displacement,npoint, charge, mult, printlevel=2):
    displacement_bohr=displacement*1.88972612546
    print("Displacement: {:5.4f} Å ({:5.4f} Bohr)".format(displacement,displacement_bohr))
    print("Starting geometry:")
    print()
    print("Printing original geometry...")
    print_coords_all(current_coords,elems)
    print()

    # Looping over each atom and each coordinate to create displaced geometries
    # Only displacing atom if in hessatoms list. i.e. possible partial Hessian
    list_of_displaced_geos=[]
    list_of_displacements=[]
    for atom_index in range(0,len(current_coords)):
        for coord_index in range(0,3):
            val=current_coords[atom_index,coord_index]
            #Displacing in + direction
            current_coords[atom_index,coord_index]=val+displacement
            y = current_coords.copy()
            list_of_displaced_geos.append(y)
            list_of_displacements.append((atom_index, coord_index, '+'))
            if npoint == 2:
                #Displacing  - direction
                current_coords[atom_index,coord_index]=val-displacement
                y = current_coords.copy()
                list_of_displaced_geos.append(y)
                list_of_displacements.append((atom_index, coord_index, '-'))
            #Displacing back
            current_coords[atom_index, coord_index] = val

    # Original geo added here if onepoint
    if npoint == 1:
        list_of_displaced_geos.append(current_coords)
        list_of_displacements.append('Originalgeo')

    if printlevel > 2:
        print("List of displacements:", list_of_displacements)

    # Creating ASH fragments
    # Creating displacement labels as strings and adding to fragment
    # Also calclabels, currently used by runmode serial only
    list_of_labels=[]
    all_disp_fragments=[]
    for dispgeo,disp in zip(list_of_displaced_geos,list_of_displacements):
        #Original geo
        if disp == 'Originalgeo':
            calclabel = 'Originalgeo'
            stringlabel=f"Originalgeo"
        #Displacements
        else:
            atom_disp = disp[0]
            if disp[1] == 0:
                crd = 'x'
            elif disp[1] == 1:
                crd = 'y'
            elif disp[1] == 2:
                crd = 'z'
            drection = disp[2]
            calclabel="Atom: {} Coord: {} Direction: {}".format(str(atom_disp),str(crd),str(drection))
            stringlabel=f"{disp[0]}_{disp[1]}_{disp[2]}"
        #Create fragment
        frag=ash.Fragment(coords=dispgeo, elems=elems,label=stringlabel, printlevel=0, charge=charge, mult=mult)
        all_disp_fragments.append(frag)
        #list_of_labels.append(calclabel)

    return list_of_displaced_geos, list_of_displacements,all_disp_fragments





# MicroIterativeclass class
class MicroIterativeclass:
    def __init__(self, theory=None, numcores=1, printlevel=2):

        self.theory=theory
        self.theorytype="QM"
        self.iterationcount=0

        self.engine=None

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
                elems=None, Grad=False, Hessian=False, PC=False, numcores=None, restart=False, label=None,
                charge=None, mult=None, ):

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # RUN
        self.microiters_active=False
        print("Iteration:", self.iterationcount)
        if self.iterationcount == 0:
            #Do regular calculation
            eg = self.theory.run(current_coords=current_coords, qm_elems=qm_elems, Grad=Grad, 
                                                PC=False, numcores=numcores, charge=charge, mult=mult)
            self.microiters_active=True
        
        # Micro-iterations
        else:
            if self.microiters_active:
                print("Micro-iterations active ")
                print("Doing special optimization of outer-region only")
                # Do special outer region optimization
                # Calling special optimizer here
                
                # Do special inner region optimization?
                # Calling special optimizer here
                
                #Check for convergence ?
                self.microiters_active=False
            else:
                print("Micro-iterations are finished. Doing regular run")
                eg = self.theory.run(current_coords=current_coords, qm_elems=qm_elems, Grad=Grad, 
                                                    PC=False, numcores=numcores, charge=charge, mult=mult)


            #print("self.engine", self.engine)
            #self.engine.actatoms=self.theory.qmatoms
            #self.engine.ActiveRegion=True
            #Either somehow modify actatoms of ASHengine in geometric

            # Or do a special optimization here

        #Grab
        if Grad:
            self.energy, self.gradient = eg
        else:
            self.energy = eg


        self.iterationcount+=1

        if Grad:
            return self.energy, self.gradient
        else:
            return self.energy,
