from ash.modules.module_coords import print_coords_for_atoms, print_coords_all
from ash.functions.functions_general import ashexit, BC
import numpy as np
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
    def __init__(self, theory, npoint=2, displacement=0.00264589,  runmode="serial", printlevel=2):
        print("Creating NumGrad wrapper object")
        self.theory=theory
        self.theorytype="QM"
        self.theorynamelabel="NumGrad"
        self.displacement=displacement
        self.npoint=npoint
        self.runmode=runmode
        self.printlevel=printlevel

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

        list_of_displaced_geos, list_of_displacements = creating_displaced_geos(current_coords, elems, self.displacement,self.npoint, printlevel=2)
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
            print("not ready")
            exit()
            result = ash.functions.functions_parallel.Job_parallel(fragments=all_disp_fragments, theories=[self.theory], numcores=self.numcores,
                allow_theory_parallelization=True, Grad=False, printlevel=self.printlevel, copytheory=True)
        # Assemble gradient
        gradient = np.zeros((numatoms,3))
        # 2-point
        if self.npoint == 2:
            for atindex in range(0,numatoms):
                # Looping over x,yz
                for u in [0,1,2]:
                    # Pos and neg directions
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
                    posval = dispdict[(atindex,u,'+')]
                    grad_component=(posval - orig_energy)/displacement_bohr
                    gradient[atindex,u] = grad_component

        self.energy = orig_energy
        self.gradient = gradient 

        return energy, gradient

def creating_displaced_geos(current_coords,elems,displacement,npoint, printlevel=2):
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
        #frag=ash.Fragment(coords=dispgeo, elems=elems,label=stringlabel, printlevel=0, charge=charge, mult=mult)
        #all_disp_fragments.append(frag)
        list_of_labels.append(calclabel)

    return list_of_displaced_geos, list_of_displacements