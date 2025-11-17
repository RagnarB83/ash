import numpy as np
import copy
import shutil
import time
from ash.modules.module_coords import print_coords_for_atoms,print_internal_coordinate_table,write_XYZ_for_atoms,write_xyzfile,write_coords_all
from ash.functions.functions_general import ashexit, blankline,BC,print_time_rel,print_line_with_mainheader,print_line_with_subheader1,print_if_level
from ash.modules.module_coords import check_charge_mult, fullindex_to_actindex
from ash.modules.module_results import ASH_Results
from ash.modules.module_theory import NumGradclass
from ash.modules.module_singlepoint import Singlepoint
from ash.constants import hartoeV, bohr2ang

def SellaOptimizer(theory=None, fragment=None, charge=None, mult=None, printlevel=2, NumGrad=False):
    """
    Wrapper function around SellaptimizerClass
    """
    timeA=time.time()

    # EARLY EXIT
    if theory is None or fragment is None:
        print("SellaOptimizer requires theory and fragment objects provided. Exiting.")
        ashexit()
    # NOTE: Class does not take fragment and theory
    optimizer=SellaptimizerClass(charge=charge, mult=mult)

    # If NumGrad then we wrap theory object into NumGrad class object
    if NumGrad:
        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
        print("This enables numerical-gradient calculation for theory")
        theory = NumGradclass(theory=theory)

    # Providing theory and fragment to run method. Also constraints
    result = optimizer.run(theory=theory, fragment=fragment, charge=charge, mult=mult,
                           printlevel=printlevel)
    if printlevel >= 1:
        print_time_rel(timeA, modulename='Sella', moduleindex=1)

    return result

# Class for optimization.
class SellaptimizerClass:
        def __init__(self,theory=None, charge=None, mult=None, printlevel=2):

            self.printlevel=printlevel
            print_line_with_mainheader("SellaOptimizer initialization")
            print_if_level("Creating optimizer object", self.printlevel,2)

        def run(self, theory=None, fragment=None, charge=None, mult=None,printlevel=2):

            from sella import Sella
            import ase

            # Creating ASE object
            atoms = ase.atoms.Atoms(fragment.elems,positions=fragment.coords)
            #
            print("Creating ASH-ASE calculator")
            atoms.calc = ASHcalc(fragment=fragment, theory=theory, charge=charge, mult=mult)

            # Set up a Sella Dynamics object
            dyn = Sella(
                atoms,
                trajectory='sella.traj')

            dyn.run(1e-3, 1)


class ASHcalc():
    def __init__(self, fragment=None, theory=None, charge=None, mult=None):
        self.gradientcalls=0
        self.fragment=fragment
        self.theory=theory
        self.results={}
        self.name='ash'
        self.parameters={}
        self.atoms=None
        self.forces=[]
        self.charge=charge
        self.mult=mult
    def get_potential_energy(self, atomsobj):
        return self.potenergy
    def get_forces(self, atomsobj):
        timeA = time.time()
        print("Called ASHcalc get_forces")
        # Check if coordinates have changed. If not, return old forces
        if np.array_equal(atomsobj.get_positions(), self.fragment.coords) == True:
            #coordinates have not changed
            print("Coordinates unchanged.")
            if len(self.forces)==0:
                print("No forces available (1st step?). Will do calulation")
            else:
                print("Returning old forces")
                print_time_rel(timeA, modulename="get_forces: returning old forces")
                return self.forces
        print("Will calculate new forces")

        self.gradientcalls+=1

        # Copy ASE coords into ASH fragment
        self.fragment.coords=copy.copy(atomsobj.positions)
        print("atomsobj.positions:", atomsobj.positions)
        # Calculate E+G
        result = Singlepoint(theory=self.theory, fragment=self.fragment, Grad=True, charge=self.charge, mult=self.mult)
        energy = result.energy
        gradient = result.gradient
        # Converting E and G from Eh and Eh/Bohr to ASE units: eV and eV/Angstrom
        self.potenergy = energy * hartoeV
        print("gradient:", gradient)
        self.forces = -1 * gradient * hartoeV / bohr2ang
        print("Forces:", self.forces)
        # Adding forces to results also (sometimes called)
        self.results['forces'] = self.forces
        # print("potenergy:", self.potenergy)

        print("ASHcalc get_forces done")
        print_time_rel(timeA, modulename="get_forces")
        return self.forces