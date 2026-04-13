import numpy as np
import copy
import shutil
import time
from ash.modules.module_coords import print_coords_for_atoms,write_XYZ_for_atoms,write_xyzfile,write_coords_all, print_internal_coordinate_table_new
from ash.functions.functions_general import ashexit, blankline,BC,print_time_rel,print_line_with_mainheader,print_line_with_subheader1,print_if_level
from ash.modules.module_coords import check_charge_mult, fullindex_to_actindex
from ash.modules.module_results import ASH_Results
from ash.modules.module_theory import NumGradclass
from ash.modules.module_singlepoint import Singlepoint
from ash.constants import hartoeV, bohr2ang

# Sella TS optimizer
# TODO active region
# TODO PBC


def SellaOptimizer(theory=None, fragment=None, charge=None, mult=None, printlevel=2, NumGrad=False,
                   convergence_gmax=1e-4, maxiter=150, result_write_to_disk=False,
                   constraints=None, gamma=0.03):
    """
    Wrapper function around SellaoptimizerClass
    """
    timeA=time.time()

    # EARLY EXIT
    if theory is None or fragment is None:
        print("SellaOptimizer requires theory and fragment objects provided. Exiting.")
        ashexit()
    # NOTE: Class does not take fragment and theory
    optimizer = SellaoptimizerClass(charge=charge, mult=mult, convergence_gmax=convergence_gmax, 
                                   printlevel=printlevel, maxiter=maxiter, result_write_to_disk=result_write_to_disk, 
                                   constraints=constraints, gamma=gamma)

    # If NumGrad then we wrap theory object into NumGrad class object
    if NumGrad:
        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
        print("This enables numerical-gradient calculation for theory")
        theory = NumGradclass(theory=theory)

    # Providing theory and fragment to run method.
    result = optimizer.run(theory=theory, fragment=fragment, charge=charge, mult=mult)
    if printlevel >= 1:
        print_time_rel(timeA, modulename='Sella', moduleindex=1)

    return result

# Class for optimization.
class SellaoptimizerClass:
        def __init__(self,theory=None, charge=None, mult=None, printlevel=2, constraints=None,
                     convergence_gmax=3e-4, maxiter=150, result_write_to_disk=False,
                     gamma=0.4):

            self.printlevel=printlevel
            print_line_with_mainheader("SellaOptimizer initialization")
            print_if_level("Creating optimizer object", self.printlevel,2)

            # Input maxg tolerance in Eh/Bohr
            # Converting to eV/Angstrom for Sella
            self.convergence_gmax=convergence_gmax
            self.tolerance_ev_ang = convergence_gmax * hartoeV / bohr2ang
            self.maxiter = maxiter
            self.result_write_to_disk = result_write_to_disk
            self.constraints = constraints
            self.gamma = gamma


            print_if_level(f"GradMax convergence tolerance: {self.convergence_gmax} Eh/Bohr", self.printlevel, 2)
            print_if_level(f"Converted tolerance for Sella: {self.tolerance_ev_ang} eV/Angstrom", self.printlevel, 2)
            print_if_level(f"Maximum optimization steps: {self.maxiter}", self.printlevel, 2)
            print_if_level(f"Constraints: {self.constraints}", self.printlevel, 2)
            print_if_level(f"Gamma (convergence crit. for iterative eigensolver): {self.gamma}", self.printlevel, 2)

        def setup_constraints(self, atoms, constraints):
            from sella import Constraints
            sellacons = Constraints(atoms)

            # Bonds
            if 'bond' in constraints:
                for bondcon in constraints['bond']:
                    sellacons.fix_bond(tuple(bondcon))
            # Angles
            if 'angle' in constraints:
                for anglecon in constraints['angle']:
                    sellacons.fix_angle(tuple(anglecon))
            # Dihedrals
            if 'dihedral' in constraints:
                for dihedralcon in constraints['dihedral']:
                    sellacons.fix_dihedral(tuple(dihedralcon))
            # XYZ
            if 'xyz' in constraints:
                for xyzcon in constraints['xyz']:
                    sellacons.fix_translation(xyzcon)
            # TODO: partial XYZ constraints
            print("sellacons:", sellacons)
            return sellacons

        def run(self, theory=None, fragment=None, charge=None, mult=None,printlevel=2, constraints=None):

            print_line_with_subheader1("Running Sella optimization")
            from sella import Sella
            import ase

            if constraints is None:
                constraints = self.constraints
            print("constraints:", constraints)

            # Creating ASE object
            fragment.printlevel=0
            atoms = ase.atoms.Atoms(fragment.elems,positions=fragment.coords)

            # Setup constraints for Sella
            sella_constraints=None
            if self.constraints is not None:
                sella_constraints = self.setup_constraints(atoms, constraints)
            print("sella_constraints:", sella_constraints)

            # Attaching calculator
            print("Creating ASH-ASE calculator")
            atoms.calc = ASH_ASE_calculator(theory=theory, fragment=fragment)

            # Set up a Sella Dynamics object
            dyn = Sella(
                atoms, constraints=sella_constraints,
                gamma=self.gamma)

            def write_traj(a=atoms, trajname="sella_optim"):
                fragment.coords = copy.copy(a.get_positions())
                fragment.write_xyzfile(xyzfilename=trajname+'.xyz', writemode='a')

            # Attaching traj function
            #dyn.attach(print_step, interval=1)
            dyn.attach(write_traj, interval=1)

            # Running optimization step by step
            for step in range(self.maxiter):
                conv = dyn.run(self.tolerance_ev_ang, 1)
                # print("Sella step completed. Converged?", conv)
                if conv:
                    print("Converged")
                    break
            if conv is False:
                print()
                print(f"Sella Geometry optimization did not converge in {self.maxiter} steps. Exiting.")
                fragment.write_xyzfile(xyzfilename='Fragment-current.xyz')
                print()
                ashexit()

            # DONE
            if self.printlevel >= 1:
                print()
                print(f"Sella Geometry optimization converged in {step+1} steps!")
                print()

            finalenergy = atoms.calc.energy_eH

            if self.printlevel >= 1:
                print(f"Final optimized energy: {finalenergy} Eh")

            # Writing out fragment file and XYZ file
            fragment.print_system(filename='Fragment-optimized.ygg')
            fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
            fragment.set_energy(finalenergy)

            print("Final geometry")
            fragment.print_coords()
            print()


            # TODO active region
            #Active region XYZ-file
            #if self.ActiveRegion is True:
            #    write_XYZ_for_atoms(fragment.coords, fragment.elems, self.actatoms, "Fragment-optimized_Active")
            #QM-region XYZ-file
            #if isinstance(theory,QMMMTheory):
            #    write_XYZ_for_atoms(fragment.coords, fragment.elems, theory.qmatoms, "Fragment-optimized_QMregion")

            # Printing internal coordinate table
            if self.printlevel >= 2:
                print_internal_coordinate_table_new(fragment,actatoms=fragment.allatoms)
            print()

            # Now returning final Results object
            # Note: could include the geometry in object but can be very large causing printing head-aches on screen, 
            # ignoring for now since the geometry is in the Fragment object anyway
            result = ASH_Results(label="SellaOptimizer", energy=finalenergy)
            if self.result_write_to_disk is True:
                result.write_to_disk(filename="SellaOptimizer.result")
            return result


# Simpler ASH-ASE calculator
class ASH_ASE_calculator:
    def __init__(self, theory=None, fragment=None):
        self.theory = theory
        # Used for elems, charge and mult
        self.fragment = fragment
        self.forcecalls = 0
        self.forces = None
        self.energycalls = 0
        self.energy_eH = None
        self.energy_eV = None
        self.gradient = None
        self.coords=fragment.coords

    def get_potential_energy(self, atomsobj):
        #print("Called ASHcalc get_potential_energy")
        #print("Energy call number:", self.energycalls)
        self.energycalls += 1
        # Have coordinates changed?
        if np.array_equal(atomsobj.get_positions(), self.coords):
            if self.energy_eV is not None:
                # Returning old energy
                return self.energy_eV
            else:
                print("No energy available (1st step?). Will do calculation")
                # ?
                exit()
        return self.energy_eV

    def get_forces(self, atomsobj):
        #print("Force call number:", self.forcecalls)
        self.forcecalls+=1
        #print("Called ASHcalc get_forces")
        # Have coordinates changed?
        if np.array_equal(atomsobj.get_positions(), self.coords):
            if self.forces is not None:
                return self.forces
            else:
                print("Running first E+G calculation")
                print("Note: following Sella printout units are in eV and eV/Ang")
        #print("Will calculate new forces")

        self.coords = copy.copy(atomsobj.positions)
        energy, gradient = self.theory.run(current_coords=self.coords, elems=self.fragment.elems, 
                                 charge=self.fragment.charge, mult=self.fragment.mult, Grad=True)
        #print("New energy:", energy)
        self.energy_eH = energy
        self.energy_eV = energy*hartoeV
        self.gradient=gradient
        self.forces = -gradient * 51.4220674763
        return self.forces
