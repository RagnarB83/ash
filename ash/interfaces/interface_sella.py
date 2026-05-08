import numpy as np
import copy
import shutil
import time
from ash.modules.module_coords import Fragment, print_coords_for_atoms,write_XYZ_for_atoms,write_xyzfile,write_coords_all, print_internal_coordinate_table_new
from ash.functions.functions_general import ashexit, blankline,BC, listdiff,print_time_rel,print_line_with_mainheader,print_line_with_subheader1,print_if_level
from ash.modules.module_coords import check_charge_mult, fullindex_to_actindex
from ash.modules.module_results import ASH_Results
from ash.modules.module_theory import NumGradclass
from ash.modules.module_singlepoint import Singlepoint
from ash.constants import hartoeV, bohr2ang
from ash.modules.module_QMMM import QMMMTheory

# Sella TS optimizer
# TODO active region
# TODO PBC


def SellaOptimizer(theory=None, fragment=None, charge=None, mult=None, printlevel=2, NumGrad=False,
                   convergence_gmax=1e-4, maxiter=150, result_write_to_disk=False,
                   constraints=None, actatoms=None, frozenatoms=None,
                   gamma=0.03, eta=1e-4):
    """
    Wrapper function around SellaoptimizerClass
    """
    timeA=time.time()

    # EARLY EXIT
    #if theory is None or fragment is None:
    #    print("SellaOptimizer requires theory and fragment objects provided. Exiting.")
    #    ashexit()
    # NOTE: Class does not take fragment and theory
    optimizer = SellaoptimizerClass(convergence_gmax=convergence_gmax, 
                                   printlevel=printlevel, maxiter=maxiter, result_write_to_disk=result_write_to_disk, 
                                   constraints=constraints, actatoms=actatoms, frozenatoms=frozenatoms,
                                   gamma=gamma, eta=eta)

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
        def __init__(self,printlevel=2, 
                     convergence_gmax=3e-4, maxiter=150, result_write_to_disk=False,
                     constraints=None, actatoms=None, frozenatoms=None,
                     gamma=0.03, eta=1e-4):

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

            # Active and frozen atoms
            # Check if both defined
            if actatoms is not None and frozenatoms is not None:
                print("Error: both active and frozen atoms defined. Please specify only one of them. Exiting.")
                ashexit()

            self.frozenatoms = frozenatoms
            self.actatoms = actatoms
            self.gamma = gamma
            self.eta = eta


            print_if_level(f"GradMax convergence tolerance: {self.convergence_gmax} Eh/Bohr", self.printlevel, 2)
            print_if_level(f"Converted tolerance for Sella: {self.tolerance_ev_ang} eV/Angstrom", self.printlevel, 2)
            print_if_level(f"Maximum optimization steps: {self.maxiter}", self.printlevel, 2)
            print_if_level(f"Constraints: {self.constraints}", self.printlevel, 2)
            print_if_level(f"Gamma (convergence crit. for iterative eigensolver): {self.gamma}", self.printlevel, 2)
            print_if_level(f"Eta (step size for iterative eigensolver): {self.eta}", self.printlevel, 2)

        # If using Active region then we define the system geometry as only 
        def setup_active_region_geometry(self,fragment):

            if len(self.actatoms) == 0:
                print("Error: List of active atoms (actatoms) provided is empty. This is not allowed.")
                ashexit()
            # Sorting list, otherwise trouble
            self.actatoms.sort()
            print("Active Region option Active. Passing only active-region coordinates to Sella.")
            print("Active atoms list:", self.actatoms)
            print("Number of active atoms:", len(self.actatoms))

            # Check that the actatoms list does not contain atom indices higher than the number of atoms
            largest_atom_index = max(self.actatoms)
            if largest_atom_index >= fragment.numatoms:
                print(BC.FAIL,f"Found active-atom index ({largest_atom_index}) that is larger or equal (>=) than the number of atoms of system ({fragment.numatoms})!",BC.END)
                print(BC.FAIL,"This does not make sense. Please provide a correct actatoms list. Exiting.",BC.END)
                ashexit()

            # Get active region coordinates and elements
            actcoords, actelems = fragment.get_coords_for_atoms(self.actatoms)
            newfrag = Fragment(coords=actcoords, elems=actelems, charge=fragment.charge, mult=fragment.mult, printlevel=0)
            return newfrag

        def setup_constraints(self, atoms, constraints, fragment):
            from sella import Constraints

            sellacons = Constraints(atoms)

            # Bonds
            if constraints is not None:
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
                # XYZ and partial Cart constraints
                if 'xyz' in constraints:
                    for xyzcon in constraints['xyz']:
                        sellacons.fix_translation(xyzcon)
                elif 'xy' in constraints:
                    for xycon in constraints['xy']:
                        sellacons.fix_translation(xycon, directions=[0,1])
                elif 'x' in constraints:
                    for xcon in constraints['x']:
                        sellacons.fix_translation(xcon, directions=[0])
                elif 'y' in constraints:
                    for ycon in constraints['y']:
                        sellacons.fix_translation(ycon, directions=[1])
                elif 'z' in constraints:
                    for zcon in constraints['z']:
                        sellacons.fix_translation(zcon, directions=[2])
                elif 'yz' in constraints:
                    for yzcon in constraints['yz']:
                        sellacons.fix_translation(yzcon, directions=[1,2])
                elif 'xz' in constraints:
                    for xzcon in constraints['xz']:
                        sellacons.fix_translation(xzcon, directions=[0,2])

            # Frozen atoms specified, same as XYZ constraint but specified differently by user
            if self.frozenatoms is not None:
                print("Frozen atoms specified. Adding XYZ constraints for frozen atoms:", self.frozenatoms)
                for frozenatom in self.frozenatoms:
                    sellacons.fix_translation(index=frozenatom)
            print("All Sella constraints:", sellacons)
            return sellacons

        def run(self, theory=None, fragment=None, charge=None, mult=None,constraints=None):

            print_line_with_subheader1("Running Sella optimization")
            from sella import Sella
            import ase

            # Constraints provided to run or at initialization
            if constraints is None:
                constraints = self.constraints
            print("constraints:", constraints)

            # Active region setup. For a big system, we have to pass only the active region geometry to Sella
            if self.actatoms is not None:
                self.original_fragment = copy.deepcopy(fragment)
                self.active_fragment = self.setup_active_region_geometry(fragment)
                print(f"Active region fragment contains {self.active_fragment.numatoms} atoms")
            else:
                self.original_fragment=None # 
                self.active_fragment = fragment


            # Creating ASE object
            fragment.printlevel=0
            atoms = ase.atoms.Atoms(self.active_fragment.elems,positions=self.active_fragment.coords)


            # Setup constraints for Sella
            sella_constraints = None
            if self.constraints is not None or self.frozenatoms is not None:
                sella_constraints = self.setup_constraints(atoms, constraints,fragment)
            print("sella_constraints:", sella_constraints)

            # Attaching calculator
            print("Creating ASH-ASE calculator")
            atoms.calc = ASH_ASE_calculator(theory=theory, fragment=self.active_fragment, 
                                            full_fragment=self.original_fragment, actatoms=self.actatoms)

            # Set up a Sella Dynamics object
            dyn = Sella(
                atoms, constraints=sella_constraints,
                gamma=self.gamma, eta=self.eta)

            def write_traj(a=atoms, trajname="sella_optim"):
                print(f"Writing (active) trajectory to file: {trajname}.xyz")
                self.active_fragment.coords = copy.copy(a.get_positions())
                self.active_fragment.write_xyzfile(xyzfilename=trajname+'.xyz', writemode='a')

            def write_full_traj(a=atoms, trajname="sella_optim_full"):
                print(f"Writing full trajectory to file: {trajname}.xyz")
                #self.original_fragment = copy.copy(a.get_positions())
                atoms.calc.full_fragment.write_xyzfile(xyzfilename=trajname+'.xyz', writemode='a')
            def write_qmregion_traj(a=atoms, trajname="sella_optim_qmregion"):
                print(f"Writing QM-region trajectory to file: {trajname}.xyz")
                qm_elems = [atoms.calc.full_fragment.elems[i] for i in theory.qmatoms]
                qm_coords = np.array([atoms.calc.full_fragment.coords[i] for i in theory.qmatoms])
                frag = Fragment(coords=qm_coords, elems=qm_elems, printlevel=0)
                frag.write_xyzfile(xyzfilename=trajname+'.xyz', writemode='a')


            # Attaching traj function
            #dyn.attach(print_step, interval=1)
            dyn.attach(write_traj, interval=1)
            # Attaching full traj write also if using active region
            if self.actatoms is not None:
                dyn.attach(write_full_traj, interval=1)
            if isinstance(theory, QMMMTheory):
                dyn.attach(write_qmregion_traj, interval=1)

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


            if self.actatoms is None:
                print("Final geometry:")
                fragment.print_coords()
            print()

            #Active region XYZ-file
            if self.actatoms is None:
                write_XYZ_for_atoms(fragment.coords, fragment.elems, self.actatoms, 
                                    "Fragment-optimized_Active")

            #QM-region XYZ-file
            if isinstance(theory,QMMMTheory):
                write_XYZ_for_atoms(fragment.coords, fragment.elems, theory.qmatoms, 
                                    "Fragment-optimized_QMregion")

            # Printing internal coordinate table
            if self.printlevel >= 2:
                if fragment.numatoms < 50:
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
    def __init__(self, theory=None, fragment=None, full_fragment=None, actatoms=None):
        self.theory = theory
        # Used for elems, charge and mult
        self.fragment = fragment
        self.full_fragment = full_fragment
        self.actatoms = actatoms
        self.forcecalls = 0
        self.forces = None
        self.energycalls = 0
        self.energy_eH = None
        self.energy_eV = None
        self.gradient = None
        # Initializing coordinates used by Sella
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

        # Copying current active coordinates from Atoms object
        self.coords = copy.copy(atomsobj.positions)
        # Updating fragment geometry with new active region geometry from Sella
        self.fragment.coords = copy.copy(self.coords)
        
        # Active region or not
        if self.actatoms is not None:

            #Replacing act-region coordinates in full_coords with coords from currcoords
            self.full_fragment.coords[self.actatoms] = self.coords

            # Computing E+G of full system
            energy, fullgrad = self.theory.run(current_coords=self.full_fragment.coords, elems=self.full_fragment.elems, 
                                    charge=self.full_fragment.charge, mult=self.full_fragment.mult, Grad=True)
            # Extracting active region gradient from full gradient
            Grad_act = np.array([fullgrad[i] for i in self.actatoms])
            self.gradient = Grad_act
            self.forces = -Grad_act * 51.4220674763
        # No active region
        else:
            energy, gradient = self.theory.run(current_coords=self.coords, elems=self.fragment.elems, 
                                    charge=self.fragment.charge, mult=self.fragment.mult, Grad=True)
            self.gradient = gradient
            self.forces = -gradient * 51.4220674763

        # Energy
        self.energy_eH = energy
        self.energy_eV = energy*hartoeV

        return self.forces
