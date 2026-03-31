from __future__ import annotations
from ctypes import c_double, c_int, pointer
import functools
from typing import Callable, Optional
import numpy as np
from numpy.ctypeslib import as_array
from numpy.typing import ArrayLike
import signal
import os

import os
import time
from ash.functions.functions_general import ashexit, BC,print_time_rel,print_line_with_mainheader,search_list_of_lists_for_index,print_if_level
from ash.modules.module_coords import check_charge_mult, print_internal_coordinate_table_new,write_xyzfile,elemstonuccharges, print_coords_for_atoms
from ash.modules.module_coords_PBC import write_CIF_file, write_XSF_file, write_POSCAR_file, cell_vectors_to_params, cell_volume, align_to_standard_orientation
from ash.modules.module_theory import NumGradclass
from ash.modules.module_results import ASH_Results
from ash.modules.module_freq import NumFreq,AnFreq,calc_hessian_xtb
from ash.modules.module_QMMM import QMMMTheory
from ash.modules.module_oniom import ONIOMTheory

# Basic interface to the Fortran-based DL-FIND library via the libdlfind interface

def DLFIND_optimizer(jobtype=None, theory=None, fragment=None, fragment2=None, charge=None, mult=None, 
                     maxcycle=250, tolerance=4.5E-4, tolerance_e=1E-6,
                     actatoms=None, frozenatoms=None, residues=None, constraints=None,
                     printlevel=2, NumGrad=False, delta=0.01,
                     icoord=None, iopt=None, nimage=None, 
                     hessian_choice="numfreq", inithessian=0, 
                     numfreq_npoint=1, numfreq_displacement=0.005, numfreq_hessatoms=None,
                     numfreq_force_projection=None, print_atoms_list=None,
                     force_noPBC=False, PBC_format_option='CIF'):
    """
    Wrapper function around DLFIND_optimizerClass
    """
    timeA=time.time()
    #EARLY EXIT
    if theory is None or fragment is None:
        print("DLFIND_optimizer requires theoryNumFreq and fragment objects provided. Exiting.")
        ashexit()
    optimizer=DLFIND_optimizerClass(jobtype=jobtype, theory=theory, fragment=fragment, fragment2=fragment2, charge=charge, mult=mult, actatoms=actatoms,
                                    frozenatoms=frozenatoms,residues=residues, constraints=constraints, delta=delta,
                                    printlevel=printlevel, icoord=icoord,iopt=iopt, maxcycle=maxcycle, 
                                    tolerance=tolerance,tolerance_e=tolerance_e, 
                                    nimage=nimage, 
                                    hessian_choice=hessian_choice, inithessian=inithessian, 
                                    numfreq_npoint=numfreq_npoint,numfreq_displacement=numfreq_displacement,
                                    numfreq_hessatoms=numfreq_hessatoms,numfreq_force_projection=numfreq_force_projection,
                                    print_atoms_list=print_atoms_list,
                                    force_noPBC=force_noPBC, PBC_format_option=PBC_format_option)

    # If NumGrad then we wrap theory object into NumGrad class object
    if NumGrad:
        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
        print("This enables numerical-gradient calculation for theory")
        theory = NumGradclass(theory=theory)

    # Providing theory and fragment to run method. Also constraints
    result = optimizer.run(theory=theory, fragment=fragment, charge=charge, mult=mult)
    if printlevel >= 1:
        print_time_rel(timeA, modulename='DL-FIND', moduleindex=1)

    return result

# Class for optimization.
class DLFIND_optimizerClass:

    def __init__(self,jobtype=None, fragment=None, fragment2=None, theory=None, charge=None, mult=None, 
                 maxcycle=250, tolerance=4.5E-4, tolerance_e=1E-6, 
                 printlevel=2, result_write_to_disk=True, actatoms=None, frozenatoms=None, residues=None, constraints=None,
                 icoord=None, iopt=None, nimage=None, delta=0.01, 
                 hessian_choice='numfreq', inithessian=None, 
                 numfreq_npoint=1,numfreq_displacement=0.005,numfreq_force_projection=None,
                 numfreq_hessatoms=None, print_atoms_list=None,
                 force_noPBC=False, PBC_format_option='CIF'):

        print_line_with_mainheader("DLFIND_optimizer initialization")
        print()
        print("If you use DL-FIND for your research make sure to cite:")
        print("DL-FIND: an Open-Source Geometry Optimizer for Atomistic Simulations, J. Kästner, J. M. Carr, T. W. Keal, W. Thiel, A. Wander, P. Sherwood, J. Phys. Chem. A, 2009, 113, 11856.")
        print()
        self.printlevel=printlevel

        print("Importing libdlfind package\n")
        try:
            from libdlfind import dl_find
            from libdlfind.callback import (dlf_get_gradient_wrapper,
                                    dlf_put_coords_wrapper, make_dlf_get_params)
        except:
            print("Error importing libdlfind")
            print("Have you installed: https://github.com/digital-chemistry-laboratory/libdlfind")
            print("Quick-fix: pip install libdlfind")
            ashexit()

        # EARLY EXITS
        #if theory is None or fragment is None:
        #    print("DLFIND_optimizer requires theory and fragment objects provided. Exiting.")
        #    ashexit()

        if jobtype is None and icoord is None:
            print("Error: You must either select a jobtype keyword (e.g. opt, neb, dimer, instanton) or select DL-FIND icoord and iopt codes")
            print("Example: DLFIND_optimizer(jobtype='opt') ")
            ashexit()
        elif jobtype == "opt":
            print("jobtype: opt chosen")
            print("Choosing icoord=1 (HDLC internal coordinates) and iopt=3 (L-BFGS minimizer)")
            print("For other coordinate-systems: choose icoord=0 (cartesian), icoord=2 (hdlc-tc), icoord=3 (dlc-prim), icoord=3 (dlc-tc)")
            print("For other opt algorithms: choose iopt codes: 0: sd, 1: cg-autorestart, 2: cg-restart10, 3: lbfgs, 10: P-RFO")
            icoord=1
            iopt=3
        elif jobtype == "tsopt" or jobtype == "ts":
            print("jobtype: tsopt chosen")
            print("Choosing icoord=3 (HDLC internal coordinates) and iopt=10 (P-RFO)")
            print("Note: inithessian option is:", inithessian)
            icoord=3
            iopt=10
        elif jobtype == "neb":
            print("jobtype: neb chosen")
            print("Choosing icoord=120 (NEB with frozen endpoints) and iopt=3 (L-BFGS)")
            icoord=120
            iopt=3
        elif jobtype == "dimer":
            print("jobtype: dimer chosen")
            print("Choosing icoord=210 (Dimer) and iopt=3 (L-BFGS)")
            icoord=210
            iopt=3
        elif jobtype == "qts" or jobtype == "instanton" :
            print("jobtype: qts chosen (a.k.a. instanton)")
            print("Choosing icoord=190 (qts) and iopt=3 (L-BFGS)")
            icoord=190
            iopt=3
        else:
            print("No jobype selected.")
            print(f"Will start job based on chosen icoord={icoord} and iopt={iopt}")


        self.fragment=fragment
        self.fragment2=fragment2
        self.theory=theory

        # Periodic
        self.PBC_format_option=PBC_format_option
        self.PBC=False # False by default unless detected in theory
        self.force_noPBC=force_noPBC
        
        #############
        #HESSIAN
        #############
        # Initial-Hessian DL-FIND option
        #inithessian options (for P-RFO, iopt=10): 
        # 0: external program. if fails goes to 2-point FD
        # 1: 1-point FD
        # 2: 2-point FD
        # 3: diagonal 1-point FD
        # 4: identity matrix
        # 5 (INSTANTON-ONLY: read from file, has to be named qts_hessian.txt)
        self.inithessian=inithessian
        # For inithessian=0 we have other multiple options, decided by hessian_choice
        self.hessian_choice=hessian_choice
        # hessian_choice: numfreq, anfreq, random
        #For choice numfreq pointchoice: 1 or 2
        self.numfreq_npoint=numfreq_npoint
        self.numfreq_displacement=numfreq_displacement
        self.numfreq_hessatoms=numfreq_hessatoms
        self.numfreq_force_projection=numfreq_force_projection

        # Optimizer options
        # internal Coordinate-system or job-type
        self.icoord=icoord # 0: cart, 1: hdlc-prim, 2: hdlc-tc, 3:dlc-prim, 4:dlc-tc, 10-14: Lagrange-Newton CI search
        # 190: Quantum TS search
        # 210: Dimer , 200 another variant
        # 120: NEB
        # Algorithms
        self.iopt=iopt # algo: 0:sd, 1:cg-autorestart, 2:cg-restart10, 3:lbfgs, 10: P-RFO
        # 11: thermal-analysis, 12: quantum thermal analysis
        self.maxcycle=maxcycle
        #Tolerances
        self.tolerance=tolerance
        self.tolerance_e=tolerance_e
        # NEB
        self.nimage=nimage
        #Dimer
        self.delta=delta

        # Residues for HDLC
        self.residues=residues

        #Constraints
        self.constraints=constraints
        self.actatoms=actatoms
        self.frozenatoms=frozenatoms

        self.print_atoms_list=print_atoms_list
        self.result_write_to_disk=result_write_to_disk

        #Tracking DL-FIND cycles
        self.dlfind_eg_calls=0
        self.dlfind_opt_cycles=0
        self.dlfind_neb_cycles=0
        self.dlfind_dimer_cycles=0


        self.NEB_energies_dict={}
        self.NEB_geometries={}

        self.runcounter=0


        # Create function to calculate energies and gradients
        @dlf_get_gradient_wrapper
        def ash_e_g_func(coordinates, iimage, kiter, theory):
            self.dlfind_eg_calls+=1
            coordinates_ang = coordinates*0.5291772109303

            if self.PBC:
                print("Inside PBC")

                # Split  coords into atomic and lattic
                R_geo = coordinates_ang[:-4]
                origin = coordinates_ang[-4]
                H_geo = coordinates_ang[-3:] - origin

                # --- Enforce Standard Orientation in each step ---
                print("Enforcing orientation")
                # 1. Ensure the Origin dummy atom stays at exactly 0,0,0
                origin[:] = 0.0
                # 2. Force H_geo to be strictly upper-triangular
                # Vector A: Only Ax is allowed (Ay and Az are zero)
                H_geo[0, 1] = 0.0  # ay = 0
                H_geo[0, 2] = 0.0  # az = 0
                # Vector B: Only Bx and By are allowed (Bz is zero)
                H_geo[1, 2] = 0.0  # bz = 0
                # -----------------------------------------------------
                s = np.dot(R_geo - origin, self.H_ref_inv)
                R_phys = np.dot(s, H_geo) + origin
                #Update cell parameters in theory
                self.theory.update_cell(H_geo)

                self.full_current_coords=R_phys
                self.fragment.replace_coords(self.fragment.elems, self.full_current_coords, conn=False)

                #PRINTING ACTIVE GEOMETRY IN EACH GEOMETRIC ITERATION
                self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")
                if self.printlevel >= 1:
                    print(f"Current geometry (Å) in step {self.dlfind_opt_cycles} (print_atoms_list region)")
                    print("---------------------------------------------------")
                    print_coords_for_atoms(R_phys, self.elems_phys, self.print_atoms_list)
                    print("")
                    print("Note: printed only print_atoms_list (this is not necessarily all atoms) ")
                    print(f"Current cell vectors (Å):{H_geo}")
                    print(f"Current cell volume (Å):{cell_volume(H_geo)}")

                # E + G from theory
                energy,grad_phys=self.theory.run(current_coords=R_phys, elems=self.elems_phys, 
                                            charge=self.charge, mult=self.mult, Grad=True)
                self.energy = energy

                # Transformation
                # M is the transformation matrix: R_phys = R_geo @ M
                M = np.dot(self.H_ref_inv, H_geo)
                grad_Rgeo = np.dot(grad_phys, M.T)

                # Convection, implicit lattice gradient
                #grad_convection = np.dot(s.T, grad_phys)

                # Lattice gradient and masking
                #Total lattice gradient: current theory cell-gradient + convection
                #grad_latt_total = self.theory.cell_gradient
                grad_latt_total = self.theory.get_cell_gradient()
                # Standard orientation mask:
                # This zeros out: a_y, a_z, and b_z
                mask = np.array([
                    [1, 0, 0], # dE/dax (ay, az frozen)
                    [1, 1, 0], # dE/dbx, dE/dby (bz frozen)
                    [1, 1, 1]  # dE/dcx, dE/dcy, dE/dcz (all free)
                ])
                grad_latt_masked = grad_latt_total * mask
                # Making sure origin is zero
                grad_origin = np.zeros((1, 3))
                # Final modified gradient to pass to geomeTRIC
                mod_gradient = np.concatenate([
                        grad_Rgeo,         # (N, 3)
                        grad_origin,       # (1, 3)
                        grad_latt_masked   # (3, 3)
                    ], axis=0)
                return energy, mod_gradient

            else:
                self.fragment.coords=coordinates_ang
                #PRINTING ACTIVE GEOMETRY IN EACH GEOMETRIC ITERATION
                self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")
                if self.printlevel >= 1:
                    print(f"Current geometry (Å) in step {self.dlfind_opt_cycles} (print_atoms_list region)")
                    print("---------------------------------------------------")
                    print_coords_for_atoms(coordinates_ang, self.fragment.elems, self.print_atoms_list)
                    print("")
                    print("Note: printed only print_atoms_list (this is not necessarily all atoms) ")
                energy, gradient = self.theory.run(current_coords=coordinates_ang, elems=self.fragment.elems, charge=self.charge, mult=self.mult, Grad=True)

                # NEB: Storing current geometry for each image
                # Note: spawned climbing image will be number nimage
                if self.icoord >= 100 and self.icoord < 150 :
                    self.NEB_geometries[iimage] = coordinates_ang
                    self.NEB_energies_dict[iimage] = energy

                return energy, gradient

        # Modified wrapper function
        def dlf_get_hessian_wrapper(func: Callable) -> Callable:
            """Factory function for dlf_get_hessian."""
            @functools.wraps(func)
            def wrapper(
                nvar2: int,
                coords: pointer[c_double],
                hessian: pointer[c_double],
                status: pointer[c_int],
                *args,
                **kwargs,
            ) -> None:
                nvar2=self.fragment.numatoms*3
                coords_ = as_array(coords, shape=(nvar2,)).reshape((-1, 3))
                hessian_ = as_array(hessian, shape=(nvar2,nvar2))
                hessian = func(coords_, *args, **kwargs)
                hessian_[:, :] = hessian
                status[0] = c_int(0)
                return
            return wrapper

        # How we get the Hessian from ASH
        @dlf_get_hessian_wrapper
        def hess_func(coords):
            nvar=self.fragment.numatoms*3
            #Updating coordinates in fragment, just to be sure
            self.fragment.coords=coords*0.5291772109303
            # Get Hessian
            #TODO: parallelization
            if type(self.hessian_choice) == np.ndarray:
                print("A Numpy array was detected as Hessian choice. Passing over to DL-FIND")
                hessian = self.hessian_choice
            elif self.hessian_choice == "numfreq":
                print("NumFreq option requested")
                print("NumFreq Npoint:", self.numfreq_npoint)

                result_freq = NumFreq(theory=self.theory, fragment=self.fragment, printlevel=0, 
                                        npoint=self.numfreq_npoint, displacement=self.numfreq_displacement,
                                        hessatoms=self.numfreq_hessatoms,force_projection=self.numfreq_force_projection,
                                        runmode='serial', 
                                        numcores=self.theory.numcores)
                hessian = result_freq.hessian
            elif self.hessian_choice == "anfreq":
                print("AnFreq option requested")
                result_freq = AnFreq(theory=self.theory, fragment=self.fragment, printlevel=0)
                hessian = result_freq.hessian
            elif self.hessian_choice == "xtb":
                print("xTB Hessian option requested")
                #Calling xtb to get Hessian, written to disk. Returns name of Hessianfile
                hessianfile = calc_hessian_xtb(fragment=fragment, actatoms=self.fragment.allatoms, 
                                                numcores=self.theory.numcores, use_xtb_feature=True, 
                                                charge=charge, mult=mult)
                hessian = np.loadtxt("Hessian_from_xtb")
            elif 'file:' in self.hessian_choice:
                print("A file was detected as Hessian choice:", self.hessian_choice)
                hessianfile = self.hessian_choice.replace("file:","")
                if os.path.isfile(hessianfile) is False:
                    print(f"File {hessianfile} does not exist.")
                    ashexit()
                hessian=hessian = np.loadtxt(hessianfile)

            elif self.hessian_choice == "random":
                hessian = np.random.random((nvar,nvar))
            print("ASH hessian:", hessian)

            return hessian

        # Create function to store results from DL-FIND
        #@dlf_put_coords_wrapper
        def store_results(a,nvar,switch, energy, coordinates, iam):
            if switch > 0:
                coords = as_array(coordinates, (nvar,)).reshape(-1, 3)
                coordinates_ang = coords*0.5291772109303
            else:
                #print("switch neg")
                # Write out NEB path if switch -1
                if self.icoord >= 100 and self.icoord < 150 and switch == -1:
                    self.dlfind_neb_cycles+=1
                    print("="*70)
                    print(f"DLFIND NEB-OPTIMIZATION CYCLE {self.dlfind_neb_cycles}")
                    print("="*70)
                    #print("Switch -1, writing out current NEB-path without CI")
                    with open("DLFIND_NEBpath_all.xyz", "a") as trajfile:
                        for imageid in list(range(1,self.nimage)):
                            trajfile.write(str(self.fragment.numatoms) + "\n")
                            trajfile.write(f"Image {imageid}. Energy: {self.NEB_energies_dict[imageid]}  \n")
                            for el, cord in zip(self.fragment.elems, self.NEB_geometries[imageid]):
                                trajfile.write(el + "  " + str(cord[0]) + " " + str(cord[1]) + " " + str(cord[2]) + "\n")
                    with open("DLFIND_NEBpath_current.xyz", "w") as trajfile:
                        for imageid in list(range(1,self.nimage)):
                            trajfile.write(str(self.fragment.numatoms) + "\n")
                            trajfile.write(f"Image {imageid}. Energy: {self.NEB_energies_dict[imageid]}  \n")
                            for el, cord in zip(self.fragment.elems, self.NEB_geometries[imageid]):
                                trajfile.write(el + "  " + str(cord[0]) + " " + str(cord[1]) + " " + str(cord[2]) + "\n")

                    # Writing out CI once it has been spawned
                    if nimage in self.NEB_geometries:
                        print("Writing out current trajectory for climbing image as: DLFIND_CIgeo_traj.xyz ")
                        write_xyzfile(fragment.elems, self.NEB_geometries[nimage], "DLFIND_CIgeo_traj", printlevel=2, writemode='a', title=f"Energy: {self.NEB_energies_dict[nimage]}")
                return
            # Write traj to disk

            # Traj-writing for regular opt
            if self.icoord < 100:
                self.dlfind_opt_cycles+=1
                #print("="*70)
                #print(f"DLFIND OPTIMIZATION CYCLE {self.dlfind_opt_cycles}")
                #print("="*70)
                #Storing current coordinates
                #traj_coords.append(np.array(coordinates_ang))
                print_if_level(f"Writing regular-opt traj",self.printlevel,1)
                write_xyzfile(self.fragment.elems, coordinates_ang, "DLFIND_opt_traj", printlevel=self.printlevel, writemode='a', title=f"Energy: {energy}")
                self.current_geo=coordinates_ang
            # Traj-writing for dimer
            elif self.icoord >= 200:
                print("Writing Dimer traj")
                if switch == 1:
                    # 1: actual geometry
                    write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_dimertraj_1", printlevel=self.printlevel, writemode='a', title=f"Energy: {energy}")
                    self.current_geo=coordinates_ang
                elif switch == 2:
                    # Approximate:
                    self.dlfind_dimer_cycles+=1
                    # transition mode
                    write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_dimertraj_2", printlevel=self.printlevel, writemode='a', title=f"Energy: {energy}")
                elif switch == 3:
                    write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_dimertraj_3", printlevel=self.printlevel, writemode='a', title=f"Energy: {energy}")
            self.traj_energies.append(energy)

            return

        self.dlf_get_gradient = functools.partial(ash_e_g_func, theory=self.theory)
        self.dlf_get_hessian = functools.partial(hess_func)
        self.dlf_put_coords = functools.partial( store_results, None)

    # Should be run only once
    def setup_PBC(self):
        
        # Real elements
        self.elems_phys=self.fragment.elems
        # Align to standard orientation
        aligned_atom_coords, aligned_vectors = align_to_standard_orientation(self.fragment.coords, 
                                                                                  self.theory.periodic_cell_vectors)
        self.fragment.coords=aligned_atom_coords
        self.theory.update_cell(aligned_vectors)

        # Reference
        self.H_ref = aligned_vectors.copy()
        self.H_ref_inv = np.linalg.inv(self.H_ref)

        # Defining DLFIND_coords to have aligned coords and 4 dummyatoms
        self.DLFIND_coords = np.concatenate((aligned_atom_coords,[[0.0,0.0,0.0]],aligned_vectors),axis=0)
        self.DLFIND_elems = self.fragment.elems+ ['F','F','F','F']

    def print_settings(self):
        # Print-atoms choice
        # If not specified then active-region or all-atoms
        if self.print_atoms_list is None:
            #Print-atoms list not specified. What to do:
            if self.actatoms is not None:
                #If QM/MM object then QM-region:
                if isinstance(self.theory,QMMMTheory):
                    print("Theory class: QMMMTheory")
                    print("Will by default print only QM-region in output (use print_atoms_list option to change)")
                    self.print_atoms_list=self.theory.qmatoms
                elif isinstance(self.theory,ONIOMTheory):
                    print("Theory class: ONIOMTheory")
                    print("Will by default print only Region1 in output (use print_atoms_list option to change)")
                    self.print_atoms_list=self.theory.regions_N[0]
                else:
                    # Print actatoms since using Active Region (can be too much)
                    self.print_atoms_list=self.actatoms
            else:
                #No act-region. Print all atoms
                self.print_atoms_list=self.fragment.allatoms

    def setup_constraints_act_frozen(self):

        ########################################
        # ACTIVE/FROZEN AND RESIDUE HANDLING
        ########################################
        if self.residues is None:
            print_if_level("No residues provided to optimizer. Creating a single residue for whole active system.",self.printlevel,2)
        else:
            print("Residues provided to optimizer:", self.residues)

        # What to optimize etc.
        self.spec=[]

        if self.PBC:
            allatoms = self.fragment.allatoms + [self.fragment.numatoms, self.fragment.numatoms+1, self.fragment.numatoms+2, self.fragment.numatoms+3]
            numatoms=self.fragment.numatoms + 4
            elems = self.fragment.elems + ['F','F','F','F']
        else:
            allatoms = self.fragment.allatoms
            numatoms=self.fragment.numatoms
            elems = self.fragment.elems


        # First identify possible frozen constraints defined in constraints dict
        if self.constraints is not None:
            print("RB here")
            # Check if any Cartesian constraint is present 
            if any(k in self.constraints for k in {'x','y','z','xy','xz','yz','xyz'}):
                if self.frozenatoms is None:
                    self.frozenatoms=[]
                print_if_level(f"Cartesian constraints found in constraints dict.", self.printlevel,2 )
                # Grab possible xyz constraints frm constraints dict
                frozenatoms_x = self.constraints.get('x',[])
                frozenatoms_y = self.constraints.get('y',[])
                frozenatoms_z = self.constraints.get('z',[])
                frozenatoms_xy = self.constraints.get('xy',[])
                frozenatoms_xz = self.constraints.get('xz',[])
                frozenatoms_yz = self.constraints.get('yz',[])
                frozenatoms_xyz = self.constraints.get('xyz',[])
                # XYZ constraints are the same frozenatoms, adding
                self.frozenatoms = self.frozenatoms+frozenatoms_xyz
                print("frozenatoms_z:", frozenatoms_z)
        if self.actatoms is not None:
            print_if_level("Actatoms provided:", self.actatoms)

            if self.PBC:
                print("PBC detected. Adding 4 dummy atoms to actatoms if not already present")
                for i in range(self.fragment.numatoms, self.fragment.numatoms+4):
                    if i not in self.actatoms:
                        self.actatoms.append(i)

            if self.frozenatoms is not None:
                if len(self.frozenatoms) > 0:
                    print("frozenatoms:", self.frozenatoms)
                    print("Error: actatoms and frozenatoms cannot both be defined")
                    ashexit()
            print_if_level(f"All atoms: {allatoms}", self.printlevel,2 )

            for i in self.fragment.allatoms:
                if i in self.actatoms:
                    if self.residues is not None:
                        self.spec.append(search_list_of_lists_for_index(i,self.residues)+1)
                    else:
                        self.spec.append(1)
                else:
                    self.spec.append(-1)
        elif self.frozenatoms is not None:
            print_if_level(f"Frozenatoms provided: {self.frozenatoms}", self.printlevel,2 )
            print_if_level(f"All atoms: {self.fragment.allatoms}", self.printlevel,2 )
            # Loopign over all atoms, 
            # Adding -1 for frozen, +1 for active, and if residues provided then adding residue number for active atoms
            # Also adding -2,-3,-4 for frozen atoms with Cartesian constraints in x,y,z and -23,-24,-34 for frozen atoms with xy,xz,yz constraints
            for i in allatoms:
                if i in self.frozenatoms:
                    self.spec.append(-1)
                elif i in frozenatoms_x:
                    self.spec.append(-2)
                elif i in frozenatoms_y:
                    self.spec.append(-3)
                elif i in frozenatoms_z:
                    self.spec.append(-4)
                elif i in frozenatoms_xy:
                    self.spec.append(-23)
                elif i in frozenatoms_xz:
                    self.spec.append(-24)
                elif i in frozenatoms_yz:
                    self.spec.append(-34)
                else:
                    if self.residues is not None:
                        self.spec.append(search_list_of_lists_for_index(i,self.residues)+1)
                    else:
                        self.spec.append(1)
        else:
            print_if_level("Case: no actatoms or frozenatoms provided. All atoms will be active.", self.printlevel,2)
            print_if_level(f"All atoms: {allatoms}", self.printlevel,2)
            if self.residues is None:
                # If no residues provided then all atoms get spec 1 (active)
                # Doing all real atoms
                #for i in self.fragment.allatoms:
                self.spec=[1 for i in self.fragment.allatoms]
                print("self.spec:", self.spec)
                if self.PBC:
                    print("PBC detected. Adding 4 dummy atoms as a separate residue")
                    self.spec = self.spec + [2,2,2,2]
            else:
                print_if_level(f"Residues provided: {self.residues}", self.printlevel,2)
                for i in allatoms:
                    resid = search_list_of_lists_for_index(i,self.residues)
                    self.spec.append(resid+1)

        # Nuclear charges
        nuccharges  = elemstonuccharges(elems)
        self.spec=self.spec + nuccharges

        # Constraints. should be dict: constraints={'bond':[[0,1]], 'angle':[[98,99,100]]}
        if self.constraints is not None:
            print_if_level(f"Constraints passed: {self.constraints}", self.printlevel,2)
            self.numcons=0
            conlist=[]
            for k,v in self.constraints.items():
                if k == 'bond' or k == 'distance':
                    print_if_level(f"Found bond constraint between atoms: {v}", self.printlevel,2)
                    for x in v:
                        b = [1,x[0]+1,x[1]+1,0,0]
                        conlist += b
                        self.numcons+=1
                elif k == 'angle':
                    print_if_level(f"Found angle constraint between atoms: {v}", self.printlevel,2)
                    for x in v:
                        b = [2,x[0]+1,x[1]+1,x[2]+1,0]
                        conlist += b
                        self.numcons+=1
                elif k == 'dihedral' or k == 'torsion':
                    print_if_level(f"Found dihedral constraint between atoms: {v}", self.printlevel,2)
                    for x in v:
                        b = [3,x[0]+1,x[1]+1,x[2]+1,x[3]+1]
                        conlist += b
                        self.numcons+=1
            print_if_level(f"DL-FIND constraints-list: {conlist}", self.printlevel,2)
            print_if_level(f"Number of constraints: {self.numcons}", self.printlevel,2)
            self.spec = self.spec + conlist
        else:
            print_if_level("No constraints present", self.printlevel,2)
            self.numcons=0

        # Spec
        self.spec=self.spec+[1 for i in list(range(numatoms))] #?

        self.nspec=len(self.spec)

        print("DL-FIND spec list:", self.spec)


    def prepare_run(self):

        from libdlfind.callback import make_dlf_get_params
        self.traj_energies = []
        self.current_geo = []
        # Converting coordinates from Angstrom to Bohr
        positions = self.fragment.coords * 1.88972612546
        nz = self.fragment.numatoms
        if self.PBC:
            print("Preparing for PBC optimization. Using aligned coordinates with 4 dummy atoms")
            print("self.DLFIND_coords:", self.DLFIND_coords)
            positions = self.DLFIND_coords * 1.88972612546
            nz = self.fragment.numatoms + 4

        # Possible Fragment2 handling
        if self.fragment2 is not None:
            print("Fragment2 provided. This only makes sense for NEB and dimer jobs")
            positions2 = self.fragment2.coords * 1.88972612546
            nframe=1
        else:
            positions2=None
            nframe=0

        # Setup constraints and frozen/active stuff
        self.setup_constraints_act_frozen()

        self.dlf_get_params = make_dlf_get_params(coords=positions, coords2=positions2, icoord=self.icoord, 
                                                  iopt=self.iopt, maxcycle=self.maxcycle,tolerance=self.tolerance,
                                                  tolerance_e=self.tolerance_e, inithessian=self.inithessian,
                                                  nframe=nframe, nz=nz,
                                                  ncons=self.numcons, delta=self.delta,
                                                  spec=self.spec, printl=self.printlevel, nimage=self.nimage)

        # Delete old traj file before beginning
        remove_files=['DLFIND_opt_traj.xyz','DLFIND_dimertraj_1.xyz', 'DLFIND_dimertraj_2.xyz','DLFIND_dimertraj_3.xyz','DLFIND_NEBpath_current.xyz', 'DLFIND_NEBpath_all.xyz', 'DLFIND_CIgeo_traj.xyz']
        print_if_level(f"Removing possible old files: {remove_files}", self.printlevel,2)
        for rfile in remove_files:
            try:
                os.remove(rfile)
                print_if_level(f"removed {rfile} ", self.printlevel,2)
            except FileNotFoundError:
                #print(f"file {rfile} not found")
                pass

        print_if_level(f"\nArguments passed to DL-FIND:", self.printlevel,2)
        print_if_level(f"icoord: {self.icoord}", self.printlevel,2)
        print_if_level(f"iopt: {self.iopt}", self.printlevel,2)
        print_if_level(f"maxcycle: {self.maxcycle}", self.printlevel,2)
        print_if_level(f"spec ({len(self.spec)}): {self.spec}", self.printlevel,2)
        if self.icoord == 120:
            print_if_level(f"NEB nimage: {self.nimage}", self.printlevel,2)

    def run(self, theory=None, fragment=None, fragment2=None, constraints=None, charge=None, mult=None):
        from libdlfind import dl_find


        # Update self fragment if a run fragment was provided
        if fragment is not None:
            self.fragment=fragment

        # Update self theory if a run fragment was provided
        if theory is not None:
            self.theory=theory

        # Check if PBCs used by theory
        if getattr(self.theory, "periodic", False):
            print("Detected periodicity in Theory object")
            if self.force_noPBC is True:
                print("force_noPBC flag is True. Will run optimization without PBC")
                self.PBC=False
            else:
                print("Activating periodic routines ")
                print("Setting up PBC for DL-FIND optimization")
                self.setup_PBC()
                self.PBC=True
                print("PBC setup complete")
                if fragment2 is None and self.fragment2 is None:
                    nvarin=self.fragment.numatoms * 3 + 4*3 # 4 dummy atoms with 3 coords each
                    nvarin2=0
                # TODO: fragment2 
                #elif fragment2 is not None:
                #    nvarin = self.fragment.numatoms * 3 + 4*3 # 4 dummy atoms with 3 coords each
                #    nvarin2 = self.fragment2.numatoms * 3 
                #elif self.fragment2 is not None:
                #    nvarin = self.fragment.numatoms * 3
                #    nvarin2 = self.fragment2.numatoms * 3
                # Update constraints if provided
        else:
            if fragment2 is None and self.fragment2 is None:
                nvarin=self.fragment.numatoms * 3
                nvarin2=0
            elif fragment2 is not None:
                nvarin = self.fragment.numatoms * 3
                nvarin2 = self.fragment2.numatoms * 3
            elif self.fragment2 is not None:
                nvarin = self.fragment.numatoms * 3
                nvarin2 = self.fragment2.numatoms * 3
            # Update constraints if provided
        if constraints is not None:
            self.constraints=constraints

        if self.runcounter == 0:
            self.print_settings()

        # Prepare run, including constraints etc.
        self.prepare_run()
        self.charge, self.mult = check_charge_mult(charge, mult, self.theory.theorytype, self.fragment, 
                                         "DLFIND-optimizer", theory=self.theory, printlevel=self.printlevel)

        # Run DL-FIND
        print("Now starting DL-FIND")

        def _sigint_handler(signum, frame):
            print("\nCtrl-C caught! Aborting DL-FIND run...")
            signal.signal(signal.SIGINT, signal.SIG_DFL)  # restore default handler
            os.kill(os.getpid(), signal.SIGINT)            # re-send signal at OS level

        signal.signal(signal.SIGINT, _sigint_handler)

        dl_find(
                nvarin=nvarin, nvarin2=nvarin2, nspec=self.nspec,
                dlf_get_gradient=self.dlf_get_gradient, 
                dlf_get_params=self.dlf_get_params,
                dlf_put_coords=self.dlf_put_coords,
                dlf_get_hessian=self.dlf_get_hessian)

        # Regular optimization
        if self.icoord < 100:

            print(f"\nDL-FIND optimization finished in {self.dlfind_opt_cycles} steps!")
            print("Number of DL-FIND energy-gradient evaluations:", self.dlfind_eg_calls)

            # Print results
            finalenergy=self.traj_energies[-1]
            print("Final optimized energy:",  finalenergy)

            # Final coordinate handling
            if self.PBC:
                self.fragment.print_system(filename='Fragment-optimized.ygg')
                self.fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
                self.fragment.set_energy(finalenergy)
                print("Final geometry")
                self.fragment.print_coords()
                print("PBC True. Writing final optimized geometry in PBC-format")
                print("PBC_format_option:", self.PBC_format_option)
                if self.PBC_format_option.upper() == "CIF":
                    convert_to_pbcfile=write_CIF_file
                    file_ext='cif'
                elif self.PBC_format_option.upper() == "XSF":
                    convert_to_pbcfile=write_XSF_file
                    file_ext='xsf'
                elif self.PBC_format_option.upper() == "POSCAR":
                    convert_to_pbcfile=write_POSCAR_file
                    file_ext='POSCAR'
                pbcfile = convert_to_pbcfile(self.fragment.coords,self.fragment.elems,cellvectors=theory.periodic_cell_vectors,
                                             filename=f"Fragment-optimized.{file_ext}")
                print(f"Final cell vectors (Å):{theory.periodic_cell_vectors}")
                print(f"Final cell parameters: {cell_vectors_to_params(theory.periodic_cell_vectors)}")
                print(f"Final cell volume (Å): {cell_volume(theory.periodic_cell_vectors)}")
            else:
                # Writing out fragment file and XYZ file
                self.fragment.print_system(filename='Fragment-optimized.ygg')
                self.fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
                self.fragment.set_energy(finalenergy)
                print("Final geometry")
                self.fragment.print_coords()

            # Printing internal coordinate table
            if self.printlevel >= 2:
                print_internal_coordinate_table_new(self.fragment,actatoms=self.print_atoms_list)
            print()

            # Results object
            result = ASH_Results(label="DLFIND_optimizer", energy=finalenergy)

            if self.result_write_to_disk is True:
                result.write_to_disk(filename="DLFIND_optimizer.result", printlevel=self.printlevel)

        elif self.icoord >= 100 and self.icoord < 150:
            # NEB job complete
            print(f"\nDL-FIND NEB job finished in {self.dlfind_neb_cycles} steps!")
            print("Number of DL-FIND energy-gradient evaluations:", self.dlfind_eg_calls)

            CI_fragment_energy=None #TODO
            CI_fragment_coords=None #TODO

            # Now returning final Results object
            #result = ASH_Results(label="DLFIND_optimizer", energy=finalenergy)
            result = ASH_Results(label="DLFIND_NEB-CI calc", energy=CI_fragment_energy, geometry=CI_fragment_coords,
                charge=self.charge, mult=self.mult, MEP_energies_dict=self.NEB_energies_dict,
                barrier_energy=None)

            if self.result_write_to_disk is True:
                result.write_to_disk(filename="DLFIND_NEB.result")

        elif self.icoord >= 200:
            # Dimer
            print(f"\nDL-FIND Dimer job finished in {self.dlfind_dimer_cycles} steps!")
            print("Number of DL-FIND energy-gradient evaluations:", self.dlfind_eg_calls)

            finalenergy=self.traj_energies[-1]
            print("Final optimized energy:",  finalenergy)

            # Final coordinate handling
            final_coords=self.current_geo
            self.fragment.replace_coords(self.fragment.elems,final_coords, conn=False)
            # Writing out fragment file and XYZ file
            self.fragment.print_system(filename='Fragment-optimized.ygg')
            self.fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
            self.fragment.set_energy(finalenergy)

            # Printing internal coordinate table
            if self.printlevel >= 2:
                print_internal_coordinate_table_new(self.fragment,actatoms=self.print_atoms_list)
            print()

            # Results object
            result = ASH_Results(label="DLFIND_dimer", energy=finalenergy)

            if self.result_write_to_disk is True:
                result.write_to_disk(filename="DLFIND_dimer.result", printlevel=self.printlevel)


        return result


# Helper function to define residues
def define_residues(fragment=None, min_size=5, max_size=15):
    _COVALENT_RADII = {
    'H': 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
    'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
    'Mn': 1.61, 'Fe': 1.52, 'Co': 1.50, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
    'Hf': 1.75, 'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41,
    'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48,
    }
    elems=fragment.elems
    coords=fragment.coords
    num_atoms = len(elems)
    coords = np.array(coords)

    # 1. Build Connectivity (Adjacency List)
    adj = [[] for _ in range(num_atoms)]
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            # Threshold: sum of radii + 0.45A tolerance
            threshold = _COVALENT_RADII.get(elems[i], 0.7) + \
                        _COVALENT_RADII.get(elems[j], 0.7) + 0.45
            if dist < threshold:
                adj[i].append(j)
                adj[j].append(i)

    # 2. Split into Residues using Greedy BFS
    unvisited = set(range(num_atoms))
    residues = []

    while unvisited:
        # Start a new residue from an arbitrary unvisited atom
        root = min(unvisited)
        current_res = []
        queue = [root]

        while queue and len(current_res) < max_size:
            node = queue.pop(0)
            if node in unvisited:
                unvisited.remove(node)
                current_res.append(node)
                # Add neighbors to the queue to keep the residue contiguous
                for neighbor in adj[node]:
                    if neighbor in unvisited:
                        queue.append(neighbor)

        # Cleanup: If a residue is too small, merge it with the last one
        if len(current_res) < min_size and residues:
            residues[-1].extend(current_res)
        else:
            residues.append(current_res)

    return residues