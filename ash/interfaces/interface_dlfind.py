from __future__ import annotations
from ctypes import c_double, c_int, pointer
import functools
from typing import Callable, Optional
import numpy as np
from numpy.ctypeslib import as_array
from numpy.typing import ArrayLike

import os
import time
from ash.functions.functions_general import ashexit, blankline,BC,print_time_rel,print_line_with_mainheader,listdiff,search_list_of_lists_for_index
from ash.modules.module_coords import check_charge_mult, fullindex_to_actindex,print_internal_coordinate_table,write_xyzfile,elemstonuccharges
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
                     numfreq_force_projection=None, print_atoms_list=None):
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
                                    print_atoms_list=print_atoms_list)

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
                 numfreq_hessatoms=None, print_atoms_list=None):

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
        if theory is None or fragment is None:
            print("DLFIND_optimizer requires theory and fragment objects provided. Exiting.")
            ashexit()

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
            print("Choosing icoord=120 (HDLC internal coordinates) and iopt=10 (P-RFO)")
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
        self.theory=theory

        nuccharges  = elemstonuccharges(self.fragment.elems)

        charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "DLFIND-optimizer", theory=theory)

        # Possible Fragment2 handling
        self.fragment2=fragment2
        if self.fragment2 is not None:
            print("Fragment2 provided. This only makes sense for NEB and dimer jobs")
            positions2 = self.fragment2.coords * 1.88972612546
            nframe=1
        else:
            positions2=None
            nframe=0

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

        #Connectivity ?


        ########################################
        # ACTIVE/FROZEN AND RESIDUE HANDLING
        ########################################
        if self.residues is None:
            print("No residues provided to optimizer. Creating a single residue for whole active system.")
        else:
            print("Residues provided to optimizer:", self.residues)
        # What to optimize etc.
        self.spec=[]
        if actatoms is not None:
            print("Actatoms provided:", actatoms)
            print("All atoms:", fragment.allatoms)
            for i in fragment.allatoms:
                if i in actatoms:
                    if self.residues is not None:
                        self.spec.append(search_list_of_lists_for_index(i,self.residues)+1)
                    else:
                        self.spec.append(1)
                else:
                    self.spec.append(-1)
        elif frozenatoms is not None:
            print("Frozenatoms provided:", frozenatoms)
            print("All atoms:", fragment.allatoms)
            for i in fragment.allatoms:
                if i in frozenatoms:
                    self.spec.append(-1)
                else:
                    if self.residues is not None:
                        self.spec.append(search_list_of_lists_for_index(i,self.residues)+1)
                    else:
                        self.spec.append(1)
        else:
            print("Case: no actatoms or frozenatoms provided. All atoms will be active.")
            print("All atoms:", fragment.allatoms)
            if self.residues is None:
                self.spec=[1 for i in list(range(fragment.numatoms))]
            else:
                print("Residues provided:", self.residues)
                for i in fragment.allatoms:
                    resid = search_list_of_lists_for_index(i,self.residues)
                    self.spec.append(resid+1)

        # Nuclear charges
        self.spec=self.spec + nuccharges

        # Constraints. should be dict: constraints={'bond':[[0,1]], 'angle':[[98,99,100]]}
        if self.constraints is not None:
            print("Constraints passed: ", constraints)
            self.numcons=0
            conlist=[]
            for k,v in constraints.items():
                if k == 'bond':
                    print("Found bond constraint between atoms:", v)
                    for x in v:
                        b = [1,x[0]+1,x[1]+1,0,0]
                        conlist += b
                        self.numcons+=1
                elif k == 'angle':
                    print("Found angle constraint between atoms:", v)
                    for x in v:
                        b = [2,x[0]+1,x[1]+1,x[2]+1,0]
                        conlist += b
                        self.numcons+=1
                elif k == 'dihedral':
                    print("Found dihedral constraint between atoms:", v)
                    for x in v:
                        b = [3,x[0]+1,x[1]+1,x[2]+1,x[3]+1]
                        conlist += b
                        self.numcons+=1
            print("DL-FIND constraints-list:", conlist)
            print("Number of constraints:", self.numcons)
            self.spec = self.spec + conlist
        else:
            print("No constraints present")
            self.numcons=0

        # Spec
        self.spec=self.spec+[1 for i in list(range(fragment.numatoms))] #?

        self.nspec=len(self.spec)


        # Print-atoms choice
        # If not specified then active-region or all-atoms
        if print_atoms_list is None:
            #Print-atoms list not specified. What to do:
            if actatoms is not None:
                #If QM/MM object then QM-region:
                if isinstance(theory,QMMMTheory):
                    print("Theory class: QMMMTheory")
                    print("Will by default print only QM-region in output (use print_atoms_list option to change)")
                    self.print_atoms_list=theory.qmatoms
                elif isinstance(theory,ONIOMTheory):
                    print("Theory class: ONIOMTheory")
                    print("Will by default print only Region1 in output (use print_atoms_list option to change)")
                    self.print_atoms_list=theory.regions_N[0]
                else:
                    # Print actatoms since using Active Region (can be too much)
                    self.print_atoms_list=self.actatoms
            else:
                #No act-region. Print all atoms
                self.print_atoms_list=fragment.allatoms

        self.result_write_to_disk=result_write_to_disk

        #Tracking DL-FIND cycles
        self.dlfind_eg_calls=0
        self.dlfind_opt_cycles=0
        self.dlfind_neb_cycles=0
        self.dlfind_dimer_cycles=0


        self.NEB_energies_dict={}
        self.NEB_geometries={}

        # Create function to calculate energies and gradients
        @dlf_get_gradient_wrapper
        def ash_e_g_func(coordinates, iimage, kiter, theory):
            self.dlfind_eg_calls+=1
            coordinates_ang = coordinates*0.5291772109303
            energy, gradient = theory.run(current_coords=coordinates_ang, elems=self.fragment.elems, charge=charge, mult=mult, Grad=True)

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
            if self.hessian_choice == "numfreq":
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
                print("="*70)
                print(f"DLFIND OPTIMIZATION CYCLE {self.dlfind_opt_cycles}")
                print("="*70)
                #Storing current coordinates
                #traj_coords.append(np.array(coordinates_ang))
                print("Writing regular-opt traj")
                write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_opt_traj", printlevel=2, writemode='a', title=f"Energy: {energy}")
                self.current_geo=coordinates_ang
            # Traj-writing for dimer
            elif self.icoord >= 200:
                print("Writing Dimer traj")
                if switch == 1:
                    # 1: actual geometry
                    write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_dimertraj_1", printlevel=2, writemode='a', title=f"Energy: {energy}")
                    self.current_geo=coordinates_ang
                elif switch == 2:
                    # Approximate:
                    self.dlfind_dimer_cycles+=1
                    # transition mode
                    write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_dimertraj_2", printlevel=2, writemode='a', title=f"Energy: {energy}")
                elif switch == 3:
                    write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_dimertraj_3", printlevel=2, writemode='a', title=f"Energy: {energy}")
            self.traj_energies.append(energy)

            return

        self.traj_energies = []
        self.current_geo = []
        positions = self.fragment.coords * 1.88972612546
        self.dlf_get_params = make_dlf_get_params(coords=positions, coords2=positions2, icoord=self.icoord, 
                                                  iopt=self.iopt, maxcycle=self.maxcycle,tolerance=self.tolerance,
                                                  tolerance_e=self.tolerance_e, inithessian=self.inithessian,
                                                  nframe=nframe, nz = self.fragment.numatoms,
                                                  ncons=self.numcons, delta=self.delta,
                                                  spec=self.spec, printl=self.printlevel, nimage=self.nimage)

        self.dlf_get_gradient = functools.partial(ash_e_g_func, theory=theory)
        self.dlf_get_hessian = functools.partial(hess_func)
        self.dlf_put_coords = functools.partial( store_results, None)

        # Delete old traj file before beginning
        remove_files=['DLFIND_opt_traj.xyz','DLFIND_dimertraj_1.xyz', 'DLFIND_dimertraj_2.xyz','DLFIND_dimertraj_3.xyz','DLFIND_NEBpath_current.xyz', 'DLFIND_NEBpath_all.xyz', 'DLFIND_CIgeo_traj.xyz']
        print("Removing possible old files:", remove_files)
        for rfile in remove_files:
            try:
                os.remove(rfile)
                print("removed ", rfile)
            except FileNotFoundError:
                #print(f"file {rfile} not found")
                pass

        print("\nArguments passed to DL-FIND:")
        print("icoord:", self.icoord)
        print("iopt:", self.iopt)
        print("maxcycle:", maxcycle)
        print("spec:", self.spec)
        if icoord == 120:
            print("NEB nimage:", nimage)

    def run(self, theory=None, fragment=None, charge=None, mult=None):

        from libdlfind import dl_find

        if self.fragment2 is None:
            nvarin=self.fragment.numatoms * 3
            nvarin2=0
        else:
            # Fragment 1 and 2
            nvarin = self.fragment.numatoms * 3
            nvarin2 = self.fragment2.numatoms * 3

        # Run DL-FIND
        print("Now starting DL-FIND")
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
            final_coords=self.current_geo
            fragment.replace_coords(fragment.elems,final_coords, conn=False)
            # Writing out fragment file and XYZ file
            fragment.print_system(filename='Fragment-optimized.ygg')
            fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
            fragment.set_energy(finalenergy)

            # Printing internal coordinate table
            if self.printlevel >= 2:
                print_internal_coordinate_table(fragment,actatoms=self.print_atoms_list)
            print()

            # Now returning final Results object
            result = ASH_Results(label="DLFIND_optimizer", energy=finalenergy)
            if self.result_write_to_disk is True:
                result.write_to_disk(filename="DLFIND_optimizer.result")
            return result

        elif self.icoord >= 100 and self.icoord < 150:
            # NEB job complete
            print(f"\nDL-FIND NEB job finished in {self.dlfind_neb_cycles} steps!")
            print("Number of DL-FIND energy-gradient evaluations:", self.dlfind_eg_calls)

            CI_fragment_energy=None #TODO
            CI_fragment_coords=None #TODO

            # Now returning final Results object
            #result = ASH_Results(label="DLFIND_optimizer", energy=finalenergy)
            result = ASH_Results(label="DLFIND_NEB-CI calc", energy=CI_fragment_energy, geometry=CI_fragment_coords,
                charge=charge, mult=mult, MEP_energies_dict=self.NEB_energies_dict,
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
            fragment.replace_coords(fragment.elems,final_coords, conn=False)
            # Writing out fragment file and XYZ file
            fragment.print_system(filename='Fragment-optimized.ygg')
            fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
            fragment.set_energy(finalenergy)

            # Printing internal coordinate table
            if self.printlevel >= 2:
                print_internal_coordinate_table(fragment,actatoms=self.print_atoms_list)
            print()

            # Now returning final Results object
            result = ASH_Results(label="DLFIND_optimizer", energy=finalenergy)
            if self.result_write_to_disk is True:
                result.write_to_disk(filename="DLFIND_optimizer.result")
            return result