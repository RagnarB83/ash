
import time
import numpy as np
import os
import copy
from collections import defaultdict

from ash.functions.functions_general import BC, ashexit, print_time_rel,print_line_with_mainheader,listdiff
from ash.modules.module_theory import Theory
from ash.interfaces.interface_ORCA import grabatomcharges_ORCA
from ash.interfaces.interface_xtb import grabatomcharges_xTB,grabatomcharges_xTB_output


#TODO: deal with GBW file and ORCA autostart mismatching for different regions and theory-levels
# Keep separate GBW files for each region and theory-level

class ONIOMTheory(Theory):
    def __init__(self, theory1=None, theory2=None, theories_N=None, regions_N=None, regions_chargemult=None,
                 embedding="mechanical", full_pointcharges=None, chargemodel="CM5",
                 fullregion_charge=None, fullregion_mult=None, fragment=None, label=None,
                 printlevel=2, numcores=1,):
        super().__init__()
        self.theorytype="ONIOM"
        self.theory1=theory1
        self.theory2=theory2
        self.printlevel=printlevel
        self.label=label
        self.filename=""
        self.theorynamelabel="ONIOMTheory"
        print_line_with_mainheader("ONIOM Theory")
        print("A N-layer ONIOM module")

        # Early exits
        # If fragment object has not been defined
        if fragment is None:
            print("Error: fragment= keyword has not been defined for ONIOM. Exiting")
            ashexit()
        if fullregion_charge is None or fullregion_mult is None:
            print("Error: Full-region charge and multiplicity must be provided (fullregion_charge, fullregion_mult keywords)")
            ashexit()

        if type(theories_N) != list:
            print("Error: theories_N should be a list")
            ashexit()
        print(f"{len(theories_N)} theories provided. This is a {len(theories_N)}-layer ONIOM.")
        if regions_N is None:
            print("Error: regions_N must be provided for N-layer ONIOM")
            ashexit()
        if regions_chargemult is None:
            print("Error: regions_chargemult must be provided for N-layer ONIOM (list of lists of charge,mult for each region)")
            ashexit()
        if len(theories_N) != len(regions_N):
            print("Error: Number of theories and regions must match")
            ashexit()
        if len(theories_N) != len(regions_chargemult):
            print("Error: Number of theories and regions_chargemult must match")
            ashexit()
        # Full system
        self.fragment=fragment
        self.allatoms = self.fragment.allatoms
        self.theories_N=theories_N
        self.regions_N=regions_N
        self.regions_chargemult=regions_chargemult # List of list of charge,mult combos

        # Linkatoms False by default. Later checked.
        self.linkatoms = False

        # Embedding
        # Note: by default no embedding, meaning LL theory for everything
        self.embedding=embedding
        self.chargemodel=chargemodel
        # Defining pointcharges for full system
        self.full_pointcharges=full_pointcharges

        # N-layer ONIOM
        self.fullregion_charge=fullregion_charge
        self.fullregion_mult=fullregion_mult

        # Defining charge/mult here as well (ASH jobtypes stop otherwise)
        self.charge = self.fullregion_charge
        self.mult = self.fullregion_mult

        #
        print("Embedding:", self.embedding)
        print("Theories:")
        for i,t in enumerate(self.theories_N):
            print(f"Theory {i+1}:", t.theorynamelabel)
        print("\nRegions provided:")
        #
        for i,r in enumerate(self.regions_N):
            print(f"Region {i+1} ({len(r)} atoms):", r)
        print("Allatoms:", self.allatoms)
        print("\nRegion-chargemult info provided:")
        #
        for i,r in enumerate(self.regions_chargemult):
            print(f"Region {i+1} Charge:{r[0]} Mult:{r[1]}")

        # REGIONS AND BOUNDARY
        conn_scale=1.0
        conn_tolerance=0.2
        if len(self.theories_N) == 2:
            self.theorylabels = ["HL","LL"]
            # Atom labels
            atomlabels = ["HL" if b in self.regions_N[0] else "LL" for b in self.fragment.allatoms]
            # If HL-LL covalent boundary issue and ASH exits then printing QM-coordinates is useful
            print("\nFull-system coordinates (before any linkatoms):")
            ash.modules.module_coords.print_coords_for_atoms(self.fragment.coords, self.fragment.elems, self.fragment.allatoms, labels=atomlabels)
            print()
            self.boundaryatoms = ash.modules.module_coords.get_boundary_atoms(self.regions_N[0], self.fragment.coords, self.fragment.elems, conn_scale,
                conn_tolerance, excludeboundaryatomlist=None, unusualboundary=None)

        else:
            print("todo")
            self.regionlabels = ["HL","IL", "LL"]
            ashexit()

        if len(self.boundaryatoms) > 0:
            print("Found covalent ONIOM boundary. Linkatoms option set to True")
            print("Boundaryatoms (HL:LL pairs):", self.boundaryatoms)
            print("Note: used connectivity settings, scale={} and tol={} to determine boundary.".format(conn_scale,conn_tolerance))
            self.linkatoms = True
            # Get MM boundary information. Stored as self.MMboundarydict
            self.get_MMboundary(conn_scale,conn_tolerance)
        else:
            print("No covalent ONIOM boundary. Linkatoms and dipole_correction options set to False")
            self.linkatoms=False
            self.dipole_correction=False

        if len(self.theories_N) > 3:
            print("Error: N>3 layer ONIOM is not yet supported")
            ashexit()

    def create_linkatoms(self, current_coords,region_atoms, region_elems):
        checkpoint=time.time()
        # Get linkatom coordinates
        self.linkatoms_dict = ash.modules.module_coords.get_linkatom_positions(self.boundaryatoms,region_atoms, current_coords, region_elems)
        print("linkatoms_dict:", self.linkatoms_dict)
        if self.printlevel > 1:
            print("Adding linkatom positions to region coords")
        self.linkatom_indices = [len(region_atoms)+i for i in range(0,len(self.linkatoms_dict))]
        self.num_linkatoms = len(self.linkatom_indices)
        linkatoms_coords = [self.linkatoms_dict[pair] for pair in sorted(self.linkatoms_dict.keys())]

        print_time_rel(checkpoint, modulename='create_linkatoms', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
        return linkatoms_coords

    def ShiftMMCharges(self):
        if self.printlevel > 1:
            print("new. Shifting MM charges at ONIOM boundary.")
        # Convert lists to NumPy arrays for faster computations
        self.pointcharges = np.array(self.charges_qmregionzeroed)
        self.charges=np.array(self.charges)

        # Extract charges for MM boundary atoms
        MM1_charges = self.charges[self.MMboundary_indices]
        # Set charges of MM boundary atoms to 0
        self.pointcharges[self.MMboundary_indices] = 0.0

        # Calculate charge fractions to distribute
        MM1charge_fract = MM1_charges / self.MMboundary_counts

        # Distribute charge fractions to neighboring MM atoms
        for indices, fract in zip(self.MMboundarydict.values(), MM1charge_fract):
            self.pointcharges[[indices]] += fract

        self.chargeshifting_done=True
        return

    # From QM1:MM1 boundary dict, get MM1:MMx boundary dict (atoms connected to MM1)
    def get_MMboundary(self,scale,tol):
        timeA=time.time()
        # if boundarydict is not empty we need to zero MM1 charge and distribute charge from MM1 atom to MM2,MM3,MM4
        #Creating dictionary for each MM1 atom and its connected atoms: MM2-4
        self.MMboundarydict={}
        for (QM1atom,MM1atom) in self.boundaryatoms.items():
            connatoms = ash.modules.module_coords.get_connected_atoms(self.fragment.coords, self.fragment.elems, scale,tol, MM1atom)
            #Deleting QM-atom from connatoms list
            connatoms.remove(QM1atom)
            self.MMboundarydict[MM1atom] = connatoms

        # Used by ShiftMMCharges
        self.MMboundary_indices = list(self.MMboundarydict.keys())
        self.MMboundary_counts = np.array([len(self.MMboundarydict[i]) for i in self.MMboundary_indices])

        print("")
        print("MM boundary (MM1:MMx pairs):", self.MMboundarydict)
        print_time_rel(timeA, modulename="get_MMboundary")

    def run(self, current_coords=None, Grad=False, elems=None, charge=None, mult=None, label=None, numcores=None):

        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Charge/mult. Note: ignoring charge/mult from run keyword.
        # Charge/mult definitions for full system and regions must have been provided on object creation

        if numcores is None:
            numcores = self.numcores

        # Full coordinates
        full_coords=current_coords
        full_elems=elems

        # Dicts to keep energy and gradient for each theory-region combo
        E_dict={} # (theory,region) -> energy
        G_dict={} # (theory,region) -> gradient
        num_theories = len(self.theories_N)

        # First doing LowLevel (LL) theory on Full region
        ll_theory = self.theories_N[-1]
        print(f"Running Theory LL ({ll_theory.theorynamelabel}) on Full-region ({len(full_elems)} atoms)")

        # Derive pointcharges unless full_pointcharges were already provided
        if self.full_pointcharges is None and self.embedding.lower() == "elstat":
            print("Electrostatic embedding but no full-system pointcharges provided yet")
            print("This means that we must derive pointcharges for full system")
            # TODO: How to do this in general
            # Check if the low-level theory is compatible with some charge model
            # Should probably do this in init instead though
            if isinstance(ll_theory, ash.ORCATheory):
                print(f"Theory is ORCATheory. Using {self.chargemodel} charge model")
                ll_theory.extraline+="\n! hirshfeld "
            elif isinstance(ll_theory, ash.xTBTheory):
                 print(f"Theory is xTBTheory. Using default xtb charge model")
            else:
                print("Problem: Theory-level not compatible with pointcharge-creation")
                ashexit()

        ###############################################
        # RUN FULL REGION
        ###############################################
        #Copying theory object for full-region to avoid interference with other regions
        ll_theory_full = copy.deepcopy(ll_theory)
        #Changing filename here for GBW-file creation
        label="LL_full"
        ll_theory_full.filename = f"{label}"
        if Grad:
            e_LL_full, g_LL_full = ll_theory_full.run(current_coords=full_coords,
                                                    elems=full_elems, Grad=Grad, numcores=numcores,
                                                    label=label, charge=self.fullregion_charge, mult=self.fullregion_mult)
        else:
            e_LL_full = ll_theory_full.run(current_coords=full_coords,
                                    elems=full_elems, Grad=Grad, numcores=numcores,
                                    label=label, charge=self.fullregion_charge, mult=self.fullregion_mult)

        # Grabbing atom charges from ORCA output
        # TODO: Remove theory-specific code
        if self.full_pointcharges is None and self.embedding == "Elstat":
            print("Grabbing atom charges for whole system")
            if isinstance(ll_theory_full, ash.ORCATheory):
                self.full_pointcharges = grabatomcharges_ORCA(self.chargemodel,f"{ll_theory_full.filename}.out")
            elif isinstance(ll_theory_full, ash.xTBTheory):
                print(f"{ll_theory_full.filename}.out")
                #Note: format issue
                #self.full_pointcharges = grabatomcharges_xTB_output(ll_theory.filename+'.out', chargemodel=self.chargemodel)
                self.full_pointcharges = grabatomcharges_xTB()
                #Peptide example using charges as ORCA does it
                #self.full_pointcharges = [-0.14285028,  0.06521348,  0.05413019,  0.05573540,  0.27118983, -0.46028381, -0.18504079,  0.21696679, -0.04008202,  0.07878930,  0.04626556,  0.03996635, -0.14694734,  0.04354141,  0.07669187,  0.07252365,  0.27177521, -0.45820150, -0.16872126,  0.16697404, -0.04781068,  0.08857382,  0.05067241,  0.05092835]
            print("self.full_pointcharges:", self.full_pointcharges)
            print(len(self.full_pointcharges))

        E_dict[(num_theories-1,-1)] = e_LL_full
        if Grad:
            G_dict[(num_theories-1,-1)] = g_LL_full

        # LOOPING OVER OTHER THEORY-REGION COMBOS
        for j,region in enumerate(self.regions_N):
            # Skipping last region
            if j == len(self.regions_N)-1:
                print("Last region always skipped")
                continue
            for i,theory in enumerate(self.theories_N):
                # Skipping HL on everything but first region
                if i == 0 and j>0:
                    print("HL theory on non-first region. Skipping")
                    continue
                # 3-layer ONIOM case (LL theory on first region). Skipping
                if i == 2 and j == 0:
                    print("Case 3-layer ONIOM: LL theory on first region. Skipping")
                    continue

                # Taking coordinates for region
                r_coords = np.take(current_coords,region,axis=0)
                r_elems = [elems[x] for x in region]

                #############
                # Linkatoms
                #############
                if self.linkatoms:
                    linkatoms_coords = self.create_linkatoms(current_coords, region, r_elems)

                    # Add linkatom coordinates to region-coordinates
                    region_coords_final = np.append(r_coords, np.array(linkatoms_coords), axis=0)

                    if self.printlevel > 1:
                        print(f"There are {self.num_linkatoms} linkatoms")
                    region_elems_final = r_elems + ['H']*self.num_linkatoms
                else:
                    region_coords_final = r_coords
                    region_elems_final = r_elems
                #############

                # Activate embedding for HL and LL theories in non-Full regions
                if self.embedding == "Elstat":
                    print("Region:", region)
                    print("Embedding activated")
                    PC = True
                    # Which region?
                    PCregion=listdiff(self.allatoms,region)
                    print("PCregion:", PCregion)

                    if self.printlevel > 1:
                        print("Doing charge-shifting...")

                    # Do Charge-shifting. MM1 charge distributed to MM2 atoms
                    self.ShiftMMCharges() # Creates self.pointcharges


                    pointchargecoords = np.take(current_coords,PCregion,axis=0)
                    # Pointcharges
                    if self.full_pointcharges is None:
                        print("Warning: Pointcharges for full system not available")
                        ashexit()
                    pointcharges=[self.full_pointcharges[x] for x in PCregion]



                    #TODO: dipole correction

                else:
                    PC=False
                    pointchargecoords=None
                    pointcharges=None

                # Running
                #Changing filename here for GBW-file creation
                label=f"{self.theorylabels[i]}_region{j+1}"
                theory.filename = f"{label}"
                print(f"Running Theory {i+1} ({theory.theorynamelabel}) on Region {j+1} ({len(region_elems_final)} atoms)")
                res = theory.run(current_coords=region_coords_final, elems=region_elems_final, Grad=Grad, numcores=numcores,
                                                PC=PC, current_MM_coords=pointchargecoords, MMcharges=pointcharges,
                                                label=label, charge=self.regions_chargemult[j][0], mult=self.regions_chargemult[j][1])
                if PC and Grad:
                    e,g,pg = res
                elif not PC and Grad:
                    e,g = res
                elif not PC and not Grad:
                    e = res
                # Keeping E and G in dicts
                E_dict[(i,j)] = e
                if Grad:
                    G_dict[(i,j)] = g

        # COMBINING ENERGY AND GRADIENTS
        # 2-layer ONIOM Energy and Gradient expression
        if len(self.theories_N) == 2:
            # E_dict: {(1, -1): -14.73707412056, (0, 0): -115.486915570077, (1, 0): -8.9611505694}
            self.energy = E_dict[(1,-1)] + E_dict[(0,0)] - E_dict[(1,0)]
            if Grad:
                # Gradient assembled
                self.gradient = G_dict[(1,-1)]
                for at, g in zip(self.regions_N[0], G_dict[(0,0)]):
                    self.gradient[at] += g
                for at, g in zip(self.regions_N[0], G_dict[(1,0)]):
                    self.gradient[at] -= g

        # 3-layer ONIOM Energy and Gradient expression
        elif len(self.theories_N) == 3:
            self.energy = E_dict[(2,-1)] + E_dict[(0,0)] - E_dict[(1,0)] + E_dict[(1,1)] - E_dict[(2,1)]
            # E(High,Real) = E(Low,Real) + [E(High,Model) - E(Medium,Model)]
            #                +  [E(Medium,Inter) - E(Low,Inter)].
            if Grad:
                # Gradient assembled
                self.gradient = G_dict[(2,-1)]
                for at, g in zip(self.regions_N[0], G_dict[(0,0)]):
                    self.gradient[at] += g
                for at, g in zip(self.regions_N[0], G_dict[(1,0)]):
                    self.gradient[at] -= g
                for at, g in zip(self.regions_N[1], G_dict[(1,1)]):
                    self.gradient[at] += g
                for at, g in zip(self.regions_N[1], G_dict[(2,1)]):
                    self.gradient[at] -= g
        # 4-layer ONIOM
        elif len(self.theories_N) == 4:
            print("4-layer ONIOM not yet available")
            ashexit()

        print("ONIOM energy:", self.energy)

        if Grad:
            return self.energy,self.gradient
        else:
            return self.energy
