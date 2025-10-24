
import time
import numpy as np
import os
import copy
from collections import defaultdict
import math

from ash.functions.functions_general import BC, ashexit, print_time_rel,print_line_with_mainheader,listdiff, printdebug
from ash.modules.module_theory import Theory
from ash.interfaces.interface_ORCA import grabatomcharges_ORCA, ORCATheory
from ash.interfaces.interface_xtb import grabatomcharges_xTB,grabatomcharges_xTB_output,xTBTheory
from ash.modules.module_QMMM import linkatom_force_adv, linkatom_force_lever,linkatom_force_chainrule
from ash.modules.module_coords import print_coords_for_atoms,get_boundary_atoms,get_linkatom_positions,distance,get_connected_atoms

#TODO: deal with GBW file and ORCA autostart mismatching for different regions and theory-levels
# Keep separate GBW files for each region and theory-level

class ONIOMTheory(Theory):
    def __init__(self, theories_N=None, regions_N=None, regions_chargemult=None,
                 embedding="mechanical", full_pointcharges=None, chargemodel="CM5", dipole_correction=False,
                 fullregion_charge=None, fullregion_mult=None, fragment=None, label=None, 
                 chargeboundary_method="chargeshift", excludeboundaryatomlist=None,
                 linkatom_method='ratio', linkatom_simple_distance=None, linkatom_forceproj_method="adv",
                 linkatom_ratio=0.723, linkatom_type='H', printlevel=2, numcores=1):
        super().__init__()
        self.theorytype="ONIOM"
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
            print("Error: theories_N should be defined and be a list")
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

        # Linkatom method strategy to determine linkatom position or QM-L distance
        self.linkatom_type=linkatom_type # Usually 'H'
        self.linkatom_method = linkatom_method # Options: 'simple' or 'ratio'
        self.linkatom_simple_distance = linkatom_simple_distance # For method simple, Default 1.09 Angstrom
        # For method ratio. see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9314059/
        self.linkatom_ratio=linkatom_ratio
        # Linkatom projection method Options: 'adv', 'lever', 'chain', 'none'
        self.linkatom_forceproj_method = linkatom_forceproj_method
        if self.linkatom_forceproj_method is None:
            self.linkatom_forceproj_method="none"

        # Embedding
        # Note: by default no embedding, meaning LL theory for everything
        self.embedding=embedding
        self.chargemodel=chargemodel
        self.chargeboundary_method=chargeboundary_method  # Options: 'chargeshift', 'rcd'

        # Defining pointcharges for full system
        self.full_pointcharges=full_pointcharges
        # Dipole correction for charge-shifting
        self.dipole_correction=dipole_correction

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
            print(f"Theory {i+1}: {t.theorynamelabel} . Numcores: {t.numcores}")
        # Setting numcores of ONIOMTheory object.

        # numcores-setting in ONIOMTheory takes precedent
        if numcores != 1:
            print(f"Numcores keyword (numcores={numcores}) was set for ONIOMTheory.")
            self.numcores=numcores
            print(f"Warning: ONIOM will use {numcores} cores in general for all Theories (overriding numcores in Theory objects)")
            for i,t in enumerate(self.theories_N):
                t.numcores=numcores
                print(f"Warning: Setting numcores={numcores} for Theory {i+1}: {t.theorynamelabel}")
        else:
            print("Warning: numcores attribute was not set for ONIOMTheory.")
            print("This is fine, but check if numcores settings above are appropriate for each Theory object")
            self.numcores=1

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
            print_coords_for_atoms(self.fragment.coords, self.fragment.elems, self.fragment.allatoms, labels=atomlabels)
            print()
            self.boundaryatoms = get_boundary_atoms(self.regions_N[0], self.fragment.coords, self.fragment.elems, conn_scale,
                conn_tolerance, excludeboundaryatomlist=excludeboundaryatomlist, unusualboundary=None)
        elif len(self.theories_N) == 3:
            self.theorylabels = ["HL","IL", "LL"]
            # Atom labels
            atomlabels = ["HL" if b in self.regions_N[0] else "ML" if b in self.regions_N[1] else "LL" for b in self.fragment.allatoms ]
            print("atomlabels:", atomlabels)
            # If HL-LL covalent boundary issue and ASH exits then printing QM-coordinates is useful
            print("\nFull-system coordinates (before any linkatoms):")
            print_coords_for_atoms(self.fragment.coords, self.fragment.elems, self.fragment.allatoms, labels=atomlabels)
            #
            self.boundaryatoms_HL_ML = get_boundary_atoms(self.regions_N[0], self.fragment.coords, self.fragment.elems, conn_scale,
                conn_tolerance, excludeboundaryatomlist=excludeboundaryatomlist, unusualboundary=None)
            print("boundaryatoms_HL_ML:", self.boundaryatoms_HL_ML)
            self.boundaryatoms_ML_LL = get_boundary_atoms(self.regions_N[1], self.fragment.coords, self.fragment.elems, conn_scale,
                conn_tolerance, excludeboundaryatomlist=excludeboundaryatomlist, unusualboundary=None)
            print("boundaryatoms_ML_LL:", self.boundaryatoms_ML_LL)
            print("XX")
            #self.boundaryatoms=[]

        elif len(self.theories_N) == 4:
            print("4-layer ONION is not supported yet")
            ashexit()
        else:
            print("Weird unsupported theory specifaction:", self.theories_N)
            ashexit()

        # Checking for covalent boundary
        if len(self.theories_N) == 2 and len(self.boundaryatoms) > 0:
            print("Found covalent ONIOM boundary. Linkatoms option set to True")
            print("Boundaryatoms (HL:LL pairs):", self.boundaryatoms)
            print("Note: used connectivity settings, scale={} and tol={} to determine boundary.".format(conn_scale,conn_tolerance))
            self.linkatoms = True
            # Get MM boundary information. Stored as self.MMboundarydict
            self.get_MMboundary(self.boundaryatoms,conn_scale,conn_tolerance)
        elif len(self.theories_N) == 3 and len(self.boundaryatoms_HL_ML) > 0:
            print("Found covalent ONIOM boundary between HL and ML regions. Linkatoms option set to True")
            print("Boundaryatoms (HL:LL pairs):", self.boundaryatoms_HL_ML)
            print("Note: used connectivity settings, scale={} and tol={} to determine boundary.".format(conn_scale,conn_tolerance))
            self.linkatoms = True
            # Get MM boundary information. Stored as self.MMboundarydict
            self.get_MMboundary(self.boundaryatoms_HL_ML,conn_scale,conn_tolerance)
        elif len(self.theories_N) == 3 and len(self.boundaryatoms_ML_LL) > 0:
            print("Found covalent ONIOM boundary between ML and LL regions. Linkatoms option set to True")
            print("Boundaryatoms (HL:LL pairs):", self.boundaryatoms_ML_LL)
            print("Note: used connectivity settings, scale={} and tol={} to determine boundary.".format(conn_scale,conn_tolerance))
            self.linkatoms = True
            # Get MM boundary information. Stored as self.MMboundarydict
            self.get_MMboundary(self.boundaryatoms_ML_LL,conn_scale,conn_tolerance)
        else:
            print("No covalent ONIOM boundary. Linkatoms and dipole_correction options set to False")
            self.linkatoms=False
            self.dipole_correction=False
            self.MMboundary_indices=[]

        if len(self.theories_N) > 3:
            print("Error: N>3 layer ONIOM is not yet supported")
            ashexit()

    def create_linkatoms(self, current_coords,region_atoms, elems):
        checkpoint=time.time()
        # Get linkatom coordinates
        # NOTE: Option to change linkatom_distance, now 1.08736
        self.linkatoms_dict = get_linkatom_positions(self.boundaryatoms,region_atoms, 
                                                                               current_coords, elems, linkatom_type=self.linkatom_type,
                                                                               linkatom_method=self.linkatom_method,
                                                                               linkatom_simple_distance=self.linkatom_simple_distance,
                                                                               linkatom_ratio=self.linkatom_ratio)
        print("linkatoms_dict:", self.linkatoms_dict)
        if self.printlevel > 1:
            print("Adding linkatom positions to region coords")
        self.linkatom_indices = [len(region_atoms)+i for i in range(0,len(self.linkatoms_dict))]
        self.num_linkatoms = len(self.linkatom_indices)
        linkatoms_coords = [self.linkatoms_dict[pair] for pair in sorted(self.linkatoms_dict.keys())]

        print_time_rel(checkpoint, modulename='create_linkatoms', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
        return linkatoms_coords

    def ZeroQMCharges(self, atoms):
        print("Setting Region charges to Zero")
        # Looping over charges and setting region atoms to zero
        # 1. Copy charges to charges_qmregionzeroed
        charges_qmregionzeroed=copy.copy(self.charges)
        # 2. change charge for QM-atom
        for i, c in enumerate(charges_qmregionzeroed):
            # Setting QMatom charge to 0
            if i in atoms:
                charges_qmregionzeroed[i] = 0.0
        # 3. Flag that this has been done
        self.ChargesZeroed = True

        return charges_qmregionzeroed

    # Create dipole charge (twice) for each MM2 atom that gets fraction of MM1 charge
    def get_dipole_charge(self,delq,direction,mm1index,mm2index,current_coords, dipole_shift=0.15):

        # Coordinates and distance
        mm1coords=np.array(current_coords[mm1index])
        mm2coords=np.array(current_coords[mm2index])
        MM_distance = distance(mm1coords,mm2coords) # Distance between MM1 and MM2

        # Normalize vector
        def vnorm(p1):
            r = math.sqrt((p1[0]*p1[0])+(p1[1]*p1[1])+(p1[2]*p1[2]))
            v1=np.array([p1[0] / r, p1[1] / r, p1[2] /r])
            return v1
        diffvector = mm2coords-mm1coords
        normdiffvector = vnorm(diffvector)

        # Dipole
        d = delq*2.5
        # Charge (abs value)
        q0 = 0.5 * d / dipole_shift
        # Actual shift
        shift = direction * dipole_shift * ( MM_distance / 2.5 )
        # Position
        pos = mm2coords+np.array((shift*normdiffvector))
        # Returning charge with sign based on direction and position
        # Return coords as regular list
        return -q0*direction,list(pos)

    def SetDipoleCharges(self,current_coords, dipole_model="v2"):
        checkpoint=time.time()
        if self.printlevel > 1:
            print("Adding extra charges to preserve dipole moment for charge-shifting")
            print("MMboundarydict:", self.MMboundarydict)
        # Adding 2 dipole pointcharges for each MM2 atom
        dipole_charges = []
        dipole_coords = []

        for MM1,MMx in self.MMboundarydict.items():
            # Getting original MM1 charge (before set to 0)
            MM1charge = self.charges[MM1]
            MM1charge_fract=MM1charge/len(MMx)

            sign=-1
            for MM in MMx:
                if dipole_model == "v1":
                    q_d1, pos_d1 = self.get_dipole_charge(MM1charge_fract,1,MM1,MM,current_coords, dipole_shift=0.15)
                    q_d2, pos_d2 = self.get_dipole_charge(MM1charge_fract,-1,MM1,MM,current_coords, dipole_shift=0.15)
                    dipole_charges.append(q_d1)
                    dipole_charges.append(q_d2)
                    dipole_coords.append(pos_d1)
                    dipole_coords.append(pos_d2)
                elif dipole_model == "v2":
                    # EXPERIMENTAL
                    MM1charge_fract=MM1charge/(len(MMx)+1)
                    q_d1, pos_d1 = self.get_dipole_charge(MM1charge_fract,1,MM1,MM,current_coords, dipole_shift=0.15)
                    q_d2, pos_d2 = self.get_dipole_charge(MM1charge_fract,-1,MM1,MM,current_coords, dipole_shift=0.15)
                    #q_d3, pos_d3 = self.get_dipole_charge(MM1charge_fract,0,MM1,MM,current_coords, dipole_shift=0.15)
                    dipole_charges.append(q_d2)
                    dipole_charges.append(q_d1)

                    dipole_coords.append(pos_d2)
                    dipole_coords.append(pos_d1)

                    # Add MM atom pos
                    dipole_charges.append(q_d2*sign)
                    pos_d3 = current_coords[MM]
                    dipole_coords.append(pos_d3)
                    sign = sign*-1

        print_time_rel(checkpoint, modulename='SetDipoleCharges', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
        return dipole_charges, dipole_coords

    # Standard chargeshifting, modifies pointcharges list (no additional charges)
    def ShiftMMCharges(self, charges_qmregionzeroed):
        if self.printlevel > 1:
            print("Shifting MM charges at ONIOM boundary.")
        # Convert lists to NumPy arrays for faster computations
        pointcharges = np.array(charges_qmregionzeroed)
        self.charges = np.array(self.charges)

        # Extract charges for MM boundary atoms
        MM1_charges = self.charges[self.MMboundary_indices]
        # Set charges of MM boundary atoms to 0
        pointcharges[self.MMboundary_indices] = 0.0

        # Calculate charge fractions to distribute
        MM1charge_fract = MM1_charges / self.MMboundary_counts

        # Charge-shifting method
        # Distribute charge fractions to neighboring MM atoms
        for indices, fract in zip(self.MMboundarydict.values(), MM1charge_fract):
            pointcharges[[indices]] += fract

        self.chargeshifting_done=True
        return pointcharges

    def RCD_shifting(self, charges_qmregionzeroed, pointchargecoords, fullcoords, PCregion):
        if self.printlevel > 1:
            print("Shifting MM charges at ONIOM boundary by RCD.")
        # Convert lists to NumPy arrays for faster computations
        pointcharges = np.array(charges_qmregionzeroed)
        self.charges = np.array(self.charges)
        # Extract charges for MM boundary atoms
        MM1_charges = self.charges[self.MMboundary_indices]
        # Set charges of MM boundary atoms to 0
        pointcharges[self.MMboundary_indices] = 0.0
        # Calculate charge fractions to distribute
        MM1charge_fract = MM1_charges / self.MMboundary_counts

        # Only keep pointcharges for PC region
        pointcharges=[pointcharges[x] for x in PCregion]

        # Distribute charge fractions to neighboring MM atoms
        for MM1index, MM2indices, fract in zip(self.MMboundarydict.keys(), self.MMboundarydict.values(), MM1charge_fract):
            newfract = fract*2 #q0*2
            # Looping over MM2 atoms
            for i in MM2indices:
                # RC/RCD: Instead of adding the M1 charge to the M2 atoms we create new RC/RCD sites wi
                pointcharges = np.append(pointcharges,newfract)
                # RCD: Reduce the MM2 charge by q0
                pointcharges[i] -= fract
                # Add new RCD sites to pointchargecoords and pointcharges
                newsite = (fullcoords[i] + fullcoords[MM1index])/2
                pointchargecoords = np.append(pointchargecoords, [newsite], axis=0)
            #print("RCD-modified pointcharges:", pointcharges)
            #print("RCD-modified pointchargecoords:", pointchargecoords)
        self.chargeshifting_done=True

        return pointcharges, pointchargecoords

    # From QM1:MM1 boundary dict, get MM1:MMx boundary dict (atoms connected to MM1)
    def get_MMboundary(self,boundaryatoms,scale,tol):
        timeA=time.time()
        # if boundarydict is not empty we need to zero MM1 charge and distribute charge from MM1 atom to MM2,MM3,MM4
        #Creating dictionary for each MM1 atom and its connected atoms: MM2-4
        self.MMboundarydict={}
        for (QM1atom,MM1atom) in boundaryatoms.items():
            connatoms = get_connected_atoms(self.fragment.coords, self.fragment.elems, scale,tol, MM1atom)
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

        module_init_time=time.time()
        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Charge/mult. Note: ignoring charge/mult from run keyword.
        # Charge/mult definitions for full system and regions must have been provided on object creation

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
            if isinstance(ll_theory, ORCATheory):
                print(f"Theory is ORCATheory. Using {self.chargemodel} charge model")
                if self.chargemodel == "CM5" or self.chargemodel.lower() == "hirshfeld":
                    ll_theory.extraline+="\n! hirshfeld "
                else:
                    print("Unknown charge model")
                    ashexit()
            elif isinstance(ll_theory, xTBTheory):
                 print(f"Theory is xTBTheory. Using default xtb charge model")
            else:
                print("Problem: Theory-level not compatible with pointcharge-creation")
                ashexit()

        ###############################################
        # RUN FULL REGION
        ###############################################
        # Copying theory object for full-region to avoid interference with other regions
        ll_theory_full = copy.deepcopy(ll_theory)
        # Changing filename here for GBW-file creation
        label = "LL_full"
        ll_theory_full.filename = f"{label}"
        # RUN FULL
        res_full = ll_theory_full.run(current_coords=full_coords,
                                    elems=full_elems, Grad=Grad, numcores=ll_theory.numcores,
                                    label=label, charge=self.fullregion_charge, mult=self.fullregion_mult)
        if Grad:
            e_LL_full,g_LL_full = res_full
        else:
            e_LL_full = res_full

        # Grabbing atom charges from ORCA output
        # TODO: Remove theory-specific code
        if self.embedding.lower() == "elstat" and self.full_pointcharges is None:
            print("Grabbing atom charges for whole system")
            print("Chargemodel:", self.chargemodel)
            if isinstance(ll_theory_full, ORCATheory):
                self.full_pointcharges = grabatomcharges_ORCA(self.chargemodel,f"{ll_theory_full.filename}.out")
                # Remove ll_theory.extraline from ORCATheory for next LL calculations
                ll_theory.extraline=""
            elif isinstance(ll_theory_full, xTBTheory):
                print(f"{ll_theory_full.filename}.out")
                # Note: format issue
                # self.full_pointcharges = grabatomcharges_xTB_output(ll_theory.filename+'.out', chargemodel=self.chargemodel)
                self.full_pointcharges = grabatomcharges_xTB()
            print("self.full_pointcharges:", self.full_pointcharges)
            print(len(self.full_pointcharges))

        if self.embedding.lower() == "elstat":
            # Defining charges for full system
            self.charges=self.full_pointcharges
            print("Num full system charges:", len(self.charges))
            print("Full system charges:", self.charges)

        E_dict[(num_theories-1,-1)] = e_LL_full
        if Grad:
            G_dict[(num_theories-1,-1)] = g_LL_full

        ###############################################
        # RUN OTHER REGIONS
        ###############################################
        
        # LOOPING OVER OTHER THEORY-REGION COMBOS
        for j,region in enumerate(self.regions_N):
            print("\nj:",j)
            print("region:", region )
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

                other_elems = listdiff(full_elems, r_elems)

                other_region = listdiff(self.allatoms, region)

                #############
                # Linkatoms
                #############
                if self.linkatoms:
                    linkatoms_coords = self.create_linkatoms(current_coords, region, elems)

                    # Add linkatom coordinates to region-coordinates
                    region_coords_final = np.append(r_coords, np.array(linkatoms_coords), axis=0)

                    if self.printlevel > 1:
                        print(f"There are {self.num_linkatoms} linkatoms")
                    region_elems_final = r_elems + [self.linkatom_type]*self.num_linkatoms
                else:
                    region_coords_final = r_coords
                    region_elems_final = r_elems
                #############

                # Activate embedding for HL and LL theories in non-Full regions
                if self.embedding.lower() == "elstat":
                    mm_elems_for_qmprogram = other_elems # Overwritten later, only matters for CP2K GEEP
                    # Pointcharges
                    if self.full_pointcharges is None:
                        print("Warning: Pointcharges for full system not available")
                        ashexit()

                    print("Region:", region)
                    print("Embedding activated")

                    PC = True
                    # Which region?
                    PCregion=listdiff(self.allatoms,region)
                    print("PCregion:", PCregion)

                    if self.printlevel > 1:
                        print("Doing charge-shifting...")

                    # Zero region charges
                    # This will set self.charges_qmregionzeroed
                    charges_qmregionzeroed = self.ZeroQMCharges(region)
                    print("Num charges_qmregionzeroedcharges:", len(charges_qmregionzeroed))
                    print("charges_qmregionzeroedcharges:", charges_qmregionzeroed)

                    pointchargecoords = np.take(current_coords,PCregion,axis=0)
                    print("Num pointchargecoords:", len(pointchargecoords))

                    if self.linkatoms is True:
                        print("Case: Electrostatic embedding with linkatoms")
                        #########################
                        # Do Charge-shifting
                        # #######################
                        # Sherwoord charge-shifting: MM1 charge distributed to MM2 atoms
                        if self.chargeboundary_method.lower() == "chargeshift":
                            print("Chargeshiftmethod is: (regular) chargeshift")
                            pointcharges = self.ShiftMMCharges(charges_qmregionzeroed) # Creates pointcharges
                            print("Num pointcharges after shift:", len(pointcharges))
                            print("Pointcharges after shift:", pointcharges)

                            # Only keep pointcharges for PC region
                            pointcharges=[pointcharges[x] for x in PCregion]

                            # DIPOLE CORRECTION (adds extra dipolecharges to pointcharges and pointchargecoords)
                            if self.dipole_correction is True:
                                print("Dipole correction is on. Adding dipole charges")
                                dipole_charges, dipole_coords = self.SetDipoleCharges(current_coords) # Creates

                                # Adding dipole charge coords to MM coords (given to QM code) and defining pointchargecoords
                                if self.printlevel > 1:
                                    print("Adding {} dipole charges to PC environment".format(len(dipole_charges)))

                                # Adding dipole charges to MM charges list (given to QM code)
                                pointcharges = list(pointcharges)+list(dipole_charges)
                                print("Num pointcharges after dipole addition", len(pointcharges))
                                print("pointcharges after dipole addition:", pointcharges)

                                pointchargecoords = np.append(pointchargecoords, np.array(dipole_coords), axis=0)
                                print("Num pointchargecoords after dipole addition", len(pointchargecoords))
                                print("pointchargecoords after dipole addition:", pointchargecoords)
                                # Using element H for dipole charges. Only matters for CP2K GEEP
                                print("other_elems:", other_elems)
                                mm_elems_for_qmprogram = other_elems + ['H']*len(dipole_charges)

                                if self.printlevel > 1: print("Number of pointcharges after dipole addition: ", len(pointcharges))

                            else:
                                print("Dipole correction is off. Not adding any dipole charges")
                                if self.printlevel > 1: print("Number of pointcharges: ", len(pointcharges))

                        elif self.chargeboundary_method.lower() == "rcd":
                            print("Chargeshiftmethod is: rcd")
                            pointcharges, pointchargecoords = self.RCD_shifting(charges_qmregionzeroed, pointchargecoords, current_coords, PCregion)
                    else:
                        print("Case: Electrostatic embedding but no linkatoms")
                        pointcharges = np.array(charges_qmregionzeroed)
                        pointcharges=[float(pointcharges[x]) for x in PCregion]

                    print("Region Num pointcharges after shift:", len(pointcharges))
                    print("Region Num pointchargecoords after shift:", len(pointchargecoords))
                    print("Region Pointcharges after shift:", pointcharges)

                    assert len(pointcharges) == len(pointchargecoords), f"Num pointcharges ({len(pointcharges)}) and pointchargecoords ({len(pointchargecoords)}) do not match"

                else:
                    PC=False
                    pointchargecoords=None
                    pointcharges=None
                    mm_elems_for_qmprogram = other_elems

                # Running
                # Changing filename here for GBW-file creation
                label=f"{self.theorylabels[i]}_region{j+1}"
                theory.filename = f"{label}"
                print(f"Running Theory {i+1} ({theory.theorynamelabel}) on Region {j+1} ({len(region_elems_final)} atoms)")

                # For an MM-theory like OpenMM we have to do some special handling
                if theory.theorytype == "MM":
                    print("Case: Theory is MM")
                    # Other region (i.e. not region1)
                    theory.update_charges(other_region,[0.0 for x in other_region])
                    theory.update_LJ_epsilons(other_region,[0.0 for x in other_region])
                    theory.modify_bonded_forces(other_region)
                    # NOTE: Fullsystem coordinates still passed here
                    res = theory.run(current_coords=full_coords, elems=full_elems, Grad=Grad, numcores=theory.numcores, label=label)
                # if the theory is QM/MM then this
                elif theory.theorytype == "QM/MM":
                    print("Case: Theory is QM/MM object")
                    print("Warning: A subtractive correction for a region where the theory is a QM/MM Theory does not make much sense")
                    print("Warning: Using only the QM-theory part of the QM/MM object to calculate the subtraction")
                    res = theory.qm_theory.run(current_coords=region_coords_final, elems=region_elems_final, Grad=Grad, numcores=theory.numcores,
                                                    PC=PC, current_MM_coords=pointchargecoords, MMcharges=pointcharges, mm_elems=mm_elems_for_qmprogram,
                                                    label=label, charge=self.regions_chargemult[j][0], mult=self.regions_chargemult[j][1])
                else:
                    res = theory.run(current_coords=region_coords_final, elems=region_elems_final, Grad=Grad, numcores=theory.numcores,
                                                    PC=PC, current_MM_coords=pointchargecoords, MMcharges=pointcharges, mm_elems=mm_elems_for_qmprogram,
                                                    label=label, charge=self.regions_chargemult[j][0], mult=self.regions_chargemult[j][1])
                if PC and Grad:
                    e,g,pg = res
                elif PC and not Grad:
                    e = res
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
            self.energy = E_dict[(1,-1)] + E_dict[(0,0)] - E_dict[(1,0)]
            print(f"Energy (Full-LL): {E_dict[(1,-1)]} Eh")
            print(f"Energy (Region1-HL): {E_dict[(0,0)]} Eh")
            print(f"Energy (Region1-LL): {E_dict[(1,0)]} Eh")
            if Grad:
                # Gradient assembled
                self.gradient = G_dict[(1,-1)]
                for at, g in zip(self.regions_N[0], G_dict[(0,0)]):
                    self.gradient[at] += g
                for at, g in zip(self.regions_N[0], G_dict[(1,0)]):
                    self.gradient[at] -= g

                # Linkatom force projection
                if self.linkatoms is True:
                    print("Linkatom force projection now")
                    print("Looping over linkatoms")
                    for i,linkatomindex in enumerate(self.linkatom_indices):
                        pair = sorted(self.linkatoms_dict.keys())[i]
                        Lcoord=self.linkatoms_dict[pair]
                        print("Lcoord:", Lcoord)
                        # Looping over theory-levels calculated
                        diffgrad=G_dict[(0,0)]-G_dict[(1,0)]
                        for theory_grad in [diffgrad]:
                            # for theory_grad in [G_dict[(0,0)], G_dict[(1,0)]]:
                            # Region gradient
                            Lgrad=theory_grad[linkatomindex]
                            print("Lgrad:", Lgrad)
                            # Getting QM1 info
                            fullatomindex_qm=pair[0]
                            regionatomindex=self.regions_N[0].index(fullatomindex_qm)
                            r_coords = np.take(current_coords,self.regions_N[0],axis=0)
                            Qcoord=r_coords[regionatomindex]
                            print("Qcoord:", Qcoord)
                            # Grabbing MMatom info
                            fullatomindex_mm=pair[1]
                            Mcoord=full_coords[fullatomindex_mm]
                            print("Mcoord:", Mcoord)
                            print("self.linkatom_forceproj_method:", self.linkatom_forceproj_method)
                            # Getting gradient contribution to QM1 and MM1 atoms from linkatom
                            if self.linkatom_forceproj_method == "adv":
                                QM1grad_contrib, MM1grad_contrib = linkatom_force_adv(Qcoord, Mcoord, Lcoord, Lgrad)
                            elif self.linkatom_forceproj_method == "lever": 
                                QM1grad_contrib, MM1grad_contrib = linkatom_force_lever(Qcoord, Mcoord, Lcoord, Lgrad)
                            elif self.linkatom_forceproj_method == "chain":
                                QM1grad_contrib, MM1grad_contrib = linkatom_force_chainrule(Qcoord, Mcoord, Lcoord, Lgrad)
                            elif self.linkatom_forceproj_method.lower() == "none" or self.linkatom_forceproj_method == None:
                                QM1grad_contrib = np.zeros(3)
                                MM1grad_contrib = np.zeros(3)
                            else:
                                print("Unknown linkatom_forceproj_method. Exiting")
                                ashexit()
                            print("QM1grad_contr:", QM1grad_contrib)
                            print("MM1grad_contr:", MM1grad_contrib)
                            self.gradient[fullatomindex_qm] += QM1grad_contrib
                            self.gradient[fullatomindex_mm] += MM1grad_contrib

        # 3-layer ONIOM Energy and Gradient expression
        elif len(self.theories_N) == 3:
            self.energy = E_dict[(2,-1)] + E_dict[(0,0)] - E_dict[(1,0)] + E_dict[(1,1)] - E_dict[(2,1)]
            # E(High,Real) = E(Low,Real) + [E(High,Model) - E(Medium,Model)]
            #                +  [E(Medium,Inter) - E(Low,Inter)].
            print(f"Energy (Full-LL): {E_dict[(2,-1)]} Eh")
            print(f"Energy (Region1-HL): {E_dict[(0,0)]} Eh")
            print(f"Energy (Region1-IL): {E_dict[(1,0)]} Eh")
            print(f"Energy (Region2-IL): {E_dict[(1,1)]} Eh")
            print(f"Energy (Region2-LL): {E_dict[(2,1)]} Eh")

            if Grad:
                print("Gradient for 3-layer ONIOM is not yet ready")
                ashexit()

                if self.linkatoms is True:
                    print("Linkatom projection for ONIOM-3 not ready")
                    ashexit()

                # NOTE: We need to finish linkatom handling
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

        print("Final ONIOM energy:", self.energy)
        print_time_rel(module_init_time, modulename='ONIOM run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        if Grad:
            return self.energy,self.gradient
        else:
            return self.energy
