import copy
import time
import numpy as np
import math

import ash.modules.module_coords
from ash.modules.module_coords import Fragment, write_pdbfile
from ash.functions.functions_general import ashexit, BC, blankline, listdiff, print_time_rel,printdebug,print_line_with_mainheader,writelisttofile,print_if_level
import ash.settings_ash
from ash.modules.module_MM import coulombcharge

# QM/MM theory object.
# Required at init: qm_theory and qmatoms. Fragment not. Can come later
# TODO NOTE: If we add init arguments, remember to update Numfreq QMMM option as it depends on the keywords

class QMMMTheory:
    def __init__(self, qm_theory=None, qmatoms=None, fragment=None, mm_theory=None, charges=None,
                 embedding="elstat", printlevel=2, numcores=1, actatoms=None, frozenatoms=None, excludeboundaryatomlist=None,
                 unusualboundary=False, openmm_externalforce=False, TruncatedPC=False, TruncPCRadius=55, TruncatedPC_recalc_iter=50,
                qm_charge=None, qm_mult=None, dipole_correction=True):

        module_init_time = time.time()
        timeA = time.time()
        print_line_with_mainheader("QM/MM Theory")

        # Check for necessary keywords
        if qm_theory is None or qmatoms is None:
            print("Error: QMMMTheory requires defining: qm_theory, qmatoms, fragment")
            ashexit()
        # If fragment object has not been defined
        if fragment is None:
            print("fragment= keyword has not been defined for QM/MM. Exiting")
            ashexit()

        # Defining charge/mult of QM-region
        self.qm_charge = qm_charge
        self.qm_mult = qm_mult

        # Indicate that this is a hybrid QM/MM type theory
        self.theorytype = "QM/MM"

        # External force energy. Zero except when using openmm_externalforce
        self.extforce_energy = 0.0

        # Linkatoms False by default. Later checked.
        self.linkatoms = False

        # Counter for how often QMMMTheory.run is called
        self.runcalls = 0

        # Whether we are using OpenMM custom external forces or not
        # NOTE: affects runmode
        self.openmm_externalforce = openmm_externalforce

        # Theory level definitions
        self.printlevel=printlevel
        self.qm_theory=qm_theory
        self.qm_theory_name = self.qm_theory.__class__.__name__
        self.mm_theory=mm_theory
        self.mm_theory_name = self.mm_theory.__class__.__name__
        if self.mm_theory_name == "str":
            self.mm_theory_name="None"

        print("QM-theory:", self.qm_theory_name)
        print("MM-theory:", self.mm_theory_name)

        self.fragment=fragment
        self.coords=fragment.coords
        self.elems=fragment.elems
        self.connectivity=fragment.connectivity

        # Region definitions
        self.allatoms=list(range(0,len(self.elems)))
        print("All atoms in fragment:", len(self.allatoms))
        self.num_allatoms=len(self.allatoms)
        # Sorting qmatoms list
        self.qmatoms = sorted(qmatoms)

        # All-atom Bool-array for whether atom-index is a QM-atom index or not
        # Used by make_QM_PC_gradient
        self.xatom_mask = np.isin(self.allatoms, self.qmatoms)
        self.sum_xatom_mask = np.sum(self.xatom_mask)

        if len(self.qmatoms) == 0:
            print("Error: List of qmatoms provided is empty. This is not allowed.")
            ashexit()
        # self.mmatoms = listdiff(self.allatoms, self.qmatoms)
        self.mmatoms = np.setdiff1d(self.allatoms, self.qmatoms)

        # FROZEN AND ACTIVE ATOM REGIONS for NonbondedTheory
        if self.mm_theory_name == "NonBondedTheory":
            #NOTE: To be looked at. actatoms and frozenatoms have no meaning in OpenMMTHeory. NonbondedTheory, however.
            if actatoms is None and frozenatoms is None:
                #print("Actatoms/frozenatoms list not passed to QM/MM object. Will do all frozen interactions in MM (expensive).")
                #print("All {} atoms active, no atoms frozen in QM/MM definition (may not be frozen in optimizer)".format(len(self.allatoms)))
                self.actatoms=self.allatoms
                self.frozenatoms=[]
            elif actatoms is not None and frozenatoms is None:
                print("Actatoms list passed to QM/MM object. Will skip all frozen interactions in MM.")
                #Sorting actatoms list
                self.actatoms = sorted(actatoms)
                self.frozenatoms = listdiff(self.allatoms, self.actatoms)
                print("{} active atoms, {} frozen atoms".format(len(self.actatoms), len(self.frozenatoms)))
            elif frozenatoms is not None and actatoms is None:
                print("Frozenatoms list passed to QM/MM object. Will skip all frozen interactions in MM.")
                self.frozenatoms = sorted(frozenatoms)
                self.actatoms = listdiff(self.allatoms, self.frozenatoms)
                print("{} active atoms, {} frozen atoms".format(len(self.actatoms), len(self.frozenatoms)))
            else:
                print("active_atoms and frozen_atoms can not be both defined")
                ashexit()

        # print("List of all atoms:", self.allatoms)
        print("QM region ({} atoms): {}".format(len(self.qmatoms),self.qmatoms))
        print("MM region ({} atoms)".format(len(self.mmatoms)))

        # Setting QM/MM qmatoms in QMtheory also (used for Spin-flipping currently)
        self.qm_theory.qmatoms=self.qmatoms


        #Setting numcores of object.
        # This will be when calling QMtheory and probably MMtheory

        # numcores-setting in QMMMTheory takes precedent
        if numcores != 1:
            self.numcores=numcores
        # If QMtheory numcores was set (and QMMMTHeory not)
        elif self.qm_theory.numcores != 1:
            self.numcores=self.qm_theory.numcores
        # Default 1 proc
        else:
            self.numcores=1
        print("QM/MM object selected to use {} cores".format(self.numcores))

        # Embedding type: mechanical, electrostatic etc.
        self.embedding=embedding

        if self.embedding.lower() == "elstat" or self.embedding.lower() == "electrostatic" or self.embedding.lower() == "electronic":
            self.embedding="elstat"
            self.PC = True
        elif self.embedding.lower() == "mechanical" or self.embedding.lower() == "mech":
            self.embedding="mech"
            self.PC = False
        else:
            print("Unknown embedding. Valid options are: elstat (synonyms: electrostatic, electronic), mech (synonym: mechanical)")
            ashexit()
        print("Embedding:", self.embedding)
        # Whether to do dipole correction or not
        # Note: For regular electrostatic embedding this should be True
        # Turn off for charge-shifting
        self.dipole_correction=dipole_correction

        # Whether MM-shifted performed or not. Will be set to True by self.ShiftMMCharges
        self.chargeshifting_done=False

        # if atomcharges are not passed to QMMMTheory object, get them from MMtheory (that should have been defined then)
        if charges is None:
            print("No atomcharges list passed to QMMMTheory object")
            self.charges=[]
            if self.mm_theory_name == "OpenMMTheory":
                print("Getting system charges from OpenMM object")
                self.charges = mm_theory.charges
            elif self.mm_theory_name == "NonBondedTheory":
                print("Getting system charges from NonBondedTheory object")
                #Todo: normalize charges vs atom_charges
                self.charges=mm_theory.atom_charges

            else:
                print("Unrecognized MM theory for QMMMTheory")
                ashexit()
        else:
            print("Reading in charges")
            if len(charges) != len(fragment.atomlist):
                print(BC.FAIL,"Number of charges not matching number of fragment atoms. Exiting.",BC.END)
                ashexit()
            self.charges=charges
            self.mm_theory.update_charges(self.fragment.allatoms,self.charges)

        if len(self.charges) == 0:
            print("No charges present in QM/MM object. Exiting...")
            ashexit()

        # Flag to check whether QMCharges have been zeroed in self.charges_qmregionzeroed list
        self.QMChargesZeroed=False

        # CHARGES DEFINED FOR OBJECT:
        # Self.charges are original charges that are defined above (on input, from OpenMM or from NonBondedTheory)
        # self.charges_qmregionzeroed is self.charges but with 0-value for QM-atoms
        # self.pointcharges are pointcharges that the QM-code will see (dipole-charges, no zero-valued charges etc)
        # Length of self.charges: system size
        # Length of self.charges_qmregionzeroed: system size
        # Length of self.pointcharges: unknown. does not contain zero-valued charges (e.g. QM-atoms etc.), contains dipole-charges

        # self.charges_qmregionzeroed will have QM-charges zeroed (but not removed)
        self.charges_qmregionzeroed = []

        # Self.pointcharges are pointcharges that the QM-program will see (but not the MM program)
        # They have QM-atoms zeroed, zero-charges removed, dipole-charges added etc.
        # Defined later
        self.pointcharges = []

        # Truncated PC-region option
        self.TruncatedPC = TruncatedPC
        self.TruncPCRadius = TruncPCRadius
        self.TruncatedPCcalls = 0
        self.TruncatedPC_recalc_flag = False
        self.TruncatedPC_recalc_iter = TruncatedPC_recalc_iter

        if self.TruncatedPC is True:
            print("Truncated PC approximation in QM/MM is active.")
            print("TruncPCRadius:", self.TruncPCRadius)
            print("TruncPC Recalculation iteration:", self.TruncatedPC_recalc_iter)

        # If MM THEORY (not just pointcharges)
        if mm_theory is not None:

            # Sanity check. Same number of atoms in fragment and MM object ?
            if fragment.numatoms != mm_theory.numatoms:
                print("")
                print(BC.FAIL,"Number of atoms in fragment ({}) and MMtheory object differ ({})".format(fragment.numatoms,mm_theory.numatoms),BC.END)
                print(BC.FAIL,"This does not make sense. Check coordinates and forcefield files. Exiting...", BC.END)
                ashexit()

            # Add possible exception for QM-QM atoms here.
            # Maybe easier to just just set charges to 0. LJ for QM-QM still needs to be done by MM code
            # if self.mm_theory_name == "OpenMMTheory":
            #
                # print("Now adding exceptions for frozen atoms")
                # if len(self.frozenatoms) > 0:
                #    print("Here adding exceptions for OpenMM")
                #    print("Frozen-atom exceptions currently inactive...")
                # print("Num frozen atoms: ", len(self.frozenatoms))
                # Disabling for now, since so bloody slow. Need to speed up
                # mm_theory.addexceptions(self.frozenatoms)

            # Check if we need linkatoms by getting boundary atoms dict:

            # Update: Tolerance modification to make sure we definitely catch connected atoms and get QM-MM boundary right.
            # Scale=1.0 and tol=0.1 fails for S-C bond in rubredoxin from a classical MD run
            # Bumping up a bit here.
            # 21 Sep 2023. bumping from +0.1 to +0.2. C-C bond in lysine failed
            conn_scale = ash.settings_ash.settings_dict["scale"]
            conn_tolerance = ash.settings_ash.settings_dict["tol"]+0.2

            # If QM-MM boundary issue and ASH exits then printing QM-coordinates is useful
            print("QM-region coordinates (before linkatoms):")
            ash.modules.module_coords.print_coords_for_atoms(self.coords, self.elems, self.qmatoms, labels=self.qmatoms)
            print()
            self.boundaryatoms = ash.modules.module_coords.get_boundary_atoms(self.qmatoms, self.coords, self.elems, conn_scale,
                conn_tolerance, excludeboundaryatomlist=excludeboundaryatomlist, unusualboundary=unusualboundary)
            if len(self.boundaryatoms) >0:
                print("Found covalent QM-MM boundary. Linkatoms option set to True")
                print("Boundaryatoms (QM:MM pairs):", self.boundaryatoms)
                print("Note: used connectivity settings, scale={} and tol={} to determine boundary.".format(conn_scale,conn_tolerance))
                self.linkatoms = True
                # Get MM boundary information. Stored as self.MMboundarydict
                self.get_MMboundary(conn_scale,conn_tolerance)
            else:
                print("No covalent QM-MM boundary. Linkatoms and dipole_correction options set to False")
                self.linkatoms=False
                self.dipole_correction=False

            # Removing possible QM atom constraints in OpenMMTheory
            # Will only apply when running OpenMM_Opt or OpenMM_MD
            if self.mm_theory_name == "OpenMMTheory":
                self.mm_theory.remove_constraints_for_atoms(self.qmatoms)

            # Remove bonded interactions in MM part. Only in OpenMM. Assuming they were never defined in NonbondedTHeory
                print("Removing bonded terms for QM-region in MMtheory")
                self.mm_theory.modify_bonded_forces(self.qmatoms)

                # NOTE: Temporary. Adding exceptions for nonbonded QM atoms. Will ignore QM-QM Coulomb and LJ interactions.
                # NOTE: For QM-MM interactions Coulomb charges are zeroed below (update_charges and delete_exceptions)
                print("Removing nonbonded terms for QM-region in MMtheory (QM-QM interactions)")
                self.mm_theory.addexceptions(self.qmatoms)

            ########################
            # CHANGE CHARGES
            ########################
            # Keeping self.charges as originally defined.
            # Setting QM charges to 0 since electrostatic embedding
            # and Charge-shift QM-MM boundary

            # Zero QM charges for electrostatic embedding
            # TODO: DO here or inside run instead?? Needed for MM code.
            if self.embedding.lower() == "elstat":
                print("Charges of QM atoms set to 0 (since Electrostatic Embedding):")
                self.ZeroQMCharges() #Modifies self.charges_qmregionzeroed
                # print("length of self.charges_qmregionzeroed :", len(self.charges_qmregionzeroed))
                # TODO: make sure this works for OpenMM and for NonBondedTheory
                # Updating charges in MM object.
                self.mm_theory.update_charges(self.qmatoms,[0.0 for i in self.qmatoms])

            if self.mm_theory_name == "OpenMMTheory":
                # Deleting Coulomb exception interactions involving QM and MM atoms
                self.mm_theory.delete_exceptions(self.qmatoms)
                # Option to create OpenMM externalforce that handles full system
                if self.openmm_externalforce == True:
                    print("openmm_externalforce is True")
                    # print("Creating new OpenMM custom external force")
                    # MOVED FROM HERE TO OPENMM_MD

            # Printing charges: all or only QM
            if self.printlevel > 2:
                for i in self.allatoms:
                    if i in self.qmatoms:
                        if self.embedding.lower() == "elstat":
                            print("QM atom {} ({}) charge: {}".format(i, self.elems[i], self.charges_qmregionzeroed[i]))
                        else:
                            print("QM atom {} ({}) charge: {}".format(i, self.elems[i], self.charges[i]))
                    else:
                        if self.printlevel > 3:
                            print("MM atom {} ({}) charge: {}".format(i, self.elems[i], self.charges_qmregionzeroed[i]))
            blankline()
        else:
            # Case: No actual MM theory but we still want to zero charges for QM elstate embedding calculation
            # TODO: Remove option for no MM theory or keep this ??
            if self.embedding.lower() == "elstat":
                self.ZeroQMCharges() #Modifies self.charges_qmregionzeroed
            self.linkatoms=False
            self.dipole_correction=False
        print_time_rel(module_init_time, modulename='QM/MM object creation', currprintlevel=self.printlevel, moduleindex=3)


    #From QM1:MM1 boundary dict, get MM1:MMx boundary dict (atoms connected to MM1)
    def get_MMboundary(self,scale,tol):
        timeA=time.time()
        # if boundarydict is not empty we need to zero MM1 charge and distribute charge from MM1 atom to MM2,MM3,MM4
        #Creating dictionary for each MM1 atom and its connected atoms: MM2-4
        self.MMboundarydict={}
        for (QM1atom,MM1atom) in self.boundaryatoms.items():
            connatoms = ash.modules.module_coords.get_connected_atoms(self.coords, self.elems, scale,tol, MM1atom)
            #Deleting QM-atom from connatoms list
            connatoms.remove(QM1atom)
            self.MMboundarydict[MM1atom] = connatoms

        # Used by ShiftMMCharges
        self.MMboundary_indices = list(self.MMboundarydict.keys())
        self.MMboundary_counts = np.array([len(self.MMboundarydict[i]) for i in self.MMboundary_indices])

        print("")
        print("MM boundary (MM1:MMx pairs):", self.MMboundarydict)
        print_time_rel(timeA, modulename="get_MMboundary")
    # Set QMcharges to Zero and shift charges at boundary
    #TODO: Add both L2 scheme (delete whole charge-group of M1) and charge-shifting scheme (shift charges to Mx atoms and add dipoles for each Mx atom)

    def ZeroQMCharges(self):
        timeA=time.time()
        print("Setting QM charges to Zero")
        # Looping over charges and setting QM atoms to zero
        # 1. Copy charges to charges_qmregionzeroed
        self.charges_qmregionzeroed=copy.copy(self.charges)
        # 2. change charge for QM-atom
        for i, c in enumerate(self.charges_qmregionzeroed):
            # Setting QMatom charge to 0
            if i in self.qmatoms:
                self.charges_qmregionzeroed[i] = 0.0
        # 3. Flag that this has been done
        self.QMChargesZeroed = True
        print_time_rel(timeA, modulename="ZeroQMCharges")

    def ShiftMMCharges(self):
        if self.chargeshifting_done is False:
            self.ShiftMMCharges_new2()
        else:
            print("Charge shifting already done. Using previous charges")
    #TODO: Delete old version below

    def ShiftMMCharges_old(self):
        timeA=time.time()
        if self.printlevel > 1:
            print("Shifting MM charges at QM-MM boundary.")
        # print("len self.charges_qmregionzeroed: ", len(self.charges_qmregionzeroed))
        # print("len self.charges: ", len(self.charges))

        # Create self.pointcharges list
        self.pointcharges=copy.copy(self.charges_qmregionzeroed)

        # Looping over charges and setting QM/MM1 atoms to zero and shifting charge to neighbouring atoms
        for i, c in enumerate(self.pointcharges):

            # If index corresponds to MMatom at boundary, set charge to 0 (charge-shifting
            if i in self.MMboundarydict.keys():
                MM1charge = self.charges[i]
                # print("MM1atom charge: ", MM1charge)
                self.pointcharges[i] = 0.0
                # MM1 charge fraction to be divided onto the other MM atoms
                MM1charge_fract = MM1charge / len(self.MMboundarydict[i])
                # print("MM1charge_fract :", MM1charge_fract)

                # Putting the fractional charge on each MM2 atom
                for MMx in self.MMboundarydict[i]:
                    self.pointcharges[MMx] += MM1charge_fract
        self.chargeshifting_done=True
        print_time_rel(timeA, modulename="ShiftMMCharges-old", currprintlevel=self.printlevel, currthreshold=1)

    def ShiftMMCharges_new(self):
        timeA=time.time()
        if self.printlevel > 1:
            print("new. Shifting MM charges at QM-MM boundary.")
        # Convert lists to NumPy arrays for faster computations
        self.pointcharges = np.array(self.charges_qmregionzeroed)
        for i, c in enumerate(self.pointcharges):
            if i in self.MMboundarydict:
                MM1charge = self.charges[i]
                self.pointcharges[i] = 0.0
                # Calculate MM1 charge fraction to be divided onto the other MM atoms
                MM1charge_fract = MM1charge / len(self.MMboundarydict[i])
                # Distribute charge fraction to neighboring MM atoms
                self.pointcharges[list(self.MMboundarydict[i])] += MM1charge_fract
        self.chargeshifting_done=True
        print_time_rel(timeA, modulename="ShiftMMCharges-new", currprintlevel=self.printlevel, currthreshold=1)
        return
    def ShiftMMCharges_new2(self):
        timeA=time.time()
        if self.printlevel > 1:
            print("new. Shifting MM charges at QM-MM boundary.")

        # Convert lists to NumPy arrays for faster computations
        print_time_rel(timeA, modulename="x0", currprintlevel=self.printlevel, currthreshold=1)
        self.pointcharges = np.array(self.charges_qmregionzeroed)
        self.charges=np.array(self.charges)

        print_time_rel(timeA, modulename="x1", currprintlevel=self.printlevel, currthreshold=1)
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
        print_time_rel(timeA, modulename="ShiftMMCharges-new2", currprintlevel=self.printlevel, currthreshold=1)
        return

    # Create dipole charge (twice) for each MM2 atom that gets fraction of MM1 charge
    def get_dipole_charge(self,delq,direction,mm1index,mm2index,current_coords):
        # oldMM_distance = ash.modules.module_coords.distance_between_atoms(fragment=self.fragment,
        #                                                               atoms=[mm1index, mm2index])
        # Coordinates and distance
        mm1coords=np.array(current_coords[mm1index])
        mm2coords=np.array(current_coords[mm2index])
        MM_distance = ash.modules.module_coords.distance(mm1coords,mm2coords) # Distance between MM1 and MM2

        SHIFT=0.15
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
        q0 = 0.5 * d / SHIFT
        # Actual shift
        shift = direction * SHIFT * ( MM_distance / 2.5 )
        # Position
        pos = mm2coords+np.array((shift*normdiffvector))
        # Returning charge with sign based on direction and position
        # Return coords as regular list
        return -q0*direction,list(pos)

    def SetDipoleCharges(self,current_coords):
        checkpoint=time.time()
        if self.printlevel > 1:
            print("Adding extra charges to preserve dipole moment for charge-shifting")
            print("MMboundarydict:", self.MMboundarydict)
        # Adding 2 dipole pointcharges for each MM2 atom
        self.dipole_charges = []
        self.dipole_coords = []


        for MM1,MMx in self.MMboundarydict.items():
            # Getting original MM1 charge (before set to 0)
            MM1charge = self.charges[MM1]
            MM1charge_fract=MM1charge/len(MMx)

            for MM in MMx:
                q_d1, pos_d1 = self.get_dipole_charge(MM1charge_fract,1,MM1,MM,current_coords)
                q_d2, pos_d2 = self.get_dipole_charge(MM1charge_fract,-1,MM1,MM,current_coords)
                self.dipole_charges.append(q_d1)
                self.dipole_charges.append(q_d2)
                self.dipole_coords.append(pos_d1)
                self.dipole_coords.append(pos_d2)
        print_time_rel(checkpoint, modulename='SetDipoleCharges', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)

    # Reasonably efficient version (this dominates QM/MM gradient prepare)
    def make_QM_PC_gradient_old(self):
        self.QM_PC_gradient = np.zeros((len(self.allatoms), 3))
        qmatom_indices = np.where(np.isin(self.allatoms, self.qmatoms))[0]
        pcatom_indices = np.where(~np.isin(self.allatoms, self.qmatoms))[0] # ~ is NOT operator in numpy
        self.QM_PC_gradient[qmatom_indices] = self.QMgradient_wo_linkatoms[:len(qmatom_indices)]
        self.QM_PC_gradient[pcatom_indices] = self.PCgradient[:len(pcatom_indices)]
        return

    # Faster version. Also, uses precalculated mask.
    def make_QM_PC_gradient_optimized(self):
        self.QM_PC_gradient[self.xatom_mask] = self.QMgradient_wo_linkatoms
        self.QM_PC_gradient[~self.xatom_mask] = self.PCgradient[:self.num_allatoms - self.sum_xatom_mask]
        return

    make_QM_PC_gradient=make_QM_PC_gradient_optimized

    # TruncatedPCfunction control flow for pointcharge field passed to QM program
    def TruncatedPCfunction(self, used_qmcoords):
        self.TruncatedPCcalls+=1
        print("TruncatedPC approximation!")
        if self.TruncatedPCcalls == 1 or self.TruncatedPCcalls % self.TruncatedPC_recalc_iter == 0:
            self.TruncatedPC_recalc_flag=True
            # print("This is first QM/MM run. Will calculate Full-Trunc correction in this step")
            print(f"This is QM/MM run no. {self.TruncatedPCcalls}.  Will calculate Full-Trunc correction in this step")
            # Origin coords point is center of QM-region
            origincoords=ash.modules.module_coords.get_centroid(used_qmcoords)
            # Determine the indices associated with the truncated PC field once
            self.determine_truncatedPC_indices(origincoords)
            print("Truncated PC-region size: {} charges".format(len(self.truncated_PC_region_indices)))
            # Saving full PCs and coords for 1st iteration
            # NOTE: Here using self.pointcharges_original (set by runprep)
            # since self.pointcharges may be truncated-version from last iter
            self.pointcharges_full = copy.copy(self.pointcharges_original)
            self.pointchargecoords_full = copy.copy(self.pointchargecoords)

            # Determining truncated PC-field
            self.pointcharges=[self.pointcharges_full[i] for i in self.truncated_PC_region_indices]
            self.pointchargecoords=np.take(self.pointchargecoords_full, self.truncated_PC_region_indices, axis=0)
        else:
            self.TruncatedPC_recalc_flag=False
            print("This is QM/MM run no. {}. Using approximate truncated PC field: {} charges".format(self.TruncatedPCcalls,len(self.truncated_PC_region_indices)))
            # NOTE: Here taking 1st-iter full PCs (values have not changed during opt/md)
            self.pointcharges = [self.pointcharges_full[i] for i in self.truncated_PC_region_indices]
            # NOTE: Here taking from CURRENT full pointchargecoords (not old full from step 1) since coords have changed
            self.pointchargecoords = np.take(self.pointchargecoords, self.truncated_PC_region_indices, axis=0)

    # Determine truncated PC field indices based on initial coordinates
    # Coordinates and charges for each Opt cycle defined later.
    def determine_truncatedPC_indices(self,origincoords):
        region_indices=[]
        for index,allc in enumerate(self.pointchargecoords):
            dist=ash.modules.module_coords.distance(origincoords,allc)
            if dist < self.TruncPCRadius:
                region_indices.append(index)
        # Only unique and sorting:
        self.truncated_PC_region_indices = np.unique(region_indices).tolist()
        # Removing dipole charges also (end of list)

    def oldcalculate_truncPC_gradient_correction(self,QMgradient_full, PCgradient_full, QMgradient_trunc, PCgradient_trunc):
        #Correction for QM-atom gradient
        self.original_QMcorrection_gradient=np.zeros((len(QMgradient_full)-self.num_linkatoms, 3))
        #Correction for PC gradient
        self.original_PCcorrection_gradient=np.zeros((len(PCgradient_full), 3))

        for i in range(0,len(QMgradient_full)-self.num_linkatoms):
            #print("QM index:", i)
            qmfullgrad=QMgradient_full[i]
            qmtruncgrad=QMgradient_trunc[i]
            #print("qmfullgrad:", qmfullgrad)
            #print("qmtruncgrad:", qmtruncgrad)
            qmdifference=qmfullgrad-qmtruncgrad
            #print("qmdifference:", qmdifference)
            self.original_QMcorrection_gradient[i] = qmdifference
        count=0
        for i in range(0,len(PCgradient_full)):
            if i in self.truncated_PC_region_indices:
                #print("TruncPC index:", i)
                pcfullgrad=PCgradient_full[i]
                pctruncgrad=PCgradient_trunc[count]
                #print("pcfullgrad:", pcfullgrad)
                #print("pctruncgrad:", pctruncgrad)
                difference=pcfullgrad-pctruncgrad
                #print("difference:", difference)
                self.original_PCcorrection_gradient[i] = difference
                count+=1
            else:
                # Keep original full contribution
                self.original_PCcorrection_gradient[i] = PCgradient_full[i]
        return
    #New more efficient version
    def calculate_truncPC_gradient_correction(self, QMgradient_full, PCgradient_full, QMgradient_trunc, PCgradient_trunc):
        # QM part
        qm_difference = QMgradient_full[:len(QMgradient_full)-self.num_linkatoms] - QMgradient_trunc[:len(QMgradient_full)-self.num_linkatoms]
        self.original_QMcorrection_gradient = qm_difference
        # PC part
        truncated_indices = np.array(self.truncated_PC_region_indices)
        pc_difference = np.zeros((len(PCgradient_full), 3))
        pc_difference[truncated_indices] = PCgradient_full[truncated_indices] - PCgradient_trunc
        pc_difference[~np.isin(np.arange(len(PCgradient_full)), truncated_indices)] = PCgradient_full[~np.isin(np.arange(len(PCgradient_full)), truncated_indices)]
        self.original_PCcorrection_gradient = pc_difference
        return

    #This updates the calculated truncated PC gradient to be full-system gradient
    #by combining with the original 1st step correction
    def oldTruncatedPCgradientupdate(self,QMgradient_wo_linkatoms,PCgradient):

        #QM part
        newQMgradient_wo_linkatoms = QMgradient_wo_linkatoms + self.original_QMcorrection_gradient
        #PC part
        new_full_PC_gradient=np.zeros((len(self.original_PCcorrection_gradient), 3))
        count=0
        for i in range(0,len(new_full_PC_gradient)):
            if i in self.truncated_PC_region_indices:
                #Now updating with gradient from active region
                new_full_PC_gradient[i] = self.original_PCcorrection_gradient[i] + PCgradient[count]
                count+=1
            else:
                new_full_PC_gradient[i] = self.original_PCcorrection_gradient[i]
        return newQMgradient_wo_linkatoms, new_full_PC_gradient

    def TruncatedPCgradientupdate(self, QMgradient_wo_linkatoms, PCgradient):
        newQMgradient_wo_linkatoms = QMgradient_wo_linkatoms + self.original_QMcorrection_gradient

        new_full_PC_gradient = np.copy(self.original_PCcorrection_gradient)
        new_full_PC_gradient[self.truncated_PC_region_indices] += PCgradient

        return newQMgradient_wo_linkatoms, new_full_PC_gradient

    def set_numcores(self,numcores):
        print(f"Setting new numcores {numcores}for QMtheory and MMtheory")
        self.qm_theory.set_numcores(numcores)
        self.mm_theory.set_numcores(numcores)
    # Method to grab dipole moment from outputfile (assumes run has been executed)
    def get_dipole_moment(self):
        try:
            print("Grabbing dipole moment from QM-part of QM/MM theory.")
            dipole = self.qm_theory.get_dipole_moment()
        except:
            print("Error: Could not grab dipole moment from QM-part of QM/MM theory.")
        return dipole
    #Method to polarizability from outputfile (assumes run has been executed)
    def get_polarizability_tensor(self):
        try:
            print("Grabbing polarizability from QM-part of QM/MM theory.")
            polarizability = self.qm_theory.get_polarizability_tensor()
        except:
            print("Error: Could not grab polarizability from QM-part of QM/MM theory.")
        return polarizability
    # General run
    def run(self, current_coords=None, elems=None, Grad=False, numcores=1, exit_after_customexternalforce_update=False, label=None, charge=None, mult=None):

        if self.printlevel >= 2:
            print(BC.WARNING, BC.BOLD, "------------RUNNING QM/MM MODULE-------------", BC.END)
            print("QM Module:", self.qm_theory_name)
            print("MM Module:", self.mm_theory_name)

        # OPTION: QM-region charge/mult from QMMMTheory definition
        # If qm_charge/qm_mult defined then we use. Otherwise charge/mult may have been defined by jobtype-function and passed on via run
        if self.qm_charge is not None:
            if self.printlevel > 1:
                print("Charge provided from QMMMTheory object: ", self.qm_charge)
            charge = self.qm_charge
        if self.qm_mult is not None:
            if self.printlevel > 1:
                print("Mult provided from QMMMTheory object: ", self.qm_mult)
            mult = self.qm_mult

        # Checking if charge and mult has been provided. Exit if not.
        if charge is None or mult is None:
            print(BC.FAIL, "Error. charge and mult has not been defined for QMMMTheory.run method", BC.END)
            ashexit()

        if self.printlevel >1 :
            print("QM-region Charge: {} Mult: {}".format(charge,mult))

        if self.embedding.lower() == "mech":
            return self.mech_run(current_coords=current_coords, elems=elems, Grad=Grad, numcores=numcores, exit_after_customexternalforce_update=exit_after_customexternalforce_update,
                label=label, charge=charge, mult=mult)
        elif self.embedding.lower() == "elstat":
            return self.elstat_run(current_coords=current_coords, elems=elems, Grad=Grad, numcores=numcores, exit_after_customexternalforce_update=exit_after_customexternalforce_update,
                label=label, charge=charge, mult=mult)
        else:
            print("Unknown embedding. Exiting")
            ashexit()

    # Mechanical embedding run
    def mech_run(self, current_coords=None, elems=None, Grad=False, numcores=1, exit_after_customexternalforce_update=False, label=None, charge=None, mult=None):
        print("Mechanical embedding not ready yet")
        module_init_time=time.time()
        CheckpointTime = time.time()
        if self.printlevel >= 2:
            print("Embedding: Mechanical")

        #############################################
        # If this is first run then do QM/MM runprep
        # Only do once to avoid cost in each step
        #############################################
        if self.runcalls == 0:
            print("First QMMMTheory run. Running runprep")
            self.runprep(current_coords)
            # This creates self.current_qmelems,
            # self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords

        # Updating runcalls
        self.runcalls+=1

        #########################################################################################
        # General QM-code energy+gradient call.
        #########################################################################################

        # Split current_coords into MM-part and QM-part efficiently.
        used_mmcoords, used_qmcoords = current_coords[~self.xatom_mask], current_coords[self.xatom_mask]

        if self.linkatoms is True:
            # Update linkatom coordinates. Sets: self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords
            linkatoms_coords = self.create_linkatoms(current_coords)
            # Add linkatom coordinates to QM-coordinates
            used_qmcoords = np.append(used_qmcoords, np.array(linkatoms_coords), axis=0)

        # If numcores was set when calling QMMMTheory.run then using, otherwise use self.numcores
        if numcores == 1:
            numcores = self.numcores

        if self.printlevel >= 2:
            print("Running QM/MM object with {} cores available".format(numcores))

        ################
        # QMTheory.run
        ################
        print_time_rel(module_init_time, modulename='before-QMstep', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()
        if self.qm_theory_name == "None" or self.qm_theory_name == "ZeroTheory":
            print("No QMtheory. Skipping QM calc")
            QMenergy=0.0;self.linkatoms=False
            QMgradient=np.array([0.0, 0.0, 0.0])
        else:
            #Calling QM theory, providing current QM and MM coordinates.
            if Grad is True:
                QMenergy, QMgradient = self.qm_theory.run(current_coords=used_qmcoords, qm_elems=self.current_qmelems, Grad=True, 
                                                          PC=False, numcores=numcores, charge=charge, mult=mult)
            else:
                QMenergy = self.qm_theory.run(current_coords=used_qmcoords,qm_elems=self.current_qmelems, Grad=False, 
                                              PC=False, numcores=numcores, charge=charge, mult=mult)

        print_time_rel(CheckpointTime, modulename='QM step', moduleindex=2,currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()

        ##################################################################################
        # QM/MM gradient: Initializing and then adding QM gradient, linkatom gradient
        ##################################################################################

        self.QMenergy = QMenergy

        # Initializing QM/MM gradient
        self.QM_MM_gradient = np.zeros((len(current_coords), 3))
        if Grad:
            Grad_prep_CheckpointTime = time.time()
            # Defining QMgradient without linkatoms if present
            if self.linkatoms is True:
                self.QMgradient = QMgradient
                self.QMgradient_wo_linkatoms=QMgradient[0:-self.num_linkatoms] #remove linkatoms
            else:
                self.QMgradient = QMgradient
                self.QMgradient_wo_linkatoms=QMgradient

            # Adding QM gradient (without linkatoms) to QM_MM_gradient
            self.QM_MM_gradient[self.qmatoms] += self.QMgradient_wo_linkatoms

            # LINKATOM FORCE PROJECTION
            # Add contribution to QM1 and MM1 contribution???
            if self.linkatoms is True:
                CheckpointTime = time.time()

                for pair in sorted(self.linkatoms_dict.keys()):
                    # Grabbing linkatom data
                    linkatomindex = self.linkatom_indices.pop(0)
                    Lgrad = self.QMgradient[linkatomindex]
                    Lcoord = self.linkatoms_dict[pair]
                    # Grabbing QMatom info
                    fullatomindex_qm = pair[0]
                    qmatomindex = fullindex_to_qmindex(fullatomindex_qm, self.qmatoms)
                    Qcoord = used_qmcoords[qmatomindex]
                    # Grabbing MMatom info
                    fullatomindex_mm = pair[1]
                    Mcoord = current_coords[fullatomindex_mm]
                    # Getting gradient contribution to QM1 and MM1 atoms from linkatom
                    QM1grad_contr, MM1grad_contr = linkatom_force_contribution(Qcoord, Mcoord, Lcoord, Lgrad)

                    # Updating full QM_MM_gradient
                    self.QM_MM_gradient[fullatomindex_qm] += QM1grad_contr
                    self.QM_MM_gradient[fullatomindex_mm] += MM1grad_contr

            print_time_rel(CheckpointTime, modulename='linkatomgrad prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            print_time_rel(Grad_prep_CheckpointTime, modulename='QM/MM gradient prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            CheckpointTime = time.time()
        else:
            # No Grad
            self.QMenergy = QMenergy

        ################
        # MM THEORY
        ################
        if self.mm_theory_name == "NonBondedTheory":
            if self.printlevel >= 2:
                print("Running MM theory as part of QM/MM.")
                print("Using MM on full system.")
                print("Passing QM atoms to MMtheory run so that QM-QM pairs are skipped in LJ pairlist")
            self.MMenergy, self.MMgradient = self.mm_theory.run(current_coords=current_coords,
                                                               charges=self.charges, connectivity=self.connectivity,
                                                               qmatoms=self.qmatoms, actatoms=self.actatoms)
            # NOTE: Special: For mechanical embedding the charges have not been set to zero
            # Means we get QM-QM charge interactions (double-counting) that we need to correct
            # TODO: Should move this logic into module_MM instead. However, we have to implement for numpy, julia etc.
            # Calculating QM-QM contribution 
            qm_charges = [self.charges[i] for i in self.qmatoms]
            qm_coords = current_coords[self.qmatoms]
            E_qm_qm_elstat, G_qm_qm_elstat  = coulombcharge(qm_charges, qm_coords, mode="numpy")
            # Correcting E and G
            self.MMenergy -= E_qm_qm_elstat
            self.MMgradient[self.qmatoms] -= G_qm_qm_elstat

        elif self.mm_theory_name == "OpenMMTheory":
            if self.printlevel >= 2:
                print("Using OpenMM theory as part of QM/MM.")
            if Grad:
                CheckpointTime = time.time()
                # print("QM/MM Grad is True")
                # Provide self.QM_MM_gradient to OpenMMTheory
                if self.openmm_externalforce == True:
                    print_if_level(f"OpenMM externalforce is True", self.printlevel,2)
                    # Calculate energy associated with external force so that we can subtract it later
                    # self.extforce_energy = 3 * np.mean(np.sum(self.QM_MM_gradient * current_coords * 1.88972612546, axis=0))
                    scaled_current_coords = current_coords * 1.88972612546
                    self.extforce_energy = 3 * np.mean(np.sum(self.QM_MM_gradient * scaled_current_coords, axis=0))
                    print_if_level(f"Extforce energy: {self.extforce_energy}", self.printlevel,2)
                    print_time_rel(CheckpointTime, modulename='extforce prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
                    # NOTE: Now moved mm_theory.update_custom_external_force call to MD simulation instead
                    # as we don't have access to simulation object here anymore. Uses self.QM_PC_gradient
                    if exit_after_customexternalforce_update==True:
                        print_if_level(f"OpenMM custom external force updated. Exit requested", self.printlevel,2)
                        # This is used if OpenMM MD is handling forces and dynamics
                        return

                self.MMenergy, self.MMgradient = self.mm_theory.run(current_coords=current_coords, qmatoms=self.qmatoms, Grad=True)
            else:
                print("QM/MM Grad is false")
                self.MMenergy = self.mm_theory.run(current_coords=current_coords, qmatoms=self.qmatoms)
        else:
            self.MMenergy=0
        print_time_rel(CheckpointTime, modulename='MM step', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()

        if Grad:
            # Now assemble full QM/MM gradient by adding MM gradient
            assert len(self.QM_MM_gradient) == len(self.MMgradient)
            self.QM_MM_gradient = self.QM_MM_gradient + self.MMgradient

        # Final QM/MM Energy.
        self.QM_MM_energy = self.QMenergy+self.MMenergy

        if self.printlevel >= 2:
            blankline()
            print("{:<20} {:>20.12f}".format("QM energy: ", self.QMenergy))
            print("{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
            print("{:<20} {:>20.12f}".format("QM/MM energy: ", self.QM_MM_energy))
            blankline()

        # FINAL QM/MM GRADIENT ASSEMBLY and return
        if Grad is True:
            if self.printlevel >=3:
                print("Printlevel >=3: Printing all gradients to disk")
                # Writing QM gradient only
                ash.modules.module_coords.write_coords_all(self.QMgradient_wo_linkatoms, self.qmelems, indices=self.qmatoms, file="QMgradient-without-linkatoms_{}".format(label), description="QM gradient w/o linkatoms {} (au/Bohr):".format(label))
                # Writing QM+Linkatoms gradient
                ash.modules.module_coords.write_coords_all(self.MMgradient, self.elems, indices=self.allatoms, file="MMgradient_{}".format(label), description="MM gradient {} (au/Bohr):".format(label))
                # Writing full QM/MM gradient
                ash.modules.module_coords.write_coords_all(self.QM_MM_gradient, self.elems, indices=self.allatoms, file="QM_MMgradient_{}".format(label), description="QM/MM gradient {} (au/Bohr):".format(label))
            if self.printlevel >= 2:
                print(BC.WARNING,BC.BOLD,"------------ENDING QM/MM MODULE-------------",BC.END)
            print_time_rel(module_init_time, modulename='QM/MM mech run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_MM_energy, self.QM_MM_gradient
        else:
            print_time_rel(module_init_time, modulename='QM/MM mech run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_MM_energy

    def create_linkatoms(self, current_coords):
        checkpoint=time.time()
        # Get linkatom coordinates
        self.linkatoms_dict = ash.modules.module_coords.get_linkatom_positions(self.boundaryatoms,self.qmatoms, current_coords, self.elems)
        printdebug("linkatoms_dict:", self.linkatoms_dict)
        if self.printlevel > 1:
            print("Adding linkatom positions to QM coords")
        self.linkatom_indices = [len(self.qmatoms)+i for i in range(0,len(self.linkatoms_dict))]
        self.num_linkatoms = len(self.linkatom_indices)
        linkatoms_coords = [self.linkatoms_dict[pair] for pair in sorted(self.linkatoms_dict.keys())]

        print_time_rel(checkpoint, modulename='create_linkatoms', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
        return linkatoms_coords

    # Run-preparation (for both electrostatic and mechanical)
    # Things that only have to be done in the first QM/MM run
    def runprep(self, current_coords):
        init_time_runprep=time.time()
        CheckpointTime=time.time()

        # Set basic element lists
        self.qmelems=[self.elems[i] for i in self.qmatoms]
        self.mmelems=[self.elems[i] for i in self.mmatoms]

        # LINKATOMS (both mech and elstat)
        check_before_linkatoms=time.time()
        if self.linkatoms is True:
            linkatoms_coords = self.create_linkatoms(current_coords)
            self.current_qmelems = self.qmelems + ['H']*self.num_linkatoms
            if self.printlevel > 1:
                print("Number of MM atoms:", len(self.mmatoms))
                print(f"There are {self.num_linkatoms} linkatoms")
            # Do possible Charge-shifting. MM1 charge distributed to MM2 atoms
            if self.embedding.lower() == "elstat":
                if self.printlevel > 1:
                    print("Doing charge-shifting...")
                self.ShiftMMCharges() # Creates self.pointcharges
                if self.printlevel > 1: 
                    print("Number of pointcharges defined for whole system: ", len(self.pointcharges))
                # Defining pointcharges as only containing MM atoms
                self.pointcharges = [self.pointcharges[i] for i in self.mmatoms]
                if self.printlevel > 1:
                    print("Number of pointcharges defined for MM region: ", len(self.pointcharges))
                # TODO: Code alternative to Charge-shifting: L2 scheme which deletes whole charge-group that MM1 belongs to
                if self.dipole_correction is True:
                    print("Dipole correction is on. Adding dipole charges")
                    self.SetDipoleCharges(current_coords) # Creates self.dipole_charges and self.dipole_coords

                    # Adding dipole charge coords to MM coords (given to QM code) and defining pointchargecoords
                    if self.printlevel > 1:
                        print("Adding {} dipole charges to PC environment".format(len(self.dipole_charges)))

                    # Adding dipole charges to MM charges list (given to QM code)
                    self.pointcharges = list(self.pointcharges)+list(self.dipole_charges)
                    # Using element H for dipole charges. Only matters for CP2K GEEP
                    self.mm_elems_for_qmprogram = self.mmelems + ['H']*len(self.dipole_charges)
                    if self.printlevel > 1: print("Number of pointcharges after dipole addition: ", len(self.pointcharges))
                    print_time_rel(check_before_linkatoms, modulename='Linkatom-dipolecorrection', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
                else:
                    print("Dipole correction is off. Not adding any dipole charges")
                    self.mm_elems_for_qmprogram = self.mmelems
                    if self.printlevel > 1: print("Number of pointcharges: ", len(self.pointcharges))
        # CASE: No Linkatoms
        else:
            self.mm_elems_for_qmprogram = self.mmelems
            self.num_linkatoms = 0
            # If no linkatoms then use original self.qmelems
            self.current_qmelems = self.qmelems
            # If no linkatoms then self.pointcharges are just original charges with QM-region zeroed
            if self.embedding.lower() == "elstat":
                self.pointcharges = [self.charges_qmregionzeroed[i] for i in self.mmatoms]

        # NOTE: Now we have updated MM-coordinates (if doing linkatoms, with dipolecharges etc) and updated mm-charges (more, due to dipolecharges if linkatoms)
        # We also have MMcharges that have been set to zero due to QM/MM
        # We do not delete charges but set to zero
        # If no qmatoms then do MM-only
        if len(self.qmatoms) == 0:
            print("No qmatoms list provided. Setting QMtheory to None")
            self.qm_theory_name="None"
            self.QMenergy=0.0

        # For truncatedPC option.
        self.pointcharges_original=copy.copy(self.pointcharges)

        # Populate QM_PC_gradient for efficiency
        if self.embedding.lower() == "elstat":
            self.QM_PC_gradient = np.zeros((len(self.allatoms), 3))

        print_time_rel(init_time_runprep, modulename='runprep', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)

    # Electrostatic embedding run
    def elstat_run(self, current_coords=None, elems=None, Grad=False, numcores=1, exit_after_customexternalforce_update=False, label=None, charge=None, mult=None):
        module_init_time=time.time()
        CheckpointTime = time.time()

        if self.printlevel >= 2:
            print("Embedding: Electrostatic")

        #############################################
        # If this is first run then do QM/MM runprep
        # Only do once to avoid cost in each step
        #############################################
        if self.runcalls == 0:
            print("First QMMMTheory run. Running runprep")
            self.runprep(current_coords)
            # This creates self.pointcharges, self.current_qmelems, self.mm_elems_for_qmprogram
            # self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords

        # Updating runcalls
        self.runcalls+=1

        #########################################################################################
        # General QM-code energy+gradient call.
        #########################################################################################

        # Split current_coords into MM-part and QM-part efficiently.
        used_mmcoords, used_qmcoords = current_coords[~self.xatom_mask], current_coords[self.xatom_mask]

        if self.linkatoms is True:
            # Update linkatom coordinates. Sets: self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords
            linkatoms_coords = self.create_linkatoms(current_coords)
            # Add linkatom coordinates to QM-coordinates
            used_qmcoords = np.append(used_qmcoords, np.array(linkatoms_coords), axis=0)

        # Update self.pointchargecoords based on new current_coords
        # print("self.dipole_correction:", self.dipole_correction)
        if self.dipole_correction:
            self.SetDipoleCharges(current_coords) # Note: running again
            self.pointchargecoords = np.append(used_mmcoords, np.array(self.dipole_coords), axis=0)
        else:
            self.pointchargecoords = used_mmcoords

        # TRUNCATED PC Option: Speeding up QM/MM jobs of large systems by passing only a truncated PC field to the QM-code most of the time
        # Speeds up QM-pointcharge gradient that otherwise dominates
        # TODO: TruncatedPC is inactive
        if self.TruncatedPC is True:
            self.TruncatedPCfunction(used_qmcoords)

            # Modifies self.pointcharges and self.pointchargecoords
            # print("Number of charges after truncation :", len(self.pointcharges))
            # print("Number of charge coordinates after truncation :", len(self.pointchargecoords))

        # If numcores was set when calling QMMMTheory.run then using, otherwise use self.numcores
        if numcores == 1:
            numcores = self.numcores

        if self.printlevel > 1:
            print("Number of pointcharges (to QM program):", len(self.pointcharges))
            print("Number of charge coordinates:", len(self.pointchargecoords))
        if self.printlevel >= 2:
            print("Running QM/MM object with {} cores available".format(numcores))
        ################
        # QMTheory.run
        ################
        print_time_rel(module_init_time, modulename='before-QMstep', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()
        if self.qm_theory_name == "None" or self.qm_theory_name == "ZeroTheory":
            print("No QMtheory. Skipping QM calc")
            QMenergy=0.0;self.linkatoms=False;PCgradient=np.array([0.0, 0.0, 0.0])
            QMgradient=np.array([0.0, 0.0, 0.0])
        else:

            #TODO: Add check whether QM-code supports both pointcharges and pointcharge-gradient?

            #Calling QM theory, providing current QM and MM coordinates.
            if Grad is True:
                if self.PC is True:
                    QMenergy, QMgradient, PCgradient = self.qm_theory.run(current_coords=used_qmcoords,
                                                                                         current_MM_coords=self.pointchargecoords,
                                                                                         MMcharges=self.pointcharges,
                                                                                         qm_elems=self.current_qmelems, mm_elems=self.mm_elems_for_qmprogram,
                                                                                         charge=charge, mult=mult,
                                                                                         Grad=True, PC=True, numcores=numcores)
                else:
                    QMenergy, QMgradient = self.qm_theory.run(current_coords=used_qmcoords,
                                                      current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges,
                                                      qm_elems=self.current_qmelems, Grad=True, PC=False, numcores=numcores, charge=charge, mult=mult)
            else:
                QMenergy = self.qm_theory.run(current_coords=used_qmcoords,
                                                      current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges, mm_elems=self.mm_elems_for_qmprogram,
                                                      qm_elems=self.current_qmelems, Grad=False, PC=self.PC, numcores=numcores, charge=charge, mult=mult)

        print_time_rel(CheckpointTime, modulename='QM step', moduleindex=2,currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()

        # Final QM/MM gradient. Combine QM gradient, MM gradient, PC-gradient (elstat MM gradient from QM code).
        # Do linkatom force projections in the end
        # UPDATE: Do MM step in the end now so that we have options for OpenMM extern force
        if Grad == True:
            Grad_prep_CheckpointTime = time.time()
            # assert len(self.allatoms) == len(self.MMgradient)
            # Defining QMgradient without linkatoms if present
            if self.linkatoms==True:
                self.QMgradient = QMgradient
                QMgradient_wo_linkatoms=QMgradient[0:-self.num_linkatoms] #remove linkatoms
            else:
                self.QMgradient = QMgradient
                QMgradient_wo_linkatoms=QMgradient

            # if self.printlevel >= 2:
            #    ash.modules.module_coords.write_coords_all(self.QMgradient_wo_linkatoms, self.qmelems, indices=self.allatoms, file="QMgradient_wo_linkatoms", description="QM+ gradient withoutlinkatoms (au/Bohr):")


            # TRUNCATED PC Option:
            if self.TruncatedPC is True:
                # DONE ONCE: CALCULATE FULL PC GRADIENT TO DETERMINE CORRECTION
                if self.TruncatedPC_recalc_flag is True:
                    CheckpointTime = time.time()
                    truncfullCheckpointTime = time.time()

                    # We have calculated truncated QM and PC gradient
                    QMgradient_trunc = QMgradient
                    PCgradient_trunc = PCgradient

                    print("Now calculating full QM and PC gradient")
                    print("Number of PCs provided to QM-program:", len(self.pointcharges_full))
                    QMenergy_full, QMgradient_full, PCgradient_full = self.qm_theory.run(current_coords=used_qmcoords,
                                                                                         current_MM_coords=self.pointchargecoords_full,
                                                                                         MMcharges=self.pointcharges_full,
                                                                                         qm_elems=self.current_qmelems, charge=charge, mult=mult,
                                                                                         Grad=True, PC=True, numcores=numcores)
                    print_time_rel(CheckpointTime, modulename='trunc-pc full calculation', moduleindex=3)
                    CheckpointTime = time.time()

                    #TruncPC correction to QM energy
                    self.truncPC_E_correction = QMenergy_full - QMenergy
                    print(f"Truncated PC energy correction: {self.truncPC_E_correction} Eh")
                    self.QMenergy = QMenergy + self.truncPC_E_correction
                    #Now determine the correction once and for all
                    CheckpointTime = time.time()
                    self.calculate_truncPC_gradient_correction(QMgradient_full, PCgradient_full, QMgradient_trunc, PCgradient_trunc)
                    print_time_rel(CheckpointTime, modulename='calculate_truncPC_gradient_correction', moduleindex=3)
                    CheckpointTime = time.time()

                    #Now defining final QMgradient and PCgradient
                    self.QMgradient_wo_linkatoms, self.PCgradient =  self.TruncatedPCgradientupdate(QMgradient_wo_linkatoms,PCgradient)
                    print_time_rel(CheckpointTime, modulename='truncPC_gradient update ', moduleindex=3)
                    print_time_rel(truncfullCheckpointTime, modulename='trunc-full-step pcgrad update', moduleindex=3)

                else:
                    CheckpointTime = time.time()
                    #TruncPC correction to QM energy
                    self.QMenergy = QMenergy + self.truncPC_E_correction
                    self.QMgradient_wo_linkatoms, self.PCgradient =  self.TruncatedPCgradientupdate(QMgradient_wo_linkatoms,PCgradient)
                    print_time_rel(CheckpointTime, modulename='trunc pcgrad update', moduleindex=3)
            else:
                self.QMenergy = QMenergy
                #No TruncPC approximation active. No change to original QM and PCgradient from QMcode
                self.QMgradient_wo_linkatoms = QMgradient_wo_linkatoms
                if self.embedding.lower() == "elstat":
                    self.PCgradient = PCgradient

            # Populate QM_PC gradient (has full system size)
            CheckpointTime = time.time()
            self.make_QM_PC_gradient() #populates self.QM_PC_gradient
            print_time_rel(CheckpointTime, modulename='QMpcgrad prepare', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
            # self.QM_PC_gradient = np.zeros((len(self.allatoms), 3))
            #qmcount=0;pccount=0
            #for i in self.allatoms:
            #    if i in self.qmatoms:
            #        #QM-gradient. Linkatom gradients are skipped
            #        self.QM_PC_gradient[i]=self.QMgradient_wo_linkatoms[qmcount]
            #        qmcount+=1
            #    else:
            #        #Pointcharge-gradient. Dipole-charge gradients are skipped (never reached)
            #        self.QM_PC_gradient[i] = self.PCgradient[pccount]
            #        pccount += 1
            #assert qmcount == len(self.qmatoms)
            #assert pccount == len(self.mmatoms)
            #assert qmcount+pccount == len(self.allatoms)

            #print(" qmcount+pccount:", qmcount+pccount)
            #print("len(self.allatoms):", len(self.allatoms))
            #print("len self.QM_PC_gradient", len(self.QM_PC_gradient))
            #ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM_PC_gradient", description="QM_PC_gradient (au/Bohr):")

            #if self.printlevel >= 2:
            #    ash.modules.module_coords.write_coords_all(self.PCgradient, self.mmatoms, indices=self.allatoms, file="PCgradient", description="PC gradient (au/Bohr):")



            #if self.printlevel >= 2:
            #    ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM+PCgradient_before_linkatomproj", description="QM+PC gradient before linkatomproj (au/Bohr):")


            #LINKATOM FORCE PROJECTION
            # Add contribution to QM1 and MM1 contribution???
            if self.linkatoms is True:
                CheckpointTime = time.time()

                for pair in sorted(self.linkatoms_dict.keys()):
                    printdebug("pair: ", pair)
                    #Grabbing linkatom data
                    linkatomindex=self.linkatom_indices.pop(0)
                    printdebug("linkatomindex:", linkatomindex)
                    Lgrad=self.QMgradient[linkatomindex]
                    printdebug("Lgrad:",Lgrad)
                    Lcoord=self.linkatoms_dict[pair]
                    printdebug("Lcoord:", Lcoord)
                    #Grabbing QMatom info
                    fullatomindex_qm=pair[0]
                    printdebug("fullatomindex_qm:", fullatomindex_qm)
                    printdebug("self.qmatoms:", self.qmatoms)
                    qmatomindex=fullindex_to_qmindex(fullatomindex_qm,self.qmatoms)
                    printdebug("qmatomindex:", qmatomindex)
                    Qcoord=used_qmcoords[qmatomindex]
                    printdebug("Qcoords: ", Qcoord)

                    Qgrad=self.QM_PC_gradient[fullatomindex_qm]
                    printdebug("Qgrad (full QM/MM grad)s:", Qgrad)

                    #Grabbing MMatom info
                    fullatomindex_mm=pair[1]
                    printdebug("fullatomindex_mm:", fullatomindex_mm)
                    Mcoord=current_coords[fullatomindex_mm]
                    printdebug("Mcoord:", Mcoord)

                    Mgrad=self.QM_PC_gradient[fullatomindex_mm]
                    printdebug("Mgrad (full QM/MM grad): ", Mgrad)

                    #Now grabbed all components, calculating new projected gradient on QM atom and MM atom
                    newQgrad,newMgrad = linkatom_force_fix(Qcoord, Mcoord, Lcoord, Qgrad,Mgrad,Lgrad)
                    printdebug("newQgrad: ", newQgrad)
                    printdebug("newMgrad: ", newMgrad)

                    #Updating full QM_PC_gradient (used to be QM/MM gradient)
                    #self.QM_MM_gradient[fullatomindex_qm] = newQgrad
                    #self.QM_MM_gradient[fullatomindex_mm] = newMgrad
                    self.QM_PC_gradient[fullatomindex_qm] = newQgrad
                    self.QM_PC_gradient[fullatomindex_mm] = newMgrad


            print_time_rel(CheckpointTime, modulename='linkatomgrad prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            print_time_rel(Grad_prep_CheckpointTime, modulename='QM/MM gradient prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            CheckpointTime = time.time()
        else:
            #No Grad
            self.QMenergy = QMenergy

        #if self.printlevel >= 2:
        #    ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM+PCgradient", description="QM+PC gradient (au/Bohr):")



        # MM THEORY
        if self.mm_theory_name == "NonBondedTheory":
            if self.printlevel >= 2:
                print("Running MM theory as part of QM/MM.")
                print("Using MM on full system. Charges for QM region  have to be set to zero ")
                #printdebug("Charges for full system is: ", self.charges)
                print("Passing QM atoms to MMtheory run so that QM-QM pairs are skipped in pairlist")
                print("Passing active atoms to MMtheory run so that frozen pairs are skipped in pairlist")
            assert len(current_coords) == len(self.charges_qmregionzeroed)

            # NOTE: charges_qmregionzeroed for full system but with QM-charges zeroed (no other modifications)
            #NOTE: Using original system coords here (not with linkatoms, dipole etc.). Also not with deleted zero-charge coordinates.
            #charges list for full system, can be zeroed but we still want the LJ interaction

            self.MMenergy, self.MMgradient= self.mm_theory.run(current_coords=current_coords,
                                                               charges=self.charges_qmregionzeroed, connectivity=self.connectivity,
                                                               qmatoms=self.qmatoms, actatoms=self.actatoms)

        elif self.mm_theory_name == "OpenMMTheory":
            if self.printlevel >= 2:
                print("Using OpenMM theory as part of QM/MM.")
            if self.QMChargesZeroed==True:
                if self.printlevel >= 2:
                    print("Using MM on full system. Charges for QM region {} have been set to zero ".format(self.qmatoms))
            else:
                print("QMCharges have not been zeroed")
                ashexit()
            #printdebug("Charges for full system is: ", self.charges)
            #Todo: Need to make sure OpenMM skips QM-QM Lj interaction => Exclude
            #Todo: Need to have OpenMM skip frozen region interaction for speed  => => Exclude
            if Grad==True:
                CheckpointTime = time.time()
                #print("QM/MM Grad is True")
                #Provide QM_PC_gradient to OpenMMTheory
                if self.openmm_externalforce == True:
                    print_if_level(f"OpenMM externalforce is True", self.printlevel,2)
                    #Calculate energy associated with external force so that we can subtract it later
                    #self.extforce_energy = 3 * np.mean(np.sum(self.QM_PC_gradient * current_coords * 1.88972612546, axis=0))
                    scaled_current_coords = current_coords * 1.88972612546
                    self.extforce_energy = 3 * np.mean(np.sum(self.QM_PC_gradient * scaled_current_coords, axis=0))
                    print_if_level(f"Extforce energy: {self.extforce_energy}", self.printlevel,2)
                    print_time_rel(CheckpointTime, modulename='extforce prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
                    #NOTE: Now moved mm_theory.update_custom_external_force call to MD simulation instead
                    # as we don't have access to simulation object here anymore. Uses self.QM_PC_gradient
                    if exit_after_customexternalforce_update==True:
                        print_if_level(f"OpenMM custom external force updated. Exit requested", self.printlevel,2)
                        #This is used if OpenMM MD is handling forces and dynamics
                        return

                self.MMenergy, self.MMgradient= self.mm_theory.run(current_coords=current_coords, qmatoms=self.qmatoms, Grad=True)
            else:
                print("QM/MM Grad is false")
                self.MMenergy= self.mm_theory.run(current_coords=current_coords, qmatoms=self.qmatoms)
        else:
            self.MMenergy=0
        print_time_rel(CheckpointTime, modulename='MM step', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()


        #Final QM/MM Energy. Possible correction for OpenMM external force term
        self.QM_MM_energy= self.QMenergy+self.MMenergy-self.extforce_energy
        if self.printlevel >= 2:
            blankline()
            if self.embedding.lower() == "elstat":
                print("Note: You are using electrostatic embedding. This means that the QM-energy is actually the polarized QM-energy")
                print("Note: MM energy also contains the QM-MM Lennard-Jones interaction\n")
            energywarning=""
            if self.TruncatedPC is True:
                #if self.TruncatedPCflag is True:
                print("Warning: Truncated PC approximation is active. This means that QM and QM/MM energies are approximate.")
                energywarning="(approximate)"

            print("{:<20} {:>20.12f} {}".format("QM energy: ",self.QMenergy,energywarning))
            print("{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
            print("{:<20} {:>20.12f} {}".format("QM/MM energy: ", self.QM_MM_energy,energywarning))
            blankline()

        #FINAL QM/MM GRADIENT ASSEMBLY
        if Grad is True:
            #If OpenMM external force method then QM/MM gradient is already complete
            #NOTE: Not possible anymore
            if self.openmm_externalforce is True:
                pass
            #    self.QM_MM_gradient = self.MMgradient
            #Otherwise combine
            else:
                #Now assemble full QM/MM gradient
                #print("len(self.QM_PC_gradient):", len(self.QM_PC_gradient))
                #print("len(self.MMgradient):", len(self.MMgradient))
                assert len(self.QM_PC_gradient) == len(self.MMgradient)
                self.QM_MM_gradient=self.QM_PC_gradient+self.MMgradient


            if self.printlevel >=3:
                print("Printlevel >=3: Printing all gradients to disk")
                #print("QM gradient (au/Bohr):")
                #module_coords.print_coords_all(self.QMgradient, self.qmelems, self.qmatoms)
                ash.modules.module_coords.write_coords_all(self.QMgradient_wo_linkatoms, self.qmelems, indices=self.qmatoms, file="QMgradient-without-linkatoms_{}".format(label), description="QM gradient w/o linkatoms {} (au/Bohr):".format(label))

                #Writing QM+Linkatoms gradient
                ash.modules.module_coords.write_coords_all(self.QMgradient, self.qmelems+['L' for i in range(self.num_linkatoms)], indices=self.qmatoms+[0 for i in range(self.num_linkatoms)], file="QMgradient-with-linkatoms_{}".format(label), description="QM gradient with linkatoms {} (au/Bohr):".format(label))

                #blankline()
                #print("PC gradient (au/Bohr):")
                #module_coords.print_coords_all(self.PCgradient, self.mmelems, self.mmatoms)
                ash.modules.module_coords.write_coords_all(self.PCgradient, self.mmelems, indices=self.mmatoms, file="PCgradient_{}".format(label), description="PC gradient {} (au/Bohr):".format(label))
                #blankline()
                #print("QM+PC gradient (au/Bohr):")
                #module_coords.print_coords_all(self.QM_PC_gradient, self.elems, self.allatoms)
                ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM+PCgradient_{}".format(label), description="QM+PC gradient {} (au/Bohr):".format(label))
                #blankline()
                #print("MM gradient (au/Bohr):")
                #module_coords.print_coords_all(self.MMgradient, self.elems, self.allatoms)
                ash.modules.module_coords.write_coords_all(self.MMgradient, self.elems, indices=self.allatoms, file="MMgradient_{}".format(label), description="MM gradient {} (au/Bohr):".format(label))
                #blankline()
                #print("Total QM/MM gradient (au/Bohr):")
                #print("")
                #module_coords.print_coords_all(self.QM_MM_gradient, self.elems,self.allatoms)
                ash.modules.module_coords.write_coords_all(self.QM_MM_gradient, self.elems, indices=self.allatoms, file="QM_MMgradient_{}".format(label), description="QM/MM gradient {} (au/Bohr):".format(label))
            if self.printlevel >= 2:
                print(BC.WARNING,BC.BOLD,"------------ENDING QM/MM MODULE-------------",BC.END)
            print_time_rel(module_init_time, modulename='QM/MM run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_MM_energy, self.QM_MM_gradient
        else:
            print_time_rel(module_init_time, modulename='QM/MM run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_MM_energy


#Micro-iterative QM/MM Optimization
# NOTE: Not ready
#Wrapper around QM/MM run and geometric optimizer for performing microiterative QM/MM opt
# I think this is easiest
# Thiel: https://pubs.acs.org/doi/10.1021/ct600346p
#Look into new: https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.6b00547

def microiter_QM_MM_OPT_v1(theory=None, fragment=None, chargemodel=None, qmregion=None, activeregion=None, bufferregion=None):

    ashexit()
    #1. Calculate single-point QM/MM job and get charges. Maybe get gradient to judge convergence ?
    energy=ash.Singlepoint(theory=theory,fragment=fragment)
    #grab charges
    #update charges
    #2. Change active region so that only MM atoms are in active region
    conv_criteria="something"
    sdf=ash.Optimizer(theory=theory,fragment=fragment, coordsystem='hdlc', maxiter=50, ActiveRegion=False, actatoms=[],
                           convergence_setting=None, conv_criteria=conv_criteria)
    #3. QM/MM single-point with new charges?
    #3b. Or do geometric job until a certain threshold and then do MM again??


#frozen-density micro-iterative QM/MM
def microiter_QM_MM_OPT_v2(theory=None, fragment=None, maxiter=500, qmregion=None, activeregion=None, bufferregion=None,xtbdir=None,xtbmethod='GFN2-xTB'):
    sdf="dsds"
    ashexit()
#xtb instead of charges
# def microiter_QM_MM_OPT_v3(theory=None, fragment=None, maxiter=500, qmregion=None, activeregion=None, bufferregion=None,xtbdir=None,xtbmethod='GFN2-xTB', charge=None, mult=None):
#     ashexit()
#     #Make copy of orig theory
#     orig_theory=copy.deepcopy(theory)
#     # TODO: If BS-spinflipping, use Hsmult instead of regular mul6
#     xtbtheory=ash.xTBTheory(xtbdir=None, charge=charge, mult=mult, xtbmethod=xtbmethod,
#                         runmode='inputfile', numcores=1, printlevel=2)
#     ll_theory=copy.deepcopy(theory)
#     ll_theory.qm_theory=xtbtheory
#     #Convergence criteria
#     loose_conv_criteria = { 'convergence_energy' : 1e-1, 'convergence_grms' : 1e-1, 'convergence_gmax' : 1e-1, 'convergence_drms' : 1e-1,
#                      'convergence_dmax' : 1e-1 }
#     final_conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 3e-4, 'convergence_gmax' : 4.5e-4, 'convergence_drms' : 1.2e-3,
#                         'convergence_dmax' : 1.8e-3 }


#     #Remove QM-region from actregion, optimize everything else.
#     act_original=copy.deepcopy(act)
#     for i in qmatoms:
#         activeregion.remove(i)


#     for macroiteration in range(0,maxiter):
#         oldHLenergy=Hlenergy
#         print("oldHLenergy:", oldHLenergy)
#         #New Macro-iteration step
#         HLenergy,HLgrad=Singlepoint(theory=orig_theory,fragment=fragment,Grad=True, charge=charge, mult=mult)
#         print("HLenergy:", HLenergy)
#         #Check if HLgrad matches convergence critera for gradient?
#         if macroiteration > 0:
#             #Test step acceptance
#             if HLenergy > oldHLenergy:
#                 #Reject step. Reduce step size, use old geo, not sure how
#                 pass
#             if RMS_Hlgrad < final_conv_criteria['convergence_grms'] and MaxHLgrad < final_conv_criteria['convergence_gmax']:
#                 print("Converged.")
#                 return
#             #Make step using Hlgrad

#         LLenergy,LLgrad=Singlepoint(theory=ll_theory,fragment=fragment,Grad=True, charge=charge, mult=mult)
#         #Compare gradient, calculate G0 correction

#         print("Now starting low-level theory QM/MM microtierative optimization")
#         print("activeregion:", activeregion)
#         print("ll_theory qm theory:", ll_theory.qm_theory)
#         bla=geomeTRICOptimizer(theory=ll_theory,fragment=fragment, coordsystem='hdlc', maxiter=200, ActiveRegion=True, actatoms=activeregion,
#                             conv_criteria=loose_conv_criteria, charge=charge, mult=mult)
#         print("Now starting finallevel theory QM/MM microtierative optimization")
#         print("act_original:", act_original)
#         print("orig_theory qm theory:", orig_theory.qm_theory)
#         final=geomeTRICOptimizer(theory=orig_theory,fragment=fragment, coordsystem='hdlc', maxiter=200, ActiveRegion=True, actatoms=act_original,
#                             conv_criteria=final_conv_criteria, charge=charge, mult=mult)
#         print("Micro-iterative QM/MM opt complete !")
#     return final


#This projects the linkatom force onto the respective QM atom and MM atom
def linkatom_force_fix(Qcoord, Mcoord, Lcoord, Qgrad,Mgrad,Lgrad):
    #print("Qcoord:", Qcoord)
    #print("Mcoord:", Mcoord)
    #print("Lcoord:", Lcoord)
    #print("Qgrad:", Qgrad)
    #print("Mgrad:", Mgrad)
    #print("Lgrad:", Lgrad)
    #QM1-L and QM1-MM1 distances
    QLdistance=ash.modules.module_coords.distance(Qcoord,Lcoord)
    #print("QLdistance:", QLdistance)
    MQdistance=ash.modules.module_coords.distance(Mcoord,Qcoord)
    #print("MQdistance:", MQdistance)
    #B and C: a 3x3 arrays
    B=np.zeros([3,3])
    C=np.zeros([3,3])
    for i in range(0,3):
        for j in range(0,3):
            B[i,j]=-1*QLdistance*(Mcoord[i]-Qcoord[i])*(Mcoord[j]-Qcoord[j]) / (MQdistance*MQdistance*MQdistance)
    for i in range(0,3):
        B[i,i] = B[i,i] + QLdistance / MQdistance
    for i in range(0,3):
        for j in range(0,3):
            C[i,j]= -1 * B[i,j]
    for i in range(0,3):
        C[i,i] = C[i,i] + 1.0

    #Multiplying C matrix with Linkatom gradient
    #temp
    g_x=C[0,0]*Lgrad[0]+C[0,1]*Lgrad[1]+C[0,2]*Lgrad[2]
    g_y=C[1,0]*Lgrad[0]+C[1,1]*Lgrad[1]+C[1,2]*Lgrad[2]
    g_z=C[2,0]*Lgrad[0]+C[2,1]*Lgrad[1]+C[2,2]*Lgrad[2]

    #print("g_x:", g_x)
    #print("g_y:", g_y)
    #print("g_z:", g_z)

    #Multiplying B matrix with Linkatom gradient
    gg_x=B[0,0]*Lgrad[0]+B[0,1]*Lgrad[1]+B[0,2]*Lgrad[2]
    gg_y=B[1,0]*Lgrad[0]+B[1,1]*Lgrad[1]+B[1,2]*Lgrad[2]
    gg_z=B[2,0]*Lgrad[0]+B[2,1]*Lgrad[1]+B[2,2]*Lgrad[2]

    #print("gg_x:", gg_x)
    #print("gg_y:", gg_y)
    #print("gg_z:", gg_z)
    #QM atom gradient
    #print("Qgrad before:", Qgrad)
    #print("Lgrad:", Lgrad)
    #print("C: ", C)
    #print("B:", B)
    #Multiply grad by C-diagonal
    #Qgrad[0] = Qgrad[0]*C[0][0]
    #Qgrad[1] = Qgrad[1]*C[1][1]
    #Qgrad[2] = Qgrad[2]*C[2][2]
    Qgrad[0]=Qgrad[0]+g_x
    Qgrad[1]=Qgrad[1]+g_y
    Qgrad[2]=Qgrad[2]+g_z
    #print("Qgrad after:", Qgrad)
    #MM atom gradient
    #print("Mgrad before", Mgrad)
    #Mgrad[0] = Mgrad[0]*B[0][0]
    #Mgrad[1] = Mgrad[1]*B[1][1]
    #Mgrad[2] = Mgrad[2]*B[2][2]
    Mgrad[0]=Mgrad[0]+gg_x
    Mgrad[1]=Mgrad[1]+gg_y
    Mgrad[2]=Mgrad[2]+gg_z
    #print("Mgrad after:", Mgrad)

    return Qgrad,Mgrad

def fullindex_to_qmindex(fullindex,qmatoms):
    qmindex=qmatoms.index(fullindex)
    return qmindex


#Grab resid column from PDB-file and return list of resids
# NOTE: New resid-indices are used to avoid problem of PDB-file having
# repeating sequences of resids, additional chains or segments
def grab_resids_from_pdbfile(pdbfile):
    resids=[] #New list of resid indices, starting from 0
    actual_resids=[] #Actual resid values from PDB-file, used to check if resid has changed
    indexcount=0 #This will be used to define residues
    with open(pdbfile) as f:
        for line in f:
            if 'ATOM' in line or 'HETATM' in line:
                #Based on: https://cupnet.net/pdb-format/
                resid_part=int(line[22:26].replace(" ",""))
                #Very first atom and first residue
                if len(resids) == 0:
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
                #Checking if resid in PDB-file is the same
                elif resid_part == actual_resids[-1]:
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
                # Resid changed, meaning new residue
                else:
                    indexcount+=1
                    #New residue
                    resids.append(indexcount)
                    actual_resids.append(resid_part)

    return resids

#Grab resid column from PSF-file and return list of resids
# NOTE: New resid-indices are used to avoid problem of PSF-file having
# repeating sequences of resids, additional chains or segments
def grab_resids_from_psffile(psffile):
    resids=[] #New list of resid indices, starting from 0
    actual_resids=[] #Actual resid values from PSF-file, used to check if resid has changed
    indexcount=0 #This will be used to define residues
    #resnames=[]
    with open(psffile) as f:
        for line in f:
            if 'REMARKS' in line:
                continue
            if len(line.split()) > 8:
                resname_part=line.split()[3]
                resid_part=int(line.split()[2])
                #resnames.append(resname_part)
                #Very first atom and first residue
                if len(resids) == 0:
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
                #Checking if resid in PDB-file is the same
                elif resid_part == actual_resids[-1]:
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
                # Resid changed, meaning new residue
                else:
                    indexcount+=1
                    #New residue
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
    return resids

#Read atomic charges present in PSF-file. assuming Xplor format
def read_charges_from_psf(file):
    charges=[]
    grab=False
    with open(file) as f:
        for line in f:
            if len(line.split()) == 9:
                if 'REMARKS' not in line:
                    grab=True
            if len(line.split()) < 8:
                grab=False
            if 'NBOND' in line:
                return charges
            if grab is True:
                charge=float(line.split()[6])
                charges.append(charge)
    return charges

#Define active region based on radius from an origin atom.
#Requires fragment (for coordinates) and residue information from either:
# 1. resids list inside OpenMMTheory object
# 2. residues taken from PDB-file
# 3. residues taken from PSF-file

def actregiondefine(pdbfile=None, mmtheory=None, psffile=None, fragment=None, radius=None, originatom=None):

    print_line_with_mainheader("ActregionDefine")

    #Checking if proper information has been provided
    if radius == None or originatom == None:
        print("actregiondefine requires radius and originatom keyword arguments")
        ashexit()
    if pdbfile == None and fragment == None:
        print("actregiondefine requires either fragment or pdbfile arguments (for coordinates)")
        ashexit()
    if pdbfile == None and mmtheory == None and psffile == None:
        print("actregiondefine requires either pdbfile, psffile or mmtheory arguments (for residue topology information)")
        ashexit()

    #Creating fragment from pdbfile
    if fragment == None:
        print("No ASH fragment provided. Creating ASH fragment from PDBfile")
        fragment = Fragment(pdbfile=pdbfile, printlevel=1)

    print("Radius:", radius)
    print("Origin atom: {} ({})".format(originatom,fragment.elems[originatom]))
    print("Will find all atoms within {} Å from atom: {} ({})".format(radius,originatom,fragment.elems[originatom]))
    print("Will select all whole residues within region and export list")
    if mmtheory != None:
        if mmtheory.__class__.__name__ == "NonBondedTheory":
            print("MMtheory: NonBondedTheory currently not supported.")
            ashexit()
        if not mmtheory.resids :
            print(BC.FAIL,"mmtheory.resids list is empty! Something wrong with OpenMMTheory setup. Exiting",BC.END)
            ashexit()
        #Defining list of residue from OpenMMTheory object
        resids=mmtheory.resids
    elif psffile != None:
        print("PSF-file provided. Using residue information")
        resids = grab_resids_from_psffile(psffile)
    else:
        print("PDB-file provided. Using residue information")
        #No mmtheory but PDB file should have been provided
        #Defining resids list from PDB-file
        #NOTE: Call grab_resids_from_pdbfile
        resids = grab_resids_from_pdbfile(pdbfile)

    origincoords=fragment.coords[originatom]
    print("Origin-atom coordinates:", origincoords)
    #print("resids:", resids)
    act_indices=[]
    #print("resids:", resids)
    for index,allc in enumerate(fragment.coords):
        #print("index:", index)
        #print("allc:", allc)
        dist=ash.modules.module_coords.distance(origincoords,allc)
        if dist < radius:
            #Get residue ID for this atom index
            resid_value=resids[index]
            #Get all residue members (atom indices)
            resid_members = [i for i, x in enumerate(resids) if x==resid_value]
            #Adding to act_indices list unless already added
            for k in resid_members:
                if k not in act_indices:
                    act_indices.append(k)

    #Only unique and sorting:
    print("act_indices:", act_indices)
    act_indices = np.unique(act_indices).tolist()

    #Print indices to output
    #Print indices to disk as file
    writelisttofile(act_indices, "active_atoms")
    #Print information on how to use
    print("Active region size:", len(act_indices))
    print("Active-region indices written to file: active_atoms")
    print("The active_atoms list  can be read-into Python script like this:	 actatoms = read_intlist_from_file(\"active_atoms\")")
    #Print XYZ file with active region shown
    ash.modules.module_coords.write_XYZ_for_atoms(fragment.coords,fragment.elems, act_indices, "ActiveRegion")
    print("Wrote Active region XYZfile: ActiveRegion.xyz  (inspect with visualization program)")
    return act_indices


# General QM-PC gradient calculation
def General_QM_PC_gradient(qm_coords,qm_nuc_charges,mol,mm_coords,mm_charges,dm):
    print("not ready")
    exit()
    if dm.shape[0] == 2:
        dmf = dm[0] + dm[1] #unrestricted
    else:
        dmf=dm
    # The interaction between QM atoms and MM particles
    # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
    #qm_coords = mol.atom_coords()
    #qm_charges = mol.atom_charges()
    dr = qm_coords[:,None,:] - mm_coords
    r = np.linalg.norm(dr, axis=2)
    g = np.einsum('r,R,rRx,rR->Rx', qm_nuc_charges, mm_charges, dr, r**-3)
    # The interaction between electron density and MM particles
    # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
    #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
    for i, q in enumerate(mm_charges):
        with mol.with_rinv_origin(mm_coords[i]):
            v = mol.intor('int1e_iprinv')
        f =(np.einsum('ij,xji->x', dmf, v) +
            np.einsum('ij,xij->x', dmf, v.conj())) * -q
        g[i] += f
    return g


# This projects the linkatom force onto the respective QM atom and MM atom
def linkatom_force_contribution(Qcoord, Mcoord, Lcoord, Lgrad):
    print("Qcoord:", Qcoord)
    print("Mcoord:", Mcoord)
    print("Lcoord:", Lcoord)
    print("Lgrad:", Lgrad)
    # QM1-L and QM1-MM1 distances
    QLdistance=ash.modules.module_coords.distance(Qcoord,Lcoord)
    print("QLdistance:", QLdistance)
    MQdistance=ash.modules.module_coords.distance(Mcoord,Qcoord)
    print("MQdistance:", MQdistance)
    # B and C: a 3x3 arrays
    B=np.zeros([3,3])
    C=np.zeros([3,3])
    for i in range(0,3):
        for j in range(0,3):
            B[i,j]=-1*QLdistance*(Mcoord[i]-Qcoord[i])*(Mcoord[j]-Qcoord[j]) / (MQdistance*MQdistance*MQdistance)
    for i in range(0,3):
        B[i,i] = B[i,i] + QLdistance / MQdistance
    for i in range(0,3):
        for j in range(0,3):
            C[i,j]= -1 * B[i,j]
    for i in range(0,3):
        C[i,i] = C[i,i] + 1.0

    # Multiplying C matrix with Linkatom gradient
    # temp
    g_x=C[0,0]*Lgrad[0]+C[0,1]*Lgrad[1]+C[0,2]*Lgrad[2]
    g_y=C[1,0]*Lgrad[0]+C[1,1]*Lgrad[1]+C[1,2]*Lgrad[2]
    g_z=C[2,0]*Lgrad[0]+C[2,1]*Lgrad[1]+C[2,2]*Lgrad[2]

    print("g_x:", g_x)
    print("g_y:", g_y)
    print("g_z:", g_z)

    # Multiplying B matrix with Linkatom gradient
    gg_x=B[0,0]*Lgrad[0]+B[0,1]*Lgrad[1]+B[0,2]*Lgrad[2]
    gg_y=B[1,0]*Lgrad[0]+B[1,1]*Lgrad[1]+B[1,2]*Lgrad[2]
    gg_z=B[2,0]*Lgrad[0]+B[2,1]*Lgrad[1]+B[2,2]*Lgrad[2]

    print("gg_x:", gg_x)
    print("gg_y:", gg_y)
    print("gg_z:", gg_z)
    # Return QM1_gradient and MM1_gradient contribution (to be added)
    return [g_x,g_y,g_z],[gg_x,gg_y,gg_z]

def linkatom_force_chainrule(Qcoord, Mcoord, Lcoord, Lgrad):
    print("Qcoord:", Qcoord)
    print("Mcoord:", Mcoord)
    print("Lcoord:", Lcoord)
    print("Lgrad:", Lgrad)
    # QM1-L and QM1-MM1 distances
    print("ang2bohr:", ash.constants.ang2bohr)
    QLdistance=ash.modules.module_coords.distance(Qcoord,Lcoord)*ash.constants.ang2bohr

    vec = (Mcoord - Qcoord)*ash.constants.ang2bohr
    print("vec:", vec)
    R2 = vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]
    oneR = 1.0 / math.sqrt(R2)
    print("oneR:", oneR)
    lnk_dis_oneR = QLdistance*oneR
    print("lnk_dis_oneR:", lnk_dis_oneR)
    vec = vec*oneR
    print("vec:", vec)
    print("Lgrad[0]:", Lgrad[0])
    print("Lgrad[0]*(-1)", Lgrad[0]*(-1))
    print("Lgrad[0]*(-1)*vec[0]:", Lgrad[0]*(-1)*vec[0])
    print("Lgrad[1]:", Lgrad[1])
    print("Lgrad[1]*(-1)", Lgrad[1]*(-1))
    print("Lgrad[1]*(-1)*vec[1]:", Lgrad[1]*(-1)*vec[1])
    print("Lgrad[2]:", Lgrad[2])
    print("Lgrad[2]*(-1)", Lgrad[2]*(-1))
    print("Lgrad[2]*(-1)*vec[2]:", Lgrad[2]*(-1)*vec[2])
    dotprod = Lgrad[0]*(-1)*vec[0]+Lgrad[1]*(-1)*vec[1]+Lgrad[2]*(-1)*vec[2]
    print("dotprod:", dotprod)
    forcemod=np.zeros(3)
    print("forcemod:", forcemod)
    forcemod[0] = lnk_dis_oneR*(-1*Lgrad[0]-(dotprod*vec[0])) 
    forcemod[1] = lnk_dis_oneR*(-1*Lgrad[1]-(dotprod*vec[1])) 
    forcemod[2] = lnk_dis_oneR*(-1*Lgrad[2]-(dotprod*vec[2])) 

    print("forcemod:", forcemod)

    return forcemod