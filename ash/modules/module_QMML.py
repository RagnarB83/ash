import copy
import time
import numpy as np
import math

import ash.modules.module_coords
from ash.modules.module_coords import Fragment, write_pdbfile
from ash.functions.functions_general import ashexit, BC, blankline, listdiff, print_time_rel,printdebug,print_line_with_mainheader,writelisttofile,print_if_level
import ash.settings_ash
from ash.modules.module_QMMM import linkatom_force_adv,linkatom_force_chainrule,linkatom_force_lever,fullindex_to_qmindex

# Additive QM/ML theory object.
# Required at init: qm_theory and qmatoms and fragment

class QMMLTheory:
    def __init__(self, qm_theory=None, qmatoms=None, fragment=None, ml_theory=None, charges=None,
                embedding="elstat", printlevel=2, numcores=1, excludeboundaryatomlist=None,
                unusualboundary=False, TruncatedPC=False, TruncPCRadius=55, TruncatedPC_recalc_iter=50,
                qm_charge=None, qm_mult=None, chargeboundary_method="shift", 
                dipole_correction=True, linkatom_method='simple', linkatom_simple_distance=None,
                linkatom_forceproj_method="adv", linkatom_ratio=0.723, linkatom_type='H',
                update_QMregion_charges=False):

        module_init_time = time.time()
        timeA = time.time()
        print_line_with_mainheader("QM/ML Theory")

        # Check for necessary keywords
        if qm_theory is None or qmatoms is None:
            print("Error: QMMLTheory requires defining: qm_theory, qmatoms, fragment")
            ashexit()
        # If fragment object has not been defined
        if fragment is None:
            print("fragment= keyword has not been defined for QM/ML. Exiting")
            ashexit()

        # Defining charge/mult of QM-region
        self.qm_charge = qm_charge
        self.qm_mult = qm_mult

        # Indicate that this is a hybrid QM/ML type theory
        self.theorytype = "QM/ML"
        self.theorynamelabel="QMMLTheory"

        # Subtractive corrections that might be defined later on
        # Added due to pbcmm-elstat
        self.subtractive_correction_E =0.0
        self.subtractive_correction_G = np.zeros((len(fragment.coords), 3))

        # update_QMregion_charges
        # After each QM-region calculation, the charges of the QM-region may have been calculated
        # These charges can be used to update the charges of the whole system. Only used for mechanical embedding
        self.update_QMregion_charges=update_QMregion_charges

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
            linkatom_forceproj_method="none"

        # Counter for how often QMLMTheory.run is called
        self.runcalls = 0

        # Theory level definitions
        self.printlevel=printlevel
        self.qm_theory=qm_theory
        self.qm_theory_name = self.qm_theory.__class__.__name__
        self.ml_theory=ml_theory
        self.ml_theory_name = self.ml_theory.__class__.__name__

        print("QM-theory:", self.qm_theory_name)
        print("ML-theory:", self.ml_theory_name)

        self.fragment=fragment
        self.coords=fragment.coords
        self.elems=fragment.elems
        self.connectivity=fragment.connectivity

        self.excludeboundaryatomlist=excludeboundaryatomlist
        self.unusualboundary = unusualboundary

        # Region definitions
        self.allatoms=list(range(0,len(self.elems)))
        print("All atoms in fragment:", len(self.allatoms))
        self.num_allatoms=len(self.allatoms)

        # Sorting qmatoms list making sure only unique values are taken
        self.qmatoms = sorted(list(set(qmatoms)))

        # All-atom Bool-array for whether atom-index is a QM-atom index or not
        # Used by make_QM_PC_gradient
        self.xatom_mask = np.isin(self.allatoms, self.qmatoms)
        self.sum_xatom_mask = np.sum(self.xatom_mask)

        if len(self.qmatoms) == 0:
            print("Error: List of qmatoms provided is empty. This is not allowed.")
            ashexit()
        self.mlatoms = np.setdiff1d(self.allatoms, self.qmatoms)

        # print("List of all atoms:", self.allatoms)
        print("QM region ({} atoms): {}".format(len(self.qmatoms),self.qmatoms))
        print("ML region ({} atoms)".format(len(self.mlatoms)))

        # Setting QM/ML qmatoms in QMtheory also (used for Spin-flipping currently)
        self.qm_theory.qmatoms=self.qmatoms

        # numcores-setting in QMMLTheory takes precedent
        if numcores != 1:
            self.numcores=numcores
        # If QMtheory numcores was set (and QMMLTHeory not)
        elif self.qm_theory.numcores != 1:
            self.numcores=self.qm_theory.numcores
        # Default 1 proc
        else:
            self.numcores=1
        print("QM/ML object selected to use {} cores".format(self.numcores))

        # Embedding type: mechanical, electrostatic etc.
        self.embedding=embedding
        # Charge-boundary method
        self.chargeboundary_method=chargeboundary_method  # Options: 'chargeshift', 'rcd'

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

        # if atomcharges are not passed to QMMLTheory object, get them from MLtheory (that should have been defined then)
        if charges is None:
            print("No atomcharges list passed to QMMLTheory object yet")
            self.charges=[]
        else:
            print("Reading in charges")
            if len(charges) != len(fragment.atomlist):
                print(BC.FAIL,"Number of charges not matching number of fragment atoms. Exiting.",BC.END)
                ashexit()
            self.charges=charges

        #TODO
        #if len(self.charges) == 0:
        #    print("No charges present in QM/ML object. Exiting...")
        #    ashexit()

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
            print("Truncated PC approximation in QM/ML is active.")
            print("TruncPCRadius:", self.TruncPCRadius)
            print("TruncPC Recalculation iteration:", self.TruncatedPC_recalc_iter)

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
            conn_tolerance, excludeboundaryatomlist=self.excludeboundaryatomlist, unusualboundary=self.unusualboundary)
        if len(self.boundaryatoms) >0:
            print("Found covalent QM-ML boundary. Linkatoms option set to True")
            print("Boundaryatoms (QM:ML pairs):", self.boundaryatoms)
            print("Note: used connectivity settings, scale={} and tol={} to determine boundary.".format(conn_scale,conn_tolerance))
            self.linkatoms = True
            # Get ML boundary information. Stored as self.MLboundarydict
            self.get_MLboundary(conn_scale,conn_tolerance)
        else:
            print("No covalent QM-ML boundary. Linkatoms and dipole_correction options set to False")
            self.linkatoms=False
            self.dipole_correction=False

        ########################
        # CHANGE CHARGES
        ########################
        # Keeping self.charges as originally defined.
        # Setting QM charges to 0 since electrostatic embedding
        # and Charge-shift QM-ML boundary

        # Zero QM charges for electrostatic embedding
        if self.embedding.lower() == "elstat":
            print("Charges of QM atoms set to 0 (since Electrostatic Embedding):")
            self.ZeroQMCharges() #Modifies self.charges_qmregionzeroed
            # print("length of self.charges_qmregionzeroed :", len(self.charges_qmregionzeroed))
            # TODO: make sure this works for OpenMM and for NonBondedTheory

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
                        print("ML atom {} ({}) charge: {}".format(i, self.elems[i], self.charges_qmregionzeroed[i]))
        blankline()
        print_time_rel(module_init_time, modulename='QM/ML object creation', currprintlevel=self.printlevel, moduleindex=3)


    # From QM1:ML1 boundary dict, get MM1:MMx boundary dict (atoms connected to ML1)
    def get_MLboundary(self,scale,tol):
        timeA=time.time()
        # if boundarydict is not empty we need to zero ML1 charge and distribute charge from ML1 atom to ML2,ML3,ML4
        #Creating dictionary for each ML1 atom and its connected atoms: ML2-4
        self.MLboundarydict={}
        for (QM1atom,ML1atom) in self.boundaryatoms.items():
            connatoms = ash.modules.module_coords.get_connected_atoms(self.coords, self.elems, scale,tol, ML1atom)
            #Deleting QM-atom from connatoms list
            connatoms.remove(QM1atom)
            self.MLboundarydict[ML1atom] = connatoms


        # Used by ShiftMMCharges
        self.MLboundary_indices = list(self.MLboundarydict.keys())
        print("self.MLboundary_indices:", self.MLboundary_indices)
        self.MLboundary_counts = np.array([len(self.MLboundarydict[i]) for i in self.MLboundary_indices])

        print("")
        print("ML boundary (ML1:MLx pairs):", self.MLboundarydict)
        print_time_rel(timeA, modulename="get_MLboundary")
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

    def RCD_shifting_prep(self, charges_qmregionzeroed):
        timeA=time.time()
        if self.printlevel > 1:
            print("Shifting ML charges at QM/ML boundary by RCD.")
        # Convert lists to NumPy arrays for faster computations
        pointcharges = np.array(charges_qmregionzeroed)
        self.charges = np.array(self.charges)
        # Extract charges for ML boundary atoms
        ML1_charges = self.charges[self.MLboundary_indices]
        # Set charges of ML boundary atoms to 0
        pointcharges[self.MLboundary_indices] = 0.0
        # Calculate charge fractions to distribute
        ML1charge_fract = ML1_charges / self.MLboundary_counts

        # Only keep pointcharges for PC region
        pointcharges=[pointcharges[x] for x in self.mlatoms]

        # Distribute charge fractions to neighboring MM atoms
        RCD_additional_charges=[]
        for MM1index, MM2indices, fract in zip(self.MLboundarydict.keys(), self.MLboundarydict.values(), ML1charge_fract):
            newfract = fract*2 #q0*2
            # Looping over ML2 atoms
            for i in ML2indices:
                # RC/RCD: Instead of adding the M1 charge to the M2 atoms we create new RC/RCD sites
                pointcharges = np.append(pointcharges,newfract)
                RCD_additional_charges.append(newfract)
                # RCD: Reduce the ML2 charge by q0
                pointcharges[i] -= fract
            # print("RCD-modified pointcharges:", pointcharges)
        self.chargeshifting_done=True

        print_time_rel(timeA, modulename="RCD_shifting_prep", currprintlevel=self.printlevel, currthreshold=1)
        return pointcharges, RCD_additional_charges

    def RCD_shifting_update(self, used_mlcoords, fullcoords):
        timeA = time.time()
        if self.printlevel > 1:
            print("Adding updated RCD charges at QM/ML boundary by RCD.")

        # Distribute charge fractions to neighboring ML atoms
        for ML1index, ML2indices in zip(self.MLboundarydict.keys(), self.MLboundarydict.values()):
            # Looping over MM2 atoms
            for i in ML2indices:
                # Add new RCD sites to pointchargecoords and pointcharges
                newsite = (fullcoords[i] + fullcoords[ML1index])/2
                pointchargecoords = np.append(used_mlcoords, [newsite], axis=0)
            # print("RCD-modified pointchargecoords:", pointchargecoords)

        print_time_rel(timeA, modulename="RCD_shifting_update", currprintlevel=self.printlevel, currthreshold=1)
        return pointchargecoords

    def ShiftMLCharges(self):
        if self.chargeshifting_done is False:
            self.ShiftMLCharges_new2()
        else:
            print("Charge shifting already done. Using previous charges")
    #TODO: Delete old version below
    def ShiftMLCharges_old(self):
        timeA=time.time()
        if self.printlevel > 1:
            print("Shifting ML charges at QM-ML boundary.")
        # print("len self.charges_qmregionzeroed: ", len(self.charges_qmregionzeroed))
        # print("len self.charges: ", len(self.charges))

        # Create self.pointcharges list
        self.pointcharges=copy.copy(self.charges_qmregionzeroed)

        # Looping over charges and setting QM/ML1 atoms to zero and shifting charge to neighbouring atoms
        for i, c in enumerate(self.pointcharges):

            # If index corresponds to MLatom at boundary, set charge to 0 (charge-shifting
            if i in self.MLboundarydict.keys():
                ML1charge = self.charges[i]
                # print("ML1atom charge: ", ML1charge)
                self.pointcharges[i] = 0.0
                # ML1 charge fraction to be divided onto the other ML atoms
                ML1charge_fract = ML1charge / len(self.MLboundarydict[i])
                # print("ML1charge_fract :", ML1charge_fract)

                # Putting the fractional charge on each ML2 atom
                for MLx in self.MLboundarydict[i]:
                    self.pointcharges[MLx] += ML1charge_fract
        self.chargeshifting_done=True
        print_time_rel(timeA, modulename="ShiftMLCharges-old", currprintlevel=self.printlevel, currthreshold=1)

    def ShiftMLCharges_new(self):
        timeA=time.time()
        if self.printlevel > 1:
            print("new. Shifting ML charges at QM-ML boundary.")
        # Convert lists to NumPy arrays for faster computations
        self.pointcharges = np.array(self.charges_qmregionzeroed)
        for i, c in enumerate(self.pointcharges):
            if i in self.MLboundarydict:
                ML1charge = self.charges[i]
                self.pointcharges[i] = 0.0
                # Calculate ML1 charge fraction to be divided onto the other ML atoms
                ML1charge_fract = ML1charge / len(self.MLboundarydict[i])
                # Distribute charge fraction to neighboring ML atoms
                self.pointcharges[list(self.MLboundarydict[i])] += ML1charge_fract
        self.chargeshifting_done=True
        print_time_rel(timeA, modulename="ShiftMLCharges-new", currprintlevel=self.printlevel, currthreshold=1)
        return
    def ShiftMLCharges_new2(self):
        timeA=time.time()
        if self.printlevel > 1:
            print("new. Shifting ML charges at QM-ML boundary.")

        # Convert lists to NumPy arrays for faster computations
        print_time_rel(timeA, modulename="x0", currprintlevel=self.printlevel, currthreshold=1)
        self.pointcharges = np.array(self.charges_qmregionzeroed)
        self.charges=np.array(self.charges)

        print_time_rel(timeA, modulename="x1", currprintlevel=self.printlevel, currthreshold=1)
        # Extract charges for ML boundary atoms
        ML1_charges = self.charges[self.MLboundary_indices]
        # Set charges of ML boundary atoms to 0
        self.pointcharges[self.MLboundary_indices] = 0.0

        # Calculate charge fractions to distribute
        ML1charge_fract = ML1_charges / self.MLboundary_counts

        # Distribute charge fractions to neighboring ML atoms
        for indices, fract in zip(self.MLboundarydict.values(), ML1charge_fract):
            self.pointcharges[[indices]] += fract

        self.chargeshifting_done=True
        print_time_rel(timeA, modulename="ShiftMLCharges-new2", currprintlevel=self.printlevel, currthreshold=1)
        return

    # Create dipole charge (twice) for each ML2 atom that gets fraction of ML1 charge
    def get_dipole_charge(self,delq,direction,ml1index,ml2index,current_coords):
        # oldMM_distance = ash.modules.module_coords.distance_between_atoms(fragment=self.fragment,
        #                                                               atoms=[mm1index, mm2index])
        # Coordinates and distance
        ml1coords=np.array(current_coords[ml1index])
        ml2coords=np.array(current_coords[ml2index])
        ML_distance = ash.modules.module_coords.distance(ml1coords,ml2coords) # Distance between ML1 and ML2

        SHIFT=0.15
        # Normalize vector
        def vnorm(p1):
            r = math.sqrt((p1[0]*p1[0])+(p1[1]*p1[1])+(p1[2]*p1[2]))
            v1=np.array([p1[0] / r, p1[1] / r, p1[2] /r])
            return v1
        diffvector = ml2coords-ml1coords
        normdiffvector = vnorm(diffvector)

        # Dipole
        d = delq*2.5
        # Charge (abs value)
        q0 = 0.5 * d / SHIFT
        # Actual shift
        shift = direction * SHIFT * ( ML_distance / 2.5 )
        # Position
        pos = ml2coords+np.array((shift*normdiffvector))
        # Returning charge with sign based on direction and position
        # Return coords as regular list
        return -q0*direction,list(pos)

    def SetDipoleCharges(self,current_coords):
        checkpoint=time.time()
        if self.printlevel > 1:
            print("Adding extra charges to preserve dipole moment for charge-shifting")
            print("MLboundarydict:", self.MLboundarydict)
        # Adding 2 dipole pointcharges for each ML2 atom
        self.dipole_charges = []
        self.dipole_coords = []


        for ML1,MLx in self.MLboundarydict.items():
            # Getting original ML1 charge (before set to 0)
            ML1charge = self.charges[ML1]
            ML1charge_fract=ML1charge/len(MLx)

            for ML in MLx:
                q_d1, pos_d1 = self.get_dipole_charge(ML1charge_fract,1,ML1,ML,current_coords)
                q_d2, pos_d2 = self.get_dipole_charge(ML1charge_fract,-1,ML1,ML,current_coords)
                self.dipole_charges.append(q_d1)
                self.dipole_charges.append(q_d2)
                self.dipole_coords.append(pos_d1)
                self.dipole_coords.append(pos_d2)
        print_time_rel(checkpoint, modulename='SetDipoleCharges', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)

    # Reasonably efficient version (this dominates QM/ML gradient prepare)
    #def make_QM_PC_gradient_old(self):
    #    self.QM_PC_gradient = np.zeros((len(self.allatoms), 3))
    #    qmatom_indices = np.where(np.isin(self.allatoms, self.qmatoms))[0]
    #    pcatom_indices = np.where(~np.isin(self.allatoms, self.qmatoms))[0] # ~ is NOT operator in numpy
    #    self.QM_PC_gradient[qmatom_indices] = self.QMgradient_wo_linkatoms[:len(qmatom_indices)]
    #    self.QM_PC_gradient[pcatom_indices] = self.PCgradient[:len(pcatom_indices)]
    #    return

    # Faster version. Also, uses precalculated mask.
    def make_QM_PC_gradient(self):
        self.QM_PC_gradient[self.xatom_mask] = self.QMgradient_wo_linkatoms
        self.QM_PC_gradient[~self.xatom_mask] = self.PCgradient[:self.num_allatoms - self.sum_xatom_mask]
        return
    # make_QM_PC_gradient=make_QM_PC_gradient_optimized
    # TruncatedPCfunction control flow for pointcharge field passed to QM program
    def TruncatedPCfunction(self, used_qmcoords):
        self.TruncatedPCcalls+=1
        print("TruncatedPC approximation!")
        if self.TruncatedPCcalls == 1 or self.TruncatedPCcalls % self.TruncatedPC_recalc_iter == 0:
            self.TruncatedPC_recalc_flag=True
            # print("This is first QM/ML run. Will calculate Full-Trunc correction in this step")
            print(f"This is QM/ML run no. {self.TruncatedPCcalls}.  Will calculate Full-Trunc correction in this step")
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
            print("This is QM/ML run no. {}. Using approximate truncated PC field: {} charges".format(self.TruncatedPCcalls,len(self.truncated_PC_region_indices)))
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
        print(f"Setting new numcores {numcores}for QMtheory and MLtheory")
        self.qm_theory.set_numcores(numcores)
        self.ml_theory.set_numcores(numcores)
    # Method to grab dipole moment from outputfile (assumes run has been executed)
    def get_dipole_moment(self):
        try:
            print("Grabbing dipole moment from QM-part of QM/ML theory.")
            dipole = self.qm_theory.get_dipole_moment()
        except:
            print("Error: Could not grab dipole moment from QM-part of QM/ML theory.")
        return dipole
    # Method to polarizability from outputfile (assumes run has been executed)
    def get_polarizability_tensor(self):
        try:
            print("Grabbing polarizability from QM-part of QM/ML theory.")
            polarizability = self.qm_theory.get_polarizability_tensor()
        except:
            print("Error: Could not grab polarizability from QM-part of QM/ML theory.")
        return polarizability
    # General run
    def run(self, current_coords=None, elems=None, Grad=False, numcores=1, exit_after_customexternalforce_update=False, label=None, charge=None, mult=None,
            current_MM_coords=None, MMcharges=None, qm_elems=None, PC=None, mm_elems=None):

        if self.printlevel >= 2:
            print(BC.WARNING, BC.BOLD, "------------RUNNING QM/ML MODULE-------------", BC.END)
            print("QM Module:", self.qm_theory_name)
            print("ML Module:", self.ml_theory_name)

        # OPTION: QM-region charge/mult from QMMLTheory definition
        # If qm_charge/qm_mult defined then we use. Otherwise charge/mult may have been defined by jobtype-function and passed on via run
        if self.qm_charge is not None:
            if self.printlevel > 1:
                print("Charge provided from QMMLTheory object: ", self.qm_charge)
            charge = self.qm_charge
        if self.qm_mult is not None:
            if self.printlevel > 1:
                print("Mult provided from QMMLTheory object: ", self.qm_mult)
            mult = self.qm_mult

        # Checking if charge and mult has been provided. Exit if not.
        if charge is None or mult is None:
            print(BC.FAIL, "Error. charge and mult has not been defined for QMMLTheory.run method", BC.END)
            ashexit()

        if self.printlevel >1 :
            print("QM-region Charge: {} Mult: {}".format(charge,mult))

        if self.embedding.lower() == "mech":
            return self.mech_run(current_coords=current_coords, elems=elems, Grad=Grad, numcores=numcores, exit_after_customexternalforce_update=exit_after_customexternalforce_update,
                label=label, charge=charge, mult=mult)
        elif self.embedding.lower() == "elstat":
            return self.elstat_run(current_coords=current_coords, elems=elems, Grad=Grad, numcores=numcores, exit_after_customexternalforce_update=exit_after_customexternalforce_update,
                label=label, charge=charge, mult=mult)
        elif self.embedding.lower() == "pbcmm-elstat":
            # Things should be the same except QM-charges have not been zeroed in MM-program
            # MM-program thus double-counts (SR QM-QM and SR QM-MM) and we need subtractive corrections
            return self.elstat_run(current_coords=current_coords, elems=elems, Grad=Grad, numcores=numcores, exit_after_customexternalforce_update=exit_after_customexternalforce_update,
                label=label, charge=charge, mult=mult)
        else:
            print("Unknown embedding. Exiting")
            ashexit()

    # Mechanical embedding run
    def mech_run(self, current_coords=None, elems=None, Grad=False, numcores=1, exit_after_customexternalforce_update=False, label=None, charge=None, mult=None):
        module_init_time=time.time()
        CheckpointTime = time.time()
        if self.printlevel >= 2:
            print("Embedding: Mechanical")

        #############################################
        # If this is first run then do QM/ML runprep
        # Only do once to avoid cost in each step
        #############################################
        if self.runcalls == 0:
            print("First QMMLTheory run. Running runprep")
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

        # If numcores was set when calling QMMLTheory.run then using, otherwise use self.numcores
        if numcores == 1:
            numcores = self.numcores

        if self.printlevel >= 2:
            print("Running QM/ML object with {} cores available".format(numcores))

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
            # Calling QM theory, providing current QM and MM coordinates.
            if Grad is True:
                QMenergy, QMgradient = self.qm_theory.run(current_coords=used_qmcoords, qm_elems=self.current_qmelems, Grad=True, 
                                                          PC=False, numcores=numcores, charge=charge, mult=mult)
            else:
                QMenergy = self.qm_theory.run(current_coords=used_qmcoords,qm_elems=self.current_qmelems, Grad=False, 
                                              PC=False, numcores=numcores, charge=charge, mult=mult)

        print_time_rel(CheckpointTime, modulename='QM step', moduleindex=2,currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()

        ############################
        # Update QM-region charges
        ############################

        if self.update_QMregion_charges:
            print("update_QMregion_charges is True")
            print("Will try to find charges attribute in QM-object")
            try:
                newqmcharges = self.qm_theory.charges
            except:
                print("error: found no charges attribute of QMTheory object. update_QMregion_charges can not be used")
                ashexit()
            #Removing linkatoms
            newqmcharges = newqmcharges[0:-self.num_linkatoms]
            for i, index in enumerate(self.qmatoms):
                self.charges[index] = newqmcharges[i]
            print("Updating charges of QM-region in MMTheory object")
            self.mm_theory.update_charges(self.qmatoms,[i for i in newqmcharges])
        print("Defined charges of QM-region:")
        for i in self.qmatoms:
            print(f"QM atom {i} has charge : {self.charges[i]}")

        ##################################################################################
        # QM/ML gradient: Initializing and then adding QM gradient, linkatom gradient
        ##################################################################################

        self.QMenergy = QMenergy

        # Initializing QM/ML gradient
        self.QM_ML_gradient = np.zeros((len(current_coords), 3))
        if Grad:
            Grad_prep_CheckpointTime = time.time()
            # Defining QMgradient without linkatoms if present
            if self.linkatoms is True:
                self.QMgradient = QMgradient
                self.QMgradient_wo_linkatoms=QMgradient[0:-self.num_linkatoms] #remove linkatoms
            else:
                self.QMgradient = QMgradient
                self.QMgradient_wo_linkatoms=QMgradient

            # Adding QM gradient (without linkatoms) to QM_ML_gradient
            self.QM_ML_gradient[self.qmatoms] += self.QMgradient_wo_linkatoms

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
                    fullatomindex_ml = pair[1]
                    Mcoord = current_coords[fullatomindex_ml]
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
                    #print("QM1grad contrib:", QM1grad_contrib)
                    #print("L1grad contrib:", ML1grad_contrib)
                    # Updating full QM_ML_gradient
                    self.QM_ML_gradient[fullatomindex_qm] += QM1grad_contrib
                    self.QM_ML_gradient[fullatomindex_ml] += ML1grad_contrib

            # Defining QM_PC_gradient for simplicity (used by OpenMM_MD)
            self.QM_PC_gradient = self.QM_ML_gradient


            print_time_rel(CheckpointTime, modulename='linkatomgrad prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            print_time_rel(Grad_prep_CheckpointTime, modulename='QM/ML gradient prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            CheckpointTime = time.time()
        else:
            # No Grad
            self.QMenergy = QMenergy

        ################
        # ML THEORY
        ################
        self.MLenergy, self.MLgradient = self.ml_theory.run(current_coords=current_coords,
                                                            charges=self.charges, connectivity=self.connectivity,
                                                            qmatoms=self.qmatoms, actatoms=self.actatoms)

        print_time_rel(CheckpointTime, modulename='ML step', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()

        if Grad:
            # Now assemble full QM/ML gradient by adding ML gradient
            assert len(self.QM_ML_gradient) == len(self.MLgradient)
            self.QM_ML_gradient = self.QM_ML_gradient + self.MLgradient

        # Final QM/ML Energy
        self.QM_ML_energy = self.QMenergy+self.MLenergy-self.subtractive_correction_E

        # Final QM/ML Gradient
        # Possible subtractive correction
        self.QM_ML_gradient -= self.subtractive_correction_G

        if self.printlevel >= 2:
            blankline()
            print("{:<20} {:>20.12f}".format("QM energy: ", self.QMenergy))
            print("{:<20} {:>20.12f}".format("ML energy: ", self.MLenergy))
            print("{:<20} {:>20.12f}".format("Subtractive correction energy: ", self.subtractive_correction_E))
            print("{:<20} {:>20.12f}".format("QM/ML energy: ", self.QM_ML_energy))
            blankline()

        # FINAL QM/ML GRADIENT ASSEMBLY and return
        if Grad is True:
            if self.printlevel >=3:
                print("Printlevel >=3: Printing all gradients to disk")
                # Writing QM gradient only
                ash.modules.module_coords.write_coords_all(self.QMgradient_wo_linkatoms, self.qmelems, indices=self.qmatoms, file="QMgradient-without-linkatoms_{}".format(label), description="QM gradient w/o linkatoms {} (au/Bohr):".format(label))
                # Writing QM+Linkatoms gradient
                ash.modules.module_coords.write_coords_all(self.MLgradient, self.elems, indices=self.allatoms, file="MLgradient_{}".format(label), description="ML gradient {} (au/Bohr):".format(label))
                # Writing full QM/ML gradient
                ash.modules.module_coords.write_coords_all(self.QM_ML_gradient, self.elems, indices=self.allatoms, file="QM_MLgradient_{}".format(label), description="QM/ML gradient {} (au/Bohr):".format(label))
            if self.printlevel >= 2:
                print(BC.WARNING,BC.BOLD,"------------ENDING QM/ML MODULE-------------",BC.END)
            print_time_rel(module_init_time, modulename='QM/ML mech run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_ML_energy, self.QM_ML_gradient
        else:
            print_time_rel(module_init_time, modulename='QM/ML mech run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_ML_energy

    def create_linkatoms(self, current_coords):
        checkpoint=time.time()
        # Get linkatom coordinates
        self.linkatoms_dict = ash.modules.module_coords.get_linkatom_positions(self.boundaryatoms,self.qmatoms, current_coords, self.elems,
                                                                               linkatom_method=self.linkatom_method, linkatom_type=self.linkatom_type,
                                                                               linkatom_simple_distance=self.linkatom_simple_distance,
                                                                               linkatom_ratio=self.linkatom_ratio)
        printdebug("linkatoms_dict:", self.linkatoms_dict)
        if self.printlevel > 1:
            print("Adding linkatom positions to QM coords")
        self.linkatom_indices = [len(self.qmatoms)+i for i in range(0,len(self.linkatoms_dict))]
        self.num_linkatoms = len(self.linkatom_indices)
        linkatoms_coords = [self.linkatoms_dict[pair] for pair in sorted(self.linkatoms_dict.keys())]

        print_time_rel(checkpoint, modulename='create_linkatoms', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
        return linkatoms_coords

    # Run-preparation (for both electrostatic and mechanical)
    # Things that only have to be done in the first QM/ML run
    def runprep(self, current_coords):
        print("Inside QMMLTheory runprep")
        init_time_runprep=time.time()
        CheckpointTime=time.time()

        # Set basic element lists
        self.qmelems = [self.elems[i] for i in self.qmatoms]
        self.mlelems = [self.elems[i] for i in self.mlatoms]

        # LINKATOMS (both mech and elstat)
        check_before_linkatoms=time.time()
        if self.linkatoms is True:
            linkatoms_coords = self.create_linkatoms(current_coords)
            self.current_qmelems = self.qmelems + [self.linkatom_type]*self.num_linkatoms
            if self.printlevel > 1:
                print("Number of ML atoms:", len(self.mlatoms))
                print(f"There are {self.num_linkatoms} linkatoms")
            # Do possible Charge-shifting. ML1 charge distributed to ML2 atoms
            if self.embedding.lower() == "elstat":
                self.ml_elems_for_qmprogram = self.mlelems # Overwritten later, only matters for CP2K GEEP
                if self.printlevel > 1:
                    print("Doing charge-shifting...")

                # CHARGEBOUNDARY METHOD
                if self.chargeboundary_method == "shift":
                    print("Chargeboundary method is:  shift  ")
                    self.ShiftMLCharges() # Creates self.pointcharges
                    # Defining pointcharges as only containing ML atoms
                    self.pointcharges = [self.pointcharges[i] for i in self.mlatoms]

                    if self.dipole_correction is True:
                        print("Dipole correction is on. Adding dipole charges")
                        self.SetDipoleCharges(current_coords) # Creates self.dipole_charges and self.dipole_coords

                        # Adding dipole charge coords to ML coords (given to QM code) and defining pointchargecoords
                        if self.printlevel > 1:
                            print("Adding {} dipole charges to PC environment".format(len(self.dipole_charges)))

                        # Adding dipole charges to ML charges list (given to QM code)
                        self.pointcharges = list(self.pointcharges)+list(self.dipole_charges)
                        # Using element H for dipole charges. Only matters for CP2K GEEP
                        self.ml_elems_for_qmprogram = self.mlelems + ['H']*len(self.dipole_charges)
                        if self.printlevel > 1: print("Number of pointcharges after dipole addition: ", len(self.pointcharges))
                        print_time_rel(check_before_linkatoms, modulename='Linkatom-dipolecorrection', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
                    else:
                        print("Dipole correction is off. Not adding any dipole charges")
                        if self.printlevel > 1: print("Number of pointcharges: ", len(self.pointcharges))
                # RCD
                elif self.chargeboundary_method == "rcd":
                    print("Chargeboundary method is:  rcd  ")
                    self.pointcharges, RCD_additional_charges = self.RCD_shifting_prep(self.charges_qmregionzeroed)
                    self.ml_elems_for_qmprogram = self.mlelems + ['H']*len(RCD_additional_charges)
                else:
                    print("Unknown chargeboundary_method. Exiting")
                    ashexit()

                if self.printlevel > 1: 
                    print("Number of pointcharges defined for whole system: ", len(self.pointcharges))
                if self.printlevel > 1:
                    print("Number of pointcharges defined for ML region: ", len(self.pointcharges))

        # CASE: No Linkatoms
        else:
            self.ml_elems_for_qmprogram = self.mlelems
            self.num_linkatoms = 0
            # If no linkatoms then use original self.qmelems
            self.current_qmelems = self.qmelems
            # If no linkatoms then self.pointcharges are just original charges with QM-region zeroed
            if self.embedding.lower() == "elstat":
                self.pointcharges = [self.charges_qmregionzeroed[i] for i in self.mlatoms]

        # NOTE: Now we have updated ML-coordinates (if doing linkatoms, with dipolecharges etc) and updated mm-charges (more, due to dipolecharges if linkatoms)
        # We also have MMcharges that have been set to zero due to QM/ML
        # We do not delete charges but set to zero
        # If no qmatoms then do ML-only
        if len(self.qmatoms) == 0:
            print("No qmatoms list provided. Setting QMtheory to None")
            self.qm_theory_name="None"
            self.QMenergy=0.0

        # For truncatedPC option.
        self.pointcharges_original=copy.copy(self.pointcharges)

        # Initialize QM_PC_gradient for efficiency
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
        # If this is first run then do QM/ML runprep
        # Only do once to avoid cost in each step
        #############################################
        if self.runcalls == 0:
            print("First QMMLTheory run. Running runprep")
            self.runprep(current_coords)
            # This creates self.pointcharges, self.current_qmelems, self.ml_elems_for_qmprogram
            # self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords

        # Updating runcalls
        self.runcalls+=1

        #########################################################################################
        # General QM-code energy+gradient call.
        #########################################################################################

        # Split current_coords into ML-part and QM-part efficiently.
        used_mlcoords, used_qmcoords = current_coords[~self.xatom_mask], current_coords[self.xatom_mask]

        if self.linkatoms is True:
            # Update linkatom coordinates. Sets: self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords
            linkatoms_coords = self.create_linkatoms(current_coords)
            # Add linkatom coordinates to QM-coordinates
            used_qmcoords = np.append(used_qmcoords, np.array(linkatoms_coords), axis=0)

        # Update self.pointchargecoords based on new current_coords
        # print("self.dipole_correction:", self.dipole_correction)
        if self.chargeboundary_method == "shift" and self.dipole_correction is True:
            self.SetDipoleCharges(current_coords) # Note: running again
            self.pointchargecoords = np.append(used_mlcoords, np.array(self.dipole_coords), axis=0)
        elif self.chargeboundary_method == "rcd":
            #Appends RCD chargepositions to MM-coords
            self.pointchargecoords = self.RCD_shifting_update(used_mlcoords, current_coords)
        else:
            self.pointchargecoords = used_mlcoords

        # TRUNCATED PC Option: Speeding up QM/ML jobs of large systems by passing only a truncated PC field to the QM-code most of the time
        # Speeds up QM-pointcharge gradient that otherwise dominates
        # TODO: TruncatedPC is inactive
        if self.TruncatedPC is True:
            self.TruncatedPCfunction(used_qmcoords)

            # Modifies self.pointcharges and self.pointchargecoords
            # print("Number of charges after truncation :", len(self.pointcharges))
            # print("Number of charge coordinates after truncation :", len(self.pointchargecoords))

        # If numcores was set when calling QMMLTheory.run then using, otherwise use self.numcores
        if numcores == 1:
            numcores = self.numcores

        if self.printlevel > 1:
            print("Number of pointcharges (to QM program):", len(self.pointcharges))
            print("Number of charge coordinates:", len(self.pointchargecoords))
        if self.printlevel >= 2:
            print("Running QM/ML object with {} cores available".format(numcores))
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
                                                                                         qm_elems=self.current_qmelems, mm_elems=self.ml_elems_for_qmprogram,
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

        # Final QM/ML gradient. Combine QM gradient, ML gradient, PC-gradient (elstat ML gradient from QM code).
        # Do linkatom force projections in the end
        # UPDATE: Do ML step in the end now so that we have options for OpenML extern force
        if Grad is True:
            Grad_prep_CheckpointTime = time.time()
            # assert len(self.allatoms) == len(self.MLgradient)
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

            # Populatee QM_PC gradient (has full system size)
            CheckpointTime = time.time()
            self.make_QM_PC_gradient() #populates self.QM_PC_gradient
            print_time_rel(CheckpointTime, modulename='QMpcgrad prepare', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
            #ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM+PCgradient_{}_init".format(label), description="QM+PC gradient {} (au/Bohr):".format(label))
            #LINKATOM FORCE PROJECTION
            if self.linkatoms is True:
                CheckpointTime = time.time()

                for pair in sorted(self.linkatoms_dict.keys()):
                    #Grabbing linkatom data
                    linkatomindex=self.linkatom_indices.pop(0)
                    Lgrad=self.QMgradient[linkatomindex]
                    Lcoord=self.linkatoms_dict[pair]
                    #Grabbing QMatom info
                    fullatomindex_qm=pair[0]
                    qmatomindex=fullindex_to_qmindex(fullatomindex_qm,self.qmatoms)
                    Qcoord=used_qmcoords[qmatomindex]
                    #Grabbing MMatom info
                    fullatomindex_ml=pair[1]
                    Mcoord=current_coords[fullatomindex_ml]

                    # Getting gradient contribution to QM1 and ML1 atoms from linkatom
                    if self.linkatom_forceproj_method == "adv":
                        QM1grad_contrib, ML1grad_contrib = linkatom_force_adv(Qcoord, Mcoord, Lcoord, Lgrad)
                    elif self.linkatom_forceproj_method == "lever": 
                        QM1grad_contrib, ML1grad_contrib = linkatom_force_lever(Qcoord, Mcoord, Lcoord, Lgrad)
                    elif self.linkatom_forceproj_method == "chain":
                        QM1grad_contrib, ML1grad_contrib = linkatom_force_chainrule(Qcoord, Mcoord, Lcoord, Lgrad)
                    elif self.linkatom_forceproj_method.lower() == "none" or self.linkatom_forceproj_method == None:
                        QM1grad_contrib = np.zeros(3)
                        ML1grad_contrib = np.zeros(3)
                    else:
                        print("Unknown linkatom_forceproj_method. Exiting")
                        ashexit()
                    #print("QM1grad contrib:", QM1grad_contrib)
                    #print("MM1grad contrib:", MM1grad_contrib)

                    self.QM_PC_gradient[fullatomindex_qm] += QM1grad_contrib
                    self.QM_PC_gradient[fullatomindex_ml] += ML1grad_contrib

            #ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM+PCgradient_{}_afterlink".format(label), description="QM+PC gradient {} (au/Bohr):".format(label))
            print_time_rel(CheckpointTime, modulename='linkatomgrad prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            print_time_rel(Grad_prep_CheckpointTime, modulename='QM/ML gradient prepare', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            CheckpointTime = time.time()
        else:
            #No Grad
            self.QMenergy = QMenergy

        # ML THEORY
        self.MLenergy, self.MLgradient= self.ml_theory.run(current_coords=current_coords, Grad=Grad,
                                                            charges=self.charges_qmregionzeroed, connectivity=self.connectivity,
                                                            qmatoms=self.qmatoms)

        print_time_rel(CheckpointTime, modulename='ML step', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()


        #Final QM/ML Energy.
        self.QM_ML_energy= self.QMenergy+self.MLenergy
        if self.printlevel >= 2:
            blankline()
            if self.embedding.lower() == "elstat":
                print("Note: You are using electrostatic embedding. This means that the QM-energy is actually the polarized QM-energy")
            energywarning=""
            if self.TruncatedPC is True:
                #if self.TruncatedPCflag is True:
                print("Warning: Truncated PC approximation is active. This means that QM and QM/ML energies are approximate.")
                energywarning="(approximate)"

            print("{:<20} {:>20.12f} {}".format("QM energy: ",self.QMenergy,energywarning))
            print("{:<20} {:>20.12f}".format("ML energy: ", self.MLenergy))
            print("{:<20} {:>20.12f} {}".format("QM/ML energy: ", self.QM_ML_energy,energywarning))
            blankline()

        #FINAL QM/ML GRADIENT ASSEMBLY
        if Grad is True:
            #Now assemble full QM/ML gradient
            assert len(self.QM_PC_gradient) == len(self.MLgradient)
            self.QM_ML_gradient=self.QM_PC_gradient+self.MLgradient

            if self.printlevel >=3:
                print("Printlevel >=3: Printing all gradients to disk")
                ash.modules.module_coords.write_coords_all(self.QMgradient_wo_linkatoms, self.qmelems, indices=self.qmatoms, file="QMgradient-without-linkatoms_{}".format(label), description="QM gradient w/o linkatoms {} (au/Bohr):".format(label))
                #Writing QM+Linkatoms gradient
                ash.modules.module_coords.write_coords_all(self.QMgradient, self.qmelems+['L' for i in range(self.num_linkatoms)], indices=self.qmatoms+[0 for i in range(self.num_linkatoms)], file="QMgradient-with-linkatoms_{}".format(label), description="QM gradient with linkatoms {} (au/Bohr):".format(label))
                ash.modules.module_coords.write_coords_all(self.PCgradient, self.mlelems, indices=self.mlatoms, file="PCgradient_{}".format(label), description="PC gradient {} (au/Bohr):".format(label))
                ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM+PCgradient_{}".format(label), description="QM+PC gradient {} (au/Bohr):".format(label))
                ash.modules.module_coords.write_coords_all(self.MLgradient, self.elems, indices=self.allatoms, file="MLgradient_{}".format(label), description="ML gradient {} (au/Bohr):".format(label))
                ash.modules.module_coords.write_coords_all(self.QM_ML_gradient, self.elems, indices=self.allatoms, file="QM_MLgradient_{}".format(label), description="QM/ML gradient {} (au/Bohr):".format(label))
            if self.printlevel >= 2:
                print(BC.WARNING,BC.BOLD,"------------ENDING QM/ML MODULE-------------",BC.END)
            print_time_rel(module_init_time, modulename='QM/ML run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_ML_energy, self.QM_ML_gradient
        else:
            print_time_rel(module_init_time, modulename='QM/ML run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_ML_energy
