import time
import numpy as np
import shutil
import os 

from ash.modules.module_coords import elemstonuccharges
from ash.functions.functions_general import ashexit, BC, print_if_level,print_time_rel
from ash.functions.functions_general import print_line_with_mainheader, listdiff
import ash.constants

# Simple interface to emle-engine for getting the electrostatic embedding energy and gradient

# Note: use in 2 ways:
# 1. EMLETheory directly. Requires combining  manually with QM and MM via WrapTheory
# 2. Use standalone EMLE_run function. Currently used directly by QMMMTheory(embedding="emle-elstat")

# TODO: PBC?

class EMLETheory():
    def __init__(self, method="electrostatic", model=None, alphamode="species", qmatoms=None,
                 label=None, numcores=1, printlevel=1, platform="cpu", fragment=None,
                 charges=None,
                 ):

        # Indicate that this is a hybrid QM/MM type theory
        self.theorytype = "QM/MM"
        self.theorynamelabel="EMLETheory"
        self.label = label
        self.analytic_hessian = True
        self.numcores = numcores
        self.printlevel = printlevel
        self.properties = {}

        # Book-keeping
        self.fragment=fragment
        self.qmatoms=qmatoms
        if fragment is None:
            ashexit("Error: fragment must be provided")
        self.numatoms = self.fragment.numatoms
        self.allatoms=self.fragment.allatoms
        self.mmatoms = listdiff(self.allatoms,self.qmatoms)
        self.xatom_mask = np.isin(self.allatoms, self.qmatoms)
        self.qmelems = [self.fragment.elems[i] for i in self.qmatoms]
        self.mmelems = [self.fragment.elems[i] for i in self.mmatoms]

        # MM-region charges. If not provided by run then these fixed-system charges are used
        self.charges=charges

        # EMLE options
        self.method=method
        self.platform=platform
        self.model=model
        self.alpha_mode=alphamode

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=True, numcores=None, restart=False, label=None, Hessian=False,
            charge=None, mult=None):

        module_init_time = time.time()
        if numcores is not None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        # Early exits
        # Coords provided to run
        if current_coords is None:
            print("no coordinates. Exiting")
            ashexit()

        # Check for MM charges.
        if MMcharges is None:
            print(BC.WARNING, "Warning. MM charges are not provide to run. THis is normally the case.", BC.END)
            if self.charges is None:
                print(BC.FAIL,"Error: No charges were found in EMLETheory object either.", BC.END)
                ashexit()
            else:
                print("Found charges attribute.")
                MMcharges=self.charges
            
        # Importing libraries
        try:
            import torch
            from emle.models._emle import EMLE
        except ImportError:
            print(BC.FAIL, "Error. EMLE is not installed. Please install EMLE to use this interface.", BC.END)
            print("Also make sure the correct torch and torchani versions are installed")
            ashexit()

        # Convert to nuclear charges
        atomic_numbers = elemstonuccharges(self.qmelems)

        # Preparing coordinates.
        # Split current_coords into MM-part and QM-part efficiently.
        used_mmcoords, used_qmcoords = current_coords[~self.xatom_mask], current_coords[self.xatom_mask]

        #Pointcharges
        pointcharges = [MMcharges[i] for i in self.mmatoms]
        print("pointcharges:", pointcharges)
        # TODO: linkatoms and pointcharge shifting
        # TODO
        cell=None
        # Run EMLE to get energy, gradient and pcgradient
        if Grad:
            energy, qmgradient, pcgradient = EMLE_run(atomic_numbers, pointcharges, used_qmcoords, used_mmcoords, cell, charge, 
                                                  platform=self.platform, model=self.model, method=self.method, Grad=True)
        else:
            energy = EMLE_run(atomic_numbers, pointcharges, used_qmcoords, used_mmcoords, cell, charge, 
                                                  platform=self.platform, model=self.model, method=self.method, Grad=False)            

        # Energy
        self.energy = energy
        print("Energy (EMLE):", self.energy)
        self.QM_MM_energy = energy
        self.QMenergy = 0.0
        self.MMenergy = 0.0

        # TODO: linkatoms grad
        if Grad:
            # initializing full gradient
            self.gradient = np.zeros((self.numatoms,3))
            # QM gradient
            # fill in the QM gradient using QM-atom indices in qmatoms
            self.gradient[self.qmatoms] = qmgradient
            #print("Gradient with QM (EMLE):", self.gradient)
            # MM gradient
            self.gradient[self.mmatoms] = pcgradient
            print("Final Gradient (EMLE):", self.gradient)
        if Grad:
            return self.energy, self.gradient
        else:
            return self.energy

def EMLE_run(atomic_numbers, charges_mm, xyz_qm, xyz_mm, cell, charge, 
             platform="cpu", model=None, method="", alpha_mode="species", Grad=False):
    print("Inside EMLE_run")
    import torch
    from emle.models._emle import EMLE

    # Platform/device selection
    device=torch.device(platform)
    print("Using torch device:", device)
    #print("atomic_numbers:", atomic_numbers)
    #print("charges_mm:", charges_mm)
    #print("xyz_qm:", xyz_qm)
    #print("xyz_mm:", xyz_mm)

    # Preparing coords and charges as torch tensors
    atomic_numbers = torch.tensor(
        atomic_numbers, dtype=torch.int64, device=device)
    charges_mm = torch.tensor(
        charges_mm, dtype=torch.float32, device=device, requires_grad=True)
    xyz_qm = torch.tensor(
        xyz_qm, dtype=torch.float32, device=device, requires_grad=True)
    xyz_mm = torch.tensor(
        xyz_mm, dtype=torch.float32, device=device, requires_grad=True)
    if cell is not None:
        cell = torch.tensor(cell, dtype=torch.float32, device=device)
    
    # Create the EMLE model instance.
    print("Creating EMLE object")
    emleobj = EMLE(
                model=model,
                method=method,
                alpha_mode=alpha_mode,
                atomic_numbers=atomic_numbers,
                mm_charges=charges_mm,
                qm_charge=charge)
    #  Run
    Eobj = emleobj(atomic_numbers, charges_mm, xyz_qm, xyz_mm, cell, charge)
    # Compute the total energy
    energy = float(Eobj.sum().detach().cpu().numpy())
    if Grad:
        # Gradient
        dE_dxyz_qm, dE_dxyz_mm, dE_dcharges_mm = torch.autograd.grad(
            Eobj.sum(), (xyz_qm, xyz_mm, charges_mm), allow_unused=True)
        # Preparing results
        _BOHR_TO_ANGSTROM = 0.529177210903
        dE_dxyz_qm_bohr = dE_dxyz_qm.cpu().numpy() * _BOHR_TO_ANGSTROM
        dE_dxyz_mm_bohr = (dE_dxyz_mm.cpu().numpy() * _BOHR_TO_ANGSTROM if dE_dxyz_mm is not None else np.zeros_like(xyz_mm.cpu().numpy()))
    if Grad:
        return energy, dE_dxyz_qm_bohr, dE_dxyz_mm_bohr
    else:
        return energy