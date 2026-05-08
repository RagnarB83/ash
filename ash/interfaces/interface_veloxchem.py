import subprocess as sp
import os
import shutil
import time
import numpy as np
import pathlib
from ash.modules.module_theory import Theory
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader, writestringtofile
from ash.modules.module_coords import nucchargelist
from ash.modules.module_coords_PBC import cell_vectors_to_params, cell_params_to_vectors
import ash.settings_ash
from ash.functions.functions_parallel import check_OpenMPI

# Veloxchem Theory object.

class VeloxchemTheory(Theory):
    def __init__(self, scf_type="restricted", xcfun=None, basis=None):
        super().__init__()

        self.theorynamelabel="Veloxchem"

        try:
            import veloxchem as vlx
        except:
            print("Error: Veloxchem could not be imported")
            ashexit()

        if scf_type == "restricted":
            print("Creating ScfRestrictedDriver")
            self.scf_drv = vlx.ScfRestrictedDriver()
        elif scf_type == "unrestricted":
            print("Creating ScfUnestrictedDriver")
            self.scf_drv = vlx.ScfUnrestrictedDriver()
        elif scf_type == "restrictedopen":
            print("Creating ScfRestrictedOpenDriver")
            self.scf_drv = vlx.ScfRestrictedOpenDriver()
        self.scf_drv.filename = "vlx_output"
        # basis name
        self.basis=basis

        #if xcfun is not None:
        #    print("Setting xcfun to:", xcfun)
        #    scf_drv.xcfun = xcfun


    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None, Hessian=False,
            charge=None, mult=None):
        import veloxchem as vlx
        module_init_time=time.time()
        if numcores is None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)

        # Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # veloxchem molecule
        #lines = [str(len(qm_elems)), "title"]
        #lines += [f"{el}  {x:.6f}  {y:.6f}  {z:.6f}" for el, (x, y, z) in zip(qm_elems, current_coords)]
        #xyz_string = "\n".join(lines)
        #molecule = vlx.Molecule.read_xyz_string(xyz_string)
        
        # Creating molecule
        molecule = vlx.Molecule(qm_elems, current_coords, units='angstrom', charge=charge, mult=mult)
        molecule.print_keywords()
        # Creating basis set object
        basis = vlx.MolecularBasis.read(molecule, self.basis)

        scf_results = self.scf_drv.compute(molecule, basis)
        print("scf_results:",scf_results)
        self.energy=None
        self.gradient=None
        # Grad

        if Grad:
            return self.energy,self.gradient

        else:
            return self.energy