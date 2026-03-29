import time
import os
from ash.modules.module_coords import elemstonuccharges
from ash.modules.module_coords_PBC import cell_params_to_vectors, cell_vectors_to_params
from ash.functions.functions_general import ashexit, BC,print_time_rel
from ash.functions.functions_general import print_line_with_mainheader
from ash.interfaces.interface_mace import stress_to_grad
import ash.constants

# Simple interface to Fairchem

# Use:

# VIA MODEL NAMES
# Models available in version 2: uma-s-1p1 (faster,very good), uma-m-1p1 (slower,best)
# Example: model_name = "uma-s-1p1"
# Requires hugging-face token activated in shell
# e.g. export HF_TOKEN=xxxxxxx

#Via model_file
#model_file="uma-s-1p1.pt"

class FairchemTheory():
    def __init__(self, model_name=None, model_file=None, task_name=None, platform="cuda", device=None, seed=41, numcores=1,
                 printlevel=2, periodic=False, periodic_cell_vectors=None, periodic_cell_dimensions=None):

        module_init_time=time.time()
        # Early exits
        try:
            import fairchem
        except ImportError:
            print("Problem importing fairchem. Make sure you have installed fairchem correctly")
            print("See: https://github.com/facebookresearch/fairchem")
            print("Most likely you need to do: pip install fairchem-core")
            ashexit()

        try:
            import ase
        except ImportError:
            print("Problem importing ase. Make sure you have installed  correctly")
            print("Most likely you need to do: pip install ase")
            ashexit()

        self.theorytype="QM"
        self.theorynamelabel="Fairchem"

        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        self.printlevel=printlevel
        self.task_name=task_name

        # Platform/device
        if device is not None:
            print("Warning: device keyword is deprecated. Use platform instead")
            ashexit()
        self.platform=platform.lower()

        self.model_name=model_name
        self.model_file=model_file
        self.seed=seed
        self.numcores=numcores

        # PBC
        self.periodic=periodic
        self.periodic_cell_vectors=None # initially
        self.stress=False
        if self.periodic:
            print("PBC enabled in FairchemTheory")
            self.stress=True
            if periodic_cell_vectors is None and periodic_cell_dimensions is None:
                print("Error: for periodic calculations, you must specify either periodic_cell_vectors or  periodic_cell_dimensions")
                ashexit()
                # Convert to cell vectors
                self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)
            elif periodic_cell_vectors is not None:
                self.periodic_cell_vectors = periodic_cell_vectors
                self.periodic_cell_dimensions = cell_vectors_to_params(periodic_cell_vectors)
            elif periodic_cell_dimensions is not None:
                self.periodic_cell_dimensions = periodic_cell_dimensions
                self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)

            print("Cell vectors:", self.periodic_cell_vectors)
            print("Cell dimensions:", self.periodic_cell_dimensions)

        # Counter for runcalls
        self.runcalls=0

        if self.platform.lower() == 'cpu':
            #Works ??
            os.environ['OMP_NUM_THREADS'] = str(numcores)

        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        if self.model_name is not None:
            print("Model set:", self.model_name)
            predictor = pretrained_mlip.get_predict_unit(self.model_name, device=self.platform)
            self.calc = FAIRChemCalculator(predictor, task_name=self.task_name, seed=self.seed)
        elif self.model_file is not None:
            print("Model-file set:", self.model_file)
            # TODO: can we fix
            #print("Warning: single-atom systems do not work with this approach")
            self.calc = FAIRChemCalculator.from_model_checkpoint(self.model_file,
                                                            task_name=self.task_name, device=self.platform,
                                                            seed=self.seed)
        else:
            print("Error:Neither model or model_file was set")
            ashexit()
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} init', moduleindex=2)

    def cleanup(self):
        pass


    # Update cell using either periodic_cell_vectors or periodic_cell_dimensions
    def update_cell(self,periodic_cell_vectors=None, periodic_cell_dimensions=None):
        print("Updating cell vectors")
        if periodic_cell_vectors is not None:
            self.periodic_cell_vectors = periodic_cell_vectors
            self.periodic_cell_dimensions = cell_vectors_to_params(periodic_cell_vectors)
        elif periodic_cell_dimensions is not None:
            self.periodic_cell_dimensions=periodic_cell_dimensions
            self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)

    def get_cell_gradient(self):
        return self.cell_gradient

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None, Hessian=False,
            charge=None, mult=None):

        module_init_time=time.time()
        if self.printlevel >= 2:
            print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        import ase
        if self.runcalls == 0:
            print("First runcall. Creating atoms object")
            if self.periodic:
                self.atoms = ase.atoms.Atoms(qm_elems,positions=current_coords, cell=self.periodic_cell_vectors,
                                        pbc=True)
            else:
                self.atoms = ase.atoms.Atoms(qm_elems,positions=current_coords)

            self.atoms.info["charge"] = charge
            self.atoms.info["spin"] = mult
            # Assigning calculator
            self.atoms.calc =self.calc
        elif len(self.atoms.numbers) != len(current_coords):
            print("Number-of-atoms mismatch (new molecule?). Creating new atoms object")
            if self.periodic:
                self.atoms = ase.atoms.Atoms(qm_elems,positions=current_coords, cell=self.periodic_cell_vectors,
                                        pbc=True)
            else:
                self.atoms = ase.atoms.Atoms(qm_elems,positions=current_coords)

            self.atoms.info["charge"] = charge
            self.atoms.info["spin"] = mult
            # Assigning calculator
            self.atoms.calc =self.calc
        else:
            print("Updating coordinates in atoms object")
            self.atoms.set_positions(current_coords)

        # Energy
        en = self.atoms.get_potential_energy()
        self.energy = float(en*ash.constants.evtohar)
        if self.printlevel >= 2:
            print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        if Grad:
            forces = self.atoms.get_forces()
            self.gradient = forces/-51.422067090480645
            if self.stress:
                stress_ev_ang3 = self.atoms.get_stress(voigt=False)
                self.cell_gradient = stress_to_grad(stress_ev_ang3,self.atoms.get_volume(), self.atoms.get_cell())
                print("Cell gradient:", self.cell_gradient)

        self.runcalls+=1
        if self.printlevel >= 2:
            print(BC.OKBLUE,BC.BOLD,f"------------ENDING {self.theorynamelabel}-INTERFACE-------------", BC.END)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)

        if Grad:
            return self.energy, self.gradient
        else:
            return self.energy