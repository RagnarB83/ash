import time
import os
from ash.modules.module_coords import elemstonuccharges
from ash.functions.functions_general import ashexit, BC,print_time_rel
from ash.functions.functions_general import print_line_with_mainheader
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
    def __init__(self, model_name=None, model_file=None, task_name=None, device="cuda", seed=41, numcores=1,
                 printlevel=2):

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
        self.device=device
        self.model_name=model_name
        self.model_file=model_file
        self.seed=seed
        self.numcores=numcores

        # Counter for runcalls
        self.runcalls=0

        if self.device.lower() == 'cpu':
            #Works ??
            os.environ['OMP_NUM_THREADS'] = str(numcores)

        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        if self.model_name is not None:
            print("Model set:", self.model_name)
            predictor = pretrained_mlip.get_predict_unit(self.model_name, device=self.device)
            self.calc = FAIRChemCalculator(predictor, task_name=self.task_name, seed=self.seed)
        elif self.model_file is not None:
            print("Model-file set:", self.model_file)
            self.calc = FAIRChemCalculator.from_model_checkpoint(self.model_file,
                                                            task_name=self.task_name, device=self.device,
                                                            seed=self.seed)
        else:
            print("Error:Neither model or model_file was set")
            ashexit()
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} init', moduleindex=2)

    def cleanup(self):
        pass

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

        if self.runcalls == 0:
            print("First runcall. Creating atoms object")
            import ase
            self.atoms = ase.atoms.Atoms(qm_elems,positions=current_coords)
            self.atoms.info["charge"] = charge
            self.atoms.info["spin"] = mult
            # Assigning calculator
            self.atoms.calc =self.calc
        elif len(self.atoms.numbers) != len(current_coords):
            print("Number-of-atoms mismatch (new molecule?). Creating new atoms object")
            import ase
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

        self.runcalls+=1
        if self.printlevel >= 2:
            print(BC.OKBLUE,BC.BOLD,f"------------ENDING {self.theorynamelabel}-INTERFACE-------------", BC.END)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)

        if Grad:
            return self.energy, self.gradient
        else:
            return self.energy