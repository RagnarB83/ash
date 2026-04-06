import time
import numpy as np
import shutil
import os 

from ash.modules.module_coords_PBC import cell_params_to_vectors, cell_vectors_to_params
from ash.functions.functions_general import ashexit, BC,print_time_rel
from ash.functions.functions_general import print_line_with_mainheader
import ash.constants

# Simple interface to MACE for both using and training
class MACETheory():
    def __init__(self, config_filename="config.yml",
                 model_name=None, model_name_subtype=None, model_name_head=None,
                 model_file=None, printlevel=2, mace_load_dispersion=False, mace_dispersion_xc=None,
                 label="MACETheory", numcores=1, platform="cpu", device=None, return_zero_gradient=False, default_dtype="float64",
                 energy_weight=None, forces_weight=None, max_num_epochs=None, valid_fraction=None,
                 periodic=False, periodic_cell_vectors=None, periodic_cell_dimensions=None):

        self.theorytype = 'QM'
        self.theorynamelabel = 'MACE'
        self.label = label
        self.analytic_hessian = True
        self.numcores = numcores
        self.config_filename=config_filename
        self.printlevel = printlevel
        self.properties = {}

        # Parallelization at CPU level
        os.environ['OMP_NUM_THREADS'] = str(numcores)
        os.environ['MKL_NUM_THREADS'] = str(numcores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(numcores)

        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")
        # Early exits
        try:
            import mace
        except ImportError:
            print("Problem importing mace. Make sure you have installed mace-correctly")
            print("Most likely you need to do: pip install mace-torch")
            print("Also recommended: pip install cuequivariance_torch")
            ashexit()

        # Ignore predicted forces and return zero gradient
        self.return_zero_gradient=return_zero_gradient

        # Polarmace (activated later if detected)
        self.polarmace=False

        # New interface: activated later if needed
        self.new_interface=False

        self.default_dtype=default_dtype

        # Model attribute is None until we have loaded a model
        self.model=None
        #
        self.model_file=model_file
        self.model_name=model_name #for quickly loading foundational models
        self.model_name_subtype=model_name_subtype #subtype of foundational model
        self.model_name_head = model_name_head # choose head of multi-head foundational model
        self.mace_load_dispersion=mace_load_dispersion # activate dispersion 
        self.mace_dispersion_xc=mace_dispersion_xc # functional keyword
        # Training parameters
        self.energy_weight=energy_weight
        self.forces_weight=forces_weight
        self.max_num_epochs=max_num_epochs
        self.valid_fraction=valid_fraction

        # Platform/device
        if device is not None:
            print("Warning: device keyword is deprecated. Use platform instead")
            ashexit()
        self.platform=platform.lower()

        # PBC
        self.periodic=periodic
        self.periodic_cell_vectors=None # initially
        self.stress=False
        if self.periodic:
            print("PBC enabled in MaceTHeory")
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

        self.training_done=False

    def cleanup(self):
        print("No cleanup implemented")

    def set_numcores(self,numcores):
        self.numcores=numcores

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

    def train(self, config_file="config.yml", name="model",model="MACE", platform=None, device=None,
                      valid_fraction=0.1, train_file="train_data_mace.xyz",E0s=None,
                      energy_key='energy_REF', forces_key='forces_REF',        
                      energy_weight=1, forces_weight=100, seed=42,
                      max_num_epochs=500, swa=True, batch_size=10,
                      max_L = 0, r_max = 5.0, num_channels=128,  
                      results_dir= "MACE_models", checkpoints_dir = "MACE_models", 
                      log_dir ="MACE_models", model_dir="MACE_models"):
        module_init_time=time.time()

        if self.energy_weight is not None:
            energy_weight=self.energy_weight
        if self.forces_weight is not None:
            forces_weight=self.forces_weight
        if self.max_num_epochs is not None:
            max_num_epochs=self.max_num_epochs
        if self.valid_fraction is not None:
            valid_fraction=self.valid_fraction


        self.train_file=train_file
        self.valid_fraction=valid_fraction

        if device is not None:
            print("Warning: device keyword is deprecated. Please use platform instead")
            ashexit()

        if platform is None:
            print("Warning: platform not passed to train. Using object's platform attribute:", self.platform)
            platform=self.platform

        print("Training activated")
        print("Training parameters:")
        print("config_file", config_file)
        print("name:", model)
        print("model:", model)
        print("platform:", platform)
        print("Validation set fraction (valid_fraction):", valid_fraction)
        print("train_file:", self.train_file)
        print("E0s:", E0s)
        print("energy_key:", energy_key)
        print("forces_key:", forces_key)
        print("energy_weight:", energy_weight)
        print("forces_weight:", forces_weight)
        print("max_num_epochs:", max_num_epochs)
        print("swa:", swa)
        print("seed:", seed)
        print("batch_size:", batch_size)
        print("max_L:", max_L)
        print("r_max:", r_max)
        print("num_channels:", num_channels)
        print("results_dir:", results_dir)
        print("checkpoints_dir:", checkpoints_dir)
        print("log_dir:", log_dir)
        print("model_dir:", model_dir)

        self.check_file_exists(self.train_file)

        #Write 
        print("\nWriting MACE config file to disk as as:", self.config_filename)
        #write_mace_config(config_file=self.config_filename)
        print()
        write_mace_config(config_file=config_file, name=name, model=model, platform=platform, 
                      valid_fraction=valid_fraction, train_file=self.train_file,E0s=E0s,
                      energy_key=energy_key, forces_key=forces_key,        
                      energy_weight=energy_weight, forces_weight=forces_weight,
                      max_num_epochs=max_num_epochs, swa=swa, batch_size=batch_size,
                      max_L = max_L, r_max = r_max, seed=seed,
                      num_channels=num_channels,  
                      results_dir= results_dir, checkpoints_dir = checkpoints_dir, 
                      log_dir=log_dir, model_dir=model_dir)
        print()
        import logging
        import sys
        from mace.cli.run_train import main as mace_run_train_main

        logging.getLogger().handlers.clear()
        sys.argv = ["program", "--config", self.config_filename]
        print("="*100)
        print("MACE BEGINNING")
        print("="*100)
        mace_run_train_main()
        print("="*100)
        print("MACE DONE")
        print("="*100)
        self.training_done=True
        print("Training by MACE is done. Hopefully it produced something sensible")
        print(f"The final models are located in directory: {results_dir}")
        print(f"Recommended model files to use for production: {results_dir}/{name}_compiled.model and {results_dir}/{name}_stagetwo_compiled.model")
        print()
        if self.model_file is not None:
            print("Model_file attribute was previously set at initialization.")
            print(f"Moving and renaming file {results_dir}/{name}_stagetwo_compiled.model   to :  {self.model_file}")
            shutil.move(f"{results_dir}/{name}_stagetwo_compiled.model", self.model_file)
        else:
            self.model_file=f"{os.path.abspath(os.getcwd())}/{results_dir}/{name}_stagetwo_compiled.model"
        print("model_file attribute is:", self.model_file )
        print("MACETheory object can now be used directly.")

        # If we train with a specific device we would want to use that same device for evaluation/prediction
        self.platform=platform
        print("Setting platform of object to be ", self.platform)

        #Load model from file
        self.modelfile_load()

        #############
        #STATISTICS
        #############

        # Predicting 
        #from mlatom.MLtasks import predicting, analyzing
        #print("Now predicting for molDB")
        #predicting(model=self.model, molecular_database=molDB, value=True, gradient=True)
        #print("Now predicting for subtrainDB")
        #predicting(model=self.model, molecular_database=subtrainDB, value=True, gradient=True)
        #print("Now predicting for valDB")
        #predicting(model=self.model, molecular_database=valDB, value=True, gradient=True)

        # Analyzing
        #if molDB_xyzvecproperty_file is not None:
        #    self.result_molDB = analyzing(molDB, ref_value='energy', est_value='estimated_y', ref_grad='energy_gradients', est_grad='estimated_xyz_derivatives_y', set_name="molDB")
        #    self.result_subtrainDB = analyzing(subtrainDB, ref_value='energy', est_value='estimated_y', ref_grad='energy_gradients', est_grad='estimated_xyz_derivatives_y', set_name="subtrainDB")
        #    self.result_valDB = analyzing(valDB, ref_value='energy', est_value='estimated_y', ref_grad='energy_gradients', est_grad='estimated_xyz_derivatives_y', set_name="valDB")
        #else:
        #    self.result_molDB = analyzing(molDB, ref_value='energy', est_value='estimated_y',  set_name="molDB")
        #    self.result_subtrainDB = analyzing(subtrainDB, ref_value='energy', est_value='estimated_y', set_name="subtrainDB")
        #    self.result_valDB = analyzing(valDB, ref_value='energy', est_value='estimated_y', set_name="valDB")

        #print("Statistics saved as attributes:  result_molDB, result_subtrainDB and result_valDB of MLatomTheory object")
        #print()
        #print("self.result_molDB:", self.result_molDB)
        #print("self.result_subtrainDB:", self.result_subtrainDB)
        #print("self.result_valDB:", self.result_valDB)

        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} train', moduleindex=2)

    def check_file_exists(self, file):
        import os
        if file is None:
            print(f"Error: File is None. Exiting")
            ashexit()
        if self.printlevel >= 2:
            print(f"A filename-string ({file}) was provided, checking if file exists")
        file_present = os.path.isfile(file)
        if self.printlevel >= 2:
            print(f"File {file} exists:", file_present)
        if file_present is False:
            print(f"File {file} does not exist. Exiting.")
            ashexit()

    def modelfile_load(self):
        module_init_time=time.time()

        if 'polar' in self.model_file.lower():
            print("Model file name contains 'polar'. Assuming this is a polar MACE model. Loading via mace_polar")
            self.polarmace=True
            self.new_interface=True
            from mace.calculators import mace_polar
            self.model = mace_polar(
                model=self.model_file,
                device=self.platform,
                default_dtype=self.default_dtype) # use float32 for faster MD)
        elif 'mh' in self.model_file.lower():
            print("Model file name contains 'mh'. Assuming this is a multihead MACE model. Loading via mace_mp.")
            self.new_interface=True
            from mace.calculators import mace_mp
            print("D3 dispersion:", self.mace_load_dispersion)
            print("D3 xc:", self.mace_dispersion_xc)
            if self.model_name_head is None:
                print("Error: no head provided for an MH model. You  need to select head by ASH model_name_head keyword.")
                ashexit()
                #self.model = mace_mp(model=self.model_file, default_dtype=self.default_dtype, device=self.platform)
            else:
                print("Using head:", self.model_name_head)
                self.model = mace_mp(model=self.model_file, default_dtype=self.default_dtype, device=self.platform, head=self.model_name_head,
                                            dispersion=self.mace_load_dispersion, dispersion_xc=self.mace_dispersion_xc)
        else:
            print("Loading regular MACE via Pytorch")
            import torch
            # Load model
            print(f"Loading model from file {self.model_file}. Platform is: {self.platform}")
            self.model = torch.load(f=self.model_file, map_location=torch.device(self.platform))
            self.model = self.model.to(self.platform)  # for possible cuda problems
            print_time_rel(module_init_time, modulename=f'MACE model-load', moduleindex=2)

    # Load foundational model by name
    def modelname_load(self):
        print("Inside modelname_load")
        print("model_name:", self.model_name)
        print("model_name_subtype:", self.model_name_subtype)
        print("model_name_head:", self.model_name_head)
        print("default_dtype:", self.default_dtype)
        print()
        if self.model_name.lower() in ['mace-ani-cc','mace_anicc']:
            print("MACE-ANI-CC model requested")
            from mace.calculators import mace_anicc
            self.model = mace_anicc(device=self.platform, default_dtype=self.default_dtype)
        # MACE-OMol
        elif self.model_name.lower() in ['mace_omol','mace-omol']:
            print("MACE-OMOL model requested")
            from mace.calculators import mace_omol
            print("Loading MACE-OMol model:")
            print("Using extra_large model by default (MACE-omol-0-extra-large-1024.model)")
            self.model = mace_omol(model="extra_large", device=self.platform, default_dtype=self.default_dtype)
        # MACE-OFF
        elif self.model_name.lower() in ['mace_off23','mace_off', 'mace-off', 'mace-off23']:
            print("MACE-OFF model requested")
            from mace.calculators import mace_off
            if self.model_name_subtype is None:
                print("Loading MACE-OFF model:")
                print("Using medium model by default (use model_name_subtype keyword to choose small, medium, large)")
                self.model = mace_off(model="medium", device=self.platform, default_dtype=self.default_dtype)
            else:
                print("MACE-OFF model with modelname_subtype:", self.model_name_subtype)
                self.model = mace_off(model=self.model_name_subtype, device=self.platform, default_dtype=self.default_dtype)
        # MACE Materials Project (MP) models
        elif self.model_name.lower() in ['mace-mp','mace-mh']:
            from mace.calculators import mace_mp
            if self.model_name_subtype is None:
                print("Loading MACE-MP model:")
                print("Using medium-mpa-0 model by default (use model_name_subtype keyword to choose between small, medium, large or medium-mpa-0)")
                print("D3 dispersion:", self.mace_load_dispersion)
                print("D3 xc:", self.mace_dispersion_xc)
                self.model = mace_mp(model="medium", device=self.platform, default_dtype=self.default_dtype, 
                                     dispersion=self.mace_load_dispersion, dispersion_xc=self.mace_dispersion_xc)
            else:
                print("MACE-MP model with modelname_subtype:", self.model_name_subtype)
                if self.model_name_head is None:
                    print("No model_name_head chosen. Please choose head via keyword model_name_head.")
                    ashexit()
                else:
                    self.model = mace_mp(model=self.model_name_subtype, head=self.model_name_head, device=self.platform, default_dtype=self.default_dtype,
                                         dispersion=self.mace_load_dispersion, dispersion_xc=self.mace_dispersion_xc)
        # MACE Polar
        elif self.model_name.lower() in ['mace-polar','mace_polar', 'mace-polar-1']:
            from mace.calculators import mace_polar
            if self.model_name_subtype is None:
                print("Loading MACE-Polar model:")
                #print("Using polar-1-m model by default (use model_name_subtype keyword to choose between polar-1-s, polar-1-m, polar-1-l)")
                self.model = mace_polar(model="polar-1-m",
                    device=self.platform,
                    default_dtype=self.default_dtype) # use float32 for faster MD
            else:
                print("MACE-MP model with modelname_subtype:", self.model_name_subtype)
                self.model = mace_polar(model=self.model_name_subtype,
                    device=self.platform,
                    default_dtype=self.default_dtype) # use float32 for faster MD
        else:
            print("No valid model_name was found that could be loaded (typo?)")
            ashexit()
        # Enabling new_interface for these models
        self.new_interface = True

    def get_dipole_moment(self):
        if "dipole" not in self.properties:
            print("Dipole moment not available")
            return None
        else:
            return self.properties["dipole"]

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None, Hessian=False,
            charge=None, mult=None):

        module_init_time = time.time()
        if numcores is not None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)

        # Early exits
        # Coords provided to run
        if current_coords is None:
            print("no coordinates. Exiting")
            ashexit()
        if PC is True:
            print("PC is not supported")
            ashexit()

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # Making sure Grad is True if doing Hessian
        if Hessian:
            Grad=True

        print("Running on platform/device:", self.platform)
        # Checking if model is alreadyloaded
        if self.model is None:
            print("A model has not been loaded yet.")
            # We can only proceed if we have a model_file or model_name so checking
            if self.model_file is None and self.model_name is None:
                print("Neither model_file or model_name have been defined.")
                print("Either load a valid model (model_file or model_name keywords) or train a model (train method) before running")
                ashexit()

            # Loading will define self.model
            if self.model_file is not None:
                print("Loading MACE model from file:", self.model_file)
                # Checking first f file exists
                self.check_file_exists(self.model_file)
                #Load model
                self.modelfile_load()
            elif self.model_name is not None:
                print("Loading via model_name:", self.model_name)
                self.modelname_load()
            else:
                print("Error: Neither modelfile or modelname was defined.")
                ashexit()

        # Creating ASE atoms object (MACE has ASE has dependency anyway)
        import ase
        if self.periodic:
            atoms = ase.atoms.Atoms(qm_elems,positions=current_coords, cell=self.periodic_cell_vectors,
                                    pbc=True)
        else:
            atoms = ase.atoms.Atoms(qm_elems,positions=current_coords)
        atoms.info["charge"] = charge
        atoms.info["spin"] = mult
    
        # New simpler MACE interface via ASE
        # Works for foundational models
        if self.new_interface is True:
            # Add loaded model to ASE calculator
            atoms.calc = self.model

            # Run energy
            self.energy = atoms.get_potential_energy() * ash.constants.evtohar
            print("Energy:", self.energy)
            forces = atoms.get_forces()
            self.gradient = forces/-51.422067090480645
            if self.stress:
                stress_ev_ang3 = atoms.get_stress(voigt=False)
                self.cell_gradient = stress_to_grad(stress_ev_ang3,atoms.get_volume(), atoms.get_cell())
                print("Cell gradient:", self.cell_gradient)

            # Grab some other attributes if e.g. polarmace
            if self.polarmace:
                self.charges = self.model.results["charges"]
                print("PolarMACE: Getting charges:", self.charges)
                # dipole
                self.properties["dipole"] = self.model.results["dipole"]
                print("PolarMACE: Getting dipole:", self.properties["dipole"])

        # Older interface: suitable for loading user-trained regular MACE models
        else:
            # Call model to get energy
            from mace.cli.eval_configs import main
            from mace import data
            from mace.tools import torch_geometric, torch_tools, utils
            from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
            import torch

            # Charge and spin: only makes sense for mace_polar
            atoms.info["charge"] = charge
            atoms.info["spin"] = mult

            config = data.config_from_atoms(atoms)
            z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
            # Create dataloader
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[data.AtomicData.from_config(
                        config, z_table=z_table, cutoff=float(self.model.r_max), heads=None)],
                shuffle=False,
                drop_last=False)
            #
            # Get batch
            for batch in data_loader:
                batch = batch.to(self.platform)
            # Run model
            try:
                output = self.model(batch.to_dict(), compute_stress=self.stress, compute_force=Grad)
            except RuntimeError as e:
                print("RuntimeError occurred. Trying type changes. Message", e)
                self.model = self.model.float() # sometimes necessary to avoid type problems
                output = self.model(batch.to_dict(), compute_stress=self.stress, compute_force=Grad)
            print_time_rel(module_init_time, modulename=f'MACE run - after energy', moduleindex=2)
            # Grab energy
            en = torch_tools.to_numpy(output["energy"])[0]
            self.energy = float(en*ash.constants.evtohar)
            # Grad Boolean
            if Grad:
                self.gradient = torch_tools.to_numpy(output["forces"])/-51.422067090480645
                if self.stress:
                    stress_ev_ang3 = torch_tools.to_numpy(output["stress"][0])
                    self.cell_gradient = stress_to_grad(stress_ev_ang3,atoms.get_volume(), atoms.get_cell())
                    print("Cell gradient:",self.cell_gradient)

            # Hessian 
            if Hessian:
                print("Running Hessian")
                from mace.modules.utils import compute_hessians_vmap, compute_forces
                # 
                forces_tensor = compute_forces(output["energy"], batch["positions"])
                forces_np = torch_tools.to_numpy(forces_tensor)
                self.gradient = forces_np/-51.422067090480645
                # Calculate forces
                hess = compute_hessians_vmap(forces_tensor,batch["positions"])
                hessian = torch_tools.to_numpy(hess)
                print("hessian:", hessian)
                print_time_rel(module_init_time, modulename=f'MACE run - after hessian', moduleindex=2)
                self.hessian = hessian*0.010291772
        
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Special option
        if self.return_zero_gradient:
            print("Warning: return_zero_gradient option active")
            print("Returning zero gradient instead of real gradient")
            self.gradient = np.zeros((len(current_coords), 3))
            print("self.gradient:", self.gradient)

        # Return E and G if Grad
        if Grad:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy, self.gradient
        # Returning E only
        else:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy


###################################
# Function to write config file
###################################
# Hyper-parameters to modify
# model: MACE is default, ScaleShiftMACE another option but not suitable for bond-breaking reactions
# num_channels: size-of model. 128 is recommended, 256 (larger more accurate), 64 faster
# max_L: symmetry of messages. affects speed and accuracy. default 1 (compromise of speed/acc), 2 more accurate and slower, 0 is fast
# r_max: cutoff radius of local env. Recommended: 4-7 Ang
#NOTE: E0s="average" is easiest but not recommended.
def write_mace_config(config_file="config.yml", name="model",model="MACE", platform='cpu',device=None, 
                      valid_fraction=0.1, train_file="train_data_mace.xyz",E0s=None,
                      energy_key='energy_REF', forces_key='forces_REF',        
                      energy_weight=1, forces_weight=100,
                      max_num_epochs=500, swa=True, batch_size=10,
                      max_L = 0, r_max = 5.0, seed=42,
                      num_channels=128,
                      results_dir= "MACE_models", checkpoints_dir = "MACE_models", 
                      log_dir ="MACE_models", model_dir="MACE_models"):

    import yaml

    data = dict( model= model,
num_channels= num_channels,
max_L= max_L,
r_max= r_max,
name= name,
model_dir= model_dir,
log_dir= log_dir,
checkpoints_dir= checkpoints_dir,
results_dir= results_dir,
train_file= train_file,
valid_fraction= valid_fraction,
energy_key= energy_key,
forces_key= forces_key,
energy_weight=energy_weight,
forces_weight=forces_weight,
device= device,
seed= seed,
batch_size= batch_size,
max_num_epochs= max_num_epochs,
swa= swa)
    # E0 atomic energies should generally be in training set file.
    # Option to provide as dict here also or use E0="average"
    if E0s is not None:
        print("E0s option provided:", E0s)
        data[E0s] = E0s

    with open(config_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

def stress_to_grad(stress_ev_ang3,vol,cell):
    inv_cell_T = np.linalg.inv(cell).T
    grad_ev_ang = vol * np.dot(stress_ev_ang3, inv_cell_T)
    cell_gradient = grad_ev_ang * (0.5291772105638411 / 27.211386024367243)
    return cell_gradient