import time
import numpy as np

from ash.modules.module_coords import elemstonuccharges
from ash.functions.functions_general import ashexit, BC,print_time_rel
from ash.functions.functions_general import print_line_with_mainheader
import ash.constants
# Simple interface to MACE for both using and training

class MACETheory():
    def __init__(self, config_filename="config.yml", 
                 filename="mace.model", model_file=None, printlevel=2, 
                 label="MACETheory", numcores=1, device="cpu"):
        # Early exits
        try:
            import mace
        except ImportError:
            print("Problem importing mace. Make sure you have installed mace-correctly")
            print("Most likely you need to do: pip install mace-torch")
            print("Also recommended: pip install cuequivariance_torch")

        self.theorytype = 'QM'
        self.theorynamelabel = 'MACE'
        self.label = label
        self.analytic_hessian = True
        self.numcores = numcores
        self.config_filename=config_filename
        self.filename = filename
        self.printlevel = printlevel

        # Model attribute is None until we have loaded a model
        self.model=None

        self.model_file=model_file
        self.device=device.lower()

        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        self.training_done=False

    def cleanup(self):
        print("No cleanup implemented")

    def set_numcores(self,numcores):
        self.numcores=numcores

    def train(self, config_file="config.yml", name="model",model="MACE", device='cpu',
                      valid_fraction=0.1, train_file="train_data_mace.xyz",E0s=None,
                      energy_key='energy_REF', forces_key='forces_REF',        
                      energy_weight=1, forces_weight=1,
                      max_num_epochs=500, swa=True, batch_size=10,
                      max_L = 0, r_max = 5.0, num_channels=32,  
                      results_dir= "MACE_models", checkpoints_dir = "MACE_models", 
                      log_dir ="MACE_models", model_dir="MACE_models"):
        module_init_time=time.time()


        self.train_file=train_file
        self.valid_fraction=valid_fraction

        print("Training activated")
        print("Training parameters:")
        print("config_file", config_file)
        print("name:", model)
        print("model:", model)
        print("device:", device)
        print("Validation set fraction (valid_fraction):", valid_fraction)
        print("train_file:", self.train_file)
        print("E0s:", E0s)
        print("energy_key:", energy_key)
        print("forces_key:", forces_key)
        print("energy_weight:", energy_weight)
        print("forces_weight:", forces_weight)
        print("max_num_epochs:", max_num_epochs)
        print("swa:", swa)
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
        write_mace_config(config_file=config_file, name=name, model=model, device=device, 
                      valid_fraction=valid_fraction, train_file=self.train_file,E0s=E0s,
                      energy_key=energy_key, forces_key=forces_key,        
                      energy_weight=energy_weight, forces_weight=forces_weight,
                      max_num_epochs=max_num_epochs, swa=swa, batch_size=batch_size,
                      max_L = max_L, r_max = r_max, 
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
        self.model_file=f"{results_dir}/{name}_stagetwo_compiled.model"
        print("Setting model_file attribute to:", self.model_file )
        print("MACETheory object can now be used directly.")

        # If we train with a specific device we would want to use that same device for evaluation/prediction
        self.device=device
        print("Setting device of object to be ", self.device)

        #Load model
        self.model_load()

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
    # Get statistics for training, sub-training and validation set
    #def get_statistics():

        #FIle ./valid_indices_123.txt contains indices of training set that are validation
        #Read training file #self.train_file
        #Get validation set. Convert data into Eh and Eh/Bohr
        #Create dict: valDB
        #

    #    from mlatom.MLtasks analyzing
    #
    #    self.result_molDB = analyzing(valDB, ref_value='energy', est_value='estimated_y', ref_grad='energy_gradients', 
    #                                  est_grad='estimated_xyz_derivatives_y', set_name="valDB")

    def model_load(self):
        module_init_time=time.time()
        import torch
        # Load model
        print(f"Loading model from file {self.model_file}. Device is: {self.device}")
        self.model = torch.load(f=self.model_file, map_location=torch.device(self.device))
        self.model = self.model.to(self.device)  # for possible cuda problems
        print_time_rel(module_init_time, modulename=f'MACE model-load', moduleindex=2)

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

        # Check availability of model before proceeding further
        if self.model_file is None:
            print("MACETheory model_file has not been defined.")
            print("Either load a valid model or train a model")
            ashexit()
        # Checking if file exists
        self.check_file_exists(self.model_file)

        # Call model to get energy
        from mace.cli.eval_configs import main
        from mace import data
        from mace.tools import torch_geometric, torch_tools, utils
        from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
        import torch
        from mace.modules.utils import compute_hessians_vmap, compute_hessians_loop, compute_forces

        if self.model is None:
            print("Model has not been loaded yet.")
            self.model_load()

        # Simplest to use ase here to create Atoms object
        import ase
        atoms = ase.atoms.Atoms(qm_elems,positions=current_coords)
        config = data.config_from_atoms(atoms)
        z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        # Create dataloader
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=float(self.model.r_max), heads=None)],
            shuffle=False,
            drop_last=False)
        #
        option_1=True
        if option_1:
            # Get batch
            for batch in data_loader:
                batch = batch.to(self.device)
            # Run model
            try:
                output = self.model(batch.to_dict(), compute_stress=False, compute_force=False)
                print_time_rel(module_init_time, modulename=f'MACE run - after energy', moduleindex=2)
            except RuntimeError as e:
                print("RuntimeError occurred. Trying type changes. Message", e)
                self.model = self.model.float() # sometimes necessary to avoid type problems
                output = self.model(batch.to_dict(), compute_stress=False, compute_force=False)
            # Grab energy
            en = torch_tools.to_numpy(output["energy"])[0]
            # Calculate forces
            forces = compute_forces(output["energy"], batch["positions"])
            print_time_rel(module_init_time, modulename=f'MACE run - after forces', moduleindex=2)
            forces = torch_tools.to_numpy(forces)

            # Hessian 
            if Hessian:
                print("Running Hessian")
                hess = compute_hessians_vmap(forces,batch["positions"])
                hessian = torch_tools.to_numpy(hess)
                print("hessian:", hessian)
                print_time_rel(module_init_time, modulename=f'MACE run - after hessian', moduleindex=2)

        # This worked previously
        else:
            print("previous regular mode")
            for batch in data_loader:
                try:
                    output = self.model(batch.to_dict(), compute_stress=False, compute_force=True)
                except RuntimeError as e:
                    print("RuntimeError occurred. Trying type changes. Message", e)
                    self.model = self.model.float() # sometimes necessary to avoid type problems
                    output = self.model(batch.to_dict(), compute_stress=False, compute_force=True)

            # Get energy and forces
            en = torch_tools.to_numpy(output["energy"])[0]
            forces = np.split(
                torch_tools.to_numpy(output["forces"]),
                indices_or_sections=batch.ptr[1:],
                axis=0)[0]
        # Convert energy and forces to Eh and gradient in Eh/Bohr
        self.energy = float(en*ash.constants.evtohar)
        self.gradient = forces/-51.422067090480645
        if Hessian:
            self.hessian = hessian*0.010291772
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Return E and G if Grad
        if Grad is True:
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
#NOTE: E0s="average" is easiest but not recommended. Can provide 
##todo: seed
def write_mace_config(config_file="config.yml", name="model",model="MACE", device='cpu', 
                      valid_fraction=0.1, train_file="train_data_mace.xyz",E0s=None,
                      energy_key='energy_REF', forces_key='forces_REF',        
                      energy_weight=1, forces_weight=100,
                      max_num_epochs=500, swa=True, batch_size=10,
                      max_L = 0, r_max = 5.0, 
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