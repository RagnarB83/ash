import time
import numpy as np

from ash.modules.module_coords import elemstonuccharges
from ash.functions.functions_general import ashexit, BC,print_time_rel
from ash.functions.functions_general import print_line_with_mainheader
import ash.constants



class MaceTheory():
    def __init__(self, filename="torch.pt", model_name=None, model_object=None,
                 model_file=None, printlevel=2, label="MaceTheory", numcores=1,
                 platform=None, train=False):
        # Early exits
        #try:
        #    import torch
        #except ImportError:
        #    print("Problem importing torch library. Make sure you have installed torch correctly")
        #    ashexit()

        if model_name is None and model_object is None and model_file is None and train is False:
            print("Error: No model_name, model_object or model_file was selected and train is False.")
            print("Either give as input:")
            print("1. pretrained model_file as input (model_file keyword")
            print("2. give a valid model_name (e.g. ANI2x, requires TorchANI installed")
            print("3. give a Torch NN object as input (model_object keyword")
            print("4. set train keyword to True")
            ashexit()

        self.theorytype = 'QM'
        self.theorynamelabel = 'Torch'
        self.label = label
        self.analytic_hessian = True
        self.numcores = numcores
        self.filename = filename
        self.printlevel = printlevel
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        #try:
        #     import mace:
        #except ImportError as e:
        #    print("Import error message:", e)
        #    print("Problem importing mace or something associated. Make sure you have installed things correctly.")
        #    print("Example: pip install mace-torch ")
        #    ashexit()


        import torch
        # Device choice
        if platform == 'cuda':
            print("Platfrom CUDA selected. Will attempt to use.")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif platform == 'mps':
            print("Platfrom MPS selected. Will use.")
            self.device = torch.device('mps')
        else:
            print("Unrecognized platform. Choosing CPU")
            self.device = torch.device('cpu')
        print("Torch device selected:", self.device)




    def cleanup(self):
        print("No cleanup implemented")

    def set_numcores(self,numcores):
        self.numcores=numcores

    def train(self):
        print("Training not ready")
        ashexit()
        # TODO
        #Distinguish between pure PyTorch training and TorchANI training

    def train_ani(self):
        pass
    def load_model(self,model_file):
        import torch
        # sTODO: weights only option ?
        #self.model = torch.jit.load(model_file)
        self.model = torch.load(model_file, map_location=torch.device('cpu'))

    def save_model(self,filename=None, index=None):
        import torch
        if index is not None:
            print(f"Saving only index: {index} of ensemble model")
            compiled_model = torch.jit.script(self.model[index])
        else:
            compiled_model = torch.jit.script(self.model)
        if filename is None:
            filename=self.filename
        torch.jit.save(compiled_model, filename)
        print("Torch saved model to file:", filename)


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

        # Load Torch
        import torch

        # Early exits
        # Coords provided to run
        if current_coords is None:
            print("no coordinates. Exiting")
            ashexit()
        # Check availability of model before proceeding further
        if self.model is None:
            print("MaceTheory Model has not been defined.")
            print("Either load a valid model or train")
            ashexit()

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # Call model to get energy

        # AIMNet2
        if 'aimnet2' in str(self.model).lower():
            input_data = {'coord':current_coords, 'numbers':elemstonuccharges(qm_elems), 'charge':charge, 'mult':mult}
            results = self.model(input_data, forces=Grad, stress=False, hessian=False)
            # print("results:", results)
            self.energy = results["energy"].item() / ash.constants.hartoeV
            if Grad:
                self.gradient = -1*(0.03674932217565499/1.88972612546)*results["forces"].detach().cpu().numpy()
            if Hessian:
                self.hessian = (0.03674932217565499/1.88972612546/1.88972612546)*results["hessian"].detach().cpu().numpy()
        # TorchANI
        else:
            # Converting coordinates and element information to Torch tensors
            # dtype has to be float32 to get compatible Tensor for coordinate and nuc-charges
            coords_torch = torch.tensor(np.array([current_coords], dtype='float32'), requires_grad=True, device=self.device)
            nuc_charges_torch = torch.tensor(np.array([elemstonuccharges(qm_elems)]), device=self.device)
            print("coords_torch:", coords_torch)
            print("nuc_charges_torch:", nuc_charges_torch)
            energy_tensor = self.model((nuc_charges_torch, coords_torch))
            print("here")
            print("energy_tensor:", energy_tensor)
            exit()
            
            energy_tensor = self.model((nuc_charges_torch, coords_torch)).energies
            self.energy = energy_tensor.item()

            # Call Grad
            if Grad:
                gradient_tensor = torch.autograd.grad(energy_tensor.sum(), coords_torch)[0]
                self.gradient = gradient_tensor.detach().cpu().numpy()[0]

            if Hessian:
                print("Calculating Hessian (requires torchani)")
                import torchani
                self.hessian = torchani.utils.hessian(coords_torch , energies=energy_tensor)

        if PC is True:
            print("PC not supported yet")
            ashexit()

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


def write_mace_config(name="model",model="MACE", device='cpu',
                      train_file="train_data.xyz", valid_file="trainvalid_data.xyz", test_file="traintest_data.xyz",
                      max_num_epochs=500, 
                      num_channels=32,  results_dir= "MACE_models", checkpoints_dir = "MACE_models", log_dir ="MACE_models",
                      model_dir="MACE_models",
                      swa=True, seed=123, batch_size=10, ):

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
valid_file= valid_file,
test_file= test_file,
energy_key= "energy_xtb",
forces_key= "forces_xtb",
device= device,
batch_size= batch_size,
max_num_epochs= max_num_epochs,
swa= swa,
seed= seed,
    )

    with open('data.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)