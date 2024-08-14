import time

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader
from ash.modules.module_theory import Theory
import numpy as np
import os
import shutil

##########################
# MLatom Theory interface
##########################
# NOTE: we only intend to support mostly ML functionality in MLatom (not all basic QM methods)

# MLatom has 3 types of models:
# 1) methods: these are pre-trained or at least standalone electronic structure methods
# 2) ml_model: these are models that require training
# 3) model_tree_node: some composite models (unclear)

# methods examples: AIQM1, AIQM1@DFT, AIQM1@DFT*, ANI-1ccx, ANI-1x, ANI-1x-D4, ANI-2x, ANI-2x-D4
# ml_model examples: ani, dpmd, gap, physnet, sgdml

# MLatom may use/require interfaces to : TorchANI, DeepMD-kit, GAP/QUIP, Physnet, sGDML

# NOTES on interface:
# AIQMx methods require either MNDO or Sparrow as QM-program. Also dftd4.
# Sparrow lacks AIQMx gradient (for ODM2 part), only energies available.



class MLatomTheory(Theory):
    def __init__(self, printlevel=2, numcores=1, label="mlatom",
                 method=None, ml_model=None, model_file=None, qm_program=None, ml_program=None, device='cpu'):
        module_init_time=time.time()
        super().__init__()
        self.theorynamelabel="MLatom"
        self.theorytype="QM"
        self.printlevel=printlevel
        self.label=label
        print_line_with_mainheader(f"{self.theorynamelabel} initialization")

        try:
            #
            import mlatom as ml
        except ModuleNotFoundError as e:
            print("MLatom  requires installation of mlatom")
            print("See: http://mlatom.com/docs/installation.html")
            print("Try: pip install mlatom")
            print("You probably also have to do: pip install scipy torch torchani tqdm matplotlib statsmodels h5py pyh5md")
            print()
            print("Error message:", e)
            ashexit()

        # EARLY EXITS

        if method is None and ml_model is None:
            print("Neither a method or ml_model was selected for MLatomTheory interface. Exiting.")
            ashexit()



        # METHODS: pre-trained models
        # Note: useful method keywords in MLAatom below
        # AIQMx models require either MNDO or Sparrow. Also dftd4 (except AIQM1@DFT* but that one is bad anyway)
        # 'AIQM1', 'AIQM1@DFT', 'AIQM1@DFT*',
        # ANI models: will download parameters automatically
        # 'ANI-1ccx', 'ANI-1x', 'ANI-1x-D4', 'ANI-2x', 'ANI-2x-D4'
        self.method = method
        self.qm_program = qm_program
        self.ml_program = ml_program
        self.ml_model = ml_model
        self.model_file=model_file
        self.device = device  # 'cpu' or 'cuda' (used by Torch-based models/methods)
        print("Checking if method or ml_model was selected")
        print("Method:", self.method)
        #############
        # METHOD
        #############
        if self.method is not None:
            if 'AIQM' in self.method :
                print("An AIQMx method was selected")
                print("Warning: this requires setting qm_program keyword as either mndo or sparrow.")
                print("Also dftd4 D4-dispersion program")
                if self.qm_program not in ["mndo","sparrow"]:
                    print("QM program keyword is neither mndo or sparrow. Not allowed, exiting.")
                    ashexit()
            elif 'ANI' in self.method:
                print("An ANI type method was selected")
                print("This requires TorchANI and pytorch")
                print("Note: ANI parameters will be downloaded automatically if needed")
            elif 'ODM' in self.method:
                print("A ODMx type semi-empirical method was selected. This requires MNDO")
            elif 'OM' in self.method:
                print("A OMx type semi-empirical method was selected. This requires MNDO")
            elif 'AIMNet' in self.method:
                print("A AIMNet type method was selected")
            else:
                print(f"Either an invalid  {self.method} or unknown method (to MLatomTheory interface) was selected. Exiting.")
                ashexit()

        print("QM program:", self.qm_program)

        #############
        # QM-PROGRAM
        #############
        if self.qm_program == 'mndo':
            self.setup_mndo()
            self.setup_dftd4()

        elif self.qm_program == 'sparrow':
            self.setup_sparrow()
            self.setup_dftd4()

        #############
        # ML_MODEL
        #############
        #Boolean to check whether training of this object has been done or not
        self.training_done=False
        if self.ml_model is not None:
            print("ml_model was selected:", ml_model)
            print("model_file:", model_file)
            print("ml_program:", ml_program)

            # KREG
            # ml_program is either MLatomF or KREG_API
            if ml_model.lower() == 'kreg':
                print("KREG selected")
                if ml_program is None:
                    print("ml_program keyword was not set and is required for KREG. Options are: 'KREG_API' and 'MLatomF'. Exiting.")
                    print("Setting to MLatomF and continuing")
                    ml_program='MLatomF'
                self.model = ml.models.kreg(model_file=model_file, ml_program=ml_program)
            elif ml_model.lower() == 'ani':
                print("ANI selected")
                self.model = ml.models.ani(model_file=model_file)
                print(self.model)
            elif ml_model.lower() == 'dpmd':
                print("DMPD selected")
                self.model = ml.models.dpmd(model_file=model_file)
            elif ml_model.lower() == 'gap':
                print("GAP selected")
                self.model = ml.models.gap(model_file=model_file)
            elif ml_model.lower() == 'physnet':
                print("Physnet selected")
                self.model = ml.models.physnet(model_file=model_file)
            elif ml_model.lower() == 'sgdml':
                print("SGDML selected")
                self.model = ml.models.sgdml(model_file=model_file)
            elif ml_model.lower() == 'mace':
                print("MACE selected")
                self.model = ml.models.mace(model_file=model_file)
            else:
                print("Unknown ml_model selected. Exiting")
                ashexit()
            print("MLatomTheory model created:", self.model)

            # model_file
            if self.model_file is not None:
                print("model_file:", self.model_file)
                file_present = os.path.isfile(self.model_file)
                print("File exits:", file_present)
                # Storing absolute path of file
                self.model_file=os.path.abspath(self.model_file)
                print("Absolute path to model_file:", self.model_file)

        # Initialization done
        print_time_rel(module_init_time, modulename='MLatom creation', moduleindex=2)

    def setup_mndo(self):
        print("QM program is mndo")
        print("Make sure executable mndo is in your environment")
        print("See https://mndo.kofo.mpg.de about MNDO licenses")
        try:
            mndodir = os.path.dirname(shutil.which('mndo2020'))
            os.environ['mndobin'] = mndodir+"/mndo2020"
            print("Found mndo2020 executable in:", mndodir)
        except TypeError:
            print("Found no mndo2020 executable in your environment. Exiting.")
            ashexit()
    def setup_dftd4(self):
        print("DFTD4 needed. Making sure executable dftd4 is in your environment")
        try:
            dftd4dir = os.path.dirname(shutil.which('dftd4'))
            os.environ['dftd4bin'] = dftd4dir+"/dftd4"
            print("Found dftd4 executable in:", dftd4dir)
        except TypeError:
            print("Found no dftd4 executable in your environment. Exiting.")
            ashexit()
    def setup_sparrow(self):
        print("QM program is sparrow")
        print("Make sure executable sparrow is in your environment")
        print("See https://github.com/qcscine/sparrow. Possible installation  via: conda install scine-sparrow-python")
        print("Also make sure dftd4 (https://github.com/dftd4/dftd4) is in your environment. Possible installation  via:  conda install dftd4")
        print("Warning: sparrow lacks AIQMx gradient (for ODM2 part), only energies available.")
        try:
            sparrowdir = os.path.dirname(shutil.which('sparrow'))
            os.environ['sparrowbin'] = sparrowdir+"/sparrow"
            print("Found sparrow executable in:", sparrowdir)
        except TypeError:
            print("Found no sparrow executable in your environment. Exiting.")
            ashexit()

    def train(self, molDB_xyzfile=None, molDB_scalarproperty_file=None,
              molDB_xyzvecproperty_file=None, split_DB=False, split_fraction=[0.9, 0.1],
              property_to_learn='energy', xyz_derivative_property_to_learn='energy_gradients',
              hyperparameters=None):

        import mlatom as ml
        molDB = ml.data.molecular_database.from_xyz_file(filename = molDB_xyzfile)
        print(f"Created from file ({molDB_xyzfile}): a", molDB)

        molDB.add_scalar_properties_from_file(molDB_scalarproperty_file, property_to_learn)
        if xyz_derivative_property_to_learn == 'energy_gradients':
            molDB.add_xyz_vectorial_properties_from_file(molDB_xyzvecproperty_file, xyz_derivative_property_to_learn)

        # Split
        if self.ml_model.lower() == 'kreg':
            print("KREG selected")
            print("Splitting molDB into subtraining database (subtrainDB) and validation database (valDB).")
            print("Split fraction:", split_fraction)
            subtrainDB, valDB = molDB.split(fraction_of_points_in_splits=split_fraction)
            print(f"subtrainDB {len(subtrainDB)}):", subtrainDB)
            print(f"valDB (size: {len(valDB)}):", valDB)

            if hyperparameters is not None:
                print("Hyperparameters provided:", hyperparameters)
                # optimize its hyperparameters
                if 'sigma' in hyperparameters:
                    self.model.hyperparameters['sigma'].minval = 2**-5 # modify the default lower bound of the hyperparameter sigma
                self.model.optimize_hyperparameters(subtraining_molecular_database=subtrainDB,
                                                validation_molecular_database=valDB,
                                                optimization_algorithm='grid',
                                                hyperparameters=hyperparameters,
                                                training_kwargs={'property_to_learn': property_to_learn, 'prior': 'mean'},
                                                prediction_kwargs={'property_to_predict': 'estimated_energy'})
                if 'lambda' in hyperparameters:
                    lmbd = self.model.hyperparameters['lambda'].value
                if 'sigma' in hyperparameters:
                    sigma = self.model.hyperparameters['sigma'].value
                valloss = self.model.validation_loss

                if 'sigma' in hyperparameters:
                    print('Optimized sigma:', sigma)
                if 'lambda' in hyperparameters:
                    print('Optimized lambda:', lmbd)
                print('Optimized validation loss:', valloss)

            print("\nNow training...")
            self.model.train(molecular_database=molDB,
                            property_to_learn=property_to_learn,
                            xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)

        elif self.ml_model.lower() == 'gap':
            print("GAP selected, no splitting")
            print("\nNow training...")
            self.model.train(molecular_database=molDB,
                            property_to_learn=property_to_learn,
                            xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
        else:
            print("Splitting molDB into subtraining database (subtrainDB) and validation database (valDB).")
            print("Split fraction:", split_fraction)
            subtrainDB, valDB = molDB.split(fraction_of_points_in_splits=split_fraction)
            print("subtrainDB:", subtrainDB)
            print("valDB:", valDB)

            print("\nNow training...")
            self.model.train(molecular_database=molDB, validation_molecular_database=valDB,
                            property_to_learn=property_to_learn,
                            xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)

        self.training_done=True

    # General run function
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        import mlatom as ml

        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Prepare for run
        molecule = ml.data.molecule(charge, mult)
        molecule.read_from_numpy(coordinates=current_coords, species=np.array(elems))

        # mlatom.models
        # Comp chem models, 3 types: methods (used as is), ml_model (requires training), model_tree_node (composite)
        if self.method is not None:
            print("A method was selected: ", self.method)
            print("QM program:", self.qm_program)
            print("Creating model")
            model = ml.models.methods(method=self.method, qm_program=self.qm_program, device=self.device)
            # Create dftd4.json file before running if required
            if 'AIQM' in self.method:
                print("An AIQMx method was selected")
                #NOTE: dftd4 interface of MLatom has a bug for current dftd4 release
                if 'AIQM1@DFT*' in self.method:
                    print("AIQM1@DFT* method was selected, no disp. correction needed")
                elif 'AIQM1@DFT' in self.method:
                    print("AIQM1@DFT method was selected. Disp. correction needed")
                elif 'AIQM1' in self.method:
                    print("AIQM1 method was selected. Disp. correction needed")

        elif self.ml_model is not None:
            print("A ml_model was selected: ", self.ml_model)
            model = self.model
            if self.training_done is True:
                print("Training of MLatom model has been performed. Running should work")
            else:
                print("No training of this object has been done.")
                print("model file:", self.model_file)
            if self.model_file is not None:
                print("A modelfile was specified when MLatomTheory object was created. Checking if it exists")
                file_present = os.path.isfile(self.model_file)
                print("File exits:", file_present)
                if file_present is False:
                    print("File does not exist. Exiting.")
                    ashexit()
        else:
            print("No method or ml-model was defined yet.")
            ashexit()

        # Run
        if PC is True:
            print("PC is not yet supported by MLatomTheory interface")
            # Note: MNDO should support PCs, not sure about sparrow
            ashexit()
        else:

            if Grad is True:
                print("Running MLatom Energy + Gradient calculation")
                model.predict(molecule=molecule,calculate_energy=True,
                            calculate_energy_gradients=True,
                            calculate_hessian=False)
                self.energy = float(molecule.energy)
                self.gradient = molecule.get_energy_gradients()
                print("Single-point MLatom energy:", self.energy)

                print_time_rel(module_init_time, modulename='MLatom run', moduleindex=2)
                return self.energy,self.gradient

            else:
                print("Running MLatom Energy calculation")
                model.predict(molecule=molecule, calculate_energy=True)
                self.energy = molecule.energy
                print("Single-point MLatom energy:", self.energy)
                #std = molecule.aiqm1_nn.energy_standard_deviation
                #print("std:", std)

                print_time_rel(module_init_time, modulename='MLatom run', moduleindex=2)
                return self.energy
