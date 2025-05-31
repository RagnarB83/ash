import time

from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader
from ash.modules.module_theory import Theory
from ash.interfaces.interface_xtb import create_xtb_pcfile_general, xtbpcgradientgrab
import numpy as np
import os
import shutil

##########################
# MLatom Theory interface
##########################
# NOTE: we mostly intend to support - ML functionality in MLatom (not basic QM methods)

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
                 method=None, ml_model=None, model_file=None, qm_program=None, ml_program=None, device='cpu',
                 verbose=2):
        module_init_time=time.time()
        super().__init__()
        self.theorynamelabel="MLatom"
        self.theorytype="QM"
        self.printlevel=printlevel
        self.numcores=numcores
        self.label=label
        print_line_with_mainheader(f"{self.theorynamelabel} initialization")

        try:
            #
            import mlatom as ml
        except ModuleNotFoundError as e:
            print("MLatom  requires installation of mlatom")
            print("See: http://mlatom.com/docs/installation.html")
            print("Try: pip install mlatom")
            print("You probably also have to do: pip install joblib, scipy torch torchani tqdm matplotlib statsmodels h5py pyh5md")
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
        self.verbose=verbose
        print("Checking if method or ml_model was selected")
        print("Method:", self.method)
        #############
        # METHOD
        #############
        if self.method is not None:
            if 'AIQM1' in self.method :
                print("An AIQM1 method was selected")
                print("Warning: this requires setting qm_program keyword as either mndo or sparrow.")
                print("Also dftd4 D4-dispersion program")
                if self.qm_program not in ["mndo","sparrow"]:
                    print("QM program keyword is neither mndo or sparrow. Not allowed, exiting.")
                    ashexit()
            elif 'AIQM2' in self.method :
                print("An AIQM2 method was selected")
                print("Warning: this requires setting qm_program keyword as either mndo or sparrow.")
                print("Also dftd4 D4-dispersion program")
                self.qm_program="xtb"
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


        #PC-gradient may be calculated
        self.pcgrad=None

        #############
        # ML_MODEL
        #############
        #Boolean to check whether training of this object has been done or not
        self.training_done=False
        #Multi-model run
        self.multi_model_run=False
        self.models=None
        if self.ml_model is not None:
            print("ml_model was selected:", ml_model)
            print("model_file:", model_file)
            print("ml_program:", ml_program)

            if isinstance(model_file,list):
                print("A list of modelfiles has been provided:")
                print("Enabling multi_model_run")
                self.multi_model_run=True
            # KREG
            # ml_program is either MLatomF or KREG_API
            if ml_model.lower() == 'kreg':
                print("KREG selected")
                if ml_program is None:
                    print("ml_program keyword was not set and is required for KREG. Options are: 'KREG_API' and 'MLatomF'. Exiting.")
                    print("Setting to MLatomF and continuing")
                    ml_program='MLatomF'
                if self.multi_model_run:
                    self.models=[]
                    for mfile in model_file:
                        self.models.append(ml.models.kreg(model_file=mfile, ml_program=ml_program, nthreads=self.numcores))
                else:
                    self.model = ml.models.kreg(model_file=model_file, ml_program=ml_program, nthreads=self.numcores)
            elif ml_model.lower() == 'ani':
                print("ANI selected")
                if  self.multi_model_run:
                    self.models=[]
                    for mfile in model_file:
                        self.models.append(ml.models.ani(model_file=mfile, verbose=self.verbose, device=self.device))
                else:
                    self.model = ml.models.ani(model_file=model_file, verbose=self.verbose, device=self.device)
            elif ml_model.lower() == 'dpmd':
                print("DMPD selected")
                # Testing to see if dp is found
                print("Searching for dp binary in PATH")
                if shutil.which('dp'):
                    deepmd_path = os.path.dirname(shutil.which('dp'))
                    print("Found dp binary in:", deepmd_path)
                    os.environ['DeePMDkit'] = deepmd_path
                    print("Setting DeePMDkit environment variable")
                else:
                    print("ASH could not find dp binary, implying DPMD has not been installed")
                    print("See DeePMD-kit website: https://docs.deepmodeling.com/projects/deepmd/en/stable/install/easy-install.html")
                    print("Might be as simple as: mamba install deepmd-kit lammps horovod")
                    print("and: pip install dpgen")
                    exit()
                if  self.multi_model_run:
                    self.models=[]
                    for mfile in model_file:
                        self.models.append(ml.models.dpmd(model_file=mfile, verbose=self.verbose))
                else:
                    self.model = ml.models.dpmd(model_file=model_file, verbose=self.verbose)
            elif ml_model.lower() == 'gap':
                print("GAP selected")
                if  self.multi_model_run:
                    self.models=[]
                    for mfile in model_file:
                        self.models.append(ml.models.gap(model_file=mfile, verbose=self.verbose))
                else:
                    self.model = ml.models.gap(model_file=model_file, verbose=self.verbose)
            elif ml_model.lower() == 'physnet':
                print("Physnet selected")
                if  self.multi_model_run:
                    self.models=[]
                    for mfile in model_file:
                        self.models.append(ml.models.physnet(model_file=mfile, verbose=self.verbose))
                else:
                    self.model = ml.models.physnet(model_file=model_file, verbose=self.verbose)
            elif ml_model.lower() == 'sgdml':
                print("SGDML selected")
                if  self.multi_model_run:
                    self.models=[]
                    for mfile in model_file:
                        self.models.append(ml.models.sgdml(model_file=mfile, verbose=self.verbos))
                else:
                    self.model = ml.models.sgdml(model_file=model_file, verbose=self.verbose)
            elif ml_model.lower() == 'mace':
                print("MACE selected")
                if  self.multi_model_run:
                    self.models=[]
                    for mfile in model_file:
                        self.models.append(ml.models.mace(model_file=mfile, verbose=self.verbose))
                else:
                    self.model = ml.models.mace(model_file=model_file, verbose=self.verbose)
            else:
                print("Unknown ml_model selected. Exiting")
                ashexit()
            if self.multi_model_run is False:
                print("MLatomTheory model created:", self.model)

            # model_file
            if self.multi_model_run:
                print("Multi-model run is active. Checking if all files exist")
                self.model_files = [] #New empty list of filenames
                for mfile in self.model_file:
                    file_present = os.path.isfile(mfile)
                    print(f"File {mfile} exists:", file_present)
                    print("Absolute path to model_file:", self.model_file)
                    if file_present is False:
                        print(f"File {mfile} does not exist. Exiting.")
                        ashexit()
                    # Storing absolute path of file
                    self.model_files.append(os.path.abspath(mfile))
                    print("Absolute path to model_file:", os.path.abspath(mfile))
            #else:
                #if isinstance(self.model_file,str):
                #    print(f"A filename-string ({self.model_file}) was provided, checking if file exists")
                #    file_present = os.path.isfile(self.model_file)
                #    print(f"File {self.model_file} exists:", file_present)
                #    if file_present is False:
                #        print(f"File {self.model_file} does not exist. Exiting.")
                #        ashexit()
                #    # Storing absolute path of file
                #    self.model_file=os.path.abspath(self.model_file)
                #    print("Absolute path to model_file:", self.model_file)
                #else:
                #    print(f"Error: model_file {self.model_file} is not a valid filename")
        # Initialization done
        print_time_rel(module_init_time, modulename='MLatom creation', moduleindex=2)

    def check_file_exists(self, file):
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
        # Storing absolute path of file
        #abs_path_file=os.path.abspath(file)
        #print("Absolute path to model_file:", abs_path_file)

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
              molDB_xyzvecproperty_file=None, split_fraction=[0.9, 0.1],
              property_to_learn='energy', xyz_derivative_property_to_learn='energy_gradients',
              hyperparameters={}):
        module_init_time=time.time()
        import mlatom as ml
        molDB = ml.data.molecular_database.from_xyz_file(filename = molDB_xyzfile)
        print(f"Created from file ({molDB_xyzfile}): a", molDB)

        #Learning energy by default
        molDB.add_scalar_properties_from_file(molDB_scalarproperty_file, property_to_learn)
        # If XYZvec-property-file provided then we are doing E+G
        if molDB_xyzvecproperty_file is not None:
            print("Training on both energies and gradients")
            molDB.add_xyz_vectorial_properties_from_file(molDB_xyzvecproperty_file, xyz_derivative_property_to_learn)
        else:
            print("Training on only energies")
            # Setting var to None (important)
            xyz_derivative_property_to_learn=None

        # Split
        if self.ml_model.lower() == 'kreg':
            print("KREG selected")
            print("Splitting molDB into subtraining database (subtrainDB) and validation database (valDB).")
            print("Split fraction:", split_fraction)
            subtrainDB, valDB = molDB.split(fraction_of_points_in_splits=split_fraction)
            print(f"subtrainDB {len(subtrainDB)}):", subtrainDB)
            print(f"valDB (size: {len(valDB)}):", valDB)

            print("hyperparameters:", hyperparameters)
            if len(hyperparameters) > 0:
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
            else:
                print("No hyperparameters provided (NOT recommended)")
            print("\nNow training...")
            self.model.train(molecular_database=molDB,
                            property_to_learn=property_to_learn,
                            xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)

        elif self.ml_model.lower() == 'gap':
            print("GAP selected, no splitting")
            print("\nNow training...")
            result = self.model.train(molecular_database=molDB,
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
                            property_to_learn=property_to_learn, hyperparameters=hyperparameters,
                            xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)

        self.training_done=True

        # Predicting 
        from mlatom.MLtasks import predicting, analyzing
        print("Now predicting for molDB")
        predicting(model=self.model, molecular_database=molDB, value=True, gradient=True)
        print("Now predicting for subtrainDB")
        predicting(model=self.model, molecular_database=subtrainDB, value=True, gradient=True)
        print("Now predicting for valDB")
        predicting(model=self.model, molecular_database=valDB, value=True, gradient=True)

        # Analyzing
        if molDB_xyzvecproperty_file is not None:
            self.result_molDB = analyzing(molDB, ref_value='energy', est_value='estimated_y', ref_grad='energy_gradients', est_grad='estimated_xyz_derivatives_y', set_name="molDB")
            self.result_subtrainDB = analyzing(subtrainDB, ref_value='energy', est_value='estimated_y', ref_grad='energy_gradients', est_grad='estimated_xyz_derivatives_y', set_name="subtrainDB")
            self.result_valDB = analyzing(valDB, ref_value='energy', est_value='estimated_y', ref_grad='energy_gradients', est_grad='estimated_xyz_derivatives_y', set_name="valDB")
        else:
            self.result_molDB = analyzing(molDB, ref_value='energy', est_value='estimated_y',  set_name="molDB")
            self.result_subtrainDB = analyzing(subtrainDB, ref_value='energy', est_value='estimated_y', set_name="subtrainDB")
            self.result_valDB = analyzing(valDB, ref_value='energy', est_value='estimated_y', set_name="valDB")

        print("Statistics saved as attributes:  result_molDB, result_subtrainDB and result_valDB of MLatomTheory object")
        print()
        print("self.result_molDB:", self.result_molDB)
        print("self.result_subtrainDB:", self.result_subtrainDB)
        print("self.result_valDB:", self.result_valDB)

        print_time_rel(module_init_time, modulename='MLatom train', moduleindex=2)

    # General run function
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        import mlatom as ml

        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # Prepare for run
        molecule = ml.data.molecule(charge, mult)
        molecule.read_from_numpy(coordinates=current_coords, species=np.array(qm_elems))

        # mlatom.models
        # Comp chem models, 3 types: methods (used as is), ml_model (requires training), model_tree_node (composite)
        if self.method is not None:
            print("A method was selected: ", self.method)
            print("QM program:", self.qm_program)
            print("Creating model")
            if self.method == "AIQM1":
                model = ml.models.methods(method=self.method, qm_program=self.qm_program)
            else:
                #AIQM2 
                model = ml.models.methods(method=self.method, program=self.qm_program)
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
        # Multi-run 
        elif self.models is not None:
            print("Multiple models present")
        # Single-file run
        elif self.ml_model is not None:
            print("A single ml_model was selected: ", self.ml_model)
            model = self.model
            if self.training_done is True:
                print("Training of MLatom model has been performed. Running should work")
            else:
                print("No training of this object has been done.")
                print("model file:", self.model_file)
                self.check_file_exists(self.model_file)
        else:
            print("No method or ml-model was defined yet.")
            ashexit()

        # Run
        print("Running MLatom singlepoint calculation")
        if self.multi_model_run:
            print("Multi-model run")
            energies=[]
            gradients=[]
            for i,model in enumerate(self.models):
                print("Running model from file:", self.model_files[i])
                self.check_file_exists(self.model_files[i])
                model.predict(molecule=molecule,calculate_energy=True,
                            calculate_energy_gradients=Grad,
                            calculate_hessian=False)
                energy = float(molecule.energy)
                print(f"Energy: {energy} Eh")
                energies.append(energy)
                if Grad:
                    print("Gradient was calculated.")
                    gradient = molecule.get_energy_gradients()
                    if self.printlevel > 2:
                        print(f"Gradient for model {i}:", gradient)
                    gradients.append(gradient)

            # Combining model results
            E_ave = np.mean(energies)
            E_std = np.std(energies)
            print(f"\nAverage model energy {E_ave} Eh")
            print(f"Standard deviation {E_std} Eh")
            self.energy = E_ave
            if Grad:
                self.gradient = np.mean( np.array(gradients), axis=0 )
        else:
            # PC
            if PC is True:
                print("Pointcharges provided. This is a QM/MM calculation")
                if self.qm_program == "xtb":
                    print("QM-program is xtb. Writing pointcharge file")
                    create_xtb_pcfile_general(current_MM_coords, MMcharges)
                else:
                    print("PC-embedded QM/MM calculations are right now only possible if qm_program is xtb (e.g. AIQM2)")
                    ashexit()
            model.predict(molecule=molecule,calculate_energy=True,
                        calculate_energy_gradients=Grad,
                        calculate_hessian=False)
            self.energy = float(molecule.energy)
            print("Single-point MLatom energy:", self.energy)
            if Grad:
                print("Gradient was calculated.")
                self.gradient = molecule.get_energy_gradients()
                if PC is True and self.qm_program == "xtb":
                    self.pcgrad = xtbpcgradientgrab(len(MMcharges), file="molecule0.pcgrad")

        print_time_rel(module_init_time, modulename='MLatom run', moduleindex=2)
        if Grad:
            # If pcgrad was actually calculated
            if self.pcgrad is not None:
                return self.energy,self.gradient, self.pcgrad
            else:
                return self.energy,self.gradient
        else:
            return self.energy

