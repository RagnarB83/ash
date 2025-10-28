import os
import glob
import shutil
import random
import numpy as np
import statistics

from ash.modules.module_coords import write_xyzfile, split_multimolxyzfile
from ash.functions.functions_general import natural_sort,ashexit,listdiff
from ash import Singlepoint, Fragment
from ash.interfaces.interface_mdtraj import MDtraj_slice
from ash.modules.module_plotting import ASH_plot

# Collection of functions related to machine learning and data analysis
# Also helper tools for Torch and MLatom interfaces

# Function to create ML training data given XYZ-files and 2 ASH theories
def create_ML_training_data(xyz_dir=None, dcd_trajectory=None, xyz_trajectory=None, xyz_files=None, num_snapshots=None, random_snapshots=True,
                                dcd_pdb_topology=None, nth_frame_in_traj=1, printlevel=2,
                               theory_1=None, theory_2=None, charge=0, mult=1, Grad=True, runmode="serial", numcores=1):
    print("-"*50)
    print("create_ML_training_data function")
    print("-"*50)
    if xyz_dir is None and xyz_trajectory is None and xyz_files is None and dcd_trajectory is None:
        print("Error: create_ML_training_data requires xyz_dir, xyz_trajectory,xyz_files or dcd_trajectory option to be set!")
        ashexit()

    if theory_1 is None:
        print("create_ML_training_data requires theory_1 and theory_2 to be set")
        exit()
    if theory_1 is not None and theory_2 is not None:
        print("Two theory levels selected. A delta-learning training set will be created.")
        print("Note: Theory 2 is assumed to be the higher level theory.")
        delta=True
    else:
        delta=False

    print("xyz_dir:", xyz_dir)
    print("xyz_trajectory:", xyz_trajectory)
    print("xyz_files:",xyz_files)
    print("dcd_trajectory:", dcd_trajectory)
    print("Charge:", charge)
    print("Mult:", mult)
    print("Grad:", Grad)

    print(f"num_snapshots: {num_snapshots} # xyz_trajectory option: number of snapshots to use")
    print(f"random_snapshots: {random_snapshots} # xyz_trajectory option: randomize snapshots or not")


    print("Theory 1:", theory_1)
    print("Theory 2:", theory_2)

    # XYZ DIRECTORY
    if xyz_dir is not None:
        print("XYZ-dir specified.")
        full_list_of_xyz_files=glob.glob(f"{xyz_dir}/*.xyz")
        print("Number of XYZ-files in directory:", len(full_list_of_xyz_files))
        if num_snapshots is None:
            print("num_snapshots has not been set by user")
            print("This means that we will take all snapshots")
            print("Setting num_snapshots to:", len(full_list_of_xyz_files))
            num_snapshots=len(full_list_of_xyz_files)
        print(f"Number of snapshots (num_snapshots keyword) set to {num_snapshots}")
        if random_snapshots is True:
            print(f"random_snapshots is True. Taking {num_snapshots} random XYZ snapshots.")
            list_of_xyz_files = random.sample(full_list_of_xyz_files, num_snapshots)
        else:
            print("random_snapshots is False. Taking the first", num_snapshots, "snapshots.")
            list_of_xyz_files=full_list_of_xyz_files[:num_snapshots]
        print(f"List of XYZ-files to use (num {len(list_of_xyz_files)}):", list_of_xyz_files)

    # DCD TRAJECTORY
    elif dcd_trajectory is not None:
        print("DCD-trajectory specified.")
        #Getting absolute path of file
        dcd_trajectory=os.path.abspath(dcd_trajectory)
        print("Absolute path of DCD-trajectory:", dcd_trajectory)
        print("Now splitting the DCD trajectory into individual XYZ-files.")
        try:
            shutil.rmtree("./xyz_traj_split")
            print("Deleted old xyz_traj_split")
        except:
            pass
        os.mkdir("xyz_traj_split")
        print("Splitting DCD trajectory using mdtraj")
        #Converting DCD traj into XYZ traj
        xyztraj = MDtraj_slice(dcd_trajectory, dcd_pdb_topology, format='XYZ', frames="all")
        # Splitting XYZ
        os.chdir("xyz_traj_split")
        split_multimolxyzfile(f"../{xyztraj}", writexyz=True, skipindex=nth_frame_in_traj,return_fragments=False)        
        # Getting list of XYZ-files in directory, properly sorted
        full_list_of_xyz_files=natural_sort(glob.glob(f"molecule*.xyz"))
        print("Created directory xyz_traj_split")
        print("Number of XYZ-files in directory:", len(full_list_of_xyz_files))
        if num_snapshots is None:
            print("num_snapshots has not been set by user")
            print("This means that we will take all snapshots")
            print("Setting num_snapshots to:", len(full_list_of_xyz_files))
            num_snapshots=len(full_list_of_xyz_files)
        
        print("Note: This directory can be used as a directory for the xyz_dir option to create_ML_training_data")
        print()
        print(f"Number of snapshots (num_snapshots keyword) set to {num_snapshots}")
        if random_snapshots is True:
            print(f"random_snapshots is True. Taking {num_snapshots} random XYZ snapshots.")
            list_of_xyz_files = random.sample(full_list_of_xyz_files, num_snapshots)
        else:
            print("random_snapshots is False. Taking the first", num_snapshots, "snapshots.")
            list_of_xyz_files=full_list_of_xyz_files[:num_snapshots]
        print(f"List of XYZ-files to use (num {len(list_of_xyz_files)}):", list_of_xyz_files)

        # Prepending path
        list_of_xyz_files=[f"./xyz_traj_split/{f}" for f in list_of_xyz_files]
        print(list_of_xyz_files)
        os.chdir('..')

    # XYZ-files
    elif xyz_files is not None:
        print("XYZ-files specified.")
        full_list_of_xyz_files=xyz_files
        print("Number of XYZ-files specified:", len(full_list_of_xyz_files))
        if num_snapshots is None:
            print("num_snapshots has not been set by user")
            print("This means that we will take all snapshots")
            print("Setting num_snapshots to:", len(full_list_of_xyz_files))
            num_snapshots=len(full_list_of_xyz_files)
        print(f"Number of snapshots (num_snapshots keyword) set to {num_snapshots}")
        if random_snapshots is True:
            print(f"random_snapshots is True. Taking {num_snapshots} random XYZ snapshots.")
            list_of_xyz_files = random.sample(full_list_of_xyz_files, num_snapshots)
        else:
            print("random_snapshots is False. Taking the first", num_snapshots, "snapshots.")
            list_of_xyz_files=full_list_of_xyz_files[:num_snapshots]
        print(f"List of XYZ-files to use (num {len(list_of_xyz_files)}):", list_of_xyz_files)

    # XYZ TRAJECTORY
    elif xyz_trajectory is not None:
        print("XYZ-trajectory specified.")
        #Getting absolute path of file
        xyz_trajectory=os.path.abspath(xyz_trajectory)
        print("Absolute path of XYZ-trajectory:", xyz_trajectory)
        print("Now splitting the trajectory into individual XYZ-files.")
        try:
            shutil.rmtree("./xyz_traj_split")
            print("Deleted old xyz_traj_split")
        except:
            pass
        print("here")
        os.mkdir("xyz_traj_split")
        os.chdir("xyz_traj_split")

        print("Splitting trajectory")
        split_multimolxyzfile(xyz_trajectory, writexyz=True, skipindex=1,return_fragments=False)
        # Getting list of XYZ-files in directory, properly sorted
        full_list_of_xyz_files=natural_sort(glob.glob(f"molecule*.xyz"))
        if num_snapshots is None:
            print("num_snapshots has not been set by user")
            print("This means that we will take all snapshots")
            print("Setting num_snapshots to:", len(full_list_of_xyz_files))
            num_snapshots=len(full_list_of_xyz_files)

        print("Created directory xyz_traj_split")
        print("Number of XYZ-files in directory:", len(full_list_of_xyz_files))
        print("Note: This directory can be used as a directory for the xyz_dir option to create_ML_training_data")
        print()
        print(f"Number of snapshots (num_snapshots keyword) set to {num_snapshots}")
        if random_snapshots is True:
            print(f"random_snapshots is True. Taking {num_snapshots} random XYZ snapshots.")
            list_of_xyz_files = random.sample(full_list_of_xyz_files, num_snapshots)
        else:
            print("random_snapshots is False. Taking the first", num_snapshots, "snapshots.")
            list_of_xyz_files=full_list_of_xyz_files[:num_snapshots]
        print(f"List of XYZ-files to use (num {len(list_of_xyz_files)}):", list_of_xyz_files)

        # Prepending path
        list_of_xyz_files=[f"./xyz_traj_split/{f}" for f in list_of_xyz_files]
        print(list_of_xyz_files)
        os.chdir('..')

    # Remove old files if present
    for f in ["train_data.xyz", "train_data.energies", "train_data.gradients", "train_data_mace.xyz"]:
        try:
            os.remove(f)
        except:
            pass

    # LOOP
    energies=[]
    gradients=[]
    fragments=[]
    labels=[]
    if runmode=="serial":
        print("Runmode is serial!")
        print("Will now loop over XYZ-files")
        print("For a large dataset consider using parallel runmode")
        for file in list_of_xyz_files:
            print("\nNow running file:", file)
            basefile=os.path.basename(file)
            label=basefile.split(".")[0]
            frag = Fragment(xyzfile=file, charge=charge, mult=mult,printlevel=printlevel)
            frag.label=label
            # 1: gas 2:solv  or 1: LL  or 2: HL
            print("Now running Theory 1")
            try:
                result_1 = Singlepoint(theory=theory_1, fragment=frag, Grad=Grad,
                                   result_write_to_disk=False,printlevel=printlevel)
            except:
                print("Problem with theory calculation")
                print(f"Will skip file {file} in training")
                continue
            if delta is True:
                # Running theory 2
                print("Now running Theory 2")
                try:
                    result_2 = Singlepoint(theory=theory_2, fragment=frag, Grad=Grad,
                                        result_write_to_disk=False,printlevel=printlevel)
                except:
                    print("Problem with theory calculation")
                    print(f"Will skip file {file} in training")
                    continue
                # Delta energy
                energy = result_2.energy - result_1.energy
                if Grad is True:
                    gradient = result_2.gradient - result_1.gradient
            else:
                energy = result_1.energy
                if Grad is True:
                    gradient = result_1.gradient
            # Add E and G to lists
            energies.append(energy)
            fragments.append(frag)
            labels.append(label)
            if Grad:
                gradients.append(gradient)

    elif runmode=="parallel":
        print("Runmode is parallel!")
        print("Will now run parallel calculations")
        # Fragments
        print("Looping over fragments first")
        for file in list_of_xyz_files:
            print("Now running file:", file)
            basefile=os.path.basename(file)
            label=basefile.split(".")[0]
            labels.append(label)
            # Creating fragment with label
            frag = Fragment(xyzfile=file, charge=charge, mult=mult, label=label)
            frag.label=label
            fragments.append(frag)

        # Parallel run
        print("Making sure numcores is set to 1 for both theories")
        theory_1.set_numcores(1)

        from ash.functions.functions_parallel import Job_parallel
        print("Now starting in parallel mode Theory1 calculations")
        results_theory1 = Job_parallel(fragments=fragments, theories=[theory_1], numcores=numcores, Grad=True)
        print("results_theory1.energies_dict:", results_theory1.energies_dict)
        if delta is True:
            theory_2.set_numcores(1)
            print("Now starting in parallel mode Theory2 calculations")
            results_theory2 = Job_parallel(fragments=fragments, theories=[theory_2], numcores=numcores, Grad=True)
            print("results_theory2.energies_dict:", results_theory2.energies_dict)

        # Loop over energy dict:
        for l in labels:
            print("Label l:", l)
            if delta is True:
                energy = results_theory2.energies_dict[l] - results_theory1.energies_dict[l]
                print("energy:", energy)

            else:
                energy = results_theory1.energies_dict[l]
                print("energy:", energy)

            energies.append(energy)

            # Gradient info
            if Grad:
                if delta is True:
                    gradient = results_theory2.gradients_dict[l] - results_theory1.gradients_dict[l]
                else:
                    gradient = results_theory1.gradients_dict[l]
                gradients.append(gradient)

    # Calculate energies for atoms
    energies_atoms_dict={}
    unique_elems_per_frag = [list(set(frag.elems)) for frag in fragments]
    unique_elems = list(set([j for i in unique_elems_per_frag for j in i]))

    from dictionaries_lists import atom_spinmults
    for uniq_el in unique_elems:
        mult = atom_spinmults[uniq_el]
        atomfrag = Fragment(atom=uniq_el, charge=0, mult=mult, printlevel=0)
        print("Now running Theory 1 for atom:", uniq_el)
        theory_1.printlevel=0
        theory_1.cleanup()
        result_1 = Singlepoint(theory=theory_1, fragment=atomfrag, printlevel=0,
                               result_write_to_disk=False)
        if delta is True:
            theory_2.printlevel=0
            # Running theory 2
            print("Now running Theory 2 for atom:", uniq_el)
            theory_2.cleanup()
            result_2 = Singlepoint(theory=theory_2, fragment=atomfrag, printlevel=0,
                                   result_write_to_disk=False)
            # Delta energy
            atomenergy = result_2.energy - result_1.energy
        else:
            atomenergy = result_1.energy
        energies_atoms_dict[uniq_el] = atomenergy
    print("\nAtomic energies:", energies_atoms_dict)

    ###########################################
    # Write final data
    ###########################################
    # Write XYZ-file
    for frag in fragments:
        # MultiXYZ-file
        write_xyzfile(frag.elems, frag.coords, "train_data", printlevel=1, writemode='a', title=f"coords {frag.label}")

    # Write energy file
    energies_file=open("train_data.energies", "w")
    for energy in energies:
        # Create file for ML
        energies_file.write(f"{energy}\n")
    energies_file.close()

    # Write gradient file
    if Grad:
        # Gradients-file
        gradients_file=open("train_data.gradients", "w")
        for grad in gradients:
            gradients_file.write(f"{frag.numatoms}\n")
            gradients_file.write(f"gradient {label} \n")
            for g in grad:
                gradients_file.write(f"{g[0]:10.7f} {g[1]:10.7f} {g[2]:10.7f}\n")
        gradients_file.close()

    print("\nNow writing data in MACE-format with energies in units of eV and forces in eV/Å")
    print("Fragments labels:",[frag.label for frag in fragments])
    print("energies:", energies)
    # Write data file that MACE uses

    with open("train_data_mace.xyz", "w") as mace_file:
        print("Writing isolated atom reference energies....")
        for el, an_at in energies_atoms_dict.items():
            en_ev = an_at * 27.211386245988
            mace_file.write("1\n")
            if Grad:
                mace_file.write(f"Properties=species:S:1:pos:R:3:forces_REF:R:3 config_type=IsolatedAtom energy_REF={en_ev} pbc='F F F'\n")
                mace_file.write(f"{el:2s}{0.0:17.8f}{0.0:17.8f}{0.0:17.8f}"
                                f"{-0.0:17.8f}{-0.0:17.8f}{-0.0:17.8f}\n")
            else:
                mace_file.write(f"Properties=species:S:1:pos:R:3 config_type=IsolatedAtom energy_REF={en_ev} pbc='F F F'\n")
                mace_file.write(f"{el:2s}{0.0:17.8f}{0.0:17.8f}{0.0:17.8f}\n")
        #TODO: Nmols, comp, molindex ?
        Nmols="1"
        comp="xxx"
        molindex=0

        for i in range(len(energies)):
            # Converting energy to eV
            frag = fragments[i]
            energy_ev = energies[i]*27.211386245988
            mace_file.write(f"{frag.numatoms}\n")
            if Grad:
                force = -1 * np.array(gradients[i]) * 51.42206747
                mace_file.write(f"Properties=species:S:1:pos:R:3:molID:I:1:forces_REF:R:3 Nmols={Nmols} Comp={comp} energy_REF={energy_ev} pbc='F F F'\n")
                for j in range(frag.numatoms): # Bug fix: inner loop variable was i, now j
                    mace_file.write(f"{frag.elems[j]:2s}{frag.coords[j][0]:17.8f}{frag.coords[j][1]:17.8f}{frag.coords[j][2]:17.8f}"
                                    f"{molindex:9d}{force[j][0]:17.8f}{force[j][1]:17.8f}{force[j][2]:17.8f}\n")
            else:
                # Write without forces
                mace_file.write(f"Properties=species:S:1:pos:R:3:molID:I:1 Nmols={Nmols} Comp={comp} energy_REF={energy_ev} pbc='F F F'\n")
                for j in range(frag.numatoms): # Bug fix: inner loop variable was i, now j
                    mace_file.write(f"{frag.elems[j]:2s}{frag.coords[j][0]:17.8f}{frag.coords[j][1]:17.8f}{frag.coords[j][2]:17.8f}"
                                    f"{molindex:9d}\n")

    print("All done! Files created:\ntrain_data.xyz\ntrain_data.energies\ntrain_data_mace.xyz")
    if Grad:
        print("train_data.gradients")
    print("Number of user-chosen snapshots:", num_snapshots)
    print("Number of successfully generated datapoints:", len(energies))


# Print statistics for dict with statistics for many models
# Assumes a dictionary with modelfilenames as keys and statistics_dict_forDB as values
def Ml_print_model_stats(dbdict=None, dbname="Sub-train", Grad=True):
    print("-"*30)
    print(f"       {dbname} database  ")
    print("-"*30)
    anykey = list(dbdict.keys())[0]
    print("Num-samples:", dbdict[anykey]["values"]["length"])
    print("Energies (kcal/mol)")
    print(f"{'':<20s} {'RMSE':>8s} {'MAE':>8s} {'Corr':>8s} {'R^2':>8s} {'pos_off':>8s} {'neg_off':>8s}")
    valscaling=627.5091 # Eh->kcal/mol
    for file, stats in dbdict.items():
        vals=stats["values"]
        print(f"{file:<20s} {vals['rmse']*valscaling:8.2f} {vals['mae']*valscaling:8.2f} {vals['corr_coef']:8.4f} \
{vals['r_squared']:8.4f} {vals['pos_off']*valscaling:8.2f}  {vals['neg_off']*valscaling:8.2f}")
    print()
    if Grad:
        print("Gradients (Eh/Bohr)")
        gradscaling=1.0
        for file, stats in dbdict.items():
            try:
                grads=stats["gradients"]
                print(f"{file:<20s} {grads['rmse']*gradscaling:8.6f} {grads['mae']*gradscaling:8.6f} {grads['corr_coef']:8.4f} \
{grads['r_squared']*gradscaling:8.4f} {grads['pos_off']*gradscaling:8.6f}  {grads['neg_off']*gradscaling:8.6f}")
            except KeyError:
                print("Found no gradient stats. skipping")
        print()


def query_by_committee(mltheories=None, configs=None, Grad=True, charge=0, mult=1, selection='energy-stdev', threshold=0.1, num_snaps=5, label=""):
    print("-"*50)
    print("query_by_committee function")
    print("-"*50)
    import pandas as pd
    configs_energies=[]
    configs_grads=[]
    # Loop over all other-configs with all mltheories
    for config in configs:
        energies=[]
        gradients=[]

        # 
        if isinstance(config,str):
            frag = Fragment(xyzfile=config, charge=charge, mult=mult, printlevel=0)
        if isinstance(config,Fragment):
            frag=config

        for mltheory in mltheories:
            # Running ML (or deltaML) energy
            res1 = Singlepoint(theory=mltheory, fragment=frag,Grad=Grad, printlevel=0,result_write_to_disk=False)
            energies.append(res1.energy)
            if Grad is True:
                gradients.append(res1.gradient)
        configs_energies.append(energies)
        configs_grads.append(gradients)
    # Get stdevs
    print("configs_energies:", configs_energies)
    stdevs_e = np.array([statistics.stdev(i) for i in configs_energies])
    ranges_e = np.array([abs(abs(max(i)) - abs(min(i)))for i in configs_energies])
    # Grad
    stdevs_g=[]
    stdevs_g_p = []
    for config_g in configs_grads:
        # Global stdev
        combined = np.concatenate([a.ravel() for a in config_g])
        # Pooled
        magnitudes = [np.linalg.norm(F, axis=1) for F in config_g]
        std = np.std(combined, axis=0)
        std_p = np.std(np.concatenate(magnitudes),axis=0)
        stdevs_g.append(std)
        stdevs_g_p.append(std_p)
    stdevs_g=np.array(stdevs_g)
    stdevs_g_p=np.array(stdevs_g_p)

    # Build a dictionary of top-5 snapshots per metric
    top_snapshots = {}
    for name, m in zip(["Std_E","Range_E","Std_G","STtd_G_pooled"], [stdevs_e,ranges_e,stdevs_g,stdevs_g_p]):
        top5_indices = np.argsort(m)[-num_snaps:][::-1]
        top5_values = m[top5_indices]
        top_snapshots[name] = [
            (int(idx), float(val)) for idx, val in zip(top5_indices, top5_values)
        ]
    # Convert to a DataFrame
    df = pd.DataFrame(top_snapshots)
    # extract just the indices to check duplicates
    all_indices = [x[0] for col in df.columns for x in df[col]]
    duplicate_indices = {idx for idx in all_indices if all_indices.count(idx) > 1}

    # Step 4: format cells, highlighting duplicates with a marker (★)
    def format_cell(cell):
        idx, val = cell
        marker = " ★" if idx in duplicate_indices else ""
        return f"{idx} ({val:.4f}){marker}"

    df = df.applymap(format_cell)
    print(df)

    # PLOT data
    try:
        print("Attempting to plot")
        # Create ASH_plot object named edplot
        eplot = ASH_plot("Plotname", num_subplots=1, x_axislabel="x-axis", y_axislabel='y-axis')
        eplot.addseries(0, x_list=list(range(0,len(stdevs_e))), y_list=stdevs_e, label='Stdev-E', color='blue', line=True, scatter=True)
        eplot.savefig(f"stdevE_per_snap_{label}")

        # Create ASH_plot object named edplot
        eplot = ASH_plot("Plotname", num_subplots=1, x_axislabel="x-axis", y_axislabel='y-axis')
        eplot.addseries(0, x_list=list(range(0,len(stdevs_g))), y_list=stdevs_g, label='Stdev-G', color='blue', line=True, scatter=True)
        eplot.savefig(f"stdevG_per_snap_{label}")

        # Create ASH_plot object named edplot
        eplot = ASH_plot("Plotname", num_subplots=1, x_axislabel="x-axis", y_axislabel='y-axis')
        eplot.addseries(0, x_list=list(range(0,len(stdevs_g_p))), y_list=stdevs_g_p, label='Stdev-Gp', color='blue', line=True, scatter=True)
        eplot.savefig(f"stdevGp_per_snap_{label}")
    except:
        print("Failed to plot")
        pass


    if selection == "energy-stdev":
        print("Selection option is energy-stdev")
        print("Threshold:", threshold)
        print("Number of snapshots to grab:", num_snaps)
        used_metric=stdevs_e
    elif selection == "energy-range":
        print("Selection option is energy-range")
        print("Threshold:", threshold)
        print("Number of snapshots to grab:", num_snaps)
        used_metric=ranges_e
    elif selection == "gradient":
        print("Selection option is gradient")
        print("Threshold:", threshold)
        print("Number of snapshots to grab:", num_snaps)
        used_metric=stdevs_g
    elif selection == "gradient2":
        print("Selection option is stdevs_g_p")
        print("Threshold:", threshold)
        print("Number of snapshots to grab:", num_snaps)
        used_metric=stdevs_g_p
    else:
        print("Error")
        exit()

    # Get snapshots above threshold and up to num_snaps
    above_thresh_indices = np.where(used_metric > threshold)[0]
    print(f"Found {len(above_thresh_indices)} snapshots above threshold")
    filtered_arr = used_metric[above_thresh_indices]
    top_indices_in_filtered = np.argsort(filtered_arr)[-num_snaps:][::-1]
    top_indices = above_thresh_indices[top_indices_in_filtered]
    chosen_configs = [configs[i] for i in top_indices]
    print("chosen_configs:", chosen_configs)

    print(f"Selected {len(chosen_configs)} configs with high stdevs")
    return chosen_configs

def active_learning(ml_theories=None, e_f_weights=None, training_dir=None, maxiter=10, theory_1=None, theory_2=None, Grad=True,
                        init_base_cfgs=15, threshold=0.0001, max_add_snaps=5, maxepochs=100, selection="energy-range",
                        noupdate=False, random_selection=False, random_seed_set=False, seed=42,
                        charge=None, mult=None, runmode="serial", numcores=1):

    if ml_theories is None:
        print("Error: ml_theories needs to be set (list of ASH MLTheories)")
        ashexit()
    if e_f_weights is None:
        print("Error: e_f_weights needs to be set (list of ASH MLTheories)")
        ashexit()
    if len(ml_theories) != len(e_f_weights):
        print("Error: Length of ml_theories list and length of e_f_weights must be the same")
        ashexit()

    if training_dir is None:
        print("Error: training_dir needs to be set (path to XYZ-directory)")
        ashexit()

    if theory_1 is None:
        print("Error: theory_1 needs to be set")
        ashexit()
    if charge is None or mult is None:
        print("Error: Charge/mult need to be set")
        ashexit()

    print("training_dir:", training_dir)
    print("Max Loop Iterations:", maxiter)
    print("Initial base configurations:", init_base_cfgs)
    print("Selector:", selection)
    print(f"Energy threshold: {threshold} Eh")
    print("Number of snapshots added per loop iteration:", max_add_snaps)
    print("Number of po")
    print("random_seed_set:", random_seed_set)

    print("theory_1:", theory_1)
    print("theory_2:", theory_2)
    print("Grad:", Grad)

    print("noupdate:", noupdate)
    print("random_selection:", random_selection)

    # Move
    def move_chosen_files(chosen,dirname):
        for f in chosen:
            shutil.move(f, f"{dirname}/{os.path.basename(f)}")

    #Delete old dirs
    print("Deleting possible old dirs: current_set base")
    for d in ["current_set", "base"]:
        try:
            shutil.rmtree(d)
        except:
            pass
    # Copy full dir over as current-set
    print("Copying whole training dir to current_set")
    shutil.copytree(training_dir, "./current_set")
    xyzdir="current_set"
    # Read all XYZ-files as list
    xyzfiles = glob.glob(f"{xyzdir}/*xyz")
    print("Number of xyzfiles in Original DIR:", len(xyzfiles))
    # Create base dir
    os.mkdir("base")

    # Choose base set:
    # This can be replaced by a list of chosen XYZ-files instead
    if random_seed_set:
        random.seed(42)
    base_cfgs = random.sample(xyzfiles, init_base_cfgs)

    # Move chosen base configs to base
    #move_chosen_files(base_cfgs,"base")
    move_chosen_files(base_cfgs,"current_set")

    # Determine number of elements
    num_elems = len(list(set(Fragment(xyzfile=base_cfgs[0]).elems)))

    # ACTIVE LEARNING LOOP
    chosen_cfgs=[]
    current_xyzfiles=[]
    for iter in range(maxiter):
        print("="*50)
        print(f"ACTIVE LEARNING ITERATION {iter}")
        print("="*50)
        # Base CFGS and rest configs
        other_cfgs = listdiff(xyzfiles,base_cfgs)
        print(f"NUM CURRENT BASE CONFIGS : {len(base_cfgs)}")
        print(f"NUM NEW BASE CONFIGS : {len(chosen_cfgs)}")
        print([os.path.basename(i) for i in base_cfgs])
        print(f"NUM CURRENT OTHER CONFIGS : {len(other_cfgs)}")
        print("other_cfgs:", other_cfgs)
        if len(other_cfgs) == 0:
            print("Warning: No remaining CONFIGS left. Exiting loop")
            print("Final number of cfgs in base:", len(base_cfgs))
            break
        print()
        # Create training data for new cfgs
        if iter == 0:
            current_xyzfiles = base_cfgs
        else:
            current_xyzfiles = chosen_cfgs
        print("current_xyzfiles:", current_xyzfiles)
        create_ML_training_data(xyz_files=current_xyzfiles, random_snapshots=True, printlevel=0,
                                   theory_1=theory_1, theory_2=theory_2, charge=charge, mult=mult, Grad=Grad, 
                                   runmode=runmode, numcores=numcores)
        # Keep track of each iteration's training data
        os.rename("train_data_mace.xyz", f"train_data_mace{iter}.xyz")
        # First iter, we only have train_data_mace0.xyz
        if iter == 0:
            shutil.copyfile(f"train_data_mace{iter}.xyz", "train_data_mace.xyz")
        else:
            # Append new data to train_data_mace.xyz
            with open("train_data_mace.xyz", "w") as outfile:
                for i in range(iter+1):
                    # write atomic references only once
                    if i == 0:
                        with open(f"train_data_mace{i}.xyz", "r") as infile:
                            for line in infile:
                                outfile.write(line)
                    else:
                        with open(f"train_data_mace{i}.xyz", "r") as infile:
                            lines = infile.readlines()
                            # Skip atomic references
                            data_lines = lines[3*num_elems:]
                            for line in data_lines:
                                outfile.write(line)
        # ML Theories
        for i,(ml,efw) in enumerate(zip(ml_theories, e_f_weights)):
            # Unique model filename
            ml.model_file=f"ML_ep{maxepochs}_ew_{e_f_weights[i][0]}_fw_{e_f_weights[i][1]}_iter{iter}.model"
            # Train with selected epochs and weights
            ml.train(max_num_epochs=maxepochs, energy_weight=e_f_weights[i][0], forces_weight=e_f_weights[i][1])

        # Check consistency of models and choose outliers
        chosen_cfgs = query_by_committee(mltheories=ml_theories, configs=other_cfgs, Grad=Grad, threshold=threshold,
                        num_snaps=max_add_snaps, label=str(iter), selection=selection)
        #
        if random_selection is True:
            if random_seed_set:
                random.seed(seed)
            chosen_cfgs = random.sample(other_cfgs, max_add_snaps)

        if len(chosen_cfgs) == 0:
            print("No new cfgs found. Exiting loop")
            print("Final number of cfgs in base:", len(base_cfgs))
            print("ACTIVE LEARNING COMPLETE!")
            break
        # What to do with chosen configs
        if noupdate is True:
            chosen_cfgs=[]
        #else:
            #Move chosen configs to base
            #print("RB")
            #print("Now moving chosen configs to base dir:", chosen_cfgs)
            #move_chosen_files(chosen_cfgs,"base")
        #Add to base
        base_cfgs += chosen_cfgs

    print("Active learning is complete.")
    if iter == maxiter:
        print("Warning: Active learning loop did not converge. Check the results carefully")
    else:
        print("Active learning loop converged")
        print("Final set of configurations are found in directory: base")
        move_chosen_files(base_cfgs,"base")
