import os
import glob
import shutil
import random
import numpy as np

from ash.modules.module_coords import write_xyzfile, split_multimolxyzfile
from ash.functions.functions_general import natural_sort,ashexit
from ash import Singlepoint, Fragment
from ash.interfaces.interface_mdtraj import MDtraj_slice

# Collection of functions related to machine learning and data analysis
# Also helper tools for Torch and MLatom interfaces

# Function to create ML training data given XYZ-files and 2 ASH theories
def create_ML_training_data(xyzdir=None, dcd_trajectory=None, xyz_trajectory=None, num_snapshots=None, random_snapshots=True,
                                dcd_pdb_topology=None, nth_frame_in_traj=1,
                               theory_1=None, theory_2=None, charge=0, mult=1, Grad=True, runmode="serial", numcores=1):
    if xyzdir is None and xyz_trajectory is None and dcd_trajectory is None:
        print("Error: create_ML_training_data requires xyzdir, xyz_trajectory or dcd_trajectory option to be set!")
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

    print("xyzdir:", xyzdir)
    print("xyz_trajectory:", xyz_trajectory)
    print("dcd_trajectory:", dcd_trajectory)
    print("Charge:", charge)
    print("Mult:", mult)
    print("Grad:", Grad)

    print(f"num_snapshots: {num_snapshots} # xyz_trajectory option: number of snapshots to use")
    print(f"random_snapshots: {random_snapshots} # xyz_trajectory option: randomize snapshots or not")


    print("Theory 1:", theory_1)
    print("Theory 2:", theory_2)

    # XYZ DIRECTORY
    if xyzdir is not None:
        print("XYZ-dir specified.")
        full_list_of_xyz_files=glob.glob(f"{xyzdir}/*.xyz")
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
        
        print("Note: This directory can be used as a directory for the xyzdir option to create_ML_training_data")
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
        print("Note: This directory can be used as a directory for the xyzdir option to create_ML_training_data")
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
            frag = Fragment(xyzfile=file, charge=charge, mult=mult)
            frag.label=label
            # 1: gas 2:solv  or 1: LL  or 2: HL
            print("Now running Theory 1")
            try:
                result_1 = Singlepoint(theory=theory_1, fragment=frag, Grad=Grad,
                                   result_write_to_disk=False)
            except:
                print("Problem with theory calculation")
                print(f"Will skip file {file} in training")
                continue
            if delta is True:
                # Running theory 2
                print("Now running Theory 2")
                try:
                    result_2 = Singlepoint(theory=theory_2, fragment=frag, Grad=Grad,
                                        result_write_to_disk=False)
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
        all_fragments=[]
        for file in list_of_xyz_files:
            print("Now running file:", file)
            basefile=os.path.basename(file)
            label=basefile.split(".")[0]
            labels.append(label)
            # Creating fragment with label
            frag = Fragment(xyzfile=file, charge=charge, mult=mult, label=label)
            frag.label=label

        # Parallel run
        print("Making sure numcores is set to 1 for both theories")
        theory_1.set_numcores(1)

        from ash.functions.functions_parallel import Job_parallel
        print("Now starting in parallel mode Theory1 calculations")
        results_theory1 = Job_parallel(fragments=all_fragments, theories=[theory_1], numcores=numcores, Grad=True)
        print("results_theory1.energies_dict:", results_theory1.energies_dict)
        if delta is True:
            theory_2.set_numcores(1)
            print("Now starting in parallel mode Theory2 calculations")
            results_theory2 = Job_parallel(fragments=all_fragments, theories=[theory_2], numcores=numcores, Grad=True)
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

    #Calculate energies for atoms
    energies_atoms_dict={}
    unique_elems_per_frag = [list(set(frag.elems)) for frag in fragments]
    unique_elems = list(set([j for i in unique_elems_per_frag for j in i]))

    from dictionaries_lists import atom_spinmults
    for uniq_el in unique_elems:
        mult = atom_spinmults[uniq_el]
        print("mult:", mult)
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


        #TODO: Nmols
        Nmols="1"

        #TODO comp
        comp="xxx"
        #molindex
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