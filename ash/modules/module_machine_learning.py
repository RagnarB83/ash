import os
import glob
import shutil
import random

from ash.modules.module_coords import write_xyzfile, split_multimolxyzfile
from ash.functions.functions_general import natural_sort
from ash import Singlepoint, Fragment

# Collection of functions related to machine learning and data analysis
# Also helper tools for Torch and MLatom interfaces

# Function to create ML training data given XYZ-files and 2 ASH theories
def create_ML_training_data(xyzdir=None, xyz_trajectory=None, num_snapshots=100, random_snapshots=True,
                               theory_1=None, theory_2=None, charge=0, mult=1, Grad=True):
    if xyzdir is None and xyz_trajectory is None:
        print("Error: create_ML_training_data requires xyzdir or xyz_trajectory option to be set!")
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
    for f in ["train_data.xyz", "train_data.energies", "train_data.gradients"]:
        try:
            os.remove(f)
        except:
            pass
    energies_file=open("train_data.energies", "w")
    gradients_file=open("train_data.gradients", "w")

    # LOOP
    for file in list_of_xyz_files:
        basefile=os.path.basename(file)
        label=basefile.split(".")[0]


        frag = Fragment(xyzfile=file, charge=charge, mult=mult)

        # 1: gas 2:solv  or 1: LL  or 2: HL
        print("Now running Theory 1")
        result_1 = Singlepoint(theory=theory_1, fragment=frag, Grad=Grad)

        if delta is True:
            # Running theory 2
            print("Now running Theory 2")
            result_2 = Singlepoint(theory=theory_2, fragment=frag, Grad=Grad)
            # Delta energy
            energy = result_2.energy - result_1.energy
            if Grad is True:
                gradient = result_2.gradient - result_1.gradient
        else:
            energy = result_1.energy
            if Grad is True:
                gradient = result_1.gradient

        # Create files for ML

        energies_file.write(f"{energy}\n")
        # Gradients-file
        gradients_file.write(f"{frag.numatoms}\n")
        gradients_file.write(f"gradient {label} \n")
        for g in gradient:
            gradients_file.write(f"{g[0]:10.7f} {g[1]:10.7f} {g[2]:10.7f}\n")

        # MultiXYZ-file
        write_xyzfile(frag.elems, frag.coords, "train_data", printlevel=2, writemode='a', title=f"coords {label}")

    energies_file.close()
    gradients_file.close()

    print("All done! Files created:\ntrain_data.xyz\ntrain_data.energies\ntrain_data.gradients")
