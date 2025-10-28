"""
This function will assign SOAP vectors (mathematical fingerprint) to different conformations of the system in the traj file.
The SOAP vectory is calculated using the DScribe package. 

Install : pip install dscribe

SOAP vectors are 1D arrays that describe the local atomic environment around each atom in a structure.
Cosine_similarity between SOAP vectors can be used as a measure of similarity between different atomic environments.
Here this cosine similarity value is normalized to be between 0 and 1.
Sim_threshold can be used to extract only those conformations that have a similarity value lower than the given threshold.

For my purpose, I need to select a diverse set of geometrical conformations. 
To achieve this in a reliable way, choosing an appropriate reference for calculating similarity is crucial. To do this I can use 2 ways:
1. Select the lowest energy (minima) as reference: This approach makes sense because this will allow me to sample conformations 
    which are far away from the minima and thus should be more diverse
    This approach should have a simple function with reference and traj file as input.


2. Clustering method: This is a more complicated and suggested to be a more reliable technique, will implement later.

"""

import os
import numpy as np
import shutil
import glob
import ash
from ase import Atoms
from dscribe.descriptors import SOAP
from ash.functions.functions_general import natural_sort, ashexit
from ash.modules.module_coords import write_xyzfile, split_multimolxyzfile
from ash.functions.functions_general import ashexit 


def frame_descriptor(soap, frame, average_mode):
    """
    Computes a single, normalized descriptor vector for an entire frame.
    
    Parameters:
    soap (dscribe.descriptors.SOAP): The initialized SOAP descriptor.
    frame (ase.Atoms or str): The atomic structure. Should be an ASE object 
    average_mode (str): The averaging mode ('inner', 'outer', 'off').
    """
    
    if isinstance(frame, str):
        if not os.path.isfile(frame):
             ashexit(f"File not found in frame_descriptor: {frame}")
             
        if frame.endswith('.pdb'):
            structure_obj = ash.Fragment(pdbfile=frame, printlevel=0)
        elif frame.endswith('.xyz'):
            structure_obj = ash.Fragment(xyzfile=frame, printlevel=0)
        else:
            ashexit(f"Unsupported file format in frame_descriptor: {frame}")
    else:
        # If it's not a string, assume it's already a loaded object (like ash.Fragment)
        structure_obj = frame


    ## Convert ASH fragment to ASE Atoms object
    try:   
        ase_structure = Atoms(symbols=structure_obj.elems, positions=structure_obj.coords)

    except Exception as e:
        ashexit(f"Error creating ASE Atoms object in frame_descriptor: {e}")

    ## CREATE descriptor vector using ASE object
    v = soap.create(ase_structure, n_jobs=1) 
    
    if average_mode == "off":
        # Manually average over atoms to get a single vector for the frame
        v = v.mean(axis=0)
    
    # Normalize the vector (L2 norm)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    
    # Ensure a consistent, memory-efficient type
    return v.astype(np.float32)



def SOAPSimilarity(reference, xyztraj, sim_threshold=0.5, r_cut=5.0, n_max=8, l_max=6, sigma=0.5, average="inner", ):
    """
    Assign SOAP vectors to different conformations in the traj file and filter based on similarity threshold.

    Parameters:
    reference (str): Path to the reference structure file. Can be PDB or xyz format.
    traj (str): Path to the trajectory file containing multiple conformations. # for now only xyz trajectory is supported.
    sim_threshold (float): Similarity threshold for filtering conformations.
    r_cut (float): Cutoff radius for SOAP descriptor.
    n_max (int): Maximum number of radial basis functions.
    l_max (int): Maximum degree of spherical harmonics.
    sigma (float): Gaussian width.
    average (Optional[str]): Averaging mode for the SOAP descriptor:
            - `"inner"` (default): Averages SOAP vectors before computing the power spectrum.
            - `"outer"`: Computes the power spectrum for each atom, then averages.
            - `None`: No averaging, returns per-atom descriptors. (Will be translated to 'off')


    Returns:
    list: List of paths to the newly saved conformations in 'selected_diverse_frames'.
    """
    

    # Sanity Checks
    if not os.path.isfile(reference):
        ashexit(f"Reference file not found: {reference}")
    if not os.path.isfile(xyztraj):
        ashexit(f"Trajectory file not found: {xyztraj}")
    if sim_threshold < 0.0 or sim_threshold > 1.0:
        ashexit("Similarity threshold must be between 0 and 1.")
    if average not in ["inner", "outer", "off"]:
         print(f"Error: Invalid average mode '{average}'. Must be 'inner', 'outer', or 'off'.")
         ashexit("Invalid average mode.")
    
    
    #Read reference structure
    if reference.endswith('.pdb'):
        frag_ref = ash.Fragment(pdbfile=reference, printlevel=0)
    elif reference.endswith('.xyz'):
        frag_ref = ash.Fragment(xyzfile=reference, printlevel=0)
    else:
        ashexit(f"Unsupported reference file format: {reference}")

    species = list(set(frag_ref.elems))



    if xyztraj.endswith('.xyz'):
        print("Reading trajectory from xyz file.")
        xyz_traj = os.path.abspath(xyztraj)
        print(f"Absolute path of trajectory file: {xyz_traj}")

        try:
            shutil.rmtree("xyz_traj_split")
            print("Removed existing directory: xyz_traj_split")
        except:
            pass
        os.makedirs("xyz_traj_split", exist_ok=True)
        os.chdir("xyz_traj_split")

        print("Splitting xyz trajectory into individual frames...")
        split_multimolxyzfile(xyz_traj, writexyz=True, skipindex=1, return_fragments=False)
        list_xyz_files = natural_sort(glob.glob("*.xyz"))
        print(f"Total frames found: {len(list_xyz_files)}")
        os.chdir('..')

    
    output_dir = "selected_diverse_frames"
    shutil.rmtree(output_dir, ignore_errors=True) 
    os.makedirs(output_dir, exist_ok=True)
    print(f"Selected frames will be saved to: {output_dir}")


    # Creating SOAP vectors for reference structure
    soap = SOAP(species=species,r_cut=r_cut,n_max=n_max,l_max=l_max,sigma=sigma,average=average)
    print("Computing descriptor for reference frame...")
    ref_vector = frame_descriptor(soap, reference, average)
    print(f"Reference vector shape: {ref_vector.shape}")

    selected_conformations = []

    # Process each frame in the trajectory
    for i, frame_file in enumerate(list_xyz_files):
        frame_path = os.path.join("xyz_traj_split", frame_file)
        frame_vector = frame_descriptor(soap, frame_path, average)
        # Compute cosine similarity
        cos_sim = np.dot(ref_vector, frame_vector)
        # Normalize to [0, 1]
        norm_sim = 0.5 * (cos_sim + 1.0)

        if norm_sim <= sim_threshold:
            print(f"Frame {i} selected (diverse). Similarity: {norm_sim:.4f} (Threshold: {sim_threshold})")
            
            # Copy selected frame to output directory
            dest_file = os.path.join(output_dir, f"frame_{i:06d}.xyz") # Use 6 padding for large trajs
            shutil.copy(frame_path, dest_file)
            
           
            selected_conformations.append(dest_file)
        
        
    print(f"\nProcessing complete. {len(selected_conformations)} diverse frames selected.")
        
    return len(selected_conformations)