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

Performs diverse conformation selection from an XYZ trajectory using SOAP vectors.

Supports two main selection strategies:
1.  Reference-based: Selects frames based on their cosine similarity to a
    provided reference structure (using a threshold or a fixed total number).
2.  Clustering-based: Uses K-means clustering on all SOAP vectors to find
    a specified number of representative frames (exemplars) from the
    entire conformational space.

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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min


def frame_SOAPdescriptor(soap, frame, average_mode):

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
        pass

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



def SOAPSimilarity(reference, xyztraj, sim_threshold=0.5, total_frames=None, n_clusters=None, 
                   r_cut=5.0, n_max=8, l_max=6, sigma=0.5, average="inner"):
    """
    Assign SOAP vectors and filter based on similarity OR K-means clustering.

    Modes:
    1. Reference-based: Filters by similarity threshold (sim_threshold) or
       selects a fixed number of most diverse frames (total_frames).
    2. Clustering-based: If n_clusters is set, performs K-means clustering
       and selects one representative frame from each cluster.

    Parameters:
    reference (str): Path to the reference structure file (PDB or xyz).
                     *Always* used to determine atomic species.
                     *Also* used as the comparison point for threshold/total_frames modes.
    xyztraj (str): Path to the xyz trajectory file.
    sim_threshold (float): Similarity threshold (used if total_frames and n_clusters are None).
    total_frames (int, optional): Select this many most diverse frames (overrides sim_threshold).
    n_clusters (int, optional): Number of clusters for K-means (overrides all other modes).
    r_cut (float): Cutoff radius for SOAP descriptor.
    n_max (int): Maximum number of radial basis functions.
    l_max (int): Maximum degree of spherical harmonics.
    sigma (float): Gaussian width.
    average (str): Averaging mode for SOAP descriptor ("inner", "outer", "off").

    Returns:
    int: The number of selected conformation paths.
    """
    
    if not os.path.isfile(reference):
        ashexit(f"Reference file not found: {reference}")
    if not os.path.isfile(xyztraj):
        ashexit(f"Trajectory file not found: {xyztraj}")
    if average not in ["inner", "outer", "off"]:
         ashexit(f"Error: Invalid average mode '{average}'. Must be 'inner', 'outer', or 'off'.")

    selection_mode = "threshold" 
    
    if n_clusters is not None:
        try:
            n_clusters = int(n_clusters)
            if n_clusters > 0:
                selection_mode = "kmeans"
                print(f"Selection mode: 'K-Means'. Will select {n_clusters} representative frames.")
                print("  > 'sim_threshold' and 'total_frames' will be ignored.")
            else:
                ashexit(f"Error: 'n_clusters' must be > 0. Got {n_clusters}.")
        except ValueError:
             ashexit(f"Error: 'n_clusters' is not a valid integer. Got {n_clusters}.")
    
    elif total_frames is not None:
        try:
            total_frames = int(total_frames)
            if total_frames > 0:
                selection_mode = "total_frames"
                print(f"Selection mode: 'total_frames'. Will select the {total_frames} most diverse frames.")
                print("  > 'sim_threshold' will be ignored.")
            else:
                print(f"Warning: 'total_frames' is {total_frames}. Must be > 0. Falling back to 'sim_threshold'.")
                selection_mode = "threshold"
        except ValueError:
             print(f"Warning: 'total_frames' is not a valid integer. Falling back to 'sim_threshold'.")
             selection_mode = "threshold"
    
    if selection_mode == "threshold":
        if sim_threshold < 0.0 or sim_threshold > 1.0:
            ashexit("Similarity threshold must be between 0 and 1.")
        print(f"Selection mode: 'sim_threshold'. Will select frames with similarity <= {sim_threshold}.")


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
        
        split_dir = "xyz_traj_split"
        shutil.rmtree(split_dir, ignore_errors=True)
        os.makedirs(split_dir, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(split_dir)

        print("Splitting xyz trajectory into individual frames...")
        try:
            split_multimolxyzfile(xyz_traj, writexyz=True, skipindex=1, return_fragments=False)
        except Exception as e:
            ashexit(f"Failed to split xyz file. Error: {e}")
            
        list_xyz_files = [os.path.abspath(f) for f in natural_sort(glob.glob("*.xyz"))]
        print(f"Total frames found: {len(list_xyz_files)}")
        os.chdir(cwd) 
    else:
        ashexit(f"Unsupported trajectory file format: {xyztraj}")

    if not list_xyz_files:
        ashexit("No frames found in trajectory.")

    if selection_mode == "kmeans" and n_clusters > len(list_xyz_files):
        print(f"Warning: n_clusters ({n_clusters}) is greater than total frames ({len(list_xyz_files)}).")
        print(f"Setting n_clusters = {len(list_xyz_files)}.")
        n_clusters = len(list_xyz_files)
        
    output_dir = "selected_diverse_frames"
    shutil.rmtree(output_dir, ignore_errors=True) 
    os.makedirs(output_dir, exist_ok=True)
    print(f"Selected frames will be saved to: {output_dir}")

    soap = SOAP(species=species,r_cut=r_cut,n_max=n_max,l_max=l_max,sigma=sigma,average=average)

    all_vectors = {} 
    
    print(f"\nProcessing {len(list_xyz_files)} trajectory frames to compute SOAP vectors...")
    
    for i, frame_path in enumerate(list_xyz_files):
        try:
            frame_vector = frame_SOAPdescriptor(soap, frame_path, average)
            all_vectors[frame_path] = frame_vector
        except Exception as e:
            print(f"Warning: Failed to process frame {frame_path}. Skipping. Error: {e}")
            continue 
        
        if (i + 1) % 200 == 0 or (i + 1) == len(list_xyz_files):
             print(f"  ... processed {i + 1}/{len(list_xyz_files)} frames.")
    
    if not all_vectors:
        ashexit("Failed to compute any SOAP vectors.")

    print("\nAll frame vectors computed.")

    selected_frames_info = [] 

    if selection_mode == "kmeans":
        print(f"Running K-Means clustering to find {n_clusters} clusters...")
        
        frame_paths_list = list(all_vectors.keys())
        all_vectors_list = list(all_vectors.values())
        
        X = np.array(all_vectors_list, dtype=np.float32)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        print("Clustering complete.")

        try:
            print("Generating cluster visualization plot...")
            plot_soap_clusters(X, kmeans)
        except Exception as e:
            print(f"Warning: Could not generate cluster plot. Error: {e}")


        print("Finding the most representative frame (exemplar) for each cluster...")
        closest_frame_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, 
            X
        )
        
        unique_indices = sorted(list(set(closest_frame_indices)))
        
        selected_frames_info = []
        cluster_labels_for_exemplars = kmeans.labels_[unique_indices]
        
        for i, frame_idx in enumerate(unique_indices):
            frame_path = frame_paths_list[frame_idx]
            cluster_id = cluster_labels_for_exemplars[i]
            selected_frames_info.append((frame_path, cluster_id))
            
        print(f"Found {len(selected_frames_info)} unique exemplars.")

    else:
        print("Computing descriptor for reference frame...")
        try:
            ref_vector = frame_SOAPdescriptor(soap, reference, average)
            print(f"Reference vector shape: {ref_vector.shape}")
        except Exception as e:
            ashexit(f"Failed to compute SOAP vector for reference frame {reference}. Error: {e}")

        print("Calculating similarities relative to reference...")
        all_similarities = [] 
        for frame_path, frame_vector in all_vectors.items():
            cos_sim = np.dot(ref_vector, frame_vector)
            norm_sim = 0.5 * (cos_sim + 1.0) 
            all_similarities.append((frame_path, norm_sim))
        
        if selection_mode == "total_frames":
            if total_frames > len(all_similarities):
                print(f"Warning: Requested {total_frames} frames, but only {len(all_similarities)} available. Selecting all.")
                total_frames = len(all_similarities)
            
            sorted_similarities = sorted(all_similarities, key=lambda x: x[1])
            selected_frames_info = sorted_similarities[:total_frames]
            
        else: 
            selected_frames_info = [item for item in all_similarities if item[1] <= sim_threshold]

    print(f"Copying {len(selected_frames_info)} selected frames to {output_dir}...")
    selected_conformations_paths = []

    for item in selected_frames_info:
        frame_path = item[0]
        value = item[1] 
        
        frame_basename = os.path.basename(frame_path)
        dest_file = os.path.join(output_dir, frame_basename)
        
        try:
            shutil.copy(frame_path, dest_file)
            selected_conformations_paths.append(dest_file)
            if selection_mode == "kmeans":
                print(f"  - Copied {frame_basename} (Exemplar for cluster {value})")
            else:
                print(f"  - Copied {frame_basename} (Similarity: {value:.4f})")
        except Exception as e:
            print(f"  - Error copying {frame_basename}: {e}")
            
    print(f"\nProcessing complete. {len(selected_conformations_paths)} diverse frames saved.")
        
    return len(selected_conformations_paths)


def plot_soap_clusters(X, kmeans, title="SOAP Vector Clusters (t-SNE)"):
    """
    Visualizes high-dimensional SOAP vectors in 2D using t-SNE.

    Parameters:
    X (np.ndarray): The full array of SOAP vectors (n_frames, n_features).
    kmeans (sklearn.cluster.KMeans): The *fitted* K-means model object.
    title (str): The title for the plot.
    """
    
    # 1. Get cluster labels
    labels = kmeans.labels_
    n_clusters = kmeans.n_clusters
    
    # 2. Find the index of the exemplar (closest point) for each cluster
    # This is the same logic from your main function
    exemplar_indices, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, 
        X
    )
    # Ensure unique indices, as in your function
    exemplar_indices = sorted(list(set(exemplar_indices)))

    # 3. Reduce dimensionality with t-SNE
    print("Running t-SNE for visualization... (this may take a moment)")
    # Standardizing data first often helps t-SNE
    X_std = StandardScaler().fit_transform(X)
    
    # Use perplexity around 30-50, but it must be less than n_samples
    n_samples = X.shape[0]
    perp = min(30, n_samples - 1)
    if perp <= 0:
        print("Warning: Not enough samples for t-SNE. Skipping plot.")
        return

    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=300)
    X_2d = tsne.fit_transform(X_std)
    print("t-SNE complete.")

    # 4. Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get a color map
    colors = plt.cm.get_cmap('jet', n_clusters)
    
    # Plot all the non-exemplar points
    for i in range(n_clusters):
        # Find points belonging to this cluster
        cluster_points = (labels == i)
        
        # Remove the exemplar from this list so we can plot it differently
        is_exemplar = np.zeros_like(cluster_points, dtype=bool)
        if i in labels[exemplar_indices]:
             # find which exemplar index corresponds to this cluster label
             exemplar_idx_for_this_cluster = exemplar_indices[np.where(labels[exemplar_indices] == i)[0][0]]
             is_exemplar[exemplar_idx_for_this_cluster] = True
             
        cluster_non_exemplars = cluster_points & ~is_exemplar

        plt.scatter(
            X_2d[cluster_non_exemplars, 0], 
            X_2d[cluster_non_exemplars, 1], 
            color=colors(i), 
            label=f'Cluster {i}' if i not in labels[exemplar_indices] else None, # Avoid duplicate labels
            alpha=0.4,
            s=20 # smaller size for background points
        )

    # Plot the exemplars (selected frames)
    # This plots them on top, larger, and with a black edge
    plt.scatter(
        X_2d[exemplar_indices, 0], 
        X_2d[exemplar_indices, 1], 
        marker='*', # Star marker
        c=labels[exemplar_indices], # Color by their own cluster
        cmap=plt.cm.get_cmap('jet', n_clusters),
        edgecolor='k', # Black edge
        s=400, # Much larger size
        label='Exemplars (Selected)'
    )

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(loc='best', markerscale=1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_filename = "soap_cluster_visualization.png"
    plt.savefig(plot_filename)
    print(f"Cluster visualization saved to: {plot_filename}")
    plt.show() # Uncomment this if running interactively