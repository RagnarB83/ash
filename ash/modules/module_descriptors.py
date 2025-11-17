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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

# --- Imports for DMD and BOM ---
from scipy.spatial.distance import pdist, cdist, squareform
import matplotlib.pyplot as plt

# --- User's new imports for tblite (BOM) ---
from ash.interfaces.interface_xtb import tbliteTheory
from ash.modules.module_coords import Fragment


# ==============================================================================
# --- SECTION 1: DESCRIPTOR HELPER FUNCTIONS ---
# ==============================================================================

def frame_SOAPdescriptor(soap, frame, average_mode):
    """
    Computes a single, normalized descriptor vector for an entire frame
    using dscribe's SOAP.
    """
    ## Determine structure format and load using ASH fragment
    if isinstance(frame, str):
        if not os.path.isfile(frame):
             ashexit(f"File not found in frame_descriptor: {frame}")
        # Using direct 'Fragment' import
        if frame.endswith('.pdb'):
            structure_obj = Fragment(pdbfile=frame, printlevel=0)
        elif frame.endswith('.xyz'):
            structure_obj = Fragment(xyzfile=frame, printlevel=0)
        else:
            ashexit(f"Unsupported file format in frame_descriptor: {frame}")
    else:
        ashexit("frame_descriptor currently only supports file paths as input.")

    ## Convert ASH fragment to ASE Atoms object
    try:   
        ase_structure = Atoms(symbols=structure_obj.elems, positions=structure_obj.coords)
    except Exception as e:
        ashexit(f"Error creating ASE Atoms object in frame_descriptor: {e}")

    ## CREATE descriptor vector using ASE object
    v = soap.create(ase_structure, n_jobs=1) 
    
    if average_mode == "off":
        v = v.mean(axis=0)
    
    # Normalize the vector (L2 norm)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    
    return v.astype(np.float32)


def frame_DMdescriptor(frame):
    """
    Computes a 1D descriptor vector for a frame based on its
    internal pairwise atomic distances.
    """
    from scipy.spatial.distance import pdist
    
    if isinstance(frame, str):
        if not os.path.isfile(frame):
             ashexit(f"File not found in frame_DMdescriptor: {frame}")
        # Using direct 'Fragment' import
        if frame.endswith('.xyz'):
            structure_obj = Fragment(xyzfile=frame, printlevel=0)
        else:
            ashexit(f"Unsupported file format in frame_DMdescriptor: {frame}")
    else:
        ashexit("frame_DMdescriptor currently only supports file paths as input.")
    try:   
        coords = structure_obj.coords
    except Exception as e:
        ashexit(f"Error getting coordinates in frame_DMdescriptor: {e}")

    v = pdist(coords) 
    return v.astype(np.float32)


def calc_BOmatrix_tblite(xyz_file):
    """
    Calculates Bond Order Matrix using tblite for a given xyz file.
    Returns the BOM as a 2D numpy array.
    (Incorporating user's latest version)
    """
    try:  
        
        frag_temp = Fragment(xyzfile=xyz_file, charge=0, mult=1, printlevel=0) 
        theory_temp = tbliteTheory(method="GFN2-xTB", grab_BOs=True)
        ash.Singlepoint(theory=theory_temp, fragment=frag_temp, printlevel=0)
        abc = theory_temp.BOs

        BOs_array = np.array(abc)
        return BOs_array
    
    except Exception as e:
        print(f"Error calculating BOM for {xyz_file} using tblite: {e}")
        return None


def frame_BOMdescriptor(frame):
    """
    Computes a 1D descriptor vector for a frame based on its
    internal Bond Order Matrix (BOM).
    """
    # 1. Calculate the 2D BOM
    bom_matrix = calc_BOmatrix_tblite(frame)
    
    if bom_matrix is None:
        return None

    # 2. Flatten the 2D matrix (upper triangle) to a 1D vector
    try:
        num_atoms = bom_matrix.shape[0]
        v = bom_matrix[np.triu_indices(num_atoms, k=1)]
        return v.astype(np.float32)
    except Exception as e:
        print(f"Error flattening BOM for {frame}: {e}")
        return None

# ==============================================================================
# --- SECTION 2: PLOTTING HELPER FUNCTIONS ---
# ==============================================================================

def plot_soap_clusters(X, kmeans, title="SOAP Vector Clusters (t-SNE)"):
    """
    Visualizes high-dimensional SOAP vectors in 2D using t-SNE.
    """
    labels = kmeans.labels_
    n_clusters = kmeans.n_clusters
    exemplar_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    exemplar_indices = sorted(list(set(exemplar_indices)))
    exemplar_mask = np.zeros(X.shape[0], dtype=bool)
    exemplar_mask[exemplar_indices] = True
    print("Running t-SNE for visualization... (this may take a moment)")
    X_std = StandardScaler().fit_transform(X)
    n_samples = X.shape[0]
    perp = min(30, n_samples - 1)
    if perp <= 0:
        print("Warning: Not enough samples for t-SNE. Skipping plot.")
        return
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=300)
    X_2d = tsne.fit_transform(X_std)
    print("t-SNE complete.")
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('jet', n_clusters)
    for i in range(n_clusters):
        cluster_mask = (labels == i)
        cluster_non_exemplars = cluster_mask & ~exemplar_mask
        plt.scatter(
            X_2d[cluster_non_exemplars, 0], 
            X_2d[cluster_non_exemplars, 1], 
            color=colors(i), 
            label=f'Cluster {i}' if i not in labels[exemplar_indices] else None, 
            alpha=0.4,
            s=20 
        )
    plt.scatter(
        X_2d[exemplar_mask, 0], 
        X_2d[exemplar_mask, 1], 
        marker='*', 
        c=labels[exemplar_mask], 
        cmap=plt.cm.get_cmap('jet', n_clusters),
        edgecolor='k', 
        s=400, 
        label='Exemplars (Selected)'
    )
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) 
    plt.legend(by_label.values(), by_label.keys(), loc='best', markerscale=1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_filename = "soap_cluster_visualization.png"
    plt.savefig(plot_filename)
    print(f"Cluster visualization saved to: {plot_filename}")


def get_atom_labels(xyz_file):
    """
    Reads an xyz file and returns a list
    of unique atom labels (e.g., C1, N1, H1, H2...).
    """
    try:
        # Using direct 'Fragment' import
        fragment = Fragment(xyzfile=xyz_file, printlevel=0)
        symbols = fragment.elems
        labels = []
        counts = {}
        for sym in symbols:
            counts[sym] = counts.get(sym, 0) + 1
            labels.append(f"{sym}{counts[sym]}")
        return labels
    except Exception as e:
        print(f"Error reading {xyz_file} for labels: {e}")
        return None


def create_heatmap(diff_matrix, atom_labels, title, output_filename, cbar_label, vmin, vmax):
    """
    Generates and saves a single heatmap for a Diff_DM or Diff_BOM.
    """
    print(f"  Plotting {output_filename}...")
    try:
        num_atoms = len(atom_labels)
        fig_size = max(8, num_atoms / 2.5) 
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        cax = ax.imshow(diff_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(cax, label=cbar_label, shrink=0.8)
        ax.set_xticks(np.arange(num_atoms))
        ax.set_yticks(np.arange(num_atoms))
        ax.set_xticklabels(atom_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(atom_labels, fontsize=8)
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xticks(np.arange(num_atoms + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(num_atoms + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"    Failed to plot {output_filename}: {e}")


# ==============================================================================
# --- SECTION 3: MASTER SELECTION FUNCTION ---
# ==============================================================================

def SelectDiverseFrames(
                descriptor_type, # 'soap', 'distance', or 'bond_order'
                reference_dir=None, reference_traj=None, xyztraj=None, xyzdir=None,
                n_clusters=None,
                incremental=False,
                # SOAP params
                sim_threshold=0.9,
                visualize_clusters=False,
                r_cut=5.0,
                n_max=8,
                l_max=6,
                sigma=0.5,
                average='outer', ## "inner", "outer", or "off", outer is recommended
                # Distance/Bond Matrix params
                dissim_threshold=None, #1.0 for distance, 0.1 for bond_order
                plot_heatmaps=True, 
                heatmap_output_dir=None): 
    """
    Selects diverse frames from a trajectory using one of three descriptors:
    1. 'soap': Uses dscribe's SOAP descriptor.
    2. 'distance': Uses a pairwise atomic Distance Matrix Descriptor (DMD).
    3. 'bond_order': Uses a tblite-calculated Bond Order Matrix (BOM) descriptor.
    
    Parameters:
    descriptor_type (str): The descriptor to use. Must be 'soap', 'distance', or 'bond_order'.
    reference_dir (str, optional): Path to a directory of reference .xyz files.
    reference_traj (str, optional): Path to a multi-frame .xyz reference trajectory.
    xyztraj (str, optional): Path to the main .xyz trajectory to select from.
    xyzdir (str, optional): Path to a directory of .xyz frames to select from.
    n_clusters (int, optional): If set, performs K-means clustering (overrides other modes).
    incremental (bool, optional): If True, uses incremental selection (reference-based only).
    
    --- SOAP Parameters (if descriptor_type = 'soap') ---
    sim_threshold (float): Similarity threshold (0-1). Default: 0.5
    visualize_clusters (bool): Plot t-SNE of K-means. Default: False
    r_cut (float): Cutoff radius. Default: 5.0
    n_max (int): Max radial basis functions. Default: 8
    l_max (int): Max spherical harmonics. Default: 6
    sigma (float): Gaussian width. Default: 0.5
    average (str): "inner", "outer", or "off". Default: "inner"
    
    --- Matrix Parameters (if descriptor_type = 'distance' or 'bond_order') ---
    dissim_threshold (float): Euclidean distance threshold.
                              Default: 1.0 (for 'distance'), 0.1 (for 'bond_order')
    plot_heatmaps (bool): Plot heatmaps of the difference matrix. Default: False
    heatmap_output_dir (str): Directory for heatmaps.
                              Default: 'dmd_heatmaps' or 'bom_heatmaps'
    """

    # --- 1. Set up descriptor-specific variables ---
    
    if descriptor_type == 'soap':
        selection_mode_type = 'similarity'
        split_suffix = '_soap'
        print_descriptor_name = "SOAP"
        # plot_heatmaps is already defined as False from signature
        
    elif descriptor_type in ['distance', 'bond_order']:
        selection_mode_type = 'distance'
        
        if descriptor_type == 'distance':
            frame_descriptor_func = frame_DMdescriptor
            split_suffix = '_dmd'
            cbar_label = 'Distance Change (Å)'
            if dissim_threshold is None: dissim_threshold = 1.0
            if heatmap_output_dir is None: heatmap_output_dir = 'dmd_heatmaps'
            print_descriptor_name = "Distance Matrix (DMD)"
        else: # 'bond_order'
            frame_descriptor_func = frame_BOMdescriptor
            split_suffix = '_bom'
            cbar_label = 'Bond Order Difference'
            if dissim_threshold is None: dissim_threshold = 0.1
            if heatmap_output_dir is None: heatmap_output_dir = 'bom_heatmaps'
            print_descriptor_name = "Bond Order Matrix (BOM)"
            
        # Cluster visualization is not supported for matrix descriptors
        visualize_clusters = False

    else:
        ashexit(f"Error: Unknown descriptor_type '{descriptor_type}'. Must be 'soap', 'distance', or 'bond_order'.")

    print(f"--- Running Selection using {print_descriptor_name} ---")

    # --- 2. Sanity checks ---
    if xyzdir is None and xyztraj is None:
        ashexit("Error: Must provide either 'xyzdir' or 'xyztraj' parameters.")
    if reference_dir is None and reference_traj is None:
        print("No Reference structure provided, required to determine atomic species/reference.")
        print("will take the reference from the first frame in the trajectory/directory.")
  
    if n_clusters is not None:
        print(f"Mode: K-Means Clustering (n_clusters={n_clusters}).")
        selection_mode = "kmeans"
        if plot_heatmaps:
            print("Warning: 'plot_heatmaps' is not compatible with 'n_clusters'. Disabling plotting.")
            plot_heatmaps = False 
    else:
        selection_mode = "reference_based"
        if selection_mode_type == 'similarity':
            if not (0.0 <= sim_threshold <= 1.0):
                ashexit("Error: 'sim_threshold' must be between 0 and 1.")
            threshold_val = sim_threshold
            comparison_str = "sim <"
        else: # distance
            if dissim_threshold < 0.0:
                ashexit("Error: 'dissim_threshold' must be >= 0.")
            threshold_val = dissim_threshold
            comparison_str = "dist >"

        if incremental:
            print(f"Mode: Incremental Reference ({comparison_str} {threshold_val}).")
        else:
            print(f"Mode: Fixed-Reference Threshold ({comparison_str} {threshold_val}).")
            
    if descriptor_type == 'soap' and average not in ["inner", "outer", "off"]:
         ashexit(f"Error: Invalid average mode '{average}'. Must be 'inner', 'outer', or 'off'.")

    # --- 3. File reading/splitting ---
    if xyzdir is not None:
        print("Reading trajectory from directory of xyz files, xyztraj_would be ignored.")
        xyz_directory = os.path.abspath(xyzdir)
        list_xyz_files = natural_sort(glob.glob(os.path.join(xyz_directory, "*.xyz")))
        print(f"Total frames found: {len(list_xyz_files)}")
        if not list_xyz_files:
            ashexit(f"No .xyz files found in directory: {xyz_directory}")
    else:
        list_xyz_files = []
    
    if xyzdir is None and xyztraj is not None:
        if xyztraj.endswith('.xyz'): 
            print("Reading trajectory from xyz file.")
            xyz_traj = os.path.abspath(xyztraj)
            split_dir = "xyz_traj_split" + split_suffix
            shutil.rmtree(split_dir, ignore_errors=True)
            os.makedirs(split_dir, exist_ok=True)
            cwd = os.getcwd()
            os.chdir(split_dir)
            try:
                split_multimolxyzfile(xyz_traj, writexyz=True, skipindex=1, return_fragments=False)
            except Exception as e:
                ashexit(f"Failed to split xyz file. Error: {e}")
            list_xyz_files = [os.path.abspath(f) for f in natural_sort(glob.glob("*.xyz"))]
            print(f"Total frames found: {len(list_xyz_files)}")
            os.chdir(cwd) 
        else:
            ashexit(f"Unsupported trajectory file format: {xyztraj}")
    else:
        pass
    
    reference_files = [] 
    if reference_dir is not None or reference_traj is not None:
        if reference_dir is not None:
            reference_directory = os.path.abspath(reference_dir)
            ref_files_from_dir = natural_sort(glob.glob(os.path.join(reference_directory, "*.xyz")))
            reference_files.extend(ref_files_from_dir)
        if reference_traj is not None:
            if reference_traj.endswith('.xyz'):
                print("Reading reference structures from xyz trajectory file.")
                ref_traj = os.path.abspath(reference_traj)
                split_ref_dir = "reference_traj_split" + split_suffix
                shutil.rmtree(split_ref_dir, ignore_errors=True)
                os.makedirs(split_ref_dir, exist_ok=True)
                cwd = os.getcwd()
                os.chdir(split_ref_dir)
                try:
                    split_multimolxyzfile(ref_traj, writexyz=True, skipindex=1, return_fragments=False)
                except Exception as e:
                    ashexit(f"Failed to split reference xyz file. Error: {e}")
                ref_files_from_traj = [os.path.abspath(f) for f in natural_sort(glob.glob("*.xyz"))]
                reference_files.extend(ref_files_from_traj)
                os.chdir(cwd) 
            else:
                ashexit(f"Unsupported reference trajectory file format: {reference_traj}") 
        
    if reference_files:
        print(f"Total reference files found: {len(reference_files)}")
        reference_for_species = reference_files[0]
    elif list_xyz_files:
        print("No valid reference files found. Using first frame in trajectory/directory as reference.")
        reference_for_species = list_xyz_files[0]
        if not reference_files: 
             reference_files = [reference_for_species]
    else:
        ashexit("Error: No reference files and no trajectory files found. Cannot determine species.")
    
    if plot_heatmaps and len(reference_files) > 1:
        print(f"Warning: 'plot_heatmaps' requires a single reference. You provided {len(reference_files)}.")
        print("Disabling heatmap plotting.")
        plot_heatmaps = False
        
    # --- 4. Initialize Descriptor ---
    if descriptor_type == 'soap':
        print(f"Using structure: {reference_for_species} for atomic species determination.")
        # Using direct 'Fragment' import
        frag_ref = Fragment(xyzfile=reference_for_species, printlevel=0)
        species = frag_ref.elems
        print(f"Atomic species determined from reference: {species}")
        soap = SOAP(species=species,r_cut=r_cut,n_max=n_max,l_max=l_max,sigma=sigma,average=average)

    ## Save selected frames to a dedicated output directory (common for all modes) ##
    output_dir = "selected_diverse_frames" + split_suffix
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Selected frames will be saved to: {output_dir}")
    
    ## 5. Descriptor calculation loop ##
    all_vectors = {}
    print(f"\nProcessing {len(list_xyz_files)} trajectory frames to compute {print_descriptor_name} vectors...")
    if descriptor_type == 'bond_order':
        print("This may be slow as it requires a tblite calculation for each frame.")
        
    for i, frame_path in enumerate(list_xyz_files):
        try:
            if descriptor_type == 'soap':
                frame_vector = frame_SOAPdescriptor(soap, frame_path, average)
            else: # distance or bond_order
                frame_vector = frame_descriptor_func(frame_path)
                
            if frame_vector is not None:
                all_vectors[frame_path] = frame_vector
            else:
                print(f"Warning: Descriptor calculation failed for {frame_path}. Skipping.")
        except Exception as e:
            print(f"Warning: Failed to process frame {frame_path}. Skipping. Error: {e}")
            continue 
        
        progress_chunk = 50 if descriptor_type == 'bond_order' else 200
        if (i + 1) % progress_chunk == 0 or (i + 1) == len(list_xyz_files):
            print(f"  ... processed {i + 1}/{len(list_xyz_files)} frames.")
    
    list_xyz_files = list(all_vectors.keys())
    if not all_vectors:
        ashexit(f"Failed to compute any {print_descriptor_name} vectors.")
    print("\nAll frame vectors are computed and ready.")

    selected_frames_info = [] 

    ## 6. Selection Logic ##
    if selection_mode == "kmeans":
        print(f"Running K-Means clustering to find {n_clusters} clusters...")
        frame_paths_list = list(all_vectors.keys())
        all_vectors_list = list(all_vectors.values())
        X = np.array(all_vectors_list, dtype=np.float32)
        print(f"Data matrix shape for clustering: {X.shape}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) 
        kmeans.fit(X)
        print("Clustering complete.")
        
        if descriptor_type == 'soap' and visualize_clusters:
            try:
                print("Generating cluster visualization plot...")
                plot_soap_clusters(X, kmeans)
            except Exception as e:
                print(f"Warning: Could not generate cluster plot. Error: {e}")
                
        print("Finding the most representative frame (exemplar) for each cluster...")
        closest_frame_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        unique_indices = sorted(list(set(closest_frame_indices)))
        cluster_labels_for_exemplars = kmeans.labels_[unique_indices]
        for i, frame_idx in enumerate(unique_indices):
            frame_path = frame_paths_list[frame_idx]
            cluster_id = cluster_labels_for_exemplars[i]
            selected_frames_info.append((frame_path, cluster_id, None)) 
        print(f"Found {len(selected_frames_info)} unique exemplars.")

    elif selection_mode == "reference_based":
        # --- 1. Compute Reference Vectors ---
        print("Computing descriptor(s) for initial reference frame(s)...")
        ref_vectors = {} 
        if not reference_files:
            ashexit(f"Error: '{selection_mode}' mode requires at least one reference file.")

        for ref_path in reference_files:
            try:
                print(f"Processing reference structure: {ref_path}")
                if ref_path in all_vectors:
                     ref_vector = all_vectors[ref_path]
                else:
                    if descriptor_type == 'soap':
                        ref_vector = frame_SOAPdescriptor(soap, ref_path, average)
                    else:
                        ref_vector = frame_descriptor_func(ref_path)
                
                if ref_vector is not None:
                    ref_vectors[ref_path] = ref_vector
                else:
                    print(f"Warning: Descriptor calculation failed for {ref_path}. Skipping.")
            except Exception as e:
                print(f"Warning: Failed to process reference frame {ref_path}. Skipping. Error: {e}")
                continue
        
        if not ref_vectors:
                ashexit(f"Error: Failed to compute {print_descriptor_name} vectors for any of the provided reference files.")
        print(f"Computed {len(ref_vectors)} initial reference vectors.")
        
        ref_vector_for_plotting = ref_vectors[reference_files[0]] if plot_heatmaps else None

        # --- 2. Apply Selection Logic ---
        if incremental:
            print(f"Running incremental selection...")
            selected_vectors_list = list(ref_vectors.values())
            selected_paths_set = set(ref_vectors.keys())
            for ref_path in ref_vectors.keys():
                v_diff = (ref_vectors[ref_path] - ref_vector_for_plotting) if plot_heatmaps else None
                selected_frames_info.append((ref_path, "Initial Reference", v_diff))
            print(f"Starting with {len(selected_frames_info)} initial reference frame(s).")
            
            for frame_path, frame_vector in all_vectors.items():
                if frame_path in selected_paths_set:
                    continue
                
                if selection_mode_type == 'similarity':
                    max_sim_to_selected = 0.0
                    for selected_vec in selected_vectors_list:
                        cos_sim = np.dot(frame_vector, selected_vec)
                        if cos_sim > max_sim_to_selected:
                            max_sim_to_selected = cos_sim
                    
                    if max_sim_to_selected < sim_threshold:
                        selected_frames_info.append((frame_path, max_sim_to_selected, None)) 
                        selected_vectors_list.append(frame_vector)
                        selected_paths_set.add(frame_path)
                
                else: # 'distance'
                    min_dist_to_selected = np.inf
                    for selected_vec in selected_vectors_list:
                        dist = np.linalg.norm(frame_vector - selected_vec) 
                        if dist < min_dist_to_selected:
                            min_dist_to_selected = dist
                    
                    if min_dist_to_selected > dissim_threshold:
                        v_diff = (frame_vector - ref_vector_for_plotting) if plot_heatmaps else None
                        selected_frames_info.append((frame_path, min_dist_to_selected, v_diff)) 
                        selected_vectors_list.append(frame_vector)
                        selected_paths_set.add(frame_path)
            
            print(f"Incremental selection finished. Total diverse frames: {len(selected_frames_info)}.")

        else: # Fixed-Reference
            print(f"Running fixed-reference selection...")
            ref_vector_list = list(ref_vectors.values()) # Fixed list
            
            for frame_path, frame_vector in all_vectors.items():
                if frame_path in ref_vectors:
                    continue
                
                if selection_mode_type == 'similarity':
                    similarities_to_refs = [np.dot(frame_vector, ref_vec) for ref_vec in ref_vector_list]
                    max_sim = np.max(similarities_to_refs)
                    if max_sim < sim_threshold:
                        selected_frames_info.append((frame_path, max_sim, None))
                
                else: # 'distance'
                    distances_to_refs = [np.linalg.norm(frame_vector - ref_vec) for ref_vec in ref_vector_list]
                    min_dist = np.min(distances_to_refs)
                    if min_dist > dissim_threshold:
                        v_diff = (frame_vector - ref_vector_for_plotting) if plot_heatmaps else None
                        selected_frames_info.append((frame_path, min_dist, v_diff))
            
            print(f"Fixed-reference selection finished. Found {len(selected_frames_info)} matching frames.")

    # --- 7. Copying ---
    print(f"\nCopying {len(selected_frames_info)} selected frames to {output_dir}...")
    selected_conformations_paths = []
    for item in selected_frames_info:
        frame_path = item[0]
        value = item[1] 
        
        if not os.path.exists(frame_path):
            frame_basename = os.path.basename(frame_path)
        else:
            frame_basename = os.path.basename(frame_path)
        dest_file = os.path.join(output_dir, frame_basename)
        try:
            shutil.copy(frame_path, dest_file)
            selected_conformations_paths.append(dest_file)
            
            if selection_mode == "kmeans":
                print(f"  - Copied {frame_basename} (Exemplar for cluster {value})")
            elif isinstance(value, str): 
                print(f"  - Copied {frame_basename} ({value})")
            elif selection_mode_type == 'similarity': 
                print(f"  - Copied {frame_basename} (Max sim: {value:.4f})")
            else: # 'distance'
                print(f"  - Copied {frame_basename} (Min dist: {value:.4f})")
                
        except Exception as e:
            print(f"  - Error copying {frame_basename}: {e}")
    print(f"\nProcessing complete. {len(selected_conformations_paths)} diverse frames saved.")
    
    # --- 8. Plotting ---
    if plot_heatmaps:
        print(f"\n--- Generating Heatmaps ({print_descriptor_name}) ---")
        try:
            os.makedirs(heatmap_output_dir, exist_ok=True)
            ref_file = reference_files[0]
            atom_labels = get_atom_labels(ref_file)
            ref_basename = os.path.basename(ref_file).replace('.xyz', '')
            
            if atom_labels:
                all_v_diffs = [item[2] for item in selected_frames_info if item[2] is not None]
                if all_v_diffs:
                    v_abs_max = np.max(np.abs(np.concatenate(all_v_diffs)))
                    if v_abs_max < 0.01: v_abs_max = 0.1 # Set a floor
                    print(f"Global heatmap color scale set to: -{v_abs_max:.3f} to +{v_abs_max:.3f}")
                else:
                    v_abs_max = 1.0 # Fallback
            
                for item in selected_frames_info:
                    frame_path, value, v_diff = item
                    if v_diff is None: continue
                    
                    diff_matrix = squareform(v_diff)
                    frame_basename = os.path.basename(frame_path).replace('.xyz', '')
                    output_filename = os.path.join(
                        heatmap_output_dir, 
                        f"{ref_basename}_vs_{frame_basename}{split_suffix}.png"
                    )
                    title = f"{cbar_label.split(' ')[0]} Change: {ref_basename} vs. {frame_basename}"
                    
                    create_heatmap(
                        diff_matrix, 
                        atom_labels, 
                        title, 
                        output_filename, 
                        cbar_label=cbar_label,
                        vmin=-v_abs_max, 
                        vmax=v_abs_max
                    )
                print(f"Heatmaps saved to {heatmap_output_dir}")
            else:
                print("Could not get atom labels from reference. Skipping heatmap plotting.")
        except Exception as e:
            print(f"Could not generate heatmaps. Error: {e}")

    # --- 9. Cleanup ---
    shutil.rmtree("xyz_traj_split" + split_suffix, ignore_errors=True)
    shutil.rmtree("reference_traj_split" + split_suffix, ignore_errors=True)

    return len(selected_conformations_paths)