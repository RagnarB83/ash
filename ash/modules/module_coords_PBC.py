import numpy as np
from ash.functions.functions_general import ashexit

# This module contains functions for handling periodic boundary conditions (PBC) and related coordinate transformations.



#Function that calculates box size of a molecule in a cubic box
#with optional shift
def cubic_box_size(coords, shift=0.0):
    # max and min for x,y,z coords
    max_values = np.max(coords, axis=0)
    min_values = np.min(coords, axis=0)
    #Differences for x,y,z
    span_x = max_values[0] - min_values[0]
    span_y = max_values[1] - min_values[1]
    span_z = max_values[2] - min_values[2]
    # Max span for each x,y,z
    max_span = max(span_x, span_y, span_z)
    #Optional shift
    final_span = max_span + shift
    return final_span

#More general
def bounding_box_dimensions(coordinates,shift=0.0):
    # Get max and min values for x, y, z coordinates
    max_values = np.max(coordinates, axis=0)
    min_values = np.min(coordinates, axis=0)

    # Calculate the differences along each axis to determine dimensions
    dimensions = max_values - min_values
    final_dims = dimensions + shift
    return dimensions  # Return the dimensions of the bounding box



def cell_params_to_vectors(parameters):
    a, b, c, alpha, beta, gamma = parameters
    # Convert angles to radians
    rad_a = np.radians(alpha)
    rad_b = np.radians(beta)
    rad_g = np.radians(gamma)
    
    # Calculate components
    ax = a
    ay = 0.0
    az = 0.0
    
    bx = b * np.cos(rad_g)
    by = b * np.sin(rad_g)
    bz = 0.0
    
    cx = c * np.cos(rad_b)
    cy = c * (np.cos(rad_a) - np.cos(rad_b) * np.cos(rad_g)) / np.sin(rad_g)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    
    vectors = np.array([[ax,ay,az],[bx,by,bz],[cx,cy,cz]])
    return vectors

def cell_vectors_to_params(vectors):
    va, vb, vc = vectors[0], vectors[1], vectors[2]
    
    # Calculate lengths (norms)
    a = np.linalg.norm(va)
    b = np.linalg.norm(vb)
    c = np.linalg.norm(vc)
    
    # Calculate angles using the dot product formula: 
    # cos(theta) = (v1 . v2) / (|v1| * |v2|)
    alpha_rad = np.arccos(np.dot(vb, vc) / (b * c))
    beta_rad  = np.arccos(np.dot(va, vc) / (a * c))
    gamma_rad = np.arccos(np.dot(va, vb) / (a * b))
    
    # Convert radians to degrees
    alpha = np.degrees(alpha_rad)
    beta  = np.degrees(beta_rad)
    gamma = np.degrees(gamma_rad)
    
    return [float(a), float(b), float(c), float(alpha), float(beta), float(gamma)]

# Basic conversion of Cartesian coordinates to fractional coordinates and reverse
def cart_coords_to_fract(cart_coords, cellvectors):
    M = np.array(cellvectors)
    frac = np.dot(cart_coords, np.linalg.inv(M))
    return frac

def fract_coords_to_cart(fract_coords, cellvectors):
    cart = np.dot(fract_coords, np.array(cellvectors))
    return cart

def cell_volume(vectors):
    a = vectors[0,:]
    b = vectors[1,:]
    c = vectors[2,:]
    V = abs(np.dot(a, np.cross(b, c)))
    return V

# Write Cartesian-based POSCAR files
def write_POSCAR_file(coords,elems,cellvectors=None, celldimensions=None, filename="POSCAR"):

    if cellvectors is None and celldimensions is None:
        print("Error: Either cellvectors or celldimensions should be provided")
        ashexit()
    elif celldimensions is not None:
        # converting 
        cellvectors=cell_params_to_vectors(celldimensions)

    # Unique elements in original order
    unique_elements = []
    for e in elems:
        if e not in unique_elements:
            unique_elements.append(e)
    # Count atoms of each elemtype
    counts = [elems.count(e) for e in unique_elements]

    with open(filename, 'w') as f:
        f.write("ASH created POSCAR file"+"\n")
        f.write("1.0"+"\n")
        f.write(f"{cellvectors[0,0]:.4f} {cellvectors[0,1]:.4f} {cellvectors[0,2]:.4f} "+"\n")
        f.write(f"{cellvectors[1,0]:.4f} {cellvectors[1,1]:.4f} {cellvectors[1,2]:.4f}"+"\n")
        f.write(f"{cellvectors[2,0]:.4f} {cellvectors[2,1]:.4f} {cellvectors[2,2]:.4f}"+"\n")
        f.write(f"{'  '.join(unique_elements)}\n")
        f.write(f"{'  '.join(map(str, counts))}\n")
        f.write(f"Cartesian"+"\n")# coord system
        for target_el in unique_elements:
                    for el, c in zip(elems, coords):
                        if el == target_el:
                            f.write(f"{c[0]:.8f}  {c[1]:.8f}  {c[2]:.8f}\n")
    print("Wrote POSCAR file")
    return filename

# Write XSF files
def write_XSF_file(coords, elems, cellvectors=None, celldimensions=None, filename="structure.xsf"):

    if cellvectors is None and celldimensions is None:
        print("Error: Either cellvectors or celldimensions should be provided")
        ashexit()
    elif celldimensions is not None:
        # Assuming your helper function handles the conversion
        cellvectors = cell_params_to_vectors(celldimensions)

    with open(filename, 'w') as f:
        # Header for periodic structures
        f.write("CRYSTAL\n")
        
        # Section 1: Lattice Vectors
        f.write("PRIMVEC\n")
        for i in range(3):
            f.write(f"  {cellvectors[i,0]:.10f}  {cellvectors[i,1]:.10f}  {cellvectors[i,2]:.10f}\n")
        
        # Section 2: Atomic Coordinates
        f.write("PRIMCOORD\n")
        # Header for coordinates: [Number of atoms] [Number of units, usually 1]
        f.write(f"{len(elems)} 1\n")
        
        # XSF supports either Atomic Number or Element Symbol. 
        # Using Element Symbol is more human-readable and works perfectly in VMD.
        for el, c in zip(elems, coords):
            f.write(f"{el}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n")
            
    print(f"Wrote XSF file: {filename}")
    return filename


def write_CIF_file(coords, elems, cellvectors=None, celldimensions=None, filename="structure.cif"):

    if cellvectors is None and celldimensions is None:
        print("Error: Either cellvectors or celldimensions should be provided")
        ashexit()
    elif celldimensions is not None:
        # Assuming your helper function handles the conversion
        cellvectors = cell_params_to_vectors(celldimensions)
    elif cellvectors is not None:
        celldimensions = cell_vectors_to_params(cellvectors)

    # Cart to fract
    frac_coords = cart_coords_to_fract(coords,cellvectors)

    # celldimensions should be [a, b, c, alpha, beta, gamma]
    a, b, c, alpha, beta, gamma = celldimensions

    with open(filename, 'w') as f:
        f.write("data_ASH_output\n")
        f.write(f"_cell_length_a    {a:.6f}\n")
        f.write(f"_cell_length_b    {b:.6f}\n")
        f.write(f"_cell_length_c    {c:.6f}\n")
        f.write(f"_cell_angle_alpha {alpha:.6f}\n")
        f.write(f"_cell_angle_beta  {beta:.6f}\n")
        f.write(f"_cell_angle_gamma {gamma:.6f}\n\n")
        
        # We use P1 symmetry (no symmetry) so every atom is listed explicitly
        f.write("_symmetry_space_group_name_H-M 'P 1'\n")
        f.write("_symmetry_Int_Tables_number 1\n\n")
        
        # The Atom Loop
        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        
        for i, (el, c) in enumerate(zip(elems, frac_coords)):
            # We add an index to the label (e.g., Na1, Na2) to keep them unique
            f.write(f"{el}{i+1}  {el}  {c[0]:.8f}  {c[1]:.8f}  {c[2]:.8f}\n")

    print(f"Wrote CIF file: {filename}")
    return filename

def align_to_standard_orientation(fragment_coords, cell_vectors):
        """
        Rotates the entire system (atoms and cell) into the standard 
        upper-triangular orientation.
        
        cell_vectors: 3x3 matrix where rows are [a, b, c]
        fragment_coords: Nx3 array of atomic positions
        """
        # 1. Transpose cell_vectors because QR works on columns
        H = cell_vectors.T 
        
        # 2. QR Decomposition
        # H = Q * R  -> R is the upper triangular matrix we want
        Q, R = np.linalg.qr(H)
        
        # 3. Handle 'Flip' cases
        # QR can sometimes return negative diagonal elements. 
        # We want lengths (a_x, b_y, c_z) to be positive.
        d = np.sign(np.diag(R))
        # If a diagonal is 0, we treat it as positive
        d[d == 0] = 1
        
        # Correct Q and R so diagonals of R are positive
        Q = Q * d
        R = (R.T * d).T
        
        # 4. New Cell Vectors (R transposed back to rows)
        new_cell_vectors = R.T
        
        # 5. New Atomic Coordinates
        # We rotate the atoms using the same rotation matrix Q
        # Since H_new = Q.T @ H_old, we use Q.T for the atoms
        new_coords = np.dot(fragment_coords, Q)
        
        return new_coords, new_cell_vectors
