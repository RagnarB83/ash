import math
import shutil
import sys
import os
import re
import linecache
import itertools

# Defining covalent radii
cov_radii = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Po": 1.40,
    "At": 1.50,
    "Rn": 1.50,
}

elements = list(cov_radii.keys())


# Function to determine if supplied cubefile has .cub or .cube extension
def cub_or_cube(cubefile):
    cubename = ""
    if cubefile.endswith(".cub"):
        cubename = cubefile[:-4]
    elif cubefile.endswith(".cube"):
        cubename = cubefile[:-5]

    return cubename


# Funciton to get geometry from cubefile in angstroms
def get_geom_from_cube(cubefile):
    geometry = []
    angstrom = 0.529177
    cubename = cub_or_cube(cubefile)

    # Get number of atoms
    num_atoms = int(linecache.getline(cubefile, 3).split()[0])
    # Loop through cube, starting at beginning of geometry specification
    with open(cubefile, "r") as cube:
        for line in itertools.islice(cube, 6, 6 + num_atoms):
            # Getting element symbol and coordinates
            current = line.split()
            el_symbol = elements[int(current[0]) - 1]
            geometry.append(
                [
                    el_symbol,
                    [
                        float(current[2]) * angstrom,
                        float(current[3]) * angstrom,
                        float(current[4]) * angstrom,
                    ],
                ]
            )

    return geometry, cubename


# Function to get and return coordinates of attractors
def get_attractors(pdbfile):
    coords = []
    with open(pdbfile) as file:
        # Loop through each line and grab the attractor number and its coordinates
        for line in file:
            attractor, xcoord, ycoord, zcoord = [line.split()[i] for i in (5, 6, 7, 8)]
            coords.append([attractor, [float(xcoord), float(ycoord), float(zcoord)]])

    return coords


# Function to calculate and return a distance matrix
def calc_distances(geom, attractors):
    distance_matrix = []

    for attractor in attractors:
        for atom in geom:
            # Assign variables to make distance calculation more readable
            attractor_num = attractor[0]
            atom_symbol = atom[0]
            attr_xcoord, attr_ycoord, attr_zcoord = attractor[1]
            atom_xcoord, atom_ycoord, atom_zcoord = atom[1]

            # Distance calculation
            distance = math.sqrt(
                (attr_xcoord - atom_xcoord) ** 2
                + (attr_ycoord - atom_ycoord) ** 2
                + (attr_zcoord - atom_zcoord) ** 2
            )
            distance_matrix.append((attractor_num, atom_symbol, distance))

    # Split distance matrix into separate sublists, each containing the distance to each atom for a specific attractor
    distance_matrix_split = list(chunks(distance_matrix, len(geom)))

    return distance_matrix_split


# Function that splits a list into evenly sized chunks
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# Generate range of floats to used for range of scaling factors
def frange(radius, scaling_factor=1.00):
    while scaling_factor < 1.40:
        yield (radius * scaling_factor)
        scaling_factor += 0.05


# Function to assign attractors to atoms, and specify whether they are CORE or VALENCE
def assign(distance_matrix, core_threshold=0.4):
    assignments = []
    for attractor in distance_matrix:
        core_atom_list = []
        valence_atom_list = []
        for atom_index, atom in enumerate(attractor):
            # Get distance to current atom
            distance = atom[2]

            # Generate list of scaled covalent radii for current atom (scaling_factor = 1.0, 1.05, ... , 1.40)
            cov_radius_scaled = list(frange(cov_radii[atom[1]]))

            if distance > core_threshold:
                # Check if the distance is within any of the scaled radii for the current atom
                # If so, break from the loop to avoid going to higher scaling factors
                # This is a very safe way to test scaling factors --> Confined to individual atoms rather than using global scaling factor
                # This means that in theory, the same elements in different chemical environments will be accounted for
                for scaled_radius in cov_radius_scaled:
                    if distance < scaled_radius:
                        valence_atom_list.append((atom[1], atom_index))
                        break
            elif distance < core_threshold:
                core_atom_list.append((atom[1], atom_index))

        if len(core_atom_list) == 1:
            assignments.append((atom[0], "CORE", core_atom_list))
        elif 0 < len(valence_atom_list) <= 2:
            assignments.append((atom[0], "VALENCE", valence_atom_list))

    return assignments


# Function to output all CORE and VALENCE assignments to stdout
def output_assignments(assignments):
    assignments = sorted(assignments, key=lambda x: x[1], reverse=True)
    # Header line for listing out all attractors and their assignments
    print(f"Attractor    Core/Valence    Atoms    Indices")

    # Writing each assignment to file and to stdout
    for assignment in assignments:
        print(
            f"{assignment[0]:<13}{assignment[1]:<16}{','.join(atom[0] for atom in assignment[2]):<9}{','.join(str(atom_index[1]) for atom_index in assignment[2]):<7}"
        )


# Function to return VALENCE attractors in bohrs
# If interest_atoms is specified, only return VALENCE attractors of interest
def get_relevant_attractors_bohrs(assignments, attractors, interest_atoms):
    attractors_bohrs = []

    for i, assignment in enumerate(assignments):
        assigned_atom_indices = [atom_index[1] for atom_index in assignment[2]]
        if assignment[1] == "VALENCE" and len(interest_atoms) > 0:
            for index in assigned_atom_indices:
                if index in interest_atoms:
                    attractors_bohrs.append(
                        [
                            attractors[i][1][0] * 1.88973,
                            attractors[i][1][1] * 1.88973,
                            attractors[i][1][2] * 1.88973,
                        ]
                    )
                    break
        elif assignment[1] == "VALENCE":
            attractors_bohrs.append(
                [
                    attractors[i][1][0] * 1.88973,
                    attractors[i][1][1] * 1.88973,
                    attractors[i][1][2] * 1.88973,
                ]
            )

    return attractors_bohrs


# Function to append requested attractors to cube file for visualistion
def append_cube(cubefile, attractors_bohrs):
    cubename = cub_or_cube(cubefile)

    # Opening original cubefile
    contents = None
    with open(cubefile, "r") as original_cube:
        contents = original_cube.readlines()

    # Gettings number of atoms to know the position at which to start inserting attractors
    num_atoms = int(contents[2].split()[0])
    start_position = 6 + num_atoms

    # Replacing number of atoms to reflect added attractors
    new_num_atoms = num_atoms + len(attractors_bohrs)
    num_atom_line = re.split(f"(\s+)", contents[2])
    num_atom_line[2] = str(new_num_atoms)
    contents[2] = "".join(num_atom_line)

    # Loop through list of attractors, adding to original_cube_contents
    for attractor in attractors_bohrs:
        formatted_attractor = f"{'0':>5}{0.0:>12.6f}{attractor[0]:>12.6f}{attractor[1]:>12.6f}{attractor[2]:>12.6f}\n"
        contents.insert(start_position, formatted_attractor)
        start_position += 1

    # Create new cubefile, containing attractors
    with open(f"{cubename}_updated.cub", "w") as new_cube:
        new_cube.write("".join(contents))


# Main function of program
def auto_elf_assign(cubefile, attractorfile, interest_atoms=[]):
    # Get geometry from cube file
    geom, cube_basename = get_geom_from_cube(cubefile)

    # Get attractors from pdb file
    attractors = get_attractors(attractorfile)

    # Get distance matrix
    distance_matrix = calc_distances(geom, attractors)

    # Printing a message to indicate the start of assignment
    print("=" * 120)
    print(f"Starting assignment for {cube_basename}")
    print("=" * 120)

    # Get assignments of attractors to CORE and VALENCE
    assignments = assign(distance_matrix)

    # Get (relevant) attractors in units of bohrs
    attractors_bohrs = get_relevant_attractors_bohrs(
        assignments, attractors, interest_atoms
    )

    # Output assignments to stdout
    output_assignments(assignments)

    # Confirmation messages
    print("=" * 120)
    print(f"Success! Ending Assignment for {cube_basename}\n")

    # Begin process of editing cube file to contain (relevant) VALENCE attractors
    append_cube(cubefile, attractors_bohrs)
    print(
        f"{cube_basename}_updated.cub created, where (requested) valence attractors have been apppended to original cube file."
    )
    print("=" * 120)

    return cube_basename
