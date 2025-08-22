import subprocess as sp
import os
import shutil
import math
from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader,listdiff
from ash.modules.module_coords import writepdb_with_connectivity
import ash.settings_ash


# Basic packmol interface

def packmol_solvate(packmoldir=None,ligand_files=None, num_mols_ligands=None, solvent_files=None, solvents_ratio=None, tolerance=2.0, result_file="final", shape="box",
                    min_coordinates=[0.0, 0.0, 0.0], max_coordinates=[40.0, 40.0, 40.0],total_density=None, sphere_center=None, sphere_radius=None):

    ligands_mass= []
    solvents_mass= []
    # This list will store the final number of molecules for each solvent
    solvent_molecule_counts = []

    packmoldir=check_packmol_location(packmoldir)
    print("packmoldir:", packmoldir)
    print()

    # --- Input Validation ---
    if shape.lower() == "sphere":
        if sphere_center is None or sphere_radius is None:
            print("Error: For a 'sphere' shape, you must provide 'sphere_center' and 'sphere_radius'.")
            ashexit()
    elif shape.lower() != "box":
        print(f"Error: Unsupported shape '{shape}'. Please use 'box' or 'sphere'.")
        ashexit()

    if not ligand_files and not solvent_files:
        print("Error: No input files provided for either ligands or solvents. Exiting.")
        ashexit()

    if ligand_files is not None:
        print("Ligand files:", ligand_files)
    else:
        print("No ligand files provided, only solvent files will be packed.")
        num_mols_ligands = []

    if num_mols_ligands is None:
        print("Info: No number of molecules for ligands provided! Will use 1 molecule per ligand file!")
        num_mols_ligands = [1] * len(ligand_files) if ligand_files else []
    else:
        if ligand_files and (len(ligand_files) != len(num_mols_ligands)):
            print("Error: Input-list variables ligand_files and num_mols_ligands must have the same size! ")
            ashexit()

    if solvent_files is None:
        print("Info: No solvent provided! Only Ligand files will be packed!")
    else:
        if solvents_ratio and (len(solvent_files) != len(solvents_ratio)):
            print("Error: Input-list variables solvent_files and solvents_ratio must have the same size! ")
            ashexit()
        if total_density is None and solvents_ratio is None:
            print("Error: When providing solvent_files, you must specify either total_density or solvents_ratio (as molecule counts)!")
            ashexit()

    # --- Density Calculation ---
    if total_density is not None:
        print(f"A density of {total_density} g/ml was specified. Calculating number of molecules based on density and volume.")
        if ligand_files:
            print("Ligand files provided, will calculate mass of ligands.")
            for i,ligand_file in enumerate(ligand_files):
                if ligand_file.endswith((".pdb", ".mol2")):
                    frag = ash.Fragment(pdbfile=ligand_file, printlevel=0)
                    mw_lig = frag.mass
                    print(f"{ligand_file} MW:", mw_lig, "g/mol")
                elif ligand_file.endswith((".xyz")):
                    frag = ash.Fragment(xyzfile=ligand_file, printlevel=0)
                    mw_lig = frag.mass
                    pdb_ligand_file = ligand_file.rsplit('.', 1)[0] + ".pdb"
                    frag.write_pdbfile(pdb_ligand_file)
                    print(f"Converted {ligand_file} from xyz to pdb format.")
                    print(f"{pdb_ligand_file} MW:", mw_lig, "g/mol")
                    ligand_files[i] = pdb_ligand_file
                else:
                    print(f"Unsupported ligand file format: {ligand_file} || Supported formats: .pdb, .mol2,.xyz")
                    ashexit()
                ligands_mass.append(mw_lig)
        else:
            print("No ligand files provided, will not calculate mass of ligands.")
            ligands_mass = []

        if solvent_files:
            print("Solvent files provided, will calculate mass of solvents.")
            for i,solvent_file in enumerate(solvent_files):
                if solvent_file.endswith((".pdb", ".mol2")):
                    frag = ash.Fragment(pdbfile=solvent_file, printlevel=0)
                    mw_solvent = frag.mass
                    print(f"{solvent_file} MW:", mw_solvent, "g/mol")
                elif solvent_file.endswith((".xyz")):
                    frag = ash.Fragment(xyzfile=solvent_file, printlevel=0)
                    mw_solvent = frag.mass
                    pdb_solvent_file = solvent_file.rsplit('.', 1)[0] + ".pdb"
                    frag.write_pdbfile(pdb_solvent_file)
                    print(f"Converted {solvent_file} from xyz to pdb format.")
                    print(f"{pdb_solvent_file} MW:", mw_solvent, "g/mol")
                    solvent_files[i] = pdb_solvent_file
                else:
                    print(f"Unsupported solvent file format: {solvent_file} || Supported formats: .pdb, .mol2, .xyz")
                    ashexit()
                solvents_mass.append(mw_solvent)
        else:
            print("No solvent files provided, will not calculate mass of solvents.")
            solvents_mass = []
        
        volume = 0.0
        if shape.lower() == "box":
            volume = (max_coordinates[0]-min_coordinates[0])*(max_coordinates[1]-min_coordinates[1])*(max_coordinates[2]-min_coordinates[2])
            print("Volume of box:", volume, "A^3")
        elif shape.lower() == "sphere":
            volume = (4.0/3.0) * math.pi * (sphere_radius ** 3)
            print("Volume of sphere:", volume, "A^3")

        Avo_N = 6.02214076E+23
        volume_ml = 1.00E-24*volume
        print("Total Density Desired:", total_density, "g/ml")
        print("Will compute mass of ligands and solvents and calculate number of molecules for each solvent based on ratio and total density.")

        ligand_total_mass = sum([ligands_mass[i] * num_mols_ligands[i] for i in range(len(ligands_mass))])
        print("Total mass of ligands:", ligand_total_mass, "g/mol")
       
        if not solvent_files:
            solvent_total_mass = 1
        else:
            solvent_total_mass = sum([solvents_mass[i] * solvents_ratio[i] for i in range(len(solvents_mass))])

        if solvent_files:
            total_mass = total_density * volume_ml * Avo_N
            m = (total_mass - ligand_total_mass) / solvent_total_mass if solvent_total_mass > 0 else 0
            print("Scaling factor 'm' for solvent ratios:", m)
            solvent_molecule_counts = [int(round(m * r)) for r in solvents_ratio]
            for i in range(len(solvents_mass)):
                print(f"Number of molecules for {solvent_files[i]}: {solvent_molecule_counts[i]}")
        else:
            total_mass = ligand_total_mass
            print("No solvents provided, will not pack solvents.")
        print("Total mass of system:", total_mass, "g/mol")

    elif solvent_files:
        solvent_molecule_counts = [int(n) for n in solvents_ratio]

    # --- Filetype Determination ---
    if solvent_files:
        filetype=os.path.splitext(solvent_files[0])[1].replace(".","")
    elif ligand_files:
        filetype=os.path.splitext(ligand_files[0])[1].replace(".","")
    else:
        print("Error: No files to determine filetype from.")
        ashexit()
    print("Filetype is:", filetype)

    # --- Packing Summary ---
    print("\n--- Packing Summary ---")
    if ligand_files:
        print("Ligands to pack:")
        for f,m in zip(ligand_files,num_mols_ligands):
            print(f"  - {f} : {m} molecule(s)")
    else:
        print("No ligands to pack.")

    if solvent_files:
        print("Solvents to pack:")
        for f,m in zip(solvent_files,solvent_molecule_counts):
            print(f"  - {f} : {m} molecule(s)")
    else:
        print("No solvents to pack.")
    print("-----------------------\n")
    


    input_header=f"""
tolerance {tolerance}
filetype {filetype}
output {result_file}.{filetype}
"""
    # reusable command string for the shape
    shape_command = ""
    if shape.lower() == "box":
        shape_command = f"inside box {min_coordinates[0]} {min_coordinates[1]} {min_coordinates[2]} {max_coordinates[0]} {max_coordinates[1]} {max_coordinates[2]}"
    elif shape.lower() == "sphere":
        shape_command = f"inside sphere {sphere_center[0]} {sphere_center[1]} {sphere_center[2]} {sphere_radius}"

    ligands_input = ""
    solvents_input = ""

    if ligand_files:
        if len(ligand_files) == 1 and num_mols_ligands[0] == 1:
            print(f"Only one ligand present, fixing its position to the center of the {shape}.")
            center_x, center_y, center_z = (0, 0, 0)
            if shape.lower() == 'box':
                center_x = (max_coordinates[0] + min_coordinates[0]) / 2
                center_y = (max_coordinates[1] + min_coordinates[1]) / 2
                center_z = (max_coordinates[2] + min_coordinates[2]) / 2
            elif shape.lower() == 'sphere':
                center_x, center_y, center_z = sphere_center
            
            ligands_input = f"""
structure {ligand_files[0]}
  number 1
  fixed {center_x} {center_y} {center_z} 0. 0. 0.
end structure
"""
        else:
            # Corrected f-string and use of shape_command
            print(f"Multiple ligands present, packing them {shape_command}.")
            for struct_file, num_mol in zip(ligand_files, num_mols_ligands):
                ligands_input += f"""
structure {struct_file}
  number {num_mol}
  {shape_command}
end structure
"""
    
    if solvent_files:
        print(f"Packing solvents {shape_command}.")
        for struct_file,num_mol in zip(solvent_files,solvent_molecule_counts):
            # Use the generic shape_command here as well
            solvents_input +=f"""
structure {struct_file}
  number {num_mol}
  {shape_command}
end structure
"""
    
    input_main = ligands_input + solvents_input
    with open("inputfile.inp", "w") as f:
        f.write(input_header)
        f.write(input_main)

    # --- Running Packmol ---
    with open('inputfile.inp') as input_file, open("packmol.out", "w") as ofile:
        sp.run([f"{packmoldir}/packmol"], env=os.environ, check=True, stdin=input_file, stdout=ofile, stderr=ofile)

    result_filename=f"{result_file}.{filetype}"
    print("\nPackmol-interface finished successfully")
    print("Packmol output can be found in packmol.out")
    print(f"Created file: {result_filename}")

    if solvent_files and result_filename.endswith(".pdb"):
        print("\nWarning: Packmol-created PDB-files may not contain correct connectivity information!")
        print("Will now try to use ASH function writepdb_with_connectivity to write a new PDB-file with connectivity information.")
        writepdb_with_connectivity("final.pdb")
        result_filename=f"{result_file}_withcon.{filetype}"
        print("Created file:", result_filename)
        print("Creating also single-molecule file:")
        writepdb_with_connectivity(solvent_files[0])
    return result_filename

def check_packmol_location(packmoldir, binary_name="packmol", dirname="packmoldir"):
    if packmoldir != None:
        finaldir = packmoldir
        print(BC.OKGREEN,f"Using {dirname} path provided: {finaldir}", BC.END)
    else:
        print(BC.WARNING, f"No {dirname} argument passed. Attempting to find {dirname} variable in ASH settings file (~/ash_user_settings.ini)", BC.END)
        try:
            finaldir=ash.settings_ash.settings_dict[dirname]
            print(BC.OKGREEN,f"Using {dirname} path provided from ASH settings file (~/ash_user_settings.ini): ", finaldir, BC.END)
        except (KeyError, AttributeError):
            print(BC.WARNING,f"Found no {dirname} variable in ASH settings file either.",BC.END)
            print(BC.WARNING,f"Checking for {binary_name} in PATH environment variable.",BC.END)
            try:
                finaldir = os.path.dirname(shutil.which(binary_name))
                if finaldir:
                    print(BC.OKGREEN,f"Found {binary_name} binary in PATH. Using the following directory:", finaldir, BC.END)
                else:
                    raise TypeError
            except TypeError:
                print(BC.FAIL,f"Found no {binary_name} binary in PATH environment variable either. Giving up.", BC.END)
                print("Download latest packmol release from: https://github.com/m3g/packmol/releases")
                print("Extract archive, cd packmol dir and make (requires Fortran compiler)")
                print("For problems: see installation instructions: https://m3g.github.io/packmol/userguide.shtml")
                ashexit()
    return finaldir
