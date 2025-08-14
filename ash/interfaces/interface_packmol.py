import subprocess as sp
import os
import shutil
from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader,listdiff
from ash.modules.module_coords import writepdb_with_connectivity
import ash.settings_ash


# Basic packmol interface

def packmol_solvate(packmoldir=None,ligand_files=None, num_mols_ligands=None, solvent_files=None, solvents_ratio=None, tolerance=2.0, result_file="final", shape="box",
                    min_coordinates=[0.0, 0.0, 0.0], max_coordinates=[40.0, 40.0, 40.0],total_density=None):

    ligands_mass= []
    solvents_mass= []
    packmoldir=check_packmol_location(packmoldir)
    print("packmoldir:", packmoldir)
    print()
    if ligand_files is not None:
        print("Ligand files:", ligand_files)
    else:
        print("No ligand files provided, only solvent files will be packed.")
        num_mols_ligands = None

    if num_mols_ligands is None:
        print("Info: No number of molecules for ligands provided! Will use 1 molecule per ligand file!")
        num_mols_ligands = [1] * len(ligand_files) if ligand_files else []
    else:
        if len(ligand_files) != len(num_mols_ligands):
            print("Error: Input-list variables ligand_files and num_mols_ligands must have the same size! ")
            ashexit()

    if solvent_files is None :
        print("Info: No solvent provided! Only Ligand files would be packed!")
        #print("Error: inputfiles  input arguments are required!")
        #exit()
        solvents_ratio = None

    if solvents_ratio is not None :
        if len(solvent_files) != len(solvents_ratio):
            print("Error: Input-list variables inputfiles and num_moles must have the same size! ")
            ashexit()
    if total_density is None and solvents_ratio is None:
        print("Error: Either density or num_mols input arguments is required!")
        ashexit()


    # If density is given, calculate num_mols
    if total_density is not None:
        print(f"A density of {total_density} g/ml was specified. Calculating number of molecules based on density and volume.")
        #if len(solvent_files) != 1:
        #    print("Error: Only one input file is allowed when using density as input!")
        #    ashexit()
        # read file to get mass
        if ligand_files != None:
            print("Ligand files provided, will calculate mass of ligands and solvents to determine number of molecules for each solvent.")
            for i,ligand_file in enumerate(ligand_files):
                if ligand_file.endswith((".pdb", ".mol2")):
                    frag = ash.Fragment(pdbfile=ligand_file, printlevel=0)
                    mw_lig = frag.mass
                    print(f"{ligand_file} MW:", mw_lig, "g/mol")
                elif ligand_file.endswith((".xyz")):
                    frag = ash.Fragment(xyzfile=ligand_file, printlevel=0)
                    mw_lig = frag.mass
                    ligand_file = ligand_file.rsplit('.', 1)[0] + ".pdb"  # remove .xyz, add .pdb
                    frag.write_pdbfile(ligand_file)
                    print(f"Converted {ligand_file} from xyz to pdb format.")
                    print(f"{ligand_file} MW:", mw_lig, "g/mol")
                    ligand_files[i] = ligand_file  # update the ligand file name
                else:
                    print(f"Unsupported ligand file format: {ligand_file} || Supported formats: .pdb, .mol2,.xyz")
                    ashexit()
                ligands_mass.append(mw_lig)
        else:
            print("No ligand files provided, will not calculate mass of ligands.")
            ligands_mass = []


        if solvent_files != None:
            print("Solvent files provided, will calculate mass of solvents to determine number of molecules for each solvent.")
            for i,solvent_file in enumerate(solvent_files):
                if solvent_file.endswith((".pdb", ".mol2")):
                    frag = ash.Fragment(pdbfile=solvent_file, printlevel=0)
                    mw_solvent = frag.mass
                    print(f"{solvent_file} MW:", mw_solvent, "g/mol")
                elif solvent_file.endswith((".xyz")):
                    frag = ash.Fragment(xyzfile=solvent_file, printlevel=0)
                    mw_solvent = frag.mass
                    solvent_file = solvent_file.rsplit('.', 1)[0] + ".pdb"  # remove .xyz, add .pdb
                    frag.write_pdbfile(solvent_file)
                    print(f"Converted {solvent_file} from xyz to pdb format.")
                    print(f"{solvent_file} MW:", mw_solvent, "g/mol")
                    solvent_files[i] = solvent_file  # update the solvent file name
                else:
                    print(f"Unsupported solvent file format: {solvent_file} || Supported formats: .pdb, .mol2, .xyz")
                    ashexit()
                solvents_mass.append(mw_solvent)
        else:
            print("No solvent files provided, will not calculate mass of solvents.")
            solvents_mass = []
        
        # Volumes from coordinates
        volume = (max_coordinates[0]-min_coordinates[0])*(max_coordinates[1]-min_coordinates[1])*(max_coordinates[2]-min_coordinates[2])
        #print("Volume of box:", volume, "A^3")
        #num_mol = density_to_num_mol(total_density, volume, frag.mass)
        #print("Number of molecules:", num_mol)
        #solvents_ratio=[num_mol]
        #print(solvents_ratio)
        Avo_N = 6.02214076E+23
        volume_ml = 1.00E-24*volume
        #num_mol = math.ceil((density*volume_ml*Avo_N)/MW)
        print("Total Desity Desired:", total_density, "g/ml")
        print("Volume of box:", volume, "A^3")
        print("Will compute mass of ligands and solvents and calculate number of molecules for each solvent based on ratio and total density.")

        ligand_total_mass = sum([ligands_mass[i] * num_mols_ligands[i] for i in range(len(ligands_mass))])
        print("Total mass of ligands:", ligand_total_mass, "g/mol")
       
        solvent_total_mass = sum([solvents_mass[i] * solvents_ratio[i] for i in range(len(solvents_mass))])

        total_mass = total_density * volume_ml * Avo_N  # total mass in grams
        print("Total mass of system:", total_mass, "g/mol")
        # Solve for m:
        m = (total_mass - ligand_total_mass) / solvent_total_mass
        print("Number of molecules of each solvent to achieve desired density (for ratio 1):", m)
        # Number of molecules of each solvent
        solvent_molecule_counts = [int(round(m * solvents_ratio[j] )) for j in range(len(solvents_mass))]
        print(f"Number of molecules for {solvent_files[i]}: {m} * {solvents_ratio[i]}" for i in range(len(solvents_mass)))



    # Determining filetype
    filetype=os.path.splitext(solvent_files[0])[1].replace(".","")
    print("Filetype is:", filetype)

    # PRINT
    print("Ligand Inputfiles:", ligand_files)
    print("Num_mols Ligands:", num_mols_ligands)
    if ligand_files != None:
        for f,m in zip(ligand_files,num_mols_ligands):
            print(f"{f} : {m} molecules")
    else:
        print("No ligands provided, will not pack ligands.")

    print("Solvent Inputfiles:", solvent_files)
    print("Solvent Ratio:", solvents_ratio)
    if solvent_files != None:
        for f,m in zip(solvent_files,solvent_molecule_counts):
            print(f"{f} : {m} molecules")
    else:
        print("No solvents provided, will not pack solvents.")
    

    # WRITING INPUTFILE
    input_header=f"""
tolerance {tolerance}
filetype {filetype}
output {result_file}.{filetype}
"""

    input_main=""
    ligands_input = ""
    solvents_input = ""
    if ligand_files != None:
        if len(ligand_files) == 1 and num_mols_ligands[0] == 1:
            print("Only one ligand present, will fix it's position in the center of the box.")
            min_coordinates_single = [(max_coordinates[i] + min_coordinates[i]) / 2 for i in range(3)]
            max_coordinates_single = [(max_coordinates[i] + min_coordinates[i]) / 2 for i in range(3)]
            for struct_file, num_mol in zip(ligand_files, num_mols_ligands):
                ligands_input = f"""
            structure {struct_file}
            number {num_mol}
            fixed {min_coordinates_single[0]} {min_coordinates_single[1]} {min_coordinates_single[2]} {max_coordinates_single[0]} {max_coordinates_single[1]} {max_coordinates_single[2]}
            end structure
            """
        else:
            print("Multiple ligands present, will use the provided box coordinates for packing.")
            for struct_file, num_mol in zip(ligand_files, num_mols_ligands):
                ligands_input += f"""
            structure {struct_file}
            number {num_mol}
            inside {shape} {min_coordinates[0]} {min_coordinates[1]} {min_coordinates[2]} {max_coordinates[0]} {max_coordinates[1]} {max_coordinates[2]}
            end structure
            """
    else:
        print("No ligands provided, will not pack ligands.")
        ligands_input = ""
        
    for struct_file,num_mol in zip(solvent_files,solvent_molecule_counts):
        solvents_input +=f"""
    structure {struct_file}
    number {num_mol}
    inside {shape} {min_coordinates[0]} {min_coordinates[1]} {min_coordinates[2]} {max_coordinates[0]} {max_coordinates[1]} {max_coordinates[2]}
    end structure
    """
        input_main= ligands_input + solvents_input
    with open("inputfile.inp", "w") as f:
        f.write(input_header)
        f.write(input_main)

    input_file = open('inputfile.inp')

    # sp.run(args_list, stdout=ofile)
    ofile = open("packmol.out","w")
    sp.run([f"{packmoldir}/packmol"], env=os.environ, check=True, stdin=input_file, stdout=ofile, stderr=ofile)
    input_file.close()
    ofile.close()

    result_filename=f"{result_file}.{filetype}"
    print("\nPackmol-interface finished successfully")
    print("Packmol output can be found in packmol.out")
    print(f"Created file: {result_filename}")

    if solvent_files[0].endswith(".pdb"):
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
        except KeyError:
            print(BC.WARNING,f"Found no {dirname} variable in ASH settings file either.",BC.END)
            print(BC.WARNING,f"Checking for {binary_name} in PATH environment variable.",BC.END)
            try:
                finaldir = os.path.dirname(shutil.which(binary_name))
                print(BC.OKGREEN,f"Found {binary_name} binary in PATH. Using the following directory:", finaldir, BC.END)
            except TypeError:
                print(BC.FAIL,f"Found no {binary_name} binary in PATH environment variable either. Giving up.", BC.END)
                print("Download latest packmol release from: https://github.com/m3g/packmol/releases")
                print("Extract archive, cd packmol dir and make (requires Fortran compiler)")
                print("For problems: see installation instructions: https://m3g.github.io/packmol/userguide.shtml")
                ashexit()
    return finaldir
