import subprocess as sp
import os
import shutil
import math
from ash.functions.functions_general import ashexit, BC,print_time_rel, print_line_with_mainheader,listdiff
from ash.modules.module_coords import writepdb_with_connectivity
import ash.settings_ash

# Basic packmol interface

def packmol_solvate(packmoldir=None, inputfiles=None, num_mols=None, tolerance=2.0, result_file="final", shape="box",
                    min_coordinates=[0.0, 0.0, 0.0], max_coordinates=[40.0, 40.0, 40.0],density=None):

    packmoldir=check_packmol_location(packmoldir)
    print("packmoldir:", packmoldir)
    print()
    if inputfiles is None :
        print("Error: inputfiles  input arguments are required!")
        exit()
    if num_mols is not None :
        if len(inputfiles) != len(num_mols):
            print("Error: Input-list variables inputfiles and num_moles must have the same size! ")
            ashexit()
    if density is None and num_mols is None:
        print("Error: Either density or num_mols input arguments is required!")
        ashexit()
    if density is not None and num_mols is not None:
        print("Error: Only one of density or num_mols input arguments is required!")
        ashexit()

    # If density is given, calculate num_mols
    if density is not None:
        print(f"A density of {density} g/ml was specified. Calculating number of molecules based on density and volume.")
        if len(inputfiles) != 1:
            print("Error: Only one input file is allowed when using density as input!")
            ashexit()
        # read file to get mass
        if inputfiles[0].endswith(".pdb"):
            frag = ash.Fragment(pdbfile=inputfiles[0], use_atomnames_as_elements=True,printlevel=0)
        elif inputfiles[0].endswith(".mol"):
            frag = ash.Fragment(xyzfile=inputfiles[0], printlevel=0)
        print("Molecule MW:", frag.mass, "g/mol")
        # Volumes from coordinates
        volume = (max_coordinates[0]-min_coordinates[0])*(max_coordinates[1]-min_coordinates[1])*(max_coordinates[2]-min_coordinates[2])
        print("Volume of box:", volume, "A^3")
        num_mol = density_to_num_mol(density, volume, frag.mass)
        print("Number of molecules:", num_mol)
        num_mols=[num_mol]
        print(num_mols)

    # Determining filetype
    filetype=os.path.splitext(inputfiles[0])[1].replace(".","")
    print("Filetype is:", filetype)

    # PRINT
    print("Inputfiles:", inputfiles)
    print("Num_mols:", num_mols)
    for f,m in zip(inputfiles,num_mols):
        print(f"{f} : {m} molecules")

    # WRITING INPUTFILE
    input_header=f"""
tolerance {tolerance}
filetype {filetype}
output {result_file}.{filetype}
"""

    input_main=""
    for struct_file,num_mol in zip(inputfiles,num_mols):
        temp=f"""
    structure {struct_file}
    number {num_mol}
    inside {shape} {min_coordinates[0]} {min_coordinates[1]} {min_coordinates[2]} {max_coordinates[0]} {max_coordinates[1]} {max_coordinates[2]}
    end structure
    """
        input_main= input_main+temp
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

    if inputfiles[0].endswith(".pdb"):
        print("\nWarning: Packmol-created PDB-files may not contain correct connectivity information!")
        print("Will now try to use ASH function writepdb_with_connectivity to write a new PDB-file with connectivity information.")
        writepdb_with_connectivity("final.pdb")
        result_filename=f"{result_file}_withcon.{filetype}"
        print("Created file:", result_filename)
        print("Creating also single-molecule file:")
        writepdb_with_connectivity(inputfiles[0])
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


#Convert desired density (g/ml) and desired volume (A^3) to number of a given molecule
def density_to_num_mol(density, volume, MW):
    Avo_N = 6.02214076E+23
    volume_ml = 1.00E-24*volume
    num_mol = math.ceil((density*volume_ml*Avo_N)/MW)
    return num_mol
