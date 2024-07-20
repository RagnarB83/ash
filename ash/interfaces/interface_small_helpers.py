import subprocess as sp
import os
import shutil

from ash.functions.functions_general import ashexit, BC,print_time_rel, \
        print_line_with_mainheader,listdiff, check_program_location
import ash.settings_ash

# Small helpers
#./qvSZP --struc nh3.xyz --chrg 0 --bfile /homelocal/rb269145/CALCDIR/qvSZP-test/qvSZP-2.1/q-vSZP_basis/basisq --efile --efile /homelocal/rb269145/CALCDIR/qvSZP-test/qvSZP-2.1/q-vSZP_basis/ecpq

def create_adaptive_minimal_basis_set(directory=None, fragment=None, xyzfile=None, bin_name="qvSZP",
                                      basisfile_path=None, ecpfile_path=None, charge=None):

    #Early exits
    if fragment is None and xyzfile is None:
        print(BC.FAIL, "No fragment or xyzfile provided. Exiting.", BC.END)
        ashexit()

    if fragment != None:
        print("Using ASH fragment. Writing XYZ-file to disk")
        xyzfile = fragment.write_xyzfile(xyzfilename="fragment.xyz")
        charge=fragment.charge #Setting charge from fragment
    # Checking if charge was set (by user or from fragment)
    if charge is None:
        print("Error: you must provide a charge for the molecule")
        ashexit()

    directory=check_program_location(directory,"directory",bin_name)

    if basisfile_path is None or ecpfile_path is None:
        print("Error: you must provide a basisfile_path and ecpfile_path (download from https://github.com/grimme-lab/qvSZP/)")
        ashexit()

    print("Running qvSZP. Will write output to qvSZP.out")
    with open('qvSZP.out', 'w') as ofile:

        args_list = [directory+f'/{bin_name}', '--struc', xyzfile, '--chrg', str(charge), '--bfile', basisfile_path, '--efile', ecpfile_path]
        print("args_list:", args_list)
        sp.run(args_list, stdout=ofile)
    # Read the ORCA inputfile created and grab the basis set lines
    basis_dict, ecp_dict = grab_basis_from_ORCAinputfile('wb97xd4-qvszp.inp')

    return basis_dict, ecp_dict


def grab_basis_from_ORCAinputfile(infile):
    grab=False
    basis_grab=False
    ecpgrab=False
    atomcounter=0
    atom_dict={}
    ecp_dict={}
    with open('wb97xd4-qvszp.inp') as o:
        for line in o:
            if ecpgrab:
                if 'NewECP' in line:
                    el = line.split()[1]
                    ecp_dict[el] = [line]
                elif 'end' in line:
                    ecp_dict[el].append(line)
                    ecpgrab=False
                else:
                    ecp_dict[el].append(line)
            if basis_grab:
                if len(line.split()) == 2 or len(line.split()) == 3:
                    atom_dict[(el,atomcounter)].append(line)
            if grab:
                if len(line.split())  == 4:
                    el = line.split()[0]
                    atom_dict[(el,atomcounter)]=[]
                if 'NewGTO' in line:
                    atom_dict[(el,atomcounter)].append(line)
                    basis_grab=True
                if 'end' in line:
                    atom_dict[(el,atomcounter)].append(line)
                    atomcounter+=1
            if 'xyz' in line and '*' in line:
                grab=True
            if '%basis' in line:
                ecpgrab=True
    return atom_dict,ecp_dict
