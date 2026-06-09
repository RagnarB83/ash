import os
import time
import sys
import shutil
import subprocess as sp

from ash.modules.module_coords import split_multimolxyzfile
from ash.functions.functions_general import ashexit, BC, int_ranges, listdiff, print_line_with_subheader1,print_time_rel,pygrep, check_program_location
from ash.modules.module_coords import check_charge_mult, Fragment
import ash.settings_ash

# New crest interface that supports ASH levels of theory (Limitation: must be picklable)
def new_call_crest(fragment=None, theory=None, crestdir=None, runtype="imtd-gc", 
                   ewin=6.0, rthr=None, ethr=None, bthr=None,
                   shake=None, tstep=None, dump=None,length_ps=None,temp=None, hmass=None,
                   kpush=None, alpha=None, cvtype=None, dump_ps=None,
                   numcores=1, charge=None, mult=None, 
                   topocheck=True, constraints=None):
    module_init_time=time.time()

    if fragment is None or theory is None:
        print("new_call_crest requires a valid ASH fragment and valid ASH Theory object")
        ashexit()
    if charge is None or mult is None:
        print("Charge and multiplicity not provided to new_call_crest")
        print("Checking if defined inside fragment")
        if fragment.charge is not None:
            charge=fragment.charge
            mult=fragment.mult
        else:
            print("Found not charge defined in fragment either. Exiting ")
            ashexit()

    crestdir=check_program_location(crestdir,'crestdir','crest')

    # Write initial XYZ-file
    fragment.write_xyzfile(xyzfilename="struc.xyz")

    #####################
    # Parsing options
    #####################

    #Constraints file
    constraints_line=""
    if constraints is not None:
        print("constraints were found:", constraints)
        print("Writing constraints file: constraints.inp")
        # constraints ={'atoms':'1-26', 'metadyn_atoms':}
        constraintsfile="constraints.inp"
        with open(constraintsfile, 'w') as f:
            f.write(f"$constrain\n")
            if 'atoms' in constraints:
                f.write(f'  atoms: {constraints["atoms"]}\n')
            if 'elements' in constraints:
                f.write(f'  elements: {constraints["elements"]}\n')
            if 'bond' in constraints:
                f.write(f'  bond:{constraints["bond"]}\n')
            if 'distance' in constraints:
                f.write(f'  distance:{constraints["distance"]}\n')
            if 'angle' in constraints:
                f.write(f'  angle:{constraints["angle"]}\n')
            if 'dihedral' in constraints:
                f.write(f'  dihedral:{constraints["dihedral"]}\n')
            if 'force' in constraints:
                f.write(f'  force constant={constraints["force"]}\n')
            if 'reference' in constraints:
                f.write(f'  reference={constraints["reference"]}\n')
            if 'metadyn_atoms' in constraints:
                f.write(f"$metadyn\n")
                f.write(f'  atoms: {constraints["metadyn_atoms"]}\n')
            f.write(f"$end\n")
        constraints_line=f'constraints="{constraintsfile}"'

    #Dynamics options
    dynamics_options=[]
    if shake is not None:
        dynamics_options.append(f"shake={shake}")
    if tstep is not None:
        dynamics_options.append(f"tstep={tstep}")
    if dump is not None:
        dynamics_options.append(f"dump={dump}")
    if length_ps is not None:
        dynamics_options.append(f"length_ps={length_ps}")
    if temp is not None:
        dynamics_options.append(f"temp={temp}")
    if hmass is not None:
        dynamics_options.append(f"hmass={hmass}")
    dynamics_lines = "\n".join(dynamics_options)

    #MTD options
    mtd_options=[]
    if kpush is not None:
        mtd_options.append(f"kpush={kpush}")
    if alpha is not None:
        mtd_options.append(f"alpha={alpha}")
    if cvtype is not None:
        mtd_options.append(f"cvtype={cvtype}")
    if dump_ps is not None:
        mtd_options.append(f"dump_ps={dump_ps}")
    mtd_lines = "\n".join(mtd_options)


    #CREGEN options
    cregen_options=[]
    if ewin is not None:
        cregen_options.append(f"ewin={ewin}")
    if rthr is not None:
        cregen_options.append(f"rthr={rthr}")
    if ethr is not None:
        cregen_options.append(f"ethr={ethr}")
    if bthr is not None:
        cregen_options.append(f"bthr={bthr}")
    cregen_lines = "\n".join(cregen_options)

    # What type of theory.
    # Can be valid crest-theory string (gfn1, gfn2, gfnff) or ASH Theory

    if isinstance(theory, str):
        print(f"Theory input is a string:{theory} Checking if valid")
        if 'gfn' in theory.lower():
            print("Theory is a GFN method:", theory)
        else:
            print("Error: Invalid theory-keyword. Valid options are: gfn1, gfn2, gfnff")
            ashexit()
        theorylines=f"""method = "{theory.lower()}"
        """
    else:
        print("A Theory object was passed.")
        print("Now serializing.")
        # Pickle for serializing theory object
        import pickle
        # Serialize theory object for later use
        theoryfilename="theory.saved"
        pickle.dump(theory, open(theoryfilename, "wb" ))

        # Write ASH inputfile: ash_input.py
        #ashinput=f"""from ash import *
#from ash.interfaces.interface_ORCA import print_gradient_in_ORCAformat
#import pickle
#
#frag = Fragment(xyzfile="genericinp.xyz", charge={charge},mult={mult})
#Unpickling theory object
#theory = pickle.load(open(\"../{theoryfilename}\", \"rb\" ))
#result = Singlepoint(theory=theory, fragment=frag, Grad=True)
#print_gradient_in_ORCAformat(result.energy,result.gradient,"genericinp", extrabasename="")
#"""
    ash_server_code = """# energy_server.py

from ash import *
import pickle
import socket
import json
import numpy as np

print("Loading theory object...")

theory = pickle.load(open("./theory.saved", "rb"))

print("Theory loaded.")

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

socket_path = "/tmp/ash_energy.sock"

import os
if os.path.exists(socket_path):
    os.remove(socket_path)

sock.bind(socket_path)
sock.listen()

print("Server ready")

while True:

    conn, _ = sock.accept()

    try:
        data = b""

        while True:
            chunk = conn.recv(4096)

            if not chunk:
                break

            data += chunk

        request = json.loads(data.decode())

        coords = np.array(request["coords"])

        elements = request["elements"]

        frag = Fragment(
            elems=elements,
            coords=coords,
            charge=0,
            mult=1
        )

        result = Singlepoint(
            theory=theory,
            fragment=frag,
            Grad=True
        )

        response = {
            "energy": float(result.energy),
            "gradient": result.gradient.tolist()
        }

        conn.sendall(json.dumps(response).encode())

    except Exception as e:

        response = {
            "error": str(e)
        }

        conn.sendall(json.dumps(response).encode())

    finally:
        conn.close()
    """

    ash_client_code = """# energy_client.py
from ash import *
from ash.interfaces.interface_ORCA import print_gradient_in_ORCAformat

import socket
import json

frag = Fragment(
    xyzfile="genericinp.xyz",
    charge=0,
    mult=1
)

request = {
    "elements": frag.elems,
    "coords": frag.coords.tolist()
}

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

sock.connect("/tmp/ash_energy.sock")

sock.sendall(json.dumps(request).encode())

sock.shutdown(socket.SHUT_WR)

data = b""

while True:

    chunk = sock.recv(4096)

    if not chunk:
        break

    data += chunk

sock.close()

response = json.loads(data.decode())

if "error" in response:
    raise RuntimeError(response["error"])

energy = response["energy"]
gradient = response["gradient"]

print_gradient_in_ORCAformat(
    energy,
    gradient,
    "genericinp",
    extrabasename=""
)
"""

    # Write server code
    with open("ash_server.py", "w") as f:
        f.write(ash_server_code)

    # Write client code
    with open("ash_client.py", "w") as f:
        f.write(ash_client_code)

    # Write CREST toml file
    # Note: crest created dirs caleld calculation.level.X etc. and enters them
    theorylines=f"""method = "generic"
binary = "python3 ../ash_client.py"
gradfile = "genericinp.engrad"
gradtype = "engrad"
"""
    tomlinput=f"""# CREST 3 input file
input = "struc.xyz"
runtype="{runtype}"
threads = {numcores}
{constraints_line}
topo = {str(topocheck).lower()}
[cregen]
{cregen_lines}

[calculation]
elog="energies.log"

[dynamics]
{dynamics_lines}

[[dynamics.meta]]
{mtd_lines}

[[calculation.level]]
{theorylines}
uhf = {mult-1}
chrg = {charge}"""
    with open("input.toml", "w") as f:
        f.write(tomlinput)

    print("CREST run-type:", runtype)

    if runtype == "imtd-gc":
        print(f"Note:Energy window is {ewin} kcal/mol")

    # Launching ASH server in background via subprocess
    print("Launching ASH server in background...")
    process1 = sp.Popen(["python3", "ash_server.py"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    print(process1)
    #exit()
    print("Now calling CREST like this: crest --input input.toml")
    process = sp.run([crestdir + '/crest', '--input', 'input.toml'])

    print_time_rel(module_init_time, modulename='crest run', moduleindex=0)

    # Get conformers
    try:
        if runtype == "imtd-gc":
            list_conformers, list_energies = get_crest_conformers(charge=charge, mult=mult)
            return list_conformers, list_energies
        else:
            return None, None
    except:
        return None, None



#Very simple crest interface
def call_crest(fragment=None, xtbmethod=None, crestdir=None, charge=None, mult=None, solvent=None, energywindow=6, numcores=1,
               constrained_atoms=None, forceconstant_constraint=0.5, extraoptions=None):
    print_line_with_subheader1("call_crest")
    module_init_time=time.time()
    if crestdir == None:
        print(BC.WARNING, "No crestdir argument passed to call_crest. Attempting to find crestdir variable inside settings_ash", BC.END)
        try:
            print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
            crestdir=ash.settings_ash.settings_dict["crestdir"]
        except:
            print(BC.WARNING,"Found no crestdir variable in settings_ash module either.",BC.END)
            try:
                crestdir = os.path.dirname(shutil.which('crest'))
                print(BC.OKGREEN,"Found crest in path. Setting crestdir to:", crestdir, BC.END)
            except:
                print("Found no crest executable in path. Exiting... ")
                ashexit()

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, "QM", fragment, "call_crest", theory=None)

    try:
        shutil.rmtree('crest-calc')
    except:
        pass
    os.mkdir('crest-calc')
    os.chdir('crest-calc')

    if constrained_atoms != None:
        allatoms=range(0,fragment.numatoms)
        unconstrained=listdiff(allatoms,constrained_atoms)

        constrained_crest=[i+1 for i in constrained_atoms]
        unconstrained_crest=[j+1 for j in unconstrained]

        #Get ranges. List of tuples
        constrained_ranges=int_ranges(constrained_crest)
        unconstrained_ranges=int_ranges(unconstrained_crest)


        print("Creating .xcontrol file for constraints")
        with open(".xcontrol","w") as constrainfile:
            constrainfile.write("$constrain\n")
            #constrainfile.write("atoms: {}\n".format(','.join(map(str, constrained_ranges))))
            constrainfile.write("atoms: {}\n".format(constrained_ranges))
            constrainfile.write("force constant={}\n".format(forceconstant_constraint))
            constrainfile.write("$metadyn\n")
            constrainfile.write("atoms: {}\n".format(unconstrained_ranges ))
            constrainfile.write("$end\n")

    #Create XYZ file from fragment (for generality)
    fragment.write_xyzfile(xyzfilename="initial.xyz")
    #Theory level
    if 'GFN2' in xtbmethod.upper():
        xtbflag=2
    elif 'GFN1' in xtbmethod.upper():
        xtbflag=1
    elif 'GFN0' in xtbmethod.upper():
        xtbflag=0
    else:
        print("Using default GFN2-xTB")
        xtbflag=2


    #GBSA solvation or not
    if solvent is None:
        solventstring=""
    else:
        solventstring=f'-alpb {solvent}'
    #Extra options or empty. string with extra crest keyword flags
    if extraoptions == None:
        extraoptions=""

    #Run
    print("Now calling CREST")
    process = sp.run([crestdir + '/crest', 'initial.xyz','-T', str(numcores),  '-gfn' + str(xtbflag),
                      '-ewin', str(energywindow),  str(charge), solventstring, extraoptions,
                    '-chrg', str(charge), '-uhf', str(mult - 1)])


    os.chdir('..')
    print_time_rel(module_init_time, modulename='crest run', moduleindex=0)

    #Get conformers
    list_conformers, list_xtb_energies = get_crest_conformers(charge=charge, mult=mult)


    return list_conformers, list_xtb_energies





#Very simple crest interface for entropy calculations
def call_crest_entropy(fragment=None, crestdir=None, charge=None, mult=None, numcores=1):
    print_line_with_subheader1("call_crest")
    module_init_time=time.time()
    if crestdir == None:
        print(BC.WARNING, "No crestdir argument passed to call_crest. Attempting to find crestdir variable inside settings_ash", BC.END)
        try:
            print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
            crestdir=ash.settings_ash.settings_dict["crestdir"]
        except:
            print(BC.WARNING,"Found no crestdir variable in settings_ash module either.",BC.END)
            try:
                crestdir = os.path.dirname(shutil.which('crest'))
                print(BC.OKGREEN,"Found crest in path. Setting crestdir to:", crestdir, BC.END)
            except:
                print("Found no crest executable in path. Exiting... ")
                ashexit()

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, "QM", fragment, "call_crest", theory=None)

    try:
        shutil.rmtree('crest-calc')
    except:
        pass
    os.mkdir('crest-calc')
    os.chdir('crest-calc')


    #Create XYZ file from fragment (for generality)
    fragment.write_xyzfile(xyzfilename="initial.xyz")


    print("Running crest with entropy option")
    outputfile="crest-ash.out"
    logfile = open(outputfile, 'w')
    #Using POpen and piping like this we can write to stdout and a logfile
    process = sp.Popen([crestdir + '/crest', 'initial.xyz', '--entropy', '-T', str(numcores), '-chrg', str(charge), '-uhf', str(mult-1)],
        stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True)
    for line in process.stdout:
        sys.stdout.write(line)
        logfile.write(line)
    process.wait()
    logfile.close()
    #TODO: Grab stuff from output
    Sconf = pygrep("   Sconf   =", outputfile)[-1]
    dSrrho = pygrep(" + δSrrho  =", outputfile)[-1]
    Stot = pygrep(" = S(total)  =", outputfile)[-1]
    H_T_0_corr = pygrep("   H(T)-H(0) =", outputfile)[-1]
    G_tot = pygrep(" = G(total)  =", outputfile)[-2]
    entropydict={'Sconf':Sconf,'dSrrho':dSrrho,'Stot':Stot,'H_T_0_corr':H_T_0_corr,'G_tot':G_tot}

    print("Stot:", Stot)
    print("entropydict:", entropydict)
    os.chdir('..')
    print_time_rel(module_init_time, modulename='crest run', moduleindex=0)




    return entropydict






#Grabbing crest conformers. Goes inside rest-calc dir and finds file called crest_conformers.xyz
#Creating ASH fragments for each conformer
def get_crest_conformers(crest_calcdir='crest-calc',conf_file="crest_conformers.xyz", charge=None, mult=None):
    print("")
    print("Now finding Crest conformers and creating ASH fragments...")
    os.chdir(crest_calcdir)
    list_conformers=[]
    list_xtb_energies=[]
    all_elems, all_coords, all_titles = split_multimolxyzfile(conf_file,writexyz=True,return_fragments=False)
    print("Found {} Crest conformers".format(len(all_elems)))

    #Getting energies from title lines
    for i in all_titles:
        en=float(i[0])
        list_xtb_energies.append(en)

    for (els,cs,eny) in zip(all_elems,all_coords,list_xtb_energies):
        conf = Fragment(elems=els, coords=cs, charge=charge, mult=mult, printlevel=0)
        list_conformers.append(conf)
        conf.energy=eny

    os.chdir('..')
    print("")
    return list_conformers, list_xtb_energies
