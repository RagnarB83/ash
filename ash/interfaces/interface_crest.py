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
#
# How it works: CREST's "generic" calculator runs an external binary for every energy+gradient
# call. Starting a full ASH process per call is far too slow (`from ash import *` alone is ~0.16 s),
# so ASH starts persistent server worker(s) that hold the unpickled Theory object, and CREST calls a
# tiny client (builtin _socket only, no ASH/numpy) that sends the geometry over a UNIX socket and
# writes the ORCA-style .engrad file that CREST reads.
# - one server worker per CREST thread (server_workers, default = numcores), each in its own directory
# - absolute paths everywhere (CREST runs the binary in calculation.level.1/ and in calculation.level.1/_N/)
# - per-job socket path, charge/mult taken from the job, server readiness checked before CREST starts
 
_ASH_CREST_SERVER = r'''# ASH server for CREST generic calculator (written by ash.interfaces.interface_crest.new_call_crest)
import os, sys, socket, pickle, signal
socket_path, theoryfile, nworkers, charge, mult = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
basedir = os.getcwd()
if os.path.exists(socket_path):
    os.remove(socket_path)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(socket_path + ".tmp")
sock.listen(64)
 
def serve(workerid):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # each worker gets its own directory so file-based theories (ORCA, xtb inputfile, ...) do not collide
    wdir = os.path.join(basedir, f"ash_server_worker_{workerid}")
    os.makedirs(wdir, exist_ok=True)
    os.chdir(wdir)
    sys.stdout = open("ash_server_worker.log", "w", buffering=1)
    sys.stderr = sys.stdout
    import ash  # noqa: F401  (needed to unpickle ASH theory objects)
    import numpy as np
    theory = pickle.load(open(theoryfile, "rb"))
    ncalls = 0
    while True:
        conn, _ = sock.accept()
        try:
            chunks = []
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            # request = the genericinp.xyz file CREST wrote (Angstrom)
            lines = b"".join(chunks).decode().splitlines()
            nat = int(lines[0])
            elems = [l.split()[0] for l in lines[2:2 + nat]]
            coords = np.array([l.split()[1:4] for l in lines[2:2 + nat]], dtype=float)
            # same call that ash.Singlepoint makes, without the Fragment construction and printing
            energy, gradient = theory.run(current_coords=coords, elems=elems, Grad=True, charge=charge, mult=mult)
            response = repr(float(energy)) + "\n" + "\n".join(repr(g) for g in np.asarray(gradient, dtype=float).ravel().tolist()) + "\n"
            ncalls += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            response = f"ERROR {type(e).__name__}: {e}"
        try:
            conn.sendall(response.encode())
        finally:
            conn.close()
 
children = []
def _stop(signum, frame):
    for pid in children:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    os._exit(0)
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)
for i in range(1, nworkers + 1):
    pid = os.fork()
    if pid == 0:
        serve(i)
        os._exit(0)
    children.append(pid)
# all workers inherit the listening socket; publish it only now (atomic rename = readiness signal)
os.rename(socket_path + ".tmp", socket_path)
for pid in children:
    os.waitpid(pid, 0)
'''
 
_ASH_CREST_CLIENT = r'''# minimal client for CREST generic calculator (written by ash.interfaces.interface_crest.new_call_crest)
# CREST starts this script once per energy+gradient call, so it must start fast: it imports only the
# builtin _socket module (no ASH, numpy, json or even socket.py) and is run with `python -I -S`.
import _socket
xyz = open("genericinp.xyz", "rb").read()
s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
s.connect(SOCKET_PATH)
s.sendall(xyz)
s.shutdown(1)  # SHUT_WR
chunks = []
while True:
    chunk = s.recv(65536)
    if not chunk:
        break
    chunks.append(chunk)
s.close()
reply = b"".join(chunks).decode()
if not reply or reply.startswith("ERROR"):
    raise SystemExit("ASH server error: " + reply)
nat = xyz.split(None, 1)[0].decode()
energy, grad = reply.split("\n", 1)
with open("genericinp.engrad", "w") as f:
    f.write("#\n# Number of atoms\n#\n" + nat + "\n#\n# The current total energy in Eh\n#\n" + energy
            + "\n#\n# The current gradient in Eh/bohr\n#\n" + grad)
'''
 
 
def new_call_crest(fragment=None, theory=None, crestdir=None, runtype="imtd-gc",
                   ewin=6.0, rthr=None, ethr=None, bthr=None,
                   shake=None, tstep=None, dump=None,length_ps=None,temp=None, hmass=None,
                   kpush=None, alpha=None, cvtype=None, dump_ps=None,
                   numcores=1, charge=None, mult=None,
                   topocheck=True, constraints=None, server_workers=None):
    """Run CREST (v3 TOML input) with either a GFN string ('gfn1','gfn2','gfnff') or any picklable ASH Theory.
 
    server_workers: number of ASH server processes answering CREST's energy/gradient requests
                    (default: numcores, i.e. one per CREST thread). Use 1 for theories that themselves
                    use many cores (e.g. ORCATheory with numcores>1) to avoid oversubscription.
    Note: generic-theory iMTD-GC needs a CREST build newer than 3.0.2 (3.0.2 terminates all production MTDs).
    """
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
 
    # What type of theory: valid crest-theory string (gfn1, gfn2, gfnff) or ASH Theory
    use_server = not isinstance(theory, str)
    serverproc = None
    socketdir = None
    if not use_server:
        print(f"Theory input is a string:{theory} Checking if valid")
        if theory.lower() in ['gfn1', 'gfn2', 'gfnff', 'gfn0']:
            print("Theory is a GFN method:", theory)
        else:
            print("Error: Invalid theory-keyword. Valid options are: gfn0, gfn1, gfn2, gfnff")
            ashexit()
        theorylines=f'method = "{theory.lower()}"\n'
    else:
        import pickle
        import tempfile
        print("A Theory object was passed. Serializing it for the ASH server.")
        theoryfile = os.path.abspath("theory.saved")
        with open(theoryfile, "wb") as f:
            pickle.dump(theory, f)
        if server_workers is None:
            server_workers = numcores
        # per-job socket in a short private directory (AF_UNIX paths are limited to ~100 characters)
        socketdir = tempfile.mkdtemp(prefix="ashcrest_")
        socket_path = os.path.join(socketdir, "ash.sock")
        with open("ash_crest_server.py", "w") as f:
            f.write(_ASH_CREST_SERVER)
        with open("ash_crest_client.py", "w") as f:
            f.write(f"SOCKET_PATH = {socket_path!r}\n" + _ASH_CREST_CLIENT)
        # absolute interpreter + script paths: CREST runs the binary from different subdirectories
        theorylines=f"""method = "generic"
binary = "{sys.executable} -I -S {os.path.abspath('ash_crest_client.py')}"
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
 
    try:
        if use_server:
            print(f"Launching ASH server in background ({server_workers} worker(s))...")
            serverlog = open("ash_server.log", "w")
            serverproc = sp.Popen([sys.executable, "ash_crest_server.py", socket_path, theoryfile,
                                   str(server_workers), str(charge), str(mult)], stdout=serverlog, stderr=serverlog, start_new_session=True,
                                  env=dict(os.environ, OMP_NUM_THREADS=str(getattr(theory, "numcores", 1) or 1)))
            t0 = time.time()
            while not os.path.exists(socket_path):
                if serverproc.poll() is not None:
                    print("ASH server exited before becoming ready. See ash_server.log")
                    ashexit()
                if time.time() - t0 > 120:
                    print("ASH server did not become ready within 120 s. See ash_server.log")
                    ashexit()
                time.sleep(0.05)
            print(f"ASH server ready after {time.time()-t0:.2f} s")
        print("Now calling CREST like this: crest --input input.toml")
        sp.run([crestdir + '/crest', '--input', 'input.toml'], check=True)
    finally:
        if serverproc is not None:
            print("Terminating ASH server...")
            # the server and its forked workers share a process group: stop them all
            import signal
            try:
                os.killpg(serverproc.pid, signal.SIGTERM)
                serverproc.wait(timeout=10)
            except sp.TimeoutExpired:
                os.killpg(serverproc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            serverlog.close()
        if socketdir is not None:
            shutil.rmtree(socketdir, ignore_errors=True)
 
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
 
