import os
import re
import shutil
import subprocess as sp
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel, print_line_with_mainheader
import ash.constants
import ash.settings_ash

# Interface to MOPAC (https://github.com/openmopac/mopac)
# Energy and gradient for the NDDO methods of MOPAC (PM7, PM6, PM6-D3H4, PM3, AM1, RM1, MNDO, ...)
# QM/MM: energy + gradient + pointcharge gradient using electrostatic embedding via the MOPAC QMMM keyword
#
# Energy: MOPAC's heat of formation, converted from kcal/mol to Eh. Only energy differences are meaningful.
#
# Electrostatic embedding:
# MOPAC does not read pointcharges. The QMMM keyword reads (file mol.in) one electrostatic potential per QM atom,
# phi_A = sum_J Q_J / |R_A - R_J|, which is added to the diagonal core-Hamiltonian elements of atom A and to its core.
# The QM-pointcharge energy is thus E = sum_A q_A * phi_A, where q_A is the MOPAC atomic charge of QM atom A, i.e.
# the pointcharges interact with the atomic charges of the QM atoms (no interaction with the atomic dipoles/quadrupoles
# of the NDDO density and no short-range damping).
# MOPAC's gradient is calculated at fixed phi. The missing terms are added here (the SCF energy is variational):
#   QM atom A:      - q_A * sum_J Q_J (R_A - R_J) / r_AJ^3
#   pointcharge J:  + Q_J * sum_A q_A (R_A - R_J) / r_AJ^3


class MOPACTheory:
    """MOPAC NDDO single-point theory with optional point-charge embedding.

    Energies are MOPAC heats of formation converted from kcal/mol to Eh;
    only energy differences are physically meaningful. Gradients are returned
    in Eh/Bohr, matching the ASH theory interface convention.
    """

    def __init__(self, mopacdir=None, filename='mopac', method=None, printlevel=2, label="MOPAC",
                 numcores=1, openshell='UHF', precise=True, scfcrt=None, keywords=None):

        self.theorynamelabel = "MOPAC"
        self.label = label
        self.theorytype = "QM"
        self.analytic_hessian = False
        self.printlevel = printlevel
        if self.printlevel > 0:
            print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        if method is None:
            print(f"{self.theorynamelabel}Theory requires a method keyword (e.g. method='PM7' or 'PM6-D3H4')")
            ashexit()
        if str(openshell).upper() not in ('UHF', 'ROHF'):
            print("Error: openshell keyword should be 'UHF' or 'ROHF'")
            ashexit()

        # Finding MOPAC
        if mopacdir is None:
            if self.printlevel > 0:
                print(BC.WARNING, f"No mopacdir argument passed to {self.theorynamelabel}Theory. Attempting to find mopacdir variable inside settings_ash", BC.END)
            try:
                self.mopacdir = ash.settings_ash.settings_dict["mopacdir"]
            except KeyError:
                if self.printlevel > 0:
                    print(BC.WARNING, "Found no mopacdir variable in settings_ash module either.", BC.END)
                mopac_executable = shutil.which('mopac')
                if mopac_executable is None:
                    print(BC.FAIL, "Found no mopac executable in PATH. Exiting...", BC.END)
                    print("See https://github.com/openmopac/mopac for MOPAC")
                    ashexit()
                self.mopacdir = os.path.dirname(mopac_executable)
                if self.printlevel > 0:
                    print(BC.OKGREEN, "Found mopac in PATH. Setting mopacdir to:", self.mopacdir, BC.END)
        else:
            self.mopacdir = mopacdir
        # mopacdir can be the directory containing the mopac executable or the installation directory (containing bin)
        for exe in (os.path.join(self.mopacdir, 'mopac'), os.path.join(self.mopacdir, 'bin', 'mopac')):
            if os.path.isfile(exe):
                self.mopac_executable = exe
                break
        else:
            print(BC.FAIL, f"Found no mopac executable in {self.mopacdir} or {os.path.join(self.mopacdir, 'bin')}. Exiting...", BC.END)
            ashexit()

        self.filename = filename
        self.method = method
        self.numcores = numcores
        # Open-shell treatment for mult > 1: 'UHF' or 'ROHF' (MOPAC's default RHF half-electron treatment)
        self.openshell = str(openshell).upper()
        # PRECISE keyword: tighter SCF and other criteria (recommended for gradients)
        self.precise = precise
        # SCFCRT keyword: SCF criterion in kcal/mol. None: MOPAC's own value
        self.scfcrt = scfcrt
        # Additional MOPAC keywords (string), added to the keyword line as given
        self.keywords = keywords

        # Results
        self.energy = None
        self.gradient = None
        self.pcgradient = None
        self.charges = None
        self.properties = {}

        if self.printlevel > 0:
            print("MOPAC executable:", self.mopac_executable)
            print("Method:", self.method)

    # Set numcores method
    def set_numcores(self, numcores):
        self.numcores = numcores

    def cleanup(self):
        if self.printlevel > 0:
            print(f"Cleaning up old {self.theorynamelabel} files")
        for f in [self.filename + ext for ext in ('.mop', '.out', '.aux', '.arc', '.log', '.end', '.den', '.res')]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):

        module_init_time = time.time()
        if numcores is None:
            numcores = self.numcores

        if self.printlevel > 0:
            print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        # Coords provided to run
        if current_coords is None:
            print("no current_coords")
            ashexit()
        current_coords = np.array(current_coords, dtype=float)
        if current_coords.ndim != 2 or current_coords.shape[1] != 3:
            print(BC.FAIL, "Error: current_coords must have shape (number of atoms, 3)", BC.END)
            ashexit()

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems
        qm_elems = list(qm_elems)
        if len(qm_elems) != len(current_coords):
            print(BC.FAIL, "Error: number of QM elements does not match number of coordinates", BC.END)
            ashexit()

        # Pointcharges: used if PC is True and pointcharges were provided
        use_pc = PC is True and MMcharges is not None and len(MMcharges) > 0
        phi = None
        if use_pc:
            if self.openshell == 'ROHF' and int(mult) > 1:
                print(BC.FAIL, "Error: pointcharge embedding with MOPACTheory requires openshell='UHF' for open-shell systems.", BC.END)
                print("The gradient correction for the pointcharges is only valid for a variational SCF energy.")
                ashexit()
            current_MM_coords = np.array(current_MM_coords, dtype=float)
            MMcharges = np.array(MMcharges, dtype=float)
            if current_MM_coords.shape != (len(MMcharges), 3):
                print(BC.FAIL, "Error: MM coordinates and point charges have incompatible dimensions", BC.END)
                ashexit()
            # Potential (kcal/mol per unit charge) at each QM atom
            phi = pointcharge_potential(current_coords, current_MM_coords, MMcharges)
        self.properties = {}

        if self.printlevel > 1:
            print("Job label:", label)
            print(f"Creating inputfile: {self.filename}.mop")
            print("MOPAC method:", self.method)

        # Write inputfile (and mol.in if pointcharges)
        mol_in_existed = use_pc and os.path.exists('mol.in')
        mol_in_backup = None
        if mol_in_existed:
            with open('mol.in', 'rb') as mol_in_file:
                mol_in_backup = mol_in_file.read()
        keywordline = write_mopac_input(self.method, self.filename, qm_elems, current_coords, charge, mult, Grad=Grad,
                                        phi=phi, numcores=numcores, openshell=self.openshell, precise=self.precise,
                                        scfcrt=self.scfcrt, keywords=self.keywords)
        if self.printlevel > 1:
            print("MOPAC keywords:", keywordline)
        print_time_rel(module_init_time, modulename='mopac prep-run', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)

        # Run MOPAC
        try:
            run_MOPAC(self.mopac_executable, self.filename)
        finally:
            # mol.in is a fixed-name MOPAC QMMM file. Restore any pre-existing
            # user file, or remove the temporary file created for this call.
            if use_pc:
                if mol_in_existed:
                    with open('mol.in', 'wb') as mol_in_file:
                        mol_in_file.write(mol_in_backup)
                else:
                    try:
                        os.remove('mol.in')
                    except FileNotFoundError:
                        pass
        print_time_rel(module_init_time, modulename='mopac run-done', moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)

        # Grab energy, gradient (at fixed potential if pointcharges) and atomic charges
        self.energy, mopac_gradient, self.charges, dipole = grab_energy_gradient_mopac(self.filename, len(qm_elems), Grad=Grad)
        self.properties["charges"] = self.charges
        if dipole is not None:
            self.properties["dipole"] = dipole
        if self.printlevel > 0:
            print(f"Single-point {self.theorynamelabel} energy:", self.energy)
            print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        if Grad is True:
            self.gradient = mopac_gradient
            if use_pc:
                # Adding the QM-pointcharge gradient terms missing in the MOPAC gradient
                qm_correction, self.pcgradient = pointcharge_gradient(current_coords, current_MM_coords, MMcharges, self.charges)
                self.gradient = self.gradient + qm_correction
            elif PC is True:
                self.pcgradient = np.zeros((0, 3))
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            if PC is True:
                return self.energy, self.gradient, self.pcgradient
            else:
                return self.energy, self.gradient
        # Returning energy without gradient
        else:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.energy


# Coulomb constant in kcal/mol * Angstrom / e^2
def coulomb_constant_kcal_ang():
    return ash.constants.harkcal * ash.constants.bohr2ang


# QM-pointcharge distances for a block of pointcharges: 1/r array of shape (numQM, numPC_block)
# Pointcharges of zero charge are harmless at any distance (e.g. the MM1 atom of a link-atom boundary)
def _inverse_distances(qmcoords, pccoords, pccharges):
    diff = qmcoords[:, None, :] - pccoords[None, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    nonzero = pccharges != 0.0
    if np.any(dist[:, nonzero] < 1e-6):
        print(BC.FAIL, "Error: a pointcharge is located on top of a QM atom.", BC.END)
        ashexit()
    with np.errstate(divide='ignore'):
        inv = 1.0 / dist
    inv[:, ~nonzero] = 0.0
    return diff, inv


# Electrostatic potential (kcal/mol per unit charge) of the pointcharges at each QM atom
def pointcharge_potential(qmcoords, pccoords, pccharges, blocksize=20000):
    phi = np.zeros(len(qmcoords))
    for i in range(0, len(pccharges), blocksize):
        diff, inv = _inverse_distances(qmcoords, pccoords[i:i + blocksize], pccharges[i:i + blocksize])
        phi += inv @ pccharges[i:i + blocksize]
    return coulomb_constant_kcal_ang() * phi


# Gradient (Eh/Bohr) of E = sum_A q_A * phi_A w.r.t. QM atoms and pointcharges at fixed atomic charges q_A
def pointcharge_gradient(qmcoords, pccoords, pccharges, qmcharges, blocksize=20000):
    qmgrad = np.zeros((len(qmcoords), 3))
    pcgrad = np.zeros((len(pccharges), 3))
    for i in range(0, len(pccharges), blocksize):
        diff, inv = _inverse_distances(qmcoords, pccoords[i:i + blocksize], pccharges[i:i + blocksize])
        # q_A * Q_J * (R_A - R_J) / r^3
        term = (qmcharges[:, None] * pccharges[None, i:i + blocksize] * inv**3)[:, :, None] * diff
        qmgrad -= term.sum(axis=1)
        pcgrad[i:i + blocksize] = term.sum(axis=0)
    # e^2/Angstrom^2 => Eh/Bohr
    factor = ash.constants.bohr2ang * ash.constants.bohr2ang
    return qmgrad * factor, pcgrad * factor


def write_mopac_input(method, filename, elems, coords, charge, mult, Grad=False, phi=None, numcores=1,
                      openshell='UHF', precise=True, scfcrt=None, keywords=None):
    multnames = {1: 'SINGLET', 2: 'DOUBLET', 3: 'TRIPLET', 4: 'QUARTET', 5: 'QUINTET', 6: 'SEXTET',
                 7: 'SEPTET', 8: 'OCTET', 9: 'NONET'}
    if int(mult) not in multnames:
        print(f"Error: multiplicity {mult} is not supported by the MOPAC interface")
        ashexit()
    # 1SCF: single-point. NOSYM: no symmetry. GEO-OK: no stop for short distances (e.g. link atoms)
    kw = [str(method), '1SCF', 'NOSYM', 'GEO-OK', f'CHARGE={int(charge)}', f'THREADS={int(numcores)}', 'AUX(PRECISION=9,MOS=-1,COMP)']
    if Grad is True:
        kw.append('GRAD')
    if int(mult) > 1:
        kw.append(multnames[int(mult)])
        if openshell == 'UHF':
            kw.append('UHF')
    if precise is True:
        kw.append('PRECISE')
    if scfcrt is not None:
        kw.append('SCFCRT=' + f'{float(scfcrt):.3E}'.replace('E', 'D'))
    if phi is not None:
        kw.append('QMMM')
    if keywords:
        kw.append(str(keywords).strip())
    keywordline = ' '.join(kw)
    # A keyword line can be continued on the next line with +
    lines = []
    current = ''
    for word in keywordline.split():
        if len(current) + len(word) + 1 > 70:
            lines.append(current + ' +')
            current = word
        else:
            current = (current + ' ' + word).strip()
    lines.append(current)
    if len(lines) > 3:
        print("Error: too many MOPAC keywords (max 3 lines)")
        ashexit()
    with open(f"{filename}.mop", 'w') as f:
        f.write('\n'.join(lines) + '\n')
        f.write('MOPAC input created by ASH\n\n')
        # Cartesian coordinates. Optimization flags 1 so that the gradient is printed for all coordinates
        for el, c in zip(elems, coords):
            f.write(f"{el:<2s} {c[0]:18.10f} 1 {c[1]:18.10f} 1 {c[2]:18.10f} 1\n")
    if phi is not None:
        # mol.in: line 1 is a comment line, line 2: number of QM atoms and number of link atoms
        # then one line per QM atom: element, coordinates (not used by MOPAC) and potential in kcal/mol
        with open('mol.in', 'w') as f:
            f.write(f"\n{len(elems)} 0\n")
            for el, c, p in zip(elems, coords, phi):
                f.write(f"{el} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f} {p:.10f}\n")
    return keywordline


def run_MOPAC(mopac_executable, filename):
    # Old output files are removed so that a failed run can not be mistaken for a finished one
    for ext in ('.out', '.aux', '.arc'):
        try:
            os.remove(filename + ext)
        except FileNotFoundError:
            pass
    env = dict(os.environ)
    # MOPAC shared library: lib directory next to the bin directory
    libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(mopac_executable))), 'lib')
    if os.path.isdir(libdir):
        env['LD_LIBRARY_PATH'] = libdir + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    try:
        process = sp.run([mopac_executable, f"{filename}.mop"], env=env, stdout=sp.PIPE,
                         stderr=sp.PIPE, text=True, check=False)
    except OSError as exc:
        print(BC.FAIL, f"Error: could not start MOPAC executable {mopac_executable}: {exc}", BC.END)
        ashexit()
    if process.returncode != 0:
        print(BC.FAIL, f"Error: MOPAC ended with exit code {process.returncode}. See {filename}.out", BC.END)
        if process.stderr:
            print(process.stderr[-2000:])
        _print_mopac_output_end(filename)
        ashexit()


def _print_mopac_output_end(filename, numlines=15):
    try:
        with open(f"{filename}.out", errors='replace') as f:
            lines = [l.rstrip() for l in f.readlines() if l.strip()]
        print(f"Last lines of {filename}.out:")
        for l in lines[-numlines:]:
            print("   ", l)
    except FileNotFoundError:
        print(f"MOPAC did not write {filename}.out")


# Read a numerical block from the MOPAC AUX file
def _read_aux_block(aux, key):
    match = re.search(re.escape(key) + r"(\[\d+\])?=\s*([-+0-9.DEde\s]+)", aux)
    if match is None:
        return None
    return np.array([float(v.replace('D', 'E').replace('d', 'E')) for v in match.group(2).split()])


def grab_energy_gradient_mopac(filename, numatoms, Grad=False):
    try:
        with open(f"{filename}.aux", errors='replace') as f:
            aux = f.read()
    except FileNotFoundError:
        print(BC.FAIL, f"Error: MOPAC did not write {filename}.aux", BC.END)
        _print_mopac_output_end(filename)
        ashexit()
    hof = _read_aux_block(aux, "HEAT_OF_FORMATION:KCAL/MOL")
    charges = _read_aux_block(aux, "ATOM_CHARGES")
    if hof is None or charges is None or len(charges) != numatoms:
        print(BC.FAIL, f"Error: MOPAC did not finish (no energy or atomic charges in {filename}.aux). SCF failure?", BC.END)
        _print_mopac_output_end(filename)
        ashexit()
    # SCF convergence is reported in the output file
    with open(f"{filename}.out", errors='replace') as f:
        out = f.read()
    if "SCF FIELD WAS ACHIEVED" not in out:
        print(BC.FAIL, f"Error: the MOPAC SCF did not converge. See {filename}.out", BC.END)
        _print_mopac_output_end(filename)
        ashexit()
    energy = hof[0] / ash.constants.harkcal
    dipole = _read_aux_block(aux, "DIP_VEC:DEBYE")
    gradient = None
    if Grad is True:
        grad = _read_aux_block(aux, "GRADIENTS:KCAL/MOL/ANGSTROM")
        if grad is None or len(grad) != 3 * numatoms:
            print(BC.FAIL, f"Error: no complete gradient found in {filename}.aux", BC.END)
            _print_mopac_output_end(filename)
            ashexit()
        # kcal/mol/Angstrom => Eh/Bohr
        gradient = grad.reshape(numatoms, 3) * ash.constants.bohr2ang / ash.constants.harkcal
    return energy, gradient, charges, dipole
