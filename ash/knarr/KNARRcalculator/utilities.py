import numpy as np
from KNARRatom.utilities import DMIC


# Author: Vilhjalmur Asgeirsson, 2019.

def ExecuteCalculatorParallel(G, ncore, args):
    import multiprocessing as mp
    pool = mp.Pool(processes=ncore)
    results = pool.map(G, args)
    pool.close()
    pool.join()

    return results


def GetProgramPath(program):
    path = WhereProgram(program)
    if path is None:
        print('**Warning: unable to find %s' % program)
        if program != program.upper():
            print('**Warning: extending search to %s' % program.upper())
            path = WhereProgram(program.upper())

        if path is None:
            if program != program.lower():
                print('**Warning: extending search to %s' % program.lower())
                path = WhereProgram(program.lower())

    if path is None:
        raise RuntimeError("Unable to find program %s. Do you have it in system path?" % program)

    return path


def WhereProgram(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def ReadEONTemplateFile(fname):

    template = []
    found_pot = False
    with open(fname) as f:
        for i,line in enumerate(f):
            if 'potential' in line.lower() and '=' in line.lower():
                found_pot = True
                template = line.split('=')[1]
                
    if not found_pot:
        raise IOError("Bad EON template file")

    return template

def ReadORCATemplateFile(fname):
    start_read = False
    template = []
    end_found = False
    begin_found = False

    with open(fname) as f:
        for i, line in enumerate(f):

            if '&ORCA_TEMPLATE_END' in line.upper().strip():
                start_read = False
                end_found = True

            if start_read and line != '\n':
                template.append(line)

            if '&ORCA_TEMPLATE' in line.upper().strip():
                start_read = True
                begin_found = True

    if not end_found or not begin_found:
        raise IOError("ORCA template markers not found in %s" % fname)

    return template


def CheckORCATemplateFile(template):
    jobtypes = ('ENGRAD', 'TIGHTOPT', 'LOOSEOPT', 'OPT', 'NEB-CI', 'NEB-TS',
                'FREQ', 'NUMFREQ', 'TSOPT', 'NEB'
                , 'COPT', 'ZOPT', 'GDIIS-COPT', 'GDIIS-ZOPT',
                'GDIIS-OPT', 'Numgrad', 'MD', 'IRC', 'VERYTIGHTOPT')
    for i, line in enumerate(template):
        line = line.upper()
        for k in line.split():
            for z in jobtypes:
                if k.strip() == z:
                    raise IOError("Please remove the jobtype from the ORCA template file")

        if '*XYZ' in line or '* XYZ' in line:
            raise IOError("Please remove the geometry block from the ORCA template file")
    return


def LinearInterpolationMatrix(ndim, npts, xi, xf):
    dnpts = float(npts)
    # RB. int addition. Correct??
    natoms = int(ndim / 3)
    dkappa = np.zeros(shape=(natoms, natoms, npts))

    for i in range(natoms):
        for j in range(natoms):
            dx = (xf[i, j] - xi[i, j])
            dx /= (dnpts - 1.0)
            for ipts in range(npts):
                dkappa[i, j, ipts] = xi[i, j] + ipts * dx

    return dkappa


def GetAllConfigDistances(ndim, rxyz,
                          pbc=False, cell=None):
    natoms = int(ndim / 3)
    rcurr_dist = np.zeros(shape=(natoms, natoms))
    rcurr_dx = np.zeros(shape=(natoms, natoms))
    rcurr_dy = np.zeros(shape=(natoms, natoms))
    rcurr_dz = np.zeros(shape=(natoms, natoms))

    for i in range(0, ndim, 3):
        #RB. int addition. Correct??
        atom0 = int(i / 3)
        x0 = rxyz[i]
        y0 = rxyz[i + 1]
        z0 = rxyz[i + 2]
        r0 = np.array([x0, y0, z0])
        for j in range(0, ndim, 3):
            # RB. int addition. Correct??
            atom1 = int(j / 3)
            x1 = rxyz[j]
            y1 = rxyz[j + 1]
            z1 = rxyz[j + 2]
            r1 = np.array([x1, y1, z1])

            dr = r1 - r0
            dr = DMIC(3, dr, pbc, cell)

            dist = np.sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2])
            rcurr_dist[atom0, atom1] = dist
            rcurr_dx[atom0, atom1] = dr[0]
            rcurr_dy[atom0, atom1] = dr[1]
            rcurr_dz[atom0, atom1] = dr[2]

    return rcurr_dist, rcurr_dx, rcurr_dy, rcurr_dz


def NumForce(ndim, nim, rxyz, h, F=None):
    assert ndim * nim == len(rxyz)
    assert h < 0.01 and h > 0.0
    assert F is not None

    forces = np.zeros(shape=(ndim * nim, 1))
    for i in range(nim):
        rr = rxyz[i * ndim:(i + 1) * ndim].copy()
        rl = rxyz[i * ndim:(i + 1) * ndim].copy()
        for j in range(ndim):
            rr[j] += h
            fr = F(rr)
            rl[j] -= h
            fl = F(rl)
            forces[i * ndim + j] = - (fr - fl) / (2.0 * h)
    return forces

def NumHessDOF(ndim, ndof, func, rxyz, rxyz_dof,
               free_index, fd_step=0.0001):

    def updconf(Rfull, Rfree, index):
        Rupd = Rfull.copy()
        Rupd[index] = Rfree
        return Rupd

    x0 = rxyz_dof
    H = np.zeros(shape=(ndof, ndof))
    for i in range(ndof):
        x1 = x0.copy()
        x1[i] = x0[i] - fd_step
        x1 = updconf(rxyz, x1, free_index)
        x1 = np.reshape(x1, (ndim, 1))
        F1, E1 = func(x1)
        F1 = F1[free_index]

        x2 = x0.copy()
        x2[i] = x0[i] + fd_step
        x2 = updconf(rxyz, x2, free_index)
        x2 = np.reshape(x2, (ndim, 1))
        F2, E2 = func(x2)
        F2 = F2[free_index]

        tmp = -(F2 - F1) / (2.0 * fd_step)  # its negative because I use the forces!
        H[i, :] = tmp.T

    return H
