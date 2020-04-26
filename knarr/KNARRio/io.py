import numpy as np
import os


# Author: Vilhjalmur Asgeirsson, 2019.

def ReadFileToList(fname):
    filelist = []
    with open(fname) as f:
        f = f.readlines()
        for line in f:
            filelist.append(line)
    return filelist


def ReadFirstLineOfFile(fname):
    with open(fname) as f:
        first_line = f.readline()
    return first_line


def ReadConstraintsFile(fname):
    import numpy as np

    constraints = []
    with open(fname) as f:
        lines = f.readlines()
        natoms = int(lines[0])
        ndim = natoms * 3
        for i in range(2, len(lines)):
            line = lines[i].split()
            if len(line) == 1:

                try:
                    val = int(line[0])
                except:
                    raise ValueError("Constraints values need to be given in integer numbers")

                constraints.append(val)
                constraints.append(val)
                constraints.append(val)

            elif len(line) == 3:

                try:
                    val1 = int(line[0])
                    val2 = int(line[1])
                    val3 = int(line[2])
                except:
                    raise ValueError("Constraints values need to be given in integer numbers")

                constraints.append(val1)
                constraints.append(val2)
                constraints.append(val3)

            elif len(line) == 4:

                try:
                    val1 = int(line[1])
                    val2 = int(line[2])
                    val3 = int(line[3])
                except:
                    raise ValueError("Constraints values need to be given in integer numbers")

                constraints.append(val1)
                constraints.append(val2)
                constraints.append(val3)
            else:
                raise IOError("Incorrect format of %s", fname)

    #print(len(constraints), ndim)
    if len(constraints) != ndim:
        raise ValueError("Dimension mismatch")

    return np.reshape(constraints, (ndim, 1))


def ReadCellFile(fname):
    line = ReadFirstLineOfFile(fname)
    line = line.split()
    if len(line) != 3:
        raise ValueError("Wrong format of file %s" % fname)

    try:
        cellx = float(line[0])
        celly = float(line[1])
        cellz = float(line[2])
    except:
        raise ValueError("Cell requires floating point arguments")

    return [cellx, celly, cellz]


def ReadXYZ(fname):
    import numpy as np
    import os

    # Check if .xyz file exists
    if not os.path.isfile(fname):
        raise IOError("Input file %s not found" % fname)

    # Read coordinates and symbols
    rxyz = []
    symbols = []
    with open(fname) as f:
        lines = f.readlines()
        natoms = int(lines[0])
        ndim = natoms * 3
        for i in range(2, len(lines)):
            line = lines[i].split()
            symbols.append(line[0])
            symbols.append(line[0])
            symbols.append(line[0])
            rxyz.append(float(line[1]))
            rxyz.append(float(line[2]))
            rxyz.append(float(line[3]))

    return np.reshape(rxyz, (ndim, 1)), ndim, symbols


def WriteXYZ(fname, ndim, rxyz, symb, energy=None):
    if len(rxyz) != ndim:
        raise RuntimeError("Dimension mismatch in configuration")

    with open(fname, "w") as f:
        f.write(str(ndim / 3) + '\n')
        if energy is not None:
            f.write('E=%12.8lf\n' % energy)
        else:
            f.write('\n')

        for j in range(0, ndim, 3):
            f.write('%s %12.8f %12.8f %12.8f\n' % (symb[j], rxyz[j + 0], rxyz[j + 1], rxyz[j + 2]))
    return None


def WriteXYZF(fname, ndim, rxyz, symb, energy=None, fxyz=None):
    if len(fxyz) != ndim:
        raise RuntimeError("Dimension mismatch in forces")
    if len(rxyz) != ndim:
        raise RuntimeError("Dimension mismatch in configuration")
    
    with open(fname, "w") as f:
        f.write(str(ndim / 3) + '\n')
        if energy is not None:
            f.write('E=%12.8lf\n' % energy)
        else:
            f.write('\n')
        if fxyz is None:
            for j in range(0, ndim, 3):
                f.write('%s %12.8f %12.8f %12.8f\n' % (symb[j], rxyz[j + 0], rxyz[j + 1], rxyz[j + 2]))
        else:
            for j in range(0, ndim, 3):
                f.write('%s %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n' % (symb[j], rxyz[j + 0], rxyz[j + 1], rxyz[j + 2],
                                                       fxyz[j + 0], fxyz[j + 1], fxyz[j + 2]))

    return None


def ReadXYZF(fname):
    import numpy as np
    import os

    """
     Reads and returns .xyz (fname) file with energy and forces.
     Second line contains the energy

     "N
     E=0.0
     symb1 x1 y1 z1 fx1 fx2 fx3
     ...
     symbN xN yN zN fxN fyN fzN"

     return rxyz, fxyz, energy, ndim, symbols

    """

    # Check if .xyz file exists
    if not os.path.isfile(fname):
        raise IOError("Input file %s not found" % fname)

    # Read coordinates, forces, energy and symbols
    rxyz = []
    fxyz = []
    energy = 0.0
    symbols = []
    with open(fname) as f:
        lines = f.readlines()
        natoms = int(lines[0])
        if 'E' in lines[1]:
            try:
                energy = float(lines.split("=")[1])
            except:
                raise RuntimeError("Unable to read energy from %s", fname)

        ndim = natoms * 3
        for i in range(2, len(lines)):
            line = lines[i].split()
            if len(line) == 7:
                symbols.append(line[0])
                symbols.append(line[0])
                symbols.append(line[0])
                rxyz.append(float(line[1]))
                rxyz.append(float(line[2]))
                rxyz.append(float(line[3]))
                fxyz.append(float(line[4]))
                fxyz.append(float(line[5]))
                fxyz.append(float(line[6]))
            else:
                raise IOError("Unable to read forces from %s", fname)

    return np.reshape(rxyz, (ndim, 1)), np.reshape(fxyz, (ndim, 1)), energy, ndim, symbols


def ReadCon(fname):
    import os
    import numpy as np
    cell = []
    symbols = []
    rxyz = []
    constr = []

    if not os.path.isfile(fname):
        raise ValueError("Input file %s not found." % fname)

    natms = []
    matms = []
    with open(fname) as f:
        lines = f.readlines()
        ctmp = lines[2].split()
        for i in range(0, len(ctmp)):
            cell.append(float(ctmp[i]))
        ctmp = lines[5].split()
        Ncnstr, Nuncnstr, nimg = int(ctmp[0]), int(ctmp[1]), int(ctmp[2])
        if nimg > 1:
            raise IOError("Unable to read .con trajectory files")

        Ncmp = int(lines[6])
        ctmp = lines[7].split()
        for i in range(0, len(ctmp)):
            natms.append(int(ctmp[i]))
        ctmp = lines[8].split()
        for i in range(0, len(ctmp)):
            matms.append(float(ctmp[i]))
        ind = 8
        for i in range(0, Ncmp):
            ind += 1
            tmpsymb = lines[ind]
            tmpsymb = tmpsymb[:-1]
            ind += 1
            for j in range(0, natms[i]):
                ind += 1
                line = lines[ind].split()
                rxyz.append(float(line[0]))
                rxyz.append(float(line[1]))
                rxyz.append(float(line[2]))
                if len(line) == 5:
                    for k in range(0, 3):
                        constr.append(int(line[3]))
                elif len(line) == 7:
                    constr.append(int(line[3]))
                    constr.append(int(line[4]))
                    constr.append(int(line[5]))
                symbols.append(tmpsymb.strip())
                symbols.append(tmpsymb.strip())
                symbols.append(tmpsymb.strip())
    ndim = len(rxyz)

    return np.reshape(rxyz, (ndim, 1)), ndim, symbols, np.reshape(cell, (3, 1)), np.reshape(constr, (ndim, 1))


def WriteCon(fname, nim, R, symb, cell, constr):
    from KNARRatom.utilities import GetMasses
    usymb, ucount = np.unique(symb, return_counts=True)
    athl = len(constr)
    if athl == 0:
        constr = np.zeros(shape=(len(symb), 1))  # keep everything unconstrained if not specified otherwise

    with open(fname, "w") as f:
        f.write(' 0 Random Number Seed\n')
        f.write(' 0 Time\n')
        f.write(' %4.16f %4.16f %4.16f\n' % (cell[0], cell[1], cell[2]))
        f.write(' %4.16f %4.16f %4.16f\n' % (90.0, 90.0, 90.0))
        f.write(' 0 0\n')
        f.write(' %d %d %d \n' % (np.sum(constr) / 3, (len(symb) - np.sum(constr)) / 3, nim))
        f.write(' %d \n' % len(usymb))
        f.write(' %d ' * len(usymb) % (tuple(ucount / 3)))
        f.write('\n')
        amass = GetMasses(2, usymb)
        f.write(' %8.4f' * len(usymb) % (tuple(amass)))
        f.write('\n')
        ind = 0
        for i in range(0, len(usymb)):
            f.write(' ' + usymb[i] + '\n')
            f.write(' Components of type ' + str(i + 1) + '\n')
            for j in range(0, len(symb), 3):
                if usymb[i] == symb[j]:
                    f.write(' %12.6f %12.6f %12.6f %d %d %d %d\n' % (
                        R[j], R[j + 1], R[j + 2], constr[j], constr[j + 1], constr[j + 2], ind))
                    ind += 1
    return None


def WritePath(fname, ndimIm, nim, rxyz, symb, energy=None):
    with open(fname, 'w') as f:
        for i in range(nim):
            f.write('%i \n' % (ndimIm / 3))
            if energy is None:
                f.write('\n')
            else:
                f.write('% 8.6f \n' % energy[i])

            for j in range(0, ndimIm, 3):
                z = i * ndimIm + j
                f.write('%2s % 12.8f % 12.8f % 12.8f\n' % (symb[z], rxyz[z], rxyz[z + 1], rxyz[z + 2]))
    return None


def WriteTraj(fname, ndimIm, nim, rxyz, symb, energy=None):
    with open(fname, 'a') as f:
        for i in range(nim):
            f.write('%i \n' % (ndimIm / 3))
            if energy is None:
                f.write('\n')
            else:
                f.write('%8.6lf \n' % energy[i])

            for j in range(0, ndimIm, 3):
                z = i * ndimIm + j
                f.write('%2s % 12.8f % 12.8f % 12.8f\n' % (symb[z], rxyz[z], rxyz[z + 1], rxyz[z + 2]))
    return None


def WriteSingleImageTraj(fname, ndim, rxyz, symb, E):
    with open(fname, 'a') as f:
        f.write(str(ndim / 3) + '\n')
        f.write('%8.6lf \n' % E)
        for j in range(0, ndim, 3):
            f.write('%2s % 12.8lf % 12.8lf % 12.8lf\n' % (symb[j], rxyz[j + 0], rxyz[j + 1], rxyz[j + 2]))
    return None


def ReadTraj(fname):
    if not os.path.isfile(fname):
        raise IOError("File %s not found " % fname)

    extension = fname.split('.')[-1]
    if extension.strip().upper() != 'XYZ':
        raise IOError('Only .xyz trajectories can be read by KNARR')

    first_line = ReadFirstLineOfFile(fname)
    assert int(first_line)
    natoms = int(first_line)
    ndim = natoms * 3

    # begin by reading the contents of the file to a list
    contents = []
    f = open(fname).readlines()
    for i, line in enumerate(f):
        contents.append(line)

    # get number of lines and hence number of images
    number_of_lines = i + 1
    nim = int((number_of_lines) / (natoms + 2))

    rp = []
    ind = 0
    for i in range(nim):
        symb = []
        ind = ind + 2
        for j in range(natoms):
            geom_line = contents[ind]
            geom_line = geom_line.split()
            symb.append(geom_line[0].strip())
            symb.append(geom_line[0].strip())
            symb.append(geom_line[0].strip())
            rp.append(float(geom_line[1]))
            rp.append(float(geom_line[2]))
            rp.append(float(geom_line[3]))
            ind += 1

    rp = np.reshape(rp, (nim * ndim, 1))

    return rp, ndim, nim, symb


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


def WriteForcesFile(fname, ndim, symb, rxyz):
    with open(fname, 'w') as f:
        for i in range(0, ndim, 3):
            f.write("%2ls % 12.8lf % 12.8lf % 12.8lf \n" % (symb[i], rxyz[i + 0], rxyz[i + 1], rxyz[i + 2]))
    return None


def WriteEnergyFile(fname, energy, nim=None):
    with open(fname, 'w') as f:
        if nim is None:
            f.write('%12.8lf' % energy)
        else:
            assert len(energy) == nim
            for i in range(nim):
                f.write('%12.8lf \n' % energy[i])

    return None


def ReadModeFromFile(fname):
    if not os.path.isfile(fname):
        raise IOError("Unable to find file: %s" % fname)
    listi = []
    with open(fname) as f:
        f = f.readlines()
        for i, line in enumerate(f):
            try:
                listi.append(float(line))
            except:
                raise IOError("Wrong format of mode file: %f " % fname)

    return np.reshape(listi, (len(listi), 1))


def WriteModeToFile(fname, w):
    ndim = len(w)
    f = open(fname, 'w')
    for i in range(ndim):
        f.write('%12.8f\n' % (w[i]))
    f.close()
    return None

def ReadTwoDeeOptimization(fname):
    listi = []
    with open(fname) as f:
        for i,line in enumerate(f):
            if 'H' in line:
                listi.append(line.split())
    pos = np.zeros(shape=(len(listi),2))
    for i, val in enumerate(listi):
        pos[i,0]=float(val[1])
        pos[i,1]=float(val[2])

    return pos
