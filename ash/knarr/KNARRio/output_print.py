import numpy as np


# Author: Vilhjalmur Asgeirsson, 2019

def PrintConfiguration(header, ndim, ndof, rxyz, constr, symb, cell=[0.0, 0.0, 0.0], pbc=False):
    if header is not None:
        print('%s' % header)
    print('Number of dimensions                  : %5li' % ndim)
    print('Number of degrees of freedom          : %5li' % ndof)
    print('Number of inactive degrees of freedom : %5li' % (ndim - ndof))
    if pbc:
        if cell is not None:
            print('Cell dimensions: %6.2f %6.2f %6.2f' % (cell[0], cell[1], cell[2]))
    if constr is not None and symb is not None and rxyz is not None:
        for i in range(0, ndim, 3):
            print('% 2ls % 12.8lf % 12.8lf % 12.8lf % 2li % 2li % 2li' % (
                symb[i], rxyz[i], rxyz[i + 1], rxyz[i + 2], constr[i], constr[i + 1], constr[i + 2]))
    else:
        raise RuntimeError("Are you sure you know what you are printing?")
    return None


def PrintConfigurationPath(header, ndim, ndimIm, nim, ndof, rxyz, constr, symb,
                           cell=[0.0, 0.0, 0.0], pbc=False):
    if header is not None:
        print('%s' % header)
    print('Number of images                      : %5li' % nim)
    print('Number of dimensions (path)           : %5li' % ndim)
    print('Number of dimensions (image)          : %5li' % ndimIm)
    print('Number of degrees of freedom          : %5li' % ndof)
    print('Number of inactive degrees of freedom : %5li' % (ndim - ndof))
    if pbc:
        if cell is not None:
            print('Cell dimensions: %6.2f %6.2f %6.2f' % (cell[0], cell[1], cell[2]))
    if constr is not None and symb is not None and rxyz is not None:
        for k in range(nim):
            print('Image: %3li' % k)
            for i in range(0, ndimIm, 3):
                z = k * ndimIm + i
                print('% 2ls % 12.8lf % 12.8lf % 12.8lf % 2li % 2li % 2li' % (
                    symb[z], rxyz[z], rxyz[z + 1], rxyz[z + 2], constr[z], constr[z + 1], constr[z + 2]))
    else:
        raise RuntimeError("Are you sure you know what you are printing?")

    return None


def PrintIntegerListAs3D(head_string, ndim, x):
    if x is not None:
        print('%s' % head_string)
        for i in range(0, ndim, 3):
            print('% 4li % 4li % 4li' % (x[i + 0], x[i + 1], x[i + 2]))

    return None


def PrintFloatListAs3D(head_string, ndim, x):
    if x is not None:
        print('%s' % head_string)
        for i in range(0, ndim, 3):
            print('% 8.4lf % 8.4lf % 8.4lf' % (x[i + 0], x[i + 1], x[i + 2]))

    return None


def PrintAtomMatrix(header, ndim, x, symb):
    if header is not None:
        print('%s' % header)
    for i in range(0, ndim, 3):
        print('%2s % 6.4lf % 6.4lf % 6.4lf' % (symb[i], x[i], x[i + 1], x[i + 2]))
    print('')
    return None


def PrintNEBLogFile(fname, ndim, nim, it, forces, freal_perp, ksp, fsp_paral, fsp_perp, fneb):
    with open(fname, 'a') as f:
        f.write(' Iteration: %i\n' % it)
        f.write('F         = ')
        listi = []
        for i in range(nim):
            listi.append(np.max(abs(forces[i * ndim:(i + 1) * ndim])))
        f.write('%6.4lf ' * len(listi) % tuple(listi))
        f.write('\n')

        f.write('Fperp      = ')
        listi = []
        for i in range(nim):
            listi.append(np.max(abs(freal_perp[i * ndim:(i + 1) * ndim])))
        f.write('%6.4lf ' * len(listi) % tuple(listi))
        f.write('\n')

        f.write('Ksp.       = ')
        listi = []
        for i in range(nim - 1):
            listi.append(ksp[i])
        f.write('%6.4lf ' * len(listi) % tuple(listi))
        f.write('\n')

        f.write('Fsp_paral = ')
        listi = []
        for i in range(nim):
            listi.append(np.max(abs(fsp_paral[i * ndim:(i + 1) * ndim])))
        f.write('%6.4lf ' * len(listi) % tuple(listi))
        f.write('\n')

        f.write('Fsp_perp  = ')
        listi = []
        for i in range(nim):
            listi.append(np.max(abs(fsp_perp[i * ndim:(i + 1) * ndim])))
        f.write('%6.4lf ' * len(listi) % tuple(listi))
        f.write('\n')

        f.write('FNEB     = ')
        listi = []
        for i in range(nim):
            listi.append(np.max(abs(fneb[i * ndim:(i + 1) * ndim])))
        f.write('%6.4lf ' * len(listi) % tuple(listi))
        f.write('\n')

    return None
