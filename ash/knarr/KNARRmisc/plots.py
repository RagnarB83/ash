import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
except:
    raise ImportError("Unable to import matplotlib")


# Author: Vilhjalmur Asgeirsson, 2019.


def PlotOptimizationProfile(fname='orca.interp', start_at=0, end_at=-1, one_frame=True):
    def read_spline_file(fname):
        with open(fname) as input_data:
            listi = []
            images = []
            spline = []
            for i, line in enumerate(input_data):
                if line == '\n':
                    listi.append('E')
                else:
                    line = line.split()
                    listi.append(line)
                    if line[0] == 'Interp.:':
                        spline.append(i)
                    if line[0] == 'Images:':
                        images.append(i)
        return listi, images, spline

    def read_images(start_point, listi):
        arclength = []
        energy = []
        index = start_point
        while True:
            if listi[index][0].strip().upper() == 'INTERP.:' or listi[index][0].strip().upper() == 'E':
                break
            arclength.append(float(listi[index][1]))
            energy.append(float(listi[index][2]))
            index += 1
        return arclength, energy

    def read_spline(start_point, listi):
        arclength = []
        energy = []
        index = start_point
        while True:
            if listi[index][0].strip().upper() == 'ITERATION:' or listi[index][0].strip().upper() == 'E':
                break
            arclength.append(float(listi[index][1]))
            energy.append(float(listi[index][2]))
            index += 1
            if index == len(listi):
                break
        return arclength, energy

    def convert_bool(val):
        if val.upper() == 'FALSE':
            val = False
        elif val.upper() == 'TRUE':
            val = True
        return val

    # ============================================
    # Print header
    # ============================================
    print('==========================================')
    print('     Generation of ')
    print('             Optimization Profile   ')
    print('==========================================')
    print(' ')
    print('=> plotting from iteration %i to %i' % (start_at, end_at))

    # - - - - - - - - - - - - - - - - - - - - - - -
    # Let the plotting begin...
    # - - - - - - - - - - - - - - - - - - - - - - -

    # ==========================================================
    # We read .interp file only once into 'listi'
    # and the starting points of each 'images' and 'interp'
    # sections in the file.
    # =========================================================

    listi, start_images, start_spline = read_spline_file(fname)
    no_of_iters = len(start_spline)

    if end_at == -1:
        end_at = no_of_iters

    # ==========================================================
    # Make some checks...
    # =========================================================

    if len(start_images) != len(start_spline):
        raise RuntimeError("Corrupt spline file!")

    if start_at > no_of_iters or end_at > no_of_iters or start_at > end_at:
        raise RuntimeError("The number of iterations in the .interp file is incorrect")

    # ==========================================================
    # Create dir. neb_frames (you can comment out this section)
    # ==========================================================
    path = os.getcwd()
    working_dir = path + '/neb_frames'

    if os.path.isdir(working_dir):
        print('Directory %s found!' % working_dir)
        print('    => Existing files are overwritten!')
    else:
        os.mkdir(working_dir)
        print('Working dir: %s' % working_dir)

    os.chdir(working_dir)

    one_iter = False
    if no_of_iters == 1:
        if 'final' in fname.lower():
            print('*** Note that %s contains only the last iteration of a NEB/CI-NEB run  ***' % fname)
        else:
            print('%s contains only one iteration?   ***' % fname)
        one_iter = True

    # ==========================================================
    # Read and plot the spline and images of the .interp file
    # ==========================================================
    # if not one_iter:
    #    print('Iteration: ')

    for i in range(start_at, end_at):
        saveI = i
        #   if not one_iter:
        #       print('%3i' % i)

        arcS, Eimg = read_images(start_images[i] + 1, listi)
        arcS2, Eimg2 = read_spline(start_spline[i] + 1, listi)

        # Convert atomic units to eV and angstr.
        newE1 = np.array(Eimg)  # *27.211396132
        newE2 = np.array(Eimg2)  # *27.211396132
        newS1 = np.array(arcS)  # /1.889725989
        newS2 = np.array(arcS2)  # /1.889725989

        if not one_frame:
            # Generate and save frame
            plt.plot(newS2, newE2, '-k', label='Interp.')
            plt.plot(newS1, newE1, '.r', Markersize=5.5, label='NEB img.')
            plt.xlabel("Reaction path")
            plt.ylabel("Energy")
            plt.savefig('frame_' + str(saveI) + '.png')
            plt.clf()
        else:
            # Plot frames together
            if i == end_at - 1:
                plt.plot(newS2, newE2, '-r', label='Last iter.')
                plt.plot(newS1, newE1, '.r', Markersize=5.5)
            else:
                plt.plot(newS2, newE2, '-k', label='Interp.')
                plt.plot(newS1, newE1, '.y', Markersize=5.5, label='NEB img.')

    # save whole trajectory
    if one_frame:
        plt.xlabel("Reaction path")
        plt.ylabel("Energy")
        if not one_iter:
            plt.title("Iter.:" + str(start_at) + " to " + str(saveI))
        plt.savefig('neb_optimization.png')

        plt.clf()
        plt.plot(newS2, newE2, '-k', label='Interp.')
        plt.plot(newS1, newE1, '.r', label='NEB img.')
        plt.xlabel("Reaction path")
        plt.ylabel("Energy")
        plt.savefig('neb_lastiter.png')

    print('')
    print('==========================================')
    print('Execution terminated (see /neb_frames).')
    print('==========================================')
    return None


def PlotSurface(fname, workerfunc=None, xbound=None, ybound=None,
                cbound=None, npts=100, list_of_points=None,
                list_of_arrays=None, filled=True, ncont=200):
    if xbound is None or ybound is None:
        raise ValueError("Boundaries are missing")

    if len(xbound) != 2 or len(ybound) != 2:
        raise ValueError("Dimension mismatch")

    assert isinstance(npts, int)

    x = np.linspace(xbound[0], xbound[1], npts)
    y = np.linspace(ybound[0], ybound[1], npts)
    X, Y = np.meshgrid(x, y)
    n, m = np.shape(X)
    E = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            Etmp = workerfunc(np.array([X[i, j], Y[i, j], 0.0]))[1]
            E[i, j] = Etmp

    plt.figure(0)
    if filled:
        CS = plt.contourf(X, Y, E, ncont)
    else:
        CS = plt.contour(X, Y, E, ncont)

    if cbound is not None:
        CS.set_clim(cbound[0], cbound[1])
    # plt.colorbar(m, boundaries=np.linspace(0, 2, 6))
    m = plt.cm.ScalarMappable()
    m.set_array(E)
    if cbound is not None:
        m.set_clim(cbound[0], cbound[1])
        # plt.colorbar(m, boundaries=np.linspace(cbound[0], cbound[1],10))

    if list_of_points is not None:
        for i in list_of_points:
            plt.plot(i[0], i[1], 'ok', MarkerSize=2)

    if list_of_arrays is not None:
        assert type(list_of_arrays) == list
        for i in list_of_arrays:
            plt.plot(i[:, 0], i[:, 1], '-k', LineWidth=1.5)

    CS.set_cmap('RdYlBu_r')
    # CS.set_cmap('YlGnBu')
    plt.savefig(fname + '.png')
    plt.show()
    plt.clf()
    # print np.max(E)
    # print np.min(E)
    return None


def PlotCos2Surface(fname, calculator, atoms,
                    xbound=None, ybound=None,
                    npts=100, ncont=100):
    if xbound is None or ybound is None:
        raise ValueError("Boundaries are missing")

    if len(xbound) != 2 or len(ybound) != 2:
        raise ValueError("Dimension mismatch")

    assert isinstance(npts, int)

    x = np.linspace(xbound[0], xbound[1], npts)
    y = np.linspace(ybound[0], ybound[1], npts)
    X, Y = np.meshgrid(x, y)
    n, m = np.shape(X)
    cos2 = np.zeros(shape=(n, m))
    E = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            atoms.SetR(np.reshape([X[i, j], Y[i, j], 0.0], (3, 1)))
            atoms.UpdateCoords()
            cos2tmp = calculator.ComputeCosSq(atoms)
            cos2[i, j] = cos2tmp
            Etmp = calculator.Compute(atoms)
            E[i, j] = atoms.GetEnergy()

    with open('cos2.dat', 'w') as f:
        for i in range(n):
            for j in range(m):
                f.write('%12.8f %12.8f %12.8f\n' % (X[i, j], Y[i, j], cos2[i, j]))

    plt.contour(X, Y, cos2, 40)
    CS = plt.contourf(X, Y, cos2, ncont)
    # plt.clim([0,1])
    plt.colorbar()
    CS.set_cmap('jet')
    # plt.show()
    # plt.gca().set_aspect('equal')#, adjustable='box')
    # CS.set_cmap('viridis')
    # CS.set_cmap('YlGnBu')
    plt.savefig(fname + '.png')
    plt.show()
    return None


def PlotCos2GradSurface(fname, calculator, atoms,
                        xbound=None, ybound=None,
                        npts=100, ncont=100):
    if xbound is None or ybound is None:
        raise ValueError("Boundaries are missing")

    if len(xbound) != 2 or len(ybound) != 2:
        raise ValueError("Dimension mismatch")

    assert isinstance(npts, int)

    x = np.linspace(xbound[0], xbound[1], npts)
    y = np.linspace(ybound[0], ybound[1], npts)
    X, Y = np.meshgrid(x, y)
    n, m = np.shape(X)
    energy = np.zeros(shape=(n, m))
    cos2 = np.zeros(shape=(n, m))
    grad0 = np.zeros(shape=(n, m))
    grad1 = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            atoms.SetR(np.reshape([X[i, j], Y[i, j], 0.0], (3, 1)))
            atoms.UpdateCoords()
            calculator.Compute(atoms)
            energy[i, j] = atoms.GetEnergy()
            cos2tmp = calculator.ComputeCosSq(atoms)
            cos2[i, j] = cos2tmp
            tmp_grad = calculator.ComputeCosSqGrad(atoms)
            grad0[i, j] = tmp_grad[0]
            grad1[i, j] = tmp_grad[1]

    plt.contour(X, Y, cos2, 40)
    plt.quiver(X, Y, grad0, grad1)
    plt.savefig(fname + '_cos2_gradient.png')
    plt.clf()
    return None


def PlotEigSurface(fname, calculator, atoms, xbound=None,
                   ybound=None, npts=30, ncont=100):
    x = np.linspace(xbound[0], xbound[1], npts)
    y = np.linspace(ybound[0], ybound[1], npts)
    X, Y = np.meshgrid(x, y)
    n, m = np.shape(X)
    energy = np.zeros(shape=(n, m))
    gradx = np.zeros(shape=(n, m))
    grady = np.zeros(shape=(n, m))
    eig0 = np.zeros(shape=(n, m))
    eig1 = np.zeros(shape=(n, m))
    eigv0x = np.zeros(shape=(n, m))
    eigv0y = np.zeros(shape=(n, m))
    eigv1x = np.zeros(shape=(n, m))
    eigv1y = np.zeros(shape=(n, m))

    for i in range(n):
        for j in range(m):
            atoms.SetR(np.reshape([X[i, j], Y[i, j], 0.0], (3, 1)))
            atoms.UpdateCoords()
            calculator.Compute(atoms)
            atoms.UpdateF()
            calculator.Compute(atoms, computeHessian=True)
            energy[i, j] = atoms.GetEnergy()
            hessian = atoms.GetHessian()
            hessian = hessian[0:2, 0:2, 0]  # for 2d
            eigenval, eigenvec = np.linalg.eigh(hessian)
            gradx[i, j] = atoms.GetF()[0]
            grady[i, j] = atoms.GetF()[1]
            eig0[i, j] = eigenval[0]
            eig1[i, j] = eigenval[1]
            eigv0x[i, j] = eigenvec[0, 0]
            eigv0y[i, j] = eigenvec[1, 0]
            eigv1x[i, j] = eigenvec[0, 1]
            eigv1y[i, j] = eigenvec[1, 1]

    plt.contour(X, Y, energy, 40)
    CS = plt.contourf(X, Y, eig0, ncont)
    CS.set_cmap('jet')
    plt.colorbar()
    # plt.clim([-2, 2])
    # plt.colorbar()
    plt.savefig(fname + '_eigenvalue0.png')
    plt.clf()

    plt.contour(X, Y, energy, 40)
    CS = plt.contourf(X, Y, eig1, ncont)
    CS.set_cmap('jet')
    plt.colorbar()
    # plt.clim([-2,2])
    # plt.colorbar()
    plt.savefig(fname + '_eigenvalue1.png')
    plt.clf()

    plt.contour(X, Y, energy, 40)
    plt.quiver(X, Y, gradx, grady)
    plt.savefig(fname + '_gradient.png')
    plt.clf()

    plt.contour(X, Y, energy, 40)
    plt.quiver(X, Y, eigv0x, eigv0y)
    plt.savefig(fname + '_eigenvector0.png')
    plt.clf()

    plt.contour(X, Y, energy, 40)
    plt.quiver(X, Y, eigv1x, eigv1y)
    plt.savefig(fname + '_eigenvector1.png')
    plt.clf()

    return None


def PlotEnergyAndCosSurface(fname, calculator, atoms,
                            xbound=None, ybound=None,
                            npts=100, ncont=100):
    if xbound is None or ybound is None:
        raise ValueError("Boundaries are missing")

    if len(xbound) != 2 or len(ybound) != 2:
        raise ValueError("Dimension mismatch")

    assert isinstance(npts, int)

    x = np.linspace(xbound[0], xbound[1], npts)
    y = np.linspace(ybound[0], ybound[1], npts)
    X, Y = np.meshgrid(x, y)
    n, m = np.shape(X)
    cos2 = np.zeros(shape=(n, m))
    E = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            atoms.SetR(np.reshape([X[i, j], Y[i, j], 0.0], (3, 1)))
            atoms.UpdateCoords()
            cos2tmp = calculator.ComputeCosSq(atoms)
            cos2[i, j] = cos2tmp
            calculator.Compute(atoms)
            E[i, j] = atoms.GetEnergy()

    listi = []
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            refCOS = cos2[i, j]
            # Check for condition A
            if refCOS >= cos2[i - 1, j + 1] and refCOS >= cos2[i + 1, j - 1]:
                listi.append([0, X[i, j], Y[i, j], refCOS])
                continue
            # Check for condition B
            if refCOS >= cos2[i - 1, j - 1] and refCOS >= cos2[i + 1, j + 1]:
                listi.append([1, X[i, j], Y[i, j], refCOS])
                continue
            # Check for condition C
            if refCOS >= cos2[i - 1, j] and refCOS >= cos2[i + 1, j]:
                listi.append([2, X[i, j], Y[i, j], refCOS])
                continue
            if refCOS >= cos2[i, j + 1] and refCOS >= cos2[i, j - 1]:
                listi.append([3, X[i, j], Y[i, j], refCOS])
                continue

    plt.contour(X, Y, E, ncont)
    CS = plt.contourf(X, Y, E, ncont)
    CS.set_cmap('jet')
    # plt.clim([-3, 6])
    plt.clim([-0.15, 0.2])
    circle_size = 5
    x = np.zeros(shape=(len(listi), 1))
    y = np.zeros(shape=(len(listi), 1))
    z = np.zeros(shape=(len(listi), 1))
    for i in range(len(listi)):
        x[i] = listi[i][1]
        y[i] = listi[i][2]
        z[i] = listi[i][3]
    cmap = matplotlib.cm.gray  #

    z = z - 0.9
    for i in range(len(z)):
        if z[i] < 0.0:
            z[i] = 0.0
        z[i] *= 10.0

    plt.scatter(x, y, s=circle_size, c=1 - z, cmap=cmap)

    plt.show()
    plt.savefig(fname + '_scatter_ontop_of_contour.png')
    return None


def PlotEigLinesOnSurface(fname, calculator, atoms,
                          xbound=None, ybound=None,
                          npts=100, ncont=100, tol=1e-2):
    if xbound is None or ybound is None:
        raise ValueError("Boundaries are missing")

    if len(xbound) != 2 or len(ybound) != 2:
        raise ValueError("Dimension mismatch")

    assert isinstance(npts, int)

    x = np.linspace(xbound[0], xbound[1], npts)
    y = np.linspace(ybound[0], ybound[1], npts)
    X, Y = np.meshgrid(x, y)
    n, m = np.shape(X)
    cos2 = np.zeros(shape=(n, m))
    E = np.zeros(shape=(n, m))
    eig0 = np.zeros(shape=(n, m))
    eig1 = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            atoms.SetR(np.reshape([X[i, j], Y[i, j], 0.0], (3, 1)))
            atoms.UpdateCoords()
            cos2tmp = calculator.ComputeCosSq(atoms)
            cos2[i, j] = cos2tmp
            calculator.Compute(atoms)
            E[i, j] = atoms.GetEnergy()
            hessian = atoms.GetHessian()
            hessian = hessian[0:2, 0:2, 0]  # for 2d
            eigenval, eigenvec = np.linalg.eigh(hessian)
            eig0[i, j] = eigenval[0]
            eig1[i, j] = eigenval[1]

    plt.contour(X, Y, E, ncont)
    CS = plt.contourf(X, Y, E, ncont)
    CS.set_cmap('jet')
    #plt.clim([-3, 6])
    plt.clim([-0.15, 0.2])

    for i in range(n):
        for j in range(m):
            if abs(eig0[i, j]) < tol:
                plt.plot(X[i, j], Y[i, j], 'ow')

            if abs(eig1[i, j]) < tol:
                plt.plot(X[i, j], Y[i, j], 'ok')

            deigs = abs(eig0[i, j] - eig1[i, j])
            if deigs < tol:
                plt.plot(X[i, j], Y[i, j], 'om')


    plt.savefig(fname + '_eig001')
    plt.show()
    return None
