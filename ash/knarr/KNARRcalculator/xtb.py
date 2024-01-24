import numpy as np
import os
import shutil
import subprocess

from KNARRcalculator.utilities import ExecuteCalculatorParallel
import KNARRsettings


# Author: Vilhjalmur Asgeirsson, 2019.

def CleanXTB(working_dir):
    # clean up
    if os.path.isfile(working_dir + '/charges'):
        os.remove(working_dir + '/charges')
    if os.path.isfile(working_dir + '/gradient'):
        os.remove(working_dir + '/gradient')
    if os.path.isfile(working_dir + '/energy'):
        os.remove(working_dir + '/energy')
    if os.path.isfile(working_dir + '/hessian'):
        os.remove(working_dir + '/hessian')
    if os.path.isfile(working_dir + '/xtb_normalmodes'):
        os.remove(working_dir + '/xtb_normalmodes')
    if os.path.isfile(working_dir + '/xtbrestart'):
        os.remove(working_dir + '/xtbrestart')
    if os.path.isfile(working_dir + '/molden.input'):
        os.remove(working_dir + '/molden.input')
    if os.path.isfile(working_dir + '/wbo'):
        os.remove(working_dir + '/wbo')

    return None


def ReadXTBhess(ndim, fname='hessian'):
    Htmp = np.zeros(shape=(ndim ** 2))
    f = open(fname).readlines()
    ind = 0
    for i, line in enumerate(f):
        if '$hessian' not in line:
            k = line.split()
            for j in k:
                Htmp[ind] = float(j)
                ind = ind + 1
    H = np.reshape(Htmp, (ndim, ndim))
    return H


def XTBHess(calculator, atoms):
    # Wrapper for the XTB Hessian  calculator
    H = np.zeros(shape=(atoms.GetNDimIm(), atoms.GetNDimIm(), atoms.GetNim()))

    list_of_jobs = []

    R = atoms.GetCoords()
    for i in range(atoms.nim):
        image = i
        current_dir = calculator.path
        working_dir = current_dir + '/image_' + str(image)
        Rtmp = R[i * atoms.ndim:(i + 1) * atoms.ndim]
        ncore = calculator.GetNCore()
        list_of_jobs.append((i, current_dir, working_dir, calculator.GetQCPath(), image,
                             atoms.GetNDim(), Rtmp, atoms.GetSymbols(),
                             calculator.GetCharge(), calculator.GetMultiplicity()))

    # Run XTB in parallel execution over number of images
    results = ExecuteCalculatorParallel(XTBHessWorker, ncore, list_of_jobs)
    atoms.AddFC(atoms.nim)

    for i, val in enumerate(results):
        if val[1] == i:
            H[:, :, i] = val[0].copy()
        else:
            raise RuntimeError("Something went wrong in the parallel execution of XTBHess")

    H = H * KNARRsettings.au_to_eva2
    atoms.SetHessian(H)

    return None


def XTBHessWorker(X):
    # Actual XTB Hessian execution

    # Read XTBworker input 'X'
    current_dir = X[1]
    working_dir = X[2]
    path_to_code = X[3]
    image = X[4]
    ndim = X[5]
    rxyz = X[6]
    symb = X[7]
    charge = X[8]
    multiplicity = X[9]
    unpel = int(multiplicity - 1)

    # Check if working dir already exists
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)
    os.chdir(working_dir)

    # write xyz input to disk
    basefname = 'XTB_im_' + str(image)
    with open(working_dir + '/' + basefname + '.xyz', 'w') as f:
        f.write(str(ndim / 3) + '\n\n')
        for j in range(0, ndim, 3):
            f.write('%s %12.8f %12.8f %12.8f \n' % (symb[j], rxyz[j + 0], rxyz[j + 1], rxyz[j + 2]))

    # Execution arguments
    args = [path_to_code, working_dir + '/' + basefname + '.xyz', '--uhf ', str(unpel), '--chrg ',
            str(charge), '--acc', str(0.0001), '--hess']

    # =================== EXECUTE XTB CODE =====================
    xtbtmp = open(working_dir + '/xtb.out', 'w')  # stdout
    xtberr = open(working_dir + '/xtb.err', 'w')  # stderr
    P = subprocess.Popen(args, stdout=xtbtmp, stderr=xtberr)
    P.wait()
    xtbtmp.close()
    xtberr.close()
    # ==========================================================

    # Attempt a second calculation (with less accuracy) - if first failed.
    if not os.path.isfile(working_dir + '/' + 'hessian'):
        raise RuntimeError('No hessian found. There is no way to reocver. ' \
                           'Please inspect folder for image: %i' % image)

    H = ReadXTBhess(ndim, fname='hessian')

    CleanXTB(working_dir)

    os.chdir(current_dir)

    return H, X[0]


def XTB(calculator, atoms, list_to_compute):
    # Wrapper for the XTB calculator
    list_of_jobs = []
    R = atoms.GetCoords().copy()
    F = np.zeros(shape=(atoms.GetNDimIm() * atoms.GetNim(), 1))
    E = np.zeros(shape=(atoms.GetNim(), 1))
    current_dir = calculator.path
    ncore = calculator.GetNCore()
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.nim):
            image = i
            working_dir = current_dir + '/image_' + str(image)
            if os.path.isdir(working_dir):
                shutil.rmtree(working_dir)
            Rtmp = R[i * atoms.GetNDimIm():(i + 1) * atoms.GetNDimIm()]
            list_of_jobs.append((i, current_dir, working_dir, calculator.GetQCPath(), image,
                                 atoms.GetNDimIm(), Rtmp, atoms.GetSymbols(),
                                 calculator.GetCharge(), calculator.GetMultiplicity()))
            counter += 1
    else:
        if len(list_to_compute) > atoms.GetNim():
            raise RuntimeError("Wrong number of images")
        if not list_to_compute:
            raise RuntimeError("Nothing to be computed")

        for i, val in enumerate(list_to_compute):
            image = val
            working_dir = current_dir + '/image_' + str(image)
            if os.path.isdir(working_dir):
                shutil.rmtree(working_dir)
            Rtmp = R[val * atoms.GetNDimIm():(val + 1) * atoms.GetNDimIm()]
            list_of_jobs.append((val, current_dir, working_dir, calculator.GetQCPath(), image,
                                 atoms.GetNDimIm(), Rtmp, atoms.GetSymbols(),
                                 calculator.GetCharge(), calculator.GetMultiplicity()))
            counter += 1

    # Run XTB in parallel execution over number of images
    results = ExecuteCalculatorParallel(XTBWorker, ncore, list_of_jobs)
    atoms.AddFC(counter)

    for i, val in enumerate(results):
        F[val[2] * atoms.GetNDimIm():(val[2] + 1) * atoms.GetNDimIm()] = val[0]
        E[val[2]] = float(val[1])

    atoms.SetForces(F)
    atoms.SetEnergy(E)

    return None


def XTBWorker(X):
    # Actual XTB execution

    # Read XTBworker input 'X'
    current_dir = X[1]
    working_dir = X[2]
    path_to_code = X[3]
    image = X[4]
    ndim = X[5]
    rxyz = X[6]
    symb = X[7]
    charge = X[8]
    multiplicity = X[9]
    unpel = int(multiplicity - 1)

    # Check if working dir already exists
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)
    os.chdir(working_dir)

    # write xyz input to disk
    basefname = 'XTB_im_' + str(image)
    with open(working_dir + '/' + basefname + '.xyz', 'w') as f:
        f.write(str(ndim / 3) + '\n\n')
        for j in range(0, ndim, 3):
            f.write('%s %12.8f %12.8f %12.8f \n' % (symb[j], rxyz[j + 0], rxyz[j + 1], rxyz[j + 2]))

    # Execution arguments
    args = [path_to_code, working_dir + '/' + basefname + '.xyz', '--uhf ', str(unpel), '--chrg ',
            str(charge), '--acc', str(0.0001), '--grad']

    # =================== EXECUTE XTB CODE =====================
    xtbtmp = open(working_dir + '/xtb.out', 'w')  # stdout
    xtberr = open(working_dir + '/xtb.err', 'w')  # stderr
    P = subprocess.Popen(args, stdout=xtbtmp, stderr=xtberr)
    P.wait()
    xtbtmp.close()
    xtberr.close()
    # ==========================================================

    # Attempt a second calculation (with less accuracy) - if first failed.
    if not os.path.isfile(working_dir + '/' + 'gradient'):

        args = [path_to_code, working_dir + '/' + basefname + '.xyz', '--uhf ', str(unpel), '--chrg ',
                str(charge), '--grad']

        xtbtmp = open(working_dir + '/xtb.out', 'w')
        xtberr = open(working_dir + '/xtb.err', 'w')
        P = subprocess.Popen(args, stdout=xtbtmp, stderr=xtberr)
        P.wait()
        xtbtmp.close()
        xtberr.close()

        if not os.path.isfile(working_dir + '/gradient'):
            raise RuntimeError('Second  XTB calculation failed. There is no way to reocver. ' \
                               'Please inspect folder for image: %i' % image)

    Ftmp = np.zeros(shape=(ndim, 1))
    with open(working_dir + '/gradient') as f:
        lines = f.readlines()
        ind = 0
        for j, line in enumerate(lines):
            line = line.split()
            if len(line) == 10:
                E = float(line[6])
            if len(line) == 3:
                Ftmp[ind] = float(line[0].replace("D", "E"))
                Ftmp[ind + 1] = float(line[1].replace("D", "E"))
                Ftmp[ind + 2] = float(line[2].replace("D", "E"))
                ind = ind + 3

    CleanXTB(working_dir)

    F = -Ftmp  # Get forces - not gradient

    # Convert to eV/Angstr
    F = F * KNARRsettings.au_to_eva
    E = E * KNARRsettings.au_to_ev

    os.chdir(current_dir)

    return F, E, X[0]
