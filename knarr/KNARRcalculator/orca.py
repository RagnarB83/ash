import numpy as np
import subprocess
import os
import shutil

from KNARRcalculator.utilities import ExecuteCalculatorParallel
import KNARRsettings


# Author: Vilhjalmur Asgeirsson, 2019.


# TODO: one day I should clean up these routines and merge into one driver and worker

def ReadORCAhess(ndim, fname):
    start_read = False
    f = open(fname).readlines()
    HESS = []
    for i, line in enumerate(f):

        if '$vibrational_frequencies' in line:
            start_read = False

        if start_read == True:
            k = line.split()
            if len(k) > 1:
                HESS.append([k])

        if '$hessian' in line:
            start_read = True

    ind = 0
    a = len(HESS) / (ndim + 1)
    H = np.zeros(shape=(ndim, ndim))
    for i in range(a):
        image = HESS[ind]
        ind += 1
        a, b = np.shape(image)
        col = [0]
        for z in range(b):
            col.append(int(image[0][z]))

        for j in range(0, ndim):
            val = HESS[ind]
            line = int(val[0][0])
            for z in range(1, len(col)):
                H[line, col[z]] = float(val[0][z]) * 97.173645604
            ind += 1

    return H


def PrepareORCAInput(working_file, template, R, symb,
                     charge, multiplicity, job_type=0):
    input = template
    if job_type == 0:
        input.insert(0, '!ENGRAD\n')
    elif job_type == 1:
        input.insert(0, '!FREQ\n')
    else:
        raise RuntimeError("Unknown orca job")

    string = '%s %2i %2i\n' % ('*xyz', charge, multiplicity)
    input.append(string)
    for i in range(0, len(R), 3):
        string = '%s % 12.6f % 12.6f % 12.6f\n' % \
                 (symb[i], R[i], R[i + 1], R[i + 2])
        input.append(string)

    input.append('*')

    with open(working_file, 'w') as fout:
        for i, val in enumerate(input):
            fout.write(val)

    return None


def ORCAHess(calculator, atoms):
    H = np.zeros(shape=(atoms.GetNDimIm(), atoms.GetNDimIm(), atoms.GetNim()))
    current_dir = calculator.path
    ncore = calculator.GetNCore()
    counter = 0
    list_of_jobs = []
    for i in range(atoms.GetNim()):
        image = i
        working_dir = current_dir + '/image_' + str(image)
        if os.path.isdir(working_dir):
            shutil.rmtree(working_dir)
        list_of_jobs.append((i, current_dir, working_dir, calculator.GetQCPath(),
                             image, atoms.GetNDimIm(),
                             atoms.GetCoords()[i * atoms.GetNDimIm():(i + 1) * atoms.GetNDimIm()]
                             , atoms.GetSymbols(), calculator.GetCharge(), calculator.GetMultiplicity(),
                             calculator.GetTemplate()))
        counter += 1

    # Run XTB in parallel execution over number of images
    results = ExecuteCalculatorParallel(ORCAHessWorker, ncore, list_of_jobs)
    atoms.AddFC(counter)

    for i, val in enumerate(results):
        H[:, :, val[1]] = val[0].copy()

    atoms.SetHessian(H)
    return


def ORCAHessWorker(X):
    current_dir = X[1]
    working_dir = X[2]
    path_to_code = X[3]
    image = X[4]
    ndim = X[5]
    rxyz = X[6]
    symb = X[7]
    charge = X[8]
    multiplicity = X[9]
    template = X[10]

    basefname = 'orca_' + str(image)

    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    # Copy template file
    PrepareORCAInput(working_dir + '/' + basefname + '.inp', template, rxyz,
                     symb, charge, multiplicity, job_type=1)

    # Run template file
    args = [path_to_code, working_dir + '/' + basefname + '.inp']
    orcatmp = open(working_dir + '/' + basefname + '.out', 'w')
    orcaerr = open(working_dir + '/' + basefname + '.err', 'w')
    P = subprocess.Popen(args, stdout=orcatmp, stderr=orcaerr)
    P.wait()

    orcatmp.close()
    orcaerr.close()

    status_good = False
    if os.path.isfile(working_dir + '/' + basefname + '.out'):
        check = open(working_dir + '/' + basefname + '.out').readlines()
        for k, line in enumerate(check):
            if 'ORCA TERMINATED NORMALLY' in line:
                status_good = True

    if not status_good:
        raise RuntimeError("ORCA calculation failed. Inspect image: %i" % image)

    if not os.path.isfile(working_dir + '/' + basefname + '.hess'):
        raise RuntimeError("Unable to find .hess file. Inspect image: %i" % image)

    H = ReadORCAhess(ndim, working_dir + '/' + basefname + '.hess')

    # clean up
    if os.path.isfile(working_dir + '/' + basefname + '.hess'):
        os.remove(working_dir + '/' + basefname + '.hess')

    H = H * KNARRsettings.au_to_eva2

    os.chdir(current_dir)

    return H, X[0]



def ORCAworker(X):
    current_dir = X[1]
    working_dir = X[2]
    path_to_code = X[3]
    image = X[4]
    ndim = X[5]
    rxyz = X[6]
    symb = X[7]
    charge = X[8]
    multiplicity = X[9]
    template = X[10]

    basefname = 'orca_' + str(image)

    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    # Copy template file
    PrepareORCAInput(working_dir + '/' + basefname + '.inp', template, rxyz,
                     symb, charge, multiplicity, job_type=0)

    # Run template file
    args = [path_to_code, working_dir + '/' + basefname + '.inp']
    orcatmp = open(working_dir + '/' + basefname + '.out', 'w')
    orcaerr = open(working_dir + '/' + basefname + '.err', 'w')
    P = subprocess.Popen(args, stdout=orcatmp, stderr=orcaerr)
    P.wait()

    orcatmp.close()
    orcaerr.close()

    status_good = False
    if os.path.isfile(working_dir + '/' + basefname + '.out'):
        check = open(working_dir + '/' + basefname + '.out').readlines()
        for k, line in enumerate(check):
            if 'ORCA TERMINATED NORMALLY' in line:
                status_good = True

    if not status_good:
        raise RuntimeError("ORCA calculation failed. Inspect image: %i" % image)

    if not os.path.isfile(working_dir + '/' + basefname + '.engrad'):
        raise RuntimeError("Unable to find .engrad file. Inspect image: %i" % image)

    Ftmp = np.zeros(shape=(ndim, 1))

    with open(working_dir + '/' + basefname + '.engrad') as f:
        ind = 0
        for n, line in enumerate(f):
            if n == 7:
                Etmp = line.split()
            if n >= 11 and n < ndim + 11:
                try:
                    Ftmp[ind] = float(line)
                    ind += 1
                except:
                    raise RuntimeError("Unable to read .gradient for image: %i )")
    try:
        E = float(Etmp[0])
    except:
        raise RuntimeError("Unable to read .gradient for image: %i" % image)

    # clean up
    if os.path.isfile(working_dir + '/' + basefname + '.engrad'):
        os.remove(working_dir + '/' + basefname + '.engrad')

    F = -Ftmp * KNARRsettings.au_to_eva
    E *= KNARRsettings.au_to_ev

    os.chdir(current_dir)

    return F, E, X[0]


def ORCA(calculator, atoms, list_to_compute):
    F = np.zeros(shape=(atoms.GetNDimIm() * atoms.GetNim(), 1))
    E = np.zeros(shape=(atoms.GetNim(), 1))
    current_dir = calculator.path
    ncore = calculator.GetNCore()
    counter = 0
    list_of_jobs = []

    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            image = i
            working_dir = current_dir + '/image_' + str(image)
            if os.path.isdir(working_dir):
                shutil.rmtree(working_dir)
            list_of_jobs.append((i, current_dir, working_dir, calculator.GetQCPath(),
                                 image, atoms.GetNDimIm(),
                                 atoms.GetCoords()[i * atoms.GetNDimIm():(i + 1) * atoms.GetNDimIm()]
                                 , atoms.GetSymbols(), calculator.GetCharge(), calculator.GetMultiplicity(),
                                 calculator.GetTemplate()))
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
            list_of_jobs.append((i, current_dir, working_dir, calculator.GetQCPath(),
                                 image, atoms.GetNDimIm(),
                                 atoms.GetCoords()[i * atoms.GetNDimIm():(i + 1) * atoms.GetNDimIm()]
                                 , atoms.GetSymbols(), calculator.GetCharge(), calculator.GetMultiplicity(),
                                 calculator.GetTemplate()))
            counter += 1

    # Run parallel code
    results = ExecuteCalculatorParallel(ORCAworker, ncore, list_of_jobs)
    atoms.AddFC(counter)

    for i, val in enumerate(results):
        F[val[2] * atoms.GetNDimIm():(i + 1) * atoms.GetNDimIm()] = val[0]
        E[i] = float(val[1])

    atoms.SetForces(F)
    atoms.SetEnergy(E)

    return F, E
