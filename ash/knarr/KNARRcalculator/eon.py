import numpy as np
import os
import shutil
import subprocess
from KNARRcalculator.utilities import ExecuteCalculatorParallel
from KNARRio.io import WriteCon
import KNARRsettings

def EON(calculator, atoms, list_to_compute):
    # Wrapper for the XTB calculator
    list_of_jobs = []
    R = atoms.GetCoords().copy()
    F = np.zeros(shape=(atoms.GetNDimIm() * atoms.GetNim(), 1))
    E = np.zeros(shape=(atoms.GetNim(), 1))
    current_dir = calculator.path
    ncore = calculator.GetNCore()
    template = calculator.GetTemplate()
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
                                 atoms.GetConstraints(), atoms.GetCell(), template))
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
            list_of_jobs.append((i, current_dir, working_dir, calculator.GetQCPath(), image,
                                 atoms.GetNDimIm(), Rtmp, atoms.GetSymbols(),
                                 atoms.GetConstraints(), atoms.GetCell(), template))

            counter += 1


    # Run XTB in parallel execution over number of images
    results = ExecuteCalculatorParallel(EONWorker, ncore, list_of_jobs)
    atoms.AddFC(counter)
    for i, val in enumerate(results):
        F[val[2] * atoms.GetNDimIm():(val[2] + 1) * atoms.GetNDimIm()] = val[0]
        E[val[2]] = float(val[1])

    atoms.SetForces(F)
    atoms.SetEnergy(E)

    return None

def WriteConfigFile(template=None):

    if template is None:
        potential = 'tip4p'
    else:
        potential =template


    with open('config.ini','w') as fconfig:
        fconfig.write('[main]\n')
        fconfig.write('job=point\n')
        fconfig.write('[potential]\n')
        fconfig.write('potential=%s\n' % potential)
    return None

def ReadForces(fname='gradient.dat'):
    forces = np.loadtxt(fname)
    natoms = len(forces)
    forces = np.reshape(forces, (3*natoms,1))
    return forces

def ReadEnergy(fname='energy.dat'):
    with open(fname) as f:
        try:
            energy=float(f.readlines()[0])
        except:
            raise RuntimeError("Incorrect format of energy.dat")
    return energy


def EONWorker(X):
    current_dir = X[1]
    working_dir = X[2]
    path_to_code = X[3]
    image = X[4]
    ndim = X[5]
    rxyz = X[6]
    symb = X[7]
    constr = X[8]
    cell = X[9]
    template = X[10]

    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    os.chdir(working_dir)

    # write input
    WriteCon('pos.con', 1, rxyz, symb, cell, constr)
    WriteConfigFile(template)

    # execute EON
    args=[path_to_code]
    eon_tmp = open(working_dir + '/eon.out', 'w')
    eon_err = open(working_dir + '/eon.err', 'w')
    P = subprocess.Popen(args, stdout=eon_tmp, stderr=eon_err)
    P.wait()
    eon_tmp.close()
    eon_err.close()

    # read output
    F=ReadForces()
    E=ReadEnergy()

    os.chdir(current_dir)

    return F, E, X[0]
