import time
import KNARRsettings
import shutil
import os
import numpy as np

from KNARRio.system_print import PrintJob, PrintCallBack, PrintJobDone
from KNARRio.io import WriteModeToFile
from KNARRjobs.utilities import ComputeFreq, ComputeTc, GenerateVibrTrajectory


# Author: Vilhjalmur Asgeirsson (2019)


def DoFreq(atoms, calculator, freq):
    PrintJob('Vibrational frequency computation')
    PrintCallBack('freqjob', calculator, atoms)

    if not atoms.setup:
        raise RuntimeError("Atoms object is not properly initialized")

    if not calculator.setup:
        raise RuntimeError("Calculator is not properly initialized")

    atoms.PrintConfiguration('Input configuration:')
    start_t = time.time()

    partialHessian = freq["PARTIAL_HESSIAN"]
    
    basename = atoms.GetOutputFile()
    ischain = atoms.IsChain()

    npoints = 40  # number of points to be included in trj file
    magn_displ = 1.0  # displacement of vibr trj

    # ---------------------------------------------------------------------------------
    # Actual Hessian calculation
    # ---------------------------------------------------------------------------------
    if ischain:
        raise NotImplementedError("Frequency calculations of chains are not ready. Skipping calculation,")

    calculator.Compute(atoms)
    print 'Energy: %12.8f eV' % atoms.GetEnergy()
    calculator.Compute(atoms, computeHessian=True)
    H = atoms.GetHessian()

    if ischain:
        raise NotImplementedError("Frequency calculations of chains are not ready")
    else:
        Hcomp = H[:, :, 0]

    if partialHessian:
        if calculator.hessian:
            #if analytical hessian
            raise NotImplementedError("Frequency computations using partial analytic Hessian are not implemented")
        else:
            eig, w = ComputeFreq(atoms.GetNDof(), atoms.GetGlobalDof(), atoms.GetFreeMass(), Hcomp)
    else:
        if calculator.hessian:
            #if analytical hessian
            eig, w = ComputeFreq(atoms.GetNDim(), atoms.GetGlobalDof(), atoms.GetMass(), Hcomp)
        else:
            raise NotImplementedError("Frequency computations using full numerical Hessian is not implemented")

        
    
    i = np.where(eig == np.min(eig[np.nonzero(eig)]))
    lowest_index = int(i[0])
    wlowest = w[:, lowest_index]
    full_w = np.zeros(shape=(atoms.GetNDim(),1))
    full_w[atoms.GetMoveableAtoms()] = np.reshape(wlowest, (atoms.GetNDof(),1))

    print("\nVibr. freq:")
    for i in range(len(eig)):
        print 'eig:% 4li % 6.4lf (% 6.4lf cm-1)' % (
            i + 1, eig[i], np.sqrt(np.abs(eig[i])) / (2 * np.pi * KNARRsettings.time_unit * KNARRsettings.c))
    print ''

    # ---------------------------------------------------------------------------------
    # Print all vibr. modes?
    # ---------------------------------------------------------------------------------
    number_of_modes = len(eig)
    mode_dir = calculator.path + '/modes'
    if not os.path.isdir(mode_dir):
        os.mkdir(mode_dir)
    else:
        print 'Removing directory tree: %s\n' % mode_dir
        shutil.rmtree(mode_dir)
        os.mkdir(mode_dir)

    for i in range(atoms.GetGlobalDof(), number_of_modes):
        fname = mode_dir + '/vibr_mode_' + str(i) + '.out'
        WriteModeToFile(fname, w[:, i])

    negative_eig = np.sign(eig)
    countneg = (negative_eig == -1).sum()

    if countneg > 0:
        print('%2i negative mode(s) found' % countneg),
        if countneg == 1:  # compute crossover temperature
            Tc = ComputeTc(eig[lowest_index])
            print("The cross-over temperature is %6.2f K\n" % Tc)
            with open(basename + ".imaginary_mode","w") as f:
                for i in range(len(full_w)):
                    f.write("% 6.2lf \n" % full_w[i])

    for i in range(atoms.GetGlobalDof(), len(eig)):
        full_w = np.zeros(shape=(atoms.GetNDim(),1))
        full_w[atoms.GetMoveableAtoms()] = np.reshape(w[:,i], (atoms.GetNDof(),1))
        GenerateVibrTrajectory(mode_dir + '/' + basename + "_vibtrj_" + str(i) + ".xyz",
                               atoms.GetNDim(), atoms.GetCoords(), atoms.GetSymbols(), full_w, npts=npoints,
                               A=magn_displ)

    print('All %i vibrational modes written to path: %s' % (number_of_modes-atoms.GetGlobalDof(), mode_dir))
    with open(basename + '.eigs', 'w') as f:
        for i in range(len(eig)):
            f.write('%6.2f %6.2f \n' % (
                eig[i], np.sqrt(np.abs(eig[i])) / (2.0 * np.pi * KNARRsettings.time_unit * KNARRsettings.c)))

    # Execution done
    PrintJobDone('Freq. job', time.time() - start_t)

    return eig, w
