import numpy as np
import time

from KNARRio.system_print import PrintJob, PrintCallBack, PrintJobDone, PrintConverged, PrintMaxIter
from KNARRio.output_print import PrintConfiguration
from KNARRio.io import WriteCon, WriteXYZ, WriteSingleImageTraj, WriteForcesFile, WriteEnergyFile
from KNARRatom.utilities import RMS
from KNARRoptimization.vpo import GlobalVPO, AndriVPO
from KNARRoptimization.fire import GetFIREParam, GlobalFIRE, EulerStep
from KNARRoptimization.lbfgs import LBFGSStep, LBFGSUpdate
from KNARRoptimization.utilities import IsConverged, GlobalScaleStepByMax, TakeFDStep

import KNARRsettings

# Author: Vilhjalmur Asgeirsson, 2019

def DoOpt(atoms, calculator, optimizer):
    # ---------------------------------------------------------------------------------
    # Initialize
    # ---------------------------------------------------------------------------------
    PrintJob('Structural optimization')
    # PrintCallBack('optjob', calculator, atoms, optimizer)

    if not atoms.setup:
        raise RuntimeError("Atoms object is not properly initialized")
    if not calculator.setup:
        raise RuntimeError("Calculator is not properly initialized")

    atoms.PrintConfiguration('Input configuration:')
    start_t = time.time()
    basename = atoms.GetOutputFile()

    # ---------------------------------------------------------------------------------
    # Read optimizer parameters
    # ---------------------------------------------------------------------------------
    method_string = optimizer["OPTIM_METHOD"].upper()
    if method_string in KNARRsettings.optimizer_types:
        opttype = KNARRsettings.optimizer_types[method_string]
    else:
        raise NotImplementedError("%s is not implemented" % method_string)
    maxiter = optimizer["MAX_ITER"]
    tol_max_force = optimizer["TOL_MAX_FORCE"]
    tol_rms_force = optimizer["TOL_RMS_FORCE"]
    time_step = optimizer["TIME_STEP"]
    max_move = optimizer["MAX_MOVE"]
    lbfgs_memory = optimizer["LBFGS_MEMORY"]
    fd_step = optimizer["FD_STEP"]
    linesearch = optimizer["LINESEARCH"]
    reset_on_scaling = optimizer["RESTART_ON_SCALING"]
    lbfgs_damping = optimizer["LBFGS_DAMP"]

    if linesearch:
        raise NotImplementedError("Linesearch is not ready")
    # ---------------------------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------------------------

    # Init
    reset_opt = False
    converged = 1
    was_scaled = False
    fire_param = GetFIREParam(time_step)

    print '\nStarting structural optimization:'
    print(' %3ls %9ls   %9ls %8ls  %8ls   %8ls' % ('it', 'Energy', 'dEnergy', 'RMSF', 'MAXF', '|step|'))
    for it in range(maxiter):
        calculator.Compute(atoms)
        WriteSingleImageTraj('minimization.xyz', atoms.GetNDim(), atoms.GetCoords(),
                             atoms.GetSymbols(), atoms.GetEnergy())

        atoms.UpdateR()
        atoms.UpdateF()

        max_force = np.max(abs(atoms.GetF()))
        rms_force = RMS(atoms.GetNDof(), atoms.GetF())
        deltaE = atoms.GetEnergy() - atoms.GetOldEnergy()

        converged = IsConverged(it, maxiter, tol_rms_force, tol_max_force,
                                max_force, rms_force)

        # ==========================
        # Check for convergence
        # ==========================
        if converged == 0 and it > 0:
            PrintConverged(it, atoms.GetFC())
            print 'Last iteration:'
            print('%3li %4.6lf %4.6lf %4.6lf %4.6lf %4.4lf' %
                  (it, atoms.GetEnergy(), deltaE, rms_force, max_force, np.linalg.norm(step)))
            break
        elif converged == 1:
            PrintMaxIter(maxiter)
            break

        # ==========================
        # Get step
        # ==========================
        if opttype == 0 or opttype == 1:
            # Velocity projection optimization
            if it == 0 or reset_opt:
                reset_opt = False
                atoms.ZeroV()
            if was_scaled:
                time_step *= 0.95

            step, velo = GlobalVPO(atoms.GetF(), atoms.GetV(), time_step)
            atoms.SetV(velo)

        elif opttype == 2:
            # Andris Velocity projection optimization
            if it == 0 or reset_opt:
                reset_opt = False
                atoms.ZeroV()
            if was_scaled:
                time_step *= 0.95

            step, velo = AndriVPO(atoms.GetNDof(), atoms.GetF(), atoms.GetV(), time_step)
            atoms.SetV(velo)
            step, was_scaled = GlobalScaleStepByMax(step, max_move)

        elif opttype == 4:
            # FIRE
            if it == 0 or reset_opt:
                reset_opt = False
                fire_param = GetFIREParam(time_step)
                atoms.ZeroV()
            if was_scaled:
                time_step *= 0.95

            velo, time_step, fire_param = GlobalFIRE(atoms.GetF(), atoms.GetV(), time_step, fire_param)
            atoms.SetV(velo)
            step, velo = EulerStep(atoms.GetV(), atoms.GetF(), time_step)
            atoms.SetV(velo)

        elif opttype == 5:
            # L-BFGS
            if it == 0 or reset_opt:
                reset_opt = False
                sk = []
                yk = []
                rhok = []
                keepf = atoms.GetF().copy()
                keepr = atoms.GetR().copy()
                step = TakeFDStep(calculator, atoms, fd_step)
            else:
                sk, yk, rhok = LBFGSUpdate(atoms.GetR(), keepr, atoms.GetF(), keepf,
                                           sk, yk, rhok, lbfgs_memory)
                keepf = atoms.GetF().copy()
                keepr = atoms.GetR().copy()
                step, negativecurv = LBFGSStep(atoms.GetF(), sk, yk, rhok)
                step *= lbfgs_damping

                if negativecurv:
                    reset_opt = True

        else:
            raise RuntimeError("Choosen optimization method  %s is not available in structural opt." % method_string)

        # ===================================
        # Check if step needs to be scaled
        # ===================================
        step, was_scaled = GlobalScaleStepByMax(step, max_move)

        if reset_on_scaling and was_scaled:
            reset_opt = True

        # Take step
        atoms.SetR(atoms.GetR() + step)

        print('%3li % 4.6lf % 4.6lf % 4.6lf % 4.6lf % 4.6lf' %
              (it, atoms.GetEnergy(), deltaE, rms_force, max_force, np.linalg.norm(step)))

        atoms.UpdateCoords()
        atoms.MIC()

    # ---------------------------------------------------------------------------------
    # Job finished
    # ---------------------------------------------------------------------------------
    print('')
    if converged == 0:
        PrintConfiguration("Relaxed Configuration", atoms.GetNDim(), atoms.GetNDof(), atoms.GetCoords(),
                           atoms.GetConstraints(), atoms.GetSymbols(),
                           cell=atoms.GetCell(), pbc=atoms.GetPBC())

        print('Energy         : % 6.4lf %s' % (atoms.GetEnergy(), KNARRsettings.energystring))
        print('max|F|         : % 6.6lf %s' % (np.max(abs(atoms.GetF())), KNARRsettings.forcestring))

        if KNARRsettings.extension == 0:
            WriteXYZ(basename + ".xyz", atoms.GetNDim(), atoms.GetCoords(), atoms.GetSymbols(),
                     energy=atoms.GetEnergy())
        else:
            WriteCon(basename + ".con", 1, atoms.GetCoords(), atoms.GetSymbols(),
                     atoms.GetCell(), atoms.GetConstraints())

        WriteEnergyFile(basename + '.energy', atoms.GetEnergy())
        WriteForcesFile(basename + '.forces', atoms.GetNDim(), atoms.GetSymbols(), atoms.GetForces())

    else:
        print('Unable to converge to an energy minimum!')
        if KNARRsettings.extension == 0:
            WriteXYZ("last_iteration.xyz", atoms.GetNDim(), atoms.GetCoords(), atoms.GetSymbols(),
                     energy=atoms.GetEnergy())
        else:
            WriteCon("last_iteration.con", 1, atoms.GetCoords(), atoms.GetSymbols(),
                     atoms.GetCell(), atoms.GetConstraints())

    PrintJobDone('Opt. job', time.time() - start_t)

    return atoms.GetEnergy(), atoms.GetCoords()
