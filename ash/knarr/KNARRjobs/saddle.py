import numpy as np
import time
import KNARRsettings

from KNARRio.system_print import PrintJob, PrintCallBack, PrintJobDone, PrintConverged, PrintMaxIter
from KNARRio.output_print import PrintConfiguration
from KNARRio.io import WriteCon, WriteXYZ, WriteSingleImageTraj, WriteForcesFile, \
    WriteEnergyFile, ReadModeFromFile, WriteModeToFile
from KNARRatom.utilities import RMS
from KNARRoptimization.vpo import GlobalVPO, AndriVPO
from KNARRoptimization.fire import GetFIREParam, GlobalFIRE, EulerStep
from KNARRoptimization.lbfgs import LBFGSStep, LBFGSUpdate
from KNARRoptimization.utilities import IsConverged, GlobalScaleStepByMax, TakeFDStepWithFunction
from KNARRjobs.utilities import ComputeEffectiveMMFForce, GetMinimumModeLanczos


def DoSaddle(atoms, calculator, optimizer, parameters):
    # ---------------------------------------------------------------------------------
    # Initialize
    # ---------------------------------------------------------------------------------
    PrintJob('Saddle point search')
    # PrintCallBack('saddlejob', calculator, atoms, optimizer)

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
    reset_on_scaling = optimizer["RESTART_ON_SCALING"]
    lbfgs_damping = optimizer["LBFGS_DAMP"]

    # ---------------------------------------------------------------------------------
    # Read saddle parameters
    # ---------------------------------------------------------------------------------
    method_string = parameters["SADDLE_METHOD"].upper()
    if method_string == "MMF":
        method_type = 0
    elif method_string == "NEWTON":
        method_type = 1
    else:
        raise NotImplementedError()

    mmf_type_string = parameters["MMF_TYPE"].upper()
    if mmf_type_string == "LANCZOS":
        mmf_type = 0
    elif mmf_type_string == "DIMER":
        mmf_type = 1
    else:
        raise TypeError("Unknown choice for minimum mode following")

    l_maxiter = parameters["L_NMAX"]
    l_jumpiter = parameters["L_JUMP"]
    l_fdstep = parameters["L_FDSTEP"]
    l_tol = parameters["L_TOL"]
    l_guess = parameters["L_GUESS"]

    compute_hessian_every = parameters["COMPUTE_HESSIAN_EVERY"]
    damping_factor = parameters["DAMPING"]
    # ---------------------------------------------------------------------------------
    # Find Saddle
    # ---------------------------------------------------------------------------------
    if method_type == 0:
        if mmf_type == 0:
            saddle_energy, omega, eig = LanczosMMF(calculator, atoms, opttype, tol_max_force=tol_max_force,
                                                   tol_rms_force=tol_rms_force, maxiter=maxiter, time_step=time_step,
                                                   max_move=max_move, lbfgs_memory=lbfgs_memory,
                                                   lbfgs_damping = lbfgs_damping,
                                                   fd_step=fd_step, reset_on_scaling=reset_on_scaling,
                                                   l_maxiter=l_maxiter, l_jumpiter=l_jumpiter, l_fdstep=l_fdstep,
                                                   l_tol=l_tol, l_guess=l_guess)
        elif mmf_type == 1:
            DimerMMF()

    elif method_type == 1:
        saddle_energy, omega, eig = NewtonMethod(calculator, atoms, compute_hessian_every=compute_hessian_every,
                                                 tol_max_force=tol_max_force, tol_rms_force=tol_rms_force,
                                                 maxiter=maxiter, damping=damping_factor,
                                                 max_move=max_move, reset_on_scaling=reset_on_scaling)
    else:
        raise IOError("Unknown saddle point method")
    OK = 0
    # ---------------------------------------------------------------------------------
    # Job finished
    # ---------------------------------------------------------------------------------
    print('')
    if OK == 0:
        PrintConfiguration("Saddle Configuration", atoms.GetNDim(), atoms.GetNDof(), atoms.GetR(),
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
        WriteModeToFile(basename + '.mode', omega)
    else:
        print('Unable to converge to an energy minimum!')
        if KNARRsettings.extension == 0:
            WriteXYZ("last_iteration.xyz", atoms.GetNDim(), atoms.GetCoords(), atoms.GetSymbols(),
                     energy=atoms.GetEnergy())
        else:
            WriteCon("last_iteration.con", 1, atoms.GetCoords(), atoms.GetSymbols(),
                     atoms.GetCell(), atoms.GetConstraints())
            WriteModeToFile(basename + '_restart.mode', omega)

    PrintJobDone('Saddle job', time.time() - start_t)

    return atoms.GetCoords()


def LanczosMMF(calculator, atoms, opttype=0, tol_max_force=0.026,
               tol_rms_force=0.013, maxiter=300, time_step=0.1,
               max_move=0.1, lbfgs_memory=20, lbfgs_damping = 1.0,
               fd_step=0.01, reset_on_scaling=True,
               l_maxiter=30, l_jumpiter=5, l_fdstep=0.0001,
               l_tol=0.001, l_guess=None, omega=None):
    calculator.Compute(atoms)
    atoms.UpdateR()
    atoms.UpdateF()
    print('Energy of initial structure: % 6.4f %s' % (atoms.GetEnergy(), KNARRsettings.energystring))
    was_scaled = False
    reset_opt = False
    sk = []
    yk = []
    rhok = []
    basename = atoms.GetOutputFile()
    fire_param = GetFIREParam(time_step)

    if omega is not None:
        assert omega is type(np.ndarray)
        assert len(omega) == atoms.GetNDof() * atoms.GetNim()
    else:
        if l_guess is not None:
            omega = ReadModeFromFile(l_guess)
        else:
            import random
            random.seed(KNARRsettings.seed)
            omega = []
            for i in range(atoms.GetNDof() * atoms.GetNim()):
                omega.append(random.uniform(-1, 1))
            omega = np.asarray(omega)

    eigvalue, omega, l_it = GetMinimumModeLanczos(calculator, atoms, omega,
                                                  l_maxiter=l_maxiter, l_fdstep=l_fdstep, l_tol=l_tol)
    print '\nStarting Lanczos minimum-mode following:'
    print(' %3ls %9ls   %9ls %8ls  %8ls   %8ls' % ('it', 'Energy', 'dEnergy', 'RMSF', 'MAXF', '|step|'))

    print('Initial eigenvalue: % 6.4f' % (eigvalue))
    if eigvalue > 0.0:
        print('** Warning: lowest eigenvalue is positive. May need a better initial guess.')

    if l_it >= l_maxiter - 1:
        print('** Warning: Lanczos reached maximum number of iterations')

    for it in range(maxiter):

        calculator.Compute(atoms)
        WriteSingleImageTraj(basename + '_saddlesearch.xyz', atoms.GetNDim(), atoms.GetCoords(),
                             atoms.GetSymbols(), atoms.GetEnergy())

        atoms.UpdateR()
        atoms.UpdateF()

        max_force = np.max(abs(atoms.GetF()))
        rms_force = RMS(atoms.GetNDof(), atoms.GetF())
        deltaE = atoms.GetEnergy() - atoms.GetOldEnergy()

        converged = IsConverged(it, maxiter, tol_rms_force, tol_max_force,
                                max_force, rms_force)

        if (it % l_jumpiter == 0) and it > 0:
            eigvalue, omega, l_it = GetMinimumModeLanczos(calculator, atoms, omega,
                                                          l_maxiter=l_maxiter, l_fdstep=l_fdstep, l_tol=l_tol)
            if KNARRsettings.printlevel > 0:
                print '    Lanczos converged in %i iterations with an eigenvalue of % 6.4f' % (l_it, eigvalue)

        feff = ComputeEffectiveMMFForce(atoms.GetF(), eigvalue, omega)

        # ==========================
        # Check for convergence
        # ==========================
        if converged == 0 and eigvalue < 0.0 and it > 0:
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

            step, velo = GlobalVPO(feff, atoms.GetV(), time_step)
            step, was_scaled = GlobalScaleStepByMax(step, max_move)
            atoms.SetV(velo)

        elif opttype == 2:
            # Andris Velocity projection optimization
            if it == 0 or reset_opt:
                reset_opt = False
                atoms.ZeroV()
            if was_scaled:
                time_step *= 0.95

            step, velo = AndriVPO(atoms.GetNDof(), feff, atoms.GetV(), time_step)
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

            velo, time_step, fire_param = GlobalFIRE(feff, atoms.GetV(), time_step, fire_param)
            atoms.SetV(velo)
            step, velo = EulerStep(atoms.GetV(), feff, time_step)
            step, was_scaled = GlobalScaleStepByMax(step, max_move)
            atoms.SetV(velo)

        elif opttype == 5:
            # L-BFGS
            if it == 0 or reset_opt:
                reset_opt = False
                sk = []
                yk = []
                rhok = []
                keepf = feff.copy()
                keepr = atoms.GetR().copy()
                y = lambda x: ComputeEffectiveMMFForce(x, eigvalue, omega)
                step = TakeFDStepWithFunction(calculator, atoms, fd_step, feff, y)
                step = GlobalScaleStepByMax(step, max_move)[0]
            else:
                sk, yk, rhok = LBFGSUpdate(atoms.GetR(), keepr, feff, keepf,
                                           sk, yk, rhok, lbfgs_memory)
                keepf = feff.copy()
                keepr = atoms.GetR().copy()
                step, negativecurv = LBFGSStep(feff, sk, yk, rhok)
                step, was_scaled = GlobalScaleStepByMax(step, max_move)
                step *= lbfgs_damping

                if negativecurv:
                    reset_opt = True

        else:
            raise RuntimeError(
                "Choosen optimization method  %s is not available in saddle point optimization." % opttype)

        if reset_on_scaling and was_scaled:
            reset_opt = True

        # Take step
        atoms.SetR(atoms.GetR() + step)

        print('%3li % 4.6lf % 4.6lf % 4.6lf % 4.6lf % 4.6lf' %
              (it, atoms.GetEnergy(), deltaE, rms_force, max_force, np.linalg.norm(step)))

        atoms.UpdateCoords()
        atoms.MIC()

    return atoms.GetEnergy(), omega, eigvalue


def DimerMMF():
    return None


def NewtonMethod(calculator, atoms, compute_hessian_every=1,
                 tol_max_force=0.026, tol_rms_force=0.013,
                 maxiter=100, damping=1.0,
                 max_move=0.1, reset_on_scaling=True):

    #raise RuntimeError("Not ready")
    basename = atoms.GetOutputFile()
    reset_opt = False
    print '\nStarting Newton-Raphson saddle point search:'
    print(' %3ls %9ls   %9ls %8ls  %8ls   %8ls' % ('it', 'Energy', 'dEnergy', 'RMSF', 'MAXF', '|step|'))
    for it in range(maxiter):

        calculator.Compute(atoms)
        WriteSingleImageTraj(basename + '_saddle_search.xyz', atoms.GetNDim(), atoms.GetCoords(),
                             atoms.GetSymbols(), atoms.GetEnergy())
        atoms.UpdateR()
        atoms.UpdateF()

        current_forces = atoms.GetF()

        max_force = np.max(abs(current_forces))
        rms_force = RMS(atoms.GetNDof(), current_forces)
        deltaE = atoms.GetEnergy() - atoms.GetOldEnergy()

        converged = IsConverged(it, maxiter, tol_rms_force, tol_max_force,
                                max_force, rms_force)

        if converged == 0 and it > 0:
            PrintConverged(it, atoms.GetFC())
            print 'Last iteration:'
            print('%3li %4.6lf %4.6lf %4.6lf %4.6lf %4.4lf' %
                  (it, atoms.GetEnergy(), deltaE, rms_force, max_force, np.linalg.norm(step)))
            break
        elif converged == 1:
            PrintMaxIter(maxiter)
            break

        if (it % compute_hessian_every == 0 or it == 0 or reset_opt):
            reset_opt = False
            # compute exact hessian
            calculator.Compute(atoms, computeHessian=True)
            hessian = atoms.GetHessian()[:, :, 0]
            if atoms.IsTwoDee():
                hessian = hessian[0:2,0:2]
        else:
            # update hessian via bofill
            exit()

        if atoms.IsTwoDee():
            current_forces = current_forces[0:2]

        invH = np.linalg.inv(np.matrix(hessian))
        step = -damping * np.dot(invH, current_forces)
        step, was_scaled = GlobalScaleStepByMax(step, max_move)

        if reset_on_scaling and was_scaled:
            reset_opt = True

        if atoms.IsTwoDee():
            step_tmp = step.copy()
            step = np.zeros(shape=(3,1))
            step[0] = step_tmp[0]
            step[1] = step_tmp[1]

        # Take step
        atoms.SetR(atoms.GetR() + step)

        print('%3li % 4.6lf % 4.6lf % 4.6lf % 4.6lf % 4.6lf' %
              (it, atoms.GetEnergy(), deltaE, rms_force, max_force, np.linalg.norm(step)))

        atoms.UpdateCoords()
        atoms.MIC()

    return
