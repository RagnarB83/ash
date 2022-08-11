import os
import numpy as np
import time
import KNARRsettings

from KNARRio.system_print import PrintDivider, PrintJob, PrintCallBack, PrintJobDone, \
    PrintConverged, PrintMaxIter
from KNARRio.output_print import PrintAtomMatrix
from KNARRio.io import WriteCon, WriteXYZ, WriteTraj, WritePath, WriteEnergyFile
from KNARRjobs.utilities import ComputeLengthOfPath, ComputeEffectiveNEBForce, \
    PiecewiseCubicEnergyInterpolation, GetTangent, GenerateVibrTrajectory, GetTangent
from KNARRatom.utilities import RMS, MakeReparametrization, MakeReparametrizationWithCI, GenerateNewPath
from KNARRoptimization.vpo import GlobalVPO, LocalVPO, AndriVPO
from KNARRoptimization.fire import GetFIREParam, GlobalFIRE, EulerStep
from KNARRoptimization.lbfgs import LBFGSStep, LBFGSUpdate
from KNARRoptimization.utilities import GlobalScaleStepByMax, LocalScaleStepByMax, TakeFDStepWithFunction
from KNARRjobs.utilities import AutoZoom


# Author: Vilhjalmur Asgeirsson, 2019

def DoNEB(path, calculator, neb, optimizer, second_run=False):
    if second_run:
        PrintJob("Secondary Nudged Elastic Band Method")
    else:
        PrintJob('Nudged Elastic Band Method')
    # PrintCallBack('Linear', path)

    start_t = time.time()
    basename = path.GetOutputFile()
    if second_run == True:
        basename = basename + '_zoom'
    # ---------------------------------------------------------------------------------
    # Read optimizer input parameters
    # ---------------------------------------------------------------------------------
    method_string = optimizer["OPTIM_METHOD"].upper()
    if method_string in KNARRsettings.optimizer_types:
        opttype = KNARRsettings.optimizer_types[method_string]
    else:
        raise NotImplementedError("%s is not implemented" % method_string)
    maxiter = optimizer["MAX_ITER"]
    time_step = optimizer["TIME_STEP"]
    max_move = optimizer["MAX_MOVE"]
    fd_step = optimizer["FD_STEP"]
    lbfgs_memory = optimizer["LBFGS_MEMORY"]
    linesearch = optimizer["LINESEARCH"]
    reset_on_scaling = optimizer["RESTART_ON_SCALING"]
    lbfgs_damping = optimizer["LBFGS_DAMP"]

    if linesearch:
        raise NotImplementedError("Linesearch is not ready")

    # ---------------------------------------------------------------------------------
    # Read neb input parameters
    # --------------------------------------------------------------------------------
    doci = neb["CLIMBING"]
    restart_on_ci = neb["RESTART_OPT_ON_CI"]

    tangent_type_string = neb["TANGENT"].upper()
    if tangent_type_string == "ORIGINAL":
        tangent_type = 0
    elif tangent_type_string == "IMPROVED":
        tangent_type = 1
    else:
        raise TypeError("Unknown tangent type")

    spring_type_string = neb["SPRINGTYPE"].upper()
    if spring_type_string == "ORIGINAL":
        spring_type = 0
    elif spring_type_string == "DISTANCE":
        spring_type = 1
    elif spring_type_string == "IDEAL":
        spring_type = 2
    elif spring_type_string == "ELASTIC_BAND":
        spring_type = 3
        tangent_type = 0 #elastic band uses original tangent
    else:
        raise TypeError("Unknown spring type")

    perp_springtype_string = neb["PERP_SPRINGTYPE"].upper()
    if perp_springtype_string == "" or perp_springtype_string == "NO":
        perp_springtype = 0
    elif perp_springtype_string == "COS":
        perp_springtype = 1
    elif perp_springtype_string == "TAN":
        perp_springtype = 2
    elif perp_springtype_string == "COSTAN":
        perp_springtype = 3
    elif perp_springtype_string == "DNEB":
        perp_springtype = 4
    else:
        raise TypeError("Unknown perp. spring type")

    energy_weighted = neb["ENERGY_WEIGHTED"]
    springconst = neb["SPRINGCONST"]
    springconst2 = neb["SPRINGCONST2"]
    min_rmsd = neb["MIN_RMSD"]
    remove_extern_force = neb["REMOVE_EXTERNAL_FORCE"]
    free_end = neb["FREE_END"]

    free_end_type_str = neb["FREE_END_TYPE"].upper()
    if free_end_type_str == "PERP":
        free_end_type = 0
    elif free_end_type_str == "CONTOUR":
        free_end_type = 1
    elif free_end_type_str == "FULL":
        free_end_type = 2
    else:
        raise TypeError("Unknown free end type")

    free_end_energy1 = neb["FREE_END_ENERGY"]
    free_end_energy2 = neb["FREE_END_ENERGY2"]
    free_end_kappa = neb["FREE_END_KAPPA"]

    conv_type_string = neb["CONV_TYPE"].upper()
    if conv_type_string == "ALL":
        conv_type = 0
    elif conv_type_string == "CI_ONLY" or conv_type_string == "CIONLY":
        conv_type = 1
    else:
        raise TypeError("Unknown option for convergence monitoring")

    tol_scale = neb["TOL_SCALE"]
    tol_max_fci = neb["TOL_MAX_FCI"]
    tol_rms_fci = neb["TOL_RMS_FCI"]
    tol_max_f = neb["TOL_MAX_F"]
    tol_rms_f = neb["TOL_RMS_F"]
    tol_turn_on_ci = neb["TOL_TURN_ON_CI"]

    if tol_scale is not None and doci:

        if tol_scale <= 1:
            raise ValueError("Tol scale is too low")

        tol_max_f = tol_scale * tol_max_fci
        tol_rms_f = tol_scale * tol_rms_fci

    reparamfreq = neb["REPARAM"]
    tol_reparam = neb["TOL_REPARAM"]
    do_reparam = False
    lbfgs_reparam_on_restart = neb["LBFGS_REPARAM_ON_RESTART"]

    interp_type_str = neb["INTERPOLATION_TYPE"].upper()
    if interp_type_str == "LINEAR":
        interp_type = 0
    elif interp_type_str == "CUBIC":
        interp_type = 1
    else:
        raise TypeError("Unknown reparametrization type")

    autozoom = neb["AUTO_ZOOM"]
    zoom_offset = neb["ZOOM_OFFSET"]
    zoom_alpha = neb["ZOOM_ALPHA"]

    # ---------------------------------------------------------------------------------
    # Make checks for stupid input
    # --------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------
    # Initialize NEB
    # --------------------------------------------------------------------------------
    maxf_all = 100.0
    step = 0.0
    ci = -1
    reset_opt = False
    reparam_only_once = True
    startci = False
    was_scaled = False
    converged = False
    stop_neb = False
    fire_param = GetFIREParam(time_step)
    path.PrintPath('Initial path:')

    # --------------------------------------------------------------------------------
    # Initialize second run of neb
    # --------------------------------------------------------------------------------
    if second_run:
        # Get Zoom Region
        CI = np.argmax(path.GetEnergy())
        Ereactant = path.GetEnergy()[0]
        Eproduct = path.GetEnergy()[-1]
        if autozoom:
            print('Searching for automatic zoom interval:')
            s = ComputeLengthOfPath(path.GetNDofIm(), path.GetNim(), path.GetR(), path.GetPBC(), path.GetCell())
            fneb, freal_perp, freal_paral = \
                ComputeEffectiveNEBForce(path.GetF(), 0, path.GetNDofIm(), path.GetNim(), ci, path.GetR(),
                                         path.GetEnergy(),
                                         tangent_type, spring_type, energy_weighted, springconst, springconst2,
                                         perp_springtype,
                                         free_end, free_end_type, free_end_energy1, free_end_energy2, free_end_kappa,
                                         startci, remove_extern_force,
                                         path.GetPBC(), path.GetCell())

            OK, A, B = AutoZoom(path.GetNim(), s, path.GetEnergy(), freal_paral, CI, zoom_alpha)

            if OK == 1:
                print('==> AutoZoom was succesful')
            else:
                print('AutoZoom unable to find a good guess:')
                print('Reverting back to manual selection with offset %i' % zoom_offset)
                A = CI - zoom_offset
                B = CI + zoom_offset

        else:
            A = CI - zoom_offset
            B = CI + zoom_offset

        if A < 0:
            A = 0

        if B > path.GetNim() + 1:
            B = path.GetNim() + 1

        if A == 0 and B == path.GetNim() + 1:
            raise RuntimeError("Too large zoom interval chosen, "
                               "nothing will be gained from the zoom. Please inspect the calculation")

        print('Zoom region is defined by images: {} -to- {}'.format(A, B))
        coords = path.GetCoords()[A * path.GetNDimIm():(B + 1) * path.GetNDimIm()]
        energy = path.GetEnergy()[A:B + 1]
        znim = len(energy)
        ztang = GetTangent(path.GetNDimIm(), znim, coords, energy, tangent_type,
                           path.GetPBC(), path.GetCell())
        s = ComputeLengthOfPath(path.GetNDimIm(), znim, coords,
                                path.GetPBC(), path.GetCell())

        # if no free-end was used in first run - we need to reduce number of images by 2
        if not free_end:
            nim = path.GetNim() - 2

        free_end = True
        free_end_type = 0

        path.nim = nim
        path.SetNDim(path.GetNim() * path.GetNDimIm())
        path.SetNDof(path.GetNim() * path.GetNDofIm())
        path.energy = None
        path.energy0 = None
        constraints = path.GetConstraints()
        path.SetConstraints(np.array(list(constraints[0:path.GetNDimIm()]) * path.GetNim()))
        path.SetMoveableAtoms()
        new_coords = GenerateNewPath(interp_type, path.GetNDimIm(), znim, path.GetNim(), s, coords, ztang)
        path.SetCoords(new_coords)

        free_end = True
        free_end_type = 0

    # ---------------------------------------------------------------------------------
    # Get end points
    # --------------------------------------------------------------------------------
    if not free_end:
        #print("path:", path)
        #print(path.__dict__)
        #print("path.GetNim():", path.GetNim())
        calculator.Compute(path, list_to_compute=[0, path.GetNim() - 1])
        print('Energy of end points: ')
        #print('  Reactant: % 6.6f %s' % (path.GetEnergy()[0], KNARRsettings.energystring))
        #print('  Product : % 6.6f %s' % (path.GetEnergy()[path.GetNim() - 1], KNARRsettings.energystring))
        print('  Reactant: % 6.6f %s' % (0.03674930495120813*path.GetEnergy()[0], 'Eh'))
        print('  Product : % 6.6f %s' % (0.03674930495120813*path.GetEnergy()[path.GetNim() - 1], 'Eh'))

        path.UpdateF()
        maxf_reactant = np.max(abs(path.GetF()[0:path.GetNDofIm()]))
        maxf_product = np.max(abs(path.GetF()[(path.GetNim() - 1) * path.GetNDofIm()::]))
        if maxf_reactant > tol_max_f:
            print('**Warning: atom forces on reactant state may be too large. Max(|F|): %6.4f %s' %
                  (maxf_reactant, KNARRsettings.forcestring))
        if maxf_product > tol_max_f:
            print('**Warning: atom forces on product  state may be too large. Max(|F|): %6.4f %s' %
                  (maxf_product, KNARRsettings.forcestring))

        Ereactant = path.GetEnergy()[0]
        Eproduct = path.GetEnergy()[-1]

    if second_run:
        print('Reference energy: %6.6f' % Ereactant)

    # ---------------------------------------------------------------------------------
    # Start NEB optimization
    # --------------------------------------------------------------------------------
    print('\nStarting nudged elastic band optimization:')
    #print(' %4ls %4s  %9ls %5ls %7ls %9ls %8ls' %
    #      ('it', 'dS', 'Energy', 'HEI', 'RMSF', 'MaxF', 'step'))

    for it in range(maxiter):
        # =======================================================
        # Reparametrization and minimization of rmsd
        # =======================================================
        if it > 0:
            if maxf_all < tol_reparam and reparam_only_once:
                do_reparam = True
                reparam_only_once = False

        if (reparamfreq > 0 or do_reparam):
            if ((it % reparamfreq == 0 or do_reparam) and (it > 0)):
                s = ComputeLengthOfPath(path.GetNDim(), path.GetNim(), path.GetCoords())
                tang = GetTangent(path.GetNDim(), path.GetNim(),
                                  path.GetCoords(), path.GetEnergy(), tangent_type)
                if (startci):
                    newr = MakeReparametrizationWithCI(path.GetNDim(), path.GetNim(), ci, s,
                                                       path.GetCoords(), tang, interp_type)
                else:
                    newr = MakeReparametrization(path.GetNDim(), path.GetNim(), s,
                                                 path.GetCoords(), tang, interp_type)
                path.SetCoords(newr)
                WritePath(basename + "_reparam_" + str(it) + "_.xyz", path.GetNDimIm(), path.GetNim(),
                          path.GetCoords(), path.GetSymbols())
                print('\nReparametrization of path\n')
                reset_opt = True
                do_reparam = False
        #RB: THIS CAN BE PROBLEMATIC to do in every step. Product geometry keeps changing. Bug?
        if min_rmsd and not path.IsConstrained() and not path.IsTwoDee():
            #RB change. Only do once
            if it == 0:
                print("Minimizing RMSD in this step")
                path.MinRMSD()
        # =======================================================
        # Compute energy and gradient
        # =======================================================
        if free_end:
            #calculator.Compute(path)
            #RB change:
            calculator.Compute(path, list_to_compute=range(0, path.GetNim()))
            if it == 0 and not second_run:
                Ereactant = path.GetEnergy()[0]
                Eproduct = path.GetEnergy()[-1]
        else:
            calculator.Compute(path, list_to_compute=range(1, path.GetNim() - 1))
            path.SetEnergy(Ereactant, x=0)
            path.SetEnergy(Eproduct, x=path.GetNim() - 1)


        path.UpdateR()
        path.UpdateF()
        # =======================================================
        # Write trajectory files
        # =======================================================

        WriteTraj(basename + "_optimization.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())
        WritePath(basename + "_current.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())

        # =======================================================
        # Make checks - angle, possible intermediate images, etc.
        # =======================================================

        # =======================================================
        # Compute effective NEB force
        # =======================================================
        s = ComputeLengthOfPath(path.GetNDofIm(), path.GetNim(), path.GetR(),
                                path.GetPBC(), path.GetCell())

        fneb, freal_perp, freal_paral = \
            ComputeEffectiveNEBForce(path.GetF(), it, path.GetNDofIm(), path.GetNim(), ci, path.GetR(),
                                     path.GetEnergy(),
                                     tangent_type, spring_type, energy_weighted, springconst, springconst2,
                                     perp_springtype,
                                     free_end, free_end_type, free_end_energy1, free_end_energy2, free_end_kappa,
                                     startci, remove_extern_force,
                                     path.GetPBC(), path.GetCell(), path.IsTwoDee())


        # =======================================================
        # Compute maximum abs. and RMS force
        # =======================================================
        if startci:
            maxf_all = np.max(abs(freal_perp))
            freal_noci = np.concatenate((freal_perp[0:ci * path.GetNDofIm()],
                                         freal_perp[(ci + 1) * path.GetNDofIm()::]), axis=0)
            rmsf_noci = RMS(path.GetNDof() - path.GetNDofIm(), freal_noci)
            maxf_noci = np.max(abs(freal_noci))
            rmsf_ci = RMS(path.GetNDofIm(), path.GetF()[ci * path.GetNDofIm():(ci + 1) * path.GetNDofIm()])
            maxf_ci = np.max(abs(path.GetF()[ci * path.GetNDofIm():(ci + 1) * path.GetNDofIm()]))
            print("CI is active")
            #print(f"Current freal_noci: {freal_noci}")
            #print(f"Current rmsf_noci: {rmsf_noci} (tol_rms_fci: {tol_rms_fci})")
            #print(f"Current maxf_noci: {maxf_noci} (tol_max_fci: {tol_max_fci})")
            #print(f"Current rmsf_ci: {rmsf_ci} (tol_rms_f: {tol_rms_f})")
            #print(f"Current maxf_ci: {maxf_ci} (tol_max_f: {tol_max_f})")

        else:
            rmsf = RMS(path.GetNDof(), freal_perp)
            maxf = np.max(abs(freal_perp))
            #print("CI is NOT active")
            #print(' %4ls %4s  %9ls %5ls %7ls %9ls %8ls' %('it', 'dS', 'Energy', 'HEI', 'RMSF', 'MaxF', 'step'))
            #print(f"Current RMS-F: {rmsf} (tol_rms_f: {tol_rms_f})")
            #print(f"Current Max-F: {maxf} (tol_max_f: {tol_max_f})")

        # =======================================================
        # Output print
        # =======================================================
        if startci:
            print("HEI: Highest energy image")
            print("ΔE: Relative energy of HIE in kcal/mol (w.r.t. image 0)")
            print("Forces in eV/Ang.")
            print("RMSF/MaxF: RMS/Max force on all images.")
            print("RMSF_CI/MaxF_CI: RMS/Max force on climbing image.")
            print("-"*80)
            print('%4ls %6s %8ls %5ls %8ls %8ls %8ls %8ls %8ls' %
                    ('it', 'dS', 'ΔE', 'CI', 'RMSF', 'MaxF', 'RMSF_CI', 'MaxF_CI', 'step'))
            print(f"Thresholds:                {tol_rms_f:8.4f} {tol_max_f:8.4f} {tol_rms_fci:8.4f} {tol_max_fci:8.4f}")
            print ("%4i %6.2lf %8.3lf %5li %8.4lf %8.4lf %8.4lf %8.4lf %8.4lf"
                   % (it, s[-1], 23.060541945329334*(path.GetEnergy()[ci] - Ereactant), ci, rmsf_noci,
                      maxf_noci, rmsf_ci, maxf_ci, np.max(abs(step))))
            print("-"*80)
        else:
            hei = np.argmax(path.GetEnergy())
            print("HEI: Highest energy image")
            print("ΔE: Relative energy of HIE in kcal/mol (w.r.t. image 0)")
            print("Forces in eV/Ang")
            print("RMSF/MaxF: RMS/Max force on all images.")
            print("-"*70)
            print('%4ls %6s %10ls %5ls %9ls %9ls %9ls' %
                ('it', 'dS', 'ΔE', 'HEI', 'RMSF', 'MaxF', 'step'))
            print(f"Switch-on CI:{tol_turn_on_ci:>36.4f}")
            print ("%4i %6.2lf %10.6lf %5li %9.4lf  %9.4lf %9.4lf"
                   % (it, s[-1], 23.060541945329334*(path.GetEnergy()[hei] - Ereactant), hei, rmsf, maxf, np.max(abs(step))))
            print("-"*70)
        PiecewiseCubicEnergyInterpolation(basename + ".interp", path.GetNim(), s, path.GetEnergy(), freal_paral, it)

        # =======================================================
        # Check if calculation is converged...
        # =======================================================
        if doci:
            #print("Now checking if calculation is converged. conv_type:", conv_type)
            if conv_type == 1:
                #print("convtype 1")
                if startci:
                    if tol_max_fci > maxf_ci and tol_rms_fci > rmsf_ci:
                        converged = True
                else:
                    if tol_max_fci > maxf and tol_rms_f > rmsf:
                        converged = True
            else:
                #print("convtype diff")
                #print("startci:", startci)
                if startci:
                    if (tol_max_fci > maxf_ci and tol_rms_fci > rmsf_ci) and \
                            (tol_max_f > maxf_noci and tol_rms_f > rmsf_noci):
                        converged = True
                else:
                    if (tol_max_fci > maxf and tol_rms_fci > rmsf):
                        print("Now signalling convergence")
                        converged = True
        else:
            if (tol_max_f > maxf and tol_rms_f > rmsf):
                converged = True

        if it == maxiter - 1:
            stop_neb = True
            converged = False

        if converged or stop_neb:
            break

        # =======================================================
        # Climbing image block
        # =======================================================
        if (maxf < tol_turn_on_ci or tol_turn_on_ci == 0.0):
            if not startci and doci:
                checkmax = np.argmax(path.GetEnergy())
                if checkmax != 0 and checkmax != path.GetNim() - 1:
                    startci = True
                    #Rmod:
                    calculator.ISCION = True
                    ci = checkmax
                    if restart_on_ci:
                        reset_opt = True
                    #print(f"maxf:{maxf} < tol_turn_on_ci:{tol_turn_on_ci}")
                    print ('        Starting climbing image as image %i.\n' % ci),

                    # print CI header here.
                    # print '%6s %4s %2s %10s %8s %13s %5s %11s  %11s %11s %13s' \
                    # % ("Optim.", "iter", "FC", "Objf", "dS", "maxE", "CI", "rmsF'(i)", "rmsF(CI)", "maxF'",
                    #   "maxF(CI)")
        if startci:
            if (it % 5) == 0:
                newci = np.argmax(path.GetEnergy())
                # cant allow it to be at the end points
                if newci != ci and newci != 0 and newci < path.GetNim() - 1:
                    ci = newci
                    if restart_on_ci:
                        reset_opt = True
        if startci != True:
            print("CI not switched on yet (MaxF treshold not reached).")
        # =======================================================
        # Optimization block
        # =======================================================
        if opttype == 0:
            # (Global) Velocity projection optimization
            if it == 0 or reset_opt:
                reset_opt = False
                path.ZeroV()
            if was_scaled:
                time_step = time_step * 0.95

            step, velo = GlobalVPO(fneb, path.GetV(), time_step)
            path.SetV(velo)
            step, was_scaled = GlobalScaleStepByMax(step, max_move)

        elif opttype == 1:
            # (Local) Velocity projection optimization
            if it == 0 or reset_opt:
                reset_opt = False
                path.ZeroV()
            if was_scaled:
                time_step = time_step * 0.95

            step, velo = LocalVPO(path.GetNDofIm(), path.GetNim(), fneb, path.GetV(), time_step)
            path.SetV(velo)
            step, was_scaled = LocalScaleStepByMax(path.GetNDofIm(), path.GetNim(), step, max_move)

        elif opttype == 2:
            # Andris Velocity projection optimization
            if it == 0 or reset_opt:
                reset_opt = False
                path.ZeroV()
            if was_scaled:
                time_step = time_step * 0.95

            step, velo = AndriVPO(path.GetNDof(), fneb, path.GetV(), time_step)
            path.SetV(velo)
            step, was_scaled = GlobalScaleStepByMax(step, max_move)

        elif opttype == 4:
            # FIRE
            if it == 0 or reset_opt:
                reset_opt = False
                fire_param = GetFIREParam(time_step)
                path.ZeroV()
            if was_scaled:
                time_step = time_step * 0.95

            velo, time_step, fire_param = GlobalFIRE(fneb, path.GetV(), time_step, fire_param)
            path.SetV(velo)
            step, velo = EulerStep(path.GetV(), fneb, time_step)
            path.SetV(velo)
            step, was_scaled = GlobalScaleStepByMax(step, max_move)

        elif opttype == 5:
            # L-BFGS
            if it == 0 or reset_opt:
                reset_opt = False
                was_scaled = False
                sk = []
                yk = []
                rhok = []
                keepf = fneb.copy()
                keepr = path.GetR().copy()
                y = lambda x: ComputeEffectiveNEBForce(x, it, path.GetNDofIm(), path.GetNim(), ci, path.GetR(),
                                                       path.GetEnergy(),
                                                       tangent_type, spring_type, energy_weighted, springconst,
                                                       springconst2,
                                                       perp_springtype, free_end, free_end_type, free_end_energy1,
                                                       free_end_energy2, free_end_kappa,
                                                       startci, remove_extern_force,
                                                       path.GetPBC(), path.GetCell())
                step = TakeFDStepWithFunction(calculator, path, fd_step, fneb, y)
                step = GlobalScaleStepByMax(step, max_move)[0]
            else:
                sk, yk, rhok = LBFGSUpdate(path.GetR(), keepr, fneb, keepf,
                                           sk, yk, rhok, lbfgs_memory)
                keepf = fneb.copy()
                keepr = path.GetR().copy()
                step, negativecurv = LBFGSStep(fneb, sk, yk, rhok)
                step *= lbfgs_damping

                if negativecurv:
                    reset_opt = True

                step, was_scaled = GlobalScaleStepByMax(step, max_move)

        else:
            raise RuntimeError(
                "Choosen optimization method  %s is not available in structural opt." % method_string)
        if reset_on_scaling and was_scaled:
            reset_opt = True
            if lbfgs_reparam_on_restart:
                do_reparam = True

        path.SetR(path.GetR() + step)
        path.UpdateCoords()

    # ---------------------------------------------------------------------------------
    # END NEB OPTIMIZATION - Write output
    # ---------------------------------------------------------------------------------
    CI = np.argmax(path.GetEnergy())



    if converged:
        PrintConverged(it+1, path.GetFC())
        PrintDivider()
        print('Summary:')
        PrintDivider()
        #print('%4ls %4ls %5ls %9ls %9ls' % ('Img.', 'dS', 'E(eV)', 'ΔE(kcal/mol)', 'MaxF'))
        #for i in range(path.GetNim()):
        #    print('% 2i % 6.2f % 6.5f % 6.4f % 6.6f' % (
        #        i, s[i], path.GetEnergy()[i], 23.060541945329334*(path.GetEnergy()[i] - Ereactant),
        #        np.max(abs(freal_perp[i * path.GetNDofIm():(i + 1) * path.GetNDofIm()]))))
        print('%4ls %6ls %12ls %12ls %12ls' % ('Img.', 'dS', 'E(Eh)', 'ΔE(kcal/mol)', 'MaxF(eV/Ang)'))
        for i in range(path.GetNim()):
            if i == ci:
                extra="CI"
            else: extra=""
            print('%4i %6.2f %12.5f %12.4f %12.6f %6s' % (
                i, s[i], 0.03674930495120813*path.GetEnergy()[i], 23.060541945329334*(path.GetEnergy()[i] - Ereactant),
                np.max(abs(freal_perp[i * path.GetNDofIm():(i + 1) * path.GetNDofIm()])),extra))

        WritePath(basename + "_MEP.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())

        if os.path.isfile(basename + "_current.xyz"):
            os.remove(basename + "_current.xyz")

        E_barrier=(path.GetEnergy()[CI][0] - path.GetEnergy()[0][0])*23.060541945329334
        print(f"\nBarrier energy: {E_barrier} kcal/mol")

        PrintAtomMatrix("\nSaddle point geometry (Å):", path.GetNDimIm(),
                        path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()],
                        path.GetSymbols())

        PrintAtomMatrix("Atomic forces(eV/Å):", path.GetNDofIm(),
                        path.GetForces()[CI * path.GetNDofIm():(CI + 1) * path.GetNDofIm()],
                        path.GetSymbols())

        tang = GetTangent(path.GetNDofIm(), path.GetNim(), path.GetR(), path.GetEnergy(),
                          tangent_type, pbc=path.GetPBC(), cell=path.GetCell())

        PrintAtomMatrix("Tangent to path:", path.GetNDofIm(),
                        tang[CI * path.GetNDofIm():(CI + 1) * path.GetNDofIm()],
                        path.GetSymbols())

        WriteEnergyFile(basename + '.energy', path.GetEnergy(), path.GetNim())

        imaginary_mode = np.zeros(shape=(path.GetNDimIm(), 1))
        imaginary_mode[path.GetMoveableAtoms()[0:path.GetNDofIm()]] = \
            tang[CI * path.GetNDofIm():(CI + 1) * path.GetNDofIm()]
        GenerateVibrTrajectory(basename + '_mode.xyz', path.GetNDimIm(),
                               path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()],
                               path.GetSymbols(), imaginary_mode, npts=40, A=0.3)

        if KNARRsettings.extension == 0:
            WriteXYZ(basename + "_saddle.xyz", path.GetNDimIm(),
                     path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()], path.GetSymbols(),
                     energy=path.GetEnergy()[CI])
        else:
            WriteCon(basename + "_saddle.con", 1,
                     path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()],
                     path.GetSymbols()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()],
                     path.GetCell(),
                     path.GetConstraints()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()])
    else:
        print("NEB is not yet converged. Doing next NEB iteration")
    if stop_neb:
        PrintMaxIter(maxiter)
        WritePath(basename + "_last_iter.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())

        if os.path.isfile(basename + "_current.xyz"):
            os.remove(basename + "_current.xyz")

    # Done.
    PrintJobDone('NEB job', time.time() - start_t)

    return path.GetEnergy()[CI], path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()]
