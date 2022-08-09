import os
import time
import numpy as np
import KNARRsettings

from KNARRio.system_print import PrintJob, PrintCallBack, PrintJobDone, PrintMaxIter, PrintConverged
from KNARRatom.utilities import MinimizeRotation, RMS3, RMS, MakeReparametrization
from KNARRjobs.utilities import PathLinearInterpol, PathLinearInterpolWithInsertion, AllImageDistances, \
    ComputeLengthOfPath
from KNARRcalculator.calculator import Calculator
from KNARRio.io import WritePath


# Author: Vilhjalmur Asgeirsson, 2019

def DoPathInterpolation(path, parameters):
    PrintJob('Path Generation')
    # PrintCallBack('Linear', path)

    start_t = time.time()
    basename = path.GetOutputFile()
    insertion = path.GetInsertionConfig()

    # ------------------------------------------------------------
    # Read input parameters
    # ------------------------------------------------------------
    if parameters["METHOD"].upper() == "DOUBLE":
        double_ended = True
        single_ended = False
    elif parameters["METHOD"].upper() == "SINGLE":
        single_ended = True
        double_ended = False
    else:
        raise IOError("Unknown method option for path generation")

    interpolation_string = parameters["INTERPOLATION"].upper()
    if interpolation_string == "LINEAR":
        type_of_interp = 0
    elif interpolation_string == "IDPP":
        type_of_interp = 1
    else:
        raise TypeError("Unknown interpolation type")
    # ------------------------------------------------------------
    # Double ended path generation
    # ------------------------------------------------------------
    if double_ended:
        # ================================================
        # Perform linear interpolation without insertion
        # ================================================
        if insertion is None:
            # =======================================
            # Minimize RMSD
            # =======================================
            if not path.IsConstrained() and not path.GetPBC() and not path.twodee:
                print('Minimization of RMSD (R-to-P):')
                rmsdbefore = RMS3(path.GetNDimIm(), path.GetConfig1() - path.GetConfig2())
                prod_coords, atom_coords = MinimizeRotation(path.GetNDimIm(), path.GetConfig1(),
                                                            path.GetConfig2())
                path.SetConfig2(prod_coords)
                path.SetConfig1(atom_coords)
                rmsdafter = RMS3(path.GetNDimIm(), path.GetConfig1() - path.GetConfig2())

                print('RMSD: %6.3f -> %6.3f %s' % (rmsdbefore, rmsdafter, KNARRsettings.lengthstring))

            rp = PathLinearInterpol(path.GetNDimIm(), path.nim,
                                    path.GetConfig1(), path.GetConfig2(),
                                    path.GetPBC(), path.GetCell())

            path.SetCoords(rp)
            path.MIC()
            path.setup = True

        # ===============================================
        # Perform linear interpolation with insertion
        # ================================================
        else:
            print("Insertion of TS geometry is active")
            if not path.IsConstrained() and not path.GetPBC() and not path.twodee:
                print('Minimzation of RMSD (R-to-I):')
                rmsdbefore = RMS3(path.GetNDimIm(), path.GetConfig1() - path.GetInsertionConfig())
                insertion_coords, atom_coords = MinimizeRotation(path.GetNDimIm(), path.GetConfig1(),
                                                                 path.GetInsertionConfig())
                path.SetInsertionConfig(insertion_coords)
                path.SetConfig1(atom_coords)
                rmsdafter = RMS3(path.GetNDimIm(), path.GetConfig1() - path.GetInsertionConfig())

                print('RMSD: %6.3f -> %6.3f %s' % (rmsdbefore, rmsdafter, KNARRsettings.lengthstring))

                print('Minimzation of RMSD (I-to-P):')
                rmsdbefore = RMS3(path.GetNDimIm(), path.GetInsertionConfig() - path.GetConfig2())
                prod_coords, insertion_coords = MinimizeRotation(path.GetNDimIm(), path.GetInsertionConfig(),
                                                                 path.GetConfig2())
                path.SetConfig2(prod_coords)
                path.SetInsertionConfig(insertion_coords)
                rmsdafter = RMS3(path.GetNDimIm(), path.GetConfig1() - path.GetInsertionConfig())

                print('RMSD: %6.3f -> %6.3f %s' % (rmsdbefore, rmsdafter, KNARRsettings.lengthstring))

            optimal_index = []
            list_of_indices = range(2, int(path.GetNim() - 2))
            for index in list_of_indices:
                rp = PathLinearInterpolWithInsertion(path.GetNDimIm(), path.nim,
                                                     path.GetConfig1(), path.GetConfig2(),
                                                     path.GetInsertionConfig(), index,
                                                     path.GetPBC(), path.GetCell())
                # WritePath("path_insertion"+str(index)+'.xyz', path.GetNDimIm(), path.GetNim(), rp,
                # path.GetSymbols(), energy=None)
                distmat = AllImageDistances(path.GetNDimIm(), path.GetNim(), rp,
                                            path.GetPBC(), path.GetCell())[0]
                optimal_index.append(np.max(distmat) - np.min(distmat))

            ind = np.argmin(optimal_index)
            insertion_no = list_of_indices[ind]
            rp = PathLinearInterpolWithInsertion(path.GetNDimIm(), path.nim,
                                                 path.GetConfig1(), path.GetConfig2(),
                                                 path.GetInsertionConfig(), insertion_no,
                                                 path.GetPBC(), path.GetCell())

            s = ComputeLengthOfPath(path.GetNDimIm(), path.GetNim(), rp, pbc=path.GetPBC(), cell=path.GetCell())
            rp = MakeReparametrization(path.GetNDimIm(), path.GetNim(), s, rp,
                                       np.zeros(shape=(path.GetNim() * path.GetNDim(), 1)),
                                       type_of_interp=0)

            path.SetCoords(rp)
            path.MIC()
            path.setup = True

        if type_of_interp > 0:
            print('IDPP optimization:')
            IDPP_OPT(path, max_iter=parameters["IDPP_MAX_ITER"], spring_const=parameters["IDPP_SPRINGCONST"],
                     time_step=parameters["IDPP_TIME_STEP"], max_move=parameters["IDPP_MAX_MOVE"],
                     tol_max_f=parameters["IDPP_MAX_F"], tol_rms_f=parameters["IDPP_RMS_F"])

    else:
        # ===============================================
        # SINGLE ENDED
        # ===============================================
        raise NotImplementedError()

    # Write output
    print('Final path:')
    path.PrintPath()
    print('Distances:')
    s = ComputeLengthOfPath(path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                            path.GetPBC(), path.GetCell())
    for i in range(len(s)):
        print('% li % 6.3lf %s' % (i, s[i], KNARRsettings.lengthstring))
    path.WritePath(basename + '_path.xyz')

    # Done.
    PrintJobDone('Path interpolation job', time.time() - start_t)

    return path


def IDPP_OPT(path, max_iter=3000, spring_const=10.0, time_step=0.01,
             max_move=0.1, tol_max_f=0.01, tol_rms_f=0.005):
    from KNARRcalculator.utilities import LinearInterpolationMatrix, GetAllConfigDistances
    from KNARRio.io import WriteTraj, WritePath
    from KNARRoptimization.vpo import GlobalVPO
    from KNARRoptimization.utilities import GlobalScaleStepByMax
    from KNARRjobs.utilities import PiecewiseCubicEnergyInterpolation, ComputeLengthOfPath, ComputeEffectiveNEBForce

    # ====================================
    # Initialize IDPP calculator
    # ====================================

    calculator = Calculator(name="IDPP", ncore=1)
    calculator.Setup()

    basename = 'idpp'
    # =========================================
    # Initialize configuration and get dkappa
    # =========================================

    # Get initial and final config
    Rinitial = np.zeros(shape=(path.GetNDimIm(), 1))
    Rfinal = np.zeros(shape=(path.GetNDimIm(), 1))
    rp = path.GetCoords()
    for i in range(path.GetNDimIm()):
        Rinitial[i] = rp[i]
        Rfinal[i] = rp[(path.GetNim() - 1) * path.GetNDimIm() + i]

    # Get distance matrices for reactant and product state configurations
    initialdistancematrix, initialgdx, initialgdy, initialgdz = GetAllConfigDistances(path.GetNDimIm(), Rinitial)
    finaldistancematrix, finalgdx, finalgy, finalgdz = GetAllConfigDistances(path.GetNDimIm(), Rfinal)

    # Get interpolation matrix dkappa of distances
    dkappa = LinearInterpolationMatrix(path.GetNDimIm(), path.GetNim(), initialdistancematrix,
                                       finaldistancematrix)
    path.Setdkappa(dkappa)

    # ======================================
    # Initialize neb parameters
    # ======================================
    idpp_max_iter = max_iter
    tangent_type = 1
    spring_type = 1
    energy_weighted = False
    springconst2 = None
    springconst = spring_const

    time_step = time_step
    max_move = max_move
    tol_max_f = tol_max_f
    tol_rms_f = tol_rms_f

    perp_springtype = 0
    free_end = False
    free_end_type = None
    free_end_energy1 = None
    free_end_energy2 = None
    free_end_kappa = None
    startci = False
    ci = -1
    remove_extern_force = True

    Ereactant = 0.0
    converged = 0.0
    reset_opt = False
    stop_neb = False
    was_scaled = False
    reset_on_scaling = False
    step = 0.0
    # ----------------------------------------------------------
    # Start NEB iterations
    # ----------------------------------------------------------
    for it in range(idpp_max_iter):

        calculator.Compute(path)
        path.UpdateR()
        path.UpdateF()

        WriteTraj(basename + "_optimization.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())

        WritePath(basename + "_current.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())

        s = ComputeLengthOfPath(path.GetNDofIm(), path.GetNim(), path.GetR(), path.GetPBC(), path.GetCell())
        fneb, freal_perp, freal_paral = \
            ComputeEffectiveNEBForce(path.GetF(), it, path.GetNDofIm(), path.GetNim(), ci, path.GetR(),
                                     path.GetEnergy(),
                                     tangent_type, spring_type, energy_weighted, springconst, springconst2,
                                     perp_springtype,
                                     free_end, free_end_type, free_end_energy1, free_end_energy2, free_end_kappa,
                                     startci, remove_extern_force,
                                     path.GetPBC(), path.GetCell())
        rmsf = RMS(path.GetNDof(), freal_perp)
        maxf = np.max(abs(freal_perp))
        hei = np.argmax(path.GetEnergy())
        print ("%4i %6.2lf  % 6.6lf %3li  %8.4lf  %8.4lf %8.4lf"
               % (it, s[-1], path.GetEnergy()[hei] - Ereactant, hei, rmsf, maxf, np.max(abs(step))))
        PiecewiseCubicEnergyInterpolation(basename + ".interp", path.GetNim(), s, path.GetEnergy(), freal_paral, it)
        if (tol_max_f > maxf and tol_rms_f > rmsf):
            converged = True

        if it == idpp_max_iter - 1:
            stop_neb = True
            converged = False

        if converged or stop_neb:
            break

        if it == 0 or reset_opt:
            reset_opt = False
            path.ZeroV()
        if was_scaled:
            time_step = time_step * 0.95

        step, velo = GlobalVPO(fneb, path.GetV(), time_step)
        path.SetV(velo)
        step, was_scaled = GlobalScaleStepByMax(step, max_move)

        if reset_on_scaling and was_scaled:
            reset_opt = True

        path.SetR(path.GetR() + step)
        path.UpdateCoords()
        path.MIC()

    if converged:
        PrintConverged(it, 0)
        # WritePath(basename + "_path.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
        #          path.GetSymbols(), path.GetEnergy())

        if os.path.isfile(basename + "_current.xyz"):
            os.remove(basename + "_current.xyz")

    if stop_neb:
        PrintMaxIter(idpp_max_iter)
        raise RuntimeError("IDPP-NEB unable to converge")

    return path.GetCoords()
