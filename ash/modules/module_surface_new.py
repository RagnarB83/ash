import os
import glob
import shutil
import copy
import time
import itertools
import numpy as np
#import ash
from ash.functions.functions_general import frange, BC, natural_sort, print_line_with_mainheader, \
    print_line_with_subheader1,print_time_rel, ashexit, print_if_level
import ash.functions.functions_parallel
from ash.modules.module_coords import check_charge_mult
from ash.modules.module_coords_PBC import write_CIF_file, write_POSCAR_file, write_XSF_file
from ash.modules.module_results import ASH_Results
from ash.interfaces.interface_geometric_new import geomeTRICOptimizer,GeomeTRICOptimizerClass
from ash.interfaces.interface_dlfind import DLFIND_optimizer, DLFIND_optimizerClass
from ash.modules.module_theory import NumGradclass
from ash.constants import ang2bohr
from ash.functions.functions_optimization import Cart_optimizer_class

# New rewritten calc_surface function
def calc_surface(
    fragment=None, theory=None, charge=None, mult=None, optimizer='geometric', printlevel=2,
    scantype='UNRELAXED', resultfile='surface_results.txt',
    keepoutputfiles=True, keepmofiles=False,
    runmode='serial', coordsystem='dlc', maxiter=250,
    NumGrad=False, extraconstraints=None,
    set_geometry_via_restraint=True,
    convergence_setting=None, conv_criteria=None,
    subfrctor=1, force_noPBC=False,
    numcores=1, ActiveRegion=False, actatoms=None,
    PBC_format_option="CIF",
    # ---- New N-dimensional interface ----
    RC_list=None,
    # ---- Legacy 1D/2D interface (kept for backward compatibility) ----
    RC1_range=None, RC1_type=None, RC1_indices=None,
    RC2_range=None, RC2_type=None, RC2_indices=None,
):
    """Calculate an N-dimensional potential energy surface (1D, 2D, 3D, …).
 
    The preferred interface is *RC_list*, a list of reaction-coordinate dicts::
 
        RC_list=[
            {'type': 'bond',  'indices': [[0, 1]],    'range': [1.0, 2.0, 0.1]},
            {'type': 'angle', 'indices': [[0, 1, 2]], 'range': [90, 180, 10]},
        ]
 
    The legacy ``RC1_*`` / ``RC2_*`` keyword arguments continue to work unchanged.
 
    Args:
        fragment           : ASH Fragment object
        theory             : ASH Theory object
        charge, mult       : charge and multiplicity
        scantype           : 'UNRELAXED' or 'RELAXED'
        resultfile         : filename for surface results
        keepoutputfiles    : copy QM output files per point
        keepmofiles        : copy MO files per point
        runmode            : 'serial' or 'parallel'
        numcores           : number of cores for parallel mode
        coordsystem        : coordinate system for geomeTRIC
        maxiter            : max optimisation iterations
        NumGrad            : use numerical gradients
        extraconstraints   : additional constraints dict
        convergence_setting: geomeTRIC convergence preset
        conv_criteria      : explicit convergence criteria dict
        subfrctor          : subfrctor for geomeTRIC
        force_noPBC        : disable PBC in optimiser
        ActiveRegion       : use active region in optimisation
        actatoms           : list of active atoms
        PBC_format_option  : 'CIF', 'XSF', or 'POSCAR'
        RC_list            : list of RC dicts (new interface)
        RC1_*/RC2_*        : legacy 1D/2D parameters
 
    Returns:
        ASH_Results with surfacepoints dict
    """
    module_init_time = time.time()
    print_line_with_mainheader("CALC_SURFACE FUNCTION")

    # NOW SETTING UP OPTIMIZER
    # Defining extraconstraints
    extraconstraints={} if extraconstraints is None else extraconstraints
    if isinstance(optimizer,str):
        if optimizer.lower() == "geometric":
            print("Optimizer to use for surface scan: geomeTRIC")
            opt_arguments = {
                    'coordsystem': coordsystem,
                    'maxiter': maxiter,
                    'convergence_setting': convergence_setting,
                    'conv_criteria': conv_criteria,
                    'subfrctor': subfrctor,
                    'force_noPBC': force_noPBC, 'PBC_format_option': PBC_format_option,
                    'ActiveRegion': ActiveRegion, 
                    'result_write_to_disk':False,
                    'printlevel':printlevel,
                    }
            # Creating optimizer object
            optimizerobj = GeomeTRICOptimizerClass(**opt_arguments)
            # For geomeTRIC we use constrainvalue True
            extraoopt_run_kws={'constrainvalue':True}
            # For geometric we don't have to preset
            presetting_geometry_required=False

        elif optimizer.lower() in ['dlfind','dl-find']:
            print("Optimizer to use for surface scan: DL-FIND")
            opt_arguments={'maxcycle':maxiter,'iopt':3, 'icoord':1, 'printlevel':printlevel}

            # Creating optimizer object
            optimizerobj = DLFIND_optimizerClass(**opt_arguments)
            extraoopt_run_kws={}
            # DL-FIND: need to be preset
            presetting_geometry_required=True
        elif optimizer.lower() in ['cartopt', 'cart_opt', 'cart-opt', 'cartesian']:
            print("Optimizer to use for surface scan: Cart_optimizer")
            opt_arguments={'maxiter':maxiter,'printlevel':printlevel}

            # Creating optimizer object
            optimizerobj = Cart_optimizer_class(**opt_arguments)
            extraoopt_run_kws={}
            # Cart_optimizer: no presetting required
            presetting_geometry_required=False
        else:
            print("Wrong optimizer option chosen. Valid options are: geometric and dlfind")
            ashexit()
    elif isinstance(optimizer,GeomeTRICOptimizerClass):
        print("A GeomeTRICOptimizerClass object was provided")
        optimizerobj=optimizer
        # For geomeTRIC we use constrainvalue True
        extraoopt_run_kws={'constrainvalue':True}
        # For geometric we don't have to preset
        presetting_geometry_required=False
        # Merge constraints if defined in both optimizer object and extraconstraints argument
        extraconstraints = _merge_dicts(optimizerobj.constraints, extraconstraints)
    elif isinstance(optimizer,DLFIND_optimizerClass):
        print("A DLFIND_optimizerClass object was provided")
        optimizerobj=optimizer
        opt_arguments={}
        extraoopt_run_kws={}
        # DL-FIND: need to be preset
        presetting_geometry_required=True
        # Merge constraints if defined in both optimizer object and extraconstraints argument
        extraconstraints = _merge_dicts(optimizerobj.constraints, extraconstraints)
    elif isinstance(optimizer,Cart_optimizer_class):
        print("A Cart_optimizer_class object was provided")
        optimizerobj=optimizer
        opt_arguments={}
        extraoopt_run_kws={}
        # Cart_optimizer: no presetting required
        presetting_geometry_required=False
        # Merge constraints if defined in both optimizer object and extraconstraints argument
        extraconstraints = _merge_dicts(optimizerobj.constraints, extraconstraints)
    else:
        print("optimizer keyword should either be a string (geometric or dlfind) or an Optimizer object (GeomeTRICOptimizerClass or DLFIND_optimizerClass)")
        ashexit()

    # Build connectivity once
    conn = _build_connectivity(fragment.coords, fragment.elems)

    # Changing printlevel of fragment
    fragment.printlevel=printlevel

    # -- NumGrad wrapping ---------------------------------------------------
    if NumGrad:
        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
        theory = NumGradclass(theory=theory)

    # -- Charge/mult check --------------------------------------------------
    charge, mult = check_charge_mult(
        charge, mult, theory.theorytype, fragment, "calc_surface", theory=theory,
    )

    # -- Build RC_list (legacy compat) --------------------------------------
    if RC_list is None:
        RC_list = _legacy_to_rc_list(
            RC1_type, RC1_indices, RC1_range,
            RC2_type, RC2_indices, RC2_range,
        )
    RC_list = _normalise_rc_list(RC_list)
    dimension = len(RC_list)
    print(f"Number of reaction coordinates (dimension): {dimension}")
    # -- Build value lists and total point count ----------------------------
    RC_value_lists = _build_rc_value_lists(RC_list)
    totalnumpoints = 1
    for vl in RC_value_lists:
        totalnumpoints *= len(vl)
    for i, vl in enumerate(RC_value_lists):
        print(f"RCvalue{i + 1}_list: {vl}")
    print(f"Number of surfacepoints to calculate: {totalnumpoints}")

    # -- Read existing results ----------------------------------------------
    surfacedictionary = read_surfacedict_from_file(resultfile, dimension=dimension)
    print("Initial surfacedictionary:", surfacedictionary)

    # -- Output-file policy -------------------------------------------------
    keepoutputfiles, keepmofiles = _silence_outputfiles_for_special_theories(
        theory, keepoutputfiles, keepmofiles,
    )
    print("keepoutputfiles:", keepoutputfiles)
    print("keepmofiles:", keepmofiles)

    # -- PBC setup ----------------------------------------------------------
    if getattr(theory, "periodic", False):
        print(
            "Warning: Theory is periodic. Constrained geometry optimizations by "
            "Optimizer will optimize both atom and cell parameters"
        )
        print("Set force_noPBC=True if you do not want cell-parameter optimisation.")
        print(f"PBC_format_option: {PBC_format_option}")
    convert_to_pbcfile = _select_pbc_converter(PBC_format_option)

    # -- Create/reset output directories ------------------------------------
    _setup_directories(theory)

    # -----------------------------------------------------------------------
    # PARALLEL MODE
    # -----------------------------------------------------------------------
    if runmode == 'parallel':
        print("Parallel runmode. Number of cores:", numcores)
        if numcores == 1:
            print("Error: numcores must be > 1 for parallel runmode. Exiting.")
            ashexit()

        surfacepointfragments_list = []

        if scantype.upper() == 'UNRELAXED':
            # Geometry-setting pass with ZeroTheory
            zerotheory = ash.ZeroTheory()
            pointcount = 0
            for rc_values in itertools.product(*RC_value_lists):
                pointcount += 1
                key = _point_key(rc_values)
                label = _point_label(rc_values)
                print(f"======= Surfacepoint {pointcount}/{totalnumpoints}: {label} =======")
                if key in surfacedictionary:
                    continue
                allconstraints = set_constraints_nd(RC_list, rc_values, extraconstraints, fragment=fragment,
                                                    printlevel=printlevel)
                print_if_level(f"allconstraints: {allconstraints}",printlevel,2)

                # Copying fragment
                newfrag = copy.copy(fragment)
                newfrag.printlevel=printlevel
                newfrag.label = key

                # Here we modify geometry
                print_if_level(f"For an unrelaxed scan we need to modify geometry first (done in serial fashion)",printlevel,2)
                print_if_level(f"set_geometry_via_restraint: {set_geometry_via_restraint}",printlevel,2)
                if set_geometry_via_restraint is True:
                    print_if_level(f"Modifying geometry to set constraints via DL-FIND restraint optimization",printlevel,2)
                    # NOTE: passing extraconstraints if any
                    _preset_geometry_restraint(newfrag, RC_list, rc_values, optimizerobj,
                                opt_arguments, charge, mult,printlevel=1, extraconstraints=extraconstraints,
                                extraoopt_run_kws=extraoopt_run_kws,
                                force_constant=10000.0)
                else:
                    print_if_level(f"Modifying geometry to set constraints via coordinate manipulation",printlevel,2)
                    _set_geometry_direct(newfrag, RC_list, rc_values, conn=conn)
                _verify_geometry(fragment, RC_list, rc_values, printlevel=printlevel)

                xyzname = f"{label}.xyz"
                newfrag.write_xyzfile(xyzfilename=xyzname)
                shutil.move(xyzname, f"surface_xyzfiles/{xyzname}")
                _handle_pbc(theory, newfrag, label, convert_to_pbcfile)
                surfacepointfragments_list.append(newfrag)

            result_surface = ash.functions.functions_parallel.Job_parallel(
                fragments=surfacepointfragments_list, theories=[theory], numcores=numcores,
            )
            surfacedictionary = result_surface.energies_dict

        elif scantype.upper() == 'RELAXED':
            print("Warning: Relaxed scans in parallel mode are experimental")
            pointcount = 0
            for rc_values in itertools.product(*RC_value_lists):
                pointcount += 1
                key = _point_key(rc_values)
                label = _point_label(rc_values)
                print(f"======= Surfacepoint {pointcount}/{totalnumpoints}: {label} =======")
                if key in surfacedictionary:
                    continue
                allconstraints = set_constraints_nd(RC_list, rc_values, extraconstraints, fragment=fragment,
                                                    printlevel=printlevel)
                print_if_level(f"allconstraints: {allconstraints}", printlevel,2) 
                newfrag = copy.copy(fragment)
                newfrag.printlevel=printlevel

                if presetting_geometry_required:
                    print_if_level(f"For DL-FIND we need to modify geometry first to the desired constraint value.",printlevel,2)
                    print_if_level(f"set_geometry_via_restraint: {set_geometry_via_restraint}",printlevel,2)
                    if set_geometry_via_restraint is True:
                        print_if_level(f"Modifying geometry to get constraint value via DL-FIND restraint optimization",printlevel,2)
                        _preset_geometry_restraint(newfrag, RC_list, rc_values, optimizerobj,
                                    opt_arguments, charge, mult,printlevel=1, extraconstraints=extraconstraints,
                                    extraoopt_run_kws=extraoopt_run_kws,
                                    force_constant=10000.0)
                    else:
                        print_if_level(f"Modifying geometry to get constraint value via coordinate manipulation",printlevel,2)
                        _set_geometry_direct(newfrag, RC_list, rc_values, conn=conn)
                    _verify_geometry(newfrag, RC_list, rc_values, printlevel=printlevel)
                newfrag.label = key
                newfrag.constraints = allconstraints
                surfacepointfragments_list.append(newfrag)

            result_surface = ash.functions.functions_parallel.Job_parallel(
                fragments=surfacepointfragments_list, theories=[theory],
                numcores=numcores, Opt=True, optimizer=optimizer,
            )
            # Copy optimised XYZ files to surface_xyzfiles/
            for rc_values in itertools.product(*RC_value_lists):
                key = _point_key(rc_values)
                label = _point_label(rc_values)
                d = result_surface.worker_dirnames[key]
                shutil.copy(
                    d + "/Fragment-optimized.xyz",
                    f"surface_xyzfiles/{label}.xyz",
                )
            surfacedictionary = result_surface.energies_dict

        print("Parallel calculation done!")
        print("surfacedictionary:", surfacedictionary)
        if len(surfacedictionary) != totalnumpoints:
            print(
                f"Warning: Dictionary incomplete! "
                f"Got {len(surfacedictionary)}, expected {totalnumpoints}"
            )

    # -----------------------------------------------------------------------
    # SERIAL MODE
    # -----------------------------------------------------------------------
    elif runmode == 'serial':
        print("Serial runmode")
        zerotheory = ash.ZeroTheory()
        pointcount = 0

        for rc_values in itertools.product(*RC_value_lists):
            pointcount += 1
            key = _point_key(rc_values)
            label = _point_label(rc_values)

            # Resetting constraints is optimizer object to be safe
            optimizerobj.constraints=None

            print("=" * 50)
            print(f"Surfacepoint: {pointcount} / {totalnumpoints}")
            print(f"  {label}")
            if scantype.upper() == 'UNRELAXED':
                print("  Unrelaxed scan: first setting geometry and then doing single-point calculation")
            else:
                print("  Relaxed scan: relaxing geometry with theory + constraints.")
            print("=" * 50)

            if key in surfacedictionary:
                print(f"{label} already in dict. Skipping.")
                continue

            allconstraints = set_constraints_nd(RC_list, rc_values, extraconstraints, fragment=fragment,
                                                printlevel=printlevel)
            print_if_level(f"All constraints: {allconstraints}", printlevel,1)

            if scantype.upper() == 'UNRELAXED':
                # Here we modify geometry
                print_if_level(f"For an unrelaxed scan we need to modify geometry first (done in serial fashion)",printlevel,2)
                print_if_level(f"set_geometry_via_restraint: {set_geometry_via_restraint}",printlevel,2)
                if set_geometry_via_restraint is True:
                    print_if_level(f"Modifying geometry to set constraints via DL-FIND restraint optimization",printlevel,2)
                    # NOTE: passing extraconstraints if any
                    _preset_geometry_restraint(fragment, RC_list, rc_values, optimizerobj,
                                opt_arguments, charge, mult,printlevel=1, extraconstraints=extraconstraints,
                                extraoopt_run_kws=extraoopt_run_kws,
                                force_constant=10000.0)
                else:
                    print_if_level(f"Modifying geometry to set constraints via coordinate manipulation",printlevel,2)
                    _set_geometry_direct(fragment, RC_list, rc_values, conn=conn)
                _verify_geometry(fragment, RC_list, rc_values, printlevel=printlevel)

                print_if_level(f"Now running single-point calculation using Theory", printlevel,2)
                result = ash.Singlepoint(
                    fragment=fragment, theory=theory, charge=charge, mult=mult,
                )
            else:  # RELAXED
                if presetting_geometry_required:
                    print_if_level(f"For DL-FIND we need to modify geometry first to set constraints.", printlevel,2)
                    if set_geometry_via_restraint is True:
                        print_if_level(f"Modifying geometry to set constraints via DL-FIND restraint optimization", printlevel,2)
                        # NOTE: passing extraconstraints if any
                        _preset_geometry_restraint(fragment, RC_list, rc_values, optimizerobj,
                                    opt_arguments, charge, mult,printlevel=1, extraconstraints=extraconstraints,
                                    extraoopt_run_kws=extraoopt_run_kws,
                                    force_constant=10000.0)
                    else:
                        print_if_level(f"Modifying geometry to set constraints via coordinate manipulation", printlevel,2)
                        _set_geometry_direct(fragment, RC_list, rc_values, conn=conn)
                    _verify_geometry(fragment, RC_list, rc_values, printlevel=printlevel)
                else:
                    print_if_level(f"For geometric Optimizer we enforce constraints during optimization.", printlevel,2)
                print_if_level(f"Now running Relaxed Optimization", printlevel,2)
                # Running optimizer object

                #Running optimizer object, passing theory, fragment, constraints and possible extra kws
                result = optimizerobj.run(theory=theory,fragment=fragment, constraints=allconstraints, **extraoopt_run_kws)

                #if pointcount == 2:
                #    print("2nd point optimization result:", result)
                #    ashexit()
            energy = float(result.energy)
            print(f"  {label}  Energy: {energy}")

            # -- File I/O ---------------------------------------------------
            fragment.write_xyzfile(xyzfilename="surface_traj.xyz", writemode='a')
            xyzname = f"{label}.xyz"
            fragment.write_xyzfile(xyzfilename=xyzname)
            shutil.move(xyzname, f"surface_xyzfiles/{xyzname}")
            _handle_output_files(theory, label, keepoutputfiles, keepmofiles, printlevel=printlevel)
            _handle_pbc(theory, fragment, label, convert_to_pbcfile)
 
            surfacedictionary[key] = float(energy)
            write_surfacedict_to_file(surfacedictionary, resultfile, dimension=dimension)
 
        print("surfacedictionary:", surfacedictionary)
 
    else:
        print(f"Error: Unknown runmode '{runmode}'. Use 'serial' or 'parallel'.")
        ashexit()
 
    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------
    write_surfacedict_to_file(surfacedictionary, resultfile, dimension=dimension)
 
    # Combine all per-point XYZ files into a single trajectory
    xyzfile_list = glob.glob("surface_xyzfiles/*.xyz")

    with open("surface_traj_final.xyz", 'w') as outfile:
        for xyzfile in natural_sort(xyzfile_list):
            with open(xyzfile) as infile:
                outfile.write(infile.read())
 
    print_time_rel(module_init_time, modulename='calc_surface', moduleindex=0)
 
    result = ASH_Results(label="Surface calc", surfacepoints=surfacedictionary)
    try:
        result.write_to_disk(filename="ASH_surface.result")
    except TypeError as e:
        print("Problem writing ASH_surface.result to disk. Skipping.")
        print("Error:", e)
    return result

# FROM XYZ
def calc_surface_fromXYZ(
    xyzdir=None, multixyzfile=None, theory=None, charge=None, mult=None, optimizer="geometric",
    dimension=None, resultfile='surface_results.txt', printlevel=2,
    scantype='UNRELAXED', runmode='serial',
    coordsystem='dlc', maxiter=250, extraconstraints=None,
    convergence_setting=None, conv_criteria=None, subfrctor=1, NumGrad=False,
    numcores=None,
    keepoutputfiles=True, force_noPBC=False, PBC_format_option="CIF",
    keepmofiles=False, read_mofiles=False, mofilesdir=None,
    # New ND interface:
    RC_list=None,
    # Legacy 1D/2D interface (kept for backward compatibility):
    RC1_type=None, RC1_indices=None,
    RC2_type=None, RC2_indices=None,
):
    """Calculate an N-dimensional surface from a directory of XYZ files.

    XYZ filenames must follow the convention produced by calc_surface::

        RC1_<val1>-RC2_<val2>-...-RCN_<valN>.xyz

    RC information is only required for RELAXED scans (to rebuild constraints).
    For UNRELAXED scans all RC arguments may be omitted.

    Preferred interface uses RC_list (same format as calc_surface, but 'range'
    is ignored and may be omitted since the grid is defined by the XYZ files)::

        calc_surface_fromXYZ(
            xyzdir='surface_xyzfiles', theory=theory, charge=0, mult=1,
            scantype='Relaxed', dimension=2,
            RC_list=[
                {'type': 'bond',  'indices': [[0, 1], [0, 2]]},
                {'type': 'angle', 'indices': [[1, 0, 2]]},
            ],
        )

    Legacy 1D/2D keyword arguments (RC1_type, RC1_indices, RC2_type,
    RC2_indices) continue to work unchanged.

    Args:
        xyzdir           : directory containing XYZ files
        dimension        : number of RC coordinates; inferred from RC_list if
                           not provided, or from the first filename as fallback
        theory           : ASH Theory object
        charge, mult     : charge and multiplicity
        scantype         : 'UNRELAXED' or 'RELAXED'
        runmode          : 'serial' or 'parallel'
        numcores         : cores for parallel mode
        RC_list          : list of RC dicts (new ND interface)
        RC1_type/indices : legacy 1D/2D constraint specification
        RC2_type/indices : legacy 2D constraint specification
        read_mofiles     : read MO files from mofilesdir
        mofilesdir       : directory containing MO files

    Returns:
        ASH_Results with surfacepoints dict
    """
    module_init_time = time.time()
    print_line_with_mainheader("CALC_SURFACE_FROMXYZ FUNCTION")
    if isinstance(optimizer,str):
        if optimizer.lower() == "geometric":
            print("Optimizer to use for surface scan: geomeTRIC")
            Optimizer=geomeTRICOptimizer
            Optimizerclass=GeomeTRICOptimizerClass
            opt_arguments = {
                    'coordsystem': coordsystem,
                    'maxiter': maxiter,
                    'convergence_setting': convergence_setting,
                    'conv_criteria': conv_criteria,
                    'subfrctor': subfrctor,
                    'force_noPBC': force_noPBC,
                    'PBC_format_option': PBC_format_option}
        elif optimizer.lower() in ['dlfind','dl-find']:
            print("Optimizer to use for surface scan: DL-FIND")
            Optimizer=DLFIND_optimizer
            Optimizerclass=DLFIND_optimizerClass
            opt_arguments={}
    elif isinstance(optimizer,GeomeTRICOptimizerClass):
        print("A GeomeTRICOptimizerClass object was provided")
    elif isinstance(optimizer,DLFIND_optimizerClass):
        print("A GeomeTRICOptimizerClass object was provided")
        opt_arguments={}
    else:
        print("optimizer keyword should either be a string (geometric or dlfind) or an Optimizer object")
        ashexit()



    # -- NumGrad wrapping ---------------------------------------------------
    if NumGrad:
        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
        theory = NumGradclass(theory=theory)

    # -- Basic argument checks ----------------------------------------------
    if charge is None or mult is None:
        print(BC.FAIL, "Error: charge and mult must be defined for calc_surface_fromXYZ", BC.END)
        ashexit()
    if xyzdir is None:
        print("Error: xyzdir must be provided")
        ashexit()
    if read_mofiles and mofilesdir is None:
        print("Error: mofilesdir not set but read_mofiles=True. Exiting.")
        ashexit()

    # -- Build RC_list from legacy kwargs if needed -------------------------
    if RC_list is None and RC1_type is not None:
        # Legacy path: build RC_list without 'range' (not needed here)
        RC_list = [{'type': RC1_type, 'indices': RC1_indices}]
        if RC2_type is not None:
            RC_list.append({'type': RC2_type, 'indices': RC2_indices})

    # Normalise indices to list-of-lists
    if RC_list is not None:
        RC_list = _normalise_rc_list(RC_list)

    # For RELAXED scans RC_list is mandatory
    if scantype.upper() == 'RELAXED' and not RC_list:
        print(
            "Error: RC_list (or legacy RC1_type/RC1_indices) is required for "
            "RELAXED scans in calc_surface_fromXYZ"
        )
        ashexit()

    # -- Discover XYZ files -------------------------------------------------
    xyzfile_list = sorted(glob.glob(xyzdir + '/*.xyz'))
    totalnumpoints = len(xyzfile_list)
    if totalnumpoints == 0:
        print(f"Found no XYZ-files in directory '{xyzdir}'. Exiting")
        ashexit()

    # -- Infer dimension ----------------------------------------------------
    if dimension is None:
        if RC_list is not None:
            dimension = len(RC_list)
        else:
            # Infer from first filename: count how many 'RC' tokens appear
            first_file = os.path.basename(xyzfile_list[0])
            dimension = first_file.replace('.xyz', '').count('RC')
        print(f"Inferred dimension={dimension}")

    print("XYZdir:", xyzdir)
    print("Theory:", theory)
    print("Dimension:", dimension)
    print("Scan type:", scantype)
    print("keepoutputfiles:", keepoutputfiles)
    print("keepmofiles:", keepmofiles)
    print("read_mofiles:", read_mofiles)
    print("mofilesdir:", mofilesdir)
    print("runmode:", runmode)
    print("totalnumpoints:", totalnumpoints)

    # -- Read existing results ----------------------------------------------
    surfacedictionary = read_surfacedict_from_file(resultfile, dimension=dimension)
    print("Initial surfacedictionary:", surfacedictionary)

    if len(surfacedictionary) == totalnumpoints:
        print(
            f"Surface dictionary size {len(surfacedictionary)} matches "
            f"total number of XYZ files {totalnumpoints}. All data present."
        )
        result = ASH_Results(label="Surface calc XYZ", surfacepoints=surfacedictionary)
        result.write_to_disk(filename="ASH_surface_xyz.result")
        return result

    # -- Output-file policy -------------------------------------------------
    keepoutputfiles, keepmofiles = _silence_outputfiles_for_special_theories(
        theory, keepoutputfiles, keepmofiles,
    )
    print("keepoutputfiles:", keepoutputfiles)
    print("keepmofiles:", keepmofiles)

    # -- Directory setup ----------------------------------------------------
    if scantype.upper() == 'RELAXED':
        if os.path.exists('surface_xyzfiles'):
            print(BC.FAIL, "surface_xyzfiles directory already exists. Please remove it.", BC.END)
            ashexit()
        os.mkdir('surface_xyzfiles')

    if runmode == 'serial':
        shutil.rmtree("surface_outfiles", ignore_errors=True)
        os.makedirs("surface_outfiles", exist_ok=True)
        shutil.rmtree("surface_mofiles", ignore_errors=True)
        os.makedirs("surface_mofiles", exist_ok=True)

    # -----------------------------------------------------------------------
    # Helper: parse RC values from filename
    # Handles filenames like RC1_1.45-RC2_90.0-RC3_0.0.xyz
    # -----------------------------------------------------------------------
    def parse_rc_values(relfile):
        base = relfile.replace('.xyz', '')
        # Split on '-RC' to get ['RC1_1.45', '2_90.0', '3_0.0']
        parts = base.split('-RC')
        vals = []
        for part in parts:
            # Each part is like 'RC1_1.45' or '2_90.0' — value is after last '_'
            vals.append(float(part.split('_')[-1]))
        return tuple(vals[:dimension])

    # -----------------------------------------------------------------------
    # Helper: build geomeTRIC constraints for a given point
    # -----------------------------------------------------------------------
    def build_constraints(rc_vals, frag):
        if not RC_list:
            return {}
        return set_constraints_nd(RC_list, rc_vals, extraconstraints, fragment=frag)

    # -----------------------------------------------------------------------
    # PARALLEL
    # -----------------------------------------------------------------------
    if runmode == 'parallel':
        if numcores is None:
            print("Error: numcores argument required for parallel runmode")
            ashexit()

        surfacepointfragments_list = []
        for file in xyzfile_list:
            relfile = os.path.basename(file)
            rc_vals = parse_rc_values(relfile)
            key = _point_key(rc_vals)
            if key in surfacedictionary:
                continue
            newfrag = ash.Fragment(xyzfile=file, label=key, charge=charge, mult=mult)
            if scantype.upper() == 'RELAXED':
                newfrag.constraints = build_constraints(rc_vals,newfrag)
            surfacepointfragments_list.append(newfrag)

        if scantype.upper() == 'UNRELAXED':
            kwargs = dict(
                fragments=surfacepointfragments_list,
                theories=[theory],
                numcores=numcores,
            )
            if read_mofiles:
                kwargs['mofilesdir'] = mofilesdir
            results = ash.functions.functions_parallel.Job_parallel(**kwargs)

        else:  # RELAXED
            optimizer = Optimizerclass(
                maxiter=maxiter, 
                convergence_setting=convergence_setting, 
                **opt_arguments,
            )
            kwargs = dict(
                fragments=surfacepointfragments_list,
                theories=[theory],
                numcores=numcores,
                Opt=True,
                optimizer=optimizer,
            )
            if read_mofiles:
                kwargs['mofilesdir'] = mofilesdir
            results = ash.functions.functions_parallel.Job_parallel(**kwargs)

        print("Parallel calculation done!")
        surfacedictionary = {k: float(v) for k, v in results.energies_dict.items()}
        if len(surfacedictionary) != totalnumpoints:
            print(
                f"Warning: Dictionary incomplete! "
                f"Got {len(surfacedictionary)}, expected {totalnumpoints}"
            )

    # -----------------------------------------------------------------------
    # SERIAL
    # -----------------------------------------------------------------------
    elif runmode == 'serial':
        for count, file in enumerate(xyzfile_list):
            relfile = os.path.basename(file)
            rc_vals = parse_rc_values(relfile)
            key = _point_key(rc_vals)
            label = _point_label(rc_vals)

            print("=" * 66)
            print(f"Surfacepoint: {count + 1} / {totalnumpoints}")
            print(f"XYZ-file: {relfile}  ({label})")
            print("=" * 66)

            if read_mofiles:
                mofile = f"{mofilesdir}/{theory.filename}_{label}.gbw"
                print(f"Will read MO-file: {mofile}")
                if theory.__class__.__name__ == "ORCATheory":
                    theory.moreadfile = mofile

            if key in surfacedictionary:
                print(f"{label} already in dict. Skipping.")
                continue

            mol = ash.Fragment(xyzfile=file)

            if scantype.upper() == 'UNRELAXED':
                result = ash.Singlepoint(
                    theory=theory, fragment=mol, charge=charge, mult=mult,
                )

            else:  # RELAXED
                allconstraints = build_constraints(rc_vals,mol)
                result = Optimizer(
                    fragment=mol, theory=theory, maxiter=maxiter,
                    constraints=allconstraints,
                    convergence_setting=convergence_setting,
                    charge=charge, mult=mult, **opt_arguments,
                )
                xyzname = f"{label}.xyz"
                mol.write_xyzfile(xyzfilename=xyzname)
                shutil.move(xyzname, f"surface_xyzfiles/{xyzname}")

            energy = float(result.energy)
            print(f"Energy of {relfile}: {energy} Eh")
            _handle_output_files(theory, label, keepoutputfiles, keepmofiles, printlevel=printlevel)
            surfacedictionary[key] = energy
            # Write after every point so partial results are never lost
            write_surfacedict_to_file(surfacedictionary, resultfile, dimension=dimension)

    else:
        print(f"Error: Unknown runmode '{runmode}'. Use 'serial' or 'parallel'.")
        ashexit()

    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------
    write_surfacedict_to_file(surfacedictionary, resultfile, dimension=dimension)
    print("Final surfacedictionary:", surfacedictionary)
    print_time_rel(module_init_time, modulename='calc_surface_fromXYZ', moduleindex=0)

    result = ASH_Results(label="Surface calc XYZ", surfacepoints=surfacedictionary)
    result.write_to_disk(filename="ASH_surface_xyz.result")
    return result



# HELPER FUNCTIONS

def _merge_dicts(dict1, dict2):
    """Merge two dictionaries, concatenating lists if keys overlap."""
    if dict1 is None:
        dict1 = {}
    if dict2 is None:
        dict2 = {}
    merged = dict(dict1)  # start with dict1's keys and values
    for key, value in dict2.items():
        if key in merged:
            merged[key] = merged[key] + value  # concatenate lists
        else:
            merged[key] = value
    return merged

def read_surfacedict_from_file(resultfile, dimension=None):
    """Read surface dictionary from resultfile.

    Returns an empty dict if the file does not exist.
    Keys are tuples of floats (uniform for all dimensions).
    """
    surfacedictionary = {}
    if not os.path.isfile(resultfile):
        return surfacedictionary
    print(f"Found existing resultfile: {resultfile}. Reading entries.")
    with open(resultfile) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            try:
                energy = float(tokens[-1])
                rc_vals = tuple(float(t) for t in tokens[:-1])
                #if dimension == 1:
                #    # Legacy: 1D keys stored as bare float in old files
                #    key = rc_vals[0] if len(rc_vals) == 1 else rc_vals
                #else:
                key = rc_vals
                surfacedictionary[key] = float(energy)
            except (ValueError, IndexError):
                print(f"Warning: could not parse line: {line!r}")
    return surfacedictionary

def write_surfacedict_to_file(surfacedictionary, resultfile, dimension=None):
    with open(resultfile, 'w') as f:
        f.write("# Surface scan results\n")
        f.write("# RC1 [RC2 ...] Energy\n")
        # Normalise keys to tuples so sorted() always works regardless of
        # whether the dict came from a fresh run or a legacy result file
        normalised = {
            (k,) if not isinstance(k, tuple) else k: v
            for k, v in surfacedictionary.items()
        }
        for key, energy in sorted(normalised.items()):
            rc_str = '  '.join(str(v) for v in key)
            f.write(f"{rc_str}  {energy}\n")


# SUPPORT FUNCTIONS (not to be called by user)

def _silence_outputfiles_for_special_theories(theory, keepoutputfiles, keepmofiles):
    name = theory.__class__.__name__
    if name in ("ZeroTheory", "ORCA_CC_CBS_Theory"):
        return False, False
    return keepoutputfiles, keepmofiles

def _select_pbc_converter(PBC_format_option):
    opt = PBC_format_option.upper()
    if opt == "CIF":
        return write_CIF_file
    elif opt == "XSF":
        return write_XSF_file
    elif opt == "POSCAR":
        return write_POSCAR_file
    else:
        print(f"Warning: Unknown PBC_format_option '{PBC_format_option}', defaulting to CIF")
        return write_CIF_file

def _legacy_to_rc_list(RC1_type, RC1_indices, RC1_range,
                        RC2_type, RC2_indices, RC2_range):
    if RC1_type is None or RC1_indices is None:
        print("Error: RC1_type and RC1_indices are required")
        ashexit()
    RC_list = [{'type': RC1_type, 'indices': RC1_indices, 'range': RC1_range}]
    if RC2_type is not None:
        RC_list.append({'type': RC2_type, 'indices': RC2_indices, 'range': RC2_range})
    return RC_list

def _normalise_rc_list(RC_list):
    """Ensure every RC dict has 'indices' as a list-of-lists."""
    out = []
    for rc in RC_list:
        rc = dict(rc)  # shallow copy so we don't mutate caller's data
        indices = rc['indices']
        if not any(isinstance(el, list) for el in indices):
            indices = [indices]
        rc['indices'] = indices
        out.append(rc)
    return out

def _build_rc_value_lists(RC_list):
    """Return a list of value-lists, one per RC dimension."""
    result = []
    for rc in RC_list:
        r = rc['range']
        vals = list(frange(r[0], r[1], r[2]))
        vals.append(float(r[1]))  # always include the endpoint
        result.append(vals)
    return result

def _setup_directories(theory):
    """Create/reset the standard surface output directories."""
    for d in ("surface_xyzfiles", "surface_outfiles", "surface_mofiles"):
        shutil.rmtree(d, ignore_errors=True)
        os.mkdir(d)
    try:
        os.remove("surface_traj.xyz")
    except FileNotFoundError:
        pass
    if getattr(theory, "periodic", False):
        shutil.rmtree("surface_pbcfiles", ignore_errors=True)
        os.mkdir("surface_pbcfiles")
        print("Created directory: surface_pbcfiles")

def _point_key(rc_values):
    """Dictionary key for a surface point.
 
    A 1-tuple behaves exactly like the old scalar key for 1D surfaces,
    but we keep it as a tuple throughout so the logic is uniform.
    Callers that need the old scalar key for 1D can unpack themselves.
    """
    return tuple(rc_values)

def _point_label(rc_values):
    """Human-readable label: 'RC1_1.5-RC2_120.0-RC3_2.0' etc."""
    return '-'.join(f'RC{i + 1}_{v}' for i, v in enumerate(rc_values))

def set_constraints_nd(RC_list, rc_values, extraconstraints=None, fragment=None, printlevel=2):
    """Build a geomeTRIC constraints dict for any number of reaction coordinates.

    Args:
        RC_list          : list of RC dicts (already normalised, indices are list-of-lists)
        rc_values        : tuple of current values, one per RC
        extraconstraints : optional additional constraints dict; each entry is a
                           list of [*indices, value] or just [*indices] (no value).
                           If no value is present and fragment is provided, the
                           current geometry value is measured and appended.
                           If no value and no fragment, an error is raised.
        fragment         : ASH fragment, used to measure current constraint values
                           when extraconstraints entries have no value appended.

    Returns:
        dict suitable for geomeTRICOptimizer's ``constraints`` argument
    """
    allconstraints = {}

    # RC constraints — value always explicitly provided
    for rc, val in zip(RC_list, rc_values):
        rc_type = rc['type']
        allconstraints.setdefault(rc_type, [])
        for indices in rc['indices']:
            allconstraints[rc_type].append([*indices, val])

    if extraconstraints:
        for constraint_type, entries in extraconstraints.items():
            allconstraints.setdefault(constraint_type, [])
            # Expected atom counts per constraint type (number of index atoms)
            natoms = {'bond': 2, 'angle': 3, 'dihedral': 4, 'distance': 2,
                      'cartesian': 1, 'translation-x': 1, 'translation-y': 1,
                      'translation-z': 1, 'rotation-x': 1, 'rotation-y': 1,
                      'rotation-z': 1}
            expected_natoms = natoms.get(constraint_type.lower(), None)

            for entry in entries:
                # Determine whether a value is already appended:
                # if the entry has more elements than the expected atom count,
                # the last element is the value.
                if expected_natoms is not None and len(entry) > expected_natoms:
                    # Value already present — use as-is
                    allconstraints[constraint_type].append(list(entry))
                elif expected_natoms is not None and len(entry) == expected_natoms:
                    # No value — measure from current geometry or error
                    if fragment is None:
                        if printlevel > 1:
                            print(
                                f"Error: extraconstraint of type '{constraint_type}' "
                                f"with indices {entry} has no value, and no fragment "
                                f"was provided to measure it from."
                            )
                        ashexit()
                    val = _measure_constraint(fragment, constraint_type, entry)
                    if printlevel > 1:
                        print(
                            f"extraconstraint '{constraint_type}' {entry}: "
                            f"no value provided, using current geometry value {val:.6f}"
                        )
                    allconstraints[constraint_type].append([*entry, val])
                else:
                    # Unknown type or ambiguous length — append as-is with a warning
                    if printlevel > 1:
                        print(
                            f"Warning: cannot determine whether value is present for "
                            f"extraconstraint type '{constraint_type}', entry {entry}. "
                            f"Appending as-is."
                        )
                    if isinstance(entry,int):
                        allconstraints[constraint_type].append(entry)
                    else:
                        allconstraints[constraint_type].append(list(entry))

    return allconstraints

def _measure_constraint(fragment, constraint_type, indices):
    """Measure the current value of a geometric constraint from fragment coords.

    Args:
        fragment        : ASH fragment (must have .coords in Angstrom)
        constraint_type : 'bond', 'angle', or 'dihedral'
        indices         : list of atom indices (0-based)

    Returns:
        float — bond length in Å, angle or dihedral in degrees
    """
    coords = np.array(fragment.coords)  # shape (natoms, 3)

    ct = constraint_type.lower()

    if ct in ('bond', 'distance'):
        a, b = indices
        return float(np.linalg.norm(coords[a] - coords[b]))

    elif ct == 'angle':
        a, b, c = indices
        v1 = coords[a] - coords[b]
        v2 = coords[c] - coords[b]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    elif ct == 'dihedral':
        a, b, c, d = indices
        b1 = coords[b] - coords[a]
        b2 = coords[c] - coords[b]
        b3 = coords[d] - coords[c]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return float(np.degrees(np.arctan2(y, x)))

    else:
        print(
            f"Warning: _measure_constraint does not know how to measure "
            f"'{constraint_type}'. Returning 0.0 as placeholder value."
        )
        return 0.0

def _handle_pbc(theory, fragment, pointlabel, convert_to_pbcfile):
    """Move PBC coordinate file to surface_pbcfiles/ if theory is periodic."""
    if not getattr(theory, "periodic", False):
        return
    pbcfile = convert_to_pbcfile(
        fragment.coords, fragment.elems,
        cellvectors=theory.periodic_cell_vectors,
    )
    ext = pbcfile.split('.')[-1]
    shutil.move(pbcfile, f"surface_pbcfiles/{pointlabel}.{ext}")

def _handle_output_files(theory, pointlabel, keepoutputfiles, keepmofiles, printlevel=2):
    """Copy QM output / MO files to their surface subdirectories."""
    if not hasattr(theory, 'theorytype') or theory.theorytype != "QM":
        if keepoutputfiles or keepmofiles:
            print("Warning: For hybrid theories, outputfiles and MO-files are not kept")
        return
    if keepoutputfiles:
        try:
            shutil.copyfile(
                theory.filename + '.out',
                f'surface_outfiles/{theory.filename}_{pointlabel}.out',
            )
        except TypeError:
            print_if_level("Theory has no outputfile, probably. ignoring", printlevel,2)
            pass
        except FileNotFoundError:
            pass
    if keepmofiles:
        try:
            shutil.copyfile(
                theory.filename + '.gbw',
                f'surface_mofiles/{theory.filename}_{pointlabel}.gbw',
            )
        except FileNotFoundError:
            pass








# ---------------------------------------------------------------------------
# Covalent radii (Angstrom) — used for connectivity detection
# Subset covering most common elements; extend as needed.
# ---------------------------------------------------------------------------
_COVALENT_RADII = {
    'H': 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
    'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
    'Mn': 1.61, 'Fe': 1.52, 'Co': 1.50, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
    'Hf': 1.75, 'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41,
    'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48,
}
_DEFAULT_RADIUS = 1.50   # fallback for unknown elements
_CONNECTIVITY_TOLERANCE = 0.40  # Angstrom added to sum of covalent radii
 
 
# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
 
def _measure_bond(coords, i, j):
    """Bond length in Angstrom between atoms i and j."""
    return float(np.linalg.norm(coords[i] - coords[j]))
 
 
def _measure_angle(coords, i, j, k):
    """Angle i-j-k in degrees (j is the vertex)."""
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
 
 
def _measure_dihedral(coords, i, j, k, l):
    """Dihedral angle i-j-k-l in degrees (range -180 to 180)."""
    b1 = coords[j] - coords[i]
    b2 = coords[k] - coords[j]
    b3 = coords[l] - coords[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))
 
 
# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------
 
def _build_connectivity(coords, elems):
    coords = np.asarray(coords)
    n = len(elems)
    radii = np.array([
        _COVALENT_RADII.get(e.capitalize(), _DEFAULT_RADIUS) for e in elems
    ])
    conn = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            threshold = radii[i] + radii[j] + _CONNECTIVITY_TOLERANCE
            # Ignore very short distances (e.g. same atom or ghost atoms)
            if 0.4 < dist < threshold:
                conn[i].add(j)
                conn[j].add(i)
    return conn

def _atoms_on_side(start, fixed, conn):
    """BFS: return set of atom indices reachable from *start* without
    crossing *fixed*.  Used to find which atoms move when a bond is stretched
    or a dihedral is rotated.
 
    Args:
        start : atom index to start BFS from
        fixed : atom index that acts as the barrier (not included in result)
        conn  : adjacency list from _build_connectivity
 
    Returns:
        set of atom indices (includes *start*, excludes *fixed*)
    """
    visited = {fixed}   # seed with fixed so BFS never crosses it
    queue = [start]
    visited.add(start)
    while queue:
        current = queue.pop()
        for neighbour in conn[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    visited.discard(fixed)
    return visited
 
 
# ---------------------------------------------------------------------------
# Bond length
# ---------------------------------------------------------------------------
 
def _set_bond(coords, i, j, target, conn):
    """Set bond length i-j to *target* Angstrom by translating the smaller
    connected fragment.
 
    The atom with the smaller connected component (determined by BFS through
    *conn* with the i-j bond removed) is moved together with all atoms on its
    side.
 
    Args:
        coords : (N, 3) numpy array, modified in-place
        i, j   : atom indices defining the bond
        target : target bond length in Angstrom
        conn   : adjacency list from _build_connectivity
    """
    current = _measure_bond(coords, i, j)
    if abs(current - target) < 1e-6:
        return
 
    # Find which side is smaller — move that side
    side_i = _atoms_on_side(i, fixed=j, conn=conn)
    side_j = _atoms_on_side(j, fixed=i, conn=conn)
 
    if len(side_i) <= len(side_j):
        move_atoms = side_i
        direction = coords[i] - coords[j]   # points toward i from j
    else:
        move_atoms = side_j
        direction = coords[j] - coords[i]   # points toward j from i
 
    unit = direction / np.linalg.norm(direction)
    delta = (target - current) * unit
    for atom in move_atoms:
        coords[atom] += delta
 
 
# ---------------------------------------------------------------------------
# Bond angle
# ---------------------------------------------------------------------------
 
def _set_angle(coords, i, j, k, target_deg, conn):
    current_deg = _measure_angle(coords, i, j, k)
    delta_deg = target_deg - current_deg
    if abs(delta_deg) < 1e-6:
        return

    v1 = coords[i] - coords[j]   # vector from vertex to i
    v2 = coords[k] - coords[j]   # vector from vertex to k

    # Rotation axis perpendicular to the i-j-k plane
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        # v1 and v2 are (anti)parallel — the plane is undefined.
        # Build an arbitrary perpendicular to v1 as the rotation axis.
        axis = _arbitrary_perpendicular(v1)
    else:
        axis = axis / axis_norm

    # Rotate the smaller side
    side_i = _atoms_on_side(i, fixed=j, conn=conn)
    side_k = _atoms_on_side(k, fixed=j, conn=conn)
    
    # Fallback: if connectivity failed, move just the single terminal atom
    if len(side_i) == 0:
        print(f"Warning: _set_angle: no atoms found on i-side of bond {j}-{i}. "
            f"Check connectivity. Falling back to moving atom {i} only.")
        side_i = {i}
    if len(side_k) == 0:
        print(f"Warning: _set_angle: no atoms found on k-side of bond {j}-{k}. "
            f"Check connectivity. Falling back to moving atom {k} only.")
        side_k = {k}

    if len(side_i) <= len(side_k):
        move_atoms = side_i
        angle_rad = np.radians(delta_deg)
    else:
        move_atoms = side_k
        angle_rad = np.radians(-delta_deg)

    # --- Sign check: trial rotation ---
    R_trial = _rotation_matrix(axis, angle_rad)
    pivot = coords[j]
    coords_trial = coords.copy()
    for atom in move_atoms:
        coords_trial[atom] = pivot + R_trial @ (coords_trial[atom] - pivot)

    achieved_trial = _measure_angle(coords_trial, i, j, k)
    error_pos = abs(achieved_trial - target_deg)
    error_neg = abs(_measure_angle(
        _apply_rotation(coords, move_atoms, pivot,
                        _rotation_matrix(axis, -angle_rad)), i, j, k
    ) - target_deg)

    # Pick the direction that gets closer to target
    if error_neg < error_pos:
        angle_rad = -angle_rad

    R = _rotation_matrix(axis, angle_rad)
    for atom in move_atoms:
        coords[atom] = pivot + R @ (coords[atom] - pivot)
 
def _apply_rotation(coords, move_atoms, pivot, R):
    """Return a copy of coords with move_atoms rotated — used for trial checks."""
    coords_trial = coords.copy()
    for atom in move_atoms:
        coords_trial[atom] = pivot + R @ (coords_trial[atom] - pivot)
    return coords_trial
# ---------------------------------------------------------------------------
# Dihedral angle
# ---------------------------------------------------------------------------
 
def _set_dihedral(coords, i, j, k, l, target_deg, conn):
    current_deg = _measure_dihedral(coords, i, j, k, l)
    delta_deg = target_deg - current_deg

    # Wrap into (-180, 180]
    delta_deg = (delta_deg + 180.0) % 360.0 - 180.0

    if abs(delta_deg) < 1e-6:
        return

    axis = coords[k] - coords[j]
    axis = axis / np.linalg.norm(axis)

    move_atoms = _atoms_on_side(l, fixed=k, conn=conn)

    # Trial rotation with +delta to check sign
    R_trial = _rotation_matrix(axis, np.radians(delta_deg))
    pivot = coords[k]
    coords_trial = coords.copy()
    for atom in move_atoms:
        coords_trial[atom] = pivot + R_trial @ (coords_trial[atom] - pivot)

    achieved_trial = _measure_dihedral(coords_trial, i, j, k, l)
    error_pos = abs((achieved_trial - target_deg + 180.0) % 360.0 - 180.0)

    # If positive delta moved us away, flip the sign
    if error_pos > abs(delta_deg) * 0.5:
        delta_deg = -delta_deg

    R = _rotation_matrix(axis, np.radians(delta_deg))
    for atom in move_atoms:
        coords[atom] = pivot + R @ (coords[atom] - pivot)
 
 
# ---------------------------------------------------------------------------
# Low-level math helpers
# ---------------------------------------------------------------------------
 
def _rotation_matrix(axis, angle_rad):
    """Rodrigues' rotation formula: 3x3 rotation matrix.
 
    Args:
        axis      : unit vector (length-3 array)
        angle_rad : rotation angle in radians
 
    Returns:
        (3, 3) numpy array
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])
 
 
def _arbitrary_perpendicular(v):
    """Return a unit vector perpendicular to *v* (for collinear edge case)."""
    v = np.asarray(v, dtype=float)
    if abs(v[0]) < 0.9:
        perp = np.array([1.0, 0.0, 0.0])
    else:
        perp = np.array([0.0, 1.0, 0.0])
    perp = np.cross(v, perp)
    return perp / np.linalg.norm(perp)
 
 
# ---------------------------------------------------------------------------
# A function to set the geometry directly
# ---------------------------------------------------------------------------
 
def _set_geometry_direct(fragment, RC_list, rc_values, conn=None):
    """Move fragment.coords to the target RC values without any optimiser.
 
    Supports constraint types: 'bond', 'angle', 'dihedral'.
    Connectivity is built once from the current geometry.
    All RC coordinates are applied sequentially; if multiple RCs share atoms
    they are applied in the order given (same order as RC_list).
 
    For symmetric constraints (multiple index sets per RC, e.g. two equivalent
    bonds) all index sets are applied for the same target value.
 
    Args:
        fragment  : ASH Fragment object with .coords (Angstrom) and .elems
        RC_list   : normalised RC_list (indices already list-of-lists)
        rc_values : tuple of target values, one per RC entry in RC_list
    """
    coords = np.array(fragment.coords, dtype=float)   # working copy
    elems  = fragment.elems
 
    # Build connectivity once — cheap, done from current geometry
    print("Connectivity of atoms 0,1,2:", {a: conn[a] for a in [0,1,2]})
 
    for rc, target in zip(RC_list, rc_values):
        rc_type = rc['type'].lower()
        for indices in rc['indices']:       # rc['indices'] is list-of-lists
            if rc_type in ('bond', 'distance'):
                i, j = indices
                _set_bond(coords, i, j, float(target), conn)
 
            elif rc_type == 'angle':
                i, j, k = indices
                _set_angle(coords, i, j, k, float(target), conn)
 
            elif rc_type in ('dihedral', 'torsion'):
                i, j, k, l = indices
                _set_dihedral(coords, i, j, k, l, float(target), conn)
 
            else:
                print(
                    f"Warning: _set_geometry_direct does not support constraint "
                    f"type '{rc_type}'. Skipping."
                )
 
    # Write the modified coordinates back into the fragment
    fragment.coords =coords
 
 
# ---------------------------------------------------------------------------
# Verifying the set geometry constraint
# ---------------------------------------------------------------------------
 
def _verify_geometry(fragment, RC_list, rc_values, tol=1e-3, printlevel=2):
    coords = np.array(fragment.coords)
    print_if_level("  RC pre-set verification:", printlevel,2)
    for i, (rc, target) in enumerate(zip(RC_list, rc_values)):
        rc_type = rc['type'].lower()
        for indices in rc['indices']:
            if rc_type in ('bond', 'distance'):
                achieved = _measure_bond(coords, *indices)
                deviation = abs(achieved - target)
            elif rc_type == 'angle':
                achieved = _measure_angle(coords, *indices)
                deviation = abs(achieved - target)
            elif rc_type == 'dihedral':
                achieved = _measure_dihedral(coords, *indices)
                # Normalize deviation to (-180, 180] — 190 and -170 are identical
                deviation = abs((achieved - target + 180.0) % 360.0 - 180.0)
            else:
                continue
            flag = " <-- WARNING" if deviation > tol else ""
            if printlevel > 1:
                print(
                    f"    RC{i+1} {rc_type} {indices}: "
                    f"target={target:.4f}  achieved={achieved:.4f}  "
                    f"dev={deviation:.4f}{flag}"
                )


# ---------------------------------------------------------------------------
# Implementation of a RestraintTheory: alternative way of setting restraints
# ---------------------------------------------------------------------------

def _preset_geometry_restraint(fragment, RC_list, rc_values, optimizerobj, 
                                opt_arguments, charge, mult,printlevel=1, extraconstraints=None,
                                extraoopt_run_kws=None,
                                force_constant=10000.0):
    """Drive geometry to target RC values using RestraintTheory + any optimiser."""
    restraints = []
    for rc, target in zip(RC_list, rc_values):
        for indices in rc['indices']:
            restraints.append({
                'type':    rc['type'],
                'indices': indices,
                'target':  float(target),
            })

    restraint_theory = RestraintTheory(
        restraints=restraints,
        force_constant=force_constant,
    )

    # Strip any constraints from opt_arguments — we don't want them here
    preset_args = {k: v for k, v in opt_arguments.items()
                   if k not in ('constraints', 'constrainvalue')}
    # Optimizing with restraint theory, passing extraconstraints as contraints if present
    optimizerobj.run(theory=restraint_theory,fragment=fragment, constraints=extraconstraints, **extraoopt_run_kws)

    #optimizer(
    #    fragment=fragment, theory=restraint_theory, constraints=extraconstraints,
    #    charge=charge, mult=mult, printlevel=printlevel,
    #    **preset_args,
    #)



class RestraintTheory:
    def __init__(self, fragment=None, printlevel=None, numcores=1, label=None,
                 restraints=None, force_constant=10000.0):
        """RestraintTheory: A theory that implements harmonic restraint potentials
        on internal coordinates (bonds, angles, dihedrals). Designed to be used
        with an optimiser  to drive geometry to target RC values.

        The energy and gradient are purely from harmonic restraints:
            E = 0.5 * k * (q - q0)^2
        where q is the current value of the internal coordinate and q0 is the
        target value. Angles and dihedrals use degree units internally but the
        force constant should be chosen accordingly (see below).

        Args:
            fragment       : ASH fragment. Defaults to None.
            printlevel     : print verbosity 0-3. Defaults to None.
            numcores       : number of cores (unused, for consistency). Defaults to 1.
            label          : string label. Defaults to None.
            restraints     : list of restraint dicts, each with keys:
                                 'type'    : 'bond', 'angle', or 'dihedral'
                                 'indices' : list of atom indices
                                 'target'  : target value (Å for bonds,
                                             degrees for angles/dihedrals)
                             Example:
                                 [{'type': 'bond',     'indices': [0, 1], 'target': 1.2, 'forceconstant': 50},
                                  {'type': 'angle',    'indices': [1, 0, 2], 'target': 104.5, 'forceconstant': 20},
                                  {'type': 'dihedral', 'indices': [0,1,2,3], 'target': 180.0, 'forceconstant': 10}]
            force_constant : Global harmonic force constant k. Only used if no forceconstant in individual restraintdict.
                             Defaults to 10000.0.
                             Units: energy/Å² for bonds, energy/deg² for angles
                             and dihedrals. The default is chosen to be stiff
                             enough to reach the target closely in a few steps.
                             Reduce if the optimiser has convergence problems.
        """
        self.numcores        = numcores
        self.printlevel      = printlevel
        self.label           = label
        self.fragment        = fragment
        self.filename        = "restrainttheory"
        self.theorynamelabel = "RestraintTheory"
        self.theorytype      = "QM"    # treated as QM so ASH passes coords/grad

        self.restraints      = restraints if restraints is not None else []
        self.force_constant  = force_constant

        self.energy   = 0.0
        self.gradient = None

    # ------------------------------------------------------------------
    # Internal coordinate measurement
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_bond(coords, i, j):
        return float(np.linalg.norm(coords[i] - coords[j]))

    @staticmethod
    def _measure_angle(coords, i, j, k):
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    @staticmethod
    def _measure_dihedral(coords, i, j, k, l):
        b1 = coords[j] - coords[i]
        b2 = coords[k] - coords[j]
        b3 = coords[l] - coords[k]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))

    # ------------------------------------------------------------------
    # Analytical gradients of internal coordinates w.r.t. Cartesian coords
    # ------------------------------------------------------------------

    @staticmethod
    def _bond_gradient(coords, i, j):
        """dq/dX for bond length q = |r_i - r_j|.
        Returns (natoms, 3) sparse gradient array."""
        natoms = len(coords)
        grad = np.zeros((natoms, 3))
        r = coords[i] - coords[j]
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-10:
            return grad
        unit = r / r_norm
        grad[i] += unit
        grad[j] -= unit
        return grad

    @staticmethod
    def _angle_gradient(coords, i, j, k):
        """dq/dX for angle q (degrees) at vertex j.
        Returns (natoms, 3) sparse gradient array."""
        natoms = len(coords)
        grad = np.zeros((natoms, 3))
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return grad

        cos_a = np.dot(v1, v2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0 + 1e-10, 1.0 - 1e-10)
        sin_a = np.sqrt(1.0 - cos_a**2)
        if sin_a < 1e-10:
            return grad

        # d(angle_rad)/dX, then convert to degrees
        # Using the standard Wilson B-matrix elements
        u1 = v1 / n1
        u2 = v2 / n2
        # Gradient w.r.t. atom i
        gi = (cos_a * u1 - u2) / (n1 * sin_a)
        # Gradient w.r.t. atom k
        gk = (cos_a * u2 - u1) / (n2 * sin_a)
        # Gradient w.r.t. vertex j (negative sum)
        gj = -(gi + gk)

        deg_per_rad = 180.0 / np.pi
        grad[i] += gi * deg_per_rad
        grad[j] += gj * deg_per_rad
        grad[k] += gk * deg_per_rad
        return grad

    @staticmethod
    def _dihedral_gradient(coords, i, j, k, l):
        """dq/dX for dihedral angle q (degrees) i-j-k-l.
        Returns (natoms, 3) sparse gradient array."""
        natoms = len(coords)
        grad = np.zeros((natoms, 3))

        b1 = coords[j] - coords[i]
        b2 = coords[k] - coords[j]
        b3 = coords[l] - coords[k]

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        b2_norm = np.linalg.norm(b2)

        if n1_norm < 1e-10 or n2_norm < 1e-10 or b2_norm < 1e-10:
            return grad

        n1_u = n1 / n1_norm
        n2_u = n2 / n2_norm
        b2_u = b2 / b2_norm

        # Standard Blondel & Karplus (1996) dihedral gradient
        gi =  (b2_norm / n1_norm**2) * n1
        gl = -(b2_norm / n2_norm**2) * n2
        gj = (-np.dot(b1, b2) / (b2_norm * n1_norm**2)) * n1 \
             + (np.dot(b3, b2) / (b2_norm * n2_norm**2)) * n2
        gk = -gj - gi - gl    # translational invariance: sum = 0

        deg_per_rad = 180.0 / np.pi
        grad[i] += gi * deg_per_rad
        grad[j] += gj * deg_per_rad
        grad[k] += gk * deg_per_rad
        grad[l] += gl * deg_per_rad
        return grad

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(self, current_coords=None, elems=None, Grad=False, PC=False,
            numcores=None, charge=None, mult=None, label=None,
            current_MM_coords=None, MMcharges=None, qm_elems=None):

        # Convert coords from Å to Bohr for all internal calculations
        coords = current_coords * ang2bohr
        natoms = len(coords)

        energy = 0.0
        gradient = np.zeros((natoms, 3))

        for r in self.restraints:
            rtype  = r['type'].lower()
            idx    = r['indices']
            # Check if forceconstant in r:
            k      = float(r.get('force_constant', self.force_constant))

            if rtype in ('bond', 'distance'):
                # target given in Å — convert to Bohr
                target = float(r['target']) * ang2bohr
                q      = self._measure_bond(coords, *idx)      # now in Bohr
                dq     = q - target                            # Bohr
                energy += 0.5 * k * dq**2                     # Eh  (k in Eh/Bohr²)
                if Grad:
                    dqdX = self._bond_gradient(coords, *idx)   # dimensionless (Bohr/Bohr)
                    gradient += k * dq * dqdX                  # Eh/Bohr

            elif rtype == 'angle':
                # target given in degrees — convert to radians
                target   = float(r['target']) * np.pi / 180.0
                q        = self._measure_angle(coords, *idx) * np.pi / 180.0  # rad
                dq       = (q - target + np.pi) % (2*np.pi) - np.pi           # rad
                energy  += 0.5 * k * dq**2                    # Eh  (k in Eh/rad²)
                if Grad:
                    # _angle_gradient currently returns deg/Å — convert to rad/Bohr
                    dqdX = self._angle_gradient(coords, *idx)  # deg/Bohr (coords in Bohr)
                    dqdX *= np.pi / 180.0                      # → rad/Bohr
                    gradient += k * dq * dqdX                  # Eh/Bohr

            elif rtype in ('dihedral', 'torsion'):
                # same as angle
                target   = float(r['target']) * np.pi / 180.0
                q        = self._measure_dihedral(coords, *idx) * np.pi / 180.0
                dq       = (q - target + np.pi) % (2*np.pi) - np.pi
                energy  += 0.5 * k * dq**2                    # Eh  (k in Eh/rad²)
                if Grad:
                    dqdX = self._dihedral_gradient(coords, *idx)  # deg/Bohr
                    dqdX *= np.pi / 180.0                         # → rad/Bohr
                    gradient += k * dq * dqdX                     # Eh/Bohr

        self.energy   = energy     # Eh
        self.gradient = gradient   # Eh/Bohr

        if not Grad:
            return self.energy
        else:
            return self.energy, self.gradient


# ---------------------------------------------------------------------------
# Surface analysis
# ---------------------------------------------------------------------------


def analyze_surface(resultfile='surface_results.txt', dimension=None,
                    energy_unit='kcal/mol', tol=1e-6):
    """Analyze a surface scan result file for minima, maxima, and saddle points.

    Works for any dimension but critical point classification beyond 1D relies
    on finite-difference estimation of the Hessian on the grid, so results are
    only as good as the grid resolution.

    Args:
        resultfile  : path to surface_results.txt
        dimension   : number of RC coordinates (inferred if None)
        energy_unit : 'kcal/mol', 'kJ/mol', or 'Eh' for relative energies
        tol         : energy tolerance for detecting flat regions

    Returns:
        dict with keys 'global_min', 'local_minima', 'global_max',
                        'local_maxima', 'saddle_points'
        Each entry is a list of dicts with 'coords', 'energy', 'rel_energy'.
    """

    # -- Unit conversion ----------------------------------------------------
    conv = {'kcal/mol': 627.509, 'kJ/mol': 2625.50, 'Eh': 1.0}
    if energy_unit not in conv:
        print(f"Warning: unknown energy_unit '{energy_unit}', using kcal/mol")
        energy_unit = 'kcal/mol'
    factor = conv[energy_unit]

    # -- Read data ----------------------------------------------------------
    surfacedictionary = read_surfacedict_from_file(resultfile, dimension)
    if dimension is None:
        dimension = len(list(surfacedictionary.keys())[0])

    print(f"Read {len(surfacedictionary)} points, dimension={dimension}")

    if dimension == 1:
        return _analyze_1d(surfacedictionary, factor, energy_unit, tol)
    else:
        return _analyze_nd(surfacedictionary, dimension, factor, energy_unit, tol)


# ---------------------------------------------------------------------------
# 1D analysis
# ---------------------------------------------------------------------------

def _analyze_1d(surfacedictionary, factor, energy_unit, tol):
    # 1. Sort and extract
    keys = sorted(surfacedictionary.keys())
    coords = np.array([k if isinstance(k, tuple) else (k,) for k in keys])
    energies = np.array([surfacedictionary[k] for k in keys])
    
    # 2. Periodicity Detection & Trimming
    # If the first and last points are the same physical location (e.g., -180 and 180),
    # we remove the last point to avoid "neighboring itself" in the cycle.
    is_periodic = (abs(abs(coords[-1][0] - coords[0][0]) - 360.0) < 1.0)
    
    if is_periodic:
        print("Periodic scan detected. Wrapping boundaries for analysis.")
        analysis_energies = energies[:-1]
        analysis_coords = coords[:-1]
    else:
        analysis_energies = energies
        analysis_coords = coords

    n = len(analysis_energies)
    local_minima = []
    local_maxima = []

    # 3. Find Critical Points
    for idx in range(n):
        e = analysis_energies[idx]
        
        if is_periodic:
            left  = analysis_energies[(idx - 1) % n]
            right = analysis_energies[(idx + 1) % n]
        else:
            if idx == 0 or idx == n - 1: continue
            left  = analysis_energies[idx - 1]
            right = analysis_energies[idx + 1]

        # Use >= or <= with tol to be inclusive of "flat" minima/maxima if needed, 
        # but strict inequality is usually safer for discrete scans.
        is_min = (e < left - tol) and (e < right - tol)
        is_max = (e > left + tol) and (e > right + tol)

        entry = {'coords': tuple(analysis_coords[idx]), 'energy': e}

        if is_min:
            local_minima.append(entry)
        elif is_max:
            local_maxima.append(entry)

    # 4. Global vs Local Assignment
    if not local_minima:
        # Fallback if no local minima found due to high tol
        idx_min = np.argmin(analysis_energies)
        local_minima = [{'coords': tuple(analysis_coords[idx_min]), 'energy': analysis_energies[idx_min]}]

    local_minima.sort(key=lambda x: x['energy'])
    local_maxima.sort(key=lambda x: x['energy'], reverse=True)

    global_min = local_minima[0]
    global_max = local_maxima[0] if local_maxima else None

    # 5. Compute Relative Energies
    for entry in local_minima + local_maxima:
        entry['rel_energy'] = (entry['energy'] - global_min['energy']) * factor

    result = {
        'global_min':    global_min,
        'local_minima':  local_minima[1:],
        'global_max':    global_max,
        'local_maxima':  local_maxima[1:],
        'saddle_points': [],
    }
    
    # Assuming _print_analysis is defined elsewhere
    _print_analysis(result, factor, energy_unit, dimension=1)
    return result

# ---------------------------------------------------------------------------
# ND analysis (2D, 3D, ...)
# ---------------------------------------------------------------------------


def _analyze_nd(surfacedictionary, dimension, factor, energy_unit, tol):
    import itertools as it
    # --- 1. Grid Setup ---
    all_keys = sorted(surfacedictionary.keys())
    axes = [np.array(sorted({k[d] for k in all_keys})) for d in range(dimension)]
    shape = tuple(len(a) for a in axes)
    index_maps = [{v: i for i, v in enumerate(a)} for a in axes]
    grid = np.full(shape, np.nan)
    for key, energy in surfacedictionary.items():
        idx = tuple(index_maps[d][key[d]] for d in range(dimension))
        grid[idx] = float(energy)

    global_min_e = np.nanmin(grid)
    local_minima, local_maxima, saddle_candidates = [], [], []

    # --- 2. Iterate Interior Points ---
    ranges = [range(1, s - 1) for s in shape]
    for idx in it.product(*ranges):
        e0 = grid[idx]
        if np.isnan(e0): continue

        # A. Calculate Gradient Norm (Stationary Check)
        grads = []
        for d in range(dimension):
            i_p, i_m = list(idx), list(idx)
            i_p[d] += 1; i_m[d] -= 1
            h = axes[d][idx[d]+1] - axes[d][idx[d]-1]
            grads.append((grid[tuple(i_p)] - grid[tuple(i_m)]) / h)
        
        gnorm = np.linalg.norm(grads)

        # B. Strict Neighbor Comparison (topology test)
        # Check neighbors along principal axes
        nb_vals = []
        for d in range(dimension):
            i_p, i_m = list(idx), list(idx)
            i_p[d] += 1; i_m[d] -= 1
            nb_vals.append((grid[tuple(i_m)], grid[tuple(i_p)]))

        # Determine if it's an extreme or a saddle
        # is_min: lower than all immediate neighbors
        is_min = all(e0 < v_m - tol and e0 < v_p - tol for v_m, v_p in nb_vals)
        # is_max: higher than all immediate neighbors
        is_max = all(e0 > v_m + tol and e0 > v_p + tol for v_m, v_p in nb_vals)
        
        # is_saddle: max in one dir, min in another (for 2D)
        # We check: (Min in X and Max in Y) OR (Max in X and Min in Y)
        is_saddle = False
        if dimension == 2:
            (x_m, x_p), (y_m, y_p) = nb_vals
            saddle_1 = (e0 < x_m and e0 < x_p) and (e0 > y_m and e0 > y_p)
            saddle_2 = (e0 > x_m and e0 > x_p) and (e0 < y_m and e0 < y_p)
            is_saddle = saddle_1 or saddle_2

        coords = tuple(axes[d][idx[d]] for d in range(dimension))
        entry = {'coords': coords, 'energy': e0, 'rel_energy': (e0 - global_min_e)*factor, 'gnorm': gnorm}

        if is_min:
            local_minima.append(entry)
        elif is_max:
            local_maxima.append(entry)
        elif is_saddle:
            saddle_candidates.append(entry)

    # --- 3. Non-Maximum Suppression (Clustering) ---
    # This is the "Magic" step that deletes duplicates in flat regions
    def cluster_points(points, is_saddle=False):
        if not points: return []
        # Sort by gradient norm (we want the point closest to a true stationary point)
        points.sort(key=lambda x: x['gnorm'])
        unique = []
        for p in points:
            is_redundant = False
            for u in unique:
                # If point is within 2 grid steps of a better one, discard it
                dist = np.array([abs(p['coords'][d] - u['coords'][d]) for d in range(dimension)])
                step = np.array([axes[d][1] - axes[d][0] for d in range(dimension)])
                if all(dist <= step * 2.1): # 2-step radius
                    is_redundant = True
                    break
            if not is_redundant:
                unique.append(p)
        return unique

    refined_minima = cluster_points(local_minima)
    refined_maxima = cluster_points(local_maxima)
    refined_saddles = cluster_points(saddle_candidates, is_saddle=True)

    # Final result construction
    refined_minima.sort(key=lambda x: x['energy'])
    refined_maxima.sort(key=lambda x: x['energy'], reverse=True)
    refined_saddles.sort(key=lambda x: x['energy'])

    result = {
        'global_min': refined_minima[0] if refined_minima else None,
        'local_minima': refined_minima[1:],
        'global_max': refined_maxima[0] if refined_maxima else None,
        'local_maxima': refined_maxima[1:],
        'saddle_points': refined_saddles,
    }
    _print_analysis(result, factor, energy_unit, dimension=dimension)
    return result


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_analysis(result, factor, energy_unit, dimension):
    col_w = 12

    def fmt_coords(coords):
        return '  '.join(f'{v:>10.4f}' for v in coords)

    def fmt_entry(entry, tag=''):
        c = fmt_coords(entry['coords'])
        e = f"{entry['energy']:>18.10f} Eh"
        r = f"{entry['rel_energy']:>12.4f} {energy_unit}"
        order_str = ''
        if 'order' in entry:
            order_str = f"  ({entry['order']}-order SP)"
        return f"  {c}  {e}  {r}  {tag}{order_str}"

    print()
    print("=" * 80)
    print("SURFACE ANALYSIS")
    print("=" * 80)

    print("\nMINIMA")
    print("-" * 80)
    if result['global_min']:
        print(fmt_entry(result['global_min'], tag='(global min)'))
    if result['local_minima']:
        for entry in result['local_minima']:
            print(fmt_entry(entry, tag='(local min)'))
    if not result['global_min'] and not result['local_minima']:
        print("  None found (may be on boundary or grid too coarse)")

    print("\nMAXIMA")
    print("-" * 80)
    if result['global_max']:
        print(fmt_entry(result['global_max'], tag='(global max)'))
    if result['local_maxima']:
        for entry in result['local_maxima']:
            print(fmt_entry(entry, tag='(local max)'))
    if not result['global_max'] and not result['local_maxima']:
        print("  None found (may be on boundary or grid too coarse)")

    print("\nSADDLE POINTS")
    print("-" * 80)
    if result['saddle_points']:
        for entry in result['saddle_points']:
            print(fmt_entry(entry))
    else:
        print("  None found")

    print("=" * 80)
    print()