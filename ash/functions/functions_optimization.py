import numpy as np
import time
import os

import ash.constants
from ash.functions.functions_general import ashexit, blankline,print_time_rel_and_tot,BC,listdiff,print_time_rel
from ash.modules.module_coords import check_charge_mult , write_xyzfile, print_internal_coordinate_table_new
from ash.modules.module_coords_PBC import cell_vectors_to_params, cart_coords_to_fract, fract_coords_to_cart, cell_volume, \
                                          write_CIF_file,write_XSF_file, write_POSCAR_file
from ash.modules.module_coords import print_coords_for_atoms
from ash.interfaces.interface_geometric_new import geomeTRICOptimizer
from ash.modules.module_results import ASH_Results

#Root mean square of numpy array, e.g. gradient
def RMS_G(grad):
    sumsq = 0
    count = 0
    for row in grad:
        count += len(row)
        for col in row:
            sumsq += col ** 2
    rms = np.sqrt(sumsq / count)
    return rms

#Max abs value
def Max_G(grad):
    maxg=abs(max(grad.min(), grad.max(), key=abs))
    return maxg

def write_xyz_trajectory(file, coords, elems, titleline):
    with open (file, 'a') as f:
        f.write(str(len(elems))+'\n')
        f.write('Energy: '+str(titleline)+'\n')
        for el,c in zip(elems,coords):
            line="{:4} {:12.6f} {:12.6f} {:12.6f}".format(el,c[0], c[1], c[2])
            f.write(line+'\n')

#########################
# GEOMETRY OPTIMIZERS    #
#########################


#ASH Cartesian Optimizer function for basic usage
def SimpleOpt(fragment=None, theory=None, charge=None, mult=None, optimizer='KNARR-LBFGS', maxiter=50, 
              frozen_atoms=None, actatoms=None, RMSGtolerance=0.0001, MaxGtolerance=0.0003, FIRE_timestep=0.00009):

    if fragment is not None:
        pass
    else:
        print("No fragment provided to optimizer")
        print("Checking if theory object contains defined fragment")
        try:
            fragment=theory.fragment
            print("Found theory fragment")
        except:
            print("Found no fragment in theory object either.")
            print("Exiting...")
            ashexit()
    if frozen_atoms is None:
        frozen_atoms=[]
    if actatoms is not None:
        print("Actatoms provided:", actatoms)
        frozen_atoms = listdiff(fragment.allatoms,actatoms)

    print_atoms_list = fragment.allatoms

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "SimpleOpt", theory=theory)

    #List of active vs. frozen labels
    actfrozen_labels=[]
    #for i in range(fragment.numatoms):
    #    print("i", i)
    #    if i in frozen_atoms:
    #        actfrozen_labels.append('Frozen')
    #    else:
     #       actfrozen_labels.append('Active')

    beginTime = time.time()
    print(BC.OKMAGENTA, BC.BOLD, "------------STARTING OPTIMIZER-------------", BC.END)
    print_option='Small'
    print("Running Optimizer")
    print("Optimization algorithm:", optimizer)
    if len(frozen_atoms)> 0:
        print("Frozen atoms:", frozen_atoms)
    blankline()
    #Printing info and basic initalization of parameters
    if optimizer=="SD":
        print("Using very basic stupid Steepest Descent algorithm")
        sdscaling=0.85
        print("SD Scaling parameter:", sdscaling)
    elif optimizer=="KNARR-LBFGS":
        print("Using LBFGS optimizer from Knarr by Vilhjálmur Ásgeirsson")
        print("LBFGS parameters (currently hardcoded)")
        print(LBFGS_parameters)
        reset_opt = False
    elif optimizer=="SD2":
        sdscaling = 0.01
        print("Using different SD optimizer")
        print("SD Scaling parameter:", sdscaling)
    elif optimizer=="KNARR-FIRE":
        time_step=FIRE_timestep
        was_scaled=False
        print("FIRE Parameters for timestep:", time_step)
        print(GetFIREParam(time_step))
    else:
        print("Unknown optimizer")
        ashexit()
    print("Tolerances:  RMSG: {}  MaxG: {}  Eh/Bohr".format(RMSGtolerance, MaxGtolerance))
    #Name of trajectory file
    trajname="opt-trajectory.xyz"
    print("Writing XYZ trajectory file: ", trajname)
    #Remove old trajectory file if present
    try:
        os.remove(trajname)
    except:
        pass
    #Current coordinates
    current_coords=fragment.coords
    elems=fragment.elems

    #OPTIMIZATION LOOP
    for step in range(1,maxiter):
        CheckpointTime = time.time()
        blankline()
        print("GEOMETRY OPTIMIZATION STEP", step)
        print(f"Current geometry (Å) in step {step} (print_atoms_list region)")
        print("-------------------------------------------------")
        print_coords_for_atoms(current_coords, fragment.elems, print_atoms_list)

        #Running E+G theory job.
        E, Grad = theory.run(current_coords=current_coords, elems=fragment.elems, Grad=True, charge=charge, mult=mult)
        #print("E,", E)
        #print("Grad,", Grad)

        #Applying frozen atoms constraint. Setting gradient to zero on atoms
        if len(frozen_atoms) > 0:
            print("Applying frozen-atom constraints")
            #Setting gradient for atoms to zero
            for num,gradcomp in enumerate(Grad):
                if num in frozen_atoms:
                    Grad[num]=[0.0,0.0,0.0]
        #print("Grad (after frozen constraints)", Grad)
        #Converting to atomic forces in eV/Angstrom. Used by Knarr
        forces_evAng=Grad * (-1) * ash.constants.hartoeV / ash.constants.bohr2ang
        blankline()
        print("Step: {}    Energy: {} Eh.".format(step, E))
        if print_option=='Big':
            blankline()
            print("Gradient (Eh/Bohr): \n{}".format(Grad))
            blankline()
        RMSG=RMS_G(Grad)
        MaxG=Max_G(Grad)

        #Write geometry to trajectory (with energy)
        write_xyz_trajectory(trajname, current_coords, elems, E)

        # Convergence threshold check
        if RMSG < RMSGtolerance and MaxG < MaxGtolerance:
            print("RMSG: {:3.9f}       Tolerance: {:3.9f}    YES".format(RMSG, RMSGtolerance))
            print("MaxG: {:3.9f}       Tolerance: {:3.9f}    YES".format(MaxG, MaxGtolerance))
            print(BC.OKGREEN,"Geometry optimization Converged!",BC.END)
            write_xyz_trajectory(trajname, current_coords, elems, E)

            # Updating energy and coordinates of ASH fragment before ending
            fragment.set_energy(E)
            print("Final optimized energy:", fragment.energy)
            fragment.replace_coords(elems, current_coords, conn=False)

            # Writing out fragment file and XYZ file
            fragment.print_system(filename='Fragment-optimized.ygg')
            fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')

            blankline()
            print_time_rel_and_tot(CheckpointTime, beginTime, 'Opt Step')
            return
        elif RMSG > RMSGtolerance and MaxG < MaxGtolerance:
            print("RMSG: {:3.9f}       Tolerance: {:3.9f}    NO".format(RMSG, RMSGtolerance))
            print("MaxG: {:3.9f}       Tolerance: {:3.9f}    YES".format(MaxG, MaxGtolerance))
            print(BC.WARNING,"Not converged",BC.END)
        elif RMSG < RMSGtolerance and MaxG > MaxGtolerance:
            print("RMSG: {:3.9f}       Tolerance: {:3.9f}    YES".format(RMSG, RMSGtolerance))
            print("MaxG: {:3.9f}       Tolerance: {:3.9f}    NO".format(MaxG, MaxGtolerance))
            print(BC.WARNING,"Not converged",BC.END)
        else:
            print("RMSG: {:3.9f}       Tolerance: {:3.9f}    NO".format(RMSG, RMSGtolerance))
            print("MaxG: {:3.9f}       Tolerance: {:3.9f}    NO".format(MaxG, MaxGtolerance))
            print(BC.WARNING,"Not converged",BC.END)

        blankline()
        if optimizer=='SD':
            print("Using Basic Steepest Descent optimizer")
            print("Scaling parameter:", sdscaling)
            current_coords=steepest_descent(current_coords,Grad,sdscaling)
        elif optimizer=='SD2':
            print("Using Basic Steepest Descent optimizer, SD2 with norm")
            print("Scaling parameter:", sdscaling)
            current_coords=steepest_descent2(current_coords,Grad,sdscaling)
        elif optimizer=="KNARR-FIRE":
            print("Taking FIRE step")
            # FIRE
            if step == 1 or reset_opt:
                reset_opt = False
                fire_param = GetFIREParam(time_step)
                ZeroVel=np.zeros( (3*len(current_coords),1))
                CurrentVel=ZeroVel
            if was_scaled:
                time_step *= 0.95
            velo, time_step, fire_param = GlobalFIRE(forces_evAng, CurrentVel, time_step, fire_param)
            CurrentVel=velo
            step, velo = EulerStep(CurrentVel, forces_evAng, time_step)
            CurrentVel=velo

        elif optimizer=='NR':
            print("disabled")
            ashexit()
            #Identity matrix
            #Hess_approx=np.identity(3*len(current_coords))
            #TODO: Not active
            #current_coords = newton_raphson(current_coords, Grad, Hess_approx)
        elif optimizer=='KNARR-LBFGS':
            if step == 1 or reset_opt:
                if reset_opt == True:
                    print("Resetting optimizer")
                print("Taking SD-like step")
                reset_opt = False
                sk = []
                yk = []
                rhok = []
                #Store original atomic forces (in eV/Å)
                keepf=np.copy(forces_evAng)
                keepr = np.copy(current_coords)
                step = TakeFDStep(theory, current_coords, LBFGS_parameters["fd_step"], forces_evAng, fragment.elems, charge, mult)

            else:
                print("Doing LBFGS Update")
                sk, yk, rhok = LBFGSUpdate(current_coords, keepr, forces_evAng, keepf,
                                           sk, yk, rhok, LBFGS_parameters["lbfgs_memory"])
                keepf=np.copy(forces_evAng)
                keepr = np.copy(current_coords)
                print("Taking LBFGS Step")
                step, negativecurv = LBFGSStep(forces_evAng, sk, yk, rhok)
                step *= LBFGS_parameters["lbfgs_damping"]
                if negativecurv:
                    reset_opt = True
        else:
            print("Optimizer option not supported.")
            ashexit()
        #Take the actual step
        #Todo: Implement maxmove-scaling here if step too large
        current_coords=current_coords+step
        #Write current geometry (after step) to disk as 'Current_geometry.xyz'.
        # Can be used if optimization failed, SCF convergence problemt etc.
        write_xyzfile(elems, current_coords, 'Current_geometry')
        blankline()
        print_time_rel_and_tot(CheckpointTime, beginTime, 'Opt Step')
    print(BC.FAIL,"Optimization did not converge in {} iteration".format(maxiter),BC.END)


#Very basic bad steepest descent algorithm.
#Arbitrary scaling parameter instead of linesearch
#0.8-0.9 seems to work well for H2O
def steepest_descent(coords, Gradient,scaling):
    newcoords = coords - scaling*Gradient
    return newcoords

#Normalized forces
def steepest_descent2(coords, Gradient, scaling):
    current_forces=Gradient * (-1) * ash.constants.hartoeV / ash.constants.bohr2ang
    Fu = current_forces / np.linalg.norm(current_forces)
    step = scaling * Fu
    new_config = coords + step
    return new_config

#Optimizer SD, LBFGS and FIRE routines from Villi

LBFGS_parameters = {'fd_step' : 0.001, 'lbfgs_memory' : 20, 'lbfgs_damping' : 1.0}
SD_parameters = {'sd_step' : 0.001}

# Author: Vilhjalmur Asgeirsson, 2019
#Modified to fit ASH
def TakeFDStep(theory, current_coords, fd_step, forces, elems, charge, mult):
    # keep config / forces
    current_forces = np.copy(forces)
    #Current config is in Å
    current_config = np.copy(current_coords)
    # Get direction of force and generate step in that direction
    Fu = current_forces / np.linalg.norm(current_forces)
    step = fd_step * Fu
    # Take step - but we do not save it.
    new_config = current_config + step

    # Compute forces and energy at new step
    E, Grad = theory.run(current_coords=new_config, elems=elems, Grad=True, charge=charge, mult=mult)

    # Restore previous values and store new forces
    new_forces = Grad*(-1)*ash.constants.hartoeV/ash.constants.bohr2ang
    new_forces_unflat=new_forces.reshape(len(new_forces)*3,-1)
    current_forces_unflat=current_forces.reshape(len(current_forces)*3,-1)
    Fu_unflat=Fu.reshape(len(Fu)*3,-1)
    H = 1.0 / (np.dot((-new_forces_unflat + current_forces_unflat).T, Fu_unflat) / fd_step)
    # If curvature is positive - get new step
    if H > 0.0:
        step = np.multiply(H, current_forces)
    return step

def LBFGSUpdate(R, R0, F, F0, sk, yk, rhok, memory):
    dr = R - R0
    df = -(F - F0)
    dr_unflat = dr.reshape(len(dr) * 3, -1)
    df_unflat = df.reshape(len(df) * 3, -1)
    if abs(np.dot(dr_unflat.T, df_unflat)) < 1e-30:
        raise ZeroDivisionError()
    else:
        sk.append(dr_unflat)
        yk.append(df_unflat)
        rhok.append(1.0 / np.dot(dr_unflat.T, df_unflat))

    if len(sk) > memory:
        sk.pop(0)
        yk.pop(0)
        rhok.pop(0)
    return sk, yk, rhok


def LBFGSStep(F, sk, yk, rhok):
    #Unflattening forces array
    F_flat=F.reshape(len(F) * 3, -1)
    neg_curv = False
    C = rhok[-1] * np.dot(yk[-1].T, yk[-1])
    H0 = 1.0 / C
    if H0 < 0.0:
        print('**Warning: Negative curvature. Restarting optimizer.')
        neg_curv = True

    lengd = len(sk)
    q = -F_flat.copy()
    alpha = np.zeros(shape=(lengd, 1))

    for i in range(lengd - 1, -1, -1):
        alpha[i] = rhok[i] * np.dot(sk[i].T, q)
        q = q - (alpha[i] * yk[i])

    r = H0 * q

    for i in range(0, lengd):
        beta = rhok[i] * np.dot(yk[i].T, r)
        r = r + sk[i] * (alpha[i] - beta)

    step = -r

    #Flatten step back
    step=step.reshape(len(F), 3)

    return step, neg_curv

#FIRE

# Author: Vilhjalmur Asgeirsson

def GetFIREParam(time_step):
    fire_param = {"ALPHA": 0.1, "n": 0, "ALPHA_START": 0.1,
                  "FINC": 1.1, "FDEC": 0.5, "N": 5,
                  "FALPHA": 0.99, "MAX_TIME_STEP": 10 * time_step}
    return fire_param


def GlobalFIRE(F, velo, dt, fire_param):
    F = F.reshape(len(F) * 3, -1)
    alpha = fire_param["ALPHA"]
    n = fire_param["n"]
    alpha_start = fire_param["ALPHA_START"]
    finc = fire_param["FINC"]
    fdec = fire_param["FDEC"]
    N = fire_param["N"]
    falpha = fire_param["FALPHA"]
    dtmax = fire_param["MAX_TIME_STEP"]

    F_unit = F / np.linalg.norm(F)
    P = np.dot(F.T, velo)
    velo = (1.0 - alpha) * velo + alpha * F_unit * np.linalg.norm(velo)

    if P >= 0.0:
        n = n + 1
        if (n > N):
            dt = np.min([dt * finc, dtmax])
            alpha = alpha * falpha
    else:
        dt = dt * fdec
        velo = velo * 0.0
        alpha = alpha_start
        n = 0

    fire_param["ALPHA"] = alpha
    fire_param["n"] = n

    return velo, dt, fire_param

def EulerStep(velo, F, dt):
    F = F.reshape(len(F) * 3, -1)
    velo += F * dt
    step = velo * dt

    #Flatten step back
    step=step.reshape(int(len(velo)/3), 3)
    return step, velo


#########################
# PYBERNY Optimization interface.
# Has internal coordinates
#PyBerny: https://github.com/jhrmnn/pyberny/blob/master/README.md
#Installed via pip
#Limitations: No constraints or frozen atoms
#Todo: Add active-region option like geometric
########################

def BernyOpt(theory,fragment, charge=None, mult=None):
    blankline()
    print("Beginning Py-Berny Optimization")
    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "BernyOpt", theory=theory)
    try:
        from berny import Berny, geomlib
    except:
        blankline()
        print(BC.FAIL,"pyberny module not found!", BC.END)
        print(BC.WARNING,"Either install pyberny using pip:\n pip install pyberny\n "
                         "or manually from Github (https://github.com/jhrmnn/pyberny)", BC.END)
        ashexit(code=9)
    print("See: https://github.com/jhrmnn/pyberny")
    #Options: Berny(ethanol, steprms=0.01, stepmax=0.05, maxsteps=5)
    optimizer = Berny(geomlib.Geometry(fragment.elems,fragment.coords))
    for geom in optimizer:
        # get energy and gradients for geom
        E, Grad = theory.run(current_coords=geom.coords, elems=fragment.elems, Grad=True, charge=charge, mult=mult)
        optimizer.send((E,Grad))
    print("BernyOpt Geometry optimization converged!")
    #Updating energy and coordinates of ASH fragment before ending
    fragment.set_energy(E)
    print("Final optimized energy:",  fragment.energy)
    fragment.replace_coords(fragment.elems,geom.coords)
    blankline()

#############################
# PERIODIC OPTIMIZERS
#############################
# Alternating periodic cell optimizer: first atom-opt, then cell-step etc.
# Not really that useful
def periodic_optimizer_alternating(fragment=None, theory=None, rate=0.5, maxiter=50, tol=1e-3, step_algo="SD",
                                force_orthorhombic=True, max_step=0.25, momentum=0.5,
                                atoms_tolsetting=None, atom_opt_maxiter=100):
    ang2bohr=1.88972612546
    print("Learning rate:", rate)
    print("maxiter:", maxiter)

    # Max step in bohrs (default = 0.1 Å = 0.188 bohrs)
    print("Rate:", rate)
    max_step_au = max_step*ang2bohr
    print("force_orthorhombic:", force_orthorhombic)
    print(f"Tolerance: {tol} Eh/Bohr")
    print("Maxiter:", maxiter)
    print(f"Max step size {max_step} Å")
    print()
    print(f"Initial cell vectors in Theory object: {theory.periodic_cell_vectors} Å")

    cell_vectors_au = theory.periodic_cell_vectors*ang2bohr
    cell_vectors = theory.periodic_cell_vectors

    # Looping
    velocity = np.zeros((3, 3))
    print("Initial cell_vectors:", cell_vectors)
    for i in range(0,maxiter):
        print("="*40)
        print("Cell optimization step", i)
        print("="*40)
        # Optimize atom coordinates with frozen cell
        print("a) Will now optimize atom coordinates")
        # Note: forcing PBC to be off in geometric
        res = geomeTRICOptimizer(theory=theory, fragment=fragment, force_noPBC=True, convergence_setting=atoms_tolsetting, maxiter=atom_opt_maxiter)

        # Check convergence of cell gradient
        cell_gradient = theory.get_cell_gradient()
        grad_norm = np.linalg.norm(cell_gradient)
        print(f"Current Cell Gradient Norm: {grad_norm:.6f}")
        if grad_norm < tol:
            print(f"Cell converged in {i} cell-iterations  (Gradient norm: {grad_norm:.6f} < tol={tol} Eh/Bohr)")
            print(f"Final cell vectors: {cell_vectors} Å  and parameters: ({cell_vectors_to_params(cell_vectors)})")
            print(f"Final energy: {res.energy} Eh")
            break

        # Convert previously optimized Cart coords to Fract coords
        fract_coords = cart_coords_to_fract(fragment.coords,theory.periodic_cell_vectors)

        print("b) Will now take cell vector step")

        # Calculate cell vector step (in Bohrs)
        if step_algo.lower() =="sd":
            print("Doing steepest descent step")
            delta_au = - (rate * cell_gradient)
        elif step_algo.lower() == "damped-MD":
            print("Doing momentum step")
            print("velocity:", velocity)
            velocity = (momentum * velocity) - (rate * cell_gradient)
            print("velocity:", velocity)
            delta_au = velocity
        elif step_algo.lower() == "nesterov":
            # Storing old
            velocity_old = velocity.copy()
            print("Doing Nesterov momentum step")
            velocity = (momentum * velocity) - (rate * cell_gradient)
            nesterov_update = -momentum * velocity_old + (1 + momentum) * velocity
            delta_au = nesterov_update
        elif step_algo.lower() == "cg":
            print("Doing conjugate gradient step")
            if i == 0:
                search_dir = cell_gradient
                prev_grad=None
            else:
                # Polak-Ribière formula for beta
                diff = cell_gradient - prev_grad
                beta = np.sum(cell_gradient * diff) / np.sum(prev_grad * prev_grad)
                beta = max(0, beta) # Standard 'reset' for CG
                search_dir = cell_gradient + (beta * search_dir)

            delta_au = - (rate * search_dir)
            prev_grad = cell_gradient.copy()
        else:
            print("Unknown step_algo")
            ashexit()


        print("delta_au:", delta_au)

        # Force orthorhomic
        if force_orthorhombic:
            print("force_orthorhombic True")
            diagonal_mask = np.eye(3)
            delta_au = delta_au*diagonal_mask

        # Scale down step if required
        if np.max(np.abs(delta_au)) > max_step_au:
            print(f"Step scale down:  {np.max(np.abs(delta_au))}  > max_step_au: {max_step_au})")
            delta_au = delta_au * (max_step / np.max(np.abs(delta_au)))
            print("Actual step:", delta_au)

        # Take step
        cell_vectors_au += delta_au
        # Convert final cell vectors from Bohrs to Å
        cell_vectors = cell_vectors_au / ang2bohr
        print("Current cell vectors (Å):", cell_vectors)
        print("Current cell volume (Å):", cell_volume(cell_vectors))
        # Update Theory with new cell vectors in Å
        theory.update_cell(periodic_cell_vectors=cell_vectors)
        print("theory.periodic_cell_vectors:", theory.periodic_cell_vectors)

        # Update fragment with new XYZ coords that match cell
        new_cart_coords = fract_coords_to_cart(fract_coords,theory.periodic_cell_vectors)
        print("new_cartesian_coords:", new_cart_coords)
        fragment.coords=new_cart_coords

# Cartesian-based periodic cell optimizer


# Wrapper function around Cart_optimizer_class
def Cart_optimizer(fragment=None, theory=None, rate=2.0, 
                                scaling_rate_cell=1.0, maxiter=50, 
                                step_algo="bfgs",
                                max_step=0.25, momentum=0.5, constrain_method='soft',
                                printlevel=2, conv_criteria=None, PBC_format_option="CIF",
                                constraints=None, frozen_atoms=None, result_write_to_disk=True):
    """
    Wrapper function around Cart_optimizer_class
    """
    timeA=time.time()

    # EARLY EXIT
    if theory is None or fragment is None:
        print("Cart_optimizer requires theory and fragment objects provided. Exiting.")
        ashexit()
    optimizer=Cart_optimizer_class(fragment=fragment, theory=theory, rate=rate, scaling_rate_cell=scaling_rate_cell, 
                                            maxiter=maxiter, step_algo=step_algo,
                                            max_step=max_step, momentum=momentum, PBC_format_option=PBC_format_option,
                                            constrain_method=constrain_method,
                                            printlevel=printlevel, conv_criteria=conv_criteria, constraints=constraints, 
                                            frozen_atoms=frozen_atoms, result_write_to_disk=result_write_to_disk)

    result = optimizer.run()
    if printlevel >= 1:
        print_time_rel(timeA, modulename='Cart_optimizer', moduleindex=1)

    return result


class Cart_optimizer_class:

    def __init__(self,fragment=None, theory=None, rate=2.0, scaling_rate_cell=1.0, maxiter=50, step_algo="bfgs",
                                max_step=0.25, momentum=0.5, printlevel=2, conv_criteria=None, print_atoms_list=None,
                                PBC_format_option="CIF", constraints=None, constrain_method='soft',
                                frozen_atoms=None, result_write_to_disk=True):

        self.fragment = fragment
        self.theory = theory
        self.rate = rate
        self.scaling_rate_cell = scaling_rate_cell
        self.maxiter = maxiter
        self.step_algo=step_algo
        self.max_step=max_step
        self.momentum=momentum
        self.printlevel=printlevel
        self.PBC_format_option=PBC_format_option
        self.print_atoms_list=print_atoms_list
        self.result_write_to_disk=result_write_to_disk
        # Constraints
        self.constraints = constraints if constraints is not None else []
        self.constrain_method = constrain_method # 'hard' or 'soft'
        # Default force constant for soft restraints
        self.default_k = 10.0
        # Frozen atoms
        self.frozen_atoms = frozen_atoms if frozen_atoms is not None else []
        if self.frozen_atoms:
                print(f"Frozen atoms: {self.frozen_atoms}")

        self.ang2bohr=1.88972612546

        if conv_criteria is None:
            print("Convergence criteria not set by user. Using following")
            self.conv_criteria = {'convergence_grms':1e-4, 'convergence_gmax':3e-4}
        else:
            self.conv_criteria=conv_criteria
        print("Convergence criteria:", self.conv_criteria)
        print("Constraints:", self.constraints)
        for con in self.constraints:
            print("con:",con)

        # Max step in bohrs (default = 0.25 Å = 0.472 bohrs)
        self.max_step_au = max_step*self.ang2bohr

        print("Rate (atoms):", self.rate)
        print("Scaling for Rate (cell):", self.scaling_rate_cell)
        print("Maxiter:", self.maxiter)
        print(f"Max step size {self.max_step} Å")
        print()

        self.PBC=False

        #######################
        # INITITAL SETUP
        #######################

        #---- PERIODIC -----
        if getattr(self.theory, "periodic", False):
            print("Theory object is periodic")
            print("Will run periodic cell optimization")
            print(f"Initial cell vectors in Theory object: {theory.periodic_cell_vectors} Å")
            self.PBC=True

            self.cell_vectors_au = theory.periodic_cell_vectors*self.ang2bohr
            self.cell_vectors = theory.periodic_cell_vectors

        #---- NON-PERIODIC -----
        else:
            print("Theory object is not periodic.")

    def setup_PBC(self):
        # Align to standard orientation
        aligned_atom_coords, aligned_vectors = self.align_to_standard_orientation(self.fragment.coords, 
                                                                                self.theory.periodic_cell_vectors)
        print("Updating fragment coordinates and theory cell with aligned coords")
        self.fragment.coords=aligned_atom_coords
        self.theory.update_cell(aligned_vectors)

        # Reference
        self.H_ref = aligned_vectors.copy()
        self.H_ref_inv = np.linalg.inv(self.H_ref)

    def apply_cartesian_constraints(self, gradient):
        """
        Zero out gradient components for frozen atoms.
        Accepts either a list of atom indices to freeze, or a dict with
        per-atom frozen Cartesian components, e.g.:
            frozen_atoms=[0, 1, 5]                        # freeze all xyz
            frozen_atoms={0: 'xyz', 3: 'xz', 7: 'y'}     # freeze specific components
        """
        grad_out = gradient.copy()

        #if isinstance(self.frozen_atoms, (list, tuple)):
        #    for idx in self.frozen_atoms:
        #        grad_out[idx] = 0.0

        component_map = {'x': 0, 'y': 1, 'z': 2}
        for idx, components in self.all_cartesian_constraints.items():
            for c in components.lower():
                if c in component_map:
                    grad_out[idx, component_map[c]] = 0.0

        return grad_out

    def apply_bond_constraints(self, coords, gradient, energy):
        """
        Apply bond-length constraints to gradient (and energy for soft mode).

        coords:   (N, 3) physical atomic coordinates in Ångström
        gradient: (N+4, 3) supergradient (atoms + origin + 3 lattice rows)
        energy:   float, current energy

        Returns modified (energy, gradient).
        """
        if not self.constraints:
            return energy, gradient

        # Work on a copy so we don't mutate in-place unexpectedly
        grad_out = gradient.copy()
        energy_out = energy
        coords_au = coords * self.ang2bohr

        for c in self.constraints['bond']:
            print("Applying bond constraint")
            i, j, r0_ang  = c

            r0 = r0_ang * self.ang2bohr  # convert target bond length to Bohrs
            #print("i, j:", i, j)
            #print("r0:", r0)
            #k      = c.get('k', self.default_k)   # only used for soft
            k = self.default_k # temp
            # Current bond vector and length
            rij = coords_au[i] - coords_au[j]          # (3,)
            d   = np.linalg.norm(rij)
            if d < 1e-8:
                print(f"Warning: atoms {i} and {j} are on top of each other. Skipping constraint.")
                continue
            e_ij = rij / d                        # unit vector i→j

            delta = d - r0                        # signed deviation in Å

            if self.constrain_method == 'soft':
                # Harmonic restraint: V = 0.5 * k * delta^2
                # dV/dr_i = k * delta * e_ij
                # dV/dr_j = -k * delta * e_ij
                energy_out += 0.5 * k * delta**2
                grad_out[i] += k * delta * e_ij
                grad_out[j] -= k * delta * e_ij
                if self.printlevel >= 2:
                    print(f"  Soft constraint ({i},{j}): d={d:.4f} Å  target={r0:.4f} Å  "
                        f"delta={delta:.4f} Å  penalty={0.5*k*delta**2:.6f}")

            elif self.constrain_method == 'hard':
                # SHAKE-style: project out the component of the gradient
                # along the bond direction for both atoms.
                # g_parallel_i =  (g_i · e_ij) * e_ij
                # g_parallel_j = -(g_j · e_ij) * e_ij  (opposite sign convention)
                # We zero those components to enforce the constraint.
                g_i_par = np.dot(grad_out[i], e_ij) * e_ij
                g_j_par = np.dot(grad_out[j], e_ij) * e_ij
                grad_out[i] -= g_i_par
                grad_out[j] -= g_j_par
                if self.printlevel >= 2:
                    print(f"  Hard constraint ({i},{j}): d={d:.4f} Å  target={r0:.4f} Å  "
                        f"delta={delta:.4f} Å  |proj_i|={np.linalg.norm(g_i_par):.6f}")
            else:
                print(f"Unknown constraint method '{self.constrain_method}'. Use 'hard' or 'soft'.")

        return energy_out, grad_out

    def apply_angle_constraints(self, coords, gradient, energy):
        """
        Angle constraints for triplets (i, j, k).
        Target angle in degrees. Gradient via chain rule through arccos.
        """
        if not self.constraints:
            return energy, gradient

        grad_out = gradient.copy()
        energy_out = energy
        coords_au = coords * self.ang2bohr

        for c in self.constraints['angle']:
            print("Applying angle constraint")
            i, j, k, theta0_deg = c         # centre atom is j
            theta0 = np.deg2rad(theta0_deg)
            kf=self.default_k #temp
            #kf       = c.get('k', self.default_k)

            # Bond vectors pointing away from centre j
            u = coords_au[i] - coords_au[j]
            v = coords_au[k] - coords_au[j]
            lu = np.linalg.norm(u)
            lv = np.linalg.norm(v)

            if lu < 1e-8 or lv < 1e-8:
                print(f"Warning: degenerate angle {i}-{j}-{k}. Skipping.")
                continue

            u_hat = u / lu
            v_hat = v / lv
            cos_t = np.clip(np.dot(u_hat, v_hat), -1.0, 1.0)
            theta  = np.arccos(cos_t)
            sin_t  = np.sqrt(max(1.0 - cos_t**2, 1e-10))  # avoid /0 at 0° or 180°

            # dθ/dr_i = (u_hat × (u_hat × v_hat)) / (lu * sin_t)
            # but the simpler form via arccos derivative:
            # dcos/dr_i = (v_hat - cos_t * u_hat) / lu
            # dθ/dr_i  = -1/sin_t * dcos/dr_i
            dc_dri =  (v_hat - cos_t * u_hat) / lu
            dc_drk =  (u_hat - cos_t * v_hat) / lv
            dc_drj = -(dc_dri + dc_drk)

            dt_dri = -dc_dri / sin_t
            dt_drk = -dc_drk / sin_t
            dt_drj = -dc_drj / sin_t

            delta = theta - theta0   # deviation in radians

            if self.constrain_method == 'soft':
                energy_out    += 0.5 * kf * delta**2
                grad_out[i]   += kf * delta * dt_dri
                grad_out[j]   += kf * delta * dt_drj
                grad_out[k]   += kf * delta * dt_drk
                if self.printlevel >= 2:
                    print(f"  Soft angle ({i},{j},{k}): θ={np.rad2deg(theta):.3f}°  "
                        f"target={np.rad2deg(theta0):.3f}°  delta={np.rad2deg(delta):.3f}°  "
                        f"penalty={0.5*kf*delta**2:.6f}")

            elif self.constrain_method == 'hard':
                # Project out the gradient component along dθ/dr for each atom
                for idx, dt_dr in [(i, dt_dri), (j, dt_drj), (k, dt_drk)]:
                    proj = np.dot(grad_out[idx], dt_dr)
                    if np.linalg.norm(dt_dr) > 1e-10:
                        n_hat = dt_dr / np.linalg.norm(dt_dr)
                        grad_out[idx] -= np.dot(grad_out[idx], n_hat) * n_hat
                if self.printlevel >= 2:
                    print(f"  Hard angle ({i},{j},{k}): θ={np.rad2deg(theta):.3f}°  "
                        f"target={np.rad2deg(theta0):.3f}°  delta={np.rad2deg(delta):.3f}°")

        return energy_out, grad_out

    def apply_dihedral_constraints(self, coords, gradient, energy, kf=0.5):
        """
        Dihedral (torsion) restraints for quadruplets (i, j, k, l).

        Soft mode:
            E = 0.5 * kf * delta^2
        with delta wrapped into [-pi, pi].

        The gradient is computed by finite differences on the restraint energy.
        This is slower than analytic formulas but much more robust.
        """

        #Making sure user did not use torsion
        if 'dihedral' in self.constraints:
            condict = self.constraints['dihedral']
        elif 'torsion' in self.constraints:
            condict = self.constraints['torsion']
        else:
            return energy, gradient

        grad_out = gradient.copy()
        energy_out = energy
        coords_au = coords * self.ang2bohr

        def dihedral_phi(ca, i, j, k, l):
            """Signed dihedral angle in radians."""
            r1 = ca[i]
            r2 = ca[j]
            r3 = ca[k]
            r4 = ca[l]

            b1 = r2 - r1
            b2 = r3 - r2
            b3 = r4 - r3

            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)

            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            b2_norm = np.linalg.norm(b2)

            if n1_norm < 1e-12 or n2_norm < 1e-12 or b2_norm < 1e-12:
                return None

            n1_hat = n1 / n1_norm
            n2_hat = n2 / n2_norm
            b2_hat = b2 / b2_norm

            x = np.dot(n1_hat, n2_hat)
            y = np.dot(np.cross(n1_hat, b2_hat), n2_hat)
            return np.arctan2(y, x)

        def torsion_restraint_energy(ca, i, j, k, l, phi0_rad, kf_local):
            phi = dihedral_phi(ca, i, j, k, l)
            if phi is None:
                return None
            delta = np.arctan2(np.sin(phi - phi0_rad), np.cos(phi - phi0_rad))
            return 0.5 * kf_local * delta * delta

        h = 1.0e-4  # Bohr finite-difference step

        for c in condict:
            print("Applying torsion constraint")
            i, j, k, l, phi0_deg = c
            phi0 = np.deg2rad(phi0_deg)

            E0 = torsion_restraint_energy(coords_au, i, j, k, l, phi0, kf)
            if E0 is None:
                print(f"Warning: degenerate torsion {i}-{j}-{k}-{l}. Skipping.")
                continue

            energy_out += E0

            if self.constrain_method == 'soft':
                involved = [i, j, k, l]
                for idx in involved:
                    for a in range(3):
                        cp = coords_au.copy()
                        cm = coords_au.copy()
                        cp[idx, a] += h
                        cm[idx, a] -= h

                        Ep = torsion_restraint_energy(cp, i, j, k, l, phi0, kf)
                        Em = torsion_restraint_energy(cm, i, j, k, l, phi0, kf)

                        if Ep is None or Em is None:
                            continue

                        grad_out[idx, a] += (Ep - Em) / (2.0 * h)

                if self.printlevel >= 2:
                    phi = dihedral_phi(coords_au, i, j, k, l)
                    delta = np.arctan2(np.sin(phi - phi0), np.cos(phi - phi0))
                    print(f"  Soft torsion ({i},{j},{k},{l}): "
                        f"φ={np.rad2deg(phi):.3f}° target={phi0_deg:.3f}° "
                        f"delta={np.rad2deg(delta):.3f}° "
                        f"penalty={E0:.6f}")

            elif self.constrain_method == 'hard':
                # Hard torsion is not safely enforced by gradient projection.
                # Use shake_torsion after the geometry step instead.
                if self.printlevel >= 2:
                    phi = dihedral_phi(coords_au, i, j, k, l)
                    delta = np.arctan2(np.sin(phi - phi0), np.cos(phi - phi0))
                    print(f"  Hard torsion requested for ({i},{j},{k},{l}), "
                        f"but gradient projection is not reliable for dihedrals. "
                        f"Current φ={np.rad2deg(phi):.3f}° target={phi0_deg:.3f}° "
                        f"delta={np.rad2deg(delta):.3f}°")
            else:
                print(f"Unknown constraint method '{self.constrain_method}'. Use 'hard' or 'soft'.")

        return energy_out, grad_out

    def shake_torsion(self, coords, i, j, k, l, phi0_deg, max_iter=50, tol_deg=0.01):
        """
        Directly correct atomic positions to satisfy a torsion constraint.
        Moves only atoms i and l (the terminal atoms) along the torsion direction.
        Called after the geometry step, before the next gradient evaluation.
        """
        phi0 = np.deg2rad(phi0_deg)
        tol  = np.deg2rad(tol_deg)

        coords_new = coords.copy()

        for _ in range(max_iter):
            b1 = coords_new[j] - coords_new[i]
            b2 = coords_new[k] - coords_new[j]
            b3 = coords_new[l] - coords_new[k]

            n1  = np.cross(b1, b2)
            n2  = np.cross(b2, b3)
            ln1 = np.linalg.norm(n1)
            ln2 = np.linalg.norm(n2)
            lb2 = np.linalg.norm(b2)

            if ln1 < 1e-8 or ln2 < 1e-8:
                break

            m1    = np.cross(n1, b2 / lb2)
            cos_p = np.dot(n1, n2) / (ln1 * ln2)
            sin_p = np.dot(m1, n2) / (ln1 * ln2)
            phi   = np.arctan2(sin_p, cos_p)

            #phi_wrapped  = (phi  + np.pi) % (2*np.pi) - np.pi
            #phi0_wrapped = (phi0 + np.pi) % (2*np.pi) - np.pi
            #delta = phi_wrapped - phi0_wrapped
            #delta = (delta + np.pi) % (2*np.pi) - np.pi
            delta = np.arctan2(np.sin(phi - phi0), np.cos(phi - phi0))

            if abs(delta) < tol:
                break

            # Rotate atom i around the b2 axis by -delta/2
            # and atom l around the b2 axis by +delta/2
            # This distributes the correction symmetrically
            b2_hat = b2 / lb2

            def rotate(point, center, axis, angle):
                """Rotate point around axis through center by angle (radians)."""
                p = point - center
                c, s = np.cos(angle), np.sin(angle)
                return center + (p * c
                                + np.cross(axis, p) * s
                                + axis * np.dot(axis, p) * (1 - c))

            coords_new[i] = rotate(coords_new[i], coords_new[j],  b2_hat, -delta/2)
            coords_new[l] = rotate(coords_new[l], coords_new[k], -b2_hat,  delta/2)

        return coords_new

    def align_to_standard_orientation(self, fragment_coords, cell_vectors):
        """
        Rotates the entire system (atoms and cell) into the standard 
        upper-triangular orientation.

        cell_vectors: 3x3 matrix where rows are [a, b, c]
        fragment_coords: Nx3 array of atomic positions
        """
        # 1. Transpose cell_vectors because QR works on columns
        H = cell_vectors.T 

        # 2. QR Decomposition
        # H = Q * R  -> R is the upper triangular matrix we want
        Q, R = np.linalg.qr(H)

        # 3. Handle 'Flip' cases
        # QR can sometimes return negative diagonal elements. 
        # We want lengths (a_x, b_y, c_z) to be positive.
        d = np.sign(np.diag(R))
        # If a diagonal is 0, we treat it as positive
        d[d == 0] = 1

        # Correct Q and R so diagonals of R are positive
        Q = Q * d
        R = (R.T * d).T

        # 4. New Cell Vectors (R transposed back to rows)
        new_cell_vectors = R.T

        # 5. New Atomic Coordinates
        # We rotate the atoms using the same rotation matrix Q
        # Since H_new = Q.T @ H_old, we use Q.T for the atoms
        new_coords = np.dot(fragment_coords, Q)

        return new_coords, new_cell_vectors

    def compute_bfgs_step(self, current_grad, current_coords):
        # Flatten everything to 1D vectors for linear algebra
        g = current_grad.flatten()
        x = current_coords.flatten()
        n = len(g)

        # 1. INITIALIZATION
        # On the first step, we don't have a Hessian yet. 
        # We start with an Identity matrix (equivalent to Steepest Descent).
        if not hasattr(self, 'Hess_inv') or self.Hess_inv is None:
            print("BFGS: First step. SD step with rate:", self.rate)
            self.Hess_inv = np.eye(n) * self.rate 
            self.g_old = g
            self.x_old = x
            return -(self.rate * g).reshape(current_grad.shape)

        # 2. COMPUTE DIFFERENCES
        s = x - self.x_old  # Change in coordinates
        y = g - self.g_old  # Change in gradient

        # 3. UPDATE INVERSE HESSIAN (Sherman-Morrison-Woodbury formula)
        # We only update if the curvature condition (y.s > 0) is met to maintain stability
        rho_inv = np.dot(y, s)
        if rho_inv > 1e-9:
            rho = 1.0 / rho_inv
            I = np.eye(n)

            # BFGS Update Formula
            A = I - np.outer(s, y) * rho
            B = I - np.outer(y, s) * rho
            self.Hess_inv = np.dot(A, np.dot(self.Hess_inv, B)) + (rho * np.outer(s, s))
        else:
            # If curvature is bad, reset the Hessian to Identity to avoid exploding
            print("BFGS: Curvature condition not met, resetting Hessian.")
            self.Hess_inv = np.eye(n) * self.rate

        # 4. COMPUTE STEP
        # p = -Hess_inv * g
        step_vec = -np.dot(self.Hess_inv, g)

        # Update histories
        self.g_old = g
        self.x_old = x

        # Return reshaped to (N+4, 3)
        return step_vec.reshape(current_grad.shape)

    # Split  coords into atomic and lattice
    def split_coords(self,supercoords):

        R_geo = supercoords[:-4]
        origin = supercoords[-4]
        H_geo = supercoords[-3:] - origin
        s = np.dot(R_geo - origin, self.H_ref_inv)
        R_phys = np.dot(s, H_geo) + origin
        return R_phys, H_geo
    
    def calculate_reg_gradient(self,coords):
        # E + G from theory
        energy,gradient=self.theory.run(current_coords=coords, elems=self.elems_phys, 
                                    charge=self.fragment.charge, mult=self.fragment.mult, Grad=True)
        return energy, gradient
    def calculate_supergradient(self,supercoords):

        R_phys, H_geo = self.split_coords(supercoords)

        # E + G from theory
        energy,grad_phys=self.theory.run(current_coords=R_phys, elems=self.elems_phys, 
                                    charge=self.fragment.charge, mult=self.fragment.mult, Grad=True)

        # Transformation
        # M is the transformation matrix: R_phys = R_geo @ M
        # TODO: check units
        M = np.dot(self.H_ref_inv, H_geo)
        grad_Rgeo = np.dot(grad_phys, M.T)

        # Lattice gradient and masking
        # Total lattice gradient: current theory cell-gradient + convection
        #grad_latt_total = self.theory.cell_gradient
        grad_latt_total = self.theory.get_cell_gradient()
        # Standard orientation mask:
        # This zeros out: a_y, a_z, and b_z
        mask = np.array([
            [1, 0, 0], # dE/dax (ay, az frozen)
            [1, 1, 0], # dE/dbx, dE/dby (bz frozen)
            [1, 1, 1]  # dE/dcx, dE/dcy, dE/dcz (all free)
        ])
        grad_latt_masked = grad_latt_total * mask
        # scaling lattice gradient
        n_atoms = len(grad_Rgeo)
        scaling_factor = 1.0 / n_atoms
        grad_latt_preconditioned = grad_latt_masked * scaling_factor
        # 
        grad_latt_final=grad_latt_preconditioned
        # Making sure origin is zero
        grad_origin = np.zeros((1, 3))
        # Final modified gradient to pass to geomeTRIC
        mod_gradient = np.concatenate([
                grad_Rgeo,         # (N, 3)
                grad_origin,       # (1, 3)
                grad_latt_final   # (3, 3)
            ], axis=0)

        return energy, mod_gradient

    def compute_step(self,gradient,currcoords):

        if self.PBC:
            # 1. Separate rates for Atoms vs Cell (Preconditioning)
            # Often the cell needs a rate ~10x smaller than atoms in Cartesian space
            rate_mask = np.ones_like(gradient)
            rate_mask[-3:] *= self.scaling_rate_cell  # Dampen lattice steps
            effective_gradient = gradient * rate_mask
        else:
            effective_gradient = gradient
        # Calculate delta step (in Bohrs)
        if self.step_algo.lower() =="sd":
            print("Taking steepest descent step")
            delta_au = - (self.rate * effective_gradient)
        elif self.step_algo == "damped-MD":
            print("Taking damped-MD step")
            print("velocity:", self.velocity)
            self.velocity = (self.momentum * self.velocity) - (self.rate * effective_gradient)
            print("velocity:", self.velocity)
            delta_au = self.velocity
            # Simple "Power" check: If we go against the gradient, kill velocity
            if np.sum(delta_au * gradient) > 0:
                self.velocity *= 0.0
        elif self.step_algo.lower() == "nesterov":
            # Storing old
            velocity_old = self.velocity.copy()
            print("Taking Nesterov momentum step")
            self.velocity = (self.momentum * self.velocity) - (self.rate * effective_gradient)
            nesterov_update = -self.momentum * velocity_old + (1 + self.momentum) * self.velocity
            delta_au = nesterov_update
        elif self.step_algo.lower() == "bfgs":
            print("Taking BFGS step")
            delta_au = self.compute_bfgs_step(gradient, currcoords)
        elif self.step_algo.lower() == "cg":
            print("Taking conjugate gradient step")
            if self.iteration == 0:
                self.search_dir = effective_gradient
                self.prev_grad = None
            else:
                # Polak-Ribière formula for beta
                diff = effective_gradient - self.prev_grad
                beta = np.sum(gradient * diff) / np.sum(self.prev_grad * self.prev_grad)
                beta = max(0, beta) # Standard 'reset' for CG
                self.search_dir = effective_gradient + (beta * self.search_dir)

            delta_au = - (self.rate * self.search_dir)
            self.prev_grad = gradient.copy()
        else:
            print("Unknown step_algo")
            ashexit()

        return delta_au

    def run(self, theory=None, fragment=None, constraints=None, charge=None, mult=None):
        self.run_init_time=time.time()
        #print("Cart_opt: --------------------------------------------")
        #print("Cart_opt: time init:", time.time()-self.run_init_time, "seconds")
        # Update self fragment if a run fragment was provided
        if fragment is not None:
            self.fragment=fragment
            self.elems_phys=fragment.elems
        else:
            self.elems_phys=self.fragment.elems

        if self.print_atoms_list is None:
            self.print_atoms_list = self.fragment.allatoms

        # Update self theory if a run fragment was provided
        if theory is not None:
            self.theory=theory

        # Update constraints if provided
        if constraints is not None:
            self.constraints=constraints

        self.charge, self.mult = check_charge_mult(charge, mult, self.theory.theorytype, self.fragment, 
                                         "CartOptimizer", theory=self.theory, printlevel=self.printlevel)
        # Defining coordinates to use, PBC vs. non-PBC
        if self.PBC:
            print("Running periodic optimization in Cartesian coordinates with cell optimization")
            self.setup_PBC()
            currcoords = np.concatenate([
                    self.fragment.coords,         # (N, 3)
                    np.zeros((1, 3)),       # (1, 3)
                    self.theory.periodic_cell_vectors,   # (3, 3)
                ], axis=0)
            opt_type_label="PBC"
        else:
            print("Running non-periodic optimization in Cartesian coordinates")
            currcoords = self.fragment.coords
            opt_type_label="NonPBC"

        # Initialize velocity for momentum-based step algorithms
        self.velocity = np.zeros((len(currcoords),3))

        for file in ["Fragment-currentgeo.xyz", "PBC_opt_traj.xyz", "NonPBC_opt_traj.xyz"]:
            try:
                os.remove(file)
            except:
                pass

        #print("Cart_opt: time until LOOP:", time.time()-self.run_init_time, "seconds")
        # LOOP
        for iteration in range(0,self.maxiter):
            self.iteration=iteration
            print("="*40)
            print(f"{opt_type_label} optimization step", iteration)
            print("="*40)

            if self.PBC:
                currcoords_au = currcoords*self.ang2bohr
                R_phys, H_geo = self.split_coords(currcoords)
                #Update cell
                self.theory.update_cell(H_geo)
            else:
                R_phys = currcoords
                currcoords_au = R_phys*self.ang2bohr

            # Update coordinates of atoms and cell
            self.fragment.replace_coords(self.fragment.elems, R_phys, conn=False)

            # 0. PRINTING ACTIVE GEOMETRY IN EACH  ITERATION
            self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")
            self.fragment.write_xyzfile(xyzfilename=f"{opt_type_label}_opt_traj.xyz",writemode="a")
            if self.printlevel >= 1:
                print(f"Current geometry (Å) in step {iteration} (print_atoms_list region)")
                print("---------------------------------------------------")
                print_coords_for_atoms(R_phys, self.elems_phys,self.fragment.allatoms)
                print("")
                if self.PBC:
                    print(f"Current cell vectors (Å):{self.theory.periodic_cell_vectors}")
                    print(f"Current cell volume (Å):{cell_volume(H_geo)}")

            #print("Cart_opt: time until e+g step:", time.time()-self.run_init_time, "seconds")
            #########################################
            # 1. Compute energy and gradient
            #########################################
            if self.PBC:
                energy, supergradient = self.calculate_supergradient(currcoords)
            else:
                energy, supergradient = self.calculate_reg_gradient(currcoords)
                prev_supgrad = supergradient.copy()
            #print("Cart_opt: time until after e+g step:", time.time()-self.run_init_time, "seconds")
            # 1b. Apply all constraints
            self.all_cartesian_constraints={}
            if self.constraints:
                print("Applying constraints...")
                print("self.constraints:", self.constraints)
                if 'bond' in self.constraints or 'distance' in self.constraints:
                    energy, supergradient = self.apply_bond_constraints(R_phys, supergradient, energy)
                    #print("Cart_opt: time until after bondcon step:", time.time()-self.run_init_time, "seconds")
                if 'angle' in self.constraints:
                    energy, supergradient = self.apply_angle_constraints(R_phys, supergradient, energy)
                if 'torsion' in self.constraints or 'dihedral' in self.constraints:
                    energy, supergradient = self.apply_dihedral_constraints(R_phys, supergradient, energy)
                    #print("supergradient after dihedral constraints:", supergradient)
                    #print("Cart_opt: time until after dihedralcon step:", time.time()-self.run_init_time, "seconds")
                # Cartesian constraints. prepare
                if 'xyz' in self.constraints:
                    for i in self.constraints['xyz']:
                        self.all_cartesian_constraints[i] = 'xyz'
                if 'x' in self.constraints:
                    for i in self.constraints['x']:
                        self.all_cartesian_constraints[i] = 'x'
                if 'y' in self.constraints:
                    for i in self.constraints['y']:
                        self.all_cartesian_constraints[i] = 'y'
                if 'z' in self.constraints:
                    for i in self.constraints['z']:
                        self.all_cartesian_constraints[i] = 'z'
                if 'xy' in self.constraints:
                    for i in self.constraints['xy']:
                        self.all_cartesian_constraints[i] = 'xy'
                if 'yz' in self.constraints:
                    for i in self.constraints['yz']:
                        self.all_cartesian_constraints[i] = 'yz'
                if 'xz' in self.constraints:
                    for i in self.constraints['xz']:
                        self.all_cartesian_constraints[i] = 'xz'
            # 1c. Apply frozen atoms
            if self.frozen_atoms or len(self.all_cartesian_constraints)>0:
                print("We have frozen atoms or cartesian constraints, applying them to the gradient...")
                # Combining frozen atoms list with all_cartesian_constraints dict
                if isinstance(self.frozen_atoms, list):
                    for i in self.frozen_atoms:
                        self.all_cartesian_constraints[i] = 'xyz'
                print("All Cartesian constraints", self.all_cartesian_constraints)

                supergradient = self.apply_cartesian_constraints(supergradient)
                #print("Cart_opt: time until after cartesiancon step:", time.time()-self.run_init_time, "seconds")

            #########################################
            # 2. Check convergence
            #########################################
            grad_rms_atoms = np.sqrt(np.mean(supergradient**2))
            grad_max_atoms = abs(max(supergradient.min(), supergradient.max(), key=abs))

            if self.PBC:
                grad_rms_atoms = np.sqrt(np.mean(supergradient[:-4]**2))
                grad_max_atoms = abs(max(supergradient[:-4].min(), supergradient[:-4].max(), key=abs))
                grad_rms_cell = np.sqrt(np.mean(supergradient[-3:]**2))
                grad_max_cell = abs(max(supergradient[-3:].min(), supergradient[-3:].max(), key=abs))
                print(f"Step {iteration:3d} Energy: {energy:.10f} Eh     RMSG(atoms): {grad_rms_atoms:.6f} MaxG(atoms): {grad_max_atoms:.6f}    RMSG(cell): {grad_rms_cell:.6f} MaxG(cell): {grad_max_cell:.6f}     Cell-volume {cell_volume(self.theory.periodic_cell_vectors):.2f} Å")

                if grad_rms_atoms < self.conv_criteria['convergence_grms'] and grad_max_atoms < self.conv_criteria['convergence_gmax'] and \
                            grad_rms_cell < self.conv_criteria['convergence_grms'] and grad_max_cell < self.conv_criteria['convergence_gmax']:
                    print()
                    print(f"Optimization converged in {iteration+1} iterations. Convergence criteria ({self.conv_criteria}) fulfilled")
                    print(f"Final cell vectors (Å):{self.theory.periodic_cell_vectors}")
                    print(f"Final cell volume (Å):{cell_volume(self.theory.periodic_cell_vectors)}")
                    print(f"Final cell parameters: ({cell_vectors_to_params(self.theory.periodic_cell_vectors)})")
                    print()
                    print(f"Final optimized energy: {energy} Eh")

                    #Writing out fragment file and XYZ file
                    self.fragment.print_system(filename='Fragment-optimized.ygg')
                    self.fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
                    self.fragment.set_energy(energy)
                    print("\nFinal geometry")
                    self.fragment.print_coords()
                    print()
                    if self.printlevel >= 2:
                        print_internal_coordinate_table_new(self.fragment,actatoms=self.print_atoms_list)
                    print()

                    print("PBC_format_option:", self.PBC_format_option)
                    if self.PBC_format_option.upper() =="CIF":
                        convert_to_pbcfile=write_CIF_file
                        file_ext='cif'
                    elif self.PBC_format_option.upper() =="XSF":
                        convert_to_pbcfile=write_XSF_file
                        file_ext='xsf'
                    elif self.PBC_format_option.upper() == "POSCAR":
                        convert_to_pbcfile=write_POSCAR_file
                        file_ext='POSCAR'
                    convert_to_pbcfile(self.fragment.coords,self.fragment.elems,cellvectors=self.theory.periodic_cell_vectors,
                                                filename=f"Fragment-optimized.{file_ext}")
                    #Now returning final Results object
                    result = ASH_Results(label="Optimizer", energy=energy)
                    if self.result_write_to_disk is True:
                        result.write_to_disk(filename="ASH_Cart_optimizer.result")
                    return result
            else:
                print(f"Step {iteration:3d} Energy: {energy:.10f} Eh     RMSG(atoms): {grad_rms_atoms:.6f} MaxG(atoms): {grad_max_atoms:.6f}")
                if grad_rms_atoms < self.conv_criteria['convergence_grms'] and grad_max_atoms < self.conv_criteria['convergence_gmax'] :

                    if self.printlevel >= 1:
                        print()
                        print(f"Optimization converged in {iteration+1} iterations. Convergence criteria ({self.conv_criteria}) fulfilled")
                        print()
                        print(f"Final optimized energy: {energy} Eh")
                        print("Cart_opt: time CONVERGENCE:", time.time()-self.run_init_time, "seconds")

                    # Writing out fragment file and XYZ file
                    self.fragment.print_system(filename='Fragment-optimized.ygg')
                    self.fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
                    self.fragment.set_energy(energy)
                    print("\nFinal geometry")
                    self.fragment.print_coords()
                    print()
                    if self.printlevel >= 2:
                        print_internal_coordinate_table_new(self.fragment,actatoms=self.print_atoms_list)
                    print()

                    #Now returning final Results object
                    result = ASH_Results(label="Optimizer", energy=energy)
                    if self.result_write_to_disk is True:
                        result.write_to_disk(filename="ASH_Cart_optimizer.result")
                    return result

            #########################################
            # 3. Take step
            #########################################
            # Compute step
            #print("Cart_opt: time until before  compute step:", time.time()-self.run_init_time, "seconds")
            delta_au = self.compute_step(supergradient,currcoords)
            print("Computed step:", delta_au)

            if self.PBC:
                # Separate check for the lattice part (last 3 rows of delta_au)
                lattice_step = delta_au[-3:]
                if np.max(np.abs(lattice_step)) > (0.05 * self.ang2bohr): # Cap lattice at 0.05 Å
                    scale_latt = (0.05 * self.ang2bohr) / np.max(np.abs(lattice_step))
                    delta_au[-3:] *= scale_latt
                    print(f"Lattice-specific scaling applied: {scale_latt:.3f}")

            # Scale down step if required
            if np.max(np.abs(delta_au)) > self.max_step_au:
                print(f"Step scale down:  {np.max(np.abs(delta_au))}  > max_step_au: {self.max_step_au})")
                delta_au = delta_au * (self.max_step_au / np.max(np.abs(delta_au)))
            print("Actual step:", delta_au)
            

            # Take the step
            currcoords_au += delta_au
            print("Cart_opt: time until after step:", time.time()-self.run_init_time, "seconds")
            # Converting coordinates from Bohr to Angstrom
            currcoords = currcoords_au / self.ang2bohr

            #for c in self.constraints['dihedral']:
            #    i, j, k, l, phi0_deg = c
            #    currcoords_new_ang = self.shake_torsion(currcoords, i, j, k, l, phi0_deg)
            #currcoords_au = currcoords_new_ang * self.ang2bohr
            #currcoords = currcoords_new_ang

        if iteration == self.maxiter-1:
            print("Number of max iterations reached without reaching convergence. Sad...")