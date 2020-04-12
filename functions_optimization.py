import numpy as np
import constants
from functions_general import *
from functions_coords import *
from yggdrasill import *
import time
import os
import shutil
import yggdrasill
#Root mean square of numpy array, e.g. gradient
def RMS_G(grad):
    sumsq = 0;
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
#GEOMETRY OPTIMIZERS    #
#########################
#TODO: Implement maxmove scaling step thing
#TODO: Implement xtB Hessian option into LBFGS as starting Hessian??
#TODO: Fix Newton-Rhapson. Write so that we can easily take NR-step using whatever Hessian


#TODO: More complex optimizer options:
# https://github.com/eljost/pysisyphus/tree/master/pysisyphus/optimizers
# Has QN, RFO, LBFGS, etc, linesearches etc.
# Add Pele: https://pele-python.github.io/pele/quenching.html
# Add stuff from ASE: https://wiki.fysik.dtu.dk/ase/ase/optimize.html  Maybe GP minimizer??
# https://wiki.fysik.dtu.dk/ase/_modules/ase/optimize/gpmin/gpmin.html#GPMin
# Interface DL-FIND (internal coords, HDLC etc.): https://www.chemshell.org/dl-find



#Yggdrasill Optimizer class for basic usage
class Optimizer:
    def __init__(self, fragment=None, theory='', optimizer='', maxiter=50, frozen_atoms=[], RMSGtolerance=0.0001, MaxGtolerance=0.0003):
        if fragment is not None:
            self.fragment=fragment
        else:
            print("No fragment provided to optimizer")
            print("Checking if theory object contains defined fragment")
            try:
                self.fragment=theory.fragment
                print("Found theory fragment")
            except:
                print("Found no fragment in theory object either.")
                print("Exiting...")
                exit()
        self.theory=theory
        self.optimizer=optimizer
        self.maxiter=maxiter
        self.RMSGtolerance=RMSGtolerance
        self.MaxGtolerance=MaxGtolerance
        #Frozen atoms: List of atoms that should be frozen
        self.frozen_atoms=frozen_atoms
        #List of active vs. frozen labels
        self.actfrozen_labels=[]
        for i in range(self.fragment.numatoms):
            print("i", i)
            if i in self.frozen_atoms:
                self.actfrozen_labels.append('Frozen')
            else:
                self.actfrozen_labels.append('Active')

    def run(self):
        beginTime = time.time()
        print(BC.OKMAGENTA, BC.BOLD, "------------STARTING OPTIMIZER-------------", BC.END)
        print_option='Big'
        print("Running Optimizer")
        print("Optimization algorithm:", self.optimizer)
        if len(self.frozen_atoms)> 0:
            print("Frozen atoms:", self.frozen_atoms)
        blankline()
        #Printing info and basic initalization of parameters
        if self.optimizer=="SD":
            print("Using very basic stupid Steepest Descent algorithm")
            sdscaling=0.85
            print("SD Scaling parameter:", sdscaling)
        elif self.optimizer=="KNARR-LBFGS":
            print("Using LBFGS optimizer from Knarr by Vilhjálmur Ásgeirsson")
            print("LBFGS parameters (currently hardcoded)")
            print(LBFGS_parameters)
            reset_opt = False
        elif self.optimizer=="SD2":
            sdscaling = 0.01
            print("Using different SD optimizer")
            print("SD Scaling parameter:", sdscaling)
        elif self.optimizer=="KNARR-FIRE":
            time_step=0.01
            was_scaled=False
            print("FIRE Parameters for timestep:", timestep)
            print(GetFIREParam(time_step))

        print("Tolerances:  RMSG: {}  MaxG: {}  Eh/Bohr".format(self.RMSGtolerance, self.MaxGtolerance))
        #Name of trajectory file
        trajname="opt-trajectory.xyz"
        print("Writing XYZ trajectory file: ", trajname)
        #Remove old trajectory file if present
        try:
            os.remove(trajname)
        except:
            pass
        #Current coordinates
        current_coords=self.fragment.coords
        elems=self.fragment.elems

        #OPTIMIZATION LOOP
        #TODO: think about whether we should switch to fragment object for geometry handling
        for step in range(1,self.maxiter):
            CheckpointTime = time.time()
            blankline()
            print("GEOMETRY OPTIMIZATION STEP", step)
            print("Current geometry (Å):")
            if self.theory.__class__.__name__ == "QMMMTheory":
                print_coords_all(current_coords,elems, indices=self.fragment.allatoms, labels=self.theory.hybridatomlabels, labels2=self.actfrozen_labels)
            else:
                print_coords_all(current_coords, elems, indices=self.fragment.allatoms, labels=self.actfrozen_labels)
            blankline()

            #Running E+G theory job.
            E, Grad = self.theory.run(current_coords=current_coords, elems=self.fragment.elems, Grad=True)
            print("E,", E)
            print("Grad,", Grad)

            #Applying frozen atoms constraint. Setting gradient to zero on atoms
            if len(self.frozen_atoms) > 0:
                print("Applying frozen-atom constraints")
                #Setting gradient for atoms to zero
                for num,gradcomp in enumerate(Grad):
                    if num in self.frozen_atoms:
                        Grad[num]=[0.0,0.0,0.0]
            print("Grad (after frozen constraints)", Grad)
            #Converting to atomic forces in eV/Angstrom. Used by Knarr
            forces_evAng=Grad * (-1) * constants.hartoeV / constants.bohr2ang
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
            if RMSG < self.RMSGtolerance and MaxG < self.MaxGtolerance:
                print("RMSG: {:3.9f}       Tolerance: {:3.9f}    YES".format(RMSG, self.RMSGtolerance))
                print("MaxG: {:3.9f}       Tolerance: {:3.9f}    YES".format(MaxG, self.MaxGtolerance))
                print(BC.OKGREEN,"Geometry optimization Converged!",BC.END)
                write_xyz_trajectory(trajname, current_coords, elems, E)

                # Updating energy and coordinates of Yggdrasill fragment before ending
                self.fragment.set_energy(E)
                print("Final optimized energy:", self.fragment.energy)
                self.fragment.replace_coords(elems, current_coords)
                blankline()
                print_time_rel_and_tot(CheckpointTime, beginTime, 'Opt Step')
                return
            elif RMSG > self.RMSGtolerance and MaxG < self.MaxGtolerance:
                print("RMSG: {:3.9f}       Tolerance: {:3.9f}    NO".format(RMSG, self.RMSGtolerance))
                print("MaxG: {:3.9f}       Tolerance: {:3.9f}    YES".format(MaxG, self.MaxGtolerance))
                print(BC.WARNING,"Not converged",BC.END)
            elif RMSG < self.RMSGtolerance and MaxG > self.MaxGtolerance:
                print("RMSG: {:3.9f}       Tolerance: {:3.9f}    YES".format(RMSG, self.RMSGtolerance))
                print("MaxG: {:3.9f}       Tolerance: {:3.9f}    NO".format(MaxG, self.MaxGtolerance))
                print(BC.WARNING,"Not converged",BC.END)
            else:
                print("RMSG: {:3.9f}       Tolerance: {:3.9f}    NO".format(RMSG, self.RMSGtolerance))
                print("MaxG: {:3.9f}       Tolerance: {:3.9f}    NO".format(MaxG, self.MaxGtolerance))
                print(BC.WARNING,"Not converged",BC.END)

            blankline()
            if self.optimizer=='SD':
                print("Using Basic Steepest Descent optimizer")
                print("Scaling parameter:", sdscaling)
                current_coords=steepest_descent(current_coords,Grad,sdscaling)
            elif self.optimizer=='SD2':
                print("Using Basic Steepest Descent optimizer, SD2 with norm")
                print("Scaling parameter:", sdscaling)
                current_coords=steepest_descent2(current_coords,Grad,sdscaling)
            elif self.optimizer=="KNARR-FIRE":
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

            elif self.optimizer=='NR':
                print("disabled")
                exit()
                #Identity matrix
                Hess_approx=np.identity(3*len(current_coords))
                #TODO: Not active
                current_coords = newton_raphson(current_coords, Grad, Hess_approx)
            elif self.optimizer=='KNARR-LBFGS':
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
                    step = TakeFDStep(self.theory, current_coords, LBFGS_parameters["fd_step"], forces_evAng, self.fragment.elems)

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
                exit()
            #Take the actual step
            #Todo: Implement maxmove-scaling here if step too large
            current_coords=current_coords+step
            #Write current geometry (after step) to disk as 'Current_geometry.xyz'.
            # Can be used if optimization failed, SCF convergence problemt etc.
            write_xyzfile(elems, current_coords, 'Current_geometry')
            blankline()
            print_time_rel_and_tot(CheckpointTime, beginTime, 'Opt Step')
        print(BC.FAIL,"Optimization did not converge in {} iteration".format(self.maxiter),BC.END)





#Very basic bad steepest descent algorithm.
#Arbitrary scaling parameter instead of linesearch
#0.8-0.9 seems to work well for H2O
def steepest_descent(coords, Gradient,scaling):
    newcoords = coords - scaling*Gradient
    return newcoords

#Normalized forces
def steepest_descent2(coords, Gradient, scaling):
    current_forces=Gradient * (-1) * constants.hartoeV / constants.bohr2ang
    Fu = current_forces / np.linalg.norm(current_forces)
    step = scaling * Fu
    new_config = coords + step
    return new_config

#Optimizer SD, LBFGS and FIRE routines from Villi

LBFGS_parameters = {'fd_step' : 0.001, 'lbfgs_memory' : 20, 'lbfgs_damping' : 1.0}
SD_parameters = {'sd_step' : 0.001}

# Author: Vilhjalmur Asgeirsson, 2019
#Modified to fit Yggdrasill
def TakeFDStep(theory, current_coords, fd_step, forces, elems):
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
    E, Grad = theory.run(current_coords=new_config, elems=elems, Grad=True)

    # Restore previous values and store new forces
    new_forces = Grad*(-1)*constants.hartoeV/constants.bohr2ang
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





#NOT WORKING
def newton_raphson(coords, Gradient,Hessian):
    Hessian_inv=np.linalg.inv(Hessian)
    #Convert to 1D array
    Grad=Gradient.reshape(1,Gradient.size)
    #print(Hessian_inv)
    #print(Grad)
    print("Hessian_inv", Hessian_inv)
    print("")
    print("Gradient:", Gradient)
    print("")
    print("Grad:", Grad)
    bla = Hessian_inv*Grad
    print(bla)
    exit()
    #newcoords = coords - Hessian_inv*Grad
    return newcoords


#########################
# PYBERNY Optimization.
# Has internal coordinates
#PyBerny: https://github.com/jhrmnn/pyberny/blob/master/README.md
#Installed via pip
#Limitations: No constraints or frozen atoms
########################

def BernyOpt(theory,fragment):
    blankline()
    print("Beginning Py-Berny Optimization")
    try:
        from berny import Berny, geomlib
    except:
        blankline()
        print(BC.FAIL,"pyberny module not found!", BC.END)
        print(BC.WARNING,"Either install pyberny using pip:\n pip install pyberny\n "
                         "or manually from Github (https://github.com/jhrmnn/pyberny)", BC.END)
        exit()
    print("See: https://github.com/jhrmnn/pyberny")
    elems=fragment.elems
    coords=fragment.coords
    #Options: Berny(ethanol, steprms=0.01, stepmax=0.05, maxsteps=5)
    optimizer = Berny(geomlib.Geometry(fragment.elems,fragment.coords))
    for geom in optimizer:
        # get energy and gradients for geom
        E, Grad = theory.run(current_coords=geom.coords, elems=elems, Grad=True)
        optimizer.send((E,Grad))
    print("BernyOpt Geometry optimization converged!")
    #Updating energy and coordinates of Yggdrasill fragment before ending
    fragment.set_energy(E)
    print("Final optimized energy:",  fragment.energy)
    fragment.replace_coords(elems,geom.coords)
    blankline()

#########################
# geomeTRIC Optimization
########################

#Wrapper function around geomTRIC code. Take theory and fragment info from Yggdrasill
#Supports frozen atoms and bond constraints in native code. Use frozenatoms and bondconstraints for this.
#TODO: Get other constraints (angle-constraints etc.) working.
#New feature: Active Region for huge systems. Use ActiveRegion=True and provide actatoms list.
#Active-atom coords (e.g. only QM region) are only provided to geomeTRIC during optimization while rest is frozen.
#E+G still performed on full system
#Needed as discussed here: https://github.com/leeping/geomeTRIC/commit/584869707aca1dbeabab6fe873fdf139d384ca66#diff-2af7dd72b77dac63cea64c052a549fe0
# Todo: Add optional print-coords in each step option. Maybe only print QM-coords (if QM/MM).
def geomeTRICOptimizer(theory=None,fragment=None, coordsystem='tric', frozenatoms=[], bondconstraints=[],
                       maxiter=50, ActiveRegion=False, actatoms=[]):
    try:
        os.remove('geometric_OPTtraj.log')
        os.remove('geometric_OPTtraj.xyz')
        os.remove('constraints.txt')
        os.remove('initialxyzfiletric.xyz')
        shutil.rmtree('geometric_OPTtraj.tmp')
        shutil.rmtree('dummyprefix.tmp')
        os.remove('dummyprefix.log')
    except:
        pass
    blankline()
    print("Launching geomeTRIC optimization module")
    print("Coordinate system: ", coordsystem)
    print("Max iterations: ", maxiter)
    print("Frozen atoms: ", frozenatoms)
    print("Bond constraints: ", bondconstraints)
    if fragment==None:
        print("geomeTRIC requires fragment object")
        exit()
    try:
        import geometric
    except:
        blankline()
        print(BC.FAIL,"geomeTRIC module not found!", BC.END)
        print(BC.WARNING,"Either install geomeTRIC using pip:\n pip install geometric\n or manually from Github (https://github.com/leeping/geomeTRIC)", BC.END)
        exit()

    #ActiveRegion option where geomeTRIC only sees the QM part that is being optimized
    if ActiveRegion == True:
        print("Active Region option Active!")
        print("Note: Passing only active-region coordinates to geomeTRIC.")
        print("Number of active atoms", len(actatoms))
        actcoords, actelems = fragment.get_coords_for_atoms(actatoms)
        #Writing act-region coords (only) of Yggdrasill fragment to disk as XYZ file and reading into geomeTRIC
        write_xyzfile(actelems, actcoords, 'initialxyzfiletric')
        mol_geometric_frag=geometric.molecule.Molecule("initialxyzfiletric.xyz")
    else:
        #Write coordinates from Yggdrasill fragment to disk as XYZ-file and reading into geomeTRIC
        fragment.write_xyzfile("initialxyzfiletric.xyz")
        mol_geometric_frag=geometric.molecule.Molecule("initialxyzfiletric.xyz")

    #Defining Yggdrasill engine class used to communicate with geomeTRIC
    class Yggdrasillengineclass:
        def __init__(self,geometric_molf, theory, ActiveRegion=False):
            #Defining M attribute of engine object as geomeTRIC Molecule object
            self.M=geometric_molf
            #Defining theory from argument
            self.theory=theory
            self.ActiveRegion=ActiveRegion
            #Defining current_coords for full system (not only act region)
            self.full_current_coords=[]
            #Manual iteration count
            self.iteration_count=0
        #Defining calculator
        def clearCalcs(self):
            print("ClearCalcs option chosen by geomeTRIC. Not sure why")
        def calc(self,coords,tmp):
            self.iteration_count+=1
            #Updating coords in object
            #Need to combine with rest of full-syme coords I think
            self.M.xyzs[0] = coords.reshape(-1, 3) * constants.bohr2ang
            currcoords=self.M.xyzs[0]
            #Special act-region (for QM/MM) since GeomeTRIC does not handle huge system and constraints
            if self.ActiveRegion==True:
                #Defining full_coords as original coords temporarily
                full_coords = np.array(fragment.coords)
                #Replacing act-region coordinates with coords from currcoords
                for i, c in enumerate(full_coords):
                    if i in actatoms:
                        #Silly. Pop-ing first coord from currcoords until done
                        curr_c, currcoords = currcoords[0], currcoords[1:]
                        full_coords[i] = curr_c
                self.full_current_coords=full_coords
                #Request Engrad calc for full system
                E, Grad = self.theory.run(current_coords=full_coords, elems=fragment.elems, Grad=True)
                #Trim Full gradient down to only act-atoms gradient
                Grad_act = np.array([Grad[i] for i in actatoms])
                self.energy = E
                return {'energy': E, 'gradient': Grad_act.flatten()}
            else:
                E,Grad=self.theory.run(current_coords=currcoords, elems=self.M.elem, Grad=True)
                self.energy = E
                return {'energy': E, 'gradient': Grad.flatten()}


    class geomeTRICArgsObject:
        def __init__(self,eng,constraints, coordsys, maxiter):
            self.coordsys=coordsys
            self.maxiter=maxiter
            self.prefix='geometric_OPTtraj'
            self.input='dummyinputname'
            self.constraints=constraints
            #Created log.ini file here. Missing from pip installation for some reason?
            #Storing log.ini in yggdrasill dir
            path = os.path.dirname(yggdrasill.__file__)
            self.logIni=path+'/log.ini'
            self.customengine=eng

    #Define constraints provided. Write constraints.txt file
    #Frozen atom option. Only for small systems. Not QM/MM etc.
    if len(frozenatoms) > 0 :
        print("Writing frozen atom constraints")
        constraints='constraints.txt'
        with open("constraints.txt", 'w') as confile:
            confile.write('$freeze\n')
            for frozat in frozenatoms:
                #Changing from zero-indexing (Yggdrasill) to 1-indexing (geomeTRIC)
                frozenatomindex=frozat+1
                confile.write('xyz {}\n'.format(frozenatomindex))
    #Bond constraints
    elif len(bondconstraints) > 0 :
        constraints='constraints.txt'
        with open("constraints.txt", 'w') as confile:
            confile.write('$freeze\n')
            for bondpair in bondconstraints:
                #Changing from zero-indexing (Yggdrasill) to 1-indexing (geomeTRIC)
                print("bondpair", bondpair)
                confile.write('distance {} {}\n'.format(bondpair[0]+1,bondpair[1]+1))
    else:
        constraints=None

    #Defining Yggdrasill engine object containing geometry and theory. ActiveRegion boolean passed.
    yggdrasillengine = Yggdrasillengineclass(mol_geometric_frag,theory, ActiveRegion=ActiveRegion)
    #Defining args object, containing engine object
    args=geomeTRICArgsObject(yggdrasillengine,constraints,coordsys=coordsystem, maxiter=maxiter)

    #Starting geomeTRIC
    geometric.optimize.run_optimizer(**vars(args))
    time.sleep(1)
    blankline()
    #print("geomeTRIC Geometry optimization converged in {} steps!".format(geometric.iteration))
    print("geomeTRIC Geometry optimization converged in {} steps!".format(yggdrasillengine.iteration_count))
    #TODO Add iterations here.

    blankline()

    #Updating energy and coordinates of Yggdrasill fragment before ending
    fragment.set_energy(yggdrasillengine.energy)
    print("Final optimized energy:",  fragment.energy)
    #
    fragment.replace_coords(fragment.elems,yggdrasillengine.full_current_coords, conn=False)
    fragment.print_system(filename='Fragment-optimized.ygg')
    fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')

    return fragment
    # TODO:
    # Trajectory file. Both active atoms and full?