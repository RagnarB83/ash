import numpy as np
import KNARRsettings


# Author: Vilhjalmur Asgeirsson, 2019.

def IsConverged(it, maxiter, tol_rms_force, tol_max_force, max_force, rms_force):
    if it == maxiter - 1:
        return 1

    if tol_rms_force > rms_force and tol_max_force > max_force:
        return 0

    return 2


def GlobalScaleStepByMax(step, maxmove):
    step_scaled = False
    max_step = np.max(abs(step))

    if max_step > maxmove:
        step = maxmove * np.divide(step, max_step)
        step_scaled = True

        if KNARRsettings.printlevel > 0:
            print('**Warning: large step (%4.2f Angs) attempted. Step scaled.' % max_step)

    return step, step_scaled


def LocalScaleStepByMax(ndim, nim, step, maxmove):
    step_scaled = False
    for i in range(nim):
        max_step = np.max(abs(step[i * ndim:(i + 1) * ndim]))
        if max_step > maxmove:
            step[i * ndim:(i + 1) * ndim] = maxmove * np.divide(step[i * ndim:(i + 1) * ndim], max_step)
            step_scaled = True

    if KNARRsettings.printlevel > 0 and step_scaled:
        print('**Warning: large step (%4.2f Angs) attempted. Step scaled.' % max_step)

    return step, step_scaled


def ScaleStepByNorm(step, maxmove):
    step_scaled = False
    max_step = np.linalg.norm(step)

    if max_step > maxmove:
        step = maxmove * np.divide(step, max_step)
        step_scaled = True

        if KNARRsettings.printlevel > 0:
            print('**Warning: large step (%4.2f Angs) attempted. Step scaled.' % max_step)

    return step, step_scaled


def TakeFDStep(calculator, atoms, fd_step):

    # keep config / forces
    current_forces = atoms.GetF().copy()
    current_config = atoms.GetR().copy()
    current_energy = atoms.GetEnergy()

    # Get direction of force and generate step in that direction
    Fu = current_forces / np.linalg.norm(current_forces)
    step = fd_step * Fu
    # Take step - but we do not save it.
    new_config = current_config + step
    atoms.SetR(new_config)
    atoms.UpdateCoords()

    # Compute forces and energy at new step
    calculator.Compute(atoms)
    atoms.UpdateF()

    # Restore previous values and store new forces
    new_forces = atoms.GetF()
    atoms.SetR(current_config)
    atoms.SetF(current_forces)
    atoms.SetEnergy(current_energy)

    H = 1.0 / (np.dot((-new_forces + current_forces).T, Fu) / fd_step)
    # If curvature is position - get new step
    if H > 0.0:
        step = np.multiply(H, current_forces)

    return step

def TakeFDStepWithFunction(calculator, atoms, fd_step,
                           current_fneb, func):


    current_config = atoms.GetR().copy()
    current_forces = atoms.GetF().copy()

    # Get direction of force and generate step in that direction
    Fu = current_fneb / np.linalg.norm(current_fneb)
    step = fd_step * Fu

    # Take step - but we do not save it.
    new_config = atoms.GetR() + step
    atoms.SetR(new_config)
    atoms.UpdateCoords()

    # Compute forces and energy at new step
    calculator.Compute(atoms)
    atoms.UpdateF()
    new_forces=func(atoms.GetF())[0]
    # restore original

    H = 1.0 / (np.dot((-new_forces + current_fneb).T, Fu) / fd_step)
    # If curvature is position - get new step
    if H > 0.0:
        step = np.multiply(H, current_fneb)

    atoms.SetF(current_forces)
    atoms.SetR(current_config)
    return step
