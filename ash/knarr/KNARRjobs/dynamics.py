import time
import numpy as np
import os
import KNARRsettings
from KNARRio.system_print import PrintJob, PrintCallBack, PrintJobDone
from KNARRio.output_print import PrintAtomMatrix
from KNARRio.io import WriteXYZF
from KNARRio.io import WriteSingleImageTraj, ReadModeFromFile
from KNARRjobs.utilities import GetMaxwellBoltzmannVelocity, GetTemperature, \
    GetKineticEnergy, AndersenCollision, VelocityVerletStep, Andersen, LangevinStep


# Author: Vilhjalmur Asgeirsson, 2019

def DoDynamics(atoms, calculator, parameters):
    PrintJob('Molecular Dynamics')
    # PrintCallBack('optjob', calculator, atoms, optimizer)

    if not atoms.setup:
        raise RuntimeError("Atoms object is not properly initialized")
    if not calculator.setup:
        raise RuntimeError("Calculator is not properly initialized")

    atoms.PrintConfiguration('Input configuration:')
    start_t = time.time()
    basename = atoms.GetOutputFile()
    ensemble = 'microcanonical'
    # ========================================
    # Read parameters
    # =========================================
    timestep = parameters["TIME_STEP"]  # SIZE OF EACH STEP
    simulation_time = parameters["SIMULATION_TIME"]
    velo_distr_string = parameters["VELOCITY_DISTR"]

    if velo_distr_string.lower() == "maxwell_boltzmann":
        velo_distr = 1
    elif velo_distr_string.lower() == "zero":
        velo_distr = 0
    else:
        velo_fname = velo_distr_string
        velo_distr = 2

    thermostat = -1
    try:
        print_iter = int(parameters["PRINT_ITER"])
    except:
        raise IOError("PrintIter should be integer")
    try:
        temp_input = float(parameters["TEMPERATURE"])
    except:
        raise IOError("temperature needs to be float")

    force_temp = parameters["FORCE_TEMPERATURE"]
    thermostat_type = parameters["THERMOSTAT"]
    gamma = parameters["LANGEVIN_FRICTION"]
    collfreq = parameters["ANDERSEN_COLLISION_FREQ"]
    collstrength = parameters["ANDERSEN_COLLISION_STRENGTH"]
    no_kb = parameters["NO_KB"]
    if no_kb:
        KNARRsettings.kB = 1.0

    conf_sampling = parameters["CONFORMATIONAL_SAMPLING"]
    try:
        conf_interval = int(parameters["SAMPLE_INTERVAL"])
    except:
        raise IOError("sample interval needs to be an integer")

    if conf_sampling:
        # make checks
        if (simulation_time / timestep) % conf_interval != 0:
            raise RuntimeError("Make sure the simulation time is divisible by the sample interval")
    conformerNo = 0

    exitIfJump = parameters["EXIT_IF_JUMP"]
    criticalTime = parameters["EXIT_CRITICAL_TIME"]
    TSTcounter = 0
    thermalCounter = 0
    jumpTime = 0.0
    possibleJump = False
    keepT = []

    if thermostat_type is not None:
        if thermostat_type.lower().strip() == 'none':
            thermostat = -1
            ensemble = 'microcanonical'
        elif thermostat_type.lower().strip() == 'andersen':
            thermostat = 0
            ensemble = 'canonical (Andersen)'
        elif thermostat_type.lower().strip() == 'langevin':
            thermostat = 1
            ensemble = 'canonical (Langevin)'
        else:
            raise TypeError("Unknown option for thermostat")

    # ===========================================================================
    # Init dynamics
    # ===========================================================================
    f = open('dynamics.dat', 'w')
    simulated_time = 0.0
    mass = atoms.GetM()

    symbols = [atoms.GetSymbols()[i] for i in atoms.GetMoveableAtomsInList()]
    atoms.ZeroV()
    atoms.ZeroA()

    # If Maxwell-Boltzmann distribution of velocities... do that here
    if velo_distr == 1:
        atoms.SetV(GetMaxwellBoltzmannVelocity(atoms.GetNDof(), mass, temp_input, atoms.IsTwoDee()))
    elif velo_distr == 2:
        if os.path.isfile(velo_fname):
            velo = ReadModeFromFile(velo_fname)
            atoms.SetV(velo)
        else:
            raise IOError("Could not find %s" % velo_fname)

    PrintAtomMatrix('\nInitial velocity', atoms.GetNDof(),
                    atoms.GetV(), symbols)
    Ekin = GetKineticEnergy(atoms.GetNDof(), mass, atoms.GetV())
    temp = GetTemperature(atoms.GetNDof(), Ekin, istwodee=atoms.IsTwoDee)

    # scaling of temperature
    if force_temp and velo_distr == 1:
        atoms.SetV(np.multiply(atoms.GetV(), np.sqrt(temp_input / temp)))

    Ekin = GetKineticEnergy(atoms.GetNDof(), mass, atoms.GetV())
    temp = GetTemperature(atoms.GetNDof(), Ekin, istwodee=atoms.IsTwoDee())
    print('Initial kinetic energy : %6.4lf eV' % Ekin)
    print('Initial temperature    : %6.2lf K' % temp)

    # ===========================================================================
    # start dynamics
    # ===========================================================================

    print('\nStarting simulation in %s ensemble:' % ensemble)
    print('%4s %4ls   % 8ls     % 8ls     % 8ls      %6ls' % ('step', 'time', 'Epot', 'Ekin', 'Etot', 'Temp.'))
    step = 0
    if os.path.isfile('knarr_traj.xyz'):
        os.remove('knarr_traj.xyz')

    while simulated_time < simulation_time:
        KNARRsettings.boost_temp = temp

        # Save current step
        currPos = atoms.GetCoords().copy()

        # If conformer sampling is used - print out structure
        if conf_sampling:
            if step % conf_interval == 0 and step > 0:
                fname_str = 'conformer_%i.xyz' % conformerNo
                print('Writing conformer %i to file %s' % (conformerNo, fname_str))
                WriteXYZF(fname_str, atoms.GetNDim(), atoms.GetCoords(), atoms.GetSymbols(), energy=atoms.GetEnergy(),
                          fxyz=atoms.GetV())
                conformerNo += 1

        # Take a step
        if thermostat == -1:  # no thermostat (NVE)
            VelocityVerletStep(calculator, atoms, dt=0.01)
        elif thermostat == 0:
            Andersen(atoms, timestep, temp_input, collfreq=collfreq, collstrength=collstrength)
            VelocityVerletStep(calculator, atoms, dt=0.01)
        elif thermostat == 1:
            LangevinStep(atoms, calculator, dt=timestep, gamma=gamma, temperature=temp_input)

        # Compute the time step
        if KNARRsettings.boosted:
            dt = timestep * KNARRsettings.boost_time
        else:
            dt = timestep

        newPos = atoms.GetCoords()

        if conf_sampling:
            for i in range(0, atoms.GetNDim(), 3):
                a = np.floor(currPos[i + 0])
                b = np.floor(newPos[i + 0])
                if a != b:
                    print 'Found a TST jump in %i steps and after %12.8f time units' % (step, simulated_time)
                    print 'Need to stop execution!!!'
                    PrintJobDone('Dynamics job', time.time() - start_t)
                    return atoms
                
                    
        # if exit on jump is activated then
        if exitIfJump:
            for i in range(0, atoms.GetNDim(), 3):
                a = np.floor(currPos[i + 0])
                b = np.floor(newPos[i + 0])
                if a != b:
                    print 'Found a TST jump in %i steps and after %12.8f time units' % (step, simulated_time)
                    TSTcounter += 1
                    possibleJump = True
                    jumpTime = 0.0
                    break

            if possibleJump:
                if jumpTime > criticalTime:
                    print 'Checking if thermal jump occurred:',
                    for i in range(0, atoms.GetNDim(), 3):
                        if newPos[i+0] < 0.0 or newPos[i+0] > 1.0:
                            jumpOccurred = True
                            break
                        else:
                            jumpOccurred = False

                    # Check if jumpOccure
                    if jumpOccurred:
                        # Terminate execution.
                        print 'Yes! (Found a thermal jump) Stopping execution.'
                        thermalCounter += 1
                        with open("reynir.out","w") as f:
                            f.write('%i %i %i %12.8f %12.8f\n' % (step, TSTcounter, thermalCounter, simulated_time, np.mean(keepT)))
                        PrintJobDone('Dynamics job', time.time() - start_t)
                        return atoms
                    else:
                        print 'No'
                        possibleJump = False

                else:
                    jumpTime += dt


        # OUTPUT
        WriteSingleImageTraj(basename + '_traj.xyz', atoms.GetNDim(), atoms.GetCoords(), atoms.GetSymbols(),
                             atoms.GetEnergy())
        Ekin = GetKineticEnergy(atoms.GetNDof(), mass, atoms.GetV())
        temp = GetTemperature(atoms.GetNDof(), Ekin, istwodee=atoms.IsTwoDee())
        keepT.append(temp)
        if (step % 20) == 0:
            print '%i %6.2lf     % 6.4lf     % 6.4lf     % 6.4lf    %6.4lf' % (step,
                                                                               simulated_time, atoms.GetEnergy(), Ekin,
                                                                               atoms.GetEnergy() + Ekin, temp)
        f.write('%6.2lf     % 6.4lf     % 6.4lf     % 6.4lf    %6.4lf \n' % (
            simulated_time, atoms.GetEnergy(), Ekin, atoms.GetEnergy() + Ekin, temp))


        simulated_time += dt
        step += 1

    f.close()
    with open("reynir.out", "w") as f:
        f.write('%i %i %i %12.8f %12.8f\n' % (step, TSTcounter, thermalCounter, simulated_time, np.mean(keepT)))
    PrintJobDone('Dynamics job', time.time() - start_t)
    return atoms
