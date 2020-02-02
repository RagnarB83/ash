#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
# stdlib
import os
import tempfile
import shutil
import json
import bz2

# openmm
import simtk.openmm as mm
from simtk.unit import (nanometer, picosecond, dalton, Quantity,
                        kilojoules_per_mole)


#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

__all__ = ['RestartReporter', 'loadRestartFile']

class NotSpecified(object):
    def __str__(self):
        return 'NotSpecified'
NotSpecified = NotSpecified()

RESTART_FORMAT_VERSION = 2.0

# File signatures http://www.garykessler.net/library/file_sigs.html
magic_dict = {
    "\x1f\x8b\x08": "gz",
    "\x42\x5a\x68": "bz2",
    "\x50\x4b\x03\x04": "zip"
    }

max_len = max(len(x) for x in magic_dict)

def file_type(filename):
    with open(filename) as f:
        file_start = f.read(max_len)
    for magic, filetype in magic_dict.items():
        if file_start.startswith(magic):
            return filetype
    return "no match"

#-----------------------------------------------------------------------------
# Utilities
#-----------------------------------------------------------------------------

def isLeapFrogIntegrator(integrator):
    if isinstance(integrator, mm.VerletIntegrator):
        return True
    if isinstance(integrator, mm.VariableVerletIntegrator):
        return True
    if isinstance(integrator, mm.LangevinIntegrator):
        return True
    if isinstance(integrator, mm.VariableLangevinIntegrator):
        return True
    if hasattr(mm, 'DrudeLangevinIntegrator') and isinstance(integrator, mm.DrudeLangevinIntegrator):
        return True

    return False


def computeShiftedVelocities(context, state, velocities, timeShift, leaveShiftedVelocitiesInContext=False):
    """Shift velocities forward or backward in time. This method can be used
    to line up the velocities with the positions for leapfrog-style integrators.

    Parameters
    - context (Context)
    - state (State)
    - velocities (list of Vec3)
    - timeShift (float)
    - leaveShiftedVelocitiesInContext (bool)
    Returns: shifted velocities
    """
    if timeShift == 0:
        return velocities

    system = context.getSystem()
    numParticles = system.getNumParticles()
    particleMasses = [system.getParticleMass(i).value_in_unit(dalton) for i in range(numParticles)]

    # Compute the shifted velocities
    if isinstance(velocities, Quantity):
        velocities = velocities.value_in_unit(nanometer / picosecond)
    if isinstance(timeShift, Quantity):
        timeShift = timeShift.value_in_unit(picosecond)

    forces = state.getForces().value_in_unit(kilojoules_per_mole / nanometer)
    shiftedVelocities = [None for i in range(numParticles)]
    for i in range(numParticles):
        if particleMasses[i] > 0:
            shiftedVelocities[i] = velocities[i] + forces[i] * (timeShift / particleMasses[i])
        else:
            shiftedVelocities[i] = velocities[i]

    # Apply constraints to them by round-tripping them through the context
    context.setVelocities(shiftedVelocities)
    context.applyVelocityConstraints(1.0e-4)
    shiftedVelocities = context.getState(getVelocities=True).getVelocities()
    if not leaveShiftedVelocitiesInContext:
        context.setVelocities(velocities)

    return shiftedVelocities

#-----------------------------------------------------------------------------
# Classes and Function
#-----------------------------------------------------------------------------

class RestartReporter(object):
    """RestartReporter periodically writes restart files containg positions,
    velocities, box vectors, and other information necessary to restart a
    simulation.

    Because information like the state of OpenMM's internal random number
    generators is not saved, the trajectory produced by a restarted
    simulation should not be expected to be identical to one that would have
    been produced without the restart.

    To use it, create a RestartReporter, then add it to the Simulation's
    list of reporters.
    """

    def __init__(self, fileName, reportInterval, isLeapFrog=NotSpecified):
        """Create a RestartReporter.

         Parameters:
         - fileName (string) The file to write to, specified as a file name.
         - reportInterval (int) The interval (in time steps) at which to write restart files
         - isLeapFrog (bool) Flag indicating whether the simulation uses a leapfrog
           style integrator, in which the velocities are offset from the positions
           by 1/2 a timestep. If so, the velocities will be advanced to match
           up with the positions before writing the restart file. If not specified,
           the reporter will inspect the integrator and attempt to make that
           determination on its own.
        """
        self._reportInterval = reportInterval
        self._fileName = fileName
        self._isLeapFrog = isLeapFrog
        self._isInitialized = False

    def _initialize(self, simulation):
        """Delayed initialization that can only take place once we
        have access to the simulation object that the reporter is bound to
        """
        if self._isLeapFrog == NotSpecified:
            self._isLeapFrog = isLeapFrogIntegrator(simulation.context.getIntegrator())

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters:
         - simulation (Simulation) The Simulation to generate a report for
        Returns: A five element tuple.  The first element is the number of steps until the
        next report.  The remaining elements specify whether that report will require
        positions, velocities, forces, and energies respectively.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, True, True, True)

    def report(self, simulation, state):
        """Generate a restart file

        Parameters:
         - simulation (Simulation) The Simulation to generate a report for
         - state (State) The current state of the simulation
        """
        if not self._isInitialized:
            self._initialize(simulation)
            self._isInitialized = True

        positions = state.getPositions().value_in_unit(nanometer)
        boxVectors = state.getPeriodicBoxVectors().value_in_unit(nanometer)

        timeStep = 0.5 * int(self._isLeapFrog) * simulation.context.getIntegrator().getStepSize()
        velocities = state.getVelocities().value_in_unit(nanometer / picosecond)
        velocities = computeShiftedVelocities(simulation.context, state,
                        state.getVelocities(), timeStep).value_in_unit(nanometer / picosecond)

        time = state.getTime().value_in_unit(picosecond)
        step = simulation.currentStep
        parameters = state.getParameters()

        # Write the new restart file to a temporary file, then move
        # it to the proper location
        tmp_fd, tmp_fn = tempfile.mkstemp()

        try:
            f = bz2.BZ2File(tmp_fn, 'w')
            data = {'version': RESTART_FORMAT_VERSION,
                    'positions': positions,
                    'boxVectors': boxVectors,
                    'velocities': velocities,
                    'time': time,
                    'step': step,
                    'parameters': parameters}
            json.dump(data, f)
        finally:
            f.close()


        try:
            shutil.move(tmp_fn, self._fileName)
        except OSError:
            # Unix will overwrite the existing file silently if the user
            # has permission. On windows, OSError will be raised
            os.remove(self._fileName)
            shutil.move(tmp_fn, self._fileName)
        finally:
            os.close(tmp_fd)





#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def loadRestartFile(simulation, fileName, isLeapFrog=NotSpecified):
    """Populate a simulation with data from a restart file.

   Parameters:
    - simulation (Simulation) The Simulation to populate.
    - fileName (State) The file to read from, specified as a file name.
    - isLeapFrog (bool) Flag indicating whether the simulation uses a leapfrog
      style integrator, in which the velocities are offset from the positions
      by 1/2 a timestep. If so, the velocities will be advanced after loading.
      If not specified, we will inspect the integrator and attempt to make that
      determination automatically.
    """
    if file_type(fileName) == 'bz2':
        f = bz2.BZ2File(fileName)
    else:
        f = open(fileName, 'rb')
    data = json.load(f)
    f.close()

    if 'version' not in data or data['version'] != RESTART_FORMAT_VERSION:
        raise ValueError("I don't know how to read this restart file.")

    numParticles = simulation.context.getSystem().getNumParticles()
    fields = ['positions', 'boxVectors', 'velocities', 'time', 'step', 'parameters']
    for field in fields:
        if field not in data:
            raise KeyError('Restart file "%s" does not contain %s' % (fileName, field))

    # set positions
    numPositions = len(data['positions'])
    if numPositions != numParticles:
        raise ValueError('simulation contains %d particles, but restart '
                         'file only contains %d positions' % (numParticles, numPositions))
    simulation.context.setPositions(data['positions'])

    # set box vectors
    if len(data['boxVectors']) != 3:
        raise ValueError('Periodic box vectors were malformed.')
    simulation.context.setPeriodicBoxVectors(*data['boxVectors'])

    # set velocities
    if isLeapFrog == NotSpecified:
        isLeapFrog = isLeapFrogIntegrator(simulation.context.getIntegrator())
    timeShift = -0.5 * int(isLeapFrog) * simulation.context.getIntegrator().getStepSize()

    numVelocities = len(data['velocities'])
    if numVelocities != numParticles:
        raise ValueError('simulation contains %d particles, but restart '
                         'file only contains %d velocities' % (numParticles, numVelocities))
    if timeShift == 0:
        simulation.context.setVelocities(data['velocities'])
    else:
        state = simulation.context.getState(getForces=True)
        computeShiftedVelocities(simulation.context, state, data['velocities'],
                                 timeShift, leaveShiftedVelocitiesInContext=True)

    # set time
    simulation.context.setTime(data['time'])

    # set step
    simulation.currentStep = data['step']

    # set parameters
    for key, value in data['parameters'].iteritems():
        key = key.encode('ascii', 'ignore')
        simulation.context.setParameter(key, value)
