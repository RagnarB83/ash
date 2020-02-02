#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from simtk.openmm import CustomIntegrator

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def VelocityVerletIntegrator(timestep):
    # Velocity Verlet integrator with explicit velocities.
    integrator = CustomIntegrator(timestep)
    integrator.addPerDofVariable("x1", 0)
    integrator.addPerDofVariable("x2", 0)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("x2", "x")
    integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x2-x1)/dt")
    integrator.addConstrainVelocities()
    return integrator
