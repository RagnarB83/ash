from ash import *

# Script to perform a metadynamics simulation using the Plumed library instead of the OpenMM built-in metadynamics.
# This offers more flexiblity in collective variables etc.
# NOTE : Plumed uses 1-based atom indexing (instead of 0-based atom indexing in ASH and OpenMM)
# NOTE : Plumed uses different energy units also


#Creating the ASH fragment
frag = Fragment(databasefile="butane.xyz", charge=0, mult=1)
#Defining the xTB theory (GFN1-xTB)
theory = xTBTheory(runmode='library', xtbmethod="GFN1")

plumedstring="""
# set up two variables for Phi and Psi dihedral angles
phi: TORSION ATOMS=1,2,3,4

# Activate metadynamics in phi
# depositing a Gaussian every 100 time steps,
# with height equal to 1.0 kJ/mol,
# and width 0.5 rad
#
metad: METAD ARG=phi PACE=100 HEIGHT=1.0 SIGMA=0.5 FILE=HILLS
# monitor the two variables and the metadynamics bias potential
PRINT STRIDE=10 ARG=phi,metad.bias FILE=COLVAR
"""

#Call the OpenMM_MD_plumed function
OpenMM_MD_plumed(fragment=frag, theory=theory, timestep=0.001,
              simulation_time=10,
              temperature=300, integrator='LangevinMiddleIntegrator',
              coupling_frequency=1, traj_frequency=1,
              plumed_input_string=plumedstring)