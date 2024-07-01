from ash import *
import os
import math

# Example script for performing a basic umbrella sampling simulation using ASH
# System: Butane torsion using GFN1-XTB

numcores = 1
####################################################################
# Creating the ASH fragment
frag = Fragment(databasefile="butane.xyz", charge=0, mult=1)
# Defining the xTB theory (GFN1-xTB)
theory = xTBTheory(runmode='library', numcores=numcores)
####################################################################

# MD settings
temperature = 300 # Kelvin
timestep = 0.001 # ps
simulation_time = 1 # ps

# US restraint potential settings
RC_atoms=[0,1,2,3]
RC_FC=2000 #Unit?
traj_frequency=100 #Frames saved to trajectory and used in US
filename_prefix="US_window" # Used for created files
M = 20 # M centers of harmonic biasing potentials
theta0 = np.linspace(-math.pi, math.pi, M, endpoint = False) # array of values

# Save MD and US settings to parameterfile
import json
json.dump({'M': M, 'RC_atoms': RC_atoms, 'RC_FC': RC_FC, timestep: 'timestep',
           'simulation_time': simulation_time,
           'traj_frequency': traj_frequency, 'temperature': temperature,
           'theta0': list(theta0), 'filename_prefix': filename_prefix},
          open("ASH_US_parameters.txt", 'w'))

# Loop over windows and run biased simulation in each
# Note: It is more efficient to run these as independent simulations in parallel
for ind,RC_val in enumerate(theta0):
    print("="*50)
    print(f"NEW UMBRELLA WINDOW. Value: {RC_val}")
    print("="*50)
    # Setting restraint potential as a list: [atom_indices, value, force constant]
    restraint=RC_atoms+[RC_val]+[RC_FC] # Combining into 1 list

    # Calling OpenMM_MD with a restraint potential
    OpenMM_MD(fragment=frag, theory=theory,
             timestep=timestep, simulation_time=simulation_time, traj_frequency=traj_frequency,
             temperature=temperature, restraints=[restraint])

    os.rename("trajectory.dcd", f"{filename_prefix}_{ind}.dcd")
