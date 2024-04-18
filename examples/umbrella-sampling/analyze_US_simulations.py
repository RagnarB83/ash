from ash import *
import glob
import openmm.unit as unit
import math

import matplotlib.pyplot as plt
import json


try:
    from FastMBAR import *
except ModuleNotFoundError:
    print("FastMBAR package not found")
    print("See installation instructions at: https://fastmbar.readthedocs.io/en/latest/installation.html")
    print("pip install -U FastMBAR  should work (though see instructions for faster GPU execution)")
    ashexit()

# Run FastMBAR on GPU or not (requires FastMBAR/pytorch to be installed with GPU support)
CUDA_OPTION = False

##################################
# Analysis of umbrella sampling
##################################

# Get US restraint potential settings from settings file (created by run_US_simulations.py)
# Note: variables below can also be set manually instead
US_settings = json.load(open(f"ASH_US_parameters.txt")) # Load settings from file as a dict
RC_atoms=US_settings["RC_atoms"]
RC_FC = US_settings["RC_FC"]
M = US_settings["M"]
temperature = US_settings["temperature"]
theta0 = US_settings["theta0"]
filename_prefix=US_settings["filename_prefix"]


####################################
# Get RC values from trajectories
####################################

# Grab trajectory-filenames in current-directory as a sorted list
trajectoryfiles = natural_sort(glob.glob("*dcd"))
print(trajectoryfiles)

# Loop over trajectories and grab RCs. Save as CSV-files
# Note: this step can be skipped if CSV files have been created
for ind,trajfile in enumerate(trajectoryfiles):
    theta = MDtraj_coord_analyze(trajfile, pdbtopology="frag.pdb", indices=RC_atoms)
    # Save to file
    np.savetxt(f"./{filename_prefix}_{ind}.csv", theta, fmt = "%.5f", delimiter = ",")

# Read RC values from CSV-files
thetas = []
num_conf = []
for theta0_index in range(M):
    theta = np.loadtxt(f"./{filename_prefix}_{theta0_index}.csv", delimiter = ",")
    thetas.append(theta)
    num_conf.append(len(theta))
thetas = np.concatenate(thetas)
num_conf = np.array(num_conf)
N = len(thetas)

# compute reduced energy matrix A
A = np.zeros((M, N))
K = RC_FC
T = temperature * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB * temperature * unit.kelvin * unit.AVOGADRO_CONSTANT_NA
kbT = kbT.value_in_unit(unit.kilojoule_per_mole)

for theta0_index in range(M):
    current_theta0 = theta0[theta0_index]
    diff = np.abs(thetas - current_theta0)
    diff = np.minimum(diff, 2*math.pi-diff)
    A[theta0_index, :] = 0.5*K*diff**2/kbT

# solve MBAR equations using FastMBAR
print("Starting FastMBAR")
fastmbar = FastMBAR(energy = A, num_conf = num_conf, cuda=CUDA_OPTION, verbose = True)
print("Relative free energies: ", fastmbar.F)

# compute the reduced energy matrix B
L = 25
theta_PMF = np.linspace(-math.pi, math.pi, L, endpoint = False)
width = 2*math.pi / L
B = np.zeros((L, N))

for i in range(L):
    theta_center = theta_PMF[i]
    theta_low = theta_center - 0.5*width
    theta_high = theta_center + 0.5*width

    indicator = ((thetas > theta_low) & (thetas <= theta_high)) | \
                 ((thetas + 2*math.pi > theta_low) & (thetas + 2*math.pi <= theta_high)) | \
                 ((thetas - 2*math.pi > theta_low) & (thetas - 2*math.pi <= theta_high))

    B[i, ~indicator] = np.inf

# compute PMF using the energy matrix B
results = fastmbar.calculate_free_energies_of_perturbed_states(B)
PMF = results['F']
PMF_uncertainty = results['F_std']

# plot the PMF
fig = plt.figure(0)
fig.clf()
plt.errorbar(theta_PMF*180/math.pi, PMF, yerr = PMF_uncertainty, fmt = '-o',
ecolor = 'black', capsize = 2, capthick = 1, markersize = 6)
plt.xlim(-180, 180)
plt.xlabel("dihedral")
plt.ylabel("reduced free energy")
plt.savefig("PMF_fastmbar.pdf")
