#The conda/PyPi PACKAGES that ASH may use
#Run the lines for the packages you want within the conda environment. Make sure the correct conda environment is active
#Requirements
#Note: mamba is a faster alternative to conda but both work
mamba install -c conda-forge openmm
mamba install -c conda-forge julia
mamba install -c conda-forge xtb

#Optional: only for extra MM and MD functionality
mamba install -c conda-forge pdbfixer
mamba install -c conda-forge plumed
mamba install -c conda-forge parmed
mamba install -c conda-forge mdanalysis
mamba install -c conda-forge ase 

#Optional: only for plotting
mamba install -c conda-forge scipy
mamba install -c conda-forge matplotlib 

#Optional QM program packages
mamba install -c psi4 psi4 

############################
#Required PyPi packages
# WARNING: make sure the correct python interpreter (of the conda/mamba environment) is loaded

#PythonCall/Julicall interface (recommened)
python -m pip install juliacall

#PyJulia interface (not recommended). Alternative to PythonCall/JuliaCall
#pip3 install julia

#Optional pip packages(MD)
python -m pip  install plumed
python -m pip  install mdtraj #may not be needed anymore

#Other optional pip packages
python -m pip  install pyscf
python -m pip  install pyframe

