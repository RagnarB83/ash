#ALL conda/PyPi PACKAGES that ASH may use

#Requirements
conda install python
conda install -c conda-forge geometric
conda install -c conda-forge openmm
conda install -c conda-forge julia
conda install -c conda-forge xtb

#Optional: for MM and MD functionality
conda install -c conda-forge plumed
conda install -c conda-forge parmed
conda install -c conda-forge mdanalysis
conda install -c conda-forge ase 

#Optional: for plotting
conda install -c conda-forge scipy
conda install -c conda-forge matplotlib 

#Optional QM program packages
conda install -c psi4 psi4 

#Required pip packages (make sure which pip points to conda environment)
pip3 install julia

#Optional pip packages(MD)
pip3 install plumed
pip3 install mdtraj #may not be needed anymore

#Other optional pip packages
pip3 install pyscf
pip3 install pyframe

