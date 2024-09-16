#The conda/PyPi PACKAGES that ASH may use
#Run the lines for the packages you want within the conda/mamba environment. Make sure the correct conda environment is active
#Note: mamba is usually a faster alternative to conda (use the one you have installed or prefer)

####################################
# CRITICAL PACKAGES
# (should already have been installed)
####################################
#mamba install python
#mamba install numpy
#pip install geometric
#pip install packaging
#pip install pytest

####################################
# RECOMMENDED PACKAGES:
####################################

#OpenMM (all MM and MD functionality in ASH)
mamba install -c conda-forge openmm # 745MB
#xTB: semiempirical program
mamba install -c conda-forge xtb # 8MB
# pdbfixer: Needed for MM of biomolecules
mamba install -c conda-forge pdbfixer # 0.5 MB
#pySCF: Good QM program
python -m pip  install pyscf # 50 MB
#mdtraj: Needed for basic MD trajectory analysis
mamba install -c conda-forge mdtraj  # 8 MB

#Optional:  required for plotting in ASH
mamba install -c conda-forge scipy #17 MB
mamba install -c conda-forge matplotlib # 203 MB

####################################
# RARELY NEEDED PACKAGES:
####################################

#Optional: only for some special MM and MD functionality in ASH
mamba install -c conda-forge parmed # 36 MB
mamba install -c conda-forge plumed # 10 MB

#Julia-Python interface (sed for molecular crystal QM/MM, NonBondedTheory)
mamba install conda-forge::pyjuliacall
# Alternative: python -m pip install juliacall # 0.1 MB
#Julia installation (you can also follow the instructions at https://julialang.org/downloads/)
mamba install -c conda-forge julia # 146MB

#Optional QM program packages
#Psi4
mamba install -c psi4 psi4

####################################
# VERY RARELY NEEDED PACKAGES:
####################################
#Mdanalysis (alternative to mdtraj)
mamba install -c conda-forge mdanalysis # 38 MB
#ASE: Atomic simulation environment (if using ASH-ASE interface)
mamba install -c conda-forge ase  # 2 MB
#PyFrame for Fragment-based Multiscale Embedding
python -m pip  install pyframe # 0.2 MB
