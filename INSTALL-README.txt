
ASH is 99% Python. A Python3 distribution is required and as packages will have to be installed you need to have
write/install access to the Python.

Dependencies.
- Numpy library inside Python.
- Julia installation.
- PyJulia installation (Python package via pip)
-


#INSTALLATION AND SETUP
1. Install Ash e.g. in your home directory.

2. Check if a suitable Python distribution is available. Needs to contain numpy and you will need to be able to install
Python packages using pip. If you don't have one then go to 2b.

2b. Anaconda Python3 setup (recommended)
  i. Download Anaconda Python3 package (https://www.anaconda.com/products/individual) and install in e.g. your user directory. Set up

2c. Create a new conda Python3.7 virtual environment (here called ashpy37) that will be used for Ash:
        conda create -n ashpy37 python=3.7 numpy
     Switch to that environment: conda activate ashpy37. Alternatively you can use the default base environment

3. To make Ash part of the Python environment do: export PYTHONPATH=/path/to/ash:$PYTHONPATH  where /path/to/ash is the dir where all the ASH sourcefiles are.
    Also the LD_LIBRARY_PATH need to be set:
      export LD_LIBRARY_PATH=/path/to/ash:$LD_LIBRARY_PATH
    Put these environment definitions in your shell environment file e.g. .bashrc, .bash_profile, .zshrc etc.

3. Install necessary Python packages via pip:
   Strongly recommended:
   pip install geometric   (geomeTRIC optimizer)


4. Make sure preferred QM packages are available:
    - The path to ORCA needs to be in PATH and LD_LIBRARY_PATH
    - xTB needs be in PATH

julia


Fortran


Optional Python packages to install
   pip install pyberny     (pyBerny optimizer)
   pip install pyscf       (PySCF QM program)
   pip install pyframe     (polarizable embedding helper tool)



