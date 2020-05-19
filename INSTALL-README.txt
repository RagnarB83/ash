
ASH is 99% Python with 1 % Julia. A Python3 distribution is required and as packages will have to be installed you need to access
to install Python packages via pip.

Dependencies.
- Numpy library inside Python.
- Julia installation. PyCall library required.
- PyJulia installation (Python package via pip)
- geomeTRIC


#INSTALLATION AND SETUP
1. Install Ash e.g. in your home directory.

2. Check if a suitable Python distribution is available. Needs to contain numpy and you will need to be able to install
Python packages using pip. If you don't have one then go to 2b.

2b. Anaconda Python3 setup (recommended)
  i. Download Anaconda Python3 package (https://www.anaconda.com/products/individual) and install in e.g. your user directory. Set up

2c. Create a new conda Python3.7 virtual environment (here called ashpy37) that will be used for Ash:
        conda create -n ashpy37 python=3.7 numpy
     Switch to that environment: conda activate ashpy37. Alternatively you can use the default base environment

3. To make Ash part of the Python environment do:
    export PYTHONPATH=/path/to/ash:/path/to/ash/lib:$PYTHONPATH  where /path/to/ash is the dir where all the ASH sourcefiles are.
    Also the LD_LIBRARY_PATH need to be set:
      export LD_LIBRARY_PATH=/path/to/ash/lib:$LD_LIBRARY_PATH
    Put these environment definitions in your shell environment startup file e.g. .bashrc, .bash_profile, .zshrc etc.

3. Install necessary Python packages via pip:
   Strongly recommended:
   pip install geometric   (geomeTRIC optimizer)

5a. Install Julia from https://julialang.org/downloads
    i. Download appropriate binaries from site. Extract archive a
       Add Julia binaries to path: e.g. export PATH=/path/to/julia-1.4.1/bin:$PATH
       Put PATH definition to your shell startup file.

    ii. Launch Julia to install PyCall:
        julia      #This launches the julia interpreter
        using Pkg  # activate Pkg manager
        Pkg.add("PyCall")  #Install PyCall library
        exit()

        If there is an error like this: ERROR: SystemError: opening file "/path/to/.julia/registries/General/Registry.toml": No such file or directory
        Execute in shell: rm -rf ~/.julia/registries/General   (assuming Julia is installed in ~).


5b. Install PyJulia: https://pyjulia.readthedocs.io:
    i. pip install julia

    ii. Activate PyJulia by opening up the python3 interpreter, import julia library and install:
    python3
    import julia
    julia.install()

    #If this is successful then the python-jl binary (installed by PyJulia) should be available.


6. Make sure preferred QM packages are available:
    - The path to ORCA needs to be in PATH and LD_LIBRARY_PATH of your shell.
    - xTB needs to be in PATH



Optional Python packages to install
   pip install pyberny     (pyBerny optimizer)
   pip install pyscf       (PySCF QM program)
   pip install pyframe     (polarizable embedding helper tool)



