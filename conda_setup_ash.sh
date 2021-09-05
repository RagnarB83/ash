echo "Searching for python3 in PATH"
path_to_python3_exe=$(which python3)
path_to_julia_exe=$(which julia)
#Check if fou
  if [ $? -eq 1 ]
  then
      echo "Did not find a python3 executable in PATH. Put a Python3 installation in PATH (or load a module)"
      exit
  fi
  echo "Found: $path_to_python3_exe"

#Dirname only
path_to_python3_dir=${path_to_python3_exe%/python3}
path_to_julia_dir=${path_to_julia_exe%/julia}
echo ""
echo "Python3 path: $path_to_python3_dir"
echo "Julia path: $path_to_julia_dir"
echo ""

thisdir=$PWD

# Change python3 to be used in python3_ash to the Conda.jl python3
echo "Creating python3_ash script:"
#NOTE: This is not really necessary if set_environment_ash.sh is set
#sed -i "s:/usr/bin/env python3:/usr/bin/env ${path_to_python3_dir}/python3:g" python3_ash
echo "#!/usr/bin/env ${path_to_python3_dir}/python3" > python3_ash
echo "# -*- coding: utf-8 -*-" >> python3_ash
echo "#Note: python-jl fix so that PyJulia works without problems" >> python3_ash
echo "#Note: This file needs to be made executable: chmod +x python3_ash" >> python3_ash
echo "import sys" >> python3_ash
echo "import re" >> python3_ash
echo "" >> python3_ash
echo "from julia.python_jl import main" >> python3_ash
echo "if __name__ == '__main__':" >> python3_ash
echo "    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])" >> python3_ash
echo "    sys.exit(main())" >> python3_ash
#Making python3_ash executable
chmod uog+x python3_ash

#Create set_environment_ash.sh file
echo "Creating set_environent_ash.sh script"
echo "#!/bin/bash" > set_environment_ash.sh
echo "ulimit -s unlimited" >> set_environment_ash.sh
echo "export ASHPATH=${thisdir}" >> set_environment_ash.sh
echo "export python3path=${path_to_python3_dir}" >> set_environment_ash.sh
echo "export JULIAPATH=${path_to_julia_dir}" >> set_environment_ash.sh
#echo "export JULIA_DEPOT_PATH=${thisdir}/julia-python-bundle" >> set_environment_ash.sh
#echo "export PYTHONPATH=\$ASHPATH:\$ASHPATH/lib:$ASHPATH/pythonpackages:\$PYTHONPATH" >> set_environment_ash.sh
echo "export PATH=\$python3path:\$ASHPATH:\$JULIAPATH:\$PATH" >> set_environment_ash.sh
echo "export LD_LIBRARY_PATH=\$ASHPATH/lib:\$LD_LIBRARY_PATH" >> set_environment_ash.sh


echo "Installation of ASH was successful!"
echo ""
echo "Remember:"
echo "     - Run: source ${thisdir}/set_environment_ash.sh to activate ASH!"
echo "     - Put source command in your .bash_profile/.zshrc/.bashrc and job-submission scripts"
echo "     - use python3_ash as your main Python executable when running ASH Python scripts!"