#!/bin/bash

################################################
#AUTO-INSTALLATION SCRIPT FOR ASH
#this avoids manual setup of Python, Julia etc.
################################################
#__________________
#Settings
#__________________

#Download Julia or not (otherwise a julia tar.gz file is needed)
download_julia=true

#Julia version to download/use (1.6.1 is recommended)
juliaversion="1.6.1"

#Path to Python3 executable can be set below (uncomment first). If not set, script will try to find python3 in PATH
#path_to_python3_exe="/usr/bin/python3"

# Force pip to install in user home directory instead of in default global Python location.
localuserpipoption=false 

#Whether to install Python packages in ASH dir instead. Potentially problematic
localpipinstallation=false

# Use conda and python inside Julia. Problematic and disabled
#use_julia_conda=true #problem with python3 binary inside Conda.jl


#__________________
# END OF SETTINGS
#__________________


###############################################
echo "-------------------------------"
echo "ASH installation script"
echo "-------------------------------"
#Check if path_to_python3_exe has been set. Otherwise search for python3 in $PATH
if [ -z ${path_to_python3_exe+x} ]
then
  echo "path_to_python3_exe is unset"
  echo "Searching for python3 in PATH"
  path_to_python3_exe=$(which python3)
  #Check if fou
  if [ $? -eq 1 ]
  then
      echo "Did not find a python3 executable in PATH. Put a Python3 installation in PATH (or load a module)"
      exit
  fi
  echo "Found: $path_to_python3_exe"
else
  echo "path_to_python3_exe has been set to : $path_to_python3_exe"
fi

#Dirname only
path_to_python3_dir=${path_to_python3_exe%/python3}
echo ""
echo "Python3 path: $path_to_python3_dir"
echo ""

if [[ ! -f python3_ash ]]
then
echo "This script must be run when inside the ash directory! Please cd to it first"
exit
fi

if [[ ! -d $path_to_python3_dir ]]
then
echo "Did not find a valid Python3 dir: $path_to_python3_dir"
exit
fi

#thisdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
thisdir=$PWD
echo ""
echo "Using Python3 installation in: $path_to_python3_dir"
echo "Make sure this is the Python3 installation you want"
echo ""
echo "Current directory is:  $thisdir"

echo ""
echo "Step 1. Julia setup"

# Julia major version var: 1.6.1 => 1.6. Used in download URL
juliamajorversion=${juliaversion%??}

#Download Julia and uncompress
if [ $download_julia = true ]
then
  echo "Downloading Julia"
  rm -rf julia-${juliaversion}
  rm -rf julia-${juliaversion}-linux-x86_64.tar.gz
  rm -rf julia-${juliaversion}-linux-x86_64.tar
  wget https://julialang-s3.julialang.org/bin/linux/x64/${juliamajorversion}/julia-${juliaversion}-linux-x86_64.tar.gz
else
 echo "Skipping Julia download. Assuming file julia-${juliaversion}-linux-x86_64.tar.gz is present"
fi

#Deleting old
rm -rf julia-${juliaversion}-linux-x86_64.tar
#Decompress archive
gunzip julia-${juliaversion}-linux-x86_64.tar.gz
tar -xf julia-${juliaversion}-linux-x86_64.tar

path_to_julia=$thisdir/julia-${juliaversion}/bin

#Delete old and Create julia-python-bundle dir
rm -rf julia-python-bundle
mkdir -p julia-python-bundle

# Set Julia packages path
export JULIA_DEPOT_PATH=$thisdir/julia-python-bundle

#Install Julia packages
echo "Step 2. Downloading and installing Julia packages"
$path_to_julia/julia julia-packages-setup.jl

#Adding Julia to PATH
export PATH=$path_to_julia:$PATH

echo "Julia packages and setup done"
echo ""
echo "Step 3. Downloading and installing Python3 packages"

#Check if pip or pip3 is in correct location
path_to_pip_exe=$(which pip)
path_to_pip_dir=${path_to_pip_exe%/pip}
path_to_pip3_exe=$(which pip3)
path_to_pip3_dir=${path_to_pip3_exe%/pip3}

echo "Finding correct pip"
if [[ ${path_to_pip_dir} == $path_to_python3_dir ]]
then
  pipcommand=$path_to_pip_exe
  echo "pipcommand is : $pipcommand"
elif [[ ${path_to_pip3_dir} == $path_to_python3_dir ]]
then
  pipcommand=$path_to_pip3_exe
  echo "pipcommand is : $pipcommand"
else
  echo "Did not find pip executable in same dir as python3"
  echo "which pip gives: $path_to_pip_exe"
  echo "which pip3 gives: $path_to_pip3_exe"
  echo "something wrong with environment?"
  echo "Exiting."
  exit
fi

if [[ $localpipinstallation == true ]]
then
echo "Installing python packages in local dir: $thisdir/pythonpackages "
mkdir pythonpackages
export PIP_TARGET=$thisdir/pythonpackages
fi

# Option to force pip install in user's home directory
if [[ $localuserpipoption == true ]]
then
piparg="--user"
else
piparg=""
fi

#Install numpy in case missing
$pipcommand install numpy $piparg

#Geometric
$pipcommand install geometric $piparg

#PyJulia. Julia needs to be available
$pipcommand install julia $piparg


#elif [ $use_julia_conda = true ]
#then
#  $thisdir/julia-${juliaversion}/bin/julia julia-conda-setup.jl
  #Setting Python3 as Conda.jl Python3
   #Problem. Not working well
#  path_to_python3_dir=$thisdir/julia-python-bundle/conda/3/bin/python3
#fi



# Change python3 to be used in python3_ash to the Conda.jl python3
echo "Step 4. Modifying python3_ash binary"
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
echo "Step 5. Creating set_environent_ash.sh script"
echo "#!/bin/bash" > set_environment_ash.sh
echo "export ASHPATH=${thisdir}" >> set_environment_ash.sh
echo "export python3path=${path_to_python3_dir}" >> set_environment_ash.sh
echo "export JULIAPATH=${thisdir}/julia-${juliaversion}/bin" >> set_environment_ash.sh
echo "export JULIA_DEPOT_PATH=${thisdir}/julia-python-bundle" >> set_environment_ash.sh
echo "export PYTHONPATH=\$ASHPATH:\$ASHPATH/lib:$ASHPATH/pythonpackages:\$PYTHONPATH" >> set_environment_ash.sh
echo "export PATH=\$python3path:\$ASHPATH:\$JULIAPATH:\$PATH" >> set_environment_ash.sh
echo "export LD_LIBRARY_PATH=\$ASHPATH/lib:\$LD_LIBRARY_PATH" >> set_environment_ash.sh


echo "Installation of ASH was successful!"
echo ""
echo "Remember:"
echo "     - Run: source ${thisdir}/set_environment_ash.sh to activate ASH!"
echo "     - Put source command in your .bash_profile/.zshrc/.bashrc and job-submission scripts"
echo "     - use python3_ash as your main Python executable when running ASH Python scripts!"