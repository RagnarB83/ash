#!/bin/bash

################################################
#AUTO-INSTALLATION SCRIPT FOR ASH
#this avoids manual setup of Python, Julia etc.
################################################
#________
#Settings
#________

download_julia=true

#Julia version to use (1.6.1 is recommended)
juliaversion="1.6.1"

#Path to Python3 executable can be set here. Otherwise script will try to find python3 in PATH
#path_to_python3_exe="/usr/bin/python3"

#use_julia_conda=true #problem with python3 binary inside Conda.jl
#######################

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
echo
echo "-------------------------------"
echo "ASH installation script"
echo "Using Python3 installation in: $path_to_python3_dir"
echo "Make sure this is the Python3 installation you want"
echo ""
echo "Current directory is:  $thisdir"

echo ""
echo "Step 1. Downloading and installing Julia"

#Download previous Julia dirs
rm -rf julia-${juliaversion}
rm -rf julia-python-bundle

#Download Julia 1.6.1 and uncompress
if [ $download_julia = true ]
then
  wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz
fi

#Deleting old
rm -rf julia-${juliaversion}-linux-x86_64.tar
#Decompress archive
gunzip julia-${juliaversion}-linux-x86_64.tar.gz
tar -xvf julia-${juliaversion}-linux-x86_64.tar

path_to_julia=$thisdir/julia-${juliaversion}/bin
#Create julia-python-bundle dir
mkdir -p julia-python-bundle

#TODO: get full path?
# Set Julia packages path
export JULIA_DEPOT_PATH=$thisdir/julia-python-bundle

#Install Julia packages
echo "Step 2. Downloading and install Julia packages"
$path_to_julia/julia julia-packages-setup.jl

#Adding Julia to PATH
export PATH=$path_to_julia:$PATH

echo "Julia packages and setup done"
echo ""
echo "Step 3. Downloading and installing Python3 packages"
if [ $path_to_python3_dir ]
then

  #Install numpy just in case
  pip3 install numpy
  #Geometric
  pip3 install geometric
  #PyJulia. Julia needs to be available
  pip3 install julia
fi
#elif [ $use_julia_conda = true ]
#then
#  $thisdir/julia-${juliaversion}/bin/julia julia-conda-setup.jl
  #Setting Python3 as Conda.jl Python3
   #Problem. Not working well
#  path_to_python3_dir=$thisdir/julia-python-bundle/conda/3/bin/python3
#fi



# Change python3 to be used in python3_ash to the Conda.jl python3
echo "Step 4. Modifying python3_ash binary"
sed -i "s:/usr/bin/env python3:/usr/bin/env ${path_to_python3_dir}/python3:g" python3_ash

#Making python3_ash executable
chmod uog+x python3_ash

#Create set_environment_ash.sh file
echo "Step 5. Creating set_environent_ash.sh script"
echo "#!/bin/bash" > set_environment_ash.sh
echo "export ASHPATH=${thisdir}" >> set_environment_ash.sh
echo "export python3path=${path_to_python3_dir}" >> set_environment_ash.sh
echo "export JULIAPATH=${thisdir}/julia-${juliaversion}/bin" >> set_environment_ash.sh
echo "export JULIA_DEPOT_PATH=${thisdir}/julia-python-bundle" >> set_environment_ash.sh
echo "export PYTHONPATH=\$ASHPATH:\$ASHPATH/lib:\$PYTHONPATH" >> set_environment_ash.sh
echo "export PATH=\$python3path:\$ASHPATH:\$JULIAPATH:\$PATH" >> set_environment_ash.sh
echo "export LD_LIBRARY_PATH=$ASHPATH/lib:\$LD_LIBRARY_PATH" >> set_environment_ash.sh


echo "Installation of ASH is successful!"
echo ""
echo "Remember:"
echo "     - Run: source ${thisdir}/set_environment_ash.sh to activate ASH!"
echo "     - Put source command in your .bash_profile/.zshrc/.bashrc and job-submission scripts"
echo "     - use python3_ash as your main Python executable when running ASH Python scripts!"