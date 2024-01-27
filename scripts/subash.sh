#!/bin/bash
#subash: Submission script for ASH for Slurm
#Usage: subash.sh ash_script.py

#######################################
#CLUSTER SETTINGS (to be modified by user)

#Default name of cluster queue to submit to. Can be verridden by -q or --queue option
queue=LocalQ

#Default walltime. Overridden by -w or --walltime
walltime=900

#Default memory per CPU
memory_per_cpu=10

#Default number of threads (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENMM_CPU_THREADS)
#NOTE: This should almost always be set to 1 (ASH will control threads on the fly)
threads=1  #If threads=auto then threads are set to SLURM CPU cores (usually not recommended)

#Default number of GPU slots to ask Slurm for
number_of_gpus=0
gpu_memory=1
#-g gpu_mem:4G

# Path to local or global scratch on computing node (Script will create temporary user and job-directory here)
#8TB HDD scratch: /data-hdd/SCRATCH
#1 TB SSD scratch. /data-nvme/SCRATCH
SCRATCHLOCATION=/data-hdd/SCRATCH

#Path to bash file containing PATH and LD_LIBRARY_PATH definitions
#This shell-script will be sourced upon submission.
#This script should define Python environment for ASH and all external QM programs
ENVIRONMENTFILE=$HOME/set_environment_ash.sh

#######################################
# End of user modifications (hopefully)
#######################################


#Colors
green=`tput setaf 2`
yellow=`tput setaf 3`
normal=`tput sgr0`
cyan=`tput setaf 6`

print_usage () {
  echo "${green}subash${normal}"
  echo "${yellow}Usage: subash input.py      Dir should contain .py Python script.${normal}"
  echo "${yellow}Or: subash input.py -p 8      Submit with 8 cores.${normal}"
  echo "${yellow}Or: subash input.py -g 1:1GB      Submit with 1 GPU core and request 1 GB of memory.${normal}"
  echo "${yellow}Or: subash input.py -m X      Memory setting per CPU (in GB): .${normal}"
  echo "${yellow}Or: subash input.py -t T      Submit using T threads (for OpenMP/OpenMM/MKL).${normal}"
  echo "${yellow}Or: subash input.py -s /path/to/scratchdir      Submit using specific scratchdir .${normal}"
  echo "${yellow}Or: subash input.py -mw            Submit multi-Python job (multiple walkers) .${normal}"
  echo "${yellow}Or: subash input.py -n node54      Submit to specific node (-n, --node).${normal}"
  echo "${yellow}Or: subash input.py -q queuename    Submit to specific queue: .${normal}"
  exit
}

arguments=$@
argument_first=$1
file=$argument_first
argumentnum=$#
#echo "Arguments provided : $arguments"

#If positional argument not .py then exit
if [[ $argument_first != *".py"* ]]; then
echo "No .py file provided. Exiting..."
echo
print_usage
fi

#multiwalker default false
multiwalker=false

#Go through arguments
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
      -branch|--branch)
      ashbranch="$2"
      shift # past argument
      ;;
      -p|--procs|--cores|--numcores) #Number of cores
      numcores="$2"
      shift # past argument
      shift # past value
      ;;
      -q|--queue) #Name of queue
      queue="$2"
      shift # past argument
      shift # past value
      ;;
      -t|--threads) #Number of threads
      threads="$2"
      shift # past argument
      shift # past value
      ;;
      -m|--mempercpu) #Memory per core (GB)
      memory_per_cpu="$2"
      shift # past argument
      shift # past value
      ;;
      -g|--gpu) #Number of GPUcores and memory
      gpustuff="$2"
      gpuoptions=(${gpustuff//:/ })
      number_of_gpus=${gpuoptions[0]}
      gpu_memory=${gpuoptions[1]}
      shift # past argument
      shift # past value
      ;;
      -mw|--multiwalker) #Multiwalker
      multiwalker=true
      shift # past argument
      #shift # past value
      ;;
      -w|--walltime) #Walltime
      walltime="$2"
      shift # past argument
      shift # past value
      ;;
      -n|--node) #Name of node
      specificnode="$2"
      shift # past argument
      shift # past value
      ;;
      -s|--scratchdir) #Name of scratchdir
      SCRATCHLOCATION="$2"
      shift # past argument
      shift # past value
      ;;
      --default)
      DEFAULT=YES
      ;;
      *)    #
      shift # past argument
  esac
done

#Now checking if numcores are defined
if [[ $numcores == "" ]]
then
  #Grabbing numcores from input-file.py if not using -p flag
  echo "Numcores not provided (-p option). Trying to grab cores from Python script."
  var=$(grep '^numcores' $file)
  NPROC=$(echo $var | awk -F'=' '{print $NF}')
  numcores=$(echo $NPROC | sed -e 's/^[[:space:]]*//')
  if ((${#numcores} == 0))
  then
    echo "No numcores variable defined Python script. Exiting..."
    exit
  fi
fi

#Memory setting
echo "Memory setting per CPU: $memory_per_cpu GB"
slurm_mem_line="#SBATCH --mem-per-cpu=${memory_per_cpu}G"

#################
#THREADS-settings
#################
#Applies to programs using multithreading that needs to be controlled by:
#OMP_NUM_THREADS, MKL_NUM_THREADS or OPENMM_CPU_THREADS variables
#WARNING: Typically we want threads=1 and have ASH program
#modify all external-program parallelization (whether MPI, OpenMP or other threading) via numcores option to Theories.

#If threads is set to auto:
if [[ $threads == auto ]]
then
echo "Threads-setting is auto. Setting threads equal to numcores: $numcores"
threads=$numcores
else
echo "Threads set to $threads"
fi
#################

#################
#GPU settings
#################
if [[ $number_of_gpus != 0 ]]
then
echo "GPUs requested: $number_of_gpus"
echo "GPU memory: $gpu_memory"
slurm_gpu_line1="#SBATCH --gres=gpu:$number_of_gpus"
slurm_gpu_line2="#SBATCH --mem-per-gpu $gpu_memory"
else
slurm_gpu_line1=""
slurm_gpu_line2=""
fi

######################
#Job-script creation
######################
rm -rf ash.job
cat <<EOT >> ash.job
#!/bin/bash

#SBATCH -N 1
#SBATCH --tasks-per-node=$numcores
#SBATCH --time=$walltime:00:00
#SBATCH -p $queue
$slurm_gpu_line1
$slurm_gpu_line2
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
$slurm_mem_line

export job=\$SLURM_JOB_NAME
export job=\${job%%.*}

#Outputname
outputname="\$job.out"

#Multiwalker option
multiwalker=$multiwalker


#NUM_CORES
NUM_CORES=\$((SLURM_JOB_NUM_NODES*SLURM_CPUS_ON_NODE))


#Setting MKL_NUM_THREADS, OMP_NUM_THREADS,OPENMM_CPU_THREADS to threads variable (should be 1 usually)
#Note: Both OpenMM and pyscf threading behaved oddly unless we set this to 1 initially
export MKL_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OPENMM_CPU_THREADS=$threads
export OMP_STACKSIZE=1G
export OMP_MAX_ACTIVE_LEVELS=1

echo "OPENMM_CPU_THREADS: \$OPENMM_CPU_THREADS"
echo "MKL_NUM_THREADS: \$MKL_NUM_THREADS"
echo "OMP_NUM_THREADS: \$OMP_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS :\$OPENBLAS_NUM_THREADS"
echo "Note: ASH may change these environment variables on the fly"

# Usage:
#ulimit -u unlimited
#limit stacksize unlimited

#Create scratch
scratchlocation=$SCRATCHLOCATION
echo "scratchlocation: \$scratchlocation"
#Checking if scratch drive exists
if [ ! -d \$scratchlocation ]
then
echo "Problem with scratch directory location: \$scratchlocation"
echo "Is scratchlocation in subash script set correctly ?"
echo "Exiting"
exit
fi

#Creating user-directory on scratch if not available
if [ ! -d \$scratchlocation/\$USER ]
then
  mkdir -p \$scratchlocation/\$USER
fi
#Creating temporary dir on scratch
tdir=\$(mktemp -d \$scratchlocation/\$USER/ashjob__\$SLURM_JOB_ID-XXXX)
echo "Creating temporary tdir : \$tdir"

#Checking if directory exists
if [ -z \$tdir ]
then
echo "tdir variable empty: \$tdir"
echo "Problem creating temporary dir: \$scratchlocation/\$USER/ashjob__\$SLURM_JOB_ID-XXXX"
echo "Is scratch-disk (\$scratchlocation) writeable on node: \$SLURM_JOB_NODELIST  ?"
echo "Exiting"
exit
fi

#Checking if tdir exists
if [ ! -d \$tdir ]
then
echo "Problem creating temporary dir: \$scratchlocation/\$USER/ashjob__\$SLURM_JOB_ID-XXXX"
echo "Is scratch-disk (\$scratchlocation) writeable on node: \$SLURM_JOB_NODELIST  ?"
echo "Exiting"
exit
fi
chmod +xr \$tdir
echo "tdir: \$tdir"

cp \$SLURM_SUBMIT_DIR/*.py \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.dat \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.cif \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.xyz \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.c \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.gbw \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.molden \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*nat \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.chk \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.xtl \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.ff \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.ygg \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.pdb \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.info \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/POTENTIAL \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/BASIS_MOLOPT \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/qmatoms \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/hessatoms \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/Hessian* \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/act* \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.xml \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.txt \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.rtf \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.prm \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.gro \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.psf \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.rst7 \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.crd \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.top \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*.itp \$tdir/ 2>/dev/null
cp \$SLURM_SUBMIT_DIR/*prmtop \$tdir/ 2>/dev/null

echo "Node(s): \$SLURM_JOB_NODELIST"
# cd to scratch
echo "Entering scratchdir: \$tdir"
cd \$tdir
header=\$(df -h | grep Filesy)
scratchsize=\$(df -h | grep \$scratchlocation)

# Copy job and node info to beginning of outputfile
echo "Starting job in scratch dir: \$tdir" > \$SLURM_SUBMIT_DIR/\$outputname
echo "Job execution start: \$(date)" >> \$SLURM_SUBMIT_DIR/\$outputname
echo "Shared library path: \$LD_LIBRARY_PATH" >> \$SLURM_SUBMIT_DIR/\$outputname
echo "Slurm Job ID is: \${SLURM_JOB_ID}" >> \$SLURM_SUBMIT_DIR/\$outputname
echo "Slurm Job name is: \${SLURM_JOB_NAME}" >> \$SLURM_SUBMIT_DIR/\$outputname
echo "Nodes: \$SLURM_JOB_NODELIST" >> \$SLURM_SUBMIT_DIR/\$outputname
echo "Scratch size before job:" >> \$SLURM_SUBMIT_DIR/\$outputname
echo "\$header" >> \$SLURM_SUBMIT_DIR/\$outputname
echo "\$scratchsize" >> \$SLURM_SUBMIT_DIR/\$outputname

#ASH environment
#This activates the correct Python and ASH environment and some external programs
source $ENVIRONMENTFILE

echo "PATH is \$PATH"
echo "LD_LIBRARY_PATH is \$LD_LIBRARY_PATH"
export OMPI_MCA_btl=vader,self
export OMPI_MCA_btl_vader_single_copy_mechanism=none
echo "Running ASH job"

#Start ASH job from scratch dir.  Output file is written directly to submit directory
export PYTHONUNBUFFERED=1


# Multiple walker ASH run (intended for multiwalker metadynamics primarily

if [ "\$multiwalker" = true ]
then
  echo "Multiwalker True! NUM_CORES: \$NUM_CORES"
  #Creating multiple subdir walkersim$i HILLS files are stored in $tdir
  for (( i=0; i<\$NUM_CORES; i++ ))
  do
      echo "Creating dir: walkersim\$i"  >> \$SLURM_SUBMIT_DIR/\$outputname
      mkdir walkersim\$i
      echo "Copying files to dir: walkersim\$i"  >> \$SLURM_SUBMIT_DIR/\$outputname
      cp * walkersim\$i/
      cd walkersim\$i
      echo "Entering dir: walkersim\$i"  >> \$SLURM_SUBMIT_DIR/\$outputname
      echo "Process launched : \$i"  >> \$SLURM_SUBMIT_DIR/\$outputname
      sleep 2
      python3 \$job.py >> \$SLURM_SUBMIT_DIR/\${job}_walker\${i}.out 2>&1 &
      declare P\$i=\$!
      cd ..
  done
  wait
  # \$P1 \$P2  #Does not matter?

else
  #Regular job
  python3 \$job.py >> \$SLURM_SUBMIT_DIR/\$outputname 2>&1

fi

#Making sure to delete potentially massive  files before copying back
rm -rf core.*  #Fortran-program  segfaults
rm -rf orca.*tmp* #ORCA tmp files from e.g. MDCI

header=\$(df -h | grep Filesy)
echo "header: \$header"
scratchsize=\$(df -h | grep \$scratchlocation)
echo "Scratch size after job: \$scratchsize"

# Ash has finished. Now copy important stuff back.
outputdir=\$SLURM_SUBMIT_DIR/\${job}_\${SLURM_JOB_ID}
cp -r \$tdir \$outputdir

# Removing scratch folder
rm -rf \$tdir

EOT
######################

#Submit job.
if [[  -z "$specificnode" ]]; then
   sbatch -J $file ash.job
else
   #Submit to a specific node
   echo "Submitting to specific node: $specificnode"
   sbatch -J $file -w $specificnode ash.job
fi
echo "${cyan}ASH job: $file submitted using $numcores cores.${normal}"
echo "Queue: $queue and walltime: $walltime"

#Multiwalker
if [[ "$multiwalker" == true ]]
then
  echo "Multiwalker option chosen. ASH will create multiple dirs on scratch and submit $numcores jobs"
  echo "Make sure to adjust numcores inside ASH script!"
fi
