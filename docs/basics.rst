==========================
Basic usage
==========================

#####################
Input structure
#####################
You create a Python3 script (e.g. called system.py) and import the Yggdrasill functionality:

.. code-block:: python

    from yggdrasill import *
    import settings_yggdrasill

For convenience you may want to initalize standard global settings (connectivity etc.):

.. code-block:: python

    settings_yggdrasill.init()

From then on you have the freedom of writing a Python script in any way you prefer but taking the advantage
of Yggdrasill functionality. Typically you would first create one (or more) molecule fragments, then define a theory
object and then call a specific job-module (an optimizer, numerical-frequencies, MD).
See  :doc:`coordinate-input` for various ways of dealing with coordinates and fragments.

#####################
Example script
#####################

Here is a basic Yggdrasill Python script, e.g. named: yggtest.py

.. code-block:: python

    from yggdrasill import *
    import settings_yggdrasill
    settings_yggdrasill.init()

    #Create fragment
    Ironhexacyanide = Fragment(xyzfile="fecn6.xyz")

    #Defining ORCA-related variables
    orcadir='/opt/orca_4.2.1'
    orcasimpleinput="! BP86 def2-SVP Grid5 Finalgrid6 tightscf"
    orcablocks="%scf maxiter 200 end"

    ORCASPcalculation = ORCATheory(orcadir=orcadir, fragment=Ironhexacyanide, charge=0, mult=1,
                                orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

    #Simple Energy SP calc
    ORCASPcalculation.run()


The script above loads Yggdrasill, creates a new fragment from an XYZ file (see :doc:`coordinate-input` for other ways),
defines variables related to the ORCA-interface (see :doc:`orca-interface`), creates an ORCA-theory object
(see :doc:`QM-interfaces`)and finally runs a single-point calculation (see :doc:`job-types`)

#####################
Running script directly
#####################

For a simple job we can just run the script directly

.. code-block:: shell

    python3 yggtest.py

#####################
Submitting job
#####################

For a more complicated job we would probably want to create a job-script that would handle various environmental variables,
dealing with local scratch, copy files back when done etc.
Here is an example SLURM jobscript:


.. code-block:: shell

    #!/bin/bash

    #SBATCH -N 1
    #SBATCH --tasks-per-node=8
    #SBATCH --time=8760:00:00
    #SBATCH -p compute

    # Usage of this script:
    #sbatch -J jobname job-orca-SLURM.sh , where jobname is the name of your ORCA inputfile (jobname.inp).

    # Jobname below is set automatically when submitting like this: sbatch -J jobname job-orca.sh
    #Can alternatively be set manually below. job variable should be the name of the inputfile without extension (.inp)
    job=${SLURM_JOB_NAME}
    job=$(echo ${job%%.*})

    #Python settings
    export PATH=/path/to/python:$PATH
    export PYTHONPATH=/path/to/yggdrasill

    #Setting OPENMPI paths here for QM-codes like ORCA
    export PATH=/users/home/user/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=/users/home/user/openmpi/lib:$LD_LIBRARY_PATH

    export RSH_COMMAND="/usr/bin/ssh -x"
    #ORCA path and LD_LIBRARY_PATH
    export orcadir=/path/to/orca
    export PATH=$orcadir:$PATH
    export LD_LIBRARY_PATH=$orcadir:$LD_LIBRARY_PATH

    # Creating local scratch folder for the user on the computing node.
    #Set the scratchlocation variable to the location of the local scratch, e.g. /scratch or /localscratch
    export scratchlocation=/scratch
    if [ ! -d $scratchlocation/$USER ]
    then
        mkdir -p $scratchlocation/$USER
    fi
    tdir=$(mktemp -d $scratchlocation/$USER/yggdrasilljob__$SLURM_JOB_ID-XXXX)

    # cd to scratch
    cd $tdir

    # Copy job and node info to beginning of outputfile
    echo "Job execution start: $(date)" >  $SLURM_SUBMIT_DIR/$job.out
    echo "Shared library path: $LD_LIBRARY_PATH" >>  $SLURM_SUBMIT_DIR/$job.out
    echo "Slurm Job ID is: ${SLURM_JOB_ID}" >>  $SLURM_SUBMIT_DIR/$job.out
    echo "Slurm Job name is: ${SLURM_JOB_NAME}" >>  $SLURM_SUBMIT_DIR/$job.out
    echo $SLURM_NODELIST >> $SLURM_SUBMIT_DIR/$job.out

    #Run Yggdrasill
    python3 $job.py >>  $SLURM_SUBMIT_DIR/$job.out

    # Yggdrasill has finished here. Now copy important stuff back (xyz files, GBW files etc.). Add more here if needed.

    cp $tdir/*.xyz $SLURM_SUBMIT_DIR




