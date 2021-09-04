#!/bin/bash
ulimit -s unlimited
export ASHPATH=/Users/bjornsson/ASH/ash-dev/ash
export python3path=/Users/bjornsson/miniconda/envs/ASHnewv1/bin
export PYTHONPATH=$ASHPATH:$ASHPATH/lib:/pythonpackages:$PYTHONPATH
export PATH=$python3path:$ASHPATH:$JULIAPATH:$PATH
export LD_LIBRARY_PATH=$ASHPATH/lib:$LD_LIBRARY_PATH
