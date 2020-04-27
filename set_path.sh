
CUR_PATH=$(pwd)
echo "CUR_PATH : $CUR_PATH"
SCRIPT=$(readlink -f "$0")
echo "SCRIPT : $SCRIPT"
yggpath=/home/bjornsson/YGGDRASILL-DEV_GIT/yggdrasill

echo "yggpath : $yggpath"

export PYTHONPATH=$yggpath:$PYTHONPATH
export LD_LIBRARY_PATH=$yggpath/lib:$LD_LIBRARY_PATH