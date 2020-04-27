#Getting path of this shell-script (resids in Yggdrasill main dir)
SCRIPT=$(readlink -f "$0")
yggdrasill_path=$(dirname "$SCRIPT")

echo "Setting PYTHONPATH and LD_LIBRARY_PATH for Yggdrasill"
#PYTHONPATH
export PYTHONPATH=$yggdrasill_path:$PYTHONPATH
#Setting LD_LIBRARY_PATH for lib dir
export LD_LIBRARY_PATH=$yggdrasill_path/lib:$LD_LIBRARY_PATH