#
SCRIPT=$(readlink -f "$0")
yggdrasill_path=$(dirname "$SCRIPT")
echo "yggdrasill_path : $yggdrasill_path"

echo "Setting PATH and LD_LIBRARY_PATH for Yggdrasill"
export PYTHONPATH=$yggdrasill_path:$PYTHONPATH
export LD_LIBRARY_PATH=$yggpath/lib:$LD_LIBRARY_PATH