#Simple script to rename xyz files from GMTKN55 subset archive
#dir=~/ASH/ash-dev/ash/databases/Benchmarking-sets/GMTKN55/MB16-43/data/bla
#TODO: read also charge/mult info from file and add to XYZ header

for i in $(ls $dir/)
do
cp $dir/$i/struc.xyz ./$i.xyz
done
