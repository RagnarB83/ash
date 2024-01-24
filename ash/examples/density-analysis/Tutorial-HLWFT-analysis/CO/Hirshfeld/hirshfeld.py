from ash import *

moldenfiles=glob.glob("*molden*")

for moldenfile in moldenfiles:
    multiwfn_run(moldenfile, option='hirshfeld', grid=3, numcores=1)
