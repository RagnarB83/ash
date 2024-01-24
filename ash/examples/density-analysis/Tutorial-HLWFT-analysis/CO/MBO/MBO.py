from ash import *

moldenfiles=glob.glob("*molden*")

for moldenfile in moldenfiles:
    multiwfn_run(moldenfile, option='MBO', grid=3, numcores=1)
    os.rename("bndmat.txt", f"{moldenfile}_bndmat.txt")
