from ash import *
import sys

frag=Fragment(xyzfile="/Users/bjornsson/ownCloud/ASH-tests/testsuite/phenoldonor-r1.8-withH2O.xyz")

qmatoms=[0,1,2,3,4,5,6,7,8,9,10,11,12]
atomcharges=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.834, 0.417, 0.417]

xtbpath="/opt/xtb-6.4.0/bin"
xtbcalc=xTBTheory(xtbmethod="GFN2", charge=0, mult=1, xtbdir=xtbpath)
hybrid=QMMMTheory(fragment=frag, qm_theory=xtbcalc,  embedding='Elstat', qmatoms=qmatoms, charges=atomcharges)

result=Singlepoint(fragment=frag, theory=hybrid)
xtbcalc.cleanup()



reference=-20.0354357
print("Reference energy:", reference)
print("Actual energy:", result.energy)
print("Deviation:", result.energy-reference)
if result.energy-reference >0.000001:
    print("FAIL. Bad deviation between xtB/MM and reference. Did xTB version change??")
    sys.exit(1)
else:
    print("Success!")
    sys.exit(0)

# REFERENCE value: -20.0354357, 
# xtb version 6.1, 6.2.3 and 6.4.0 are correct
# xtb version 6.3 etc. are bad
