from ash import *

#Define coordinate string
coords="""
O       -1.377626260      0.000000000     -1.740199718
H       -1.377626260      0.759337000     -1.144156718
H       -1.377626260     -0.759337000     -1.144156718
"""
#Defining fragment
H2Ofragment=Fragment(coordsstring=coords)

#Defining dummy theory
zerotheorycalc = ZeroTheory()

#Optimize with dummy theory
result = geomeTRICOptimizer(fragment=H2Ofragment, theory=zerotheorycalc)

if result == 0.0:
    print("ASH and geometric finished. Everything looks good")