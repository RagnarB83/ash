"""
Conversion factors
"""
# hartree conversions
hartokcal = 627.5094740631
harkcal = 627.5094740631
hartocm = 219474.6313702
# hartokj = 2625.5002
hartokj = 2625.4996394799
hartree2j = 4.3597438e-18
hartoeV = 27.211386245988

# speed of light in cm/s
c = 2.99792458e10

# GHz to cm-1
GHztocm = 0.0333564095198152
# 0.5*h*c: 0.5 * 6.62607015E-34 Js*2.293712317E+17hartree/J*29979245800cm/s
halfhcfactor = 2.27816766479806E-06

# Planck's constant J*s
h_planck = 6.62607015E-34
# in Eh*s
h_planck_hartreeseconds = 1.5198298716361000E-16

#hc  (J cm)
hc = h_planck*2.99792458e10

# Distance conversions
bohr2ang = 0.52917721067
ang2bohr = 1.88972612546
bohr2m = 5.29177210903e-11
amu2kg = 1.66053906660e-27
ang2m = 1e-10

# R in hartree/K. Converted from 8.31446261815324000 J/Kmol
R_gasconst = 3.16681161675373E-06
R_gasconst_JK = 8.31446261815324 #J/mol/K
R_gasconst_kcalK = 1.987191683e-3 #kcal/mol/K
k_b_JK = 1.380649e-23
k_b_cmK = 0.695034800 #cm-1/K (k/(hc))

#From OpenMM
BOLTZMANN=1.380649e-23 # J/K
AVOGADRO=6.02214076e23
RGAS=BOLTZMANN*AVOGADRO # J/(mol K)
KILO=1e3 # grams
BOLTZ=RGAS/KILO # kJ/(mol K)

pi = 3.14159265359

# Electrochemistry
SHE = 4.28
