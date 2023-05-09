from ash import *

frag = Fragment(databasefile="butane.xyz", charge=0, mult=1)

theory = xTBTheory(runmode='library')

surfacedictionary = calc_surface(fragment=frag, theory=theory, scantype='Relaxed', 
    resultfile='surface_results.txt', runmode='serial', 
    RC1_range=[-180,180,10], RC1_type='dihedral', RC1_indices=[0,1,2,3])
