from ash import *

#Function to calculate a small molecule reaction energy at the near-FullCI limit at a fixed basis set
#with comparison to simpler methods
#QM code: ORCA
#Near-FCI method: ICE-CI
#Basis set: cc-pVDZ
#Molecule: H2O
#Property: VIP

numcores = 16

####################################################################################
#Defining reaction: Vertical IP of H2O
h2o_n = Fragment(xyzfile="h2o.xyz", charge=0, mult=1)
h2o_o = Fragment(xyzfile="h2o.xyz", charge=1, mult=2)
reaction = Reaction(fragments=[h2o_n, h2o_o], stoichiometry=[-1,1], label='H2O_IP', unit='eV')

#What Tgen thresholds to calculate in ICE-CI?
tgen_thresholds=[1e-1,5e-2,1e-2,5e-3, 1e-3,5e-4]

Reaction_FCI_Analysis(reaction=reaction, basis="cc-pVDZ", tgen_thresholds=tgen_thresholds, ice_nmin=1.999, ice_nmax=0,
                DoHF=True, DoMP2=True, DoCC=True, maxcorememory=10000, numcores=1,
                plot=True, y_axis_label='IP', yshift=0.3)