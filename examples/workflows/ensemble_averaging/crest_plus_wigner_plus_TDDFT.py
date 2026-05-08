from ash import *
from ash.functions.functions_elstructure import boltzmann_populations
from ash.interfaces.interface_ORCA import tddftgrab, tddftintens_grab

numcores=10

#########
#System
#########
charge=0; mult=1
frag = Fragment(databasefile="h2o.xyz", charge=charge,mult=mult)
temperature=298.15
num_wigner_samples=10

#############
# Theories
#############
ll_theory = xTBTheory(xtbmethod="GFN2")
hl_theory = ORCATheory(orcasimpleinput="! CAM-B3LYP 6-311++G(d,p) CPCM(DMSO) tightscf", numcores=numcores)
#hl_theory = ORCATheory(orcasimpleinput="! hf-3c")
# Spectroscopy theory
blocks="""%tddft
nroots 10
tda false
end
"""
tddft_theory = ORCATheory(orcasimpleinput="! CAM-B3LYP 6-311++G(d,p) CPCM(DMSO) tightscf",orcablocks=blocks, numcores=numcores)

#############################
# 1. Conformational sampling
#############################
#new_call_crest(fragment=frag, theory=ll_theory, runtype="imtd-gc", numcores=numcores)
call_crest(fragment=frag, xtbmethod='GFN2-xTB', charge=charge, mult=mult, energywindow=6, numcores=numcores)

#Get xtB conformers as fragments
frags = get_molecules_from_trajectory("crest_conformers.xyz")
# Set charge/mult for each conformer
for frag in frags: frag.charge = charge; frag.mult=mult

##########################################
# 2. Opt+Freq (for Boltzmann populations)
##########################################
G_frags=[]
# Loop over frag and do Opt+Freq
for frag in frags:
    FinalE, componentsdict, thermochem = thermochemprotocol_single(fragment=frag, Opt_theory=hl_theory, numcores=numcores, memory=5000,
                   analyticHessian=True, temp=temperature, pressure=1.0)
    G=FinalE+thermochem["Gcorr"]
    G_frags.append(G)
boltzmann_weights = boltzmann_populations(G_frags, temperature=temperature)
print("boltzmann_weights:", boltzmann_weights)

##########################################
# 3. Wigner distributed TDDFT
##########################################
all_trans_energies=[]
all_trans_intensities=[]
# Loop over conformer fragments
for i,frag in enumerate(frags):
    # Wigner distributions
    wigner_frags = wigner_distribution(fragment=frag, hessian=frag.hessian, temperature=temperature, num_samples=num_wigner_samples)
    # TDDFT on each Wigner fragment
    conf_trans_energies=[]
    conf_trans_intensities=[]
    for wfrag in wigner_frags:
        Singlepoint(theory=tddft_theory, fragment=wfrag)
        # Get transition energies and intensities
        transition_energies = tddftgrab(f"{tddft_theory.filename}.out")
        transition_intensitites = tddftintens_grab(f"{tddft_theory.filename}.out")
        # Weight intensities by Boltzmann weight for that conformer
        transition_intensitites= np.array(transition_intensitites) * boltzmann_weights[i]
        # Conformer-specific E and intens
        conf_trans_energies += transition_energies
        conf_trans_intensities += [float(transi) for transi in transition_intensitites]
        # All
        all_trans_energies += transition_energies
        all_trans_intensities += [float(transi) for transi in transition_intensitites]

    # Plot spectrum (applies broadening to every stick)
    plot_Spectrum(xvalues=conf_trans_energies, yvalues=conf_trans_intensities, plotname=f'TDDFT_conf{i}',
        range=[0,10], unit='eV', broadening=0.075, points=10000, imageformat='png', dpi=200, matplotlib=True,
        CSV=True, color='blue', plot_sticks=True)

# Plot spectrum (applies broadening to every stick)
plot_Spectrum(xvalues=all_trans_energies, yvalues=all_trans_intensities, plotname='TDDFT_final',
    range=[0,10], unit='eV', broadening=0.075, points=10000, imageformat='png', dpi=200, matplotlib=True,
    CSV=True, color='blue', plot_sticks=True)