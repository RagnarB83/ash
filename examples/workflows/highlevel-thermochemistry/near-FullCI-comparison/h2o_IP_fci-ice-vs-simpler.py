from ash import *

#Script to calculate a small molecule reaction energy at the near-FullCI limit at a fixed basis set
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
reaction = Reaction(fragments=[h2o_n, h2o_o], stoichiometry=[-1,1])
reaction_energy_unit='eV'
reaction_label='H2O_IP'

#Plotting options
y_axis_label='IP'
# To specify the y-axis limits either define ylimits like this: ylimits=[11.5,12.0] or use yshift. 
# ylimits=[11.5,12.0] #Values of y-axis
# yshift will define y-axis limits based on last ICE-CI energy
yshift=0.3 #Shift (in reaction_energy_unit) in + and - direction of the last ICE-CI energy calculated 

#Basis set to use
basis = "cc-pVDZ"
#ORCA maxcore setting in MB
maxcorememory = 11000
#What single-reference methods to do
DoHF = True
DoMP2 = True
DoCC = True

#What Tgen thresholds to calculate in ICE-CI?
tgen_thresholds=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
#ICE thresholds for selecting active space 
ice_nmin = 1.999 #Good value for getting O 1s frozen-core for H2O
ice_nmax = 0 #All virtual orbitals

####################################################################################

#Looping over TGen thresholds in ICE-CI
results_ice = {}
results_ice_genCFGs={}
results_ice_selCFGs={}
results_ice_SDCFGs={}
for tgen in tgen_thresholds:
    print("="*100)
    print(f"Now doing tgen: {tgen}")
    input=f"! Auto-ICE {basis} tightscf"
    #Setting ICE-CI so that frozen-core is applied (1s oxygen frozen). Note: Auto-ICE also required.
    blocks=f"""
    %maxcore {maxcorememory}
    %ice
    nmin {ice_nmin}
    nmax {ice_nmax}
    tgen {tgen}
    useMP2nat true
    end
    """
    ice = ORCATheory(orcasimpleinput=input, orcablocks=blocks, numcores=numcores, label=f'ICE_{tgen}_', save_output_with_label=True)
    rel_energy_ICE = Singlepoint_reaction(reaction=reaction, theory=ice, unit=reaction_energy_unit)
    num_genCFGs,num_selected_CFGs,num_after_SD_CFGs = ICE_WF_CFG_CI_size(ice.filename+'_last.out')
    #Keeping in dict
    results_ice[tgen] = rel_energy_ICE
    results_ice_genCFGs[tgen] = num_genCFGs
    results_ice_selCFGs[tgen] = num_selected_CFGs
    results_ice_SDCFGs[tgen] = num_after_SD_CFGs



#Running regular single-reference WF methods
results_cc={}
blocks=f"""
%maxcore 11000
%mdci
maxiter	300
end
"""
brucknerblocks=f"""
%maxcore 11000
%mdci
maxiter 300
Brueckner true
end
"""

if DoHF is True:
    hf = ORCATheory(orcasimpleinput=f"! HF {basis} tightscf", orcablocks=blocks, numcores=4, label='HF', save_output_with_label=True)
    relE_HF = Singlepoint_reaction(reaction=reaction, theory=hf, unit=reaction_energy_unit)
    results_cc['HF'] = relE_HF
if DoMP2 is True:
    mp2 = ORCATheory(orcasimpleinput=f"! MP2 {basis} tightscf", orcablocks=blocks, numcores=4, label='MP2', save_output_with_label=True)
    scsmp2 = ORCATheory(orcasimpleinput=f"! SCS-MP2 {basis} tightscf", orcablocks=blocks, numcores=4, label='SCSMP2', save_output_with_label=True)
    oomp2 = ORCATheory(orcasimpleinput=f"! OO-RI-MP2 autoaux {basis} tightscf", orcablocks=blocks, numcores=4, label='OOMP2', save_output_with_label=True)
    scsoomp2 = ORCATheory(orcasimpleinput=f"! OO-RI-SCS-MP2 {basis} autoaux tightscf", orcablocks=blocks, numcores=4, label='OOSCSMP2', save_output_with_label=True)

    relE_MP2 = Singlepoint_reaction(reaction=reaction, theory=mp2, unit=reaction_energy_unit)
    relE_SCSMP2 = Singlepoint_reaction(reaction=reaction, theory=scsmp2, unit=reaction_energy_unit)
    relE_OOMP2 = Singlepoint_reaction(reaction=reaction, theory=oomp2, unit=reaction_energy_unit)
    relE_SCSOOMP2 = Singlepoint_reaction(reaction=reaction, theory=scsoomp2, unit=reaction_energy_unit)

    results_cc['MP2'] = relE_MP2
    results_cc['SCS-MP2'] = relE_SCSMP2
    results_cc['OO-MP2'] = relE_OOMP2
    results_cc['OO-SCS-MP2'] = relE_SCSOOMP2

if DoCC is True:
    ccsd = ORCATheory(orcasimpleinput=f"! CCSD {basis} tightscf", orcablocks=blocks, numcores=4, label='CCSD', save_output_with_label=True)
    bccd = ORCATheory(orcasimpleinput=f"! CCSD {basis} tightscf", orcablocks=brucknerblocks, numcores=4, label='BCCD', save_output_with_label=True)
    ooccd = ORCATheory(orcasimpleinput=f"! OOCCD {basis} tightscf", orcablocks=blocks, numcores=4, label='OOCCD', save_output_with_label=True)
    pccsd_1a = ORCATheory(orcasimpleinput=f"! pCCSD/1a {basis} tightscf", orcablocks=blocks, numcores=4, label='pCCSD1a', save_output_with_label=True)
    pccsd_2a = ORCATheory(orcasimpleinput=f"! pCCSD/2a {basis} tightscf", orcablocks=blocks, numcores=4, label='pCCSD2a', save_output_with_label=True)
    ccsdt = ORCATheory(orcasimpleinput=f"! CCSD(T) {basis} tightscf", orcablocks=blocks, numcores=4, label='CCSDT', save_output_with_label=True)
    ccsdt_qro = ORCATheory(orcasimpleinput=f"! CCSD(T) {basis} UNO tightscf", orcablocks=blocks, numcores=4, label='CCSDT_QRO', save_output_with_label=True)
    ooccd = ORCATheory(orcasimpleinput=f"! OOCCD {basis} tightscf", orcablocks=blocks, numcores=4, label='OOCCD', save_output_with_label=True)
    ooccdt = ORCATheory(orcasimpleinput=f"! OOCCD(T) {basis} tightscf", orcablocks=blocks, numcores=4, label='OOCCDT', save_output_with_label=True)
    bccdt = ORCATheory(orcasimpleinput=f"! CCSD(T) {basis} tightscf", orcablocks=brucknerblocks, numcores=4, label='BCCDT', save_output_with_label=True)
    ccsdt_bp = ORCATheory(orcasimpleinput=f"! CCSD(T) BP86 {basis} tightscf", orcablocks=blocks, numcores=4, label='CCSDT_BP', save_output_with_label=True)

    relE_CCSD = Singlepoint_reaction(reaction=reaction, theory=ccsd, unit=reaction_energy_unit)
    relE_OOCCD = Singlepoint_reaction(reaction=reaction, theory=ooccd, unit=reaction_energy_unit)
    relE_BCCD = Singlepoint_reaction(reaction=reaction, theory=bccd, unit=reaction_energy_unit)
    relE_pCCSD1a = Singlepoint_reaction(reaction=reaction, theory=pccsd_1a, unit=reaction_energy_unit)
    relE_pCCSD2a = Singlepoint_reaction(reaction=reaction, theory=pccsd_2a, unit=reaction_energy_unit)
    relE_CCSDT = Singlepoint_reaction(reaction=reaction, theory=ccsdt, unit=reaction_energy_unit)
    relE_CCSDT_QRO = Singlepoint_reaction(reaction=reaction, theory=ccsdt_qro, unit=reaction_energy_unit)
    relE_OOCCDT = Singlepoint_reaction(reaction=reaction, theory=ooccdt, unit=reaction_energy_unit)
    relE_BCCDT = Singlepoint_reaction(reaction=reaction, theory=bccdt, unit=reaction_energy_unit)
    relE_CCSDT_BP = Singlepoint_reaction(reaction=reaction, theory=ccsdt_bp, unit=reaction_energy_unit)

    results_cc['CCSD'] = relE_CCSD
    results_cc['BCCD'] = relE_BCCD
    results_cc['OOCCD'] = relE_OOCCD
    results_cc['pCCSD/1a'] = relE_pCCSD1a
    results_cc['pCCSD/2a'] = relE_pCCSD2a
    results_cc['CCSD(T)'] = relE_CCSDT
    results_cc['CCSD(T)-QRO'] = relE_CCSDT_QRO
    results_cc['OOCCD(T)'] = relE_OOCCDT
    results_cc['BCCD(T)'] = relE_BCCDT
    results_cc['CCSD(T)-BP'] = relE_CCSDT_BP


##########################################
#Printing final results
##########################################
#Create ASH_plot object named edplot

#y-limits based on last ICE calculation rel energy
if 'ylimits' in locals():
    print(f"Using y-limits: {ylimits} {reaction_energy_unit} in plot")
else:
    ylimits = [rel_energy_ICE+yshift,rel_energy_ICE-yshift]
    print(f"Using y-limits: {ylimits} {reaction_energy_unit} in plot")

eplot = ASH_plot("Plotname", num_subplots=2, x_axislabels=["TGen", "Method"], y_axislabels=[f'{y_axis_label} ({reaction_energy_unit})',f'{y_axis_label} ({reaction_energy_unit})'], subplot_titles=["ICE-CI","Single ref. methods"],
    ylimit=ylimits, horizontal=True)
xvals=[];yvals=[]
x2vals=[];y2vals=[];labels=[]

print()
print()
print("ICE-CI CIPSI wavefunction")
print(f" Tgen      Energy ({reaction_energy_unit})        # gen. CFGs       # sel. CFGs     # max S+D CFGs")
print("---------------------------------------------------------------------------------")
for t, e in results_ice.items():
    gen_cfg=results_ice_genCFGs[t]
    sel_cg=results_ice_selCFGs[t]
    sd_cfg=results_ice_SDCFGs[t]
    print("{:<10.2e} {:13.10f} {:15} {:15} {:15}".format(t,e,gen_cfg,sel_cg, sd_cfg))
    #Add data to lists for plotting
    xvals.append(t)
    yvals.append(e)
print()
print()
print("Other methods:")
print(f" WF   Energy ({reaction_energy_unit})")
print("----------------------------")
for i,(w, e) in enumerate(results_cc.items()):
    print("{:<10} {:13.10f}".format(w,e))
    x2vals.append(i)
    y2vals.append(e)
    labels.append(w)

#Add dataseries to subplot 0
#Inverting x-axis and using log-scale for ICE-CI data
eplot.addseries(0, x_list=xvals, y_list=yvals, label=reaction_label, color='blue', line=True, scatter=True, x_scale_log=True, invert_x_axis=True)
#Plotting method labels on x-axis with rotation to make things fit
eplot.addseries(1, x_list=x2vals, y_list=y2vals, x_labels=labels, label=reaction_label, color='red', line=True, scatter=True, xticklabelrotation=80)

#Save figure
eplot.savefig(f'{reaction_label}_FCI')