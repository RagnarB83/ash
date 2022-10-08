from ash import *

#Script to calculate a small molecule reaction energy at the near-FullCI limit at a fixed basis set
#with comparison to simpler methods
#QM code: ORCA
#Near-FCI method: ICE-CI
#Basis set: cc-pVDZ
#Molecule: H2O
#Property: VIP

numcores = 16

#Defining reaction: Vertical IP of H2O 
h2o_n = Fragment(xyzfile="h2o.xyz", charge=0, mult=1)
h2o_o = Fragment(xyzfile="h2o.xyz", charge=1, mult=2)
IE = Reaction(fragments=[h2o_n, h2o_o], stoichiometry=[-1,1])

#Looping over TGen thresholds in ICE-CI
results_ice = {}
results_ice_genCFGs={}
results_ice_selCFGs={}
results_ice_SDCFGs={}
for tgen in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
    print("="*100)
    print(f"Now doing tgen: {tgen}")
    input="! Auto-ICE cc-pVDZ tightscf"
    #Setting ICE-CI so that frozen-core is applied (1s oxygen frozen). Note: Auto-ICE also required.
    blocks=f"""
    %maxcore 11000
    %ice
    nmin 1.999
    nmax 0
    tgen {tgen}
    useMP2nat true
    end
    """
    ice = ORCATheory(orcasimpleinput=input, orcablocks=blocks, numcores=numcores)
    IP_energy = Singlepoint_reaction(reaction=IE, theory=ice, unit='eV')
    #Grabbing wavefunction compositions
    num_genCFGs,num_selected_CFGs,num_after_SD_CFGs = ash.interfaces.interface_ORCA.ICE_WF_CFG_CI_size(ice.filename+'_last.out')
    #Keeping in dict
    results_ice[tgen] = IP_energy
    results_ice_genCFGs[tgen] = num_genCFGs
    results_ice_selCFGs[tgen] = num_selected_CFGs
    results_ice_SDCFGs[tgen] = num_after_SD_CFGs



#Running regular single-reference WF methods
results_cc={}
blocks=f"""
%maxcore 11000
"""
brucknerblocks=f"""
%maxcore 11000
%mdci
Brueckner true
end
"""
hf = ORCATheory(orcasimpleinput="! HF cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
mp2 = ORCATheory(orcasimpleinput="! MP2 cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
scsmp2 = ORCATheory(orcasimpleinput="! SCS-MP2 cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
oomp2 = ORCATheory(orcasimpleinput="! OO-RI-MP2 autoaux cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
scsoomp2 = ORCATheory(orcasimpleinput="! OO-RI-SCS-MP2 cc-pVDZ autoaux tightscf", orcablocks=blocks, numcores=4)

ccsd = ORCATheory(orcasimpleinput="! CCSD cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
bccd = ORCATheory(orcasimpleinput="! CCSD cc-pVDZ tightscf", orcablocks=brucknerblocks, numcores=4)
ooccd = ORCATheory(orcasimpleinput="! OOCCD cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
pccsd_1a = ORCATheory(orcasimpleinput="! pCCSD/1a cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
pccsd_2a = ORCATheory(orcasimpleinput="! pCCSD/2a cc-pVDZ tightscf", orcablocks=blocks, numcores=4)

ccsdt = ORCATheory(orcasimpleinput="! CCSD(T) cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
ooccd = ORCATheory(orcasimpleinput="! OOCCD cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
ooccdt = ORCATheory(orcasimpleinput="! OOCCD(T) cc-pVDZ tightscf", orcablocks=blocks, numcores=4)
bccdt = ORCATheory(orcasimpleinput="! CCSD(T) cc-pVDZ tightscf", orcablocks=brucknerblocks, numcores=4)
ccsdt_bp = ORCATheory(orcasimpleinput="! CCSD(T) BP86 cc-pVDZ tightscf", orcablocks=blocks, numcores=4)

IP_HF = Singlepoint_reaction(reaction=IE, theory=hf, unit='eV')
IP_MP2 = Singlepoint_reaction(reaction=IE, theory=mp2, unit='eV')
IP_SCSMP2 = Singlepoint_reaction(reaction=IE, theory=scsmp2, unit='eV')
IP_OOMP2 = Singlepoint_reaction(reaction=IE, theory=oomp2, unit='eV')
IP_SCSOOMP2 = Singlepoint_reaction(reaction=IE, theory=scsoomp2, unit='eV')

IP_CCSD = Singlepoint_reaction(reaction=IE, theory=ccsd, unit='eV')
IP_OOCCD = Singlepoint_reaction(reaction=IE, theory=ooccd, unit='eV')
IP_BCCD = Singlepoint_reaction(reaction=IE, theory=bccd, unit='eV')
IP_pCCSD1a = Singlepoint_reaction(reaction=IE, theory=pccsd_1a, unit='eV')
IP_pCCSD2a = Singlepoint_reaction(reaction=IE, theory=pccsd_2a, unit='eV')

IP_CCSDT = Singlepoint_reaction(reaction=IE, theory=ccsdt, unit='eV')
IP_OOCCDT = Singlepoint_reaction(reaction=IE, theory=ooccdt, unit='eV')
IP_BCCDT = Singlepoint_reaction(reaction=IE, theory=bccdt, unit='eV')
IP_CCSDT_BP = Singlepoint_reaction(reaction=IE, theory=ccsdt_bp, unit='eV')

results_cc['HF'] = IP_HF
results_cc['MP2'] = IP_MP2
results_cc['SCS-MP2'] = IP_SCSMP2
results_cc['OO-MP2'] = IP_OOMP2
results_cc['OO-SCS-MP2'] = IP_SCSOOMP2
results_cc['CCSD'] = IP_CCSD
results_cc['BCCD'] = IP_BCCD
results_cc['pCCSD/1a'] = IP_pCCSD1a
results_cc['pCCSD/2a'] = IP_pCCSD2a
results_cc['OOCCD'] = IP_OOCCD

results_cc['CCSD(T)'] = IP_CCSDT
results_cc['OOCCD(T)'] = IP_OOCCDT
results_cc['BCCD(T)'] = IP_BCCDT
results_cc['CCSD(T)-BP'] = IP_CCSDT_BP


##########################################
#Printing and plotting final results
##########################################
#Create ASH_plot object named edplot
eplot = ASH_plot("Plotname", num_subplots=2, x_axislabels=["TGen", "Method"], y_axislabels=['IP(eV)','IP(eV)'], subplot_titles=["ICE-CI TGen calc.","WFs"],
    ylimit=[10.5,12.0], horizontal=True)
xvals=[];yvals=[]
x2vals=[];y2vals=[];labels=[]

print()
print()
print("ICE-CI CIPSI wavefunction")
print(" Tgen      Energy (eV)        # gen. CFGs       # sel. CFGs     # max S+D CFGs")
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
print(" WF   Energy (eV)")
print("----------------------------")
for i,(w, e) in enumerate(results_cc.items()):
    print(f"i:{i} w:{w} e:{e}")
    print("{:<10} {:13.10f}".format(w,e))
    #Add a dataseries to subplot 0 (the only subplot)
    x2vals.append(i)
    y2vals.append(e)
    labels.append(w)

print("x2vals:", x2vals)
print("y2vals:", y2vals)
print("labels:", labels)

#Add a dataseries to subplot 0 (the only subplot)
eplot.addseries(0, x_list=xvals, y_list=yvals, label='IP', color='blue', line=True, scatter=True)
eplot.addseries(1, x_list=x2vals, y_list=y2vals, x_labels=labels, label='IP', color='red', line=True, scatter=True)

#Save figure
eplot.savefig('H2O_IP_WF')