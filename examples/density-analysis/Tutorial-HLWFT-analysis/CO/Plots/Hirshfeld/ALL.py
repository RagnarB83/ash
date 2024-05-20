from ash import *

#Hirshfeld charge on C
hirshfeld_dict={
'RHF':0.1412859997,
'MP2-unrelax':0.1169798965,
'MP2-relax':0.0527007827,
'CCSD-lin':0.0471348620,
'CCSD-unrelax':0.0666239883,
'CCSD-orbopt':0.0777477981,
'CCSD(T)-unrelax':0.0711719313,
'CCSDT-relax':0.0683192187,
'CCSDTQ-relax':0.0689681631,
'SHCI_CCSD_T_natorbs_noPT_eps_1e-05.chg':0.0691342776,
'ICE_CI_mp2nat_tgen_1e-06':0.0722440864}



#Gettings lists from dict
method_labels= list(hirshfeld_dict.keys())
charges= list(hirshfeld_dict.values())

dplot = ASH_plot("Hirshfeld charge on C", num_subplots=1, x_axislabel="", y_axislabel='Hirshfeld charge (el)', )
dplot.addseries(0, x_list=list(range(0,len(charges))), x_labels=method_labels, y_list=charges, legend=False,
    label='Hirshfeld', color='blue', line=False, scatter=False, bar=True, colormap='viridis')

#Save figure
dplot.savefig('Hirshfeld',imageformat='png',dpi=300)
