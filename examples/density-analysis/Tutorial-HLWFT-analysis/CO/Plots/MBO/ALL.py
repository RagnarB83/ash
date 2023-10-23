from ash import *

#Mayer bond order
MBO_dict={
'RHF':2.50769225,
'MP2-unrelax':2.33010456,
'MP2-relax':2.44210934,
'CCSD-lin':2.43249387,
'CCSD-unrelax':2.41951329,
'CCSD-orbopt':2.40512503,
'CCSD(T)-unrelax':2.36829120,
'CCSDT-relax':2.36746655,
'CCSDTQ-relax':2.36293149,
'SHCI_CCSD_T_natorbs_noPT_eps_1e-05.chg':2.36346180,
'ICE_CI_mp2nat_tgen_1e-06':2.36653032}



#Gettings lists from dict
method_labels= list(MBO_dict.keys())
charges= list(MBO_dict.values())

dplot = ASH_plot("MBO C-O", num_subplots=1, x_axislabel="", y_axislabel='Mayer bond order', )
dplot.addseries(0, x_list=list(range(0,len(charges))), x_labels=method_labels, y_list=charges, legend=False,
    label='MBO', color='blue', line=False, scatter=False, bar=True, colormap='viridis')

#Save figure
dplot.savefig('MBO',imageformat='png',dpi=300)
