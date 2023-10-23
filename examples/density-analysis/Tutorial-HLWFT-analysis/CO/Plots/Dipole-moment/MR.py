from ash import *

#Dipole Z-value taken from ASH output or individual QM-program outputs
dipole_dict={
'RHF':-0.09421,
'CASSCF_2_2':-0.13942,
'CASSCF_6_5':0.10798,
'CASSCF_10_8':0.14510,
'MRCIQ_2_2_tseldef':0.05980885008541,
'MRCIQ_2_2_tsel0':0.057475824950517,
'MRCIQ_6_5_tseldef':0.093415427998156,
'MRCIQ_6_5_tsel0':0.089095593937258,
'MRCIQ_10_8_tseldef':0.094603579078658,
'MRCIQ_10_8_tsel0':0.090531604349123}

#Gettings lists from dict
method_labels= list(dipole_dict.keys())
dipole_moments= list(dipole_dict.values())
#Note: using Z-component

dplot = ASH_plot("Dipole moment", num_subplots=1, x_axislabel="", y_axislabel='Dipole-mom (A.U.)', )
dplot.addseries(0, x_list=list(range(0,len(dipole_moments))), x_labels=method_labels, y_list=dipole_moments,
    label='Dipole', color='blue', line=False, scatter=False, bar=True, colormap='viridis')

#Save figure
dplot.savefig('DM-MR',imageformat='png',dpi=300)
