from ash import *

#Dipole Z-value taken from ASH output or individual QM-program outputs
dipole_dict={
'MRCIQ_10_8_tsel0':0.090531604349123,
'MRCC_CCSDTQ_relax':0.08786930,
'ICE-CI_CCSDrelaxNat_tgen1e-6':0.08875,
'SHCI-CCSD_T_nat_noPT_eps_1e-5':0.08867,
'DMRG-CCSD_T_nat_M_2000':8.88415320e-02}

#Gettings lists from dict
method_labels= list(dipole_dict.keys())
dipole_moments= list(dipole_dict.values())
#Note: using Z-component

dplot = ASH_plot("Dipole moment", num_subplots=1, x_axislabel="", y_axislabel='Dipole-mom (A.U.)', )
dplot.addseries(0, x_list=list(range(0,len(dipole_moments))), x_labels=method_labels, y_list=dipole_moments,
    label='Dipole', color='blue', line=True, scatter=True, bar=False, colormap='viridis')

#Save figure
dplot.savefig('Dipole-moment-near-FCI',imageformat='png',dpi=300)
