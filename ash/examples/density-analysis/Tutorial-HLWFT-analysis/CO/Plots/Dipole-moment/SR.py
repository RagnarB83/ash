from ash import *

#Dipole Z-value taken from ASH output or individual QM-program outputs
dipole_dict={
'RHF':-0.09421,
'ORCA_MP2-unrelax':-0.03678,
'ORCA_MP2-relax':0.14881,
'ORCA_OOMP2':0.15152,
'ORCA_CCSD-lin':0.15610,
'ORCA_CCSD-unrelax':0.09793,
'ORCA_CCSD-orbopt':0.06531,
'pySCF-CCSD-unrelax':0.09794,
'pySCF-BCCD-unrelax':0.09066,
'pySCF-CCSD(T)-unrelax':0.08498,
'pySCF-BCCD(T)-unrelax':0.08578,
'CFOUR-CCSD-relax':0.0662791648,
'CFOUR-CCSD(T)-relax':0.0854236795,
'MRCC_CCSD_unrelax':0.09793883,
'MRCC_CCSD_relax':0.06627899,
'MRCC_CCSDT_unrelax':0.09148115,
'MRCC_CCSDT_relax':0.08444705,
'MRCC_CCSDTQ_unrelax':0.08893355,
'MRCC_CCSDTQ_relax':0.08786930}
#Gettings lists from dict
method_labels= list(dipole_dict.keys())
dipole_moments= list(dipole_dict.values())
#Note: using Z-component

dplot = ASH_plot("Dipole moment", num_subplots=1, x_axislabel="", y_axislabel='Dipole-mom (A.U.)', )
dplot.addseries(0, x_list=list(range(0,len(dipole_moments))), x_labels=method_labels, y_list=dipole_moments,
    label='Dipole', color='blue', line=False, scatter=False, bar=True, colormap='viridis')

#Save figure
dplot.savefig('DM_SR',imageformat='png',dpi=300)
