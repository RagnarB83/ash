from ash import *

#Dipole Z-value taken from ASH output or individual QM-program outputs
dipole_dict={
'RHF':-0.09421,
'CASSCF_2_2':-0.13942,
'CASSCF_6_5':0.10798,
'CASSCF_10_8':0.14510,
'MRCIQ_2_2_tseldef':0.05980885008541,
'MRCIQ_2_2_tsel0':0.057475824950517,
'MRCIQ_10_8_tseldef':0.094603579078658,
'MRCIQ_10_8_tsel0':0.090531604349123,
'ORCA_MP2-unrelax':-0.03678,
'ORCA_MP2-relax':0.14881,
'ORCA_OOMP2':0.15152,
'pySCF-CCSD-unrelax':0.09794,
'ORCA_CCSD-lin':0.15610,
'ORCA_CCSD-unrelax':0.09793,
'ORCA_CCSD-orbopt':0.06531,
'MRCC_CCSD_unrelax':0.09793883,
'MRCC_CCSD_relax':0.06627899,
'CFOUR-CCSD-relax':0.0662791648,
'CFOUR-CCSD(T)-relax':0.0854236795,
'pySCF-BCCD-unrelax':0.09066,
'pySCF-BCCD(T)-unrelax':0.08578,
'pySCF-CCSD(T)-unrelax':0.08498,
'MRCC_CCSDT_unrelax':0.09148115,
'MRCC_CCSDT_relax':0.08444705,
'MRCC_CCSDTQ_unrelax':0.08893355,
'MRCC_CCSDTQ_relax':0.08786930,
'SHCI_eps4e-5_PT':0.08737,
'SHCI_eps6e-5_noPT':0.08863,
'ICE-CI_MP2unrelaxNat_tgen1e-6':0.08879,
'ICE-CI_MP2RelaxNat_tgen1e-6':0.08898}

#Gettings lists from dict
method_labels= list(dipole_dict.keys())
dipole_moments= list(dipole_dict.values())
#Note: using Z-component

dplot = ASH_plot("Dipole moment", num_subplots=1, x_axislabel="", y_axislabel='Dipole-mom (A.U.)', )
dplot.addseries(0, x_list=list(range(0,len(dipole_moments))), x_labels=method_labels, y_list=dipole_moments,
    label='Dipole', color='blue', line=False, scatter=False, bar=True, colormap='viridis')

#Save figure
dplot.savefig('Dipole-moment-ALL',imageformat='png',dpi=300)
