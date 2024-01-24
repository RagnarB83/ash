from ash import *

#Plotting DMRG DM values as function of M states,  and orbital choices
M_values=[50,100,200,300,400,500,750,1000,1500]
#
MP2_unrelaxdens_natorbs=[8.57275541e-02, 8.73755446e-02,8.76275308e-02, 8.82905555e-02, 8.85406407e-02, 8.86419056e-02, 8.87365794e-02, 8.87894943e-02, 8.88254739e-02]
CCSD_unrelaxdens_natorbs=[8.84190212e-02,8.73351753e-02,8.89834789e-02,8.85887126e-02,8.86162553e-02,8.86416353e-02,8.87773534e-02,8.88128262e-02,8.88332503e-02]
CCSD_T_unrelaxdens_natorbs=[8.93856104e-02,8.80656901e-02,8.85246505e-02,8.85726937e-02,8.86701557e-02,8.87547739e-02,8.87890603e-02,8.88051162e-02,8.88308582e-02]

print(len(M_values))
print(len(MP2_unrelaxdens_natorbs))
print(len(CCSD_unrelaxdens_natorbs))
print(len(CCSD_T_unrelaxdens_natorbs))
#Note: using inverse logarithmic x-scale
dplot = ASH_plot("DMRG DMs for different natorbs", num_subplots=1, x_axislabel="DMRG-M-value", y_axislabel='Dipole-mom (A.U.)', )
dplot.addseries(0, x_list=M_values, y_list=MP2_unrelaxdens_natorbs, x_scale_log=False,
    label='MP2_unrelaxdens_natorbs', color='blue', line=True, scatter=True)
dplot.addseries(0, x_list=M_values, y_list=CCSD_unrelaxdens_natorbs, x_scale_log=False,
    label='CCSD_unrelaxdens_natorbs', color='green', line=True, scatter=True)
dplot.addseries(0, x_list=M_values, y_list=CCSD_T_unrelaxdens_natorbs, x_scale_log=False,
    label='CCSD_T_unrelaxdens_natorbs', color='red', line=True, scatter=True)
#Save figure
dplot.savefig('Dipole_DMRG',imageformat='png',dpi=300)
