from ash import *

#Plotting SHCI DM values as function of eps values, noPT vs PT and orbital choices
eps_values=[10, 1, 5.00E-01, 1.00E-01, 5.00E-02, 1.00E-02, 9.00E-03, 5.00E-03, 3.00E-03, 2.00E-03, 1.00E-03, 9.00E-04, 7.00E-04, 5.00E-04, 4.00E-04, 3.00E-04, 2.50E-04, 2.00E-04]

#
MP2_unrelaxdens_natorbs_noPT=[-0.05202, -0.04891, -0.05459, -0.06063, -0.069, 0.09296, 0.06381, 0.1064, 0.08493, 0.08144, 0.08668, 0.08776, 0.08854, 0.08838, 0.08803, 0.0882, 0.0882, 0.08824]
CCSD_unrelaxdens_natorbs_noPT=[-0.09421, -0.09421, -0.09421, -0.09421, -0.09421, -0.0481, -0.04938, -0.05203, -0.06199, -0.06332, 0.09164, 0.08153, 0.1066, 0.0818, 0.08266, 0.08599, 0.08804, 0.08836]
CCSD_unrelaxdens_natorbs_withPT=[0.12616, 0.11892, 0.115, 0.11543, 0.1199, 0.08548, 0.11121, 0.09299, 0.10326, 0.08695, 0.07382, 0.06861, 0.07606, 0.08141, 0.08553, 0.08373, 0.0865, 0.08737]
CCSD_T_unrelaxdens_natorbs_noPT=[-0.09421, -0.09421, -0.09421, -0.09421, -0.09421, -0.04696, -0.04509, -0.05515, -0.06133, -0.06319, 0.06271, 0.07093, 0.09887, 0.07937, 0.08137, 0.08837, 0.08849, 0.088]
CCSD_T_unrelaxdens_natorbs_withPT=[0.11421, 0.11421, 0.11421, 0.11421, 0.11421, 0.1274, 0.1264, 0.09754, 0.08624, 0.06832, 0.0793, 0.08091, 0.08567, 0.08557, 0.08628, 0.08761, 0.08793, 0.08841]

print(len(MP2_unrelaxdens_natorbs_noPT))
print(len(CCSD_unrelaxdens_natorbs_noPT))
print(len(CCSD_unrelaxdens_natorbs_withPT))
print(len(CCSD_T_unrelaxdens_natorbs_noPT))
print(len(CCSD_T_unrelaxdens_natorbs_withPT))

#Note: using inverse logarithmic x-scale
dplot = ASH_plot("SHCI DMs for different natorbs and with/without PT", num_subplots=1, x_axislabel="SHCI eps", y_axislabel='Dipole-mom (A.U.)', )
dplot.invert_x_axis(0)
dplot.addseries(0, x_list=eps_values, y_list=MP2_unrelaxdens_natorbs_noPT, x_scale_log=True,
    label='MP2_unrelaxdens_natorbs_noPT', color='blue', line=True, scatter=True)
dplot.addseries(0, x_list=eps_values, y_list=CCSD_unrelaxdens_natorbs_noPT, x_scale_log=True,
    label='CCSD_unrelaxdens_natorbs_noPT', color='green', line=True, scatter=True)
dplot.addseries(0, x_list=eps_values, y_list=CCSD_unrelaxdens_natorbs_withPT, x_scale_log=True,
    label='CCSD_unrelaxdens_natorbs_withPT', color='red', line=True, scatter=True)
dplot.addseries(0, x_list=eps_values, y_list=CCSD_T_unrelaxdens_natorbs_noPT, x_scale_log=True,
    label='CCSD_T_unrelaxdens_natorbs_noPT', color='purple', line=True, scatter=True)
dplot.addseries(0, x_list=eps_values, y_list=CCSD_T_unrelaxdens_natorbs_withPT, x_scale_log=True,
    label='CCSD_T_unrelaxdens_natorbs_withPT', color='black', line=True, scatter=True)

#Save figure
dplot.savefig('Dipole_SHCI',imageformat='png',dpi=300)
