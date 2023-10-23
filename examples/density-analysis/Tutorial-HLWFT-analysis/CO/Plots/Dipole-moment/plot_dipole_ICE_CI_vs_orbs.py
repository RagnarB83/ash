from ash import *

#Plotting dipole moments from ICE-CI as a function of Tgen but using different natural orbitals
tgen_values=[10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 5.00E-05, 1.00E-05, 5.00E-06, 1.00E-06]

#
RHF_orbs=[0.029, 0.02739, 0.02747, 0.02749, 0.02748, 0.0275, 0.02753, 0.09323, 0.08878, 0.08776, 0.08721, 0.08866, 0.08892, 0.0889]
MP2_unrelaxdens_natorbs=[-0.06634, -0.06607, -0.06595, -0.06602, -0.06603, -0.06596, 0.10504, 0.07921, 0.09059, 0.089, 0.08902, 0.08836, 0.08858, 0.08879]
MP2_relaxdens_natorbs=[0.12052, 0.1119, 0.11468, 0.11459, 0.11463, 0.1148, 0.10424, 0.09187, 0.09067, 0.09029, 0.09023, 0.08921, 0.08902, 0.08898]
CCSD_lindens_natorbs=[0.1157, 0.11519, 0.11563, 0.11566, 0.11619, 0.11623, 0.10991, 0.09104, 0.08982, 0.0893, 0.08982, 0.08897, 0.08899, 0.08881]
CCSD_unrelaxdens_natorbs=[0.10022, 0.10296, 0.10468, 0.10483, 0.10485, 0.10483, 0.10483, 0.08569, 0.08777, 0.08835, 0.08963, 0.0889, 0.08882, 0.08877]
CCSD_orboptdens_natorbs=[0.09125, 0.09742, 0.09736, 0.09754, 0.09749, 0.098, 0.09949, 0.08418, 0.08794, 0.08981, 0.08927, 0.08886, 0.08875, 0.08875]
#Note: using inverse logarithmic x-scale
dplot = ASH_plot("ICE-CI DM for different natorbs", num_subplots=1, x_axislabel="ICE-CI Tgen", y_axislabel='Dipole-mom (A.U.)', )
dplot.invert_x_axis(0)
dplot.addseries(0, x_list=tgen_values, y_list=RHF_orbs, x_scale_log=True,
    label='RHF_orbs', color='purple', line=True, scatter=True)
dplot.addseries(0, x_list=tgen_values, y_list=MP2_unrelaxdens_natorbs, x_scale_log=True,
    label='MP2-ur-natorbs', color='blue', line=True, scatter=True)
dplot.addseries(0, x_list=tgen_values, y_list=MP2_relaxdens_natorbs, x_scale_log=True,
    label='MP2-r-natorbs', color='green', line=True, scatter=True)
dplot.addseries(0, x_list=tgen_values, y_list=CCSD_lindens_natorbs, x_scale_log=True,
    label='CCSD-lin-natorbs', color='orange', line=True, scatter=True)
dplot.addseries(0, x_list=tgen_values, y_list=CCSD_unrelaxdens_natorbs, x_scale_log=True,
    label='CCSD-ur-natorbs', color='red', line=True, scatter=True)
dplot.addseries(0, x_list=tgen_values, y_list=CCSD_orboptdens_natorbs, x_scale_log=True,
    label='CCSD-OO-natorbs', color='black', line=True, scatter=True)

#Save figure
dplot.savefig('Dipole_ICE-CI',imageformat='png',dpi=300)
