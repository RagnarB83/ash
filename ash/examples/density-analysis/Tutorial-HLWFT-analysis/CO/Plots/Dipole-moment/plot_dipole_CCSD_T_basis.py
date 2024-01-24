from ash import *

#Plotting CCSD(T) DM values as function of basis set and unrelaxed and relaxed density
basis_sets =[2,3,4,5]

pyscf_CCSD_T_unrelaxdens=[8.49846838e-02,6.66072853e-02,5.09848534e-02,4.71905587e-02]
cfour_CCSD_T_relaxdens=[0.0854236795,0.0640831953,0.0477566565,0.0442905011]
experiment=0.04799
#Debye to a.u. conversion: 2.541874535611931
#CFour_CCSD_T_relaxdens=[8.49846838e-02,6.66072853e-02,5.09848534e-02,4.71905587e-02]

#Note: using inverse logarithmic x-scale
dplot = ASH_plot("CCSD(T) DMs with increasing basis sets", num_subplots=1, x_axislabel="Basis set cardinal", y_axislabel='Dipole-mom (A.U.)', )
dplot.addseries(0, x_list=basis_sets, y_list=pyscf_CCSD_T_unrelaxdens,
    label='pyscf_CCSD_T_unrelaxdens', color='blue', line=True, scatter=True)
dplot.addseries(0, x_list=basis_sets, y_list=cfour_CCSD_T_relaxdens,
    label='cfour_CCSD_T_relaxdens', color='purple', line=True, scatter=True)
dplot.addseries(0, x_list=[2,5], y_list=[experiment,experiment],
    label='Experiment', color='black', line=True, scatter=False,linestyle="--")
#Save figure
dplot.savefig('Dipole_CCSD_T_basis',imageformat='png',dpi=300)
