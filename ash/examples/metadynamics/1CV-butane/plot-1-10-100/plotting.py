from ash import *

xvalues=np.loadtxt("CV1_coord_values.txt")
rel_free_energy_1 = np.loadtxt("MTD_free_energy-rel_1ps.txt")
rel_free_energy_10 = np.loadtxt("MTD_free_energy-rel_10ps.txt")
rel_free_energy_100 = np.loadtxt("MTD_free_energy-rel_100ps.txt")

eplot = ASH_plot("Metadynamics", num_subplots=1, x_axislabel="Dihedral (°)", y_axislabel="Energy (kcal/mol")
eplot.addseries(0, x_list=xvalues, y_list=rel_free_energy_1, legend=True, label="1 ps", color='blue', line=True, scatter=False)
eplot.addseries(0, x_list=xvalues, y_list=rel_free_energy_10, legend=True, label="10 ps", color='red', line=True, scatter=False)
eplot.addseries(0, x_list=xvalues, y_list=rel_free_energy_100, legend=True, label="100 ps", color='green', line=True, scatter=False)
eplot.savefig('MTD_1-10-100-ps')
