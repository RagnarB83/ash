from ash import *
import os

eplot = ASH_plot("Metadynamics", num_subplots=1, x_axislabel="Dihedral (°)", y_axislabel="Energy (kcal/mol")

#Filenames to grab data from
cv_coord_file_name="CV1_coord_values.txt"
free_energy_file_name="MTD_free_energy_rel.txt"

colors=['black', 'green', 'blue', 'orange', 'orange', 'indigo', 'lavender', 'hotpink', 'khaki', 'saddlebrown', 'magenta', 'limegreen', 'maroon', 'olive', 'orchid', 'navy', 'purple', 'rosybrown', 'silver']

#Looping over directories (using dirnames as labels)
for i,dir in enumerate(os.listdir(os.getcwd())):
    try:
        xvalues=np.loadtxt(f"{dir}/{cv_coord_file_name}")
        rel_free_energy = np.loadtxt(f"{dir}/{free_energy_file_name}")
        eplot.addseries(0, x_list=xvalues, y_list=rel_free_energy, legend=True, label=dir, color=colors[i], line=True, scatter=False)
    except:
        pass
eplot.savefig('MTD_biaswidth_tests')
