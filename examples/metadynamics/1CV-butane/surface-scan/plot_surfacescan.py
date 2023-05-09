from ash import *


#Plot potential energy plot
surfacedictionary = read_surfacedict_from_file("surface_results.txt", dimension=1)

reactionprofile_plot(surfacedictionary, finalunit='kcal/mol', label='Potential energy scan', 
                        x_axislabel='Dihedral (°)', y_axislabel='Energy', 
                        dpi=200, mode='pyplot', filename='butane_potenergy_scan',
                        imageformat='png', RelativeEnergy=True, 
                        pointsize=40, scatter_linewidth=1, line_linewidth=1, color='blue')
