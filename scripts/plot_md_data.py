#Simple script to plot csv data from MD simulation
#Use like this: python3 plot_md_data.py md_data.csv
from ash import *
import pandas as pd
import sys

#Read file as first argument
file = sys.argv[1]

# Read using pandas
df = pd.read_csv(file)

#Grabbing columns from pandas df
steps = df['#"Step"']
time = df['Time (ps)']
temperature = df['Temperature (K)']

#Creating data list
data_list=[temperature]
#Density and Volume only present for NPT
try:
    density = df['Density (g/mL)']
    data_list.append(density)
    volume = df['Box Volume (nm^3)']
    data_list.append(volume)
except:
    pass

#Looping over data_list and plot
for pd_col in data_list:
    np_array = pd_col.to_numpy()
    label=pd_col.name
    label_no_unit = label.split('(')[0].replace(' ','')
    print(label_no_unit)
    eplot = ASH_plot(label, num_subplots=1, x_axislabel="Steps", y_axislabel=label)
    eplot.addseries(0, x_list=steps.to_numpy(), y_list=np_array, label=label, color='blue', line=True, scatter=True)
    eplot.savefig(label_no_unit)
