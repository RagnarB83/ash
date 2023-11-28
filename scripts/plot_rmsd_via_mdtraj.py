from ash import *
import mdtraj

#Plotting RMSD of trajectory using mdtraj
#Usage: python3 plot_rmsd_via_mdtraj.py trajfilename pdbfilename

#The name of the trajectory file to load
trajfile=sys.argv[1]
#The name of the PDB-file to use for topology
pdbfile=sys.argv[2]


#Loading using mdtraj
system = mdtraj.load(pdbfile)
traj = mdtraj.load(trajfile, top=system)

print(f"This trajectory contains {traj.n_frames} frames")

#Calculating full RMSD (flawed) w.r.t. first frame
rmsd_all= mdtraj.rmsd(traj, traj[0], 0)

#Sub-system selection: Defining heavy atoms

#Selection: All non-H atoms (also flawed because of solvent)
heavy_atoms = [atom.index for atom in traj.topology.atoms if atom.element.symbol != 'H']

#Selection: All non-H atoms in protein
heavy_protein_atoms = traj.topology.select("protein and (element !=  H)")

#RMSD for heavy protein atoms  w.r.t. first frame
rmsd_heavy_protein = mdtraj.rmsd(traj, traj[0], 0, atom_indices=heavy_protein_atoms)

#Plotting using ASH-plot (matplotlib)
x_label="Frames in trajectory"
y_label="RMSD (nm)"
filelabel="RMSD"
eplot = ASH_plot(filelabel, num_subplots=1, x_axislabel=x_label, y_axislabel=y_label)
eplot.addseries(0, x_list=traj.time, y_list=rmsd_heavy_protein, label=y_label, color='blue', line=True, scatter=True)
eplot.savefig(filelabel)
