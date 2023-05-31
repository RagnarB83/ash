import os
from ash.functions.functions_general import ashexit
import numpy as np

def MDtraj_import():
    print("Importing mdtraj (https://www.mdtraj.org)")
    try:
        import mdtraj
    except ImportError:
        print("Problem importing mdtraj. Try: 'pip install mdtraj' or 'conda install -c conda-forge mdtraj'")
        ashexit()
    return mdtraj


def MDtraj_RMSF(trajectory, pdbtopology, print_largest_values=True, threshold=0.005, largest_values=10):
    print("Inside MDtraj_RMSF")
    #Import mdtraj library
    mdtraj = MDtraj_import()
    
    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)
    firstframe=traj[0]
    rmsflist = mdtraj.rmsf(traj, reference=None, frame=0, atom_indices=None, parallel=True)
    
    if print_largest_values is True:
        print(f"Will print RMSF largest_values={largest_values}")
        large_rmsf_indices = rmsflist.argsort()[::-1][:largest_values]
    else:
        print(f"Will print atom RMSF values larger than threshold={threshold}")
        large_rmsf_indices = np.where(rmsflist > threshold)[0]
    if len(large_rmsf_indices) > 0:
        print("Printing atoms with high root-mean-square fluctuations:")
        print("Index    Residue-atom           Coordinates                              RMSF")
        for i in large_rmsf_indices:
            atom_string=str(firstframe.topology.atom(i))
            rmsfvalue=rmsflist[i]
            print(f"{i:>6} {atom_string:<14} {firstframe.xyz[0][i][0]:>12.6f} {firstframe.xyz[0][i][1]:>12.6f} {firstframe.xyz[0][i][2]:>12.6f}      {rmsfvalue:>12.6f}")
    return large_rmsf_indices


# anchor_molecules. Use if automatic guess fails
def MDtraj_imagetraj(trajectory, pdbtopology, format='DCD', unitcell_lengths=None, unitcell_angles=None,
                     solute_anchor=None):
    #Trajectory basename
    traj_basename = os.path.splitext(trajectory)[0]
    #PDB-file basename
    pdb_basename = os.path.splitext(pdbtopology)[0]
    
    #Import mdtraj library
    mdtraj = MDtraj_import()

    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)

    #Also load the pdbfile as a trajectory-snapshot (in addition to being topology)
    pdbsnap = mdtraj.load(pdbtopology, top=pdbtopology)
    pdbsnap_imaged = pdbsnap.image_molecules()

    numframes = len(traj._time)
    print("Found {} frames in trajectory.".format(numframes))
    print("PBC information in trajectory:")
    print("Unitcell lengths:", traj.unitcell_lengths[0])
    print("Unitcell angles", traj.unitcell_angles[0])
    # If PBC information is missing from traj file (OpenMM: Charmmfiles, Amberfiles option etc) then provide this info
    if unitcell_lengths is not None:
        print("unitcell_lengths info provided by user.")
        unitcell_lengths_nm = [i / 10 for i in unitcell_lengths]
        traj.unitcell_lengths = np.array(unitcell_lengths_nm * numframes).reshape(numframes, 3)
        traj.unitcell_angles = np.array(unitcell_angles * numframes).reshape(numframes, 3)
    # else:
    #    print("Missing PBC info. This can be provided by unitcell_lengths and unitcell_angles keywords")

    # Manual anchor if needed
    # NOTE: not sure how well this works but it's something
    if solute_anchor is True:
        anchors = [set(traj.topology.residue(0).atoms)]
        print("anchors:", anchors)
        # Re-imaging trajectory
        imaged = traj.image_molecules(anchor_molecules=anchors)
    else:
        imaged = traj.image_molecules()

    # Save trajectory in format
    if format == 'DCD':
        imaged.save(traj_basename + '_imaged.dcd')
        print("Saved reimaged trajectory:", traj_basename + '_imaged.dcd')
    elif format == 'PDB':
        imaged.save(traj_basename + '_imaged.pdb')
        print("Saved reimaged trajectory:", traj_basename + '_imaged.pdb')
    else:
        print("Unknown trajectory format.")

    #Save PDB-snapshot
    pdbsnap_imaged.save(pdb_basename + '_imaged.pdb')
    print("Saved reimaged PDB-file:", pdb_basename + '_imaged.pdb')
    #Return last frame as coords or ASH fragment ?
    #Last frame coordinates as Angstrom
    lastframe=imaged[-1]._xyz[-1]*10

    return lastframe


# Slicing trajectory. Mostly to grab specific snapshot
#TODO: allow option to grab by ps? Requires information about timestep and traj-frequency
def MDtraj_slice(trajectory, pdbtopology, format='PDB', frames=None):

    if frames is None:
        print("frames needs to be set")
        ashexit()

    #Trajectory basename
    traj_basename = os.path.splitext(trajectory)[0]
    
    #Import mdtraj library
    mdtraj = MDtraj_import()

    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)
    print(f"This trajectory contains {traj.n_frames} frames")
    #Slicing trajectory
    print("Slicing trajectory using frame selection:", frames)
    tslice = traj[frames[0]:frames[1]]
    print(f"Trajectory slice contains {tslice.n_frames} frames")
    if tslice.n_frames == 0:
        print(f"0 frames found when slicing. You probably should do: frames=[{frames[0]},{frames[1]+1}] instead")
        print("Exiting")
        ashexit()

    # Save trajectory in format
    if format == 'DCD':
        tslice.save(traj_basename + f'_frame{frames[0]}_{frames[1]}.dcd')
        print("Saved sliced trajectory:", traj_basename + f'_frame{frames[0]}_{frames[1]}.dcd')
    elif format == 'PDB':
        tslice.save(traj_basename + f'_frame{frames[0]}_{frames[1]}.pdb')
        print("Saved sliced trajectory:", traj_basename + f'_frame{frames[0]}_{frames[1]}.pdb')
    elif format == 'XYZ':
        tslice.save(traj_basename + f'_frame{frames[0]}_{frames[1]}.xyz')
        print("Saved sliced trajectory:", traj_basename + f'_frame{frames[0]}_{frames[1]}.xyz')
    else:
        print("Unknown trajectory format.")
    return

#Function to get internal coordinates from trajectory fast
#Give trajectory file
def MDtraj_coord_analyze(trajectory, pdbtopology=None, periodic=True, indices=None):
    print("Inside MDtraj_coord_analyze")
    if indices is None:
        print("indices needs to be set")
        ashexit()
    print("Trajectory:", trajectory)
    print("Topology:", pdbtopology)
    print("Atom indices:", indices)
    #Import mdtraj library
    mdtraj = MDtraj_import()

    if pdbtopology == None:
        print("A topology is required but was not provided")
        print("Checking if trajectory.pdb file (created by ASH_OpenMM_MD) is available:")
        try:
            pdbtopology=("trajectory.pdb")
        except:
            print("Found no file. Exiting")
            ashexit()

    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)
    print(f"This trajectory contains {traj.n_frames} frames")
    if len(indices) == 4:
        print("4 atom indices given. This must be a dihedral angle.  Returning dihedral in radians")
        output = mdtraj.compute_dihedrals(traj, [indices], periodic=periodic, opt=True)
        unit_label = "radians"
    elif len(indices) == 3:
        print("3 atom indices given. This must be an angle. Returning angle in radians")
        output = mdtraj.compute_angles(traj, [indices], periodic=periodic, opt=True)
        unit_label = "radians"
    elif len(indices) == 2:
        print("2 atom indices given. This must be a distance.  Returning angle in Angstrom")
        output = mdtraj.compute_distances(traj, [indices], periodic=periodic, opt=True)
        output = 10*output
        unit_label = "Angstrom"
    else:
        print("something wrong with indices supplied:", indices)
        ashexit()
    print(f"List of coordinates ({len(output)}) for each frame:", output)

    ave = np.mean(output)
    stdev = np.std(output)
    print(f"Mean: {ave} {unit_label}")
    print(f"Standard deviation: {stdev} {unit_label}")

    return output



#Initial unfinished interface to mdanalysis
def MDAnalysis_transform(topfile, trajfile, solute_indices=None, trajoutputformat='PDB', trajname="MDAnalysis_traj"):
    # Load traj
    print("MDAnalysis interface: transform")
    ashexit()
    try:
        import MDAnalysis as mda
        import MDAnalysis.transformations as trans
    except ImportError:
        print("Problem importing MDAnalysis library.")
        #print("Install via: 'pip install mdtraj'")
        ashexit()

    print("Loading trajecory using MDAnalysis")
    print("Topology file:", topfile)
    print("Trajectory file:", trajfile)
    print("Solute_indices:", solute_indices)
    print("Trajectory output format:", trajoutputformat)
    print("Will unwrap solute and center in box.")
    print("Will then wrap full system.")

    # Load trajectory
    u = mda.Universe(topfile, trajfile, in_memory=True)
    print(u.trajectory.ts, u.trajectory.time)

    # Grab solute
    numatoms = len(u.atoms)
    solutenum = len(solute_indices)
    solute = u.atoms[:solutenum]
    # solvent = u.atoms[solutenum:numatoms]
    fullsystem = u.atoms[:numatoms]
    elems_list = list(fullsystem.types)
    # Guess bonds. Could also read in vdW radii. Could also read in connectivity from ASH if this fails
    solute.guess_bonds()
    # Unwrap solute, center solute and wraps full system (or solvent)
    workflow = (trans.unwrap(solute),
                trans.center_in_box(solute, center='mass'),
                trans.wrap(fullsystem, compound='residues'))

    u.trajectory.add_transformations(*workflow)
    if trajoutputformat == 'PDB':
        fullsystem.write(trajname + ".pdb", frames='all')

    # TODO: Distinguish between transforming whole trajectory vs. single geometry
    # Maybe just read in single-frame trajectory so that things are general
    # Returning last frame. To be used in ASH workflow
    lastframe = u.trajectory[-1]

    return elems_list, lastframe._pos
