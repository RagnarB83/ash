#!/usr/bin/env python3
"""Re-image a solvated box so that the solute (first N atoms) sits in the box centre.

ASH's own re-imaging (MDtraj_imagetraj) picks anchor molecules from the bond graph and
fails for a solute without bonds such as the nonbonded FeCl4 model. Here the anchor is
given explicitly; coordinates are written back with the original OpenMM topology so atom
and residue names stay exactly as the force-field XML expects.
"""
import argparse

import mdtraj
import numpy as np
import openmm.app as app
import openmm.unit as unit

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("pdb_in")
p.add_argument("pdb_out")
p.add_argument("--solute-atoms", type=int, required=True, help="number of leading solute atoms")
a = p.parse_args()

traj = mdtraj.load(a.pdb_in)
anchor = [set(traj.topology.atom(i) for i in range(a.solute_atoms))]
imaged = traj.image_molecules(anchor_molecules=anchor)
pdb = app.PDBFile(a.pdb_in)
with open(a.pdb_out, "w") as f:
    app.PDBFile.writeFile(pdb.topology, np.asarray(imaged.xyz[0]) * unit.nanometer, f, keepIds=True)
box = traj.unitcell_lengths[0] * 10
com = imaged.xyz[0][:a.solute_atoms].mean(axis=0) * 10
print(f"box {box} A, solute centroid {com} A")
