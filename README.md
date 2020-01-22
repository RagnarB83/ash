 # Yggdrasill: a computational chemistry environment

 ### Current features: 
- Read in coordinates in multiple ways:
    - coordinate string
    - XYZ file
    - Python lists
- Single-point QM energies with ORCA and xTB:
    - Flexible input structure support any Hamiltonian/basis set available in ORCA or xTB.
    - ORCA parallelization available via OpenMPI
    - To do: Broken-symmetry, ECP-embedding, basis set on specific atoms
- Single-point electrostic embedding QM/MM with ORCA.
    - To do: Lennard-Jones
- Nonbonded Molecular Mechanics (MM) via pointcharges and Lennard-Jones potentials
    - Flexible definition of charges and Lennard-Jones potentials. Either via flexible forcefield inputfile or 
    via 
    - Both energy and gradient available.
    - Limitation: No bonded MM yet.
- Geometry optimization with multiple optimizers
     - Python LBFGS-optimizer in Cartesian coordinates (credit: Vilhjálmur Ásgeirsson). 
     No internal coordinates. Coming: frozen atoms
     - PyBerny optimizer interface with internal coordinates. Limitation: No frozen atoms or constraints.
     - geomeTRIC interface: powerful optimizer supporting multiple internal coordinates 
     (TRIC, HDLC, DLC etc.), frozen atoms, constraints.
     - To do: DL-FIND interface: powerful optimizer supporting DLC, HDLC internal coordinates, frozen atoms, constraints.
     - To do: Support for additional QM codes besides ORCA: xTB, Psi4
- Nonbonded QM/MM Geometry optimization:
    - Possible with geomeTRIC
- Numerical frequencies: one-point (forward difference) and two-point (central difference)
     - Partial Hessian
     - QM=ORCA supported. Todo: xTB
     - QM/MM not yet supported.
     - Todo: Anfreq read-in from ORCA and xTB
- Hessian analysis
  - Diagonalization of Hessian (from Yggdrasill or ORCA). Print frequencies and normal modes.
  - Todo: projection of translation/rotational modes
  - Normal mode composition analysis in terms of individual atoms, elements and atom groups.
  - Print vibrational densities of states files (with linebroadening)
  - Mode mapping: compare normal modes of 2 Hessians (e.g. with isotope substitution) for similarity
  - Read/write ORCA-style Hessian files
  - Print XYZ-trajectory file for individual modes
  - Print thermochemistry. TODO: finish
  - Write frequency output as pseudo ORCA-outputfile (enables visualization of modes in Chemcraft/Avogadro)

  
- Python multiprocessing parallelization functionality for running multiple jobs in parallel.
   - Running many single-point calculations in parallel
   - Running numerical frequency displacements in parallel
- Molecular dynamics
    - To be done
    
- Submodules:
    - molcrys: Automatic Molecular crystal QM/MM
      - Read-in CIF-file, extract cell information and coordinates of asymmetric unit.
      - Fill-up coordinates of unitcell.
      - Expand unit cell.
      - Create spherical cluster from unitcell (with only whole molecules).
      - Near-automatic fragment indentification.
      - Intelligent reordering of fragments (supports messy CIF-files)
      - Automatic creation of nonbonded MM forcefield (charges and LJ potentials).
      - Self-consistent QM/MM for charge definition of cluster.
      - QM/MM Geometry optimization of central fragment of cluster to capture solid-state geometrical effects.
      - QM/MM Numerical frequencies of central fragment of cluster.
      
    - solvshell: Multi-shell solvation for redox potentials
    
 ## How to use: 
 
 
 ### Input structure
You create a Python3 script (e.g. called system.py) and import the Yggdrasill functionality:
```sh
from yggdrasill import *
import settings_yggdrasill
from functions_MM import *
 ```
For convenience you may want to initalize standard global settings (connectivity etc.):
```sh
settings_yggdrasill.init() #initialize
 ```
From then one you have the freedom of writing a Python script in any way you prefer but taking the advantage
of Yggdrasill functionality. Typically you would first create one (or more) molecule fragments.
 
### Creating/modifying fragment objects

Fragments in Yggdrasill are Python objects containing basic information about a molecule. You can create as many fragment objects
as you want. A typical fragment will contain at least Cartesian coordinates about a molecule and the elemental information.
Fragments can be created in multiple ways but will behave the same after creation.
 
 
***Direct creation***
*From string* 
First define multi-line string (called fragcoords here) with element and coordinates (Å) separated by space:
```sh
fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
```
Then define object (here called **HF_frag**) of class *Fragment* using coordinates from the string "fragcoords".
The *Fragment* class is an Yggdrasill class.

```sh
HF_frag=Fragment(coordsstring=fragcoords)
```
---
*From list* 
Another way is if you have lists of coordinates and element information already available.
```sh
elems=['H', 'Cl']
coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
HCl_frag.add_coords(elems,coords)
```
*From external XYZ file* 
Perhaps most convenient is to define the fragment directly from reading an XYZ-file (that exists in the same directory as the script)
```sh
HI_frag = Fragment(xyzfile="hi.xyz")
```
***Via empty object***
An alternative is to first create an empty fragment and then add the coordinate and element information later. This can sometimes be useful and demonstrates here the built-in fragment object functions available (coords_from_string, add_coords, read_xyzfile)
First create empty fragment: HCl_frag:
```sh
HCl_frag=Fragment()
```

*Add coordinates from string* 
Here adding coordinates from a string
```sh
fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
HCl_frag.coords_from_string(fragcoords)
```
*Add coordinates from lists* 
```sh
HCl_frag.add_coords(elems,coords)
```
where elems and coords are lists:
```sh
elems=['H', 'Cl']
coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
```
*Add coordinates from XYZ file* 
```sh
HF_frag.read_xyzfile("hcl.xyz")
```
***Modifying coordinates of object***
If you want to replace coords and elems of a fragment object with new information this can be done conveniently through lists.
```sh
elems=['H', 'Cl']
coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]
HF_frag.replace_coords(elems,coords)
```
TODO:
Add option here of replacing coords from XYZ file and string as well.
Maybe already works via add_coords and coords_from_string object functions. Need to check.


***Calculate connectivity of fragment object***
Connectivity is an important aspect of the fragment as it distinguishes atoms that are in close-contact (i.e. forming some kind of stable covalent bond) and atoms further apart and obviously not bonded. Correct connectivity is crucial for some Yggdrasill functionality.
Currently, connectivity is calculated based on a distance and radii-based criterion (to be documented later).

To calculate the connectivity table for a table
```sh
mol_frag.calc_connectivity()
```
This creates a connectivity table which is a Python list of lists:
An example of a connectivity table would be: [[0,1,2],[3,4,5],[6,7,8,9,10]]
Atoms 0,1,2 are here bonded to each other as a sub-fragment (migh e.g. be an H2O molecule) and so are atoms 3,4,5 and also 6,7,8,9,10.
The connectivity table is available as:
```sh
mol_frag.connectivity
```
TODO: The connectivity table should ideally be calculated automatically when coordinates are added/updated to the fragment.

***Inspect defined fragment objects***
To inspect a defined fragment one can print out a Python dictionary of all defined attributes of the object.
```sh
print("HF_frag dict", HF_frag.__dict__)
```
One can also access individual attributes like this:
```sh
print("HF_frag.coords")
```
```sh
conn = FeFeH2ase.connectivity
print("conn:", conn)
```
```sh
print("Number of subfragments in FeFeH2ase", len(conn))
print("Number of atoms in FeFeH2ase", FeFeH2ase.numatoms)
print("Number atoms in connectivity in FeFeH2ase", FeFeH2ase.connected_atoms_number)
```
