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
HCl_frag=Fragment(elems,coords)
```
---
*From external XYZ file* 

Perhaps most convenient is to define the fragment directly from reading an XYZ-file (that exists in the same directory as the script):
```sh
HI_frag = Fragment(xyzfile="hi.xyz")
```~~~~~~~~~~~~
---
***Via empty object***

An alternative is to first create an empty fragment and then add the coordinate and element information later. This can sometimes be useful and demonstrates here the built-in fragment object functions available (coords_from_string, add_coords, read_xyzfile)
First create empty fragment:
```sh
HCl_frag=Fragment()
```
---
*Add coordinates from string* 

Here adding coordinates from a string
```sh
fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
HCl_frag.coords_from_string(fragcoords)
```

Note. This will append coordinates to fragment. If fragment already contains some coordinates the added coordinates
will follow.

---
*Add coordinates from lists* 

```sh
HCl_frag.add_coords(elems,coords)
```
where elems and coords are lists:
```sh
elems=['H', 'Cl']
coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
```

Note. This will append coordinates to fragment. If fragment already contains some coordinates the added coordinates
will follow.

---
*Add coordinates from XYZ file* 

```sh
HF_frag.read_xyzfile("hcl.xyz")
```

Note. This will append coordinates to fragment. If fragment already contains some coordinates the added coordinates
will follow.

---
***Replace coordinates of object***

If you want to replace coords and elems of a fragment object with new information this can be done conveniently through lists.
```sh
elems=['H', 'Cl']
coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]
HF_frag.replace_coords(elems,coords)
```
Todo: Add option here of replacing coords from XYZ file and string as well.

---
***Delete coordinates of object***

If you want to delete coordinates from object (both coords list and elems lists) then this is easily done.
```sh
HF_frag.delete_coords()
```
---
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

Note. The connectivity table is calculated or recalculated automatically when coordinates are added or when modified to the fragment.

---
***Inspect defined fragment objects***

To inspect a defined fragment one can print out a Python dictionary of all defined attributes of the object.
```sh
print("HF_frag dict", HF_frag.__dict__)
```
One can also access individual attributes like accessing the pure coordinates only:
```sh
print("HF_frag.coords")
```
More conveniently would be to use the print_coords function though (to print elems and coords):
```sh
print("HF_frag.print_coords")
```

Get coords and elems of specific atom indices:
```sh
specific_coords,specific_elems=HF_frag.get_coords_for_atoms([0,1,2])
```
Print connectivity:
```sh
conn = FeFeH2ase.connectivity
print("conn:", conn)
print("Number of subfragments in FeFeH2ase", len(conn))
```
Print number of atoms and number of connected atoms:
```sh
print("Number of atoms in FeFeH2ase", FeFeH2ase.numatoms)
print("Number atoms in connectivity in FeFeH2ase", FeFeH2ase.connected_atoms_number)
```
All defined system attributed can be printed conveniently to disk:
```sh
HF_frag.print_system(filename='fragment-info.txt')
```
An XYZ file of coordinates can be printed out:
```sh
HF_frag.write_xyzfile(xyzfilename="Fragment-xyzfile.xyz")
```