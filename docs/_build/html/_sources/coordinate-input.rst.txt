==========================
Coordinates and fragments
==========================


Creating/modifying fragment objects
***********************************

Fragments in Yggdrasill are Python objects containing basic information about a molecule. You can create as many fragment objects
as you want. A typical fragment will contain at least Cartesian coordinates about a molecule and the elemental information.
Fragments can be created in multiple ways but will behave the same after creation.

Fragments are Python objects created from the Yggdrasill *Fragment* object class.
See XXFragment-class-page-linkXX for an overview of all Fragment class attributes and functions.

Direct creation of fragment from coordinates
==============================================

*From string*

First define multi-line string (called fragcoords here) with element and coordinates (Å) separated by space:

.. code-block:: python

    fragcoords="""
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

Then define object (here called **HF_frag**) of class *Fragment* by passing the coordinates to *coordsstring*, using coordinates from the string "fragcoords".
The *Fragment* class is an Yggdrasill class.

.. code-block:: python

    HF_frag=Fragment(coordsstring=fragcoords)



*From list*

Another way is if you have lists of coordinates and element information already available.

.. code-block:: python

    elems=['H', 'Cl']
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
    HCl_frag=Fragment(elems=elems,coords=coords)


*From external XYZ file*

Perhaps most convenient is to define the fragment directly from reading an XYZ-file (that exists in the same directory as the script):

.. code-block:: python

    HI_frag = Fragment(xyzfile="hi.xyz")

Adding coordinates to empty object
=====================================

An alternative to the direct way is to first create an empty fragment and then add the coordinate and element information later.
This can sometimes be useful and demonstrates here the built-in fragment object functions available (coords_from_string, add_coords, read_xyzfile)
First create empty fragment:

.. code-block:: python

    HCl_frag=Fragment()


*Add coordinates from string*


.. code-block:: python

    fragcoords="""
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """
    HCl_frag.add_coords_from_string(fragcoords)


**Note:** This will append coordinates to fragment. If fragment already contains some coordinates the specified coordinates
will be appended.

*Add coordinates from lists*

.. code-block:: python

    HCl_frag.add_coords(elems,coords)

where elems and coords are lists:

.. code-block:: python

    elems=['H', 'Cl']
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]


**Note:** This will append coordinates to fragment. If fragment already contains some coordinates the added coordinates
will follow.

*Add coordinates from XYZ file*

.. code-block:: python

    HF_frag.read_xyzfile("hcl.xyz")


**Note:** This will append coordinates to fragment. If fragment already contains some coordinates the added coordinates
will follow.


Replace coordinates of object
==============================
If you want to replace coords and elems of a fragment object with new information this can be done conveniently through lists.

.. code-block:: python

    elems=['H', 'Cl']
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]
    HF_frag.replace_coords(elems,coords)

**TODO:** Add option here of replacing coords from XYZ file and string as well.


Delete coordinates of object
==============================
If you want to delete coordinates from object (both coords list and elems lists) then this is easily done.

.. code-block:: python

    HF_frag.delete_coords()


Calculate connectivity of fragment object
===========================================

Connectivity is an important aspect of the fragment as it distinguishes atoms that are in close-contact (i.e. forming some kind of stable covalent bond) and atoms further apart and obviously not bonded. Correct connectivity is crucial for some Yggdrasill functionality.
Currently, connectivity is calculated based on a distance and radii-based criterion (to be documented later).

.. role:: red

:red:`DOCUMENT BASIC CONNECTIVITY HERE`

To calculate the connectivity table for a table

.. code-block:: python

    mol_frag.calc_connectivity()

This creates a connectivity table which is a Python list of lists:
An example of a connectivity table would be: [[0,1,2],[3,4,5],[6,7,8,9,10]]
Atoms 0,1,2 are here bonded to each other as a sub-fragment (migh e.g. be an H2O molecule) and so are atoms 3,4,5 and also 6,7,8,9,10.
The connectivity table is available as:

.. code-block:: python

    mol_frag.connectivity


Note. The connectivity table is calculated or recalculated automatically when coordinates are added or when modified to the fragment.


Inspect defined fragment objects
==============================

To inspect a defined fragment one can print out a Python dictionary of all defined attributes of the object.
.. code-block:: python

    print("HF_frag dict", HF_frag.__dict__)

One can also access individual attributes like accessing the pure coordinates only:

.. code-block:: python

    print("HF_frag.coords")

More conveniently would be to use the print_coords function though (to print elems and coords):

.. code-block:: python

    print("HF_frag.print_coords")


Get coords and elems of specific atom indices:

.. code-block:: python

    specific_coords,specific_elems=HF_frag.get_coords_for_atoms([0,1,2])

Print connectivity:

.. code-block:: python

    conn = FeFeH2ase.connectivity
    print("conn:", conn)
    print("Number of subfragments in FeFeH2ase", len(conn))

Print number of atoms and number of connected atoms:

.. code-block:: python

    print("Number of atoms in FeFeH2ase", FeFeH2ase.numatoms)
    print("Number atoms in connectivity in FeFeH2ase", FeFeH2ase.connected_atoms_number)

All defined system attributed can be printed conveniently to disk:

.. code-block:: python

    HF_frag.print_system(filename='fragment-info.txt')

An XYZ file of coordinates can be printed out:

.. code-block:: python

    HF_frag.write_xyzfile(xyzfilename="Fragment-xyzfile.xyz")

