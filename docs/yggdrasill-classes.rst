=======================
Yggdrasill object classes
=======================

Information about the Yggdrasill object classes.

.. code-block:: python

    class Fragment:
        def __init__(self, coordsstring=None, xyzfile=None, pdbfile=None, coords=None, elems=None):

        #Add coordinates from geometry string. Will replace.
        def add_coords_from_string(self, coordsstring):

        #Replace coordinates by providing elems and coords lists.
        def replace_coords(self, elems, coords):

        #Delete coordinates
        def delete_coords(self):

        #Add coordinates
        def add_coords(self, elems,coords):

        #Print coordinates
        def print_coords(self):

        #Read PDB file
        def read_pdbfile(self,filename):

        #Read XYZ file
        def read_xyzfile(self,filename):

        # Get coordinates for specific atoms (from list of atom indices)
        def get_coords_for_atoms(self, atoms):

        #Calculate connectivity (list of lists) of coords
        def calc_connectivity(self, conndepth=99, scale=None, tol=None ):

        #Update atomcharges of object
        def update_atomcharges(self, charges):

        #Adding fragment-type info (used by molcrys, identifies whether atom is mainfrag, counterfrag1 etc.)
        def add_fragment_type_info(self,fragmentobjects):

        #Write XYZfile with provided name
        def write_xyzfile(self,xyzfilename="Fragment-xyzfile.xyz"):

        #Print system-fragment information to file. Default name of file: "fragment-info
        def print_system(self,filename='fragment-info.txt'):

        #Set energy of fragmentobject
        def set_energy(self,energy):
