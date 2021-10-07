from ash import *
import sys

#Define global system settings ( scale, tol and conndepth keywords for connectivity)


#Testing multiple ways of creating fragments. Currently just checking if we avoid Python errors.
#TO be checked whether fragment attributes are correct in all cases.

fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
####################################################
#Creation of empty fragment and add coords
HF_frag=Fragment()

#Add coordinates to fragment
HF_frag.add_coords_from_string(fragcoords)
###########################################
#From lists
elems=['H', 'Cl']
coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
HCl_frag=Fragment(elems=elems,coords=coords)
##############################
#From XYZ file
HI_frag = Fragment(xyzfile="/home/bjornsson/ASH-DEV_GIT/testsuite/hi.xyz")
#######################################
#New frag from fragcoords directly
HF_frag2=Fragment(coordsstring=fragcoords)
##################################
#Replace coordinates in fragment
elems=['H', 'Cl']
coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]
HCl_frag.replace_coords(elems,coords)
##############################
# Delete coordinates
HF_frag.delete_coords()

#########################
#Recalculate connectivity
HCl_frag.calc_connectivity()
print(HCl_frag.connectivity)


assert HF_frag2.numatoms == 2, "Number of atoms is not correct"
assert HI_frag.numatoms == 2, "Number of atoms is not correct"
assert HF_frag.numatoms == 2, "Number of atoms is not correct"
assert HCl_frag.numatoms == 2, "Number of atoms is not correct"

sys.exit(0)
