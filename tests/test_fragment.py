from ash import *

def test_fragread():
    #Testing multiple ways of creating/modifying fragments.

    fragcoords="""
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """
    HF_frag=Fragment(coordsstring=fragcoords)
    ####################################################
    #From lists
    elems=['H', 'Cl']
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
    HCl_frag=Fragment(elems=elems,coords=coords)
    ##############################
    #From np array
    elems2=['H', 'Cl']
    coords2=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]])
    HCl_frag_np=Fragment(elems=elems2,coords=coords2)
    ##############################
    #From XYZ file
    HI_frag = Fragment(xyzfile="xyzfiles/hi.xyz")
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
    assert HCl_frag_np.numatoms == 2, "Number of atoms is not correct"

def test_fragread_files():
    #Creating fragment, write to disk and read-in again

    #Create fragment from string
    fragcoords="""
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """
    HF_frag=Fragment(coordsstring=fragcoords)
    print("HF_frag conn", HF_frag.connectivity)
    #Print frag to disk
    HF_frag.print_system('HF_frag.ygg')

    #Creating Ash fragment by reading file (above)
    New_frag=Fragment(fragfile='HF_frag.ygg')

    print("New_frag:", New_frag)
    print("New_frag dict:", New_frag.__dict__)

    assert New_frag.numatoms == 2, "Number of atoms is not correct"
    assert New_frag.nuccharge == 10, "Nuccharge of fragment is incorrect"

def test_read_pdb():
    #Define global system settings ( scale, tol and conndepth keywords for connectivity)

    #PDB read in
    PDB_frag = Fragment(pdbfile="pdbfiles/1aki.pdb", conncalc=False)
    print("PDB_frag:", PDB_frag)
    #print("PDB frag dict", PDB_frag.__dict__)
    print(PDB_frag.numatoms)

    assert PDB_frag.numatoms == 1079, "Number of atoms in fragment is incorrect"

