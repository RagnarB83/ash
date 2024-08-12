from ash import *


#QM/MM tests with OpenMMTheory and NonBondedTheory, using PySCFTheory for QM-part

def test_qm_mm_pyscf_nonbondedtheory_MeOH_H2O():

    # H2O...MeOH fragment defined. Reading XYZ file
    H2O_MeOH = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/h2o_MeOH.xyz")

    # Specifying the QM atoms (3-8) by atom indices (MeOH). The other atoms (0,1,2) is the H2O and MM.
    # IMPORTANT: atom indices begin at 0.
    qmatoms=[3,4,5,6,7,8]

    # Charge definitions for whole system.
    # Charges for the QM atoms are zero (since ASH will always set QM atoms to zero in elstat embedding)
    atomcharges=[-0.834, 0.417, 0.417, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Defining atomtypes for whole system
    atomtypes=['OT','HT','HT','CX','HX', 'HX', 'HX', 'OT', 'HT']

    # Read forcefield (here containing LJ-part only) from file
    MM_forcefield=MMforcefield_read(f"{ashpath}/tests/extra_files/MeOH_H2O-sigma.ff")

    # QM object (RI off for less numerical noise)
    qm = PySCFTheory(scf_type="RKS", functional="PBE", basis="def2-SVP", densityfit=False)

    # Defining NonBondedTheory object from atomcharges, atomtypes and forcefield
    MMpart = NonBondedTheory(charges = atomcharges, atomtypes=atomtypes, forcefield=MM_forcefield,
        LJcombrule='geometric', codeversion="py")
    # Creating QM/MM object
    QMMMobject = QMMMTheory(fragment=H2O_MeOH, qm_theory=qm, mm_theory=MMpart, qmatoms=qmatoms,
                            embedding='elstat')

    # Single-point energy calculation of QM/MM object
    result = Singlepoint(theory=QMMMobject, fragment=H2O_MeOH, charge=0, mult=1, Grad=True)

    # Determined 11 aug 2024  pscf 2.6.2
    # Note: 0.01 kcal/mol different w.r.t. OpenMM (below).
    # Considered satisfactory, not clear where difference comes from.
    ref_energy=-115.816202526812
    ref_gradient=np.array([[-0.0976002,   0.06207825,  0.02913397],
                    [0.02114902, -0.07293206, -0.04991815],
                    [0.07600821,  0.010929,    0.02069938],
                    [-0.00145506, -0.00227548, -0.01546497],
                    [-0.0020466,   0.00636848,  0.00799227],
                    [0.00983804, -0.0036295,   0.00482348],
                    [-0.00546189, -0.00833844,  0.00609906],
                    [-0.00195135,  0.01570807, -0.00057301],
                    [0.00151874, -0.00790899, -0.00279125]])

    assert np.isclose(result.energy, ref_energy, atol=2e-6), "Energy is not correct"
    assert np.allclose(result.gradient, ref_gradient, atol=1e-5), "Gradient is not correct"


def test_qm_mm_pyscf_openmm_MeOH_H2O():

    # H2O...MeOH fragment defined. Reading XYZ file
    H2O_MeOH = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/h2o_MeOH.xyz")

    # Write PDB-file for OpenMM (used for topology)
    H2O_MeOH.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    pdbfile="h2o_MeOH.pdb"

    # Specifying the QM atoms (3-8) by atom indices (MeOH). The other atoms (0,1,2) is the H2O and MM.
    # IMPORTANT: atom indices begin at 0.
    qmatoms=[3,4,5,6,7,8]

    # QM
    qm = PySCFTheory(scf_type="RKS", functional="PBE", basis="def2-SVP", densityfit=False)

    # MM: OpenMMTheory using XML-file
    MMpart = OpenMMTheory(xmlfiles=[f"{ashpath}/tests/extra_files/MeOH_H2O-sigma.xml"], pdbfile=pdbfile, autoconstraints=None, rigidwater=False)

    # Creating QM/MM object
    QMMMobject = QMMMTheory(fragment=H2O_MeOH, qm_theory=qm, mm_theory=MMpart, qmatoms=qmatoms,
                            embedding='Elstat')

    # Single-point energy calculation of QM/MM object
    result = Singlepoint(theory=QMMMobject, fragment=H2O_MeOH, charge=0, mult=1, Grad=True)

    # Determined 11 aug 2024 using OpenMM 8.1 and pscf 2.6.2
    # Note: 0.01 kcal/mol difference w.r.t. NonBondedTheory (above).
    ref_energy=-115.816183777727
    ref_gradient=np.array([[-0.09756495,  0.06210685,  0.02911808],
                [0.02114902, -0.07293212, -0.04991817],
                [0.07600821,  0.01092899,  0.0206994],
                [-0.00148712, -0.00230238, -0.01544885],
                [-0.00204659,  0.00636847,  0.00799225],
                [0.00983799, -0.00362948,  0.00482346],
                [-0.00546187, -0.00833839,  0.00609904],
                [-0.00195452,  0.01570642, -0.00057316],
                [0.00151874, -0.00790902, -0.00279129]])

    assert np.isclose(result.energy, ref_energy, atol=2e-6), "Energy is not correct"
    assert np.allclose(result.gradient, ref_gradient, atol=1e-5), "Gradient is not correct"

#Read Lysozyme solvated PDB-file, create OpenMMTheory job, define pyscftheory, create QM/MM and run SP
def test_qm_mm_pyscf_openmm_lysozyme():

    numcores=2
    #Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment or PDB-file)
    pdbfile=f"{ashpath}/tests/pdbfiles/1aki_solvated.pdb"
    fragment=Fragment(pdbfile=pdbfile)

    #Creating new OpenMM object from OpenMM full system file
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"], pdbfile=pdbfile, periodic=True,
            numcores=numcores, autoconstraints=None, rigidwater=False)
    #QM
    qmatomlist = [1013,1014,1015,1016,1017,1018]
    qm = PySCFTheory(scf_type="RKS",  functional="BP86", basis="def2-SVP", densityfit=True)
    #qm = xTBTheory()
    # Create QM/MM OBJECT
    qmmmobject = QMMMTheory(qm_theory=qm, mm_theory=omm, qm_charge=-1, qm_mult=1,
        fragment=fragment, embedding="Elstat", qmatoms=qmatomlist, printlevel=2)

    Singlepoint(theory=qmmmobject, fragment=fragment, Grad=True)
