# OpenMM Modeller: residue variants and custom residues

This note explains the special-residue options used by ASH's
`OpenMM_Modeller`. The runnable preparation examples are in
[`rubredoxin/1-Preparation`](rubredoxin/1-Preparation/).

## Choosing a protonation variant

Use `residue_variants` to override OpenMM's automatic protonation/tautomer
choice for individual standard residues. Keys are the PDB chain IDs; inner keys
are the integer residue IDs from the PDB file. These are PDB residue identifiers,
not ASH's zero-based atom indices.

```python
residue_variants = {
    "A": {
        6: "CYX",   # cysteine without a sulfur hydrogen
        14: "HID",  # neutral histidine with hydrogen on ND1
        39: "GLU",  # deprotonated glutamate
    }
}

OpenMM_Modeller(
    pdbfile="protein.pdb",
    forcefield="CHARMM36",
    residue_variants=residue_variants,
)
```

Replace the chain and residue IDs with those in your PDB. ASH prints a residue
table during setup showing the chain ID, PDB residue ID, and the requested
variant; use it to check that each override matched the intended residue.
Without an override, OpenMM chooses variants based on pH and local structure.

OpenMM's built-in hydrogen builder recognizes these common variants:

| Residue | Variants | Meaning |
| --- | --- | --- |
| Aspartate | `ASH`, `ASP` | Neutral, protonated / negatively charged |
| Cysteine | `CYS`, `CYX` | Thiol / no sulfur hydrogen (including disulfide form) |
| Glutamate | `GLH`, `GLU` | Neutral, protonated / negatively charged |
| Histidine | `HID`, `HIE`, `HIP`, `HIN` | ND1-protonated / NE2-protonated / doubly protonated / unprotonated |
| Lysine | `LYS`, `LYN` | Positively charged / neutral |

Names and availability can depend on the force field. Check the force-field
templates and the [OpenMM `Modeller.addHydrogens` documentation](https://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html)
for the OpenMM version used in the calculation.

## Nonstandard residues and cofactors

`residue_variants` changes protonation forms of residues recognized by the
force field. It does not parameterize an arbitrary modified residue. For a
cofactor or other nonstandard residue, provide a force-field XML file that
defines its atoms, bonds, charges, and nonbonded parameters:

```python
openmmobject, fragment = OpenMM_Modeller(
    pdbfile="protein_with_cofactor.pdb",
    forcefield="CHARMM36",
    extraxmlfile="specialresidue.xml",
    residue_variants={"A": {6: "CYX"}},
)
```

For an ambiguous residue-template match, use `residuetemplate_choice` to map
the residue name from the PDB to the intended template name in the XML, for
example `{"FER": "FE2"}`. If you construct a separate `OpenMMTheory` from
the resulting PDB later, include the same custom XML among its force-field
files.

ASH's rubredoxin preparation scripts show this pattern with a custom
iron-sulfur cofactor XML and cysteine variants:
[`1-modelsetup_good_simple.py`](rubredoxin/1-Preparation/1-modelsetup_good_simple.py)
and
[`1-modelsetup_good_advanced.py`](rubredoxin/1-Preparation/1-modelsetup_good_advanced.py).
