from ash import *

forcefielddir="/Users/bjornsson/ownCloud/ASH-tests/testsuite/OpenMM-files-for-tests/2koc/gromacs/"
topfile=forcefielddir+"2koc.pbc.top"
grofile=forcefielddir+"2koc.pbc.gro"
gromacstopdir="/home/bjornsson/gromacs-2018.3-install/share/gromacs/top"

frag=Fragment(grofile=grofile)

#Periodic OpenMM on 2KOC system using GROMACS files:
# Reference data from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549999/ 
#Periodic cell info from GRO file
#Long-range dispersion used. Needed
#Nonbonded cutoff: 9 Å. Used in reference scripts
#Ewald error tolerance: 5e-5 . Used in reference scripts. Needs to be this magnitude at least
#Constraints applied. Needed to get reference energies right (bonded and nonbonded). Unknown whether needed in general
openmmobject_parmed = OpenMMTheory(GROMACSfiles=True, gromacstopfile=topfile, grofile=grofile, gromacstopdir=gromacstopdir,
               periodic=True, applyconstraints=True, use_parmed=True, do_energy_decomposition=True, dispersion_correction=True, periodic_nonbonded_cutoff=9,
                ewalderrortolerance=5e-5)

result = Singlepoint(theory=openmmobject_parmed, fragment=frag)
energy_parmed = result.energy

#Comparing energy components to reference values (kJ/mol):
threshold=0.1

#CHARMM program values in paper
ref_energy_components_gromacs_paper={'Bond': 7976.96, 'Angle':277.21, 'Dihedral':1416.77, 'Nonbonded':-235817.06, 'Total':-226146.12}
#OpenMM values in paper
ref_energy_components_openmm_paper={'Bond': 7976.95, 'Angle':277.21, 'Dihedral':1416.76, 'Nonbonded':-235793.81, 'Total':-226122.89}
#OpenMM recalculated values (calculated using OpenMM 7.4.1 in June 2021). Some minor deviation for angle/dihedral that cancels out
ref_energy_components_openmm_RB={'Bond': 7976.951151846831, 'Angle':277.2072582794935, 'Dihedral':1416.7618701357062, 'Nonbonded':-235793.82982554947, 'Total':-226122.90954528746}

#Which set of reference values to use below
chosen_ref=ref_energy_components_openmm_RB

print("Reference energy components table:", chosen_ref)

print("Reference MM Bond energy:", chosen_ref['Bond'])
print("Calculated MM Bond energy:", openmmobject_parmed.energy_components['Bond']._value)
assert abs(float(openmmobject_parmed.energy_components['Bond']._value)-chosen_ref['Bond']) < threshold, "MM Bond Energy-error above threshold"

#Angle energy is sum of Angle+Urey-Bradley+Impropers+CMAP
print("Reference MM Angle energy:", chosen_ref['Angle'])
print("Calculated MM Angle energy:", openmmobject_parmed.energy_components['Angle'])
assert abs(float(openmmobject_parmed.energy_components['Angle']._value)-chosen_ref['Angle']) < threshold, "MM Angle Energy-error above threshold"

print("Reference MM Dihedral energy:", chosen_ref['Dihedral'])
print("Calculated MM Dihedral energy:", openmmobject_parmed.energy_components['Dihedrals']._value)
assert abs(float(openmmobject_parmed.energy_components['Dihedrals']._value)-chosen_ref['Dihedral']) < threshold, "MM Dihedral Energy-error above threshold"

print("Reference MM Nonbonded energy:", chosen_ref['Nonbonded'])
print("Calculated MM Nonbonded energy:", openmmobject_parmed.energy_components['Nonbonded']._value)
assert abs(float(openmmobject_parmed.energy_components['Nonbonded']._value)-chosen_ref['Nonbonded']) < threshold, "MM Nonbonded Energy-error above threshold"

print("Reference MM Total energy:", chosen_ref['Total'])
print("Calculated MM Sum Total energy:", openmmobject_parmed.energy_components['Sum'])
assert abs(openmmobject_parmed.energy_components['Sum']-chosen_ref['Total']) < threshold, "MM Total Energy-error above threshold"
print("")
print("All energies within a threshold of {} kJ/mol.".format(threshold))
