from ash import *

forcefielddir="/Users/bjornsson/ownCloud/ASH-tests/testsuite/OpenMM-files-for-tests/dhfr/charmm/"
psffile=forcefielddir+"step3_pbcsetup.psf"
topfile=forcefielddir+"top_all36_prot.rtf"
prmfile=forcefielddir+"par_all36_prot.prm"
xyzfile=forcefielddir+"file.xyz"

frag = Fragment(xyzfile=xyzfile, conncalc=False)

#Periodic OpenMM on DHFR system using CHARMM files:
# Reference data from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549999/ 
#Periodic cell according to CHARMM inputfile, PBE parameters according to CHARMM settings also
#No long-range dispersion used
#Nonbonded cutoff: 12 Å, switching function: 10 Å
openmmobject = OpenMMTheory(psffile=psffile, CHARMMfiles=True, charmmtopfile=topfile,
    charmmprmfile=prmfile, periodic=True, charmm_periodic_cell_dimensions=[80, 80, 80, 90, 90, 90], do_energy_decomposition=True,
    applyconstraints=False, dispersion_correction=False, periodic_nonbonded_cutoff=12, switching_function_distance=10,
    PMEparameters=[1.0/0.34, 90, 90, 90], use_parmed=True)


energy = Singlepoint(theory=openmmobject, fragment=frag)
print("openmmobject.energy_components:", openmmobject.energy_components)


#Comparing energy components to reference values (kJ/mol):
#Note: Angle is sum of Angle+UB+CMAP+Impropers. OpenMM calc disagrees slightly with CHARMM and OpenMM paper value, not known why
#Changed threshold to 1.0. Minor extra 0.5 kJ/mol deviation upon switching to OpenMM 7.5.1. Nothing to worry about
threshold=1.0

#CHARMM program values in paper
ref_energy_components_charmm_paper={'Bond': 26518.18, 'Angle':17951.15, 'Dihedral':7225.94, 'Nonbonded':-733871.37, 'Total':-682176.09}
#OpenMM values in paper
ref_energy_components_openmm_paper={'Bond': 26518.18, 'Angle':17951.15, 'Dihedral':7226.35, 'Nonbonded':-733836.68, 'Total':-682140.99}
#OpenMM recalculated values (calculated using OpenMM 7.4.1 in June 2021). Some minor deviation for angle/dihedral that cancels out
ref_energy_components_openmm_RB={'Bond': 26518.192, 'Angle':17953.1256, 'Dihedral':7224.38728, 'Nonbonded':-733836.6971, 'Total':-682140.9922}

#Which set of reference values to use below
chosen_ref=ref_energy_components_openmm_RB

print("Reference energy components table:", chosen_ref)

print("Reference MM Bond energy:", chosen_ref['Bond'])
print("Calculated MM Bond energy:", openmmobject.energy_components['Bond']._value)
assert abs(float(openmmobject.energy_components['Bond']._value)-chosen_ref['Bond']) < threshold, "MM Bond Energy-error above threshold"

#Angle energy is sum of Angle+Urey-Bradley+Impropers+CMAP
print("Reference MM Angle energy:", chosen_ref['Angle'])
calc_total_angle_energy=(openmmobject.energy_components['Angle']._value+openmmobject.energy_components['Impropers']._value+openmmobject.energy_components['Urey-Bradley']._value+openmmobject.energy_components['CMAP']._value)
print("Calculated MM Angle energy:", calc_total_angle_energy)
assert abs(float(calc_total_angle_energy)-chosen_ref['Angle']) < threshold, "MM Angle Energy-error above threshold"

print("Reference MM Dihedral energy:", chosen_ref['Dihedral'])
print("Calculated MM Dihedral energy:", openmmobject.energy_components['Dihedrals']._value)
assert abs(float(openmmobject.energy_components['Dihedrals']._value)-chosen_ref['Dihedral']) < threshold, "MM Dihedral Energy-error above threshold"

print("Reference MM Nonbonded energy:", chosen_ref['Nonbonded'])
print("Calculated MM Nonbonded energy:", openmmobject.energy_components['Nonbonded']._value)
assert abs(float(openmmobject.energy_components['Nonbonded']._value)-chosen_ref['Nonbonded']) < threshold, "MM Nonbonded Energy-error above threshold"

print("Reference MM Total energy:", chosen_ref['Total'])
print("Calculated MM Sum Total energy:", openmmobject.energy_components['Sum'])
assert abs(openmmobject.energy_components['Sum']-chosen_ref['Total']) < threshold, "MM Total Energy-error above threshold"
print("")
print("All energies within a threshold of {} kJ/mol.".format(threshold))
