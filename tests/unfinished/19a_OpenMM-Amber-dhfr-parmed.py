from ash import *

forcefielddir="/Users/bjornsson/ownCloud/ASH-tests/testsuite/OpenMM-files-for-tests/dhfr/amber/"
inpcrdfile=forcefielddir+"dhfr.pbc.rst7"
prmtopfile=forcefielddir+"dhfr.pbc.parm7"


frag=Fragment(amber_prmtopfile=prmtopfile, amber_inpcrdfile=inpcrdfile, conncalc=False)



#Periodic OpenMM on DHFR system using Amber files
# Reference data from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549999/ 
openmmobject = OpenMMTheory(Amberfiles=True, amberprmtopfile=prmtopfile,
    periodic=True, do_energy_decomposition=True, use_parmed=True,
    applyconstraints=False, dispersion_correction=True, periodic_nonbonded_cutoff=10)

energy = Singlepoint(theory=openmmobject, fragment=frag)
print("openmmobject.energy_components:", openmmobject.energy_components)


#Comparing energy components to reference values (kJ/mol):
bondthreshold=2.5
anglethreshold=0.1
dihedralthreshold=0.1
nonbondedthreshold=3
totalthreshold=3.0

#Amber PBC program values in paper
ref_energy_components_amber_paper={'Bond': 613.34, 'Angle':1611.89, 'Dihedral':8844.32, 'Nonbonded':-433365.84, 'Total':-422296.30}
#OpenMM values in paper
ref_energy_components_openmm_paper={'Bond': 613.34, 'Angle':1611.89, 'Dihedral':8844.32, 'Nonbonded':-433410.31, 'Total':-422340.77}
#OpenMM recalculated values (calculated using OpenMM 7.4.1 in June 2021). Some minor deviation for angle/dihedral that cancels out
ref_energy_components_openmm_RB={'Bond': 613.339388011621278, 'Angle':1611.889656353512146, 'Dihedral':8844.317091243306856, 'Nonbonded':-433189.66784224110128, 'Total':-422120.1217066}


#NOTE: The Bond term we calculate via ASH-OpenMM is 611.05 without parmed and 613.34 with parmed. 611.05 agrees perfectly with Amber non-PBC values in paper


#Which set of reference values to use below
chosen_ref=ref_energy_components_amber_paper

print("Reference energy components table:", chosen_ref)

print("Reference MM Bond energy:", chosen_ref['Bond'])
print("Calculated MM Bond energy:", openmmobject.energy_components['Bond']._value)
assert abs(float(openmmobject.energy_components['Bond']._value)-chosen_ref['Bond']) < bondthreshold, "MM Bond Energy-error above threshold"

#Angle energy is sum of Angle+Urey-Bradley+Impropers+CMAP
print("Reference MM Angle energy:", chosen_ref['Angle'])
print("Calculated MM Angle energy:", openmmobject.energy_components['Angle']._value)
assert abs(float(openmmobject.energy_components['Angle']._value)-chosen_ref['Angle']) < anglethreshold, "MM Angle Energy-error above threshold"

print("Reference MM Dihedral energy:", chosen_ref['Dihedral'])
print("Calculated MM Dihedral energy:", openmmobject.energy_components['Dihedrals']._value)
assert abs(float(openmmobject.energy_components['Dihedrals']._value)-chosen_ref['Dihedral']) < dihedralthreshold, "MM Dihedral Energy-error above threshold"

print("Reference MM Nonbonded energy:", chosen_ref['Nonbonded'])
print("Calculated MM Nonbonded energy:", openmmobject.energy_components['Nonbonded']._value)
assert abs(float(openmmobject.energy_components['Nonbonded']._value)-chosen_ref['Nonbonded']) < nonbondedthreshold, "MM Nonbonded Energy-error above threshold"

print("Reference MM Total energy:", chosen_ref['Total'])
print("Calculated MM Sum Total energy:", openmmobject.energy_components['Sum'])
assert abs(openmmobject.energy_components['Sum']-chosen_ref['Total']) < totalthreshold, "MM Total Energy-error above threshold"
print("")
print("All energies within the chosen threshold.")
print("Bond threshold was: {} kJ/mol.".format(bondthreshold))
print("Angle threshold was: {} kJ/mol.".format(anglethreshold))
print("Dihedral threshold was: {} kJ/mol.".format(dihedralthreshold))
print("Nonbonded threshold was: {} kJ/mol.".format(nonbondedthreshold))
print("Total-energy threshold was: {} kJ/mol.".format(totalthreshold))

