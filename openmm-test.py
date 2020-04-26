#openMM import charmm xml file

forcefield = app.ForceField(xml)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)

forcefield = ForceField('charmm/toppar_all36_prot_model.xml')

#--------------------------------

CHARMM
import and simulate

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout, exit, stderr

psf = CharmmPsfFile('input.psf')
pdb = PDBFile('input.pdb')
params = CharmmParameterSet('charmm22.rtf', 'charmm22.prm')
system = psf.createSystem(params, nonbondedMethod=NoCutoff,
                          nonbondedCutoff=1 * nanometer, constraints=HBonds)
integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
simulation = Simulation(psf.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('output.pdb', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
                                              potentialEnergy=True, temperature=True))
simulation.step(10000)

#------------------

#CHARMM
#import and do
#energy + gradient

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout, exit, stderr

psf = CharmmPsfFile('input.psf')
pdb = PDBFile('input.pdb')
params = CharmmParameterSet('charmm22.rtf', 'charmm22.prm')
system = psf.createSystem(params, nonbondedMethod=NoCutoff,
                          nonbondedCutoff=1 * nanometer, constraints=HBonds)

integrator = LangevinIntegrator(100 * kelvin, 1 / picosecond, 0.002 * picoseconds)
simulation = Simulation(po_top, omm_sys, integrator, platform=Platform.getPlatformByName('Reference'))

# Create positions as desired by OpenMM; multiplication by .1 is to transform units from A to nm
coords = []
for i in range(len(mol.xyz)):
    coords.append(Vec3(mol.xyz[i][0] * .1, mol.xyz[i][1] * .1, mol.xyz[i][2] * .1))
positions = coords * nanometer
simulation.context.setPositions(positions)

state = simulation.context.getState(getEnergy=True, getForces=False)
# forces = state.getForces(asNumpy=True)
energy = state.getPotentialEnergy()

#----------------------------------------
#OTHer
#via
#geometric
#code
#way

forcefield = app.ForceField(xml)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)

integrator = mm.VerletIntegrator(1.0 * u.femtoseconds)
platform = mm.Platform.getPlatformByName('Reference')
self.simulation = app.Simulation(pdb.topology, system, integrator, platform)

self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
pos = [Vec3(self.M.xyzs[0][i, 0] / 10, self.M.xyzs[0][i, 1] / 10, self.M.xyzs[0][i, 2] / 10) for i in
       range(self.M.na)] * u.nanometer
self.simulation.context.setPositions(pos)

state = self.simulation.context.getState(getEnergy=True, getForces=True)
energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / eqcgmx
gradient = state.getForces(asNumpy=True).flatten() / fqcgmx








