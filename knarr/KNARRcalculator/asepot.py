import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from KNARRatom.utilities import Convert1To3, Convert3To1


def ASEPot(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()
    symbols = atoms.GetSymbols()
    pbc = atoms.GetPBC()
    cell = atoms.GetCell()
    template = calculator.GetTemplate()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = ASEWorker(template, ndim, rxyz[i * ndim:(i + 1) * ndim],
                                   symbols[0:ndim], pbc, cell)

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = ASEWorker(template, ndim, rxyz[val * ndim:(val + 1) * ndim],
                                   symbols[0:ndim], pbc, cell)

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None


def ASEWorker(template, ndim, rxyz,
              symbols, pbc, cell):
    # TODO: template can later be introduced to use different ASE calculators

    # Set up atoms object

    asecoords = Convert1To3(ndim, rxyz)
    asesymbols = []
    for i in range(0, ndim, 3):
        asesymbols.append(symbols[i])

    asecell = None
    if cell is not None:
        asecell = np.reshape(cell, (3,))
    ase_system = Atoms(positions=asecoords, symbols=asesymbols,
                       cell=asecell, pbc=pbc, calculator=None)

    # Set up calculator
    ase_system.set_calculator(EMT())

    # Compute energy and forces
    energy = ase_system.get_potential_energy()
    #print 'energy:'+str(energy)
    forces = ase_system.get_forces()
    forces = Convert3To1(ndim, forces)
    return forces, energy
