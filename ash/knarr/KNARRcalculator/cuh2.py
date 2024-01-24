import numpy as np
import eam


def CuH2Worker(X):
    cell = X[0]
    rxyz = X[1]
    forces, energy = eam.force_eam(cell, rxyz)
    forces = np.reshape(forces, (len(rxyz), 1))
    return forces, energy


def EAM(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()
    cell = atoms.GetCell()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            X = (cell, rxyz[i * ndim:(i + 1) * ndim])
            ftmp, etmp = CuH2Worker(X)

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            X = (cell, rxyz[val * ndim:(val + 1) * ndim])
            ftmp, etmp = CuH2Worker(X)

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1

    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None
