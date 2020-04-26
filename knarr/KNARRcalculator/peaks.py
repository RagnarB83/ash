import numpy as np


def Peaks(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = PeaksWorker(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = PeaksWorker(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1
    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)
    return None


def SinglePeaks(rxyz):
    x = rxyz[0]
    y = rxyz[1]

    E = 3.0 * (1.0 - x) ** 2 * np.exp(-x ** 2 - (y + 1) ** 2) - 10.0 * (x / 5.0 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - (1 / 3.0) * np.exp(-(x + 1) ** 2 - y ** 2)

    gradient = np.zeros(shape=(2, 1))
    gradient[0] = -6 * np.exp(-x ** 2 - (1 + y) ** 2) * (1 - x) - 6 * np.exp(-x ** 2 - (1 + y) ** 2) * (
            1 - x) ** 2 * x + 2 / 3 * np.exp(-(1 + x) ** 2 - y ** 2) * (1 + x) - 10 * np.exp(-x ** 2 - y ** 2) * (
                          1 / 5 - 3 * x ** 2) + 20 * np.exp(-x ** 2 - y ** 2) * x * (x / 5 - x ** 3 - y ** 5)

    gradient[1] = 2 / 3 * np.exp(-(1 + x) ** 2 - y ** 2) * y + 50 * np.exp(-x ** 2 - y ** 2) * y ** 4 - 6 * np.exp(
        -x ** 2 - (1 + y) ** 2) * (1 - x) ** 2 * (1 + y) + 20 * np.exp(-x ** 2 - y ** 2) * y * (x / 5 - x ** 3 - y ** 5)

    F = -gradient

    return F, E


def PeaksWorker(rxyz):
    r = np.array([rxyz[0], rxyz[1]])
    ndim = 2
    F = np.zeros(shape=(ndim, 1))
    Ftmp, Etmp = SinglePeaks(r)
    energy = Etmp
    F[0] = Ftmp[0][0]
    F[1] = Ftmp[1][0]
    forces = np.zeros(shape=(3, 1))
    forces[0] = F[0]
    forces[1] = F[1]
    forces[2] = 0.0
    return forces, energy
