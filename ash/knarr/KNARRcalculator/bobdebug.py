import numpy as np

import KNARRsettings

def BobDebug(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = BobDebugWorker(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = BobDebugWorker(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1
    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)
    return None


def BobDebugWorker(rxyz):
    r = np.array([rxyz[0], rxyz[1]])
    ndim = 2
    F = np.zeros(shape=(ndim, 1))
    Ftmp, Etmp = SingleBobDebug(r)
    energy = Etmp
    forces = np.zeros(shape=(3, 1))
    forces[0] = Ftmp[0]
    forces[1] = Ftmp[1]
    forces[2] = 0.0
    return forces, energy

def SingleBobDebug(rxyz):
    x = rxyz[0]
    y = rxyz[1]

    A1 = -0.4
    x01 = 0.5
    y01 = -0.5
    sigma_x1 = 0.4
    sigma_y1 = 0.8

    A2 = -0.6
    x02 = 3.0
    y02 = 2.6
    sigma_x2 = 1.2
    sigma_y2 = 0.4

    A3 = 2.5
    x03 = 2.5
    y03 = -0.5
    sigma_x3 = 0.7
    sigma_y3 = 0.7

    A4 = 1.1
    x04 = -2.0
    y04 = 4.0
    sigma_x4 = 0.7
    sigma_y4 = 0.6

    gauss1 = A1 * np.exp(-(x - x01) ** 2 / (2 * sigma_x1) ** 2) * np.exp(-(y - y01) ** 2 / (2 * sigma_y1) ** 2)
    gauss2 = A2 * np.exp(-(x - x02) ** 2 / (2 * sigma_x2) ** 2) * np.exp(-(y - y02) ** 2 / (2 * sigma_y2) ** 2)
    gauss3 = A3 * np.exp(-(x - x03) ** 2 / (2 * sigma_x3) ** 2) * np.exp(-(y - y03) ** 2 / (2 * sigma_y3) ** 2)
    gauss4 = A4 * np.exp(-(x - x04) ** 2 / (2 * sigma_x4) ** 2) * np.exp(-(y - y04) ** 2 / (2 * sigma_y4) ** 2)

    E = gauss1 + gauss2 + gauss3 + gauss4

    forces = np.zeros(shape=(2, 1))

    gauss1x = (A1 * np.exp(-((-x01 + x) ** 2 / (4 * sigma_x1 ** 2)) - (-y01 + y) ** 2 / (4 * sigma_y1 ** 2)) * (
                -x01 + x)) / (2 * sigma_x1 ** 2)
    gauss1y = (A1 * np.exp(-((-x01 + x) ** 2 / (4 * sigma_x1 ** 2)) - (-y01 + y) ** 2 / (4 * sigma_y1 ** 2)) * (
                -y01 + y)) / (2 * sigma_y1 ** 2)

    gauss2x = (A2 * np.exp(-((-x02 + x) ** 2 / (4 * sigma_x2 ** 2)) - (-y02 + y) ** 2 / (4 * sigma_y2 ** 2)) * (
                -x02 + x)) / (2 * sigma_x2 ** 2)
    gauss2y = (A2 * np.exp(-((-x02 + x) ** 2 / (4 * sigma_x2 ** 2)) - (-y02 + y) ** 2 / (4 * sigma_y2 ** 2)) * (
                -y02 + y)) / (2 * sigma_y2 ** 2)

    gauss3x = (A3 * np.exp(-((-x03 + x) ** 2 / (4 * sigma_x3 ** 2)) - (-y03 + y) ** 2 / (4 * sigma_y3 ** 2)) * (
                -x03 + x)) / (2 * sigma_x3 ** 2)
    gauss3y = (A3 * np.exp(-((-x03 + x) ** 2 / (4 * sigma_x3 ** 2)) - (-y03 + y) ** 2 / (4 * sigma_y3 ** 2)) * (
                -y03 + y)) / (2 * sigma_y3 ** 2)

    gauss4x = (A4 * np.exp(-((-x04 + x) ** 2 / (4 * sigma_x4 ** 2)) - (-y04 + y) ** 2 / (4 * sigma_y4 ** 2)) * (
                -x04 + x)) / (2 * sigma_x4 ** 2)
    gauss4y = (A4 * np.exp(-((-x04 + x) ** 2 / (4 * sigma_x4 ** 2)) - (-y04 + y) ** 2 / (4 * sigma_y4 ** 2)) * (
                -y04 + y)) / (2 * sigma_y4 ** 2)

    forces[0] = gauss1x + gauss2x + gauss3x + gauss4x
    forces[1] = gauss1y + gauss2y + gauss3y + gauss4y

    return forces, E