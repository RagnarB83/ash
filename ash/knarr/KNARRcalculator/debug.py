import numpy as np

import KNARRsettings

def Debug(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = DebugWorker(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = DebugWorker(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1
    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)
    return None


def DebugWorker(rxyz):
    r = np.array([rxyz[0], rxyz[1]])
    ndim = 2
    F = np.zeros(shape=(ndim, 1))
    Ftmp, Etmp = SingleDebug(r)
    energy = Etmp
    forces = np.zeros(shape=(3, 1))
    forces[0] = Ftmp[0]
    forces[1] = Ftmp[1]
    forces[2] = 0.0
    return forces, energy

def SingleDebug(rxyz):
    x = rxyz[0]
    y = rxyz[1]

    E = x**2 + KNARRsettings.debug_a * y**2 + KNARRsettings.debug_b*x + KNARRsettings.debug_c * y + \
        KNARRsettings.debug_d * x**3 + KNARRsettings.debug_e*(x-y)**3
    gradient = np.zeros(shape=(2,1))
    gradient[0] = 2*x + KNARRsettings.debug_b + 3*KNARRsettings.debug_d*x**2 + KNARRsettings.debug_e*3*(x-y)**2
    gradient[1] = 2*y*KNARRsettings.debug_a+KNARRsettings.debug_c - KNARRsettings.debug_e*3*(x-y)**2
    F = -gradient

    return F, E