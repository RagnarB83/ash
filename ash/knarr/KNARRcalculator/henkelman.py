import numpy as np
import KNARRsettings

def Henkelman10D(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    assert ndim == 15

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = HenkelmanWorker10D(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = HenkelmanWorker10D(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1

    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None

def Henkelman100D(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    assert ndim == 150

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = HenkelmanWorker100D(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = HenkelmanWorker100D(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1

    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None


def Henkelman20D(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    assert ndim == 30

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = HenkelmanWorker20D(rxyz[i * ndim:(i + 1) * ndim])
            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = HenkelmanWorker20D(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1

    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None

def Henkelman(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = HenkelmanWorker(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = HenkelmanWorker(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1
    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None


def HenkelmanWorker(rxyz):
    r = np.array([rxyz[0], rxyz[1]])
    ndim = 2
    Etmp, F = V_2D(r)
    energy = Etmp
    forces = np.zeros(shape=(3, 1))
    forces[0] = F[0]
    forces[1] = F[1]
    forces[2] = 0.0
    return forces, energy


def HenkelmanWorker10D(rxyz):
    ndim = len(rxyz)

    assert ndim == 15

    rnew = np.zeros(shape=(10, 1))
    ind = 0
    # cut out z-dimension
    for i in range(0, ndim, 3):
        rnew[ind] = rxyz[i]
        ind += 1
        rnew[ind] = rxyz[i + 1]
        ind += 1

    Etmp, F = V_10D(rnew) # perform computation
    energy = Etmp
    forces = np.zeros(shape=(ndim, 1))

    ind = 0
    for i in range(0, ndim, 3):
        forces[i] = F[ind]
        ind += 1
        forces[i+1] = F[ind]
        ind += 1
        forces[i+2] = 0.0

    return forces, energy

def HenkelmanWorker100D(rxyz):
    ndim = len(rxyz)

    assert ndim == 150

    rnew = np.zeros(shape=(100, 1))
    ind = 0
    # cut out z-dimension
    for i in range(0, ndim, 3):
        rnew[ind] = rxyz[i]
        ind += 1
        rnew[ind] = rxyz[i + 1]
        ind += 1

    Etmp, F = V_100D(rnew) # perform computation
    energy = Etmp
    forces = np.zeros(shape=(ndim, 1))

    ind = 0
    for i in range(0, ndim, 3):
        forces[i] = F[ind]
        ind += 1
        forces[i+1] = F[ind]
        ind += 1
        forces[i+2] = 0.0

    return forces, energy

def HenkelmanWorker20D(rxyz):
    ndim = len(rxyz)

    assert ndim == 30

    rnew = np.zeros(shape=(20, 1))
    ind = 0
    # cut out z-dimension
    for i in range(0, ndim, 3):
        rnew[ind] = rxyz[i]
        ind += 1
        rnew[ind] = rxyz[i + 1]
        ind += 1

    Etmp, F = V_20D(rnew) # perform computation
    energy = Etmp
    
    forces = np.zeros(shape=(ndim, 1))

    ind = 0
    for i in range(0, ndim, 3):
        forces[i] = F[ind]
        ind += 1
        forces[i+1] = F[ind]
        ind += 1
        forces[i+2] = 0.0


    return forces, energy


def V_2D(r):
    boost = KNARRsettings.boost
    KNARRsettings.boost_time = 1.0
    KNARRsettings.boosted = False
    T = KNARRsettings.boost_temp
    x, y = r[0], r[1]
    V0 = 1.203
    V = np.cos(2 * np.pi * x) * (1 + 4 * y) + 0.5 * (2 * np.pi * y) ** 2 + V0
    if V <= boost:
        KNARRsettings.boosted = True
        Vold = V
        V = boost
        KNARRsettings.boost_time = np.exp( (V-Vold) / 0.2) #NOTE HERE THE TEMPERATURE FOR BOOST IS HARD-CODED AS 0.2 KbT 
        print('boost (%6.4f %6.4f) time:%6.4f' % (V, Vold, KNARRsettings.boost_time))
        force = np.zeros(shape=(2,1))
    else:
        force = np.zeros(shape=(2, 1))
        force[0] = 2.0 * np.pi * (4.0 * y + 1.0) * np.sin(2 * np.pi * x)
        force[1] = -4 * np.cos(2.0 * np.pi * x) - 4.0 * np.pi * np.pi * y
    return V, force


def V_10D(r):
    ndim = len(r)
    assert ndim == 10
    boost = KNARRsettings.boost
    KNARRsettings.boost_time = 1.0
    KNARRsettings.boosted = False
    T = KNARRsettings.boost_temp

    E = 0.0
    force = np.zeros(shape=(ndim, 1))

    for i in range(0, ndim, 2):
        x, y = r[i], r[i + 1]
        V0 = 1.203
        E = E + np.cos(2 * np.pi * x) * (1 + 4 * y) + 0.5 * (2 * np.pi * y) ** 2 + V0
        force[i] = 2.0 * np.pi * (4.0 * y + 1.0) * np.sin(2 * np.pi * x)
        force[i + 1] = -4 * np.cos(2.0 * np.pi * x) - 4.0 * np.pi * np.pi * y

    if E <= boost:
        KNARRsettings.boosted = True
        Vold = E
        V = boost
        KNARRsettings.boost_time = np.exp( (V-Vold) / 0.2) #NOTE HERE THE TEMP. FOR BOOST IS HARD-CODED AS 0.2 Kb
        force = np.zeros(shape=(10,1))
        #print 'boost (%6.4f %6.4f) time:%6.4f' % (V, Vold, KNARRsettings.boost_time)


    return E, force


def V_100D(r):
    ndim = len(r)
    assert ndim == 100

    E = 0.0
    force = np.zeros(shape=(ndim, 1))
    for i in range(0, ndim, 2):
        x, y = r[i], r[i + 1]
        V0 = 1.203
        E = E + np.cos(2 * np.pi * x) * (1 + 4 * y) + 0.5 * (2 * np.pi * y) ** 2 + V0
        force[i] = 2.0 * np.pi * (4.0 * y + 1.0) * np.sin(2 * np.pi * x)
        force[i + 1] = -4 * np.cos(2.0 * np.pi * x) - 4.0 * np.pi * np.pi * y

    return E, force

def V_20D(r):
    ndim = len(r)
    assert ndim == 20

    E = 0.0
    force = np.zeros(shape=(ndim, 1))
    for i in range(0, ndim, 2):
        x, y = r[i], r[i + 1]
        V0 = 1.203
        E = E + np.cos(2 * np.pi * x) * (1 + 4 * y) + 0.5 * (2 * np.pi * y) ** 2 + V0
        force[i] = 2.0 * np.pi * (4.0 * y + 1.0) * np.sin(2 * np.pi * x)
        force[i + 1] = -4 * np.cos(2.0 * np.pi * x) - 4.0 * np.pi * np.pi * y

    return E, force


if __name__ == "__main__":
    r_saddle = np.array([0.5, 0.1013, 0.0, 0.0, -0.1013, 0.0, 0.5, 0.1013, 0.0, 0.5, 0.1013, 0.0, 0.5, 0.1013, 0.0])
    r_min = np.array([0.5, 0.1013, 0.0, 0.5, 0.1013, 0.0, 0.5, 0.1013, 0.0, 0.5, 0.1013, 0.0, 0.5, 0.1013, 0.0])
    r_custom = np.array([0.38, 0.12, 0.0, 0.5, 0.1013, 0.0, 0.5, 0.1013, 0.0, 0.5, 0.1013, 0.0, 0.5, 0.1013, 0.0])
    r = r_custom
    forces, energy = HenkelmanWorker10D(r)
    print('Force = ', forces)
    print('Energy = ', energy)
