import numpy as np


def LEPSHO(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = LEPSHOWorker(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = LEPSHOWorker(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1

    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)
    return None

def LEPSHOGauss(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = LEPSHOGaussWorker(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = LEPSHOGaussWorker(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1

    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)
    return None

def LEPSHOWorker(rxyz):
    r = np.array([rxyz[0], rxyz[1]])
    ndim = 2
    F = np.zeros(shape=(ndim, 1))
    Ftmp, Etmp = SingleLEPSHO(r)
    energy = Etmp
    F[0] = Ftmp[0][0]
    F[1] = Ftmp[1][0]
    forces = np.zeros(shape=(3, 1))
    forces[0] = F[0]
    forces[1] = F[1]
    forces[2] = 0.0
    return forces, energy


def LEPSHOGaussWorker(rxyz):
    r = np.array([rxyz[0], rxyz[1]])
    ndim = 2
    F = np.zeros(shape=(ndim, 1))
    Ftmp, Etmp = SingleLEPSHO(r)
    energy = Etmp
    F[0] = Ftmp[0][0]
    F[1] = Ftmp[1][0]
    # =========================
    # add two gaussians
    # =========================
    A1 = 1.5
    x01 = 2.02083
    y01 = -0.172881
    sigmax1 = 0.1
    sigmay1 = 0.35
    A2 = 7.0
    x02 = 0.8
    y02 = 2.0
    sigmax2 = 0.447213
    sigmay2 = 1.195229
    gauss1 = A1 * np.exp(-(r[0] - x01) ** 2 / (2.0 * sigmax1 ** 2)) * np.exp(-(r[1] - y01) ** 2 / (2.0 * sigmay1 ** 2))
    gauss2 = A2 * np.exp(-(r[0] - x02) ** 2 / (2.0 * sigmax2 ** 2)) * np.exp(-(r[1] - y02) ** 2 / (2.0 * sigmay2 ** 2))
    dxgauss1 = ((r[0] - x01) / (sigmax1 ** 2)) * gauss1
    dxgauss2 = ((r[0] - x02) / (sigmax2 ** 2)) * gauss2
    dygauss1 = ((r[1] - y01) / (sigmay1 ** 2)) * gauss1
    dygauss2 = ((r[1] - y02) / (sigmay2 ** 2)) * gauss2

    energy = energy + gauss1 + gauss2
    forces = np.zeros(shape=(3, 1))
    forces[0] = F[0] + (dxgauss1 + dxgauss2)
    forces[1] = F[1] + (dygauss1 + dygauss2)
    forces[2] = 0.0
    return forces, energy


def SingleLEPSHO(r):
    r1 = r[0]  # -10.0  # THIS IS RAB
    r2 = r[1]  # -10.0  # THIS IS X
    alphaAB = 1.942
    alphaBC = 1.942
    alphaAC = 1.942
    ap1 = 1.05
    bp1 = 1.80
    cp1 = 1.05
    r0AB = 0.742
    r0BC = 0.742
    r0AC = 0.742
    dAB = 4.746
    dBC = 4.746
    dAC = 3.445
    l = 3.0 + r0AB
    # k = 0.09
    # c = 1.5
    k = 0.2025
    c = 1.154

    def djAB(r):
        return 0.25 * dAB * (-2.0 * alphaAB * np.exp(-2.0 * alphaAB * (r - r0AB)) + 6.0 * alphaAB * np.exp(
            -alphaAB * (r - r0AB)))

    def djBC(r):
        return 0.25 * dBC * (-2.0 * alphaBC * np.exp(-2.0 * alphaBC * (r - r0BC)) + 6.0 * alphaBC * np.exp(
            -alphaBC * (r - r0BC)))

    def dqAB(r):
        return 0.25 * dAB * (-6.0 * alphaAB * np.exp(-2.0 * alphaAB * (r - r0AB)) + 2.0 * alphaAB * np.exp(
            -alphaAB * (r - r0AB)))

    def dqBC(r):
        return 0.25 * dBC * (-6.0 * alphaBC * np.exp(-2.0 * alphaBC * (r - r0BC)) + 2.0 * alphaBC * np.exp(
            -alphaBC * (r - r0BC)))

    def jAB(r):
        return 0.25 * dAB * (np.exp(-2.0 * alphaAB * (r - r0AB)) - 6.0 * np.exp(-alphaAB * (r - r0AB)))

    def jAC(r):
        return 0.25 * dAC * (np.exp(-2.0 * alphaAC * (r - r0AC)) - 6.0 * np.exp(-alphaAC * (r - r0AC)))

    def jBC(r):
        return 0.25 * dBC * (np.exp(-2.0 * alphaBC * (r - r0BC)) - 6.0 * np.exp(-alphaBC * (r - r0BC)))

    def qAB(r):
        return 0.25 * dAB * (3 * np.exp(-2.0 * alphaAB * (r - r0AB)) - 2.0 * np.exp(-alphaAB * (r - r0AB)))

    def qAC(r):
        return 0.25 * dAC * (3 * np.exp(-2.0 * alphaAC * (r - r0AC)) - 2.0 * np.exp(-alphaAC * (r - r0AC)))

    def qBC(r):
        return 0.25 * dBC * (3 * np.exp(-2.0 * alphaBC * (r - r0BC)) - 2.0 * np.exp(-alphaBC * (r - r0BC)))

    grad = np.zeros(shape=(2, 1))
    # l = RAC and l-r1 = RAB
    ELEPS = qAB(r1) / ap1 + qBC(l - r1) / bp1 + qAC(r1 + (l - r1)) / cp1 - \
            np.sqrt((jAB(r1) / ap1) ** 2 + (jBC(l - r1) / bp1) ** 2 + (jAC(r1 + (l - r1)) / cp1) ** 2 \
                    - (jAB(r1) * jBC(l - r1)) / (ap1 * bp1) - (jBC(l - r1) * jAC(r1 + l - r1)) / (bp1 * cp1) - (
                            jAB(r1) * jAC(r1 + l - r1)) / (ap1 * cp1))
    # r2 = x
    # E = ELEPS + 2 * k * c ** 2 * (r1 - (0.5 * l - 1.3 * r2 / c)) ** 2  # HO addition to LEPS
    E = ELEPS + 2 * k * (r1 - (0.5 * l - r2 / c)) ** 2

    grad[0] = (dBC * (6 * alphaBC * np.exp(-2 * alphaBC * (l - r0BC - r1)) - 2 * alphaBC * np.exp(
        -alphaBC * (l - r0BC - r1)))) / (4 * bp1) + (dAB * (
            -6 * alphaAB * np.exp(-2 * alphaAB * (-r0AB + r1)) + 2 * alphaAB * np.exp(-alphaAB * (-r0AB + r1)))) / (
                      4 * ap1) - (-((dAC * dBC * (
            np.exp(-2 * alphaAC * (l - r0AC)) - 6 * np.exp(-alphaAC * (l - r0AC))) * (2 * alphaBC * np.exp(
        -2 * alphaBC * (l - r0BC - r1)) - 6 * alphaBC * np.exp(-alphaBC * (l - r0BC - r1)))) / (16 * bp1 * cp1)) + (
                                          dBC ** 2 * (np.exp(-2 * alphaBC * (l - r0BC - r1)) - 6 * np.exp(
                                      -alphaBC * (l - r0BC - r1))) * (2 * alphaBC * np.exp(
                                      -2 * alphaBC * (l - r0BC - r1)) - 6 * alphaBC * np.exp(
                                      -alphaBC * (l - r0BC - r1)))) / (8 * bp1 ** 2) - (dAB * dBC * (
            2 * alphaBC * np.exp(-2 * alphaBC * (l - r0BC - r1)) - 6 * alphaBC * np.exp(
        -alphaBC * (l - r0BC - r1))) * (np.exp(-2 * alphaAB * (-r0AB + r1)) - 6 * np.exp(
        -alphaAB * (-r0AB + r1)))) / (16 * ap1 * bp1) - (dAB * dAC * (
            np.exp(-2 * alphaAC * (l - r0AC)) - 6 * np.exp(-alphaAC * (l - r0AC))) * (-2 * alphaAB * np.exp(
        -2 * alphaAB * (-r0AB + r1)) + 6 * alphaAB * np.exp(-alphaAB * (-r0AB + r1)))) / (16 * ap1 * cp1) - (
                                          dAB * dBC * (np.exp(-2 * alphaBC * (l - r0BC - r1)) - 6 * np.exp(
                                      -alphaBC * (l - r0BC - r1))) * (-2 * alphaAB * np.exp(
                                      -2 * alphaAB * (-r0AB + r1)) + 6 * alphaAB * np.exp(
                                      -alphaAB * (-r0AB + r1)))) / (16 * ap1 * bp1) + (dAB ** 2 * (
            np.exp(-2 * alphaAB * (-r0AB + r1)) - 6 * np.exp(-alphaAB * (-r0AB + r1))) * (-2 * alphaAB * np.exp(
        -2 * alphaAB * (-r0AB + r1)) + 6 * alphaAB * np.exp(-alphaAB * (-r0AB + r1)))) / (8 * ap1 ** 2)) / (2 * np.sqrt(
        (dAC ** 2 * (np.exp(-2 * alphaAC * (l - r0AC)) - 6 * np.exp(-alphaAC * (l - r0AC))) ** 2) / (16 * cp1 ** 2) - (
                dAC * dBC * (np.exp(-2 * alphaAC * (l - r0AC)) - 6 * np.exp(-alphaAC * (l - r0AC))) * (
                np.exp(-2 * alphaBC * (l - r0BC - r1)) - 6 * np.exp(-alphaBC * (l - r0BC - r1)))) / (
                16 * bp1 * cp1) + (dBC ** 2 * (
                np.exp(-2 * alphaBC * (l - r0BC - r1)) - 6 * np.exp(-alphaBC * (l - r0BC - r1))) ** 2) / (
                16 * bp1 ** 2) - (
                dAB * dAC * (np.exp(-2 * alphaAC * (l - r0AC)) - 6 * np.exp(-alphaAC * (l - r0AC))) * (
                np.exp(-2 * alphaAB * (-r0AB + r1)) - 6 * np.exp(-alphaAB * (-r0AB + r1)))) / (
                16 * ap1 * cp1) - (
                dAB * dBC * (np.exp(-2 * alphaBC * (l - r0BC - r1)) - 6 * np.exp(-alphaBC * (l - r0BC - r1))) * (
                np.exp(-2 * alphaAB * (-r0AB + r1)) - 6 * np.exp(-alphaAB * (-r0AB + r1)))) / (
                16 * ap1 * bp1) + (
                dAB ** 2 * (np.exp(-2 * alphaAB * (-r0AB + r1)) - 6 * np.exp(-alphaAB * (-r0AB + r1))) ** 2) / (
                16 * ap1 ** 2)))
    grad[0] += 4 * k * (r1 - 0.5 * l + r2 / c)
    grad[1] = 4 * k * (r2 / c - 0.5 * l + r1) / c
    F = -grad
    return F, E

# def NewSingleLEPSHO(r):
#    r1 = r[0]
#    r2 = r[1]
#    alphaAB = 1.942
#    alphaBC = 1.942
#    alphaAC = 1.942
#    ap1 = 1.05
#    bp1 = 1.80
#    cp1 = 1.05
#    r0AB = 0.742
#    r0BC = 0.742
#    r0AC = 0.742
#    dAB = 4.746
#    dBC = 4.746
#    dAC = 3.445
#    l = 3.0 + r0AB
#    k = 0.2025
#    c = 1.154

#    def jAB(r):
#        return 0.25 * dAB * (np.exp(-2.0 * alphaAB * (r - r0AB)) - 6.0 * np.exp(-alphaAB * (r - r0AB)))

#    def jAC(r):
#        return 0.25 * dAC * (np.exp(-2.0 * alphaAC * (r - r0AC)) - 6.0 * np.exp(-alphaAC * (r - r0AC)))

#    def jBC(r):
#        return 0.25 * dBC * (np.exp(-2.0 * alphaBC * (r - r0BC)) - 6.0 * np.exp(-alphaBC * (r - r0BC)))

#    def qAB(r):
#        return 0.25 * dAB * (3 * np.exp(-2.0 * alphaAB * (r - r0AB)) - 2.0 * np.exp(-alphaAB * (r - r0AB)))

#    def qAC(r):
#        return 0.25 * dAC * (3 * np.exp(-2.0 * alphaAC * (r - r0AC)) - 2.0 * np.exp(-alphaAC * (r - r0AC)))

#    def qBC(r):
#        return 0.25 * dBC * (3 * np.exp(-2.0 * alphaBC * (r - r0BC)) - 2.0 * np.exp(-alphaBC * (r - r0BC)))

#    energy = qAB(r1) / ap1 + qBC(l-r1) / bp1 + qAC(r1 + l-r1) / cp1 - (
#            (jAB(r1) / ap1) ** 2 + (jBC(l-r1) / bp1) ** 2 + (jAC(r1 + l-r1) / cp1) ** 2 - jAB(r1) * jBC(l-r1) / (
#            ap1 * bp1) - jBC(l-r1) * jAC(r1 + l-r1) / (bp1 * cp1) - jAB(r1) * jAC(r1 + l-r1) / (ap1 * cp1)) ** 0.5
#    energy += 2*k*(r1-(0.5*l-r2/c))**2

#    forces = np.zeros(shape=(2,1))
#    return forces, energy
