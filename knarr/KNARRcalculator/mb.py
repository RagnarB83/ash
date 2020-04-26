import numpy as np


def MBG(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = MullerBrownGaussWorker(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = MullerBrownGaussWorker(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1
    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None

def MB(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = MullerBrownWorker(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = MullerBrownWorker(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1
    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None


def MullerBrown2(rxyz):

    A = [-0.200, -0.100, -0.170, 0.015]
    a = [-1.0, -1.0, -6.5, 0.7]
    b = [0.0, 0.0,  11.0, 0.6]
    c = [-10.0, -10.0, -6.5, 0.7]
    x0 = [1.0, 0.0, -0.5, -1.0]
    y0 = [0.0, 0.5, 1.5, 1.0]

    E = 0.0
    gradient = np.zeros(shape=(2,1))
    for i in range(4):
        term = A[i] * np.exp(a[i] * (rxyz[0] - x0[i])**2
                         + b[i] * (rxyz[0] - x0[i]) * (rxyz[1] - y0[i]) + c[i] * (rxyz[1] - y0[i])**2)
        E = E + term
        gradient[0] += term * (2.0 * a[i] * (rxyz[0] - x0[i]) + b[i] * (rxyz[1] - y0[i]))
        gradient[1] += term * (b[i] * (rxyz[0] - x0[i]) + 2.0 * c[i] * (rxyz[1] - y0[i]))

    F = -gradient
    return F, E


def MullerBrown(ndim, nim, x):
    E = np.zeros(shape=(nim, 1))
    G = np.zeros(shape=(ndim * nim, 1))

    c1 = -0.2
    c2 = -0.1
    c3 = -0.17
    c4 = 0.015
    F1 = np.array([[-2.0, 0.0], [0.0, -20.0]])
    F2 = np.array([[-2.0, 0.0], [0.0, -20.0]])
    F3 = np.array([[-13.0, 11.0], [11.0, -13.0]])
    F4 = np.array([[7.0 / 5.0, 3.0 / 5.0], [3.0 / 5.0, 7.0 / 5.0]])
    q1 = np.array([[1.0, 0.0]])
    q1 = q1.T
    q2 = np.array([[0.0, 0.5]])
    q2 = q2.T
    q3 = np.array([[-0.5, 1.5]])
    q3 = q3.T
    q4 = np.array([[-1.0, 1.0]])
    q4 = q4.T
    ind = 0
    for i in range(0, nim):
        Rtmp = x[ndim * i:(i + 1) * ndim]
        R = np.reshape(Rtmp, (2, 1))
        E1 = c1 * np.exp(0.5 * np.dot(np.dot((R - q1).T, F1), (R - q1)))
        E2 = c2 * np.exp(0.5 * np.dot(np.dot((R - q2).T, F2), (R - q2)))
        E3 = c3 * np.exp(0.5 * np.dot(np.dot((R - q3).T, F3), (R - q3)))
        E4 = c4 * np.exp(0.5 * np.dot(np.dot((R - q4).T, F4), (R - q4)))
        E[i] = E1 + E2 + E3 + E4
        G1_1 = E1 * np.dot(np.array([[F1[0, 0], (F1[0, 1] + F1[1, 0]) / 2]]), R - q1)
        G1_2 = E1 * np.dot(np.array([[(F1[0, 1] + F1[1, 0]) / 2, F1[1, 1]]]), R - q1)
        G2_1 = E2 * np.dot(np.array([[F2[0, 0], (F2[0, 1] + F2[1, 0]) / 2]]), R - q2)
        G2_2 = E2 * np.dot(np.array([[(F2[0, 1] + F2[1, 0]) / 2, F2[1, 1]]]), R - q2)
        G3_1 = E3 * np.dot(np.array([[F3[0, 0], (F3[0, 1] + F3[1, 0]) / 2]]), R - q3)
        G3_2 = E3 * np.dot(np.array([[(F3[0, 1] + F3[1, 0]) / 2, F3[1, 1]]]), R - q3)
        G4_1 = E4 * np.dot(np.array([[F4[0, 0], (F4[0, 1] + F4[1, 0]) / 2]]), R - q4)
        G4_2 = E4 * np.dot(np.array([[(F4[0, 1] + F4[1, 0]) / 2, F4[1, 1]]]), R - q4)
        G[ind] = G1_1 + G2_1 + G3_1 + G4_1
        ind = ind + 1
        G[ind] = G1_2 + G2_2 + G3_2 + G4_2
        ind = ind + 1

    F = -1.0 * G.copy()
    return F, E


def MullerBrownWorker(rxyz):
    r = np.array([rxyz[0], rxyz[1]])
    ndim = 2
    F = np.zeros(shape=(ndim, 1))
    Ftmp, Etmp = MullerBrown2(r)
    energy = Etmp
    F[0] = Ftmp[0][0]
    F[1] = Ftmp[1][0]
    forces = np.zeros(shape=(3, 1))
    forces[0] = F[0]
    forces[1] = F[1]
    forces[2] = 0.0
    return forces, energy

def MullerBrownGaussWorker(rxyz):
    # ---------------------------------------------------------------------------------
    # Remove third dimension
    # ---------------------------------------------------------------------------------
    R = np.reshape([rxyz[0], rxyz[1]], (2,1))
    ndim = 2
    # ---------------------------------------------------------------------------------
    # Start calculation
    # ---------------------------------------------------------------------------------
    # MAGNITUDE
    A1 = 0.13
    A2 = 0.1
    A3 = 0.065

    # CENTERED AT
    bx1 = 1.0
    by1 = 0.25
    bx2 = 0.6
    by2 = 0.0
    bx3 = 0.8
    by3 = 0.4

    # THE WIDTH
    cx = 0.2
    cy = 0.2
    cx2 = 0.2
    cy2 = 0.2
    cx3 = 0.12
    cy3 = 0.16

    F = np.zeros(shape=(ndim, 1))

    # ---------------------------------------------------------------------------------
    # Compute F,E using MB potential
    # ---------------------------------------------------------------------------------
    Ftmp, Etmp = MullerBrown(ndim, 1, R)
    # ---------------------------------------------------------------------------------
    # Add 3 gaussian functions
    # ---------------------------------------------------------------------------------
    Ex1 = (R[0][0] - bx1) ** 2 / (2 * cx ** 2)
    Ey1 = (R[1][0] - by1) ** 2 / (2 * cy ** 2)
    Ex2 = (R[0][0] - bx2) ** 2 / (2 * cx2 ** 2)
    Ey2 = (R[1][0] - by2) ** 2 / (2 * cy2 ** 2)
    Ex3 = (R[0][0] - bx3) ** 2 / (2 * cx3 ** 2)
    Ey3 = (R[1][0] - by3) ** 2 / (2 * cy3 ** 2)

    Egauss1 = -A1 * np.exp(-(Ex1 + Ey1))
    Egauss2 = A2 * np.exp(-(Ex2 + Ey2))
    Egauss3 = -A3 * np.exp(-(Ex3 + Ey3))

    Fgauss1x = -(A1 * (R[0][0] - bx1) * np.exp(-Ex1 - Ey1)) / (cx ** 2)
    Fgauss1y = -(A1 * (R[1][0] - by1) * np.exp(-Ex1 - Ey1)) / (cy ** 2)

    Fgauss2x = (A2 * (R[0][0] - bx2) * np.exp(-Ex2 - Ey2)) / (cx2 ** 2)
    Fgauss2y = (A2 * (R[1][0] - by2) * np.exp(-Ex2 - Ey2)) / (cy2 ** 2)

    Fgauss3x = -(A3 * (R[0][0] - bx3) * np.exp(-Ex3 - Ey3)) / (cx3 ** 2)
    Fgauss3y = -(A3 * (R[1][0] - by3) * np.exp(-Ex3 - Ey3)) / (cy3 ** 2)

    energy = Etmp + Egauss1 + Egauss2 + Egauss3
    F[0] = Ftmp[0][0] + Fgauss1x + Fgauss2x + Fgauss3x
    F[1] = Ftmp[1][0] + Fgauss1y + Fgauss2y + Fgauss3y

    # ---------------------------------------------------------------------------------
    # Add back third dimension
    # ---------------------------------------------------------------------------------
    forces = np.zeros(shape=(3, 1))
    forces[0] = F[0]
    forces[1] = F[1]
    forces[2] = 0.0

    return forces, energy
