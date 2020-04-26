import numpy as np
from KNARRcalculator.utilities import GetAllConfigDistances

# Author: Vilhjalmur Asgeirsson, 2019

def IDPP(calculator, path, list_to_compute=None):
    ndim = path.GetNDimIm()
    nim = path.GetNim()
    rxyz = path.GetCoords()
    dkappa = path.Getdkappa()
    pbc = path.GetPBC()
    cell = path.GetCell()

    # Not parallel - havent found a reason to include parallel IDPP yet.
    s = np.zeros(shape=(nim, 1))
    ds = np.zeros(shape=(ndim * nim, 1))

    for i in range(nim):
        idpp_grad, idpp_objf = IDPPWorker(ndim, rxyz[i * ndim:(i + 1) * ndim],
                                          dkappa[:, :, i], pbc, cell)
        s[i] = idpp_objf
        ds[i * ndim:(i + 1) * ndim] = idpp_grad

    path.SetForces(ds)
    path.SetEnergy(s)
    return


def IDPPWorker(ndim, rxyz, dkappa, pbc=False, cell=None):
    # RB. int addition. Correct??
    natoms = int(ndim / 3)
    objf_grad = np.zeros(shape=(ndim, 1))

    rcurr_dist, rcurr_dx, rcurr_dy, rcurr_dz = GetAllConfigDistances(ndim, rxyz, pbc, cell)

    ddr = np.zeros(shape=(natoms, natoms))
    for i in range(natoms):
        for j in range(natoms):
            ddr[i, j] = rcurr_dist[i, j] - dkappa[i, j]

    for i in range(natoms):
        rcurr_dist[i, i] = 1.0

    summa = 0.0
    for i in range(natoms):
        for j in range(natoms):
            w = 1.0 / (rcurr_dist[i, j] ** 4)
            summa = summa + w * (ddr[i, j] * ddr[i, j])

    objf = 0.5 * summa

    grad = np.zeros(shape=(natoms, natoms))
    dxgrad = np.zeros(shape=(natoms, natoms))
    dygrad = np.zeros(shape=(natoms, natoms))
    dzgrad = np.zeros(shape=(natoms, natoms))

    for i in range(natoms):
        for j in range(natoms):
            grad[i, j] = (ddr[i, j] * (1.0 - 2.0 * ddr[i, j] / rcurr_dist[i, j]) / rcurr_dist[i, j] ** 5)

    for i in range(natoms):
        for j in range(natoms):
            dxgrad[i, j] = grad[i, j] * rcurr_dx[i, j]
            dygrad[i, j] = grad[i, j] * rcurr_dy[i, j]
            dzgrad[i, j] = grad[i, j] * rcurr_dz[i, j]

    grad3 = np.zeros(shape=(natoms, 3))
    for j in range(natoms):
        sumdx = 0.0
        sumdy = 0.0
        sumdz = 0.0
        for i in range(natoms):
            sumdx += dxgrad[i, j]
            sumdy += dygrad[i, j]
            sumdz += dzgrad[i, j]
        grad3[j, 0] = -2.0 * sumdx
        grad3[j, 1] = -2.0 * sumdy
        grad3[j, 2] = -2.0 * sumdz

    for j in range(0, ndim, 3):
        #RB. int addition. Correct??
        objf_grad[j + 0] = grad3[int(j / 3), 0]
        objf_grad[j + 1] = grad3[int(j / 3), 1]
        objf_grad[j + 2] = grad3[int(j / 3), 2]

    return objf_grad, objf
