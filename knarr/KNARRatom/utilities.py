import numpy as np




# Author: Vilhjalmur Asgeirsson, 2019

def InitializeAtomObject(name="reactant", input_config="reactant.xyz", pbc=False,
                         twodee=False):
    from KNARRatom.atom import Atom
    atoms = Atom(name=name, pbc=pbc, twodee=twodee)
    atoms.ReadAtomsFromFile(input_config)
    return atoms

def InitializePathObject(nim, react):

    from KNARRatom.path import Path
    # Initialize a path from the reactant structure
    path = Path(name="linear_interp", nim=nim, config1=react.GetCoords(), pbc=react.GetPBC())
    path.SetNDimIm(react.GetNDim())
    path.SetNDofIm(react.GetNDof())
    path.SetNDim(react.GetNDim() * nim)
    path.SetSymbols(react.GetSymbols() * nim)
    path.SetConstraints(np.array(list(react.GetConstraints()) * nim))
    path.ndof = path.GetNDim() - int(path.GetConstraints().sum())
    path.SetMoveableAtoms()
    path.SetOutputFile(react.GetOutputFile())
    path.pbc = react.GetPBC()
    path.twodee = react.IsTwoDee()
    if path.pbc:
        path.SetCell(react.GetCell())
    return path



def GetMasses(ndim, symbols):
    ind = 0
    mass = np.zeros(shape=(ndim, 1))
    elem = ['a', 'h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar',
            'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br',
            'kr', 'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te',
            'i',
            'xe']
    atm = [1.0, 1.00790, 4.002, 6.94, 9.012, 10.810, 12.011, 14.0067, 15.999, 18.998, 20.1797, 22.989, 24.305,
           26.981, 28.085, 30.973, 32.06, 35.45, 39.948, 39.0983, 40.078, 44.955, 47.867, 50.9415, 51.9961, 54.938,
           55.845, 58.933, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.92160, 78.96, 79.904, 83.798, 85.4678, 87.62,
           88.90585, 91.224, 92.90638, 95.96, 98.0, 101.07, 102.90, 106.42, 107.8682, 112.411, 114.818, 118.710,
           121.760, 127.60, 126.90, 131.293]
    for i in symbols:
        for j in range(0, len(elem)):
            i = i.strip()
            if i.upper() == elem[j].upper():
                mass[ind] = atm[j]
                ind += 1
    return mass


def MIC(ndim, x, pbc, cell):
    if pbc is None:
        raise RuntimeError("PBC is undefined")
    if len(x) != ndim:
        raise ValueError("Dimension mismatch")

    if pbc:
        if len(cell) != 3:
            raise ValueError("Cell dimension mismatch")
        if cell[0] == 0.0 or cell[1] == 0.0 or cell[2] == 0.0:
            raise ValueError("Cell-dimensions can not be zero")

        for i in range(0, ndim, 3):
            x[i + 0] = x[i + 0] - np.floor(x[i + 0] / cell[0]) * cell[0]
            x[i + 1] = x[i + 1] - np.floor(x[i + 1] / cell[1]) * cell[1]
            x[i + 2] = x[i + 2] - np.floor(x[i + 2] / cell[2]) * cell[2]

    return x


def DMIC(ndim, dr, pbc, cell):
    if pbc is None:
        raise RuntimeError("PBC is undefined")
    if len(dr) != ndim:
        raise ValueError("Dimension mismatch")

    if pbc:
        if len(cell) != 3:
            raise ValueError("Cell dimension mismatch")
        if cell[0] == 0.0 or cell[1] == 0.0 or cell[2] == 0.0:
            raise ValueError("Cell-dimensions can not be zero")

        for i in range(0, ndim, 3):
            dr[i + 0] = dr[i + 0] - np.rint(dr[i + 0] / cell[0]) * cell[0]
            dr[i + 1] = dr[i + 1] - np.rint(dr[i + 1] / cell[1]) * cell[1]
            dr[i + 2] = dr[i + 2] - np.rint(dr[i + 2] / cell[2]) * cell[2]
    return dr


def RMS(nlen, x):
    if len(x) != nlen:
        raise RuntimeError("Dimension mismatch")

    rms = 0.0
    for i in range(nlen):
        rms = rms + x[i] * x[i]
    return np.sqrt((1.0 / float(nlen)) * rms)


def RMS3(nlen, x):
    if len(x) != nlen:
        raise RuntimeError("Dimension mismatch")

    rms = 0.0
    for i in range(nlen):
        rms = rms + x[i] * x[i]
    return np.sqrt((3.0 / float(nlen)) * rms)


def RotationMatrixFromPoints(m0, m1):
    """Returns a rigid transformation/rotation matrix that minimizes the
    RMSD between two set of points.

    m0 and m1 should be (3, npoints) numpy arrays with
    coordinates as columns::

        (x1  x2   x3   ... xN
         y1  y2   y3   ... yN
         z1  z2   z3   ... zN)

    The centeroids should be set to origin prior to
    computing the rotation matrix.

    The rotation matrix is computed using quaternion
    algebra as detailed in::

        Melander et al. J. Chem. Theory Comput., 2015, 11,1055
    """
    v0 = np.copy(m0)
    v1 = np.copy(m1)
    # compute the rotation quaternion
    R11, R22, R33 = np.sum(v0 * v1, axis=1)
    R12, R23, R31 = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
    R13, R21, R32 = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
    f = [[R11 + R22 + R33, R23 - R32, R31 - R13, R12 - R21],
         [R23 - R32, R11 - R22 - R33, R12 + R21, R13 + R31],
         [R31 - R13, R12 + R21, -R11 + R22 - R33, R23 + R32],
         [R12 - R21, R13 + R31, R23 + R32, -R11 - R22 + R33]]

    F = np.array(f)
    w, V = np.linalg.eigh(F)
    q = V[:, np.argmax(w)]
    R = QuaternionToMatrix(q)

    return R


def QuaternionToMatrix(q):
    """Returns a rotation matrix.

    Computed from a unit quaternion Input as (4,) numpy array.
    """
    q0, q1, q2, q3 = q
    R = [[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2,
          2 * (q1 * q2 - q0 * q3),
          2 * (q1 * q3 + q0 * q2)],

         [2 * (q1 * q2 + q0 * q3),
          q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2,
          2 * (q2 * q3 - q0 * q1)],

         [2 * (q1 * q3 - q0 * q2),
          2 * (q2 * q3 + q0 * q1),
          q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]]

    return np.array(R)


def MinimizeRotation(ndim, target, atoms, fixcenter=True):
    atoms, da = TranslateToCentroid(ndim, atoms)
    target, dt = TranslateToCentroid(ndim, target)

    new_atoms = Convert1To3(ndim, atoms)
    new_target = Convert1To3(ndim, target)

    Rmat = RotationMatrixFromPoints(new_atoms.T, new_target.T)

    new_atoms = np.dot(new_atoms, Rmat.T)

    if not fixcenter:
        for i in range(0, ndim / 3):
            new_atoms[i, 0] = new_atoms[i, 0] + da[0]
            new_atoms[i, 1] = new_atoms[i, 1] + da[1]
            new_atoms[i, 2] = new_atoms[i, 2] + da[2]

            new_target[i, 0] = new_target[i, 0] + dt[0]
            new_target[i, 1] = new_target[i, 1] + dt[1]
            new_target[i, 2] = new_target[i, 2] + dt[2]

    atoms = Convert3To1(ndim, new_atoms)
    target = Convert3To1(ndim, new_target)

    return atoms, target


def TranslateToCentroid(ndim, rxyz):
    #RBmod. Py3 conversion,. Need to force int instead of float here
    rnew = np.reshape(rxyz, (int(ndim / 3), 3))
    rcenter = np.mean(rnew, axis=0)
    for i in range(0, ndim, 3):
        rxyz[i] -= rcenter[0]
        rxyz[i + 1] -= rcenter[1]
        rxyz[i + 2] -= rcenter[2]
    return rxyz, rcenter


def GetCentroid(ndim, R):
    Rx = 0.0
    Ry = 0.0
    Rz = 0.0

    Rcentr = np.zeros(shape=(3, 1))
    for i in range(0, ndim, 3):
        Rx = Rx + R[i + 0]
        Ry = Ry + R[i + 1]
        Rz = Rz + R[i + 2]

    Rcentr[0] = Rx / float(ndim / 3)
    Rcentr[1] = Ry / float(ndim / 3)
    Rcentr[2] = Rz / float(ndim / 3)
    return Rcentr


def MakeUniformDisplacement(r, dr, dir=0):
    ndim = len(r)
    if dir == 0:
        for i in range(0, ndim, 3):
            r[i + 0] = r[i + 0] + dr
    elif dir == 1:
        for i in range(0, ndim, 3):
            r[i + 1] = r[i + 1] + dr
    elif dir == 2:
        for i in range(0, ndim, 3):
            r[i + 2] = r[i + 2] + dr
    else:
        dirstring = ['x', 'y', 'z']
        raise ValueError("Unknown direction %s" % dirstring[dir])

    return r


def MakeEulerRotation(r, phi, theta, psi):
    newr = np.zeros(shape=(len(r), 1))
    A = np.zeros(shape=(3, 3))
    A[0, 0] = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi)
    A[0, 1] = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
    A[0, 2] = np.sin(psi) * np.sin(theta)

    A[1, 0] = -np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi)
    A[1, 1] = -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi)
    A[1, 2] = np.cos(psi) * np.sin(theta)

    A[2, 0] = np.sin(theta) * np.sin(phi)
    A[2, 1] = -np.sin(theta) * np.cos(phi)
    A[2, 2] = np.cos(theta)

    for i in range(0, len(r), 3):
        atom_i = np.array([r[i], r[i + 1], r[i + 2]])
        rot_atom_i = np.dot(A, atom_i)

        newr[i] = rot_atom_i[0]
        newr[i + 1] = rot_atom_i[1]
        newr[i + 2] = rot_atom_i[2]

    return newr


def Convert1To3(ndim, rxyz):
    #Rb. py3 conversion. int instead of float
    rnew = np.zeros(shape=(int(ndim / 3), 3))
    ind = 0
    for i in range(0, ndim, 3):
        rnew[ind, 0] = rxyz[i]
        rnew[ind, 1] = rxyz[i + 1]
        rnew[ind, 2] = rxyz[i + 2]
        ind = ind + 1
    return rnew


def Convert3To1(ndim, rxyz):
    rnew = np.zeros(shape=(ndim, 1))
    ind = 0
    for i in range(0, ndim, 3):
        rnew[i] = rxyz[ind, 0]
        rnew[i + 1] = rxyz[ind, 1]
        rnew[i + 2] = rxyz[ind, 2]
        ind = ind + 1
    return rnew


def MakeReparametrization(ndim, nim, s, r, tang, type_of_interp=0):
    newr = np.zeros(shape=(ndim * nim, 1))
    xi = np.linspace(s[0], s[-1], nim)

    for i in range(ndim):
        rdof = np.zeros(shape=(nim, 1))
        drdof = np.zeros(shape=(nim, 1))
        for j in range(nim):
            rdof[j] = r[j * ndim + i]
            drdof[j] = -1.0 * tang[j * ndim + i]

        if type_of_interp == 0:
            for img in range(nim):
                new_y = LinearInterpolateData(nim, s, rdof, xi[img], False)
                newr[img * ndim + i] = new_y
        else:
            for img in range(nim):
                new_y = CubicInterpolateData(nim, s, rdof, drdof, xi[img])
                newr[img * ndim + i] = new_y
    return newr


def MakeReparametrizationWithCI(ndim, nim, ci, s, r, tang, type_of_interp):
    newr = np.zeros(shape=(ndim * nim, 1))
    l_new_s = np.linspace(s[0], s[ci - 1], ci)
    r_new_s = np.linspace(s[ci + 1], ci[-1], (nim - 1) - ci)

    if (ci <= 1 or ci >= nim - 2):
        return newr
    if (ci == -1):
        return newr

    # include left side of CI
    for i in range(ndim):
        rdof = np.zeros(shape=(nim, 1))
        drdof = np.zeros(shape=(nim, 1))
        for j in range(nim):
            rdof[j] = r[j * ndim + i]
            drdof[j] = -1.0 * tang[j * ndim + i]

        if type_of_interp == 0:
            for j in range(ci):
                new_y = LinearInterpolateData(nim, s, rdof, l_new_s[j], False)
                newr[j * ndim + i] = new_y
        else:
            for j in range(ci):
                new_y = CubicInterpolateData(nim, s, rdof, drdof, l_new_s[j])
                newr[j * ndim + i] = new_y

    # include CI
    for i in range(ndim):
        newr[ci * ndim + i] = r[ci * ndim + i]

    # include right side of CI
    for i in range(ndim):
        rdof = np.zeros(shape=(nim, 1))
        drdof = np.zeros(shape=(nim, 1))
        for j in range(nim):
            rdof[j] = r[j * ndim + i]
            drdof[j] = -1.0 * tang[j * ndim + i]
        ind = 0
        if type_of_interp == 0:
            for j in range(ci + 1, nim):
                new_y = LinearInterpolateData(nim, s, rdof, r_new_s[ind], False)
                newr[j * ndim + i] = new_y
                ind += 1
        else:
            for j in range(ci + 1, nim):
                new_y = CubicInterpolateData(nim, s, rdof, drdof, r_new_s[ind])
                newr[j * ndim + i] = new_y
                ind += 1

    return newr


def GenerateNewPath(type_of_interp, ndim, nim, npoints, s, coords, tang):
    
    newr = np.zeros(shape=(ndim * npoints, 1))
    xi = np.linspace(s[0], s[-1], npoints)

    for i in range(ndim):
        Rdof = np.zeros(shape=(nim, 1))
        dRdof = np.zeros(shape=(nim, 1))
        for j in range(nim):
            Rdof[j] = coords[j * ndim + i]
            dRdof[j] = -1.0 * tang[j * ndim + i]

        if type_of_interp == 0:
            for xpt in range(npoints):
                new_y = LinearInterpolateData(nim, s, Rdof, xi[xpt], False)
                newr[xpt * ndim + i] = new_y
        else:
            for xpt in range(npoints):
                new_y = CubicInterpolateData(nim, s, Rdof, dRdof, xi[xpt])
                newr[xpt * ndim + i] = new_y

    return newr

def CubicInterpolateData(nlen, xData, yData, dyData, x):
    i = 0
    if (x >= xData[nlen - 2]):
        i = nlen - 2
    else:
        while x > xData[i + 1]:
            i += 1

    xL = xData[i]
    yL = yData[i]
    dyL = dyData[i]
    xR = xData[i + 1]
    yR = yData[i + 1]
    dyR = dyData[i + 1]
    dx = x - xL

    if dx == 0.0:
        return yL

    DR = xR - xL
    a = -2.0 * (yR - yL) / (DR ** 3.0) - (dyR + dyL) / DR ** 2.0
    b = 3.0 * (yR - yL) / (DR ** 2.0) + (2.0 * dyL + dyR) / DR
    c = -dyL
    d = yL

    return a * dx ** 3.0 + b * dx ** 2.0 + c * dx + d


def LinearInterpolateData(nlen, xData, yData, x, extrapolate):
    i = 0
    if (x >= xData[nlen - 2]):
        i = nlen - 2
    else:
        while (x > xData[i + 1]):
            i = i + 1

    xL = xData[i]
    yL = yData[i]
    xR = xData[i + 1]
    yR = yData[i + 1]

    if not extrapolate:
        if (x < xL):
            yR = yL
        if (x > xR):
            yL = yR
    dydx = (yR - yL) / (xR - xL)
    return yL + dydx * (x - xL)
