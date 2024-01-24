import KNARRsettings
import numpy as np


def HenkelmanGaussBoosted(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = HenkelmanGaussBoostedWorker2(rxyz[i * ndim:(i + 1) * ndim])

            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp

            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = HenkelmanGaussBoostedWorker2(rxyz[val * ndim:(val + 1) * ndim])

            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1
    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None


def HenkelmanGaussBoostedWorker(r):
    KNARRsettings.boost_time = 1.0
    KNARRsettings.boosted = True
    ndim = len(r)
    alfa = KNARRsettings.gauss_alpha
    A = KNARRsettings.gauss_A

    x, y, z = [], [], []
    x0, y0 = 0.5, 0.1013
    s = 0.0  # Sum of squares for the exponent of e in the Gauss function
    H, E = 0.0, 0.0
    for i in range(0, ndim, 3):
        x.append(r[i])
        y.append(r[i + 1])
        z.append(r[i + 2])

    # Compute Energy
    for i in range(ndim // 3):
        s += (x[i] - x0) ** 2 + (y[i] - y0) ** 2
        H += np.cos(2 * np.pi * x[i]) * (1 + 4 * y[i]) + 0.5 * (2 * np.pi * y[i]) ** 2 + 1.20265
    # Compute gauss function
    G = A * np.exp(-alfa * s)  # G is the Gauss function

    # check to see if we are in the 'boosted' minimum
    inMinimum = True
    f = 1.0
    for i in range(ndim / 3):
        if x[i] <= 0.0 or x[i] >= 1.0:
            inMinimum = False
            break

    if inMinimum:
        for i in range(ndim // 3):
            f *= 0.5 * (1 + np.cos(2 * np.pi * (x[i] - x0)))  # f is the switch function
    else:
        f = 0.0

    E = H + f * G
    KNARRsettings.boost_time = np.exp(f * G / 0.2)  # hard coded temp

    # Differentiation of H (the Henkelman (or Voter) potential function)
    H_diff_vector = []
    for i in range(ndim / 3):
        Hx = -2 * np.pi * np.sin(2 * np.pi * x[i]) * (1 + 4 * y[i])
        Hy = 4 * np.cos(2 * np.pi * x[i]) + 4 * np.pi ** 2 * y[i]
        Hz = 0.0
        H_diff_vector.append(Hx)
        H_diff_vector.append(Hy)
        H_diff_vector.append(Hz)

    # Differentiation of f (the switch function)
    f_vector = []  # This is f for all dimensions
    f_diff_vector = []  # This is f differentiated for each dimension
    for i in range(ndim // 3):
        f_vector.append(0.5 * (1 + np.cos(2 * np.pi * (x[i] - x0))))
        f_vector.append(1.0)
        f_vector.append(1.0)
        f_diff_vector.append(-np.pi * np.sin(2 * np.pi * (x[i] - x0)))
        f_diff_vector.append(0.0)
        f_diff_vector.append(0.0)

    fd = []  # This is f (the product over all dimensions) differentiated
    for i in range(0, ndim, 3):
        temp_vector = f_vector
        temp_vector.pop(i)
        f_diff = f_diff_vector[i]

        for j in range(ndim // 3 - 1):
            f_diff *= temp_vector[j]

        fd.append(f_diff)
        fd.append(0.0)
        fd.append(0.0)

    # Differentiation of G (the Gauss function) for all dimentions

    G_diff_vector = []
    for i in range(0, ndim, 3):
        Gx = -2 * A * alfa * (x[i] - x0) * np.exp(-alfa * s)
        Gy = -2 * A * alfa * (y[i] - y0) * np.exp(-alfa * s)
        Gz = 0.0

        G_diff_vector.append(Gx)
        G_diff_vector.append(Gy)
        G_diff_vector.append(Gz)

    # Compute the force F (F = -gradient)

    F = np.zeros(shape=(ndim, 1))
    if inMinimum:
        for i in range(0, ndim, 3):
            F[i] = -(H_diff_vector[i] + fd[i] * G + f_vector[i] * G_diff_vector[i])
            F[i + 1] = -(H_diff_vector[i + 1] + fd[i + 1] * G + f_vector[i + 1] * G_diff_vector[i])
            F[i + 2] = 0.0

    else:
        for i in range(0, ndim, 3):
            F[i] = -(H_diff_vector[i])
            F[i + 1] = -(H_diff_vector[i + 1])
            F[i + 2] = 0.0

    return F, E


def SwitchFunc(x, x0=0.5):
    return 0.5 + 0.5 * np.cos(2 * np.pi * x - 2 * np.pi * x0)


def HenkelmanGaussBoostedWorker2(r):
    KNARRsettings.boost_time = 1.0                                                                                                                           
    KNARRsettings.boosted = True  
    x0 = 0.5
    y0 = 0.1013
    A = KNARRsettings.gauss_A
    alfa = KNARRsettings.gauss_alpha
    ndim = len(r)

    # Check if we are in the minimum
    isMinimum = True
    for i in range(0, ndim, 3):
        if r[i + 0] <= 0.0 or r[i + 0] >= 1.0:
            isMinimum = False
            break

    # Get the product of the switching function
    prodS = 1.0
    for i in range(0, ndim, 3):
        prodS *= SwitchFunc(r[i + 0], x0=x0)

    # Get function values of H and G
    H = 0.0
    for i in range(0, ndim, 3):
        x = r[i + 0]
        y = r[i + 1]
        H += np.cos(2.0 * np.pi * x) * (1.0 + 4.0 * y) + 0.5 * (2.0 * np.pi * y) ** 2 + 1.20265
        exponent = (x - x0) ** 2 + (y - y0) ** 2
    G = A * np.exp(-alfa * exponent)
    
    # if we are outside the minimum - there is no Gaussian
    if not isMinimum:
        G = 0.0
    boostFunc = prodS*G
    E = H + boostFunc
    KNARRsettings.boost_time = np.exp(boostFunc/0.2)
    F = np.zeros(shape=(ndim, 1))
    
    # Calculate derivatives
    for i in range(0, ndim, 3):
        x = r[i + 0]
        y = r[i + 1]
        dHx = -2.0 * np.pi * np.sin(2 * np.pi * x) * (1 + 4 * y)
        dHy = 4.0 * np.cos(2 * np.pi * x) + 4 * np.pi ** 2 * y
        dGx = -2.0 * alfa*(x - x0) * G
        dGy = -2.0 * alfa*(y - y0) * G
        dSy = 0.0
        try:
            dSx = prodS / SwitchFunc(x, x0=x0) * (-1.0 * np.pi * np.sin(2.0 * np.pi * x - 2.0 * np.pi * x0))
        except ZeroDivisionError:
            dSx = 0.0
        F[i + 0] = -(dHx + dGx * prodS + G * dSx)
        F[i + 1] = -(dHy + dGy * prodS + G * dSy)
        F[i + 2] = 0.0

    return F, E

# def HenkelmanGaussBoostedWorker(r):
#    KNARRsettings.boost_time = 1.0
#    KNARRsettings.boosted = True
#    ndim = len(r)
#    alfa = KNARRsettings.gauss_alpha
#    A = KNARRsettings.gauss_A
#    B = KNARRsettings.gauss_B
#    x,y,z = [],[],[]
#    x0,y0 = [],0.1013
#    H,fG,E = 0.0,0.0,0.0
#    for i in range(0,ndim,3):
#        x.append(r[i])
#        y.append(r[i+1])
#        z.append(r[i+2])

# Compute Energy
#    for i in range(ndim/3):
#        x0.append(math.floor(x[i]) + 0.5)
#        H += np.cos(2*np.pi*x[i])*(1+4*y[i]) + 0.5*(2*np.pi*y[i])**2 + 1.20265
#        fG += 0.5*(1 + np.cos(2*np.pi*(x[i]-x0[i])))* A*(np.exp(-alfa*((x[i]-x0[i])**2 + (y[i] - y0)**2)))
#        print fG
#        KNARRsettings.boost_time = np.exp(fG/0.2) # hard coded temp
#
#    E = H + fG
#    F = np.zeros(shape=(ndim,1))

# Compute gradient
#    ind = 0
#    for i in range(0,ndim,3):
#        F[i]  = 2*np.pi*np.sin(2*np.pi*x[ind])*(1+4*y[ind]) + np.sin(2*np.pi*(x[ind]-x0[ind]))*np.pi*A*np.exp(-alfa*((x[ind]-x0[ind])**2 + (y[ind]-y0)**2)) + (1 + np.cos(2*np.pi*(x[ind]-x0[ind]))) * A*alfa*np.exp(-alfa*((x[ind]-x0[ind])**2 + (y[ind]-y0)**2))*(x[ind]-x0[ind])
#        F[i+1]= -4*np.cos(2*np.pi*x[ind]) - 4*np.pi**2 *y[ind] + (1+np.cos(2*np.pi*(x[ind]-x0[ind])))*alfa*A*np.exp(-alfa*((x[ind]-x0[ind])**2 + (y[ind]-y0)**2))*(y[ind]-y0)
#        F[i+2]= 0.0
#        ind += 1

#    return F,E
