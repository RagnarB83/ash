import numpy as np


# Author: Vilhjalmur Asgeirsson

def GlobalVPO(F, velo, dt):
    # Euler integrator

    c = np.dot(velo.T, F)
    if c > 0.0:
        velo = c * np.divide(F, np.dot(F.T, F))
    else:
        velo = 0.0
    velo = velo + dt * F
    step = velo * dt + 0.5 * F * (dt ** 2)

    return step, velo


def LocalVPO(ndim, nim, forces, velo, timestep):
    # Euler integrator
    step = np.zeros(shape=(ndim * nim, 1))
    for i in range(nim):
        c = np.dot(velo[i * ndim:(i + 1) * ndim].T, forces[i * ndim:(i + 1) * ndim])
        if c > 0.0:
            velo[i * ndim:(i + 1) * ndim] = c * forces[i * ndim:(i + 1) * ndim] / np.dot(
                forces[i * ndim:(i + 1) * ndim].T, forces[i * ndim:(i + 1) * ndim])
        else:
            velo[i * ndim:(i + 1) * ndim] = 0.0

        velo[i * ndim:(i + 1) * ndim] = velo[i * ndim:(i + 1) * ndim] + timestep * forces[i * ndim:(i + 1) * ndim]
        step[i * ndim:(i + 1) * ndim] = velo[i * ndim:(i + 1) * ndim] * \
                                        timestep + 0.5 * forces[i * ndim:(i + 1) * ndim] * (timestep ** 2)

    return step, velo


def AtomVPO(ndim, nim, forces, velo, timestep):
    # Euler integrator
    raise NotImplementedError()
    return


def DOFVPO(ndim, nim, forces, velo, timestep):
    # Euler integrator
    raise NotImplementedError()
    return


def AndriVPO(ndim, forces, velo, timestep):
    step = np.ones(shape=(ndim, 1))
    fv = 0.0
    fd = 0.0
    for k in range(ndim):
        t = forces[k] * velo[k]
        if (t < 0.0):
            velo[k] = 0.0
        else:
            fv = fv + t
        fd = fd + forces[k] * forces[k]

    for j in range(ndim):
        velo[j] = forces[j] * fv / fd
        velo[j] = velo[j] + forces[j] * timestep
        step[j] = velo[j] * timestep

    return step, velo
