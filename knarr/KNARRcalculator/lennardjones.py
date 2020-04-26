import KNARRsettings
import numpy as np
import math

def LennardJones(calculator, atoms, list_to_compute=[]):
    ndim = atoms.GetNDimIm()
    nim = atoms.GetNim()
    rxyz = atoms.GetCoords()

    energy = np.zeros(shape=(nim, 1))
    forces = np.zeros(shape=(nim * ndim, 1))
    counter = 0
    if list_to_compute is None:
        for i in range(atoms.GetNim()):
            ftmp, etmp = LennardJonesWorker(rxyz[i * ndim:(i + 1) * ndim])
            
            energy[i] = etmp
            forces[i * ndim:(i + 1) * ndim] = ftmp
            
            counter += 1
    else:
        for i, val in enumerate(list_to_compute):
            ftmp, etmp = LennardJonesWorker(rxyz[val * ndim:(val + 1) * ndim])
            
            energy[val] = etmp
            forces[val * ndim:(val + 1) * ndim] = ftmp

            counter += 1
    atoms.AddFC(counter)
    atoms.SetForces(forces)
    atoms.SetEnergy(energy)

    return None

def LennardJonesWorker(r):
    #init parameters
    ndim = len(r)
    sigma=KNARRsettings.ljsigma
    epsilon=KNARRsettings.ljepsilon
    rcut=KNARRsettings.ljrcut

    attr = (sigma/rcut)**6
    Ecut = 4*epsilon*attr*(attr-1)
    gradcut = 24*epsilon*attr/rcut*(1-2*attr) 
    F = np.zeros(shape=(ndim,1))
    E = 0.0
    repulsive = 0.0
    attractive = 0.0    
    for i in range(0,ndim,3):
        x0 = r[i]
        y0 = r[i+1]
        z0 = r[i+2]
        for j in range(i+3,ndim,3):
            dx1 = r[j]-x0
            dy1 = r[j+1]-y0
            dz1 = r[j+2]-z0
            rdist = np.sqrt(dx1*dx1+dy1*dy1+dz1*dz1)
            if rdist < rcut:
                tterm = (sigma/rdist)**6
                E =E+4*epsilon*tterm*(tterm - 1)-Ecut
                gradi = 24*epsilon*tterm/rdist*(1-2*tterm) - gradcut 
                Fx=gradi*dx1/rdist           
                Fy=gradi*dy1/rdist
                Fz=gradi*dz1/rdist

                F[i] = F[i]+Fx
                F[i+1] = F[i+1]+Fy
                F[i+2] = F[i+2]+Fz
                F[j] = F[j]-Fx
                F[j+1] = F[j+1]-Fy
                F[j+2] = F[j+2]-Fz
            
    return F,E
