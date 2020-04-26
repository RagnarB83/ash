import numpy as np
from math import exp, sqrt




def getEnergyAndForce(r, cell=None, De=1.0, a=1.0, re=1.0):
    ndim = len(r)
    box = self.atoms.get_cell().copy()
    Rnew = np.reshape(r, (ndim,1))
    
    De = self.parameters.epsilon
    a = self.parameters.rho0
    re = self.parameters.r0
    
    cutoff = 10000.0
    diffR=0.0
    diffRX=0.0
    diffRY=0.0
    diffRZ=0.0
    E = 0.0
    F = np.zeros(shape=(ndim,1))
    for i in range(0,ndim/3):
        for j in range(i+1,ndim/3):
            diffRX = Rnew[3*i]  - Rnew[3*j]
            diffRY = Rnew[3*i+1]-  Rnew[3*j+1]
            diffRZ = Rnew[3*i+2]-  Rnew[3*j+2]
            if cell is not None:
                diffRX = diffRX - cell[0]*np.floor(diffRX/cell[0]+0.5)
                diffRY = diffRY - cell[1]*np.floor(diffRY/cell[1]+0.5)
                diffRZ = diffRZ - cell[2]*np.floor(diffRZ/cell[2]+0.5)
            diffR = np.sqrt(diffRX*diffRX+diffRY*diffRY+diffRZ*diffRZ);
            #expression for energy and force
            
            d=1.0-np.exp(-a*(diffR-re))
            energy=De*d*d-De
            force= 2.0*De*d*(d-1.0)*a
            E = E + energy
            F[ 3*i ]+=force*diffRX/diffR
            F[3*i+1]+=force*diffRY/diffR
            F[3*i+2]+=force*diffRZ/diffR
            F[ 3*j ]-=force*diffRX/diffR
            F[3*j+1]-=force*diffRY/diffR
            F[3*j+2]-=force*diffRZ/diffR
    return F,E
    

