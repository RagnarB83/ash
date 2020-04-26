import numpy as np
import KNARRsettings
import random

from KNARRatom.utilities import DMIC, RMS3

# Author: Vilhjalmur Asgeirsson, 2019

def ComputeFreq(free_ndim, globdof, mass, H):
    for i in range(free_ndim):
        H[i, i] = H[i, i] / mass[i]
        for j in range(i + 1, free_ndim):
            H[i, j] = H[i, j] / (np.sqrt(mass[i]) * np.sqrt(mass[j]))
            H[j, i] = H[i, j]

    eigenval, eigenvec = np.linalg.eigh(H)

    if globdof > 0:
        eigenval = np.sort(eigenval)
        findmin = abs(eigenval).copy()
        keepind = []
        for k in range(globdof):
            ind = np.argmin(findmin)
            keepind.append(ind)
            eigenvec[:, ind] = 0.0
            findmin[ind] = 1e6  # not the most elegant solution
        eigenval[keepind] = 0.0

    eigenval = np.real(eigenval)

    return eigenval, eigenvec


def ComputeTc(w):
    return KNARRsettings.hbar * np.sqrt(-w) / (2.0 * np.pi * KNARRsettings.kB)


def GenerateVibrTrajectory(fname, ndim, R, symb, w, npts=40, A=1.0):
    f = open(fname, 'w')
    with open(fname, 'w') as f:
        dtheta = np.linspace(0.0, 2 * np.pi, npts)
        dx = A * np.cos(dtheta)
        for k in range(0, len(dx)):
            f.write('%i\n\n' % (ndim / 3))
            for j in range(0, ndim, 3):
                f.write('%s %12.8f %12.8f %12.8f  \n' \
                        % (symb[j], dx[k] * w[j] + R[j], dx[k] * w[j + 1] + R[j + 1],
                           dx[k] * w[j + 2] + R[j + 2]))
    return None


def PathLinearInterpol(ndim, nim, rxyz1, rxyz2,
                       pbc=False, cell=None):
    if pbc is None:
        raise RuntimeError("PBC has not been set")

    rp = np.zeros(shape=(ndim * nim, 1))
    dr = rxyz2 - rxyz1
    dr = DMIC(ndim, dr, pbc, cell)

    n = dr / float(nim - 1)
    ind = 0
    for i in range(0, nim):
        for j in range(0, ndim):
            rp[ind] = rxyz1[j] + i * n[j]
            ind += 1

    return rp


def PathLinearInterpolWithInsertion(ndim, nim, rxyz1, rxyz2,
                                    insertion_config, insertion_no,
                                    pbc=False, cell=None):
    if pbc is None:
        raise RuntimeError("PBC has not been set")

    if insertion_no == 0 or insertion_no == nim - 1:
        raise RuntimeError("Insertion can not be one of the end points")

    if insertion_no == 1 or insertion_no == nim - 2:
        raise RuntimeError("Insertion can not be adjacent to the end points")

    rp = np.zeros(shape=(ndim * nim, 1))
    rp[0:ndim] = rxyz1.copy()
    # ===================================
    # left of insertion
    # ===================================
    dr = insertion_config - rxyz1
    dr = DMIC(ndim, dr, pbc, cell)
    n = dr / float(insertion_no)
    ind = 0
    for i in range(0, insertion_no):
        for j in range(ndim):
            rp[ind] = rxyz1[j] + float(i) * n[j]
            ind += 1
    # ===================================
    # Include insertion
    # ===================================
    rp[insertion_no * ndim:(insertion_no + 1) * ndim] = insertion_config.copy()
    # ===================================
    # Right of insertion
    # ===================================
    dr = rxyz2 - insertion_config
    dr = DMIC(ndim, dr, pbc, cell)
    n = dr / float(nim - (insertion_no + 1))
    ind = (insertion_no + 1) * ndim
    for i in range(1, (nim - insertion_no)):
        for j in range(ndim):
            rp[ind] = insertion_config[j] + i * n[j]
            ind += 1

    return rp


def FindClosestStructure(ndim, nim, rxyz, rp,
                         pbc=False, cell=None):
    listi = []
    for i in range(nim):
        checkr = rp[i * ndim:(i + 1) * ndim]
        dr = rxyz - checkr
        DMIC(ndim, dr, pbc, cell)
        listi.append(RMS3(ndim, dr))
    return np.argmin(listi)


def GetKineticEnergy(ndim, mass, vel):
    Ekin = 0.0
    for i in range(ndim):
        Ekin += 0.5 * mass[i] * vel[i] ** 2
    return Ekin


def GetTemperature(ndim, Ekin, istwodee=False):
    if istwodee:
        assert ndim % 3 == 0
        return Ekin / (0.5 * KNARRsettings.kB * (ndim - (ndim / 3.0)))
    else:
        return Ekin / (0.5 * KNARRsettings.kB * ndim)


def Andersen(atoms, dt, temperature, collfreq=10, collstrength=1.0):
    scaled = False
    mass = atoms.GetMass()[atoms.GetMoveableAtoms()]
    alpha = collstrength
    tcol = collfreq
    pcol = 1.0 - np.exp(-dt / tcol)

    velo = np.zeros(shape=(atoms.GetNDof(), 1))
    vold = atoms.GetV()
    vnew = np.zeros(shape=(atoms.GetNDof(), 1))
    for i in range(0, atoms.GetNDof(), 3):
        aa = np.random.random()
        if aa < pcol:
            scaled = True
            for j in range(3):
                vnew[i + j] = np.sqrt(KNARRsettings.kB * temperature / mass[i + j]) * np.random.normal(0.0, 1.0)
                velo[i + j] = np.sqrt(1.0 - alpha * alpha) * vold[i + j] + alpha * vnew[i + j]

    if scaled:
        atoms.SetV(velo)

    return


def VelocityVerletStep(calculator, atoms, dt=0.01):
    atoms.UpdateR()  # take free-coordinates from atoms.coords and place in atoms.R

    # Compute acceleration
    calculator.Compute(atoms)  # compute energy and atom forces at current position of atoms
    atoms.UpdateF()  # take atoms.forces and put in the free atom forces.
    atoms.ComputeA()  # compute acceleration (using previously calculated forces)

    pos = atoms.GetR()  # get positions of  free atoms and store in variable pos
    velo = atoms.GetV()  # get velocity of free atoms and store in variable velo
    acc0 = atoms.GetA()  # get acceleration of free atoms and store in variable acc0

    pos += (dt * velo) + (0.5 * dt * dt * acc0)  # change coordinates according to velocity verlet

    # Nytt
    velo += 0.5 * acc0 * dt

    atoms.SetR(pos)  # change position of atoms to that of "pos" - calculated in the line above
    atoms.UpdateCoords()  # take free atom coordinates and update the full atom coordinates
    atoms.MIC()  # minimum image convention - check and fix for periodic boundary conditions

    # compute new forces
    calculator.Compute(atoms)
    atoms.UpdateF()
    atoms.ComputeA()
    acc1 = atoms.GetA()

    # velo += 0.5*dt*(acc0+acc1)
    velo += 0.5 * acc1 * dt

    atoms.SetV(velo)  # set velocity of free atoms equal to velo

    return None


def LangevinStep(atoms, calculator, dt=0.01, gamma=0.5, temperature=293.15):
    mass = atoms.GetM()
    atoms.UpdateR()
    
    # Compute acceleration
    calculator.Compute(atoms)
    atoms.UpdateF()
    atoms.ComputeA()

    # take step and compute 1/2 velo
    pos = atoms.GetR()  # save free atoms
    velo = atoms.GetV()  # get velocity
    acc = atoms.GetA()  # get acceleration

    C = np.zeros(shape=(atoms.GetNDof(), 1))
    if atoms.IsTwoDee():
        for i in range(0, atoms.GetNDof(), 3):
            sigma = np.sqrt(2.0 * KNARRsettings.kB * temperature * gamma / mass[i])

            C[i] = 0.5 * dt ** 2 * (acc[i+0] - gamma * velo[i+0]) + sigma * np.sqrt(dt ** 3) * (
                    0.5 * random.normalvariate(0, 1) + random.normalvariate(0, 1) / (2 * np.sqrt(3)))
            C[i + 1] = 0.5 * dt ** 2 * (acc[i+1] - gamma * velo[i+1]) + sigma * np.sqrt(dt ** 3) * (
                    0.5 * random.normalvariate(0, 1) + random.normalvariate(0, 1) / (2 * np.sqrt(3)))


            C[i + 2] = 0.0

    else:
        for i in range(atoms.GetNDof()):
            sigma = np.sqrt(2.0 * KNARRsettings.kB * temperature * gamma / mass[i])
            C[i] = 0.5 * dt ** 2 * (acc[i] - gamma * velo[i]) + sigma * np.sqrt(dt ** 3) * (
                    0.5 * random.normalvariate(0, 1) + random.normalvariate(0, 1) / (2 * np.sqrt(3)))
    acc_temp = acc.copy()
    pos += dt * velo + C

    # update position
    atoms.SetR(pos)
    atoms.UpdateCoords()
    atoms.MIC()

    # compute new forces
    calculator.Compute(atoms)
    atoms.UpdateF()
    atoms.ComputeA()

    friction = np.zeros(shape=(atoms.GetNDof(), 1))
    if atoms.IsTwoDee():
        for i in range(0, atoms.GetNDof(), 3):
            sigma = np.sqrt(2.0 * KNARRsettings.kB * temperature * gamma / mass[i])
            friction[i + 0] = sigma * np.sqrt(dt) * random.normalvariate(0, 1)
            friction[i + 1] = sigma * np.sqrt(dt) * random.normalvariate(0, 1)
            friction[i + 2] = 0.0
    else:
        for i in range(atoms.GetNDof()):
            sigma = np.sqrt(2.0 * KNARRsettings.kB * temperature * gamma / mass[i])
            friction[i] = sigma * np.sqrt(dt) * random.normalvariate(0, 1)

    acc = atoms.GetA()
    velo += 0.5 * dt * (acc + acc_temp) - dt * gamma * velo + friction - gamma * C

    atoms.SetV(velo)

    return None


def GetMaxwellBoltzmannVelocity(free_ndim, mass, temp_input, istwodee=False):
    magnitude = np.sqrt(np.divide(KNARRsettings.kB * temp_input, mass))
    randv = np.random.normal(0, 1, free_ndim)
    velo = magnitude * np.reshape(randv, (free_ndim, 1))
    if istwodee:
        for i in range(0, free_ndim, 3):
            velo[i + 2] = 0.0
    return velo


def Distance(ndim, x0, x1, pbc=False, cell=None):
    assert len(x0) == ndim
    assert len(x1) == ndim
    dr = x1 - x0
    dr = DMIC(ndim, dr, pbc, cell)
    return np.sqrt(np.dot(dr.T, dr))


def AllImageDistances(ndim, nim, r, pbc=False, cell=None):
    distmat = np.zeros(shape=(nim - 1, 1))
    for i in range(1, nim):
        r0 = r[(i - 1) * ndim:i * ndim]
        r1 = r[i * ndim:(i + 1) * ndim]
        distmat[i - 1] = Distance(ndim, r0, r1, pbc, cell)
    avgdist = np.sum(distmat) / (nim - 1)
    return distmat, avgdist


def ComputeLengthOfPath(ndim, nim, r, pbc=False, cell=None):
    s = np.zeros(shape=(nim,))
    for i in range(1, nim):
        r0 = r[(i - 1) * ndim:(i) * ndim]
        r1 = r[i * ndim:(i + 1) * ndim]
        s[i] = s[i - 1] + Distance(ndim, r0, r1, pbc, cell)
    return s


def ComputeEffectiveNEBForce(forces, it, ndim, nim, ci, r, energy,
                             tangent_type, spring_type, energy_weighted, springconst, springconst2,
                             perpspring_type,
                             free_end, free_end_type, free_end_energy1, free_end_energy2, free_end_kappa,
                             startci, remove_extern_force,
                             pbc=False, cell=None, twodee=False):
    from KNARRio.output_print import PrintNEBLogFile

    tang = GetTangent(ndim, nim, r, energy, tangent_type, pbc, cell)
    ksp = ComputeSpringCoefficient(nim, energy_weighted, springconst, springconst2, energy)
    fsp_parallel = ComputeFspringParallel(ndim, nim, r, tang, ksp, ci, energy, spring_type, pbc, cell)
    freal_perp, freal_paral = ComputeForcesPerp(ndim, nim, tang, forces)
    if perpspring_type > 0:
        fsp_perp = ComputeFspringPerp(ndim, nim, tang, r, freal_perp, spring_type, pbc, cell)
        freal_perp += fsp_perp
    else:
        fsp_perp = np.zeros(shape=(ndim * nim, 1))

    fneb = freal_perp + fsp_parallel
    if startci:
        fneb = ComputeClimbingImage(ci, ndim, forces, fneb, tang)

    if free_end:
        if free_end_type == 0:
            fneb, Freal_perp = ComputeFreeEndPerp(ndim, forces, fneb, freal_perp, tang)
        elif free_end_type == 2:
            fneb = ComputeFreeEndFull(ndim, forces, fneb)
        elif free_end_type == 1:
            fneb = ComputeFreeEndContour(ndim, r, forces, fneb, energy,
                                         free_end_energy1, free_end_energy2,
                                         ksp, free_end_kappa,
                                         pbc, cell)
        else:
            raise TypeError("Unknown free-end NEB type")
    else:
        # should not really be needed... but anyways
        fneb[0:ndim] = 0.0
        fneb[-ndim::] = 0.0
        freal_perp[0:ndim] = 0.0
        freal_perp[-ndim::] = 0.0

    if remove_extern_force and not pbc and not twodee:
        fneb = CentroidRemoveTranslation(ndim, nim, fneb)

    if KNARRsettings.printlevel > 0:
        PrintNEBLogFile('neb.debug', ndim, nim, it, forces, freal_perp,
                        ksp, fsp_parallel, fsp_perp, fneb)

    return fneb, freal_perp, freal_paral


def ComputeClimbingImage(im, ndim, forces, fneb, tang):
    fneb[im * ndim:(im + 1) * ndim] = forces[im * ndim:(im + 1) * ndim] - \
                                      2.0 * np.dot(forces[im * ndim:(im + 1) * ndim].T, tang[im * ndim:(im + 1) * ndim]) \
                                      * tang[im * ndim:(im + 1) * ndim]
    return fneb


def ComputeForcesPerp(ndim, nim, tang, forces):
    freal_perp = np.zeros(shape=(ndim * nim, 1))
    freal_paral = np.zeros(shape=(nim, 1))
    for i in range(nim):
        freal_paral[i] = np.dot(forces[i * ndim:(i + 1) * ndim].T, tang[i * ndim:(i + 1) * ndim])
        freal_perp[i * ndim:(i + 1) * ndim] = forces[i * ndim:(i + 1) * ndim] - \
                                              freal_paral[i] * tang[i * ndim:(i + 1) * ndim]

    return freal_perp, freal_paral


def GetTangent(ndim, nim, r, energy, tangent_type, pbc=False, cell=None):
    if tangent_type == 0:
        print('original_tangent')
        tang = IntermOriginalTangent(ndim, nim, r, pbc, cell)
    elif tangent_type == 1:
        tang = IntermImprovedTangent(ndim, nim, r, energy, pbc, cell)
    else:
        raise TypeError("Unknown tangent type")

    tau0 = r[ndim: 2 * ndim] - r[0:ndim]
    tau0 = DMIC(ndim, tau0, pbc, cell)
    tauN = r[-ndim::] - r[-2 * ndim:-ndim]
    tauN = DMIC(ndim, tauN, pbc, cell)

    tang[0:ndim] = tau0 / np.linalg.norm(tau0)
    tang[-ndim::] = tauN / np.linalg.norm(tauN)

    return tang


def IntermOriginalTangent(ndim, nim, r, pbc=False, cell=None):
    keeptang = np.zeros(shape=(ndim * nim, 1))
    for i in range(1, nim - 1):
        rl = r[i * ndim:(i + 1) * ndim] - r[(i - 1) * ndim:i * ndim]
        rl = DMIC(ndim, rl, pbc, cell)

        rh = r[(i + 1) * ndim:(i + 2) * ndim] - r[i * ndim:(i + 1) * ndim]
        rh = DMIC(ndim, rh, pbc, cell)

        tang = rh / np.linalg.norm(rh) + rl / np.linalg.norm(rl)

        keeptang[i * ndim:(i + 1) * ndim] = tang / np.linalg.norm(tang)

    return keeptang


def IntermImprovedTangent(ndim, nim, r, energy, pbc=False, cell=False):
    keeptang = np.zeros(shape=(ndim * nim, 1))
    for i in range(1, nim - 1):

        r0 = r[(i - 1) * ndim:i * ndim]
        r1 = r[i * ndim:(i + 1) * ndim]
        r2 = r[(i + 1) * ndim:(i + 2) * ndim]

        if energy[i + 1] > energy[i] and energy[i] > energy[i - 1]:
            dr = r2 - r1
            dr = DMIC(ndim, dr, pbc, cell)
            rtang = dr
        elif energy[i + 1] < energy[i] and energy[i] < energy[i - 1]:
            dr = r1 - r0
            dr = DMIC(ndim, dr, pbc, cell)
            rtang = dr
        else:
            drplus = r2 - r1
            drplus = DMIC(ndim, drplus, pbc, cell)
            drminus = r1 - r0
            drminus = DMIC(ndim, drminus, pbc, cell)

            Vmax = np.max([np.abs(energy[i + 1] - energy[i]), np.abs(energy[i - 1] - energy[i])])
            Vmin = np.min([np.abs(energy[i + 1] - energy[i]), np.abs(energy[i - 1] - energy[i])])

            if energy[i + 1] > energy[i - 1]:
                rtang = drplus * Vmax + drminus * Vmin
            else:
                rtang = drplus * Vmin + drminus * Vmax

        keeptang[i * ndim:(i + 1) * ndim] = rtang / np.linalg.norm(rtang)
    return keeptang


def ComputeSpringCoefficient(nim, energy_weighted, springconst1, springconst2=None, energy=None):
    if springconst1 < 0.0:
        raise ValueError("Springconst1 can not be negative")

    ksp = springconst1 * np.ones(shape=(nim - 1, 1))

    if energy_weighted:

        if energy is None:
            raise RuntimeError("Energy is needed for energy-weighted springs")

        if springconst1 >= springconst2:
            raise ValueError("Springconst1 >= springconst2")

        if springconst2 < 0.0:
            raise ValueError("Springconst2 can not be negative")

        if springconst2 is None:
            raise ValueError("No value set for springconst2")

        emax = np.max(energy)
        eref = np.max([energy[0], energy[-1]])
        for i in range(1, nim):
            ei = np.max([energy[i], energy[i - 1]])
            if (ei > eref):
                ksp[i - 1] = springconst2 - (springconst2 - springconst1) * ((emax - ei) / (emax - eref))
            else:
                ksp[i - 1] = springconst2 - (springconst2 - springconst1)

    return ksp


def ComputeFspringParallel(ndim, nim, r, tang, ksp, ci, energy, spring_type, pbc=False, cell=None):
    fsp_parallel = np.zeros(shape=(ndim * nim, 1))

    for i in range(1, nim - 1):
        r0 = r[(i - 1) * ndim:i * ndim]
        ri = r[i * ndim:(i + 1) * ndim]
        r1 = r[(i + 1) * ndim:(i + 2) * ndim]

        rh = r1 - ri
        rh = DMIC(ndim, rh, pbc, cell)
        rl = ri - r0
        rl = DMIC(ndim, rl, pbc, cell)
        tangi = tang[i * ndim:(i + 1) * ndim]

        # springs based on original formulation
        if spring_type == 0:  # original spring forces
            fspr = ksp[i] * (rh) - ksp[i - 1] * (rl)
            fsp_tmp = np.dot(fspr.T, tangi) * tangi

        # springs based on adjacent images
        elif spring_type == 1:
            d1 = Distance(ndim, r0, ri, pbc, cell)
            d2 = Distance(ndim, ri, r1, pbc, cell)
            fsp_tmp = (ksp[i] * d2 - ksp[i - 1] * d1) * tangi  # ksp can later become a vector of different values

        # ideal springs
        elif spring_type == 2:
            distmat, dbar = AllImageDistances(ndim, nim, r, pbc, cell)
            dideal = dbar * i
            if ci > -1:
                if i < ci:
                    dbar = np.sum(distmat[0:ci]) / float(ci)
                    dideal = i * dbar
                elif i >= ci:
                    dbar = (np.sum(distmat) - np.sum(distmat[0:ci])) / float(((nim - 1) - ci))
                    dideal = np.sum(distmat[0:ci]) + float(i - ci) * dbar
            fsp_tmp = ksp[0] * (dideal - np.sum(distmat[0:i])) / (2 * dbar) * tangi
        # elastic band springs
        elif spring_type == 3:
            Ltot = Distance(ndim, r[0:ndim], r[-ndim::], pbc, cell)
            print('nim= %i' % nim)
            Leq = Ltot / (nim - 1)
            # tau_plus = rh
            # tau_minus = rl
            taup = rh.copy()
            taum = rl.copy()

            taup_norm = np.linalg.norm(taup)
            taum_norm = np.linalg.norm(taum)

            fsp_tmp = ksp[i] * (taup_norm - Leq) * (taup / taup_norm) - ksp[i - 1] * (taum_norm - Leq) * (
                    taum / taum_norm)

            print('ci: %i' % ci)
            print('len(E):' % len(energy))
            if ci > -1:
                if i == ci - 1 or i == ci + 1:
                    print('scaling spring %i' % i)
                    Vmax = np.max([abs(energy[i + 1] - energy[i]), abs(energy[i] - energy[i - 1])])
                    Vmin = np.min([abs(energy[i + 1] - energy[i]), abs(energy[i] - energy[i - 1])])
                    fsp_tmp *= Vmin / Vmax
        else:
            raise TypeError("Unknown spring type")

        fsp_parallel[i * ndim:(i + 1) * ndim] = fsp_tmp

    return fsp_parallel


def CentroidRemoveTranslation(ndim, nim, forces):
    natoms = float(ndim) / 3.0
    for img in range(nim):
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        for j in range(0, ndim, 3):
            sum_x += forces[img * ndim + j + 0]
            sum_y += forces[img * ndim + j + 1]
            sum_z += forces[img * ndim + j + 2]

        sum_x = sum_x / float(natoms)
        sum_y = sum_y / float(natoms)
        sum_z = sum_z / float(natoms)

        for j in range(0, ndim, 3):
            forces[img * ndim + j + 0] -= sum_x
            forces[img * ndim + j + 1] -= sum_y
            forces[img * ndim + j + 2] -= sum_z
    return forces


def ComputeFreeEndPerp(ndim, forces, fneb, freal, tang):
    parallel0 = np.dot(forces[0:ndim].T, tang[0:ndim]) * tang[0:ndim]
    freal[0:ndim] = forces[0:ndim] - parallel0
    fneb[0:ndim] = freal[0:ndim]

    parallelN = np.dot(forces[-ndim::].T, tang[-ndim::]) * tang[-ndim::]
    freal[-ndim::] = forces[-ndim::] - parallelN
    fneb[-ndim::] = freal[-ndim::].copy()

    return fneb, freal


def ComputeFreeEndFull(free_ndim, forces, fneb):
    fneb[0:free_ndim] = forces[0:free_ndim]
    fneb[-free_ndim::] = forces[-free_ndim::]
    return fneb


def ComputeFreeEndContour(ndim, r, forces, fneb, energy, free_end_energy1, free_end_energy2,
                          ksp, kappa, pbc=False, cell=None):
    f0 = forces[0:ndim]  # initial configuration forces
    fN = forces[-ndim::]  # end configuration forces

    r0 = r[0:ndim]  # initial configuration (i=0)
    r1 = r[ndim:2 * ndim]  # (i=1)
    rN = r[-ndim::]  # end configuration
    rN1 = r[-2 * ndim:-ndim]  # (i=N-1)

    funit_0 = f0 / np.linalg.norm(f0)
    funit_N = fN / np.linalg.norm(fN)

    dr = r1 - r0
    dr = DMIC(ndim, dr, pbc, cell)
    gsp_0 = ksp[0] * dr

    dr = rN1 - rN
    dr = DMIC(ndim, dr, pbc, cell)
    gsp_N = ksp[-1] * dr

    fneb[0:ndim] = gsp_0 - (np.dot(gsp_0.T, funit_0) - kappa * (energy[0] - free_end_energy1)) * funit_0
    fneb[-ndim::] = gsp_N - (np.dot(gsp_N.T, funit_N) - kappa * (energy[-1] - free_end_energy2)) * funit_N

    return fneb


def ComputeFspringPerp(ndim, nim, tang, r, freal, spring_type, pbc=False, cell=None):
    fspr_perp = np.zeros(shape=(ndim * nim, 1))
    ksp_perp = 1.0

    for i in range(1, nim - 1):
        r0 = r[(i - 1) * ndim:i * ndim]
        ri = r[i * ndim:(i + 1) * ndim]
        r1 = r[(i + 1) * ndim:(i + 2) * ndim]
        fr_img = freal[i * ndim:(i + 1) * ndim]

        rh = r1 - ri
        rh = DMIC(ndim, rh, pbc, cell)
        rl = ri - r0
        rl = DMIC(ndim, rl, pbc, cell)

        dr = rh - rl
        tangi = tang[i * ndim:(i + 1) * ndim]

        fspr = dr - (np.dot(dr.T, tangi) * tangi)

        if spring_type == 1:  # COS
            xi = np.dot(rh.T, rl) / (np.linalg.norm(rh) * np.linalg.norm(rl))
            fspr_perp[i * ndim:(i + 1) * ndim] = ksp_perp * 0.5 * (1 + np.cos(np.pi * xi)) * fspr

        elif spring_type == 2:  # TAN
            fspr_perp[i * ndim:(i + 1) * ndim] = 2.0 / np.pi * np.arctan(
                np.linalg.norm(fr_img) ** 2.0 / np.linalg.norm(fspr) ** 2.0) * fspr

        elif spring_type == 3:  # COSTAN
            costheta = np.dot(rh.T, rl) / (np.linalg.norm(rh) * np.linalg.norm(rl))
            cos_func = ksp_perp * 0.5 * (1 + np.cos(np.pi * costheta))
            tan_func = 2.0 / np.pi * np.arctan(np.linalg.norm(fr_img) ** 2 / np.linalg.norm(fspr) ** 2)
            fspr_perp[i * ndim:(i + 1) * ndim] = cos_func * tan_func * fspr

        elif spring_type == 4:  # swDNEB
            FDNEB = np.dot(fspr.T, fr_img) * fr_img
            FDNEB = fspr - FDNEB
            fspr_perp[i * ndim:(i + 1) * ndim] = 2.0 / np.pi * np.arctan(
                np.linalg.norm(freal) ** 2.0 / np.linalg.norm(fspr) ** 2.0) * FDNEB
        else:
            raise TypeError("Unknown spring type for perp. spring force")

    return fspr_perp


def PiecewiseCubicEnergyInterpolation(fname, nim, s, energy, forces, it):
    xintp = []
    eintp = []
    for i in range(nim - 1):
        dr = s[i + 1] - s[i]
        a = -2 * (energy[i + 1] - energy[i]) / (dr ** 3) - (forces[i + 1] + forces[i]) / (dr ** 2)
        b = 3 * (energy[i + 1] - energy[i]) / (dr ** 2) + (2 * forces[i] + forces[i + 1]) / dr
        c = -forces[i]
        d = energy[i]
        xi = np.linspace(0, dr, 20)
        for j in xi:
            p = a * j ** 3 + b * j ** 2 + c * j + d
            eintp.append(float(p))
            xintp.append(float(j + s[i]))

    if fname:
        with open(fname, 'a') as g:
            g.write('Iteration: %i\n' % it)
            g.write('Images:\n')
            for i in range(nim):
                g.write('%.4f %12.6f %12.8f \n' % (s[i] / s[-1], s[i], energy[i]))

            g.write('Interp.:\n')
            for i in range(len(eintp)):
                g.write('%.4f %12.6f %12.8f \n' % (xintp[i] / xintp[-1], xintp[i], eintp[i]))

    return xintp, eintp


def AndersenCollision(ndim, mass, velo, inp_temp, time_step, freq, strength):
    import random
    pcol = time_step * freq
    for i in range(0, ndim, 3):
        a = random.random()
        if a < pcol:
            vx = np.sqrt(KNARRsettings.kB * inp_temp / mass[i + 0]) * np.random.normal()
            vy = np.sqrt(KNARRsettings.kB * inp_temp / mass[i + 1]) * np.random.normal()
            vz = np.sqrt(KNARRsettings.kB * inp_temp / mass[i + 2]) * np.random.normal()
            velo[i + 0] = np.sqrt(1.0 - strength ** 2) * velo[i + 0] + strength * vx
            velo[i + 1] = np.sqrt(1.0 - strength ** 2) * velo[i + 1] + strength * vy
            velo[i + 2] = np.sqrt(1.0 - strength ** 2) * velo[i + 2] + strength * vz
    return velo


def AutoZoom(nim, s, energy, forces, ci, alpha):
    from KNARRatom.utilities import CubicInterpolateData
    energy = energy - energy[0]

    if ci == 0 or ci == nim - 1:
        raise RuntimeError("Highest energy image can not be a terminal image. Please inspect the calculation")

    # Interpolate path with cubic polynomial
    npoints = 1000
    Sintp = np.linspace(s[0], s[-1], npoints)
    Eintp = np.zeros(shape=(npoints,))
    for i, x in enumerate(Sintp):
        Eintp[i] = CubicInterpolateData(nim, s, energy, forces, x)

    # find Emax, Eref
    Emax_intp_ind = np.argmax(Eintp)
    Emax_intp = Eintp[Emax_intp_ind]
    Eref = np.max([energy[0], energy[-1]])

    # shift path such that X (default: half the barrier) is at zero
    Eoffset = (1 - alpha) * (Emax_intp + Eref)
    Eintp = Eintp - Eoffset

    # find all positions along the path where the sign of the shifted energy changes

    point_of_intersect_left = []
    point_of_intersect_right = []
    for i in range(1, len(Eintp) - 1):
        if i <= Emax_intp_ind:
            if np.sign(Eintp[i]) != np.sign(Eintp[i - 1]):
                point_of_intersect_left.append(i)
        else:
            if np.sign(Eintp[i]) != np.sign(Eintp[i - 1]):
                point_of_intersect_right.append(i)

    # Make checks - if everything is OK
    countRight = len(point_of_intersect_right)
    countLeft = len(point_of_intersect_left)
    countTot = countRight + countLeft

    if countTot == 1:
        # if we only find one intersect --- use also reactant / product states
        if countLeft == 1:
            # we go from inda to product
            intp_inda = point_of_intersect_left[0]
            intp_indb = -1

        else:
            # We go from reactant to indb
            intp_inda = 0
            intp_indb = point_of_intersect_right[0]


    elif countTot == 2:
        # if we find two intersects - make sure they are on correct sides of the barrier
        if countLeft == 0:
            # raise RuntimeError("Found two intersects but none on the left side of Emax - NOT OK")
            message = "Found two intersects but none on the left side of Emax - NOT OK"
            return 0, 0, 0, message
        elif countRight == 0:
            # raise RuntimeError("Found two intersects but none on the right side of Emax- NOT OK")
            message = "Found two intersects but none on the right side of Emax- NOT OK"
            return 0, 0, 0, message
        else:
            intp_inda = point_of_intersect_left[0]
            intp_indb = point_of_intersect_right[0]

    elif countTot == 3:
        if countLeft == 0:
            # raise RuntimeError("Found three intersects but none on the left side of Emax - NOT OK")
            message = "Found three intersects but none on the left side of Emax - NOT OK"
            return 0, 0, 0, message
        elif countRight == 0:
            # raise RuntimeError("Found three intersects but none on the right side of Emax- NOT OK")
            message = "Found three intersects but none on the right side of Emax- NOT OK"
            return 0, 0, 0, message

        elif countLeft == 1 and countRight == 2:
            intp_inda = point_of_intersect_left[0]
            intp_indb = np.min(point_of_intersect_right)

        elif countLeft == 2 and countRight == 1:
            intp_inda = np.max(point_of_intersect_left)
            intp_indb = point_of_intersect_right[0]
        else:
            raise RuntimeError("Unknown problem")

    elif countTot >= 4:
        # raise RuntimeError("Too many points of intersection found")
        message = "Too many points of intersection found"
        return 0, 0, 0, message
    else:
        # raise RuntimeError("No intersection point found")
        message = "No intersection point found"
        return 0, 0, 0, message

    # Now I have the intersects at the interpolated surface - translate those to image intervals.

    interval_a = []
    if Sintp[intp_inda] == s[0]:
        interval_a = [0, 1]
    elif Sintp[intp_inda] == s[nim - 1]:
        raise RuntimeError("How can first image be on the wrong side of the barrier?")
    else:
        for i in range(1, len(s)):
            if Sintp[intp_inda] <= s[i] and Sintp[intp_inda] > s[i - 1]:
                # print 'Found interval for a - lies between images %i and %i' % (i - 1, i)
                interval_a = [i - 1, i]

    if Sintp[intp_indb] == s[0]:
        raise RuntimeError("How can first image be on the wrong side of the barrier?")

    elif Sintp[intp_indb] == s[nim - 1]:
        interval_b = [nim - 2, nim - 1]
    else:
        for i in range(1, len(s)):
            if Sintp[intp_indb] <= s[i] and Sintp[intp_indb] > s[i - 1]:
                # print 'Found interval for b - lies between images %i and %i' % (i - 1, i)
                interval_b = [i - 1, i]

    # Finally, select an image for a'
    if interval_a[1] == ci:
        A = interval_a[0]
    else:
        A = interval_a[1]

    if interval_b[0] == ci:
        B = interval_b[1]
    else:
        B = interval_b[0]

    # print 'Selecting second zoom point, B=%i' % B

    if A == 0 and B == nim - 1:
        raise RuntimeError("Selected images are the endpoints, there is no point in performing the zoom")

    return 1, A, B


def ComputeEffectiveMMFForce(F, eig, mm):
    assert len(F) == len(mm)

    if eig > 0:
        feff = -(np.dot(F.T, mm) * mm)
    elif eig < 0:
        feff = F - 2 * (np.dot(F.T, mm) * mm)
    else:
        raise RuntimeError("Zero curvature in effective force")
    return feff


def GetMinimumModeLanczos(calculator, atoms, omega,
                          l_maxiter=30, l_fdstep=0.0001, l_tol=0.001):

    ndim = atoms.GetNDof()
    nim = atoms.GetNim()
    w = omega / np.linalg.norm(omega)

    # ---------------------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------------------
    Rin = atoms.GetR().copy()
    Fin = atoms.GetF().copy()
    w = np.reshape(w, (nim * ndim, 1))
    q = np.zeros(shape=(nim * ndim, 1))
    beta = np.linalg.norm(w)
    keepalpha = []
    keepbeta = []
    keepeig = 1e6
    P = np.zeros(shape=(ndim * nim, l_maxiter))
    converged = False

    # ---------------------------------------------------------------------------------
    # Start Lanczos iterations
    # ---------------------------------------------------------------------------------
    for it in range(l_maxiter):
        qold = q
        q = w / beta
        q_tmp = np.reshape(q, (ndim * nim,))
        P[:, it] = q_tmp
        step = np.zeros(shape=(ndim * nim, 1))
        for j in range(ndim * nim):
            step[j] += l_fdstep * q[j]

        atoms.SetR(Rin + step)
        atoms.UpdateCoords()
        calculator.Compute(atoms)
        atoms.UpdateF()

        z = np.reshape(atoms.GetF() - Fin, (ndim * nim, 1))
        w = z - beta * qold
        alpha = np.dot(q.T, w)
        keepalpha.append(float(alpha))
        w = w - alpha * q
        beta = np.linalg.norm(w)
        keepbeta.append(float(beta))

        # construct the tridiagonal matrix T_(it)
        if it >= 3:
            alp = np.asarray(keepalpha).copy()
            alp = -alp / l_fdstep
            bet = np.asarray(keepbeta).copy()
            bet = -bet / l_fdstep
            T = np.zeros(shape=(it, it))
            for k in range(it - 1):
                T[k, k] = alp[k]
                T[k + 1, k] = bet[k]
                T[k, k + 1] = bet[k]
            T[k + 1, k + 1] = alp[k + 1]

            eig = np.linalg.eigh(T)[0]
            eig = np.min(eig)
            if abs((eig - keepeig)) < l_tol:
                converged = True
            keepeig = eig

            # ---------------------------------------------------------------------------------
            # Get eigenvector
            # ---------------------------------------------------------------------------------
            if (converged == True or it == l_maxiter - 1):
                # now find the lowest eigenvector
                shift = -eig + 1e-4
                alp = alp + shift  # shift matrix to be positive-definite

                for j in range(1, it):
                    t = bet[j - 1]
                    bet[j - 1] = t / alp[j - 1]
                    alp[j] = alp[j] - t * bet[j - 1]

                first_inv_iter = True
                counter = 0

                for lvec_iter in range(5000):
                    if first_inv_iter == True:
                        first_inv_iter = False
                        # start with a random vector
                        b = []
                        for i in range(it + 1):
                            b.append(random.uniform(-0.5, 0.5))
                        b = np.asarray(b)
                        b = b / np.linalg.norm(b)
                        a = b.copy()

                    for j in range(1, it):
                        b[j] = b[j] - bet[j - 1] * b[j - 1]
                    b[it] = b[it] / alp[it]

                    for j in range(it - 1, -1, -1):
                        b[j] = b[j] / alp[j] - bet[j] * b[j + 1]
                    b = b / np.linalg.norm(b)  # lanczos eigenvector

                    # construct the eigenvector from the lanczos eigenvector
                    if (abs(np.dot(a, b) - 1.0) < 1.0e-10):
                        w = np.zeros(shape=(ndim * nim, 1))
                        for k in range(it):
                            for l in range(ndim * nim):
                                w[l] = w[l] + b[k] * P[l, k]
                        w = w / np.linalg.norm(w)

                        # restore original values
                        atoms.SetR(Rin)
                        atoms.SetF(Fin)
                        atoms.UpdateCoords()
                        return eig, w, it

                    if counter > 5000:
                        raise RuntimeError("Lanczos iterations were unable to converge to a minimum mode")

                    a = b.copy()
                    counter += 1


def ComputeCosCircle(atoms, calculator,
                     radius=0.05,
                     npts=200):
    center = atoms.GetR().copy()
    x0 = center[0]
    y0 = center[1]
    t = np.linspace(0, 2 * np.pi, npts)
    circle = np.zeros(shape=(npts, 3))
    for i, val in enumerate(t):
        x = x0 + radius * np.cos(val)
        y = y0 + radius * np.sin(val)
        z = 0.0
        rxyz = [x, y, z]
        rxyz = np.reshape(rxyz, (3, 1))
        atoms.SetR(rxyz)
        atoms.UpdateCoords()
        cos = calculator.ComputeCosSq(atoms)
        circle[i, 0] = x
        circle[i, 1] = y
        circle[i, 2] = cos

    return circle
